'use strict';

const videos = [...document.querySelectorAll('video.result-video')];
const visible = new Set();
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const spatialVideo = document.querySelector('#spatial-video');
const comparisonConditions = [['ppo', 'hopper'], ['ppo', 'walker'], ['sac', 'hopper'], ['sac', 'walker']];
let comparisonIndex = 0;

function setMetricLabels(selector, entries) {
  const fields = entries[0].reward === undefined
    ? [['success5', 'S@5cm', 1, '%', true], ['success', 'S@10cm', 1, '%', true], ['smoothness', 'sm', 3, '', false]]
    : [['reward', 'R', 1, '', true], ['smoothness', 'sm', 3, '', false]];
  for (const group of document.querySelectorAll(selector)) {
    group.removeAttribute('aria-hidden');
    group.replaceChildren(...entries.map(entry => {
      const item = document.createElement('span');
      item.className = 'method-label';
      const name = document.createElement('span');
      name.textContent = entry.label;
      const metric = document.createElement('small');
      metric.className = 'video-metrics';
      metric.append('(');
      fields.forEach(([key, label, digits, suffix, higherIsBetter], index) => {
        const values = entries.map(peer => Number(peer[key].toFixed(digits)));
        const best = higherIsBetter ? Math.max(...values) : Math.min(...values);
        const value = document.createElement(Number(entry[key].toFixed(digits)) === best ? 'strong' : 'span');
        value.textContent = entry[key].toFixed(digits) + suffix;
        metric.append(`${index ? ', ' : ''}${label}: `, value);
      });
      metric.append(')');
      if (entry.targets !== undefined) {
        metric.title = `${entry.targets} targets in this video; within 5 cm: ${entry.successes5}, within 10 cm: ${entry.successes}; held for one second`;
      }
      item.append(name, document.createTextNode(' '), metric);
      return item;
    }));
  }
}

setMetricLabels('#spatial .comparison-labels', videoMetrics.spatial['ppo-hopper']);
setMetricLabels('#noise .comparison-labels', videoMetrics.noise['0.5']);
setMetricLabels('#real .comparison-labels', videoMetrics.real);

function comparisonPath() {
  const [algorithm, task] = comparisonConditions[comparisonIndex];
  spatialVideo.width = 1936;
  spatialVideo.height = 502;
  return `static/media/presentation/matched-${algorithm}-${task}-all`;
}
const initialComparisonPath = comparisonPath();
spatialVideo.querySelector('source').dataset.src = `${initialComparisonPath}.mp4`;
spatialVideo.poster = `${initialComparisonPath}.jpg`;

function loadVideo(video) {
  const source = video.querySelector('source');
  if (!source.hasAttribute('src')) {
    source.src = source.dataset.src;
    video.load();
  }
}

function synchronizedGroup(members) {
  const leader = members[0];
  const internalPlay = new WeakSet();
  const internalPause = new WeakSet();
  const internalSeek = new WeakMap();
  let wanted = false;
  let onScreen = false;
  let restarting = false;

  function halt() {
    for (const video of members) {
      if (!video.paused) {
        internalPause.add(video);
        video.pause();
      }
    }
  }

  function seek(video, time) {
    if (video.readyState < 1 || Math.abs(video.currentTime - time) < 0.015) return;
    internalSeek.set(video, time);
    video.currentTime = time;
  }

  function resume() {
    if (!wanted || document.hidden) return;
    for (const video of members) {
      if (video.paused) {
        internalPlay.add(video);
        video.play().catch(() => { internalPlay.delete(video); });
      }
    }
  }

  function start() {
    wanted = true;
    for (const video of members) {
      video.preload = 'auto';
      loadVideo(video);
      if (video !== leader) seek(video, leader.currentTime);
    }
    resume();
  }

  for (const video of members) {
    video.loop = false;
    video.addEventListener('play', () => {
      if (internalPlay.delete(video)) return;
      start();
    });
    video.addEventListener('pause', () => {
      if (internalPause.delete(video) || video.ended || restarting) return;
      wanted = false;
      halt();
    });
    video.addEventListener('seeking', () => {
      const target = internalSeek.get(video);
      if (target !== undefined && Math.abs(video.currentTime - target) < 0.08) return;
      internalSeek.delete(video);
      for (const peer of members) if (peer !== video) seek(peer, video.currentTime);
    });
    video.addEventListener('seeked', () => {
      internalSeek.delete(video);
      if (members.every(peer => !peer.seeking)) restarting = false;
      resume();
    });
    video.addEventListener('ratechange', () => {
      for (const peer of members) {
        if (peer.playbackRate !== video.playbackRate) peer.playbackRate = video.playbackRate;
      }
    });
    video.addEventListener('canplay', resume);
    video.addEventListener('ended', () => {
      if (!wanted || restarting) return;
      restarting = true;
      halt();
      for (const peer of members) seek(peer, 0);
      resume();
    });
  }

  // Use the main view as the playback clock, including after buffering or looping.
  setInterval(() => {
    if (!wanted || document.hidden || leader.paused || leader.seeking) return;
    for (const video of members.slice(1)) {
      if (!video.seeking && Math.abs(video.currentTime - leader.currentTime) > 0.06) {
        seek(video, leader.currentTime);
      }
    }
  }, 100);

  return {
    updateVisibility() {
      const next = !document.hidden && members.some(video => visible.has(video));
      if (next === onScreen) return;
      onScreen = next;
      if (!next) {
        wanted = false;
        halt();
      } else if (!reducedMotion.matches) {
        start();
      }
    },
  };
}

const groups = new Map();
for (const video of videos) {
  const name = video.dataset.syncGroup;
  if (name && !groups.has(name)) {
    groups.set(name, synchronizedGroup(videos.filter(peer => peer.dataset.syncGroup === name)));
  }
}

function playVisible(video) {
  if (document.hidden || reducedMotion.matches || !visible.has(video)) return;
  if (video.dataset.syncGroup) return;
  loadVideo(video);
  video.play().catch(() => {});
}

const playback = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const video = entry.target;
    if (entry.intersectionRatio >= 0.2) {
      visible.add(video);
      if (!video.dataset.syncGroup) playVisible(video);
    } else {
      visible.delete(video);
      if (!video.dataset.syncGroup) video.pause();
    }
  }
  for (const group of groups.values()) group.updateVisibility();
}, { threshold: [0, 0.2] });

const loading = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    loadVideo(entry.target);
    loading.unobserve(entry.target);
  }
}, { rootMargin: '200px' });

for (const video of videos) {
  video.muted = true;
  loading.observe(video);
  playback.observe(video);
  video.addEventListener('loadedmetadata', () => playVisible(video));
}

document.addEventListener('visibilitychange', () => {
  for (const video of videos) {
    if (video.dataset.syncGroup) continue;
    if (document.hidden) video.pause();
    else if (visible.has(video)) playVisible(video);
  }
  for (const group of groups.values()) group.updateVisibility();
});

function switchVideo(video, path, label) {
  const source = video.querySelector('source');
  video.pause();
  source.dataset.src = `${path}.mp4`;
  source.src = source.dataset.src;
  video.poster = `${path}.jpg`;
  video.setAttribute('aria-label', label);
  video.load();
  playVisible(video);
}

function updateComparison() {
  const [algorithm, task] = comparisonConditions[comparisonIndex];
  const condition = `${algorithm.toUpperCase()} / ${task === 'hopper' ? 'Hopper' : 'Walker'}`;
  document.querySelector('#comparison-condition').textContent = condition;
  setMetricLabels('#spatial .comparison-labels', videoMetrics.spatial[`${algorithm}-${task}`]);
  switchVideo(spatialVideo, comparisonPath(),
    `CAPS spatial-only, LipsNet++, and CATS on ${condition} at comparable smoothness`);
}
function nextComparison() {
  comparisonIndex = (comparisonIndex + 1) % comparisonConditions.length;
  updateComparison();
}
document.querySelector('#comparison-next').addEventListener('click', nextComparison);
document.querySelector('#comparison-prev').addEventListener('click', () => {
  comparisonIndex = (comparisonIndex - 1 + comparisonConditions.length) % comparisonConditions.length;
  updateComparison();
});
spatialVideo.addEventListener('ended', nextComparison);

const noiseScale = document.querySelector('#noise-scale');
const noiseVideo = document.querySelector('#noise-video');
const noiseButtons = [...noiseScale.querySelectorAll('button')];
function selectNoiseScale(button) {
  for (const peer of noiseButtons) {
    peer.setAttribute('aria-pressed', String(peer === button));
  }
  const scale = button.dataset.scale;
  setMetricLabels('#noise .comparison-labels', videoMetrics.noise[scale]);
  switchVideo(noiseVideo, `static/media/presentation/noise-${scale}`,
    `PPO and CATS on Hopper with observation noise scale ${scale}`);
}
noiseScale.addEventListener('click', event => {
  const button = event.target.closest('button[data-scale]');
  if (!button || button.getAttribute('aria-pressed') === 'true') return;
  selectNoiseScale(button);
});
noiseVideo.addEventListener('ended', () => {
  const current = noiseButtons.findIndex(button => button.getAttribute('aria-pressed') === 'true');
  selectNoiseScale(noiseButtons[(current + 1) % noiseButtons.length]);
});

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent;
  const status = document.querySelector('#copy-status');
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const input = document.createElement('textarea');
      input.value = text;
      input.style.cssText = 'position:fixed;left:-9999px';
      document.body.append(input);
      input.select();
      const copied = document.execCommand('copy');
      input.remove();
      if (!copied) throw new Error('Clipboard unavailable');
    }
    status.textContent = 'Copied';
  } catch {
    status.textContent = 'Select the citation to copy.';
  }
  setTimeout(() => { status.textContent = ''; }, 2500);
});
