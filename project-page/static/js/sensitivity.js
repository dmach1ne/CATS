'use strict';

(() => {
  const chart = document.querySelector('#sensitivity-chart');
  const buttons = [...document.querySelectorAll('.sensitivity-legend button[data-method]')];
  const selected = new Set();
  let revealRects = [];
  let revealFrame = 0;

  function prepareReveal(document) {
    const svg = document.documentElement;
    const namespace = 'http://www.w3.org/2000/svg';
    const measured = [...document.querySelectorAll('[data-method]')].map(element => {
      const box = element.getBBox();
      const matrix = svg.getCTM().inverse().multiply(element.getCTM());
      const corners = [[box.x, box.y], [box.x + box.width, box.y + box.height]]
        .map(([x, y]) => new DOMPoint(x, y).matrixTransform(matrix));
      const left = Math.min(...corners.map(point => point.x));
      const right = Math.max(...corners.map(point => point.x));
      return {element, left, right, center: (left + right) / 2};
    });
    const centers = [];
    for (const item of measured.filter(item => item.right - item.left > svg.viewBox.baseVal.width / 10).sort((a, b) => a.center - b.center)) {
      if (!centers.some(center => Math.abs(center - item.center) < 10)) centers.push(item.center);
    }
    const columns = centers.map(center => ({center, elements: [], left: Infinity, right: -Infinity}));
    for (const {element, left, right, center} of measured) {
      const column = columns.reduce((nearest, candidate) => Math.abs(candidate.center - center) < Math.abs(nearest.center - center) ? candidate : nearest);
      column.elements.push(element);
      column.left = Math.min(column.left, left - 2);
      column.right = Math.max(column.right, right + 2);
    }
    const defs = svg.querySelector('defs');
    revealRects = columns.map((column, index) => {
      const clip = document.createElementNS(namespace, 'clipPath');
      clip.id = `comparison-reveal-${index}`;
      clip.setAttribute('clipPathUnits', 'userSpaceOnUse');
      const rect = document.createElementNS(namespace, 'rect');
      rect.setAttribute('x', column.left);
      rect.setAttribute('y', 0);
      rect.setAttribute('height', svg.viewBox.baseVal.height);
      rect.setAttribute('width', column.right - column.left);
      clip.append(rect);
      defs.append(clip);
      for (const element of column.elements) {
        const wrapper = document.createElementNS(namespace, 'g');
        wrapper.dataset.revealPanel = String(index);
        wrapper.setAttribute('clip-path', `url(#${clip.id})`);
        element.before(wrapper);
        wrapper.append(element);
      }
      return {rect, width: column.right - column.left};
    });
  }

  function reveal() {
    cancelAnimationFrame(revealFrame);
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      for (const {rect, width} of revealRects) rect.setAttribute('width', width);
      return;
    }
    for (const {rect} of revealRects) rect.setAttribute('width', 0);
    const start = performance.now();
    function step(now) {
      const progress = Math.min(1, (now - start) / 800);
      for (const {rect, width} of revealRects) rect.setAttribute('width', width * progress);
      if (progress < 1) revealFrame = requestAnimationFrame(step);
    }
    revealFrame = requestAnimationFrame(step);
  }

  function muted(method) {
    return selected.size > 0 && !selected.has(method) && method !== 'CATS';
  }

  function update() {
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(selected.has(button.dataset.method) || button.dataset.method === 'CATS'));
      button.dataset.muted = String(muted(button.dataset.method));
    }
    const document = chart.contentDocument;
    if (!document) return;
    for (const element of document.querySelectorAll('[data-method]')) {
      element.dataset.muted = String(muted(element.dataset.method));
    }
  }

  chart.addEventListener('load', () => {
    prepareReveal(chart.contentDocument);
    update();
  });
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const method = button.dataset.method;
      if (selected.has(method)) selected.delete(method);
      else selected.add(method);
      update();
      reveal();
    });
  }
  document.querySelector('#sensitivity-reset').addEventListener('click', () => {
    selected.clear();
    update();
    reveal();
  });
  update();
})();
