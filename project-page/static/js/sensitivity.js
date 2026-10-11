'use strict';

(() => {
  const chart = document.querySelector('#sensitivity-chart');
  const buttons = [...document.querySelectorAll('.sensitivity-legend button[data-method]')];
  const selected = new Set();

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

  chart.addEventListener('load', update);
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const method = button.dataset.method;
      if (selected.has(method)) selected.delete(method);
      else selected.add(method);
      update();
    });
  }
  document.querySelector('#sensitivity-reset').addEventListener('click', () => {
    selected.clear();
    update();
  });
  update();
})();
