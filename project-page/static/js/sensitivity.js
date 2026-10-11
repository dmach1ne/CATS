'use strict';

(() => {
  const chart = document.querySelector('#sensitivity-chart');
  const buttons = [...document.querySelectorAll('.sensitivity-legend button')];
  let selected = null;

  function update() {
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button.dataset.method === selected));
      button.dataset.muted = String(selected !== null && button.dataset.method !== selected && button.dataset.method !== 'CATS');
    }
    const document = chart.contentDocument;
    if (!document) return;
    for (const element of document.querySelectorAll('[data-method]')) {
      element.dataset.muted = String(selected !== null && element.dataset.method !== selected && element.dataset.method !== 'CATS');
    }
  }

  chart.addEventListener('load', update);
  for (const button of buttons) {
    button.addEventListener('click', () => {
      selected = selected === button.dataset.method ? null : button.dataset.method;
      update();
    });
  }
  update();
})();
