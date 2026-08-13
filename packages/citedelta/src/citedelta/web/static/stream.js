// Attach to a pending turn shell: stream phases in, swap the finished turn.
function citedeltaStream(el) {
  const turnId = el.dataset.turnId;
  const phases = el.querySelector('[data-phases]');
  const src = new EventSource(`/ui/turn/${turnId}/stream`);

  src.addEventListener('phase', (e) => {
    // Mark every prior phase done — arrival of a new one IS the completion
    // signal for the last, so the server never has to send two events per step.
    phases.querySelectorAll('.phase.now').forEach((p) => {
      p.classList.replace('now', 'done');
      p.querySelector('.mk').textContent = '✓';
    });
    const row = document.createElement('div');
    row.className = 'phase now';
    row.innerHTML = '<span class="mk">▸</span><span></span>';
    row.lastElementChild.textContent = JSON.parse(e.data);
    phases.appendChild(row);
  });

  src.addEventListener('done', (e) => {
    src.close();
    el.outerHTML = JSON.parse(e.data);
    document.querySelector('[data-composer] input')?.focus();
  });

  src.addEventListener('error', () => {
    src.close();
    phases.innerHTML =
      '<div class="phase"><span class="mk">·</span>' +
      '<span>Lost connection while working. Ask again?</span></div>';
    document.querySelectorAll('[data-composer] button').forEach((b) => (b.disabled = false));
  });
}

document.body.addEventListener('htmx:afterSwap', (e) => {
  e.target.querySelectorAll?.('[data-turn-id]').forEach(citedeltaStream);
});
