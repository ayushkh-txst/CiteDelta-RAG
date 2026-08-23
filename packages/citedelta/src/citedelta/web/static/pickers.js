// Date pickers for the controls that htmx does not drive directly:
//   • the composer's "as of {date}" button        (data-asof-open)
//   • the compare picker's "another date…" option (data-compare-other)
// Both open a native date input anchored under the control. Picking a date
// updates the as-of the next question runs at, or fills in the compare
// dropdown's selection — it does NOT submit anything itself. The compare
// form's own "Compare" button is the only thing that fires the request, so
// picking a date is reversible and never fires a request by accident.
(function () {
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function fmt(iso) {
    const [y, m, d] = iso.split('-').map(Number);
    return `${d} ${MONTHS[m - 1]} ${y}`;
  }

  // Open a native date calendar anchored to `anchor`, call onPick(iso) on
  // choice or onCancel() if dismissed without one. The input is laid
  // invisibly *over* the anchor so the browser anchors its calendar popup to
  // the control. showPicker() must run inside the same click/change
  // gesture, so we force a synchronous layout flush first (otherwise the
  // popup anchors to the input's pre-positioned 0,0 origin and appears
  // top-left).
  function pickDate(anchor, { min, max, value }, onPick, onCancel) {
    document.querySelectorAll('.date-pop').forEach((el) => el.remove());

    const input = document.createElement('input');
    input.type = 'date';
    input.className = 'date-pop';
    if (min) input.min = min;
    if (max) input.max = max;
    if (value) input.value = value;

    const r = anchor.getBoundingClientRect();
    Object.assign(input.style, {
      position: 'absolute',
      left: `${window.scrollX + r.left}px`,
      top: `${window.scrollY + r.top}px`,
      width: `${r.width}px`,
      height: `${r.height}px`,
      margin: '0',
      padding: '0',
      border: '0',
      opacity: '0',
      zIndex: '1000',
    });
    document.body.appendChild(input);

    let done = false;
    const finish = (keep) => {
      if (done) return;
      done = true;
      if (keep && input.value) onPick(input.value);
      else if (onCancel) onCancel();
      input.remove();
    };
    input.addEventListener('change', () => finish(true));
    input.addEventListener('blur', () => finish(false));

    void input.offsetHeight; // force layout so the popup anchors to the styled box
    input.focus({ preventScroll: true });
    if (input.showPicker) {
      try { input.showPicker(); } catch (_) { /* focus already opened it */ }
    }
  }

  document.addEventListener('click', (e) => {
    const asof = e.target.closest('[data-asof-open]');
    if (asof) {
      e.preventDefault();
      const form = asof.closest('form');
      const hidden = form && form.querySelector('[data-asof-value]');
      const label = asof.querySelector('.v');
      pickDate(
        asof,
        { min: asof.dataset.asofMin, max: asof.dataset.asofMax, value: hidden && hidden.value },
        (iso) => {
          if (hidden) hidden.value = iso;
          if (label) label.textContent = fmt(iso);
        },
      );
    }
  });

  // The compare dropdown's "another date…" option opens the same native
  // picker rather than submitting anything. Picking a date inserts it as a
  // real option (so the dropdown visibly shows what's selected) and leaves
  // it selected; the form's own "Compare" button is what actually submits.
  // Cancelling without a pick reverts to whatever was selected before, so
  // "another date…" never gets left selected as if it were a real choice.
  document.addEventListener('change', (e) => {
    const select = e.target.closest('[data-compare-select]');
    if (!select || select.value !== '__other__') return;

    const otherOption = select.querySelector('[data-compare-other]');
    const previousValue = select.dataset.prevValue || '';
    const asofBtn = document.querySelector('[data-asof-open]');
    const bounds = asofBtn ? asofBtn.dataset : {};

    pickDate(
      select,
      { min: bounds.asofMin, max: bounds.asofMax },
      (iso) => {
        let opt = select.querySelector(`option[value="${iso}"]`);
        if (!opt) {
          opt = document.createElement('option');
          opt.value = iso;
          select.insertBefore(opt, otherOption);
        }
        opt.textContent = fmt(iso);
        select.value = iso;
        select.dataset.prevValue = iso;
      },
      () => { select.value = previousValue; },
    );
  });

  // Track the last real selection so a cancelled "another date…" pick has
  // something sane to revert to.
  document.addEventListener(
    'change',
    (e) => {
      const select = e.target.closest('[data-compare-select]');
      if (select && select.value !== '__other__') select.dataset.prevValue = select.value;
    },
    true,
  );
})();
