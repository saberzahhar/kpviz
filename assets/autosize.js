/* Caption boxes grow to fit their text, so an exported caption is never read
   through a scrollbar. Dash replaces the textarea node on every callback, so a
   single MutationObserver on the container is cheaper and more reliable than
   re-binding listeners; sizing is a read of scrollHeight and one style write. */
(function () {
  var SEL = ".caption-box textarea";

  function fit(el) {
    if (!el) return;
    var max = 260;
    el.style.height = "auto";
    var h = Math.min(el.scrollHeight + 2, max);
    el.style.height = h + "px";
    el.style.overflowY = el.scrollHeight + 2 > max ? "auto" : "hidden";
  }

  // takes no argument on purpose: it is also used directly as an event handler
  function fitAll() {
    document.querySelectorAll(SEL).forEach(fit);
  }

  document.addEventListener("input", function (e) {
    if (e.target && e.target.matches && e.target.matches(SEL)) fit(e.target);
  });

  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; fitAll(); });
  }

  if (document.readyState !== "loading") { fitAll(); } else {
    document.addEventListener("DOMContentLoaded", fitAll);
  }
  new MutationObserver(schedule).observe(document.documentElement,
    { childList: true, subtree: true });
  window.addEventListener("resize", schedule);
})();
