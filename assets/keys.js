/* Keyboard activation for clickable table rows (role="button"): Enter or
   Space on a focused row does what a click does. */
document.addEventListener("keydown", function (ev) {
  var t = ev.target;
  if (!t || !t.classList || !t.classList.contains("row-click")) return;
  if (ev.key === "Enter" || ev.key === " ") {
    ev.preventDefault();
    t.click();
  }
});
