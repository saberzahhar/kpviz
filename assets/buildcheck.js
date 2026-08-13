/* A tab holds the callback signatures of the build that rendered it. Restart
   the server with a changed layout and that tab posts callbacks the new process
   never registered — every poll tick becomes a 500. Comparing the server's
   build id against the one seen at load time lets the tab reload itself.

   Baseline comes from the first successful fetch rather than from the served
   HTML, so this file works unchanged in any build. */
(function () {
  var seen = null;
  var loadedAt = Date.now();

  function check() {
    fetch("kpviz-build", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.build) return;
        if (seen === null) { seen = j.build; return; }
        // give the page a moment to settle so a reload can never loop tightly
        if (j.build !== seen && Date.now() - loadedAt > 5000) {
          console.info("KPViz: server rebuilt (" + seen + " -> " + j.build +
                       "), reloading this tab");
          window.location.reload();
        }
      })
      .catch(function () { /* server down / restarting: try again later */ });
  }

  check();
  setInterval(check, 4000);
})();
