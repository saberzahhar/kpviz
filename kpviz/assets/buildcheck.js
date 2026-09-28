/* A tab holds the callback signatures of the build that rendered it. Restart
   the server with a changed layout and that tab posts callbacks the new process
   never registered — every poll tick becomes a 500. Comparing the server's
   build id against the one seen at load time lets the tab reload itself.

   The same tiny request keeps every tab's catalog current: a scan started
   in another tab (or from the command line) turns this tab's scan poll on
   while it runs, and a newer catalog version is written into its
   `catalog-version` store — the visible workbench then re-renders once,
   instead of showing figures from the previous catalog until a reload.

   Baseline comes from the first successful fetch rather than from the served
   HTML, so this file works unchanged in any build. */
(function () {
  var seen = null;
  var catalog = null;
  var loadedAt = Date.now();

  function setProps(id, props) {
    var dc = window.dash_clientside;
    if (dc && typeof dc.set_props === "function") {
      try { dc.set_props(id, props); } catch (e) { /* not mounted yet */ }
    }
  }

  function check() {
    fetch("kpviz-build", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.build) return;
        if (seen === null) { seen = j.build; catalog = j.catalog; }
        // give the page a moment to settle so a reload can never loop tightly
        if (j.build !== seen && Date.now() - loadedAt > 5000) {
          console.info("KPViz: server rebuilt (" + seen + " -> " + j.build +
                       "), reloading this tab");
          window.location.reload();
          return;
        }
        if (j.scanning) {
          setProps("scan-poll", { disabled: false });   // the poll takes over
        } else if (j.catalog !== catalog) {
          catalog = j.catalog;
          setProps("catalog-version", { data: j.catalog });
        }
      })
      .catch(function () { /* server down / restarting: try again later */ });
  }

  // on load, when the tab comes back, and every 20 s while it is visible
  // (one small GET; nothing while hidden)
  check();
  setInterval(function () {
    if (document.visibilityState === "visible") check();
  }, 20000);
  window.addEventListener("focus", check);
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") check();
  });
})();
