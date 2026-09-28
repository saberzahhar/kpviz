/* Routing and visibility, in the browser.

   Every page and workbench stays mounted (control state survives navigation),
   but only the visible ones may compute: this callback turns the URL and the
   Insights tab into one boolean store per page and per workbench (vis-*,
   read by server callbacks as State) and one "shown" counter each (shown-*,
   their Input), bumped when it appears. Leaving a page therefore wakes no
   callback, and showing one workbench wakes one set of callbacks. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  kpviz: {
    route: function (pathname, activeRq) {
      var pages = ["/", "/datasets", "/models", "/architectures", "/insights"];
      var rqs = ["rq1", "rq2", "rq3", "rq4", "rq5"];
      var nPages = pages.length, nRqs = rqs.length;
      var states = Array.prototype.slice.call(arguments, 2);
      var visSt = states.slice(0, nPages + nRqs);        // vis-* (booleans)
      var shownSt = states.slice(nPages + nRqs);         // shown-* (counters)
      var path = pages.indexOf(pathname) >= 0 ? pathname : "/";
      var noUp = window.dash_clientside.no_update;
      var styles = pages.map(function (p) {
        return { display: p === path ? "block" : "none" };
      });
      var classes = pages.map(function (p) {
        return p === path ? "nav-link active" : "nav-link";
      });
      var want = pages.map(function (p) { return p === path; }).concat(
        rqs.map(function (rq) {
          return path === "/insights" && (activeRq || "rq4") === rq;
        }));
      // vis-* follow visibility both ways (servers read them as State);
      // shown-* are bumped only when something becomes visible — the one
      // Input that wakes a page, so hiding a page costs no request at all
      var vis = want.map(function (v, i) { return v === visSt[i] ? noUp : v; });
      var shown = want.map(function (v, i) {
        return (v && !visSt[i]) ? (shownSt[i] || 0) + 1 : noUp;
      });
      return styles.concat(classes, vis, shown);
    },
    tabs: function () {
      /* Insights tab buttons: the clicked one becomes active */
      var ctx = window.dash_clientside.callback_context;
      var rqs = ["rq1", "rq2", "rq3", "rq4", "rq5"];
      var active = "rq4";
      if (ctx && ctx.triggered && ctx.triggered.length &&
          ctx.triggered[0].prop_id !== ".") {
        active = ctx.triggered[0].prop_id.split(".")[0].replace("tab-", "");
      }
      var styles = rqs.map(function (k) {
        return { display: k === active ? "block" : "none" };
      });
      var classes = rqs.map(function (k) {
        return k === active ? "rq-tab active" : "rq-tab";
      });
      return [active].concat(styles, classes);
    }
  }
});
