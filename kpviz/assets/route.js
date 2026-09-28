/* Routing and visibility, in the browser.

   Every page and workbench stays mounted (control state survives navigation),
   but only the visible ones may compute: this callback turns the URL into one
   boolean store per page and per workbench (vis-*, read by server callbacks
   as State) and one "shown" counter each (shown-*, their Input), bumped when
   it appears. Leaving a page therefore wakes no callback, and showing one
   workbench wakes one set of callbacks.

   The Insights workbench lives in the URL hash (/insights#rq3): a reload,
   the Back button and a pasted link all open the same workbench. Without a
   hash, the last one used in this tab opens (else the cost-quality one). */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  kpviz: {
    route: function (pathname, hash, lastRq) {
      var pages = ["/", "/datasets", "/models", "/architectures", "/insights"];
      var rqs = ["rq1", "rq2", "rq3", "rq4", "rq5"];
      var nPages = pages.length, nRqs = rqs.length;
      var states = Array.prototype.slice.call(arguments, 3);
      var visSt = states.slice(0, nPages + nRqs);        // vis-* (booleans)
      var shownSt = states.slice(nPages + nRqs);         // shown-* (counters)
      var path = pages.indexOf(pathname) >= 0 ? pathname : "/";
      var fromHash = (hash || "").replace("#", "");
      var active = rqs.indexOf(fromHash) >= 0 ? fromHash
                 : (rqs.indexOf(lastRq) >= 0 ? lastRq : "rq4");
      var noUp = window.dash_clientside.no_update;
      var styles = pages.map(function (p) {
        return { display: p === path ? "block" : "none" };
      });
      var classes = pages.map(function (p) {
        return p === path ? "nav-link active" : "nav-link";
      });
      var want = pages.map(function (p) { return p === path; }).concat(
        rqs.map(function (rq) {
          return path === "/insights" && active === rq;
        }));
      // vis-* follow visibility both ways (servers read them as State);
      // shown-* are bumped only when something becomes visible — the one
      // Input that wakes a page, so hiding a page costs no request at all
      var vis = want.map(function (v, i) { return v === visSt[i] ? noUp : v; });
      var shown = want.map(function (v, i) {
        return (v && !visSt[i]) ? (shownSt[i] || 0) + 1 : noUp;
      });
      var panels = rqs.map(function (k) {
        return { display: k === active ? "block" : "none" };
      });
      var tabs = rqs.map(function (k) {
        return k === active ? "rq-tab active" : "rq-tab";
      });
      return styles.concat(classes, vis, shown, panels, tabs,
                           [active === lastRq ? noUp : active]);
    },
    tabs: function () {
      /* a tab click writes the hash; route() does the rest */
      var ctx = window.dash_clientside.callback_context;
      if (!ctx || !ctx.triggered || !ctx.triggered.length ||
          ctx.triggered[0].prop_id === ".") {
        return window.dash_clientside.no_update;
      }
      return "#" + ctx.triggered[0].prop_id.split(".")[0].replace("tab-", "");
    }
  }
});
