/* Routing and visibility, in the browser.

   Every page and workbench stays mounted (control state survives navigation),
   but only the visible ones may compute: this callback turns the URL and the
   Insights tab into one boolean store per page and per workbench. Server
   callbacks listen to their own store, so a hidden page costs nothing, and a
   store is only written when its value actually changes (no_update
   otherwise), so showing one workbench wakes one set of callbacks. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  kpviz: {
    route: function (pathname, activeRq) {
      var pages = ["/", "/datasets", "/models", "/architectures", "/insights"];
      var rqs = ["rq1", "rq2", "rq3", "rq4", "rq5"];
      var nPages = pages.length;
      var states = Array.prototype.slice.call(arguments, 2);
      var path = pages.indexOf(pathname) >= 0 ? pathname : "/";
      var noUp = window.dash_clientside.no_update;
      var styles = pages.map(function (p) {
        return { display: p === path ? "block" : "none" };
      });
      var classes = pages.map(function (p) {
        return p === path ? "nav-link active" : "nav-link";
      });
      var vis = pages.map(function (p, i) {
        var v = p === path;
        return v === states[i] ? noUp : v;
      });
      var rqVis = rqs.map(function (rq, i) {
        var v = path === "/insights" && (activeRq || "rq4") === rq;
        return v === states[nPages + i] ? noUp : v;
      });
      return styles.concat(classes, vis, rqVis);
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
