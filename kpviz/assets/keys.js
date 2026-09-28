/* Keyboard, ARIA state and presentation mode — everything here is DOM
   housekeeping that needs no server round trip.

   * Tabs (WAI-ARIA tabs pattern, manual activation): Left/Right/Home/End
     move the focus along a [role=tablist]; Enter or Space opens the tab.
     aria-selected and a roving tabindex follow the active tab, and the
     current sidebar link carries aria-current="page".
   * Clickable table rows: the first cell holds the real button; a click
     elsewhere on the row presses it. An opened document scrolls into view
     (it appears below the list).
   * Hovering a series fades the others, so one model (or split, or PRMU
     class) can be followed across a crowded figure.
   * Presentation mode (P key, or ?present in the URL): larger text in the
     page *and* in every figure, for a projector at the back of a room. The
     figures are scaled in the browser (Plotly.relayout), so the server and
     the exports are untouched. Remembered per browser. */
(function () {
  var KEY = "kpviz-present";
  var SCALE = 1.3;

  // ---- ARIA state that follows the "active" classes -----------------------
  function syncAria() {
    document.querySelectorAll('[role="tab"]').forEach(function (t) {
      var on = t.classList.contains("active");
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.setAttribute("tabindex", on ? "0" : "-1");
    });
    document.querySelectorAll(".nav-link").forEach(function (a) {
      if (a.classList.contains("active")) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    // Dash's own inputs take no aria-* props: name every form field after
    // the visible label of its control (dropdowns, search, the caption box),
    // unless it already has an accessible name. Radio buttons and check
    // boxes are options: each keeps its own text, and their group takes
    // the control's label instead
    var FIELDS = "input:not([type=radio]):not([type=checkbox]), textarea, " +
                 "[role=combobox], button.dash-dropdown";
    [[".control", ".control-label"], [".exp-opt", ".exp-opt-label"]]
      .forEach(function (pair) {
        document.querySelectorAll(pair[0]).forEach(function (c) {
          var lab = c.querySelector(pair[1]);
          if (!lab) return;
          c.querySelectorAll(FIELDS).forEach(function (el) {
            if (!el.closest("[role=option], [role=listbox]")) name(el, lab.textContent);
          });
          var grp = c.querySelector(".kp-check, .segmented");
          if (grp) { grp.setAttribute("role", "group"); name(grp, lab.textContent); }
        });
      });
    document.querySelectorAll(".caption-box textarea").forEach(function (el) {
      name(el, "Figure caption used in exports");
    });
  }
  function name(el, text) {
    if (!el.getAttribute("aria-label") && !el.getAttribute("aria-labelledby") &&
        !(el.id && document.querySelector('label[for="' + CSS.escape(el.id) + '"]'))) {
      el.setAttribute("aria-label", (text || "").trim());
    }
  }

  // ---- keyboard ------------------------------------------------------------
  document.addEventListener("keydown", function (ev) {
    var t = ev.target;
    if (t && t.getAttribute && t.getAttribute("role") === "tab") {
      var list = Array.prototype.slice.call(
        t.closest('[role="tablist"]').querySelectorAll('[role="tab"]'));
      var i = list.indexOf(t), j = null;
      if (ev.key === "ArrowRight") j = (i + 1) % list.length;
      else if (ev.key === "ArrowLeft") j = (i - 1 + list.length) % list.length;
      else if (ev.key === "Home") j = 0;
      else if (ev.key === "End") j = list.length - 1;
      if (j !== null) { ev.preventDefault(); list[j].focus(); }
      return;
    }
    // P toggles presentation mode, unless the user is typing
    if ((ev.key === "p" || ev.key === "P") && !ev.ctrlKey && !ev.metaKey &&
        !ev.altKey && t && !/INPUT|TEXTAREA|SELECT/.test(t.tagName) &&
        !t.isContentEditable) {
      setPresent(!document.body.classList.contains("present"));
    }
  });

  // a click anywhere on a clickable table row presses the button in its
  // first cell (the button is what the keyboard and screen readers reach)
  document.addEventListener("click", function (ev) {
    var t = ev.target;
    if (!t || !t.closest || t.closest("button, a, input, label, summary")) return;
    var row = t.closest("tr.row-click");
    if (!row || (window.getSelection && String(window.getSelection()))) return;
    var b = row.querySelector(".row-open");
    if (b) b.click();
  });

  // ---- hover: the series under the pointer stays, the others fade ---------
  // A series is its legend group (a model and all its runs, a split, a PRMU
  // class) or, without one, its name. Only plots with two or more series
  // take part; heatmaps and pies keep their own hover.
  var FADE = 0.22;
  function seriesKey(t) { return t.legendgroup || t.name || ""; }
  function focusSeries(gd, ev) {
    if (!window.Plotly || !gd.data) return;
    var data = gd.data, keys = {}, n = 0;
    data.forEach(function (t) {
      if ((t.type === "scatter" || t.type === "bar" || !t.type) &&
          t.showlegend !== false || t.legendgroup) {
        var k = seriesKey(t);
        if (k && !keys[k]) { keys[k] = 1; n++; }
      }
    });
    if (n < 2) return;
    var want = null;
    if (ev && ev.points && ev.points.length) {
      var tr = data[ev.points[0].curveNumber];
      if (!tr || (tr.type && tr.type !== "scatter" && tr.type !== "bar")) return;
      want = seriesKey(tr) || null;
    }
    if (want === gd._kpFocus) return;
    gd._kpFocus = want;
    if (!gd._kpOpacity) {
      gd._kpOpacity = data.map(function (t) {
        return t.opacity === undefined ? 1 : t.opacity;
      });
    }
    var base = gd._kpOpacity;
    var op = data.map(function (t, i) {
      var o = base[i] === undefined ? 1 : base[i];
      if (want === null || seriesKey(t) === want || !seriesKey(t)) return o;
      return o * FADE;
    });
    window.Plotly.restyle(gd, { opacity: op });
  }

  // ---- presentation mode ---------------------------------------------------
  function scaleGraph(gd) {
    if (!window.Plotly || !gd || !gd.layout) return;
    var on = document.body.classList.contains("present");
    var base = gd.layout.font && gd.layout.font.size;
    if (!base) return;
    // _kpScaled is the font size this script set; a server re-render
    // (Plotly.react) brings the original size back and resets the question
    if (on && gd._kpScaled === base) return;           // already scaled
    if (!on) {
      if (gd._kpScaled !== base) { gd._kpScaled = 0; return; }   // not ours
    }
    var f = on ? SCALE : 1 / SCALE;
    var upd = {};
    function sz(path, v) { if (v) upd[path] = Math.round(v * f * 10) / 10; }
    sz("font.size", base);
    ["xaxis", "yaxis"].forEach(function (ax) {
      var a = gd.layout[ax];
      if (!a) return;
      sz(ax + ".tickfont.size", a.tickfont && a.tickfont.size);
      sz(ax + ".title.font.size", a.title && a.title.font && a.title.font.size);
    });
    var leg = gd.layout.legend || {};
    sz("legend.font.size", leg.font && leg.font.size);
    sz("legend.title.font.size", leg.title && leg.title.font && leg.title.font.size);
    // point labels and window labels are annotations
    (gd.layout.annotations || []).forEach(function (a, i) {
      sz("annotations[" + i + "].font.size", a.font && a.font.size);
    });
    gd._kpScaled = on ? Math.round(base * f * 10) / 10 : 0;
    window.Plotly.relayout(gd, upd);
  }
  function scaleAll() {
    document.querySelectorAll(".js-plotly-plot").forEach(scaleGraph);
  }
  function setPresent(on) {
    document.body.classList.toggle("present", on);
    try { localStorage.setItem(KEY, on ? "1" : "0"); } catch (e) { /* private */ }
    scaleAll();
  }

  // ---- one observer for all of the above -----------------------------------
  var lastDoc = "";
  var pending = false;
  function onChange() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () {
      pending = false;
      syncAria();
      document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
        if (!gd._kpHooked && gd.on) {
          gd._kpHooked = true;
          gd.on("plotly_afterplot", function () {
            if (document.body.classList.contains("present")) scaleGraph(gd);
          });
          gd.on("plotly_react", function () { gd._kpOpacity = null; gd._kpFocus = null; });
          gd.on("plotly_hover", function (ev) { focusSeries(gd, ev); });
          gd.on("plotly_unhover", function () { focusSeries(gd, null); });
        }
      });
      var doc = document.getElementById("ds-doc-view");
      var id = doc && doc.firstElementChild ? doc.textContent.slice(0, 80) : "";
      if (id && id !== lastDoc) {
        var calm = window.matchMedia &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        doc.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
      }
      lastDoc = id;
    });
  }

  function start() {
    var want = /[?&]present\b/.test(window.location.search);
    var stored = null;
    try { stored = localStorage.getItem(KEY); } catch (e) { /* private */ }
    if (want || stored === "1") document.body.classList.add("present");
    new MutationObserver(onChange).observe(document.body,
      { childList: true, subtree: true, attributes: true,
        attributeFilter: ["class"] });
    onChange();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
