"""Dash application assembly: frame, routing, global export callbacks.

Latency rules, each from a measured trace (docs/PERFORMANCE_PLAN.md):

* **Only what is on screen computes.** Every page and workbench stays
  mounted (so control state survives navigation), but a clientside callback
  turns the URL and the Insights tab into one visibility store per page and
  per workbench; server callbacks listen to their own store and do nothing
  while hidden. The first load used to fire ~100 requests and settle after
  28 s because five hidden workbenches computed at once.
* **No polling while idle.** One interval, enabled only while a scan runs.
* **Routing and tab switches never reach the server.**
* **Nothing slow on the request path.** The TeX probe runs once in a
  background thread at start-up and is persisted.
"""
from __future__ import annotations

import collections
import threading
import time
import uuid

import dash
from dash import (ALL, MATCH, ClientsideFunction, Input, Output, State, ctx,
                  dcc, html, no_update)

from . import __version__, db, scanner
from .config import settings
from .figures import geometry
from .export import (export_bundle, export_name, fig_pdf, fig_pgf, fig_png,
                     figure_env, snippets, start_tex_probe)

# Identifies this server process. A browser tab holds the callback signatures
# of the build that rendered it, so a tab left open across an upgrade posts a
# callback the new process has never heard of — Dash answers 500 once per poll
# tick, forever. assets/buildcheck.js compares this value and reloads such a
# tab by itself; _install_stale_guard keeps the log clean meanwhile.
BUILD_ID = f"{__version__}+{uuid.uuid4().hex[:8]}"

NAV = [
    ("/", "Overview", "home"),
    ("/datasets", "Datasets", "data"),
    ("/models", "Models", "model"),
    ("/architectures", "Architectures", "chip"),
    ("/insights", "Insights", "chart"),
]
PAGES = [href for href, _l, _i in NAV]
RQS = ["rq1", "rq2", "rq3", "rq4", "rq5"]


def page_key(href: str) -> str:
    return href.strip("/") or "home"


def vis_id(name: str) -> str:
    """Visibility store of a page ('home', 'datasets', …) or workbench."""
    return f"vis-{name}"


def _sidebar():
    root = settings().data_root
    return html.Div([
        html.A("Skip to content", href="#content", className="skip-link"),
        html.Div([
            html.Div(["KP", html.Span("Viz")], className="brand-name"),
            html.Div("keyphrase evaluation, from runs to paper",
                     className="brand-sub"),
        ], className="brand"),
        html.Nav([dcc.Link([html.Span(className=f"nav-ico i-{ico}",
                                      **{"aria-hidden": "true"}), label],
                           href=href, id=f"nav-{page_key(href)}",
                           className="nav-link") for href, label, ico in NAV],
                 className="nav", **{"aria-label": "Pages"}),
        html.Div([
            # the catalog state first (what a glance should tell), then the
            # data root by its last component — the full path is a tooltip
            html.Div(id="sidebar-scan", children=_sidebar_status(),
                     role="status", **{"aria-live": "polite"}),
            html.Div(root.name or str(root), className="sidebar-root",
                     title=str(root)),
        ], className="sidebar-foot"),
    ], className="sidebar")


def _sidebar_status():
    snap = scanner.STATE.snapshot()
    if snap["running"]:
        from .util import human_duration
        return [html.Span(className="scan-dot busy"),
                f"scanning — ETA {human_duration(snap.get('eta_s'))}"]
    if snap.get("error"):
        return [html.Span(className="scan-dot err"), "last scan failed"]
    if snap.get("cancelled"):
        return [html.Span(className="scan-dot warn"), "last scan cancelled"]
    v = db.scan_version()
    return [html.Span(className="scan-dot ok" if v else "scan-dot"),
            f"catalog v{v}" if v else "no scan yet"]


def build_app() -> dash.Dash:
    from .pages import architectures, datasets, home, insights, models

    from pathlib import Path
    assets = Path(__file__).resolve().parent / "assets"
    app = dash.Dash(__name__, title="KPViz",
                    assets_folder=str(assets),
                    suppress_callback_exceptions=True,
                    # no "Updating…" flicker in the browser tab on every click
                    update_title=None)
    start_tex_probe()
    db.mark_missing_optional_modules()
    from .metrics import enable_warm, warm_async
    enable_warm()
    scanner.enable_preload()
    if db.scan_version():
        warm_async()

    pages = {
        "/": home.layout, "/datasets": datasets.layout,
        "/models": models.layout, "/architectures": architectures.layout,
        "/insights": insights.layout,
    }

    def layout():
        """Built per page load, so the catalog version, the scan state and
        every picker's initial options are current without a round trip."""
        running = scanner.STATE.running
        return html.Div([
            dcc.Location(id="url"),
            dcc.Store(id="catalog-version", data=db.scan_version()),
            dcc.Store(id="ins-active", data="rq4"),
            *[dcc.Store(id=vis_id(page_key(h)), data=False) for h in PAGES],
            *[dcc.Store(id=vis_id(rq), data=False) for rq in RQS],
            # bumped each time a page / workbench is shown: the only thing
            # that wakes it (hiding one writes vis-* = False, a State only)
            *[dcc.Store(id=f"shown-{page_key(h)}", data=0) for h in PAGES],
            *[dcc.Store(id=f"shown-{rq}", data=0) for rq in RQS],
            dcc.Interval(id="scan-poll", interval=1000, n_intervals=0,
                         disabled=not running),
            _sidebar(),
            html.Main([
                html.Div([html.Div(pages[href](), id=f"page-{page_key(href)}",
                                   style={"display": "none"})
                          for href in pages]),
            ], className="main", id="content", tabIndex=-1),
        ], className="app-frame")

    app.layout = layout

    _install_build_route(app)
    _install_stale_guard(app)
    _install_timing(app)

    # ---- routing + visibility: entirely in the browser -------------------
    # the Insights workbench is the URL hash (/insights#rq3): reloads, the
    # Back button and pasted links all open the same one
    app.clientside_callback(
        ClientsideFunction(namespace="kpviz", function_name="route"),
        [Output(f"page-{page_key(h)}", "style") for h in PAGES]
        + [Output(f"nav-{page_key(h)}", "className") for h in PAGES]
        + [Output(vis_id(page_key(h)), "data") for h in PAGES]
        + [Output(vis_id(rq), "data") for rq in RQS]
        + [Output(f"shown-{page_key(h)}", "data") for h in PAGES]
        + [Output(f"shown-{rq}", "data") for rq in RQS]
        + [Output(f"panel-{rq}", "style") for rq in RQS]
        + [Output(f"tab-{rq}", "className") for rq in RQS]
        + [Output("ins-active", "data")],
        Input("url", "pathname"), Input("url", "hash"),
        State("ins-active", "data"),
        [State(vis_id(page_key(h)), "data") for h in PAGES]
        + [State(vis_id(rq), "data") for rq in RQS]
        + [State(f"shown-{page_key(h)}", "data") for h in PAGES]
        + [State(f"shown-{rq}", "data") for rq in RQS])

    # ---- page modules' own callbacks
    home.register(app)
    datasets.register(app)
    models.register(app)
    architectures.register(app)
    insights.register(app)

    _register_exports(app)
    return app


# ---------------------------------------------------------------------------
# Surviving a restart with tabs open
# ---------------------------------------------------------------------------

def _install_build_route(app):
    @app.server.route("/kpviz-build")
    def _kpviz_build():
        """Build id (a restarted server with a new layout reloads old tabs),
        plus the catalog version and scan state, so a tab that did not start
        a scan still learns that one ran (assets/buildcheck.js)."""
        from flask import jsonify
        return jsonify({"build": BUILD_ID, "catalog": db.scan_version(),
                        "scanning": bool(scanner.STATE.running)})


def _install_stale_guard(app):
    """Answer a callback this build does not have with "nothing changed".

    Dash looks the callback up as `callback_map[body["output"]]`; the same
    lookup here tells us the caller is a tab from another build *before* the
    KeyError becomes a 500 with a traceback per poll tick. One log line, then
    silence — the tab reloads itself through assets/buildcheck.js."""
    warned: set[str] = set()

    @app.server.before_request
    def _guard():
        from flask import jsonify, request
        if request.method != "POST" or \
                not request.path.endswith("_dash-update-component"):
            return None
        body = request.get_json(silent=True) or {}
        out = body.get("output")
        if not out or out in app.callback_map:
            return None
        if out not in warned:
            warned.add(out)
            print(f"· a browser tab from an older build is polling this server "
                  f"(callback {out[:56]}…) — it will reload itself; "
                  f"press Ctrl-Shift-R if it does not")
        return jsonify({"multi": True, "response": {}})


# ---------------------------------------------------------------------------
# Per-callback server timing: /kpviz-perf names the slow callback without a
# browser (p50 / p95 / max per output, response bytes, current cache use)
# ---------------------------------------------------------------------------
_TIMES: dict[str, collections.deque] = {}
_TIMES_LOCK = threading.Lock()


def _install_timing(app):
    from flask import g, jsonify, request

    @app.server.before_request
    def _t0():
        g._kpviz_t0 = time.perf_counter()

    @app.server.after_request
    def _t1(resp):
        t0 = getattr(g, "_kpviz_t0", None)
        if t0 is not None and request.path.endswith("_dash-update-component"):
            body = request.get_json(silent=True) or {}
            out = str(body.get("output") or "?")[:90]
            dt = time.perf_counter() - t0
            with _TIMES_LOCK:
                _TIMES.setdefault(out, collections.deque(maxlen=200)).append(
                    (dt, resp.calculate_content_length() or 0))
            if dt > 0.5:
                print(f"· slow callback {dt:.2f}s  {out[:70]}")
        return resp

    @app.server.route("/kpviz-perf")
    def _perf():
        from .metrics import cache_stats
        rows = []
        with _TIMES_LOCK:
            for out, dq in _TIMES.items():
                ts = sorted(d for d, _b in dq)
                rows.append({"output": out, "n": len(ts),
                             "p50_ms": round(1000 * ts[len(ts) // 2], 1),
                             "p95_ms": round(1000 * ts[min(len(ts) - 1, int(len(ts) * 0.95))], 1),
                             "max_ms": round(1000 * ts[-1], 1),
                             "kb": round(sum(b for _d, b in dq) / len(dq) / 1024, 1)})
        rows.sort(key=lambda r: -r["p95_ms"])
        rss = None
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        except Exception:
            pass
        from .metrics import warm_status
        return jsonify({"callbacks": rows, "cache": cache_stats(),
                        "warm": warm_status(),
                        "peak_rss_mb": rss, "catalog_version": db.scan_version()})


# ---------------------------------------------------------------------------
# Export callbacks (shared by every insight workbench)
# ---------------------------------------------------------------------------

_EXP_OPTS = ("venue", "span", "height", "legend", "cells")


def _with_opts(spec: dict, *vals) -> dict:
    """The figure spec plus the export options chosen under it."""
    opts = {k: v for k, v in zip(_EXP_OPTS, vals) if v}
    out = dict(spec)
    if opts:
        out["export"] = opts
    return out


def _register_exports(app):
    opt_in = [Input({"type": f"exp-{k}", "rq": MATCH}, "value") for k in _EXP_OPTS]
    opt_st = [State({"type": f"exp-{k}", "rq": MATCH}, "value") for k in _EXP_OPTS]

    @app.callback(
        Output({"type": "exp-clip-fig", "rq": MATCH}, "content"),
        Output({"type": "exp-clip-tab", "rq": MATCH}, "content"),
        Output({"type": "exp-hint", "rq": MATCH}, "children"),
        Input({"type": "fig-spec", "rq": MATCH}, "data"),
        Input({"type": "caption", "rq": MATCH}, "n_blur"),
        *opt_in,
        State({"type": "caption", "rq": MATCH}, "value"),
        prevent_initial_call=True)
    def clipboards(spec, _blur, *rest):
        """The two LaTeX snippets. Runs when the figure, the caption (on
        blur) or an export option changes — never per keystroke, never
        waiting on TeX (the probe result is read, not computed, here)."""
        *opts, caption = rest
        if not spec:
            return "", "", ""
        return snippets(_with_opts(spec, *opts), caption)

    # the caption box: the generated caption until someone edits it. A
    # recomputed figure never overwrites an edit (the new generated caption
    # is kept aside and one click restores it); with no figure, there is
    # nothing to export and the bar steps aside
    app.clientside_callback(
        """function (spec, nReset, cur, base) {
            var dc = window.dash_clientside, nu = dc.no_update;
            var ctx = dc.callback_context;
            var trig = (ctx.triggered && ctx.triggered.length)
                       ? ctx.triggered[0].prop_id : "";
            var wrap = spec ? {} : {display: "none"};
            var gen = (spec && spec.caption) ? spec.caption : "";
            if (trig.indexOf("cap-reset") >= 0) return [gen, gen, "", wrap];
            if (!spec) return [nu, nu, nu, wrap];
            var edited = !!cur && base !== null && base !== undefined &&
                         cur !== base && cur !== gen;
            if (edited) {
                return [nu, gen, "edited — the figure changed since; " +
                        "your caption is kept", wrap];
            }
            return [gen, gen, "", wrap];
        }""",
        Output({"type": "caption", "rq": MATCH}, "value"),
        Output({"type": "cap-base", "rq": MATCH}, "data"),
        Output({"type": "cap-note", "rq": MATCH}, "children"),
        Output({"type": "exp-wrap", "rq": MATCH}, "style"),
        Input({"type": "fig-spec", "rq": MATCH}, "data"),
        Input({"type": "cap-reset", "rq": MATCH}, "n_clicks"),
        State({"type": "caption", "rq": MATCH}, "value"),
        State({"type": "cap-base", "rq": MATCH}, "data"))

    # every export click is answered at once, in the browser; the server
    # replaces the line when the file is ready (or says why it is not)
    app.clientside_callback(
        """function (clicks) {
            var ctx = window.dash_clientside.callback_context;
            if (!ctx.triggered || !ctx.triggered.length ||
                !ctx.triggered[0].value) return window.dash_clientside.no_update;
            var id = JSON.parse(ctx.triggered[0].prop_id.split(".")[0]);
            var what = {pdf: "PDF", png: "PNG", pgf: ".pgf",
                        zip: "all formats"}[id.what] || id.what;
            return "Preparing " + what + "…";
        }""",
        Output({"type": "exp-status", "rq": MATCH}, "children", allow_duplicate=True),
        Input({"type": "exp-btn", "rq": MATCH, "what": ALL}, "n_clicks"),
        prevent_initial_call=True)
    app.clientside_callback(
        """function (nFig, nTab) {
            var ctx = window.dash_clientside.callback_context;
            var p = (ctx.triggered && ctx.triggered.length)
                    ? ctx.triggered[0].prop_id : "";
            return p.indexOf("exp-clip-tab") >= 0 ? "LaTeX table copied"
                                                  : "LaTeX figure copied";
        }""",
        Output({"type": "exp-status", "rq": MATCH}, "children", allow_duplicate=True),
        Input({"type": "exp-clip-fig", "rq": MATCH}, "n_clicks"),
        Input({"type": "exp-clip-tab", "rq": MATCH}, "n_clicks"),
        prevent_initial_call=True)

    @app.callback(
        Output({"type": "exp-dl", "rq": MATCH}, "data"),
        Output({"type": "exp-status", "rq": MATCH}, "children", allow_duplicate=True),
        Input({"type": "exp-btn", "rq": MATCH, "what": ALL}, "n_clicks"),
        State({"type": "fig-spec", "rq": MATCH}, "data"),
        State({"type": "caption", "rq": MATCH}, "value"),
        *opt_st,
        prevent_initial_call=True)
    def download(clicks, spec, caption, *opts):
        if not spec or not ctx.triggered_id or not any(c for c in clicks if c):
            return no_update, no_update
        what = ctx.triggered_id.get("what")
        from .pages.insights_common import catalog_unavailable
        why = catalog_unavailable()
        if why:
            return no_update, ("Export paused: " + (
                "a scan is updating the catalog." if why == "scanning" else
                "the last scan did not complete — scan again first."))
        spec = _with_opts(spec, *opts)
        if caption:
            spec["caption"] = caption
        slug, _ref = export_name(spec)
        try:
            if what == "png":
                return (dcc.send_bytes(fig_png(spec), f"{slug}.png"),
                        f"Downloaded {slug}.png")
            if what == "pdf":
                pdf, method, note = fig_pdf(spec)
                return (dcc.send_bytes(pdf, f"{slug}.pdf"),
                        f"Downloaded {slug}.pdf ({method})"
                        + (f" — {note}" if note else ""))
            if what == "pgf":
                pgf, err = fig_pgf(spec)
                if pgf is None:
                    return no_update, (f"No .pgf: {err}. PDF and PNG work "
                                       "without TeX.")
                return (dict(content=pgf, filename=f"{slug}.pgf"),
                        f"Downloaded {slug}.pgf")
            if what == "zip":
                return (dcc.send_bytes(export_bundle(spec), f"{slug}.zip"),
                        f"Downloaded {slug}.zip")
        except Exception as e:           # say what failed, never a dead button
            return no_update, (f"The {what.upper()} export failed "
                               f"({type(e).__name__}: {e})"[:200])
        return no_update, no_update

    @app.callback(
        Output({"type": "exp-preview", "rq": MATCH}, "children"),
        Output({"type": "exp-prev", "rq": MATCH}, "children"),
        Input({"type": "exp-prev", "rq": MATCH}, "n_clicks"),
        Input({"type": "fig-spec", "rq": MATCH}, "data"),
        *opt_in,
        prevent_initial_call=True)
    def preview(n, spec, *opts):
        """The exported figure as it will print: rendered by the export
        engine at the venue's width, shown at 96 px per inch (1:1 on a
        standard screen). Toggled by the button; while open it follows the
        figure and the options."""
        if not n or n % 2 == 0 or not spec:
            return None, "Print-size preview"
        import base64
        spec = _with_opts(spec, *opts)
        w, h, pt = geometry(spec)
        try:
            png = fig_png(spec, dpi=192)
        except Exception as e:
            return (html.Div(f"The preview failed ({type(e).__name__}: {e})",
                             className="muted small"), "Hide preview")
        from .figures import VENUES
        v = VENUES.get((spec.get("export") or {}).get("venue") or "generic",
                       VENUES["generic"])
        env = figure_env(spec)
        return html.Div([
            html.Div(f"{v['label']} · {env} · {w:.2f} × {h:.2f} in · "
                     f"{pt:g} pt labels · shown at print size",
                     className="paper-meta"),
            html.Img(src="data:image/png;base64," + base64.b64encode(png).decode(),
                     alt=f"Print-size preview of the exported figure "
                         f"({w:.2f} × {h:.2f} in)",
                     style={"width": f"{w * 96:.0f}px", "maxWidth": "none"}),
        ]), "Hide preview"
