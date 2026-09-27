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
from .export import (export_bundle, fig_pdf, fig_pgf, fig_png, latex_figure,
                     latex_table, slugify, start_tex_probe, tex_status)

# Identifies this server process. A browser tab holds the callback signatures
# of the build that rendered it, so a tab left open across an upgrade posts a
# callback the new process has never heard of — Dash answers 500 once per poll
# tick, forever. assets/buildcheck.js compares this value and reloads such a
# tab by itself; _install_stale_guard keeps the log clean meanwhile.
BUILD_ID = f"{__version__}+{uuid.uuid4().hex[:8]}"

NAV = [
    ("/", "Overview", "⌂"),
    ("/datasets", "Datasets", "▤"),
    ("/models", "Models", "◆"),
    ("/architectures", "Architectures", "⚙"),
    ("/insights", "Insights", "◈"),
]
PAGES = [href for href, _l, _i in NAV]
RQS = ["rq1", "rq2", "rq3", "rq4", "rq5"]


def page_key(href: str) -> str:
    return href.strip("/") or "home"


def vis_id(name: str) -> str:
    """Visibility store of a page ('home', 'datasets', …) or workbench."""
    return f"vis-{name}"


def _sidebar():
    return html.Div([
        html.Div([
            html.Div(["KP", html.Span("Viz")], className="brand-name"),
            html.Div("keyphrase evaluation cockpit", className="brand-sub"),
        ], className="brand"),
        *[dcc.Link([html.Span(ico, className="nav-ico"), label],
                   href=href, id=f"nav-{page_key(href)}",
                   className="nav-link") for href, label, ico in NAV],
        html.Div([
            html.Div(id="sidebar-scan", children=_sidebar_status()),
            html.Div(str(settings().data_root), style={
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap", "marginTop": "4px"},
                title=str(settings().data_root)),
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
        return [html.Span(className="scan-dot"), "last scan cancelled"]
    v = db.scan_version()
    return [html.Span(className="scan-dot ok" if v else "scan-dot"),
            f"catalog v{v}" if v else "no scan yet"]


def build_app() -> dash.Dash:
    from .pages import architectures, datasets, home, insights, models

    from pathlib import Path
    assets = Path(__file__).resolve().parent.parent / "assets"
    app = dash.Dash(__name__, title="KPViz",
                    assets_folder=str(assets),
                    suppress_callback_exceptions=True,
                    update_title="Updating… · KPViz")
    start_tex_probe()

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
            dcc.Interval(id="scan-poll", interval=1000, n_intervals=0,
                         disabled=not running),
            _sidebar(),
            html.Div([
                html.Div([html.Div(pages[href](), id=f"page-{page_key(href)}",
                                   style={"display": "none"})
                          for href in pages]),
            ], className="main"),
        ], className="app-frame")

    app.layout = layout

    _install_build_route(app)
    _install_stale_guard(app)
    _install_timing(app)

    # ---- routing + visibility: entirely in the browser -------------------
    app.clientside_callback(
        ClientsideFunction(namespace="kpviz", function_name="route"),
        [Output(f"page-{page_key(h)}", "style") for h in PAGES]
        + [Output(f"nav-{page_key(h)}", "className") for h in PAGES]
        + [Output(vis_id(page_key(h)), "data") for h in PAGES]
        + [Output(vis_id(rq), "data") for rq in RQS],
        Input("url", "pathname"), Input("ins-active", "data"),
        [State(vis_id(page_key(h)), "data") for h in PAGES]
        + [State(vis_id(rq), "data") for rq in RQS])

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
        from flask import jsonify
        return jsonify({"build": BUILD_ID})


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
        return jsonify({"callbacks": rows, "cache": cache_stats(),
                        "peak_rss_mb": rss, "catalog_version": db.scan_version()})


# ---------------------------------------------------------------------------
# Export callbacks (shared by every insight workbench)
# ---------------------------------------------------------------------------

def _register_exports(app):
    @app.callback(
        Output({"type": "exp-clip-fig", "rq": MATCH}, "content"),
        Output({"type": "exp-clip-tab", "rq": MATCH}, "content"),
        Output({"type": "exp-hint", "rq": MATCH}, "children"),
        Input({"type": "fig-spec", "rq": MATCH}, "data"),
        Input({"type": "caption", "rq": MATCH}, "n_blur"),
        State({"type": "caption", "rq": MATCH}, "value"),
        prevent_initial_call=True)
    def clipboards(spec, _blur, caption):
        """The two LaTeX snippets. Runs when the figure changes or the
        caption editor loses focus — never per keystroke, never waiting on
        TeX (the probe result is read, not computed, here)."""
        if not spec:
            return "", "", ""
        caption = caption or spec.get("caption", "")
        slug = slugify(spec.get("name", "figure"))
        status, eng = tex_status()
        use_pgf = status == "ready" and bool(eng)
        fig_tex = latex_figure(f"figures/{slug}" + (".pgf" if use_pgf else ".pdf"),
                               caption, slug, pgf=use_pgf,
                               size=spec.get("size", "2col"))
        tab = spec.get("table")
        tab_tex = ""
        if tab:
            tab_tex = latex_table(tab["headers"], tab["rows"], caption,
                                  tab.get("label", slug))
        if status == "probing":
            hint = "checking TeX… (PDF/PNG are ready now)"
        elif use_pgf:
            hint = f"PGF typeset with {eng}"
        else:
            hint = "no working TeX — PDF exports use Matplotlib's vector backend"
        return fig_tex, tab_tex, hint

    @app.callback(
        Output({"type": "exp-dl", "rq": MATCH}, "data"),
        Output({"type": "exp-hint", "rq": MATCH}, "children", allow_duplicate=True),
        Input({"type": "exp-btn", "rq": MATCH, "what": ALL}, "n_clicks"),
        State({"type": "fig-spec", "rq": MATCH}, "data"),
        State({"type": "caption", "rq": MATCH}, "value"),
        prevent_initial_call=True)
    def download(clicks, spec, caption):
        if not spec or not ctx.triggered_id or not any(c for c in clicks if c):
            return no_update, no_update
        what = ctx.triggered_id.get("what")
        spec = dict(spec)
        if caption:
            spec["caption"] = caption
        slug = slugify(spec.get("name", "figure"))
        try:
            if what == "png":
                return dcc.send_bytes(fig_png(spec), f"{slug}.png"), no_update
            if what == "pdf":
                pdf, method, note = fig_pdf(spec)
                return (dcc.send_bytes(pdf, f"{slug}.pdf"),
                        f"PDF via {method}" + (f" — {note}" if note else ""))
            if what == "pgf":
                pgf, err = fig_pgf(spec)
                if pgf is None:
                    return no_update, f".pgf unavailable — {err}; use PDF or PNG"
                return dict(content=pgf, filename=f"{slug}.pgf"), no_update
            if what == "zip":
                return (dcc.send_bytes(export_bundle(spec, spec.get("name", slug)),
                                       f"{slug}.zip"), no_update)
        except Exception as e:           # say what failed, never a dead button
            return no_update, f"{what} export failed: {type(e).__name__}: {e}"[:200]
        return no_update, no_update
