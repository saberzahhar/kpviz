"""Dash application assembly: frame, routing, global export callbacks."""
from __future__ import annotations

import json
import uuid

import dash
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from . import __version__, db, scanner, ui
from .config import settings
from .export import (export_bundle, fig_pdf, fig_pgf, fig_png, latex_figure,
                     latex_table, slugify, tex_engine)
from .util import human_duration

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


def _sidebar():
    return html.Div([
        html.Div([
            html.Div(["KP", html.Span("Viz")], className="brand-name"),
            html.Div("keyphrase evaluation cockpit", className="brand-sub"),
        ], className="brand"),
        *[dcc.Link([html.Span(ico, className="nav-ico"), label],
                   href=href, id=f"nav-{href.strip('/') or 'home'}",
                   className="nav-link") for href, label, ico in NAV],
        html.Div([
            html.Div(id="sidebar-scan", children=[
                html.Span(className="scan-dot"), "idle"]),
            html.Div(str(settings().data_root), style={
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap", "marginTop": "4px"},
                title=str(settings().data_root)),
        ], className="sidebar-foot"),
    ], className="sidebar")


def build_app() -> dash.Dash:
    from .pages import architectures, datasets, home, insights, models

    from pathlib import Path
    assets = Path(__file__).resolve().parent.parent / "assets"
    app = dash.Dash(__name__, title="KPViz",
                    assets_folder=str(assets),
                    suppress_callback_exceptions=True,
                    update_title=None)

    pages = {
        "/": home.layout, "/datasets": datasets.layout,
        "/models": models.layout, "/architectures": architectures.layout,
        "/insights": insights.layout,
    }
    app.layout = html.Div([
        dcc.Location(id="url"),
        dcc.Store(id="catalog-version"),
        _sidebar(),
        html.Div([
            html.Div([html.Div(pages[href](), id=f"page-{href.strip('/') or 'home'}",
                               style={"display": "block" if href == "/" else "none"})
                      for href in pages]),
        ], className="main"),
    ], className="app-frame")

    _install_build_route(app)
    _install_stale_guard(app)

    # ---- routing: show/hide pre-mounted pages (state survives navigation)
    @app.callback(
        [Output(f"page-{href.strip('/') or 'home'}", "style") for href in pages]
        + [Output(f"nav-{href.strip('/') or 'home'}", "className") for href in pages],
        Input("url", "pathname"))
    def route(path):
        path = path or "/"
        if path not in pages:
            path = "/"
        styles = [{"display": "block" if href == path else "none"}
                  for href in pages]
        classes = ["nav-link active" if href == path else "nav-link"
                   for href in pages]
        return styles + classes

    # ---- sidebar scan status (piggybacks on the home poll interval)
    @app.callback(Output("sidebar-scan", "children"),
                  Input("home-poll", "n_intervals"))
    def sidebar_scan(_n):
        snap = scanner.STATE.snapshot()
        if snap["running"]:
            eta = snap.get("eta_s")
            return [html.Span(className="scan-dot busy"),
                    f"scanning — ETA {human_duration(eta)}"]
        if snap.get("error"):
            return [html.Span(className="scan-dot err"), "last scan failed"]
        v = db.scan_version()
        return [html.Span(className="scan-dot ok" if v else "scan-dot"),
                f"catalog v{v}" if v else "no scan yet"]

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
# Export callbacks (shared by every insight workbench)
# ---------------------------------------------------------------------------

def _register_exports(app):
    @app.callback(
        Output({"type": "exp-clip-fig", "rq": MATCH}, "content"),
        Output({"type": "exp-clip-tab", "rq": MATCH}, "content"),
        Output({"type": "exp-hint", "rq": MATCH}, "children"),
        Input({"type": "fig-spec", "rq": MATCH}, "data"),
        Input({"type": "caption", "rq": MATCH}, "value"))
    def clipboards(spec, caption):
        if not spec:
            return "", "", ""
        caption = caption or spec.get("caption", "")
        slug = slugify(spec.get("name", "figure"))
        eng = tex_engine()
        fig_tex = latex_figure(f"figures/{slug}" + (".pgf" if eng else ".pdf"),
                               caption, slug, pgf=bool(eng))
        tab = spec.get("table")
        tab_tex = ""
        if tab:
            tab_tex = latex_table(tab["headers"], tab["rows"], caption,
                                  tab.get("label", slug))
        hint = (f"PGF typeset with {eng}" if eng
                else "no TeX found — PDF exports use the Matplotlib backend")
        return fig_tex, tab_tex, hint

    @app.callback(
        Output({"type": "exp-dl", "rq": MATCH}, "data"),
        Input({"type": "exp-btn", "rq": MATCH, "what": ALL}, "n_clicks"),
        State({"type": "fig-spec", "rq": MATCH}, "data"),
        State({"type": "caption", "rq": MATCH}, "value"),
        prevent_initial_call=True)
    def download(clicks, spec, caption):
        if not spec or not ctx.triggered_id or not any(c for c in clicks if c):
            return no_update
        what = ctx.triggered_id.get("what")
        spec = dict(spec)
        if caption:
            spec["caption"] = caption
        slug = slugify(spec.get("name", "figure"))
        if what == "png":
            return dcc.send_bytes(fig_png(spec), f"{slug}.png")
        if what == "pdf":
            pdf, _method = fig_pdf(spec)
            return dcc.send_bytes(pdf, f"{slug}.pdf")
        if what == "pgf":
            pgf = fig_pgf(spec)
            if pgf is None:
                return no_update
            return dict(content=pgf, filename=f"{slug}.pgf")
        if what == "zip":
            return dcc.send_bytes(export_bundle(spec, spec.get("name", slug)),
                                  f"{slug}.zip")
        return no_update
