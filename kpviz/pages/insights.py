"""Insights — five research-question workbenches with LaTeX/PGF export."""
from __future__ import annotations

from dash import Input, Output, ctx, dcc, html

from .. import ui
from ..stats import ALPHAS, ALPHA_DEFAULT, alpha_str
from .rq import rq1, rq2, rq3, rq4, rq5

TABS = [
    ("rq1", "Dataset correlation", rq1),
    ("rq2", "Data quality & bias", rq2),
    ("rq3", "Extractability & truncation", rq3),
    ("rq4", "Cost–performance", rq4),
    ("rq5", "Hyperparameters", rq5),
]


def layout():
    return html.Div([
        html.H2("Insights", className="page-title"),
        html.P("Five research questions, each a modular workbench: choose the "
               "slice, read the figure, then copy the LaTeX or download the "
               "PGF/PDF/PNG — captions are auto-written from the exact "
               "configuration and stay editable.", className="page-desc"),
        html.Div([html.Button(t, id=f"tab-{key}",
                              className="rq-tab" + (" active" if key == "rq4" else ""),
                              n_clicks=0)
                  for key, t, _ in TABS], className="rq-tabs"),
        # one significance level for every workbench: daggers, captions and
        # exported tables all read this, so a paper cannot mix thresholds
        ui.filter_row([
            ui.control("Significance level (daggers)", dcc.Slider(
                id="ins-alpha", min=0, max=len(ALPHAS) - 1, step=None,
                marks={i: f"p<{alpha_str(a)}" for i, a in enumerate(ALPHAS)},
                value=ALPHAS.index(ALPHA_DEFAULT),
                included=False), 330),
            html.Div("† marks a two-sided test below this level; every caption "
                     "states it.", className="muted small",
                     style={"alignSelf": "flex-end", "paddingBottom": "2px"}),
        ]),
        *[html.Div(mod.layout(), id=f"panel-{key}",
                   style={"display": "block" if key == "rq4" else "none"})
          for key, _, mod in TABS],
    ], className="page")


def register(app):
    for _key, _t, mod in TABS:
        mod.register(app)

    @app.callback(
        [Output(f"panel-{k}", "style") for k, _, _ in TABS]
        + [Output(f"tab-{k}", "className") for k, _, _ in TABS],
        [Input(f"tab-{k}", "n_clicks") for k, _, _ in TABS],
        prevent_initial_call=True)
    def switch(*_clicks):
        active = (ctx.triggered_id or "tab-rq4").replace("tab-", "")
        styles = [{"display": "block" if k == active else "none"}
                  for k, _, _ in TABS]
        classes = ["rq-tab active" if k == active else "rq-tab"
                   for k, _, _ in TABS]
        return styles + classes
