"""Insights — five research-question workbenches with LaTeX/PGF export."""
from __future__ import annotations

from dash import ClientsideFunction, Input, Output, dcc, html

from .. import ui
from ..stats import ADJUST, ALPHAS, ALPHA_DEFAULT, RESAMPLES, alpha_str
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
        # one inference procedure for every workbench: daggers, intervals,
        # captions and exported tables all read these, so a paper cannot mix
        # thresholds, tests or corrections
        html.Div([
            html.Div("Statistics", className="stats-bar-title",
                     title="Applies to every workbench, its captions and its "
                           "exported tables"),
            ui.control("Significance level", dcc.Slider(
                id="ins-alpha", min=0, max=len(ALPHAS) - 1, step=None,
                marks={i: f"p<{alpha_str(a)}" for i, a in enumerate(ALPHAS)},
                value=ALPHAS.index(ALPHA_DEFAULT),
                included=False), 300),
            ui.control("Tests", dcc.Dropdown(
                id="ins-family", clearable=False, className="dash-dropdown",
                value="rank", options=[
                    {"label": "Rank-based — Wilcoxon · Mann–Whitney · Friedman",
                     "value": "rank"},
                    {"label": "Mean-based — paired t · Welch t · RM-ANOVA",
                     "value": "mean"},
                    {"label": "Resampling — permutation · bootstrap",
                     "value": "resample"}]), 330),
            ui.control("Multiple comparisons", dcc.Dropdown(
                id="ins-adjust", clearable=False, className="dash-dropdown",
                value="holm",
                options=[{"label": v, "value": k} for k, v in ADJUST.items()]),
                220),
            ui.control("Intervals", dcc.Dropdown(
                id="ins-ci", clearable=False, className="dash-dropdown",
                value="t", options=[
                    {"label": "95 % Student-t", "value": "t"},
                    {"label": "95 % bootstrap", "value": "bootstrap"},
                    {"label": "none", "value": "none"}]), 160),
            ui.control("Resamples", dcc.Dropdown(
                id="ins-resamples", clearable=False, className="dash-dropdown",
                value=RESAMPLES, options=[
                    {"label": f"{n:,}", "value": n}
                    for n in (1000, 5000, 10000)]), 110),
        ], className="stats-bar filter-row"),
        _guide(),
        *[html.Div(mod.layout(), id=f"panel-{key}",
                   style={"display": "block" if key == "rq4" else "none"})
          for key, _, mod in TABS],
    ], className="page")


def _guide():
    """Conventions and test choice, one click away — what a reviewer asks."""
    def item(term, text):
        return html.Div([html.Dt(term), html.Dd(text)], className="guide-item")
    return html.Details([
        html.Summary("How to read these results — metrics, tests, intervals"),
        html.Div([
            html.Dl([
                item("P/R/F1@k", "per document, then macro-averaged: "
                     "P@k = tp / min(k, #predictions) (no padding), R@k = tp / "
                     "#gold; @O cuts at the number of gold keyphrases, @M "
                     "keeps every prediction."),
                item("PRMU", "in-order classes on stemmed tokens: Present "
                     "(contiguous, in order), Reordered (all tokens, not in "
                     "order), Mixed (some), Unseen (none)."),
                item("Matching", "predictions and gold lowercased, spaCy-"
                     "tokenised, Snowball (Porter2)-stemmed, de-duplicated "
                     "keeping rank order."),
            ], className="guide-col"),
            html.Dl([
                item("Rank-based", "robust default for per-document scores "
                     "(bounded, many ties); tests whether one condition "
                     "tends to score higher."),
                item("Mean-based", "tests the macro-average itself — the "
                     "number a paper reports; with hundreds of documents the "
                     "t distribution is accurate."),
                item("Resampling", "distribution-free: sign-flip "
                     "permutation (the approximate-randomisation test of "
                     "the NLP literature) and bootstrap; seeded from the "
                     "data, so p-values are reproducible."),
                item("Corrections", "a table tests many runs at once: Holm "
                     "controls the family-wise error (default), Benjamini–"
                     "Hochberg the false-discovery rate. Daggers follow "
                     "the adjusted p; tables show both."),
                item("Intervals", "95 % for each mean and each difference; "
                     "effect sizes say how large, p-values only whether."),
            ], className="guide-col"),
        ], className="guide-body"),
    ], className="guide")


def register(app):
    for _key, _t, mod in TABS:
        mod.register(app)

    # tab switches never reach the server: the clicked tab becomes active and
    # its visibility store (appfactory's route) wakes that workbench only
    app.clientside_callback(
        ClientsideFunction(namespace="kpviz", function_name="tabs"),
        [Output("ins-active", "data")]
        + [Output(f"panel-{k}", "style") for k, _, _ in TABS]
        + [Output(f"tab-{k}", "className") for k, _, _ in TABS],
        [Input(f"tab-{k}", "n_clicks") for k, _, _ in TABS],
        prevent_initial_call=True)
