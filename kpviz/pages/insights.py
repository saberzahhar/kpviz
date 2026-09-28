"""Insights — five research-question workbenches with LaTeX/PGF export."""
from __future__ import annotations

from dash import ClientsideFunction, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

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
        html.P("Five research questions. Pick a slice, read the figure, export "
               "it: captions are written from the exact configuration.",
               className="page-desc"),
        html.Div(id="ins-banner", className="ins-banner", role="status",
                 style={"display": "none"}),
        html.Div([html.Button(t, id=f"tab-{key}", role="tab",
                              className="rq-tab" + (" active" if key == "rq4" else ""),
                              n_clicks=0)
                  for key, t, _ in TABS], className="rq-tabs", role="tablist"),
        # one inference procedure for every workbench: daggers, intervals,
        # captions and exported tables all read these, so a paper cannot mix
        # thresholds, tests or corrections. Folded: the defaults are right
        # for most papers, and the summary says what is in force.
        html.Details([
            html.Summary(["Statistics", html.Span(id="stats-sum",
                                                  className="stats-sum")],
                         title="Applies to every workbench, its captions and "
                               "its exported tables"),
            html.Div([
            html.Div([
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
            ], id="stats-tests", className="stats-group"),
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
            ], className="more-body"),
        ], className="more-opts stats-fold"),
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
                item("PRMU", "on stemmed tokens (Boudin & Gallina, 2021): "
                     "Present — the tokens occur as a contiguous sequence, "
                     "in order, within one section; Reordered — all occur, "
                     "not as that sequence; Mixed — some; Unseen — none. A "
                     "PRMU filter restricts the gold only: every prediction "
                     "still counts in P@k."),
                item("Matching", "predictions and gold NFKC-normalised, "
                     "lowercased, split into Unicode words (letters, digits "
                     "and combining marks; one token per Chinese or Japanese "
                     "character), Snowball-stemmed in their language; predictions "
                     "de-duplicated keeping rank order, gold de-duplicated "
                     "per annotation set; a repeated document or prediction "
                     "line counts once (first line)."),
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

    @app.callback(Output("ins-banner", "children"), Output("ins-banner", "style"),
                  State("vis-insights", "data"), Input("shown-insights", "data"), Input("catalog-version", "data"),
                  Input("scan-poll", "disabled"), prevent_initial_call=True)
    def banner(visible, _shown, _v, _poll_off):
        """Why the figures are not moving: a scan is writing the store, or
        the last one stopped half-way (figures and exports then stay on the
        last coherent view until a scan completes)."""
        from .insights_common import catalog_unavailable
        if not visible:
            raise PreventUpdate
        why = catalog_unavailable()
        if why == "scanning":
            msg = ("The catalog is being updated — figures stay as they are "
                   "and refresh once, when the scan finishes.")
        elif why == "incomplete":
            msg = ("The last scan did not complete, so the catalog may mix old "
                   "and new data. Analyses and exports are paused until a scan "
                   "completes (Overview → Scan for changes).")
        else:
            return "", {"display": "none"}
        return msg, {"display": "block"}

    # what is in force, in one line, without a server round trip
    app.clientside_callback(
        """function (a, fam, adj, ci, n) {
            var alphas = %s;
            var fams = {rank: "rank tests", mean: "mean tests",
                        resample: "resampling tests"};
            var adjs = {holm: "Holm", bonferroni: "Bonferroni", bh: "BH",
                        none: "no correction"};
            var cis = {t: "95 %% t intervals", bootstrap: "95 %% bootstrap intervals",
                       none: "no intervals"};
            return " · p < " + alphas[a] + " · " + (fams[fam] || fam) + " · "
                   + (adjs[adj] || adj) + " · " + (cis[ci] || ci);
        }""" % [alpha_str(a) for a in ALPHAS],
        Output("stats-sum", "children"),
        Input("ins-alpha", "value"), Input("ins-family", "value"),
        Input("ins-adjust", "value"), Input("ins-ci", "value"),
        Input("ins-resamples", "value"))

    # RQ4 tests nothing (it plots intervals only): the test controls say so
    app.clientside_callback(
        """function (active) {
            var off = active === "rq4";
            return [{opacity: off ? 0.45 : 1, display: "contents"},
                    off ? "RQ4 runs no test: only the interval setting applies"
                        : ""];
        }""",
        Output("stats-tests", "style"), Output("stats-tests", "title"),
        Input("ins-active", "data"))

    # tab switches never reach the server: the clicked tab becomes active and
    # its visibility store (appfactory's route) wakes that workbench only
    app.clientside_callback(
        ClientsideFunction(namespace="kpviz", function_name="tabs"),
        [Output("ins-active", "data")]
        + [Output(f"panel-{k}", "style") for k, _, _ in TABS]
        + [Output(f"tab-{k}", "className") for k, _, _ in TABS],
        [Input(f"tab-{k}", "n_clicks") for k, _, _ in TABS],
        prevent_initial_call=True)
