"""Insights — five research-question workbenches with LaTeX/PGF export."""
from __future__ import annotations

import json

from dash import MATCH, ClientsideFunction, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import ui
from ..stats import ADJUST, ALPHAS, ALPHA_DEFAULT, RESAMPLES, alpha_str
from .rq import rq1, rq2, rq3, rq4, rq5

TABS = [
    ("rq1", "Dataset agreement", rq1),
    ("rq2", "Data quality", rq2),
    ("rq3", "Context windows", rq3),
    ("rq4", "Quality vs. cost", rq4),
    ("rq5", "Hyperparameters", rq5),
]

# which Statistics settings each workbench actually uses: the others are
# disabled (not merely dimmed) on that tab, and the summary says what applies
APPLIES = {
    "rq1": {"alpha", "adjust"},                 # own correlation, Fisher-z
    "rq2": {"alpha", "family", "adjust", "ci", "resamples"},
    "rq3": {"alpha", "family", "adjust", "ci", "resamples"},
    "rq4": {"ci", "resamples"},                 # intervals only, no test
    "rq5": {"alpha", "family", "adjust", "ci", "resamples"},
}


def layout():
    return html.Div([
        html.H1("Insights", className="page-title"),
        html.Div(id="ins-banner", className="ins-banner", role="status",
                 style={"display": "none"}),
        # the tab strip stays in view while a workbench scrolls; the number
        # is the paper's research question, so the default (RQ4) explains
        # itself
        html.Div([html.Button([html.Span(key.upper(), className="rq-num"), t],
                              id=f"tab-{key}", role="tab",
                              className="rq-tab" + (" active" if key == "rq4" else ""),
                              n_clicks=0, **{"aria-controls": f"panel-{key}"})
                  for key, t, _ in TABS],
                 className="rq-tabs", role="tablist",
                 **{"aria-label": "Research questions"}),
        _methods(),
        *[html.Div(mod.layout(), id=f"panel-{key}", role="tabpanel",
                   **{"aria-labelledby": f"tab-{key}"},
                   style={"display": "block" if key == "rq4" else "none"})
          for key, _, mod in TABS],
    ], className="page")


def _methods():
    """One inference procedure for every workbench — daggers, intervals,
    captions and exported tables all read it, so a paper cannot mix
    thresholds, tests or corrections — folded behind a line that says what
    is in force on this tab, with the reading guide in the same fold."""
    def item(term, text):
        return html.Div([html.Dt(term), html.Dd(text)], className="guide-item")
    guide = html.Div([
        html.Dl([
            item("P/R/F1@k", "per document, then macro-averaged: "
                 "P@k = tp / min(k, #predictions) (no padding), R@k = tp / "
                 "#gold; @O cuts at the number of gold keyphrases, @M "
                 "keeps every prediction."),
            item("PRMU", "on stemmed words (Boudin & Gallina, 2021): "
                 "Present — the words occur as a contiguous sequence, in "
                 "order, within one section; Reordered — all occur, not as "
                 "that sequence; Mixed — some; Unseen — none. A PRMU filter "
                 "restricts the gold only: every prediction still counts."),
            item("Matching", "predictions and gold NFKC-normalised, "
                 "lowercased, split into Unicode words and Snowball-stemmed "
                 "in their language; predictions de-duplicated keeping rank "
                 "order, gold de-duplicated per annotation set; a repeated "
                 "document or prediction line counts once."),
        ], className="guide-col"),
        html.Dl([
            item("Rank-based", "the default for per-document scores "
                 "(bounded, many ties): does one condition tend to score "
                 "higher?"),
            item("Mean-based", "tests the macro-average itself, the number "
                 "a paper reports; with many documents the t approximation "
                 "is usually adequate."),
            item("Resampling", "fewer distributional assumptions: sign-flip "
                 "permutation (approximate randomisation) and bootstrap, "
                 "seeded from the data so the p-values are reproducible."),
            item("Corrections", "a table tests many runs at once: Holm "
                 "controls the family-wise error, Benjamini–Hochberg the "
                 "false-discovery rate. Daggers follow the adjusted p."),
            item("Reading", "a p-value says whether a difference is "
                 "detectable, not how large it is: read it with the effect "
                 "size and the interval."),
        ], className="guide-col"),
    ], className="guide-body")
    return html.Details([
        html.Summary([html.Span("Methods", className="fold-title"),
                      html.Span(id="stats-sum", className="stats-sum")],
                     title="Statistics applied to the workbench on screen, its "
                           "captions and its exported tables"),
        html.Div([
            ui.control("Significance level", dcc.RadioItems(
                id="ins-alpha", value=ALPHAS.index(ALPHA_DEFAULT),
                options=[{"label": f"p < {alpha_str(a)}", "value": i}
                         for i, a in enumerate(ALPHAS)],
                className="segmented", inline=True), 300),
            ui.control("Tests", dcc.Dropdown(
                id="ins-family", clearable=False, className="dash-dropdown",
                value="rank", searchable=False, options=[
                    {"label": "Rank-based — Wilcoxon · Mann–Whitney · Friedman",
                     "value": "rank"},
                    {"label": "Mean-based — paired t · Welch t · RM-ANOVA",
                     "value": "mean"},
                    {"label": "Resampling — permutation · bootstrap",
                     "value": "resample"}]), 330),
            ui.control("Multiple comparisons", dcc.Dropdown(
                id="ins-adjust", clearable=False, className="dash-dropdown",
                value="holm", searchable=False,
                options=[{"label": v, "value": k} for k, v in ADJUST.items()]),
                220),
            ui.control("Intervals", dcc.Dropdown(
                id="ins-ci", clearable=False, className="dash-dropdown",
                value="t", searchable=False, options=[
                    {"label": "95 % Student-t", "value": "t"},
                    {"label": "95 % bootstrap", "value": "bootstrap"},
                    {"label": "none", "value": "none"}]), 160),
            ui.control("Resamples", dcc.Dropdown(
                id="ins-resamples", clearable=False, className="dash-dropdown",
                value=RESAMPLES, searchable=False, options=[
                    {"label": f"{n:,}", "value": n}
                    for n in (1000, 5000, 10000)]), 110),
        ], className="more-body"),
        html.Div(id="stats-scope", className="muted small stats-scope"),
        guide,
    ], className="more-opts stats-fold")


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

    # what is in force on this tab, in one line, and which settings do not
    # apply here (disabled, with the reason) — no server round trip
    app.clientside_callback(
        """function (a, fam, adj, ci, n, active) {
            var applies = %s;
            var ap = applies[active] || applies.rq2;
            var alphas = %s;
            var fams = {rank: "rank tests", mean: "mean tests",
                        resample: "resampling tests"};
            var adjs = {holm: "Holm", bonferroni: "Bonferroni", bh: "BH",
                        none: "no correction"};
            var cis = {t: "95 %% t intervals", bootstrap: "95 %% bootstrap intervals",
                       none: "no intervals"};
            var parts = [];
            if (ap.indexOf("alpha") >= 0) parts.push("p < " + alphas[a]);
            if (ap.indexOf("family") >= 0) parts.push(fams[fam] || fam);
            if (active === "rq1") parts.push("correlation tests");
            if (ap.indexOf("adjust") >= 0) parts.push(adjs[adj] || adj);
            if (active === "rq1") parts.push("Fisher-z intervals");
            else if (ap.indexOf("ci") >= 0) parts.push(cis[ci] || ci);
            if (active === "rq4") parts.push("no hypothesis test");
            var scope = {
              rq1: "Dataset agreement uses the correlation method chosen in the workbench and Fisher-z intervals; tests and intervals below do not apply to it.",
              rq4: "Quality vs. cost plots intervals only: it runs no hypothesis test, so the level, tests and correction do not apply to it."
            }[active] || "";
            var alphaOpts = alphas.map(function (v, i) {
              return {label: "p < " + v, value: i,
                      disabled: ap.indexOf("alpha") < 0};
            });
            return [" · " + parts.join(" · "), scope, alphaOpts,
                    ap.indexOf("family") < 0, ap.indexOf("adjust") < 0,
                    ap.indexOf("ci") < 0,
                    ap.indexOf("resamples") < 0 ||
                      (ci !== "bootstrap" && fam !== "resample")];
        }""" % (json.dumps({k: sorted(v) for k, v in APPLIES.items()}),
                [alpha_str(a) for a in ALPHAS]),
        Output("stats-sum", "children"), Output("stats-scope", "children"),
        Output("ins-alpha", "options"), Output("ins-family", "disabled"),
        Output("ins-adjust", "disabled"), Output("ins-ci", "disabled"),
        Output("ins-resamples", "disabled"),
        Input("ins-alpha", "value"), Input("ins-family", "value"),
        Input("ins-adjust", "value"), Input("ins-ci", "value"),
        Input("ins-resamples", "value"), Input("ins-active", "data"))

    # a workbench with nothing to show hides its (empty) graph: the reason
    # is in the scope line above it
    app.clientside_callback(
        """function (fig) {
            var empty = fig && fig.layout && fig.layout.meta === "empty";
            return empty ? {display: "none"} : {};
        }""",
        Output({"type": "rq-gwrap", "rq": MATCH}, "style"),
        Input({"type": "rq-graph", "rq": MATCH}, "figure"))

    # tab switches never reach the server: the clicked tab writes the URL
    # hash, and appfactory's route() shows its panel and wakes it alone
    app.clientside_callback(
        ClientsideFunction(namespace="kpviz", function_name="tabs"),
        Output("url", "hash"),
        [Input(f"tab-{k}", "n_clicks") for k, _, _ in TABS],
        prevent_initial_call=True)
