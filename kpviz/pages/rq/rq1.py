"""RQ1 — Are datasets linearly correlated?

A dataset-by-dataset correlation matrix where each underlying vector holds
one score per (model, run) pair — only pairs evaluated on *every* selected
dataset enter the matrix (consistent representation)."""
from __future__ import annotations

import math

from dash import Input, Output, State, dcc, html

from ... import db, scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import run_labels, run_rows
from ..insights_common import (ann_options, datasets_with_runs,
                               effective_runs, figure_block, metric_caption,
                               metric_controls, models_control, prmu_arg,
                               resolve_ann, rq_header, runs_control,
                               selected_runs)

RQ = "rq1"


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("Are datasets linearly correlated?",
                  "Each cell correlates two datasets over the scores of the "
                  "(model, run) pairs they share — a high r means the two "
                  "benchmarks rank systems the same way. Pearson tests linear "
                  "association, Spearman monotone (rank) association."),
        ui.filter_row([
            ui.control("Datasets (≥ 2)", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds, value=ds[:3], multi=True,
                className="dash-dropdown"), 300),
            *metric_controls(RQ),
            ui.control("Method", dcc.Dropdown(
                id=f"{RQ}-method",
                options=[{"label": "Pearson r", "value": "pearson"},
                         {"label": "Spearman ρ", "value": "spearman"}],
                value="pearson", clearable=False, className="dash-dropdown"), 140),
        ]),
        ui.filter_row([
            models_control(RQ),
            runs_control(RQ, 460, "all shared runs of the selected models"),
        ]),
        figure_block(RQ, height=470),
    ])


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k2 in range(i, j + 1):
            ranks[order[k2]] = r
        i = j + 1
    return ranks


def _corr(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def register(app):
    from ..insights_common import (register_dataset_refresh,
                                   register_model_run_chain)
    register_dataset_refresh(app, f"{RQ}-ds", multi=True)
    register_model_run_chain(app, RQ, multi_ds=True, require_all=True)

    @app.callback(Output(f"{RQ}-ann", "options"), Input(f"{RQ}-ds", "value"))
    def opts(ds_sel):
        return ann_options(ds_sel or [])

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "caption", "rq": RQ}, "value"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), Input(f"{RQ}-method", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"))
    def update(ds_sel, measure, k, prmu_sel, ann_choice, method, models_sel,
               runs_sel):
        from ...figures import to_plotly
        ds_sel = [d for d in (ds_sel or [])]
        if len(ds_sel) < 2:
            return (to_plotly({"kind": "bar", "series": []}), None,
                    "", html.Div("select at least two datasets",
                                 className="muted small"))
        chosen = effective_runs(ds_sel, models_sel, runs_sel, require_all=True)
        keys = selected_runs(chosen)
        prmu = prmu_arg(prmu_sel)

        vectors: dict[str, list[float]] = {}
        for ds in ds_sel:
            ann = resolve_ann(ds, ann_choice)
            sc = run_scores(ds, keys, ann, measure, k, prmu=prmu)
            vectors[ds] = [sc.get(t, {}).get("mean") for t in keys]
        # drop runs with a missing score anywhere
        keep = [i for i in range(len(keys))
                if all(vectors[ds][i] is not None for ds in ds_sel)]
        n = len(keep)
        z, ztext = [], []
        for da in ds_sel:
            row, trow = [], []
            for dbs in ds_sel:
                a = [vectors[da][i] for i in keep]
                b = [vectors[dbs][i] for i in keep]
                if method == "spearman":
                    a, b = _rank(a), _rank(b)
                r = _corr(a, b)
                row.append(r)
                trow.append("" if r is None else f"{r:.3f}")
            z.append(row)
            ztext.append(trow)

        mname = "Pearson r" if method == "pearson" else "Spearman ρ"
        spec = {
            "kind": "heatmap", "size": "1col",
            "heat": {"z": z, "x": ds_sel, "y": ds_sel, "zmin": -1, "zmax": 1,
                     "zmid": 0, "diverging": True, "text": ztext,
                     "hover": "%{y} × %{x}: %{z:.3f}<extra></extra>"},
            "name": f"dataset-correlation-{method}",
            "caption": (f"{mname} between datasets of per-run "
                        f"{metric_label(measure, k, prmu)} scores, over the "
                        f"n={n} (model, run) pairs evaluated on all "
                        f"{len(ds_sel)} datasets. "
                        + metric_caption(measure, k, prmu_sel, ann_choice)),
        }
        # table for the LaTeX export
        headers = [""] + ds_sel
        rows = [[da] + [("—" if v is None else f"{v:.3f}") for v in z[i]]
                for i, da in enumerate(ds_sel)]
        spec["table"] = {"headers": headers, "rows": rows,
                         "label": f"corr-{method}"}
        idx = scanner.cards()
        labels = run_labels(idx, run_rows(ds_sel))
        detail = ui.table(
            ["(model, run) pairs in the vectors"] + ds_sel,
            [[labels.get(chosen[i], chosen[i])]
             + [f"{vectors[ds][i]:.3f}" for ds in ds_sel] for i in keep],
            num_cols=set(range(1, 1 + len(ds_sel))))
        note = html.Div(f"n = {n} shared (model, run) pairs — pairs missing on "
                        "any dataset are excluded (consistent representation).",
                        className="muted small", style={"margin": "6px 0"})
        return to_plotly(spec), spec, spec["caption"], html.Div([note, detail])
