"""RQ1 — Are datasets linearly correlated?

A dataset-by-dataset correlation matrix where each underlying vector holds
one score per (model, run) pair — only pairs evaluated on *every* selected
dataset enter the matrix (consistent representation)."""
from __future__ import annotations

from dash import Input, Output, State, dcc, html

from ... import scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import run_labels, run_rows
from ...stats import (ADJUST, alpha_str, kendall_ci, kendall_tau_b, p_str,
                      pearson, sig_mark, spearman)
from dash.exceptions import PreventUpdate

from ..insights_common import (ann_options, datasets_with_runs,
                               effective_runs, figure_block, gate,
                               metric_caption, metric_controls, models_control,
                               p_cells, p_headers, prmu_arg, resolve_ann,
                               rq_header, runs_control, selected_runs,
                               stats_cfg, stats_inputs, value_cell, vis)

METHODS = {"pearson": "Pearson r", "spearman": "Spearman ρ",
           "kendall": "Kendall τ-b"}

RQ = "rq1"


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("Are datasets linearly correlated?",
                  "Each cell correlates two datasets over the scores of the "
                  "(model, run) pairs they share — a high r means the two "
                  "benchmarks rank systems the same way. Pearson tests linear "
                  "association, Spearman monotone (rank) association, Kendall "
                  "τ-b pairwise ranking agreement (robust with few systems). "
                  "Each cell is tested (H0: no association) and the family "
                  "of dataset pairs corrected as set under Statistics."),
        ui.filter_row([
            ui.control("Datasets (≥ 2)", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds, value=ds[:3], multi=True,
                className="dash-dropdown"), 300),
            *metric_controls(RQ),
            ui.control("Method", dcc.Dropdown(
                id=f"{RQ}-method",
                options=[{"label": v, "value": k} for k, v in METHODS.items()],
                value="pearson", clearable=False, className="dash-dropdown"), 140),
        ]),
        ui.filter_row([
            models_control(RQ),
            runs_control(RQ, 460, "all shared runs of the selected models"),
        ]),
        figure_block(RQ, height=470),
    ])


def _assoc(method: str, a: list[float], b: list[float]):
    """(coefficient, p, lo, hi) — two-sided test of no association and a
    95 % Fisher-z interval."""
    if method == "spearman":
        return spearman(a, b)
    if method == "kendall":
        tau, p = kendall_tau_b(a, b)
        lo, hi = kendall_ci(tau, len(a))
        return tau, p, lo, hi
    return pearson(a, b)


def register(app):
    from ..insights_common import (register_dataset_refresh,
                                   register_model_run_chain)
    register_dataset_refresh(app, f"{RQ}-ds", multi=True)
    register_model_run_chain(app, RQ, multi_ds=True, require_all=True)

    @app.callback(Output(f"{RQ}-ann", "options"), Input(vis(RQ), "data"),
                  Input(f"{RQ}-ds", "value"), prevent_initial_call=True)
    def opts(visible, ds_sel):
        if not visible:
            raise PreventUpdate
        return ann_options(ds_sel or [])

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "caption", "rq": RQ}, "value"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Output({"type": "fig-sig", "rq": RQ}, "data"),
        Input(vis(RQ), "data"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), Input(f"{RQ}-method", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        *stats_inputs(),
        State({"type": "fig-sig", "rq": RQ}, "data"),
        prevent_initial_call=True)
    def update(visible, *args):
        *inputs, last_sig = args
        sig = gate(visible, inputs, last_sig)
        return (*_update(*inputs), sig)

    def _update(ds_sel, measure, k, prmu_sel, ann_choice, method, models_sel,
                runs_sel, *stat_vals):
        from ...figures import to_plotly
        ds_sel = [d for d in (ds_sel or [])]
        if len(ds_sel) < 2:
            return (to_plotly({"kind": "bar", "series": []}), None,
                    "", html.Div("select at least two datasets",
                                 className="muted small"))
        chosen = effective_runs(ds_sel, models_sel, runs_sel, require_all=True)
        keys = selected_runs(chosen)
        prmu = prmu_arg(prmu_sel)

        cfg = stats_cfg(*stat_vals)
        method = method if method in METHODS else "pearson"
        vectors: dict[str, list[float]] = {}
        for ds in ds_sel:
            ann = resolve_ann(ds, ann_choice)
            sc = run_scores(ds, keys, ann, measure, k, prmu=prmu)
            vectors[ds] = [sc.get(t, {}).get("mean") for t in keys]
        # drop runs with a missing score anywhere
        keep = [i for i in range(len(keys))
                if all(vectors[ds][i] is not None for ds in ds_sel)]
        n = len(keep)
        # one test per unordered dataset pair; the family is corrected once
        pairs = [(i, j) for i in range(len(ds_sel))
                 for j in range(i + 1, len(ds_sel))]
        res = {}
        for i, j in pairs:
            res[(i, j)] = _assoc(method,
                                 [vectors[ds_sel[i]][t] for t in keep],
                                 [vectors[ds_sel[j]][t] for t in keep])
        p_raw = [res[pq][1] for pq in pairs]
        p_adj = dict(zip(pairs, cfg.adjust_all(p_raw)))
        n_tests = sum(p is not None for p in p_raw)
        z, ztext, zhover = [], [], []
        for i, da in enumerate(ds_sel):
            row, trow, hrow = [], [], []
            for j, dbs in enumerate(ds_sel):
                if i == j:
                    r = 1.0 if n >= 2 else None
                    row.append(r)
                    trow.append("" if r is None else "1")
                    hrow.append(f"{da}")
                    continue
                pq = (min(i, j), max(i, j))
                r, p, lo, hi = res[pq]
                mark = sig_mark(p_adj[pq], cfg.alpha)
                row.append(r)
                trow.append("" if r is None else f"{r:.2f}{mark}")
                hrow.append(f"{da} × {dbs}: {METHODS[method]} = "
                            + ("—" if r is None else f"{r:.3f}")
                            + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")
                            + f"<br>p = {p_str(p)}"
                            + (f", {ADJUST[cfg.adjust]} {p_str(p_adj[pq])}"
                               if cfg.adjust != "none" else "")
                            + f" {mark}<br>n = {n} (model, run) pairs")
            z.append(row)
            ztext.append(trow)
            zhover.append(hrow)

        mname = METHODS[method]
        adj_txt = (f", {ADJUST[cfg.adjust]}-adjusted over the {n_tests} "
                   "dataset pairs" if cfg.adjust != "none" and n_tests > 1 else "")
        spec = {
            "kind": "heatmap", "size": "1col",
            "heat": {"z": z, "x": ds_sel, "y": ds_sel, "zmin": -1, "zmax": 1,
                     "zmid": 0, "diverging": True, "text": ztext,
                     "customdata": zhover,
                     "hover": "%{customdata}<extra></extra>"},
            "name": f"dataset-correlation-{method}",
            "caption": (f"{mname} between datasets of per-run "
                        f"{metric_label(measure, k, prmu)} scores, over the "
                        f"n={n} (model, run) pairs evaluated on all "
                        f"{len(ds_sel)} datasets. † marks a coefficient "
                        f"significantly different from 0 (two-sided, "
                        f"p<{alpha_str(cfg.alpha)}{adj_txt}); intervals are "
                        "95 % Fisher-z. "
                        + metric_caption(measure, k, prmu_sel, ann_choice,
                                         ds_sel)),
        }
        # the LaTeX table lists the pairs with their test, the figure is the
        # matrix — a paper usually wants one of each
        headers = ["Dataset A", "Dataset B", mname] + p_headers(cfg)
        rows, rows_h = [], []
        for pq in pairs:
            r, p, lo, hi = res[pq]
            mark = sig_mark(p_adj[pq], cfg.alpha)
            cells = [value_cell(r, None, signed=True, mark=mark,
                                ci=(lo, hi) if lo is not None else None)]
            cells += p_cells(p, p_adj[pq], cfg)
            a_, b_ = ds_sel[pq[0]], ds_sel[pq[1]]
            rows.append([a_, b_] + [c[1] for c in cells])
            rows_h.append([a_, b_] + [c[0] for c in cells])
        spec["table"] = {"headers": headers, "rows": rows,
                         "label": f"corr-{method}",
                         "notes": f"{mname} over n={n} shared (model, run) "
                                  f"pairs; two-sided test of no association"
                                  f"{adj_txt}; 95 % Fisher-z intervals."}
        idx = scanner.cards()
        labels = run_labels(idx, run_rows(ds_sel))
        detail = ui.table(
            ["(model, run) pairs in the vectors"] + ds_sel,
            [[labels.get(chosen[i], chosen[i])]
             + [f"{vectors[ds][i]:.3f}" for ds in ds_sel] for i in keep],
            num_cols=set(range(1, 1 + len(ds_sel))))
        note = html.Div(f"n = {n} shared (model, run) pairs — pairs missing on "
                        "any dataset are excluded (consistent representation)."
                        + (" With fewer than 10 systems every coefficient is "
                           "fragile; read the intervals." if n < 10 else ""),
                        className="muted small", style={"margin": "6px 0"})
        pairs_tab = ui.table(headers, rows_h, num_cols={2, 3, 4})
        return (to_plotly(spec), spec, spec["caption"],
                html.Div([pairs_tab, note, detail]))
