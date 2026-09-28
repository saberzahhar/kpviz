"""RQ1 — Do scores on one dataset track scores on another?

A dataset-by-dataset correlation matrix where each underlying vector holds
one score per system (a model, its runs averaged; or every run) — only
systems evaluated on *every* selected dataset enter the matrix (consistent
representation)."""
from __future__ import annotations

from dash import Input, Output, State, dcc, html

from ... import scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import run_labels, run_rows
from ...stats import (ADJUST, alpha_str, kendall_ci, kendall_tau_b, p_str,
                      pearson, sig_mark, spearman)
from dash.exceptions import PreventUpdate

from ..insights_common import (NO_PRMU, ann_options, datasets_with_runs,
                               default_datasets, effective_runs, empty_result,
                               figure_block, gate, gold_controls, gold_phrase,
                               metric_caption, metric_control, models_control,
                               p_cells, p_headers, prmu_arg, prmu_empty,
                               resolve_ann, rq_header, runs_control,
                               scope_line, selected_runs, split_metric,
                               stats_cfg, stats_inputs, value_cell, vis, shown)

METHODS = {"pearson": "Pearson r", "spearman": "Spearman ρ",
           "kendall": "Kendall τ-b"}

RQ = "rq1"


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("Do scores on one dataset track scores on another?",
                  "Each cell correlates two datasets over the systems "
                  "evaluated on both: a high value means the two benchmarks "
                  "tell the same story about which system is better."),
        ui.filter_row([
            ui.control("Datasets (≥ 2)", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds, value=default_datasets(4), multi=True,
                className="dash-dropdown"), 320),
            metric_control(RQ),
            ui.control("Method", dcc.Dropdown(
                id=f"{RQ}-method",
                options=[{"label": v, "value": k} for k, v in METHODS.items()],
                value="pearson", clearable=False, className="dash-dropdown"), 140),
            ui.control("Systems", dcc.RadioItems(
                id=f"{RQ}-unit", value="model", className="segmented",
                inline=True,
                options=[{"label": "one per model (runs averaged)",
                          "value": "model",
                          "title": "each model's runs averaged into one point: "
                                   "beam settings or seeds of one model are "
                                   "not independent systems"},
                         {"label": "every run", "value": "run",
                          "title": "descriptive; many runs of one model "
                                   "dominate the test"}]), 300),
        ]),
        ui.more([*gold_controls(RQ), models_control(RQ),
                 runs_control(RQ, 460, "all shared runs of the selected models")]),
        figure_block(RQ, height=360,
                     label="Correlation matrix between datasets"),
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

    @app.callback(Output(f"{RQ}-ann", "options"), State(vis(RQ), "data"), Input(shown(RQ), "data"),
                  Input(f"{RQ}-ds", "value"), prevent_initial_call=True)
    def opts(visible, _shown, ds_sel):
        if not visible:
            raise PreventUpdate
        return ann_options(ds_sel or [])

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "rq-head", "rq": RQ}, "children"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Output({"type": "fig-sig", "rq": RQ}, "data"),
        State(vis(RQ), "data"), Input(shown(RQ), "data"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-metric", "value"),
        Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), Input(f"{RQ}-method", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        Input(f"{RQ}-unit", "value"),
        *stats_inputs(),
        Input("catalog-version", "data"),
        State({"type": "fig-sig", "rq": RQ}, "data"),
        prevent_initial_call=True)
    def update(visible, _shown, *args):
        # the catalog version is an input so exactly the visible workbench
        # re-renders once when a scan publishes; gate() signs it itself
        *inputs, _catalog, last_sig = args
        sig = gate(visible, inputs, last_sig)
        return (*_update(*inputs), sig)

    def _update(ds_sel, metric, prmu_sel, ann_choice, method, models_sel,
                runs_sel, unit, *stat_vals):
        from ...figures import empty_figure, to_plotly
        measure, k = split_metric(metric)
        ds_sel = [d for d in (ds_sel or [])]
        if len(ds_sel) < 2:
            return (empty_figure(), None,
                    empty_result("Pick at least two datasets to compare.",
                                 "A correlation needs two score vectors."),
                    None)
        if prmu_empty(prmu_sel):
            return empty_figure(), None, empty_result(NO_PRMU), None
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
        idx = scanner.cards()
        labels = run_labels(idx, run_rows(ds_sel))
        per_model = unit != "run"
        if per_model:
            # one system = one model: its (shared) runs averaged per dataset
            groups: dict[str, list[int]] = {}
            for i in keep:
                groups.setdefault(keys[i][0], []).append(i)
            unit_names = [idx.model(m).name + (f" ({len(ix)} runs)" if len(ix) > 1 else "")
                          for m, ix in groups.items()]
            vectors = {ds: [sum(vectors[ds][i] for i in ix) / len(ix)
                            for ix in groups.values()] for ds in ds_sel}
            keep = list(range(len(groups)))
        else:
            unit_names = [labels.get(chosen[i], chosen[i]) for i in range(len(keys))]
        n = len(keep)
        unit_txt = "models (each model's runs averaged)" if per_model else "(model, run) pairs"
        if n < 3:
            return (empty_figure(), None, empty_result(
                f"Only {n} system{'s are' if n != 1 else ' is'} evaluated on "
                f"all {len(ds_sel)} datasets — a correlation needs at least 3.",
                "Remove a dataset, or include more models or runs."), None)
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
        # the lower triangle only: each pair once, and no diagonal of 1s
        # (the darkest cells of the figure would be the uninformative ones)
        for i, da in enumerate(ds_sel[1:], start=1):
            row, trow, hrow = [], [], []
            for j, dbs in enumerate(ds_sel[:-1]):
                if j >= i:
                    row.append(None)
                    trow.append("")
                    hrow.append("")
                    continue
                pq = (min(i, j), max(i, j))
                r, p, lo, hi = res[pq]
                mark = sig_mark(p_adj[pq], cfg.alpha)
                row.append(r)
                # an undefined coefficient (a constant score vector) is
                # labelled, not left blank like the omitted upper triangle
                trow.append("n/a" if r is None else f"{r:.2f}{mark}")
                hrow.append(f"{da} × {dbs}: {METHODS[method]} = "
                            + ("—" if r is None else f"{r:.3f}")
                            + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")
                            + f"<br>p = {p_str(p)}"
                            + (f", {ADJUST[cfg.adjust]} {p_str(p_adj[pq])}"
                               if cfg.adjust != "none" else "")
                            + f" {mark}<br>n = {n} {unit_txt}")
            z.append(row)
            ztext.append(trow)
            zhover.append(hrow)

        mname = METHODS[method]
        adj_txt = (f", {ADJUST[cfg.adjust]}-adjusted over the {n_tests} "
                   "dataset pairs" if cfg.adjust != "none" and n_tests > 1 else "")
        spec = {
            "kind": "heatmap", "size": "1col",
            "heat": {"z": z, "x": ds_sel[:-1], "y": ds_sel[1:], "zmin": -1,
                     "zmax": 1,
                     "zmid": 0, "diverging": True, "text": ztext,
                     "customdata": zhover,
                     "hover": "%{customdata}<extra></extra>"},
            "name": f"dataset-correlation-{method}",
            "caption": (f"{mname} between datasets of "
                        f"{metric_label(measure, k, prmu)} scores, over "
                        f"n={n} {unit_txt} evaluated on all "
                        f"{len(ds_sel)} datasets"
                        + ("" if per_model else
                           " (runs of one model are related, not independent "
                           "systems: read the test as descriptive)")
                        + ". † marks a coefficient "
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
                         "notes": f"{mname} over n={n} shared {unit_txt}; "
                                  f"two-sided test of no association"
                                  + (" (exact permutation p up to 9 systems)"
                                     if method != "pearson" else "")
                                  + f"{adj_txt}; 95 % Fisher-z intervals."}
        detail = ui.table(
            [("models" if per_model else "(model, run) pairs") + " in the vectors"]
            + ds_sel,
            [[unit_names[i]] + [f"{vectors[ds][i]:.3f}" for ds in ds_sel]
             for i in keep],
            num_cols=set(range(1, 1 + len(ds_sel))))
        any_ci = any(res[pq][2] is not None for pq in pairs)
        note = html.Div(f"n = {n} shared {unit_txt}; runs missing on any "
                        "dataset are left out (consistent representation)."
                        + (" With fewer than 10 systems every coefficient is "
                           "fragile" + ("; read the intervals." if any_ci else ".")
                           if n < 10 else ""),
                        className="muted small", style={"margin": "6px 0"})
        pairs_tab = ui.table(headers, rows_h, num_cols={2, 3, 4})
        head = scope_line(f"{mname} of {metric_label(measure, k, prmu)}",
                          f"{len(ds_sel)} datasets",
                          f"n = {n} {'models' if per_model else 'runs'} "
                          "evaluated on all of them",
                          gold_phrase(ann_choice, ds_sel),
                          f"† p < {alpha_str(cfg.alpha)}"
                          + (f", {ADJUST[cfg.adjust]}" if cfg.adjust != "none"
                             else ""))
        return (to_plotly(spec), spec, head,
                html.Div([pairs_tab, note,
                          ui.fold(detail, len(keep),
                                  f"Scores behind the matrix ({len(keep)} "
                                  + ("models" if per_model else "runs") + ")",
                                  limit=0)]))
