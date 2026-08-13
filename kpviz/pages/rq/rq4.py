"""RQ4 — Cost–performance Pareto frontier.

Only (model, architecture, run) triples present on every selected dataset
enter the chart (consistent representation). Runs whose selected cost unit
cannot be resolved anywhere are drawn as dashed performance-only lines;
incomplete runs fade with their document coverage."""
from __future__ import annotations

import json

from dash import Input, Output, dcc, html

from ... import db, scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import encode_runs, group_key, run_labels, run_rows
from ...util import fmt_num
from ..insights_common import (ann_options, datasets_with_runs,
                               effective_runs, figure_block, metric_caption,
                               metric_controls, models_control, prmu_arg,
                               resolve_ann, rq_header, run_options,
                               runs_control, selected_runs)

RQ = "rq4"

UNIT_LABEL = {"usd": "cost (USD)", "kwh": "energy (kWh)",
              "time": "wall-clock time (s)"}


def layout():
    ds = datasets_with_runs()
    default = [d for d in ("kp20k", "kpbiomed", "kptimes") if d in ds] or ds[:3]
    return html.Div([
        rq_header("What does a point of quality cost?",
                  "Each mark is one (model, run) triple, macro-averaged over "
                  "the selected datasets — only triples evaluated on all of "
                  "them qualify. The step line is the Pareto frontier: "
                  "nothing above-left of it exists. Dashed horizontal lines "
                  "carry runs whose cost cannot be resolved (unknown "
                  "architecture or unobserved cost variables); transparency "
                  "encodes document coverage."),
        ui.filter_row([
            ui.control("Datasets", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds, value=default, multi=True,
                className="dash-dropdown"), 300),
            *metric_controls(RQ),
            ui.control("Cost unit", dcc.Dropdown(
                id=f"{RQ}-unit",
                options=[{"label": v, "value": u} for u, v in UNIT_LABEL.items()],
                value="usd", clearable=False, className="dash-dropdown"), 170),
            ui.control("Normalisation", dcc.Dropdown(
                id=f"{RQ}-norm",
                options=[{"label": "per document", "value": "per_doc"},
                         {"label": "total", "value": "total"}],
                value="per_doc", clearable=False, className="dash-dropdown"), 150),
            ui.control("x scale", dcc.Dropdown(
                id=f"{RQ}-xscale", options=["linear", "log"], value="log",
                clearable=False, className="dash-dropdown"), 110),
            ui.control("Point labels", dcc.Checklist(
                id=f"{RQ}-labels", options=[{"label": " show", "value": "y"}],
                value=["y"], className="kp-check kp-inline"), 90),
        ]),
        ui.filter_row([
            models_control(RQ),
            runs_control(RQ, 460, "all consistent runs of the selected models"),
        ]),
        figure_block(RQ, height=520),
    ])


def _frontier(pts: list[tuple[float, float]]):
    """Pareto staircase for (cost asc, perf max)."""
    pts = sorted(pts)
    xs, ys, best = [], [], None
    for x, y in pts:
        if best is None or y > best:
            best = y
            xs.append(x)
            ys.append(y)
    return xs, ys


def register(app):
    from ..insights_common import (register_dataset_refresh,
                                   register_model_run_chain)
    register_dataset_refresh(app, f"{RQ}-ds", multi=True,
                             prefer=["kp20k", "kpbiomed", "kptimes"])
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
        Input(f"{RQ}-ann", "value"), Input(f"{RQ}-unit", "value"),
        Input(f"{RQ}-norm", "value"), Input(f"{RQ}-xscale", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        Input(f"{RQ}-labels", "value"))
    def update(ds_sel, measure, k, prmu_sel, ann_choice, unit, norm, xscale,
               models_sel, runs_sel, labels_on):
        from ...figures import to_plotly
        ds_sel = ds_sel or []
        empty = to_plotly({"kind": "scatter", "series": []})
        if not ds_sel:
            return empty, None, "", html.Div("select at least one dataset",
                                             className="muted small")
        idx = scanner.cards()
        rows_meta = run_rows(ds_sel)
        chosen = effective_runs(ds_sel, models_sel, runs_sel, require_all=True)
        eligible = chosen
        keys = selected_runs(chosen)
        prmu = prmu_arg(prmu_sel)
        labels = run_labels(idx, rows_meta)
        enc = encode_runs(idx, [r for r in rows_meta
                                if group_key(r["model"], r["arch"], r["run_id"])
                                in set(chosen)])
        mlab = metric_label(measure, k, prmu)

        # per-dataset scores
        per_ds_scores: dict[str, dict] = {}
        for ds in ds_sel:
            ann = resolve_ann(ds, ann_choice)
            per_ds_scores[ds] = run_scores(ds, keys, ann, measure, k, prmu=prmu)

        # cost + coverage per (run, ds) from the runs table
        run_info: dict[tuple, dict] = {}
        for r in rows_meta:
            run_info[(r["dataset"], r["model"], r["arch"], r["run_id"])] = r

        series, hlines, table_rows = [], [], []
        known_pts = []
        for key in keys:
            gk = group_key(*key)
            lab = labels.get(gk, key[0])
            e = enc.get(gk, {})
            perfs, costs_t, docs_t, covs, flags = [], [], [], [], []
            per_ds_txt = []
            ok = True
            for ds in ds_sel:
                sc = per_ds_scores[ds].get(key)
                info = run_info.get((ds, *key))
                if not sc or sc["mean"] is None or not info:
                    ok = False
                    break
                perfs.append(sc["mean"])
                covs.append(info["coverage"] if info["coverage"] is not None else 1.0)
                c = (info["costs"] or {}).get(unit)
                per_ds_txt.append(f"{ds}: {sc['mean']:.3f}"
                                  + (f" · {fmt_num(c['total'], 4)} {unit}"
                                     if c and c.get("known") else " · cost —"))
                if c and c.get("known"):
                    costs_t.append(c["total"])
                    docs_t.append(info["n_docs"] or 0)
                    flags.extend(c.get("flags") or [])
                else:
                    costs_t.append(None)
            if not ok or not perfs:
                continue
            perf = sum(perfs) / len(perfs)
            cov = min(covs) if covs else 1.0
            alpha = max(0.15, min(1.0, cov))
            cost_known = all(c is not None for c in costs_t)
            if cost_known:
                total = sum(costs_t)
                x = (total / max(1, sum(docs_t))) if norm == "per_doc" else total
                known_pts.append((x, perf))
                hover = (f"<b>{lab}</b><br>{mlab} = {perf:.3f} (macro over "
                         f"{len(ds_sel)} datasets)<br>{UNIT_LABEL[unit]} = "
                         f"{fmt_num(x, 5)} ({norm.replace('_', ' ')})<br>"
                         f"coverage = {100 * cov:.0f}%<br>"
                         + "<br>".join(per_ds_txt)
                         + (("<br>⚠ " + ", ".join(sorted(set(flags))))
                            if flags else ""))
                series.append({
                    "name": lab, "x": [x], "y": [perf],
                    "color": e.get("color", "#2a78d6"),
                    "shape": e.get("shape", "circle"),
                    "mpl_marker": e.get("mpl_marker", "o"),
                    "size": e.get("size", 11), "alpha": alpha,
                    "text": [lab], "show_text": bool(labels_on),
                    "hover": [hover], "in_legend": not labels_on,
                })
                table_rows.append([lab, f"{perf:.3f}", fmt_num(x, 5),
                                   ui.pct(cov, 0),
                                   ", ".join(sorted(set(flags))) or "—"])
            else:
                hlines.append({"y": perf, "label": f"{lab} — no {unit}",
                               "color": e.get("color", "#898781"), "dash": True,
                               "alpha": max(0.35, alpha)})
                table_rows.append([lab, f"{perf:.3f}", "—", ui.pct(cov, 0),
                                   f"no {unit} cost resolvable"])

        fx, fy = _frontier(known_pts)
        enc_note = ""
        vals = list(enc.values())
        if vals:
            dims = [f"colour = {vals[0].get('color_dim')}"]
            if vals[0].get("shape_dim"):
                dims.append(f"shape = {vals[0]['shape_dim']}")
            if vals[0].get("size_dim"):
                dims.append(f"size = {vals[0]['size_dim']}")
            enc_note = "; ".join(dims)
        spec = {
            "kind": "scatter", "size": "2col", "xscale": xscale,
            "xlabel": f"{UNIT_LABEL.get(unit, unit)} — "
                      f"{'per document' if norm == 'per_doc' else 'total'}"
                      + (", log scale" if xscale == "log" else ""),
            "ylabel": f"{mlab} (macro over {len(ds_sel)} datasets)",
            "series": series, "hlines": hlines,
            "frontier": {"x": fx, "y": fy},
            "name": f"pareto-{unit}-{'-'.join(ds_sel)}",
            "caption": (f"Cost–performance trade-off over {', '.join(ds_sel)}: "
                        f"{mlab} macro-averaged across datasets against "
                        f"{UNIT_LABEL.get(unit, unit)} "
                        f"({'per document' if norm == 'per_doc' else 'run total'}, "
                        f"resolved through each architecture's declared linear "
                        f"rate model). Only (model, run) pairs evaluated on all "
                        f"datasets are shown ({len(series) + len(hlines)}); "
                        "dashed horizontal lines carry runs with no resolvable "
                        "cost; marker transparency encodes document coverage"
                        + (f"; {enc_note}" if enc_note else "") + ". The grey "
                        "staircase is the Pareto frontier. "
                        + metric_caption(measure, k, prmu_sel, ann_choice)),
        }
        headers = ["Run", mlab, UNIT_LABEL.get(unit, unit), "Coverage", "Notes"]
        spec["table"] = {"headers": headers,
                         "rows": [[str(c) for c in r] for r in table_rows],
                         "label": f"pareto-{unit}"}
        note = html.Div(
            f"{len(eligible)} consistent (model, arch, run) triples across "
            f"{len(ds_sel)} dataset(s); {len(hlines)} without resolvable "
            f"{unit}.", className="muted small", style={"margin": "6px 0"})
        table = ui.table(headers, table_rows, num_cols={1, 2, 3})
        return to_plotly(spec), spec, spec["caption"], html.Div([note, table])
