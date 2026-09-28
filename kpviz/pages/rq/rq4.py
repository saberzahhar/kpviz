"""RQ4 — Cost–performance Pareto frontier.

Only (model, architecture, run) triples present on every selected dataset
enter the chart (consistent representation). Runs whose selected cost unit
cannot be resolved anywhere are drawn as dashed performance-only lines;
incomplete runs fade with their document coverage."""
from __future__ import annotations

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ... import scanner, ui
from ...metrics import metric_label, run_scores_many
from ...naming import encode_runs, group_key, run_labels, run_rows
from ..insights_common import (ann_options, datasets_with_runs,
                               effective_runs, figure_block, gate,
                               metric_caption, metric_controls, models_control,
                               prmu_arg, resolve_ann, rq_header,
                               runs_control, selected_runs, stats_cfg,
                               stats_inputs, value_cell, vis, shown)
from .rq3 import _ci_hover

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
                  "nothing above-left of it exists. Only runs with a complete "
                  "cost and full document coverage can define it; the others "
                  "are drawn faded as provisional. Dashed horizontal lines "
                  "carry runs whose cost cannot be resolved (unknown "
                  "architecture or unobserved cost variables)."),
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
            ui.control("Point labels", dcc.RadioItems(
                id=f"{RQ}-labels", value="frontier", className="kp-check kp-inline",
                options=[{"label": " frontier", "value": "frontier"},
                         {"label": " all", "value": "all"},
                         {"label": " none", "value": "none"}]), 220),
        ]),
        ui.filter_row([
            models_control(RQ),
            runs_control(RQ, 460, "all consistent runs of the selected models"),
        ]),
        figure_block(RQ, height=520),
    ])


def _frontier(pts: list[tuple[float, float]]):
    """Pareto staircase for (cost asc, perf max).

    Equal costs are visited best-first (cost ascending, performance
    descending), so a dominated point at a tied cost never enters the
    frontier; non-finite costs are ignored."""
    pts = sorted(((x, y) for x, y in pts
                  if x is not None and y is not None and x == x and y == y
                  and x not in (float("inf"), float("-inf"))),
                 key=lambda p: (p[0], -p[1]))
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

    @app.callback(Output(f"{RQ}-ann", "options"), State(vis(RQ), "data"), Input(shown(RQ), "data"),
                  Input(f"{RQ}-ds", "value"), prevent_initial_call=True)
    def opts(visible, _shown, ds_sel):
        if not visible:
            raise PreventUpdate
        return ann_options(ds_sel or [])

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "caption", "rq": RQ}, "value"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Output({"type": "fig-sig", "rq": RQ}, "data"),
        State(vis(RQ), "data"), Input(shown(RQ), "data"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), Input(f"{RQ}-unit", "value"),
        Input(f"{RQ}-norm", "value"), Input(f"{RQ}-xscale", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        Input(f"{RQ}-labels", "value"), *stats_inputs(),
        Input("catalog-version", "data"),
        State({"type": "fig-sig", "rq": RQ}, "data"),
        prevent_initial_call=True)
    def update(visible, _shown, *args):
        # the catalog version is an input so exactly the visible workbench
        # re-renders once when a scan publishes; gate() signs it itself
        *inputs, _catalog, last_sig = args
        sig = gate(visible, inputs, last_sig)
        return (*_update(*inputs), sig)

    def _update(ds_sel, measure, k, prmu_sel, ann_choice, unit, norm, xscale,
                models_sel, runs_sel, labels_on, *stat_vals):
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
        shown = [r for r in rows_meta
                 if group_key(r["model"], r["arch"], r["run_id"]) in set(chosen)]
        # labels name only the parameters that differ between the runs
        # actually plotted (not every run of these datasets)
        labels = run_labels(idx, shown)
        enc = encode_runs(idx, shown)
        mlab = metric_label(measure, k, prmu)

        cfg = stats_cfg(*stat_vals)
        want_ci = cfg.ci != "none"
        # per-dataset scores, concurrently; per-document detail only when an
        # interval is asked for (the unfiltered means come precomputed)
        per_ds_scores = run_scores_many({
            ds: dict(dataset=ds, run_keys=keys,
                     ann_key=resolve_ann(ds, ann_choice), measure=measure, k=k,
                     prmu=prmu, per_doc=want_ci)
            for ds in ds_sel})

        # cost + coverage per (run, ds) from the runs table
        run_info: dict[tuple, dict] = {}
        for r in rows_meta:
            run_info[(r["dataset"], r["model"], r["arch"], r["run_id"])] = r

        series, hlines, table_rows = [], [], []
        known_pts = []            # (x, perf, series index) — frontier candidates
        n_provisional = 0
        label_mode = ("all" if labels_on in (["y"], "all") else
                      "none" if labels_on in ([], None, "none") else "frontier")
        cost_vals = []
        for key in keys:
            gk = group_key(*key)
            lab = labels.get(gk, key[0])
            e = enc.get(gk, {})
            perfs, costs_t, docs_t, covs, flags, complete_t = [], [], [], [], [], []
            per_ds_txt, per_doc_arrays = [], []
            ok = True
            for ds in ds_sel:
                sc = per_ds_scores[ds].get(key)
                info = run_info.get((ds, *key))
                if not sc or sc["mean"] is None or not info:
                    ok = False
                    break
                perfs.append(sc["mean"])
                if sc.get("per_doc") is not None:
                    per_doc_arrays.append(sc["per_doc"].vals)
                covs.append(info["coverage"] if info["coverage"] is not None else 1.0)
                c = (info["costs"] or {}).get(unit)
                per_ds_txt.append(f"{ds}: {sc['mean']:.3f}"
                                  + (f" · {c['total']:.3g} {unit}"
                                     if c and c.get("known") else " · cost —"))
                if c and c.get("known"):
                    costs_t.append(c["total"])
                    docs_t.append(info["n_docs"] or 0)
                    flags.extend(c.get("flags") or [])
                    complete_t.append(bool(c.get("complete", True)))
                else:
                    costs_t.append(None)
                    complete_t.append(False)
            if not ok or not perfs:
                continue
            perf = sum(perfs) / len(perfs)
            ci = cfg.macro_ci(per_doc_arrays) if want_ci else (None, None)
            cov = min(covs) if covs else 1.0
            alpha = max(0.15, min(1.0, cov))
            cost_known = all(c is not None for c in costs_t)
            total = sum(costs_t) if cost_known else None
            x = None if total is None else (
                (total / max(1, sum(docs_t))) if norm == "per_doc" else total)
            if x is not None and xscale == "log" and x <= 0:
                cost_known = False           # a zero cost has no log position
                flags.append(f"zero {unit}: not placeable on a log axis")
            if cost_known:
                # a partial cost total, or a run that skipped documents, is
                # not comparable to complete ones: shown, never on the frontier
                provisional = not all(complete_t) or cov < 0.999
                n_provisional += provisional
                if not provisional:
                    known_pts.append((x, perf, len(series)))
                cost_vals.append(x)
                hover = (f"<b>{lab}</b>" + (" (provisional)" if provisional else "")
                         + f"<br>{mlab} = {perf:.3f} "
                         + _ci_hover(ci) + "(macro over "
                         f"{len(ds_sel)} datasets)<br>{UNIT_LABEL[unit]} = "
                         f"{x:.3g} ({norm.replace('_', ' ')})<br>"
                         f"coverage = {100 * cov:.0f}%<br>"
                         + "<br>".join(per_ds_txt)
                         + (("<br>⚠ " + ", ".join(sorted(set(flags))))
                            if flags else ""))
                series.append({
                    "name": lab, "x": [x], "y": [perf],
                    "color": e.get("color", "#2a78d6"),
                    "shape": e.get("shape", "circle"),
                    "mpl_marker": e.get("mpl_marker", "o"),
                    "size": e.get("size", 11),
                    "alpha": min(alpha, 0.35) if provisional else alpha,
                    "text": [lab], "show_text": False,
                    "hover": [hover], "in_legend": True,
                    "err": [ci] if ci[0] is not None else None,
                })
                pc = value_cell(perf, None, ci=ci if ci[0] is not None else None)
                note_bits = sorted(set(flags))
                if provisional:
                    note_bits.insert(0, "provisional: " + (
                        "partial cost" if not all(complete_t) else
                        f"{100 * cov:.0f}% coverage"))
                table_rows.append([lab, pc, x, ui.pct(cov, 0),
                                   ", ".join(note_bits) or "—"])
            else:
                hlines.append({"y": perf, "label": f"{lab} — no {unit}",
                               "color": e.get("color", "#898781"), "dash": True,
                               "alpha": max(0.35, alpha)})
                pc = value_cell(perf, None, ci=ci if ci[0] is not None else None)
                table_rows.append([lab, pc, None, ui.pct(cov, 0),
                                   ", ".join(sorted(set(flags))) or
                                   f"no {unit} cost resolvable"])

        fx, fy = _frontier([(x, y) for x, y, _i in known_pts])
        # labels: frontier points only by default (the rest on hover); points
        # at the same place are merged into one label ("gpt-4o ×3")
        on_front = {(x, y) for x, y in zip(fx, fy)}
        front_idx = {i for x, y, i in known_pts if (x, y) in on_front}
        if label_mode != "none":
            spots: dict[tuple, list[int]] = {}
            for i, sr in enumerate(series):
                if label_mode == "frontier" and i not in front_idx:
                    continue
                spot = (float(f"{sr['x'][0]:.6g}"), round(sr["y"][0], 4))
                spots.setdefault(spot, []).append(i)
            for idxs in spots.values():
                first = series[idxs[0]]
                names = [series[i]["name"] for i in idxs]
                base = names[0].split(" (")[0]
                txt = (names[0] if len(idxs) == 1 else
                       f"{base} ×{len(idxs)}" if all(n.split(' (')[0] == base
                                                     for n in names)
                       else f"{names[0]} +{len(idxs) - 1}")
                first["text"], first["show_text"] = [txt], True

        # one number format per column: scientific throughout when any cost
        # is below 10⁻³ (9.24e-06 next to 0.000471 read as different scales)
        sci = any(0 < v < 1e-3 for v in cost_vals)
        fmt_cost = (lambda v: "—" if v is None else
                    (f"{v:.2e}" if sci else f"{v:.3g}"))
        table_rows = [[r[0], r[1], fmt_cost(r[2]), *r[3:]] for r in table_rows]
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
                        "staircase is the Pareto frontier over runs with a "
                        "complete cost and full coverage"
                        + (f" ({n_provisional} provisional run(s), with a "
                           "partial cost or missing documents, are drawn faded "
                           "and excluded from it)" if n_provisional else "")
                        + (f"; vertical whiskers: {cfg.ci_text()} of the "
                           "macro-average" if want_ci else "") + ". "
                        + metric_caption(measure, k, prmu_sel, ann_choice,
                                         ds_sel)),
        }
        headers = ["Run", mlab, UNIT_LABEL.get(unit, unit), "Coverage", "Notes"]
        spec["table"] = {"headers": headers,
                         "rows": [[r[0], r[1][1], *[str(c) for c in r[2:]]]
                                  for r in table_rows],
                         "label": f"pareto-{unit}"}
        note = html.Div(
            f"{len(eligible)} consistent (model, arch, run) triples across "
            f"{len(ds_sel)} dataset(s); {len(hlines)} without resolvable "
            f"{unit}.", className="muted small", style={"margin": "6px 0"})
        table = ui.table(headers, [[r[0], r[1][0], *r[2:]] for r in table_rows],
                         num_cols={1, 2, 3})
        return to_plotly(spec), spec, spec["caption"], html.Div([note, table])
