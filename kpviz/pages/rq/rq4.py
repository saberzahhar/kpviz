"""RQ4 — Which runs offer the best quality–cost trade-off?

Only (model, architecture, run) triples present on every selected dataset
enter the chart (consistent representation). A run whose cost cannot be
resolved is listed beside the figure, not drawn as a point it does not
have; a run with a partial cost or missing documents is drawn open (as
provisional) and never defines the frontier."""
from __future__ import annotations

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ... import scanner, ui
from ...metrics import metric_label, run_scores_many
from ...naming import (encode_runs, group_key, legend_items, run_labels,
                       run_rows)
from ..insights_common import (NO_PRMU, ann_options, datasets_with_runs,
                               default_datasets, effective_runs, empty_result,
                               figure_block, gate, gold_controls, gold_phrase,
                               metric_caption, metric_control, models_control,
                               prmu_arg, prmu_empty, resolve_ann, rq_header,
                               runs_control, scope_line, selected_runs,
                               split_metric, stats_cfg, stats_inputs,
                               value_cell, vis, shown)
from .rq3 import _ci_hover

RQ = "rq4"

UNIT_LABEL = {"usd": "cost (USD)", "kwh": "energy (kWh)",
              "time": "wall-clock time (s)"}
UNIT_SHORT = {"usd": "USD", "kwh": "kWh", "time": "s"}


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("Which runs offer the best quality–cost trade-off?",
                  "Each mark is a run, averaged over the selected datasets; "
                  "the grey staircase is the frontier of complete runs — no "
                  "run is both cheaper and better than a point on it."),
        ui.filter_row([
            ui.control("Datasets", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds, value=default_datasets(3), multi=True,
                className="dash-dropdown"), 320),
            metric_control(RQ),
            ui.control("Cost", dcc.Dropdown(
                id=f"{RQ}-unit",
                options=[{"label": v, "value": u} for u, v in UNIT_LABEL.items()],
                value="usd", clearable=False, searchable=False,
                className="dash-dropdown"), 170),
        ]),
        ui.more([
            ui.control("Cost basis", dcc.Dropdown(
                id=f"{RQ}-norm",
                options=[{"label": "per document", "value": "per_doc"},
                         {"label": "run total", "value": "total"}],
                value="per_doc", clearable=False, searchable=False,
                className="dash-dropdown"), 150),
            ui.control("Cost axis", dcc.Dropdown(
                id=f"{RQ}-xscale", options=[{"label": "log", "value": "log"},
                                            {"label": "linear", "value": "linear"}],
                value="log", clearable=False, searchable=False,
                className="dash-dropdown"), 110),
            ui.control("Point labels", dcc.RadioItems(
                id=f"{RQ}-labels", value="frontier", className="segmented",
                inline=True,
                options=[{"label": "frontier", "value": "frontier"},
                         {"label": "all", "value": "all"},
                         {"label": "none", "value": "none"}]), 220),
            *gold_controls(RQ), models_control(RQ),
            runs_control(RQ, 460, "all consistent runs of the selected models"),
        ]),
        figure_block(RQ, height=480,
                     label="Quality against cost, one mark per run, with the "
                           "Pareto frontier"),
    ])


def cost_scale(values: list[float], norm: str) -> tuple[float, str]:
    """(multiplier, basis) so that per-document costs read as ordinary
    numbers — "USD per 1,000 documents" instead of 9.13e-06 per document.
    Totals are left as they are."""
    if norm != "per_doc":
        return 1.0, "per run"
    pos = sorted(v for v in values if v and v > 0)
    if not pos:
        return 1.0, "per document"
    median = pos[len(pos) // 2]
    for mult, basis in ((1.0, "per document"), (1e3, "per 1,000 documents"),
                        (1e6, "per million documents")):
        if median * mult >= 0.01:
            return mult, basis
    return 1e6, "per million documents"


def fmt_cost(v, unit: str) -> str:
    """Plain decimals with three significant digits, the unit's symbol."""
    if v is None:
        return "—"
    txt = f"{v:,.3g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2g}"
    if "e" in txt:                      # beyond any sensible scale
        txt = f"{v:,.6f}".rstrip("0").rstrip(".")
    return ("$" + txt) if unit == "usd" else f"{txt} {UNIT_SHORT.get(unit, unit)}"


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

    def _update(ds_sel, metric, prmu_sel, ann_choice, unit, norm, xscale,
                models_sel, runs_sel, labels_on, *stat_vals):
        from ...figures import empty_figure, to_plotly
        measure, k = split_metric(metric)
        ds_sel = ds_sel or []
        if not ds_sel:
            return (empty_figure(), None,
                    empty_result("Pick at least one dataset."), None)
        if prmu_empty(prmu_sel):
            return empty_figure(), None, empty_result(NO_PRMU), None
        idx = scanner.cards()
        rows_meta = run_rows(ds_sel)
        chosen = effective_runs(ds_sel, models_sel, runs_sel, require_all=True)
        if not chosen:
            return (empty_figure(), None, empty_result(
                f"No run is evaluated on all {len(ds_sel)} selected datasets.",
                "Remove a dataset, or clear the model and run filters."), None)
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

        # pass 1: quality and cost per run
        points, unknown = [], []
        for key in keys:
            gk = group_key(*key)
            lab = labels.get(gk, key[0])
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
                per_ds_txt.append((ds, sc["mean"],
                                   c["total"] if c and c.get("known") else None))
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
            cost_known = all(c is not None for c in costs_t)
            total = sum(costs_t) if cost_known else None
            x = None if total is None else (
                (total / max(1, sum(docs_t))) if norm == "per_doc" else total)
            item = {"key": key, "gk": gk, "lab": lab, "perf": perf, "ci": ci,
                    "cov": cov, "x": x, "flags": sorted(set(flags)),
                    "complete": all(complete_t), "per_ds": per_ds_txt,
                    "e": enc.get(gk, {})}
            if not cost_known:
                item["why"] = f"{UNIT_SHORT.get(unit, unit)} not resolvable"
                unknown.append(item)
            elif xscale == "log" and x <= 0:
                # a real zero, not a missing cost: it has no place on a log
                # axis, and the reader is told how to see it
                item["why"] = "zero cost — switch the cost axis to linear"
                unknown.append(item)
            else:
                points.append(item)
        if not points and not unknown:
            return (empty_figure(), None, empty_result(
                "No selected run has scores on every selected dataset."), None)

        mult, basis = cost_scale([p_["x"] for p_ in points], norm)
        unit_txt = UNIT_LABEL.get(unit, unit).split(" (")[0]
        axis_unit = f"{UNIT_SHORT.get(unit, unit)} {basis}"
        label_mode = ("all" if labels_on in (["y"], "all") else
                      "none" if labels_on in ([], None, "none") else "frontier")

        # pass 2: marks. A partial cost total, or a run that skipped
        # documents, is not comparable to complete ones: it is drawn open
        # (provisional) and never defines the frontier
        series, known_pts, table_rows = [], [], []
        n_provisional = 0
        for it in points:
            x = it["x"] * mult
            provisional = not it["complete"] or it["cov"] < 0.999
            n_provisional += provisional
            if not provisional:
                known_pts.append((x, it["perf"], len(series)))
            e = it["e"]
            hover = (f"<b>{it['lab']}</b>" + (" (provisional)" if provisional else "")
                     + f"<br>{mlab} = {it['perf']:.3f} " + _ci_hover(it["ci"])
                     + f"(mean over {len(ds_sel)} datasets)<br>{unit_txt} = "
                     + fmt_cost(x, unit) + f" {basis}<br>"
                     + f"coverage {100 * it['cov']:.0f}%<br>"
                     + "<br>".join(f"{d}: {m:.3f}"
                                   + (f" · {fmt_cost(c, unit)} total"
                                      if c is not None else "")
                                   for d, m, c in it["per_ds"])
                     + (("<br>⚠ " + ", ".join(it["flags"])) if it["flags"] else ""))
            series.append({
                "name": it["lab"], "x": [x], "y": [it["perf"]],
                "color": e.get("color", "#2a78d6"),
                "edge": e.get("base"),
                "shape": e.get("shape", "circle"),
                "mpl_marker": e.get("mpl_marker", "o"),
                "size": e.get("size", 10), "open": provisional,
                "legendgroup": e.get("model"),
                "text": [it["lab"]], "show_text": False,
                "hover": [hover], "in_legend": True,
                "err": [it["ci"]] if it["ci"][0] is not None else None,
            })
            pc = value_cell(it["perf"], None,
                            ci=it["ci"] if it["ci"][0] is not None else None)
            notes = list(it["flags"])
            if provisional:
                notes.insert(0, "provisional: " + (
                    "partial cost" if not it["complete"] else
                    f"{100 * it['cov']:.0f}% of the documents"))
            table_rows.append([it["lab"], pc, fmt_cost(x, unit),
                               ui.pct(it["cov"], 0), ", ".join(notes) or "—"])
        for it in unknown:
            pc = value_cell(it["perf"], None,
                            ci=it["ci"] if it["ci"][0] is not None else None)
            table_rows.append([it["lab"], pc, "—", ui.pct(it["cov"], 0),
                               ", ".join([it["why"]] + it["flags"])])

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

        unknown_txt = ""
        if unknown:
            by_model: dict[str, list] = {}
            for it in unknown:
                by_model.setdefault(idx.model(it["key"][0]).name, []).append(it)
            unknown_txt = "; ".join(
                f"{m} ({len(v)} run{'s' if len(v) > 1 else ''}, "
                f"{mlab} {min(i['perf'] for i in v):.3f}"
                + (f"–{max(i['perf'] for i in v):.3f}" if len(v) > 1 else "")
                + f", {v[0]['why']})" for m, v in sorted(by_model.items()))
        spec = {
            "kind": "scatter", "size": "2col", "xscale": xscale,
            "xplain": True,
            "xlabel": f"{unit_txt}, {axis_unit}"
                      + (" (log scale)" if xscale == "log" else ""),
            "ylabel": f"{mlab} (mean over {len(ds_sel)} datasets)",
            "series": series,
            "legend_items": legend_items(enc, [it["gk"] for it in points],
                                         arch_names=idx),
            "frontier": {"x": fx, "y": fy},
            "name": f"pareto-{unit}-{'-'.join(ds_sel)}",
            "caption": (f"Quality against {unit_txt} over {', '.join(ds_sel)}: "
                        f"{mlab} averaged across datasets against "
                        f"{unit_txt} ({axis_unit}, resolved through each "
                        "architecture's declared linear rate model). Only runs "
                        f"evaluated on all datasets are shown ({len(points)}); "
                        "colour = model (lighter = another run of it), shape "
                        "= architecture. The grey staircase is the Pareto "
                        "frontier over runs with a complete cost and full "
                        "coverage"
                        + (f"; {n_provisional} provisional run(s), with a "
                           "partial cost or missing documents, are drawn open "
                           "and excluded from it" if n_provisional else "")
                        + (f"; vertical whiskers: {cfg.ci_text()} of the "
                           "mean" if want_ci else "")
                        + (f". Not drawn — {unit_txt} unknown: {unknown_txt}"
                           if unknown else "") + ". "
                        + metric_caption(measure, k, prmu_sel, ann_choice,
                                         ds_sel)),
        }
        headers = ["Run", mlab, f"{unit_txt} ({axis_unit})", "Coverage", "Notes"]
        spec["table"] = {"headers": headers,
                         "rows": [[r[0], r[1][1], *[str(c) for c in r[2:]]]
                                  for r in table_rows],
                         "label": f"pareto-{unit}"}
        table = ui.table(headers, [[r[0], r[1][0], *r[2:]] for r in table_rows],
                         num_cols={1, 2, 3})
        head = scope_line(
            f"{mlab} against {unit_txt} {axis_unit}",
            f"{len(ds_sel)} dataset{'s' if len(ds_sel) > 1 else ''}",
            f"{len(points)} runs evaluated on all of them",
            f"{len(fx)} on the frontier" if fx else None,
            f"{n_provisional} provisional (drawn open)" if n_provisional else None,
            "colour = model, shape = architecture",
            gold_phrase(ann_choice, ds_sel))
        side = (html.Div([html.B(f"Not drawn — {unit_txt} unknown: "),
                          unknown_txt], className="small muted unknown-list")
                if unknown else None)
        if not points:
            return (empty_figure(), None, empty_result(
                f"No selected run has a known {unit_txt}.",
                "Their scores are in the table; pick another cost unit, or "
                "declare the missing variables in the architecture card."),
                html.Div([side, table]))
        return (to_plotly(spec), spec, html.Div([head, side]),
                ui.fold(table, len(table_rows),
                        f"Table · {len(table_rows)} runs · "
                        f"{len(fx)} on the frontier"))
