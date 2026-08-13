"""RQ5 — How do hyperparameters move the needle?

One card-declared inference parameter, every model that actually varies it.
The figure answers "does the needle move at all?" (value on the x axis, one
line/bar per model, macro-averaged over the selected datasets); the table
answers it *independently for each (dataset, model)* — its runs share the same
documents, so the k values form paired blocks and the effect is tested with a
Friedman test (Wilcoxon when the parameter takes only two values).
"""
from __future__ import annotations

import json

from dash import Input, Output, State, dcc, html

from ... import db, scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import (natural_key, param_value_str, slot_color, value_key)
from ...stats import friedman, p_str, sig_caption, sig_mark, wilcoxon_signed_rank
from ...util import fmt_num
from ..insights_common import (alpha_of, ann_options, figure_block,
                               metric_caption, metric_controls, prmu_arg,
                               resolve_ann, rq_header, value_cell)

RQ = "rq5"


def _varying(model: str) -> dict[str, set]:
    """{parameter: {json-encoded values}} over every run of one model."""
    seen: dict[str, set] = {}
    for (resj,) in db.q("SELECT resolved FROM runs WHERE model=?", model):
        for p, info in (json.loads(resj or "{}")).items():
            v = json.dumps((info or {}).get("value"), default=str)
            seen.setdefault(p, set()).add(v)
    return {p: vs for p, vs in seen.items() if len(vs) > 1}


def _models_with_variation() -> dict[str, dict[str, set]]:
    """Only models whose runs actually differ somewhere — a model run once, or
    run many times with identical settings, has no needle to move."""
    out = {}
    for (m,) in db.q("SELECT DISTINCT model FROM runs ORDER BY 1"):
        var = _varying(m)
        if var:
            out[m] = var
    return out


def layout():
    return html.Div([
        rq_header("How do hyperparameters move the needle?",
                  "Runs grouped by the resolved value of a card-declared "
                  "inference parameter (defaults fill the blanks; illegal "
                  "values are flagged, never dropped). Only models that "
                  "actually vary the parameter are offered. The table tests "
                  "each (dataset, model) on its own: the values were run over "
                  "the same documents, so they are paired blocks — Friedman "
                  "across three or more values, Wilcoxon for two."),
        ui.filter_row([
            ui.control("Parameter", dcc.Dropdown(
                id=f"{RQ}-param", clearable=False,
                className="dash-dropdown"), 230),
            ui.control("Models", dcc.Dropdown(
                id=f"{RQ}-model", multi=True, placeholder="all models that vary it",
                className="dash-dropdown"), 320),
            ui.control("Datasets", dcc.Dropdown(
                id=f"{RQ}-ds", multi=True, className="dash-dropdown"), 300),
            *metric_controls(RQ),
        ]),
        figure_block(RQ, height=440),
    ])


def register(app):
    @app.callback(Output(f"{RQ}-param", "options"),
                  Output(f"{RQ}-param", "value"),
                  Input("catalog-version", "data"),
                  State(f"{RQ}-param", "value"))
    def refresh_params(_v, current):
        var = _models_with_variation()
        counts: dict[str, int] = {}
        spread: dict[str, int] = {}
        for params in var.values():
            for p, vals in params.items():
                counts[p] = counts.get(p, 0) + 1
                spread[p] = max(spread.get(p, 0), len(vals))
        # the parameter worth opening on is the one compared in the most
        # places, then the one with the most values to compare
        opts = [{"label": f"{p}  ({n} model{'s' if n > 1 else ''}, "
                          f"{spread[p]} values)", "value": p}
                for p, n in sorted(counts.items(),
                                   key=lambda kv: (-kv[1], -spread[kv[0]],
                                                   natural_key(kv[0])))]
        value = current if current in counts else (opts[0]["value"] if opts else None)
        return opts, value

    @app.callback(Output(f"{RQ}-model", "options"), Output(f"{RQ}-model", "value"),
                  Output(f"{RQ}-ds", "options"), Output(f"{RQ}-ds", "value"),
                  Output(f"{RQ}-ann", "options"),
                  Input(f"{RQ}-param", "value"),
                  State(f"{RQ}-model", "value"))
    def opts(param, current):
        if not param:
            return [], [], [], [], ["auto"]
        idx = scanner.cards()
        var = _models_with_variation()
        models = [m for m, params in var.items() if param in params]
        m_opts = [{"label": f"{idx.model(m).name}  "
                            f"({len(var[m][param])} values)", "value": m}
                  for m in sorted(models,
                                  key=lambda m: natural_key(idx.model(m).name))]
        kept = [m for m in (current or []) if m in models]
        ds = [r[0] for r in db.q(
            f"""SELECT DISTINCT dataset FROM runs
                 WHERE model IN ({','.join('?' * len(models))}) ORDER BY 1""",
            *models)] if models else []
        return m_opts, kept, ds, ds, ann_options(ds)

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "caption", "rq": RQ}, "value"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Input(f"{RQ}-param", "value"), Input(f"{RQ}-model", "value"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), Input("ins-alpha", "value"))
    def update(param, models_sel, ds_sel, measure, k, prmu_sel, ann_choice,
               alpha_ix):
        from ...figures import to_plotly
        empty = to_plotly({"kind": "bar", "series": []})
        var = _models_with_variation()
        if not param or not ds_sel:
            return empty, None, "", html.Div(
                "no inference parameter varies across any model's runs — "
                "nothing to compare." if not var else
                "pick a parameter and at least one dataset",
                className="muted small")
        idx = scanner.cards()
        alpha = alpha_of(alpha_ix)
        prmu = prmu_arg(prmu_sel)
        mlab = metric_label(measure, k, prmu)
        models = [m for m in (models_sel or [])
                  if m in var and param in var[m]] or \
                 [m for m, params in var.items() if param in params]

        # ---- collect: one score (and its per-document detail) per
        # (dataset, model, value); several runs at the same value keep the best
        cells: dict[tuple[str, str], dict] = {}
        illegal_any = False
        for ds in ds_sel:
            ann = resolve_ann(ds, ann_choice)
            for m in models:
                rows = db.q("""SELECT arch, run_id, resolved, violations
                               FROM runs WHERE model=? AND dataset=?""", m, ds)
                keys, meta = [], {}
                for arch, rid, resj, vj in rows:
                    info = (json.loads(resj or "{}").get(param) or {})
                    if info.get("value") is None:
                        continue
                    keys.append((m, arch, rid))
                    meta[(arch, rid)] = (
                        info["value"],
                        any(v.get("param") == param
                            for v in json.loads(vj or "[]")))
                if not keys:
                    continue
                scored = run_scores(ds, keys, ann, measure, k, prmu=prmu,
                                    per_doc=True)
                by_value: dict[str, dict] = {}
                for key in keys:
                    sc = scored.get(key) or {}
                    if sc.get("mean") is None:
                        continue
                    val, illegal = meta[(key[1], key[2])]
                    illegal_any = illegal_any or illegal
                    vj = json.dumps(val, default=str)
                    best = by_value.get(vj)
                    if best is None or sc["mean"] > best["mean"]:
                        by_value[vj] = {"value": val, "mean": sc["mean"],
                                        "n": sc["n"], "illegal": illegal,
                                        "per_doc": sc.get("per_doc", {}),
                                        "run": key[2]}
                if by_value:
                    cells[(ds, m)] = by_value
        if not cells:
            return (empty, None, "",
                    html.Div(f"no run resolves a value for “{param}” on the "
                             "selected datasets", className="muted small"))

        # every value seen anywhere, in value order — one column per value
        all_vals = {}
        for by_value in cells.values():
            for vj, info in by_value.items():
                all_vals[vj] = info["value"]
        col_order = sorted(all_vals, key=lambda vj: value_key(all_vals[vj]))

        # ---- table: one row per (dataset, model), tested on its own ---------
        headers = ["Dataset", "Model"] + [param_value_str(all_vals[vj])
                                          for vj in col_order] + ["p (effect)"]
        trs, tex, tested = [], [], 0
        for (ds, m) in sorted(cells, key=lambda t: (natural_key(t[0]),
                                                    natural_key(idx.model(t[1]).name))):
            by_value = cells[(ds, m)]
            present = [vj for vj in col_order if vj in by_value]
            # paired blocks = the documents every value was scored on
            common = None
            for vj in present:
                docs = set(by_value[vj]["per_doc"])
                common = docs if common is None else (common & docs)
            common = sorted(common or [])
            p_val, test = None, "—"
            if len(present) >= 3 and common:
                _chi, p_val = friedman([[by_value[vj]["per_doc"][d]
                                         for d in common] for vj in present])
                test = "Friedman"
            elif len(present) == 2 and common:
                _w, p_val = wilcoxon_signed_rank(
                    [by_value[present[0]]["per_doc"][d] for d in common],
                    [by_value[present[1]]["per_doc"][d] for d in common])
                test = "Wilcoxon"
            if p_val is not None:
                tested += 1
            mark = sig_mark(p_val, alpha)
            best_vj = max(present, key=lambda vj: by_value[vj]["mean"]) \
                if present else None
            row_h, row_t = [ds, idx.model(m).name], [ds, idx.model(m).name]
            for vj in col_order:
                info = by_value.get(vj)
                if info is None:
                    row_h.append(html.Span("—", className="muted"))
                    row_t.append("—")
                    continue
                cell = value_cell(info["mean"], info["n"])
                node = (html.B(cell[0]) if vj == best_vj else cell[0])
                if info["illegal"]:
                    node = html.Span([node, html.Span(" ⚠", title="value "
                                                      "violates the model card")])
                row_h.append(node)
                row_t.append(cell[1] + (" (best)" if vj == best_vj else ""))
            p_cell = f"{p_str(p_val)}" if p_val is not None else \
                ("n<6" if common else "—")
            row_h.append(html.Span([p_cell, html.Sup(mark)] if mark else p_cell,
                                   title=f"{test} over {len(common)} common documents"))
            row_t.append(f"{p_cell} {mark}".strip())
            trs.append(row_h)
            tex.append(row_t)

        # ---- figure: the headline, one series per model ---------------------
        series = []
        for i, m in enumerate(sorted(models, key=lambda m: natural_key(idx.model(m).name))):
            xs, ys, hv = [], [], []
            for vj in col_order:
                per_ds = [(ds, cells[(ds, mm)][vj]) for (ds, mm) in cells
                          if mm == m and vj in cells[(ds, mm)]]
                if not per_ds:
                    continue
                mean = sum(c["mean"] for _d, c in per_ds) / len(per_ds)
                xs.append(param_value_str(all_vals[vj]))
                ys.append(round(mean, 3))
                hv.append(f"{idx.model(m).name}<br>{param} = "
                          f"{param_value_str(all_vals[vj])}<br>{mlab} = "
                          f"{mean:.3f} (macro over {len(per_ds)} dataset"
                          f"{'s' if len(per_ds) > 1 else ''})<br>"
                          + "<br>".join(f"{d}: {c['mean']:.3f} (n={c['n']})"
                                        for d, c in per_ds))
            if xs:
                series.append({"name": idx.model(m).name, "x": xs, "y": ys,
                               "hover": hv, "color": slot_color(i)})

        spec_p = None
        for m in models:
            spec_p = spec_p or idx.model(m).inference_specs.get(param)
        rng = []
        if spec_p:
            if spec_p.min is not None:
                rng.append(f"min {fmt_num(spec_p.min)}")
            if spec_p.max is not None:
                rng.append(f"max {fmt_num(spec_p.max)}")
            if spec_p.default is not None:
                rng.append(f"default {fmt_num(spec_p.default)}")
        spec = {
            "kind": "bar", "barmode": "group", "size": "2col",
            "xlabel": f"{param}" + (f"  ({' · '.join(rng)})" if rng else ""),
            "ylabel": mlab, "series": series,
            "name": f"hyperparam-{param}",
            "caption": (f"Effect of “{param}” on {mlab}, macro-averaged over "
                        f"{', '.join(sorted(ds_sel))}, for every model whose "
                        "runs vary it. Values are resolved against each model "
                        "card (missing parameters take the declared default"
                        + ("; ⚠ marks values outside the card's legal range"
                           if illegal_any else "") + "). The table tests each "
                        "(dataset, model) separately over the documents its "
                        "runs share — Friedman for three or more values, "
                        "Wilcoxon for two; " + sig_caption(alpha) + ". "
                        + metric_caption(measure, k, prmu_sel, ann_choice)),
        }
        spec["table"] = {"headers": headers, "rows": tex, "label": f"hp-{param}"}
        note = html.Div(
            f"{tested} of {len(trs)} (dataset, model) pairs had enough paired "
            f"documents to test; bold marks the best value in the row.",
            className="muted small", style={"marginBottom": "6px"})
        return (to_plotly(spec), spec, spec["caption"],
                html.Div([note, ui.table(headers, trs,
                                         num_cols=set(range(2, len(headers))))]))
