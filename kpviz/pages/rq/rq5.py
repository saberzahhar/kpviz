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
from dash.exceptions import PreventUpdate

from ... import db, scanner, ui
from ...metrics import (PerDoc, common_ords, memo, metric_label,
                       run_scores_many, values_at)
from ...naming import (DASHES, arch_shape, model_color, natural_key,
                       param_value_str, shade, value_key)
from ...util import fmt_num
from ..insights_common import (ann_options, effect_cell, figure_block, gate,
                               gold_controls, metric_caption, metric_control,
                               p_cells, p_headers, prmu_arg, resolve_ann,
                               rq_header, split_metric, stats_cfg,
                               stats_inputs, stats_note, value_cell, vis,
                               shown)
from .rq3 import _ci_hover, _pmap

RQ = "rq5"
# the effect size a row's test reports, abbreviated in its cell
EFFECT_SHORT = {"Kendall's W": "W", "rank-biserial r": "r", "Cohen's d_z": "d_z",
                "partial η²": "η²p", "Hedges' g": "g"}


def _models_with_variation() -> dict[str, dict[str, set]]:
    """Only models whose runs actually differ somewhere — a model run once, or
    run many times with identical settings, has no needle to move.
    (One pass over `runs` per catalog version, shared by all callbacks.)"""
    def build():
        seen: dict[str, dict[str, set]] = {}
        for m, resj in db.q("SELECT model, resolved FROM runs ORDER BY 1"):
            d = seen.setdefault(m, {})
            for p, info in (json.loads(resj or "{}")).items():
                d.setdefault(p, set()).add(
                    json.dumps((info or {}).get("value"), default=str))
        out = {}
        for m, params in seen.items():
            var = {p: vs for p, vs in params.items() if len(vs) > 1}
            if var:
                out[m] = var
        return out
    return memo("rq5_variation", build)


def _controlled_sweeps() -> dict[str, int]:
    """{parameter: number of (dataset, model, fixed configuration) groups in
    which it takes at least two values while everything else is fixed} —
    what a controlled sweep can show (memoised per catalog version)."""
    def build():
        groups: dict[tuple, set] = {}
        for ds, m, arch, resj in db.q(
                "SELECT dataset, model, arch, resolved FROM runs"):
            res = {p: (i or {}).get("value") for p, i in
                   json.loads(resj or "{}").items()}
            for p, v in res.items():
                if v is None:
                    continue
                rest = json.dumps({q: w for q, w in res.items() if q != p},
                                  sort_keys=True, default=str)
                groups.setdefault((p, ds, m, arch, rest), set()).add(
                    json.dumps(v, sort_keys=True, default=str))
        out: dict[str, int] = {}
        for (p, *_k), vals in groups.items():
            if len(vals) > 1:
                out[p] = out.get(p, 0) + 1
        return out
    return memo("rq5_sweeps", build)


def _runs_of(ds: str):
    return memo(("rq5_runs", ds), lambda: [tuple(r) for r in db.q(
        """SELECT model, arch, run_id, resolved, violations FROM runs
           WHERE dataset=?""", ds)])


def layout():
    return html.Div([
        rq_header("How do hyperparameters move the needle?",
                  "Runs grouped by the value of one inference parameter "
                  "declared in the model cards (defaults fill the blanks). A "
                  "controlled sweep compares runs that differ in this "
                  "parameter alone; each row of the table is tested over the "
                  "documents its values share."),
        ui.filter_row([
            ui.control("Parameter", dcc.Dropdown(
                id=f"{RQ}-param", clearable=False,
                className="dash-dropdown"), 260),
            ui.control("Datasets", dcc.Dropdown(
                id=f"{RQ}-ds", multi=True, className="dash-dropdown"), 300),
            metric_control(RQ),
        ]),
        ui.more([
            ui.control("Models", dcc.Dropdown(
                id=f"{RQ}-model", multi=True, placeholder="all models that vary it",
                className="dash-dropdown"), 320),
            ui.control("Runs per value", dcc.RadioItems(
                id=f"{RQ}-mode", value="controlled", className="kp-check kp-inline",
                options=[{"label": " controlled sweep", "value": "controlled",
                          "title": "only runs that differ in this parameter "
                                   "alone; replicates averaged"},
                         {"label": " best-run envelope", "value": "envelope",
                          "title": "the best run at each value, whatever else "
                                   "differs (descriptive)"}]), 300),
            *gold_controls(RQ),
        ]),
        figure_block(RQ, height=420),
    ])


def register(app):
    @app.callback(Output(f"{RQ}-param", "options"),
                  Output(f"{RQ}-param", "value"),
                  State(vis(RQ), "data"), Input(shown(RQ), "data"),
                  Input("catalog-version", "data"),
                  State(f"{RQ}-param", "value"),
                  prevent_initial_call=True)
    def refresh_params(visible, _shown, _v, current):
        if not visible:
            raise PreventUpdate
        var = _models_with_variation()
        counts: dict[str, int] = {}
        spread: dict[str, int] = {}
        for params in var.values():
            for p, vals in params.items():
                counts[p] = counts.get(p, 0) + 1
                spread[p] = max(spread.get(p, 0), len(vals))
        # the parameter worth opening on is the one with the most controlled
        # sweeps (everything else fixed), then the one compared in the most
        # places, then the one with the most values
        sweeps = _controlled_sweeps()
        opts = [{"label": f"{p}  ({n} model{'s' if n > 1 else ''}, "
                          f"{spread[p]} values"
                          + (f", {sweeps[p]} controlled sweep"
                             f"{'s' if sweeps[p] > 1 else ''}" if sweeps.get(p)
                             else ", envelope only") + ")", "value": p}
                for p, n in sorted(counts.items(),
                                   key=lambda kv: (-sweeps.get(kv[0], 0), -kv[1],
                                                   -spread[kv[0]],
                                                   natural_key(kv[0])))]
        value = current if current in counts else (opts[0]["value"] if opts else None)
        return opts, value

    @app.callback(Output(f"{RQ}-model", "options"), Output(f"{RQ}-model", "value"),
                  Output(f"{RQ}-ds", "options"), Output(f"{RQ}-ds", "value"),
                  Output(f"{RQ}-ann", "options"),
                  Input(f"{RQ}-param", "value"),
                  State(f"{RQ}-model", "value"),
                  prevent_initial_call=True)
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
        Output({"type": "fig-sig", "rq": RQ}, "data"),
        State(vis(RQ), "data"), Input(shown(RQ), "data"),
        Input(f"{RQ}-param", "value"), Input(f"{RQ}-model", "value"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-mode", "value"),
        Input(f"{RQ}-metric", "value"), Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), *stats_inputs(),
        Input("catalog-version", "data"),
        State({"type": "fig-sig", "rq": RQ}, "data"),
        prevent_initial_call=True)
    def update(visible, _shown, *args):
        # the catalog version is an input so exactly the visible workbench
        # re-renders once when a scan publishes; gate() signs it itself
        *inputs, _catalog, last_sig = args
        sig = gate(visible, inputs, last_sig)
        return (*_update(*inputs), sig)

    def _update(param, models_sel, ds_sel, mode, metric, prmu_sel, ann_choice,
                *stat_vals):
        from ...figures import to_plotly
        measure, k = split_metric(metric)
        empty = to_plotly({"kind": "bar", "series": []})
        var = _models_with_variation()
        if not param or not ds_sel:
            return empty, None, "", html.Div(
                "no inference parameter varies across any model's runs — "
                "nothing to compare." if not var else
                "pick a parameter and at least one dataset",
                className="muted small")
        idx = scanner.cards()
        cfg = stats_cfg(*stat_vals)
        prmu = prmu_arg(prmu_sel)
        mlab = metric_label(measure, k, prmu)
        envelope = mode == "envelope"
        models = [m for m in (models_sel or [])
                  if m in var and param in var[m]] or \
                 [m for m, params in var.items() if param in params]

        # ---- collect runs: (dataset, model) -> their value of `param` and
        # the rest of their configuration (architecture + every other
        # resolved parameter)
        plan: dict[str, tuple] = {}
        illegal_any = False
        for ds in ds_sel:
            ann = resolve_ann(ds, ann_choice)
            keys, meta = [], {}
            for mm, arch, rid, resj, vj in _runs_of(ds):
                if mm not in models:
                    continue
                res = json.loads(resj or "{}")
                info = res.get(param) or {}
                if info.get("value") is None:
                    continue
                rest = {p_: (i_ or {}).get("value") for p_, i_ in res.items()
                        if p_ != param}
                keys.append((mm, arch, rid))
                meta[(mm, arch, rid)] = (
                    info["value"],
                    any(v.get("param") == param for v in json.loads(vj or "[]")),
                    arch, rest)
            if keys:
                plan[ds] = (keys, meta, ann)
        scored_all = run_scores_many({
            ds: dict(dataset=ds, run_keys=keys, ann_key=ann, measure=measure,
                     k=k, prmu=prmu, per_doc=True)
            for ds, (keys, _meta, ann) in plan.items()})

        # ---- series: a *controlled* sweep holds everything but `param`
        # fixed (one series per fixed configuration of a model); the
        # envelope keeps the best-scoring run at each value, whatever else
        # differs — a descriptive upper envelope, not a controlled effect
        def cfg_sig(arch, rest):
            return json.dumps([arch, rest], sort_keys=True, default=str)

        fixed_of: dict[str, dict[str, dict]] = {}     # model -> sig -> (arch, rest)
        for ds, (keys, meta, _a) in plan.items():
            for key in keys:
                _v, _il, arch, rest = meta[key]
                fixed_of.setdefault(key[0], {})[cfg_sig(arch, rest)] = (arch, rest)

        def series_label(m, sig):
            name = idx.model(m).name
            if envelope:
                return name
            sigs = fixed_of.get(m, {})
            if len(sigs) <= 1:
                return name
            arch, rest = sigs[sig]
            others = [(a_, r_) for a_, r_ in sigs.values()]
            parts = []
            if len({a_ for a_, _r in others}) > 1:
                parts.append(arch)
            names = sorted({p_ for _a, r_ in others for p_ in r_})
            for p_ in names:
                vals = {json.dumps(r_.get(p_), default=str) for _a, r_ in others}
                if len(vals) > 1 and rest.get(p_) is not None:
                    parts.append(f"{p_}={param_value_str(rest[p_])}")
            return f"{name} ({', '.join(parts)})" if parts else name

        cells: dict[tuple[str, str, str], dict] = {}   # (ds, model, sig) -> by value
        n_replicates = 0
        for ds, (keys, meta, _ann) in plan.items():
            scored = scored_all[ds]
            groups: dict[tuple, dict[str, list]] = {}
            for key in keys:
                sc = scored.get(key) or {}
                if sc.get("mean") is None or sc.get("per_doc") is None:
                    continue
                val, illegal, arch, rest = meta[key]
                illegal_any = illegal_any or illegal
                sig = "*" if envelope else cfg_sig(arch, rest)
                vj = json.dumps(val, default=str)
                groups.setdefault((key[0], sig), {}).setdefault(vj, []).append(
                    (key, sc, val, illegal))
            for (m, sig), by in groups.items():
                if len(by) < 2 and not envelope:
                    continue          # nothing varies inside this configuration
                by_value = {}
                for vj, runs in by.items():
                    if envelope:
                        key, sc, val, illegal = max(runs, key=lambda t: t[1]["mean"])
                        pd_ = sc["per_doc"]
                        rids = [key[2]]
                    else:
                        # replicates of one configuration (seeds, repeats):
                        # their per-document scores are averaged on the
                        # documents they share
                        pds = [t[1]["per_doc"] for t in runs]
                        if len(pds) > 1:
                            n_replicates += len(pds) - 1
                            co = common_ords(pds)
                            pd_ = PerDoc(co, sum(values_at(x, co) for x in pds)
                                         / len(pds), pds[0].index)
                        else:
                            pd_ = pds[0]
                        val = runs[0][2]
                        illegal = any(t[3] for t in runs)
                        rids = [t[0][2] for t in runs]
                    by_value[vj] = {"value": val, "per_doc": pd_,
                                    "illegal": illegal, "runs": rids}
                cells[(ds, m, sig)] = by_value
        if not cells:
            return (empty, None, "",
                    html.Div(f"no configuration of the selected models varies "
                             f"“{param}” alone on the selected datasets"
                             + ("" if envelope else " — switch “Runs per value” "
                                "to the best-run envelope to compare runs that "
                                "also differ in other settings"),
                             className="muted small"))

        # every value seen anywhere, in value order — one column per value
        all_vals = {}
        for by_value in cells.values():
            for vj, info in by_value.items():
                all_vals[vj] = info["value"]
        col_order = sorted(all_vals, key=lambda vj: value_key(all_vals[vj]))

        # ---- matched cohort per row: the documents every value was scored
        # on. Means, intervals and the test all use it, so the table's
        # numbers are the ones the test compares.
        order = sorted(cells, key=lambda t: (natural_key(t[0]),
                                             natural_key(series_label(t[1], t[2]))))

        def _test(row):
            by_value = cells[row]
            present = [vj for vj in col_order if vj in by_value]
            common = common_ords([by_value[vj]["per_doc"] for vj in present])
            res = {"p": None, "effect": None, "common": len(common),
                   "k": len(present), "test": cfg.test_name("multi"),
                   "effect_name": cfg.effect_name("multi")}
            for vj in present:
                pd_ = by_value[vj]["per_doc"]
                vals = values_at(pd_, common) if len(common) else pd_.vals
                by_value[vj]["vals"] = vals
                by_value[vj]["mean"] = float(vals.mean()) if len(vals) else None
                by_value[vj]["n"] = len(vals)
                by_value[vj]["ci"] = cfg.mean_ci(vals)
            if len(present) >= 2 and len(common):
                res.update(cfg.multi([by_value[vj]["vals"] for vj in present]))
            return res
        tests = _pmap(_test, order)
        p_raw = [t["p"] for t in tests]
        p_adj = cfg.adjust_all(p_raw)
        tested = sum(p is not None for p in p_raw)

        # ---- table: tests first (they are the answer), then the values ------
        headers = (["Dataset", "Runs"] + p_headers(cfg) + ["Effect size", "Docs"]
                   + [param_value_str(all_vals[vj]) for vj in col_order])
        trs, tex = [], []
        for row, t, p_val, pj in zip(order, tests, p_raw, p_adj):
            ds, m, sig = row
            by_value = cells[row]
            present = [vj for vj in col_order if vj in by_value]
            best_vj = max((vj for vj in present if by_value[vj]["mean"] is not None),
                          key=lambda vj: by_value[vj]["mean"], default=None)
            lab = series_label(m, sig)
            row_h, row_t = [ds, lab], [ds, lab]
            for c in p_cells(p_val, pj, cfg):
                row_h.append(html.Span(c[0], title=f"{t['test']} over {t['common']} "
                                                   "common documents"))
                row_t.append(c[1])
            ec = effect_cell(t["effect"])
            short = EFFECT_SHORT.get(t["effect_name"], "")
            row_h.append(html.Span([html.Span(short + " ", className="muted"), ec[0]]
                                   if short and ec[1] != "—" else ec[0],
                                   title=t["effect_name"]))
            row_t.append(f"{short} {ec[1]}".strip() if ec[1] != "—" else "—")
            row_h.append(str(t["common"]))
            row_t.append(str(t["common"]))
            for vj in col_order:
                info = by_value.get(vj)
                if info is None or info.get("mean") is None:
                    row_h.append(html.Span("—", className="muted"))
                    row_t.append("—")
                    continue
                cell = value_cell(info["mean"], None, ci=info.get("ci"))
                node = (html.B(cell[0]) if vj == best_vj else cell[0])
                title = "run" + ("s " if len(info["runs"]) > 1 else " ") + \
                    ", ".join(info["runs"])
                if info["illegal"]:
                    node = html.Span([node, html.Span(" ⚠", title="value "
                                                      "violates the model card")])
                row_h.append(html.Span(node, title=title))
                row_t.append(dict(cell[1], bold=vj == best_vj))
            trs.append(row_h)
            tex.append(row_t)

        # ---- figure: one line/bar group per series, macro-averaged over the
        # datasets it was run on — only at values present on *all* of them
        # (a macro-average must not change its datasets from one value to the
        # next); a series whose datasets share fewer than two values is
        # drawn once per dataset instead
        series = []
        # values are ordered categories, evenly spaced: 512, 1024 and 128k
        # on a linear axis put two of three points on top of each other
        xpos = {vj: i for i, vj in enumerate(col_order)}
        by_series: dict[tuple, list[str]] = {}
        for (ds, m, sig) in cells:
            by_series.setdefault((m, sig), []).append(ds)
        plots: list[tuple[str, str, str, list[str]]] = []
        for (m, sig), dss in sorted(by_series.items(),
                                    key=lambda kv: natural_key(series_label(*kv[0]))):
            common_vals = [vj for vj in col_order
                           if all(vj in cells[(d, m, sig)]
                                  and cells[(d, m, sig)][vj].get("mean") is not None
                                  for d in dss)]
            if len(dss) > 1 and len(common_vals) < 2:
                for d in sorted(dss, key=natural_key):
                    vals = [vj for vj in col_order if vj in cells[(d, m, sig)]]
                    plots.append((m, sig, f"{series_label(m, sig)} · {d}", [d], vals))
            else:
                plots.append((m, sig, series_label(m, sig), dss, common_vals))
        split_any = any(len(p_[3]) == 1 and len(by_series[(p_[0], p_[1])]) > 1
                        for p_ in plots)
        within: dict[str, int] = {}
        # series side by side inside each value's slot, so their whiskers
        # never sit on top of each other
        n_pl = len(plots)
        step = 0.36 / max(1, n_pl - 1) if n_pl > 1 else 0.0
        for i, (m, sig, name, dss, vals) in enumerate(plots):
            dodge = (i - (n_pl - 1) / 2) * step
            j = within.get(m, 0)
            within[m] = j + 1
            n_m = sum(1 for p_ in plots if p_[0] == m)
            base = model_color(m)
            archs = {fixed_of.get(m, {}).get(sig, (None, None))[0]} - {None}
            shape, marker = arch_shape(next(iter(archs))) if len(archs) == 1 \
                else ("circle", "o")
            xs, ys, hv, err = [], [], [], []
            for vj in vals:
                per_ds = [(d, cells[(d, m, sig)][vj]) for d in dss]
                mean = sum(c["mean"] for _d, c in per_ds) / len(per_ds)
                ci = cfg.macro_ci([c["vals"] for _d, c in per_ds])
                xs.append(round(xpos[vj] + dodge, 3))
                ys.append(round(mean, 4))
                err.append(ci)
                hv.append(f"{name}<br>{param} = "
                          f"{param_value_str(all_vals[vj])}<br>{mlab} = "
                          f"{mean:.3f} " + _ci_hover(ci)
                          + f"(macro over {len(per_ds)} dataset"
                          f"{'s' if len(per_ds) > 1 else ''}, matched documents)<br>"
                          + "<br>".join(f"{d}: {c['mean']:.3f} (n={c['n']})"
                                        for d, c in per_ds))
            if xs:
                series.append({"name": name, "x": xs, "y": ys,
                               "hover": hv, "err": err,
                               "mode": "lines+markers", "width": 1.8,
                               "dash": DASHES[j % len(DASHES)] if n_m > 1 else "solid",
                               "mpl_marker": marker, "shape": shape,
                               "legendgroup": m,
                               "color": shade(base, 0.55 * j / max(1, n_m - 1))
                               if n_m > 1 else base})

        # the card's range only when one model is drawn: two models declare
        # different ranges for the same name (a 1 024 vs a 128 k window)
        spec_p = None
        drawn_models = {p_[0] for p_ in plots}
        if len(drawn_models) == 1:
            spec_p = idx.model(next(iter(drawn_models))).inference_specs.get(param)
        rng = []
        if spec_p:
            if spec_p.min is not None:
                rng.append(f"min {fmt_num(spec_p.min)}")
            if spec_p.max is not None:
                rng.append(f"max {fmt_num(spec_p.max)}")
            if spec_p.default is not None:
                rng.append(f"default {fmt_num(spec_p.default)}")
        drawn = sorted({p_[0] for p_ in plots},
                       key=lambda m: natural_key(idx.model(m).name))
        spec = {
            "kind": "line", "size": "2col",
            "xlabel": f"{param}" + (f"  ({' · '.join(rng)})" if rng else ""),
            "xticks": {"vals": list(range(len(col_order))),
                       "text": [param_value_str(all_vals[vj]) for vj in col_order]},
            "xrange": [-0.4, len(col_order) - 0.6],
            "ylabel": mlab, "series": series,
            "legend_items": [{"name": idx.model(m).name, "color": model_color(m),
                              "group": m, "shape": "circle", "mpl_marker": "o",
                              "line": True} for m in drawn],
            "name": f"hyperparam-{param}",
            "caption": (f"Effect of “{param}” on {mlab}"
                        + (", best observed run at each value (an upper "
                           "envelope over runs that may also differ in other "
                           "settings — descriptive, not a controlled effect)"
                           if envelope else
                           ", one series per configuration that varies it "
                           "alone (architecture and every other parameter "
                           "fixed"
                           + (f"; {n_replicates} replicate run(s) averaged per "
                              "document" if n_replicates else "") + ")")
                        + ". Each point is macro-averaged over the datasets "
                        "its series was run on, on the documents every value "
                        "shares, and only at values present on all of them"
                        + (" (series whose datasets share fewer than two "
                           "values are drawn per dataset)" if split_any else "")
                        + ". Values are resolved against each model card "
                        "(missing parameters take the declared default"
                        + ("; ⚠ marks values outside the card's legal range"
                           if illegal_any else "") + "). "
                        + (f"Error bars: {cfg.ci_text()}. " if cfg.ci_text() else "")
                        + "The table tests each row on its own over the "
                        "documents its values share ("
                        + cfg.method_text("multi", tested) + "); a significant "
                        "test says the value matters, not which value is best. "
                        + metric_caption(measure, k, prmu_sel, ann_choice,
                                         sorted(ds_sel))),
        }
        spec["table"] = {"headers": headers, "rows": tex, "label": f"hp-{param}",
                         "notes": "Bold: highest mean in the row (descriptive). "
                                  "Means over the documents all values of the "
                                  "row share. Statistics: "
                                  + cfg.method_text("multi", tested)
                                  + (f"; {cfg.ci_text()}" if cfg.ci_text() else "")
                                  + "."}
        note = html.Div(
            f"{tested} of {len(trs)} rows had enough paired documents to test. "
            "Means are over the documents every value of the row shares; bold "
            "marks the highest (descriptive — the test asks whether the value "
            "matters at all).", className="muted small", style={"marginBottom": "6px"})
        n_sig = sum(1 for pj in p_adj if pj is not None and pj < cfg.alpha)
        return (to_plotly(spec), spec, spec["caption"],
                html.Div([note, ui.fold(ui.table(headers, trs,
                                                 num_cols=set(range(2, len(headers))),
                                                 nowrap_cols={0, 1}),
                                        len(trs),
                                        f"Table · {len(trs)} rows · {n_sig} "
                                        f"significant at p<{cfg.alpha:g}"),
                          stats_note(cfg, "multi", tested)]))
