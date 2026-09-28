"""Models explorer — cards, inference specs, runs, quick quality glance."""
from __future__ import annotations

import json

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..figures import to_plotly
from ..metrics import run_scores
from ..naming import arch_label, natural_key, window_str
from ..util import UNIT_HEAD, fmt_num, human_cost, human_count


def _model_options():
    idx = scanner.cards()
    models = {r[0] for r in db.q("SELECT DISTINCT model FROM runs")} | set(idx.models)
    return [{"label": idx.model(m).name, "value": m}
            for m in sorted(models, key=lambda m: natural_key(idx.model(m).name))]


def layout():
    return html.Div([
        html.H1("Models", className="page-title"),
        html.P("A model card, its runs (checked against the card) and how "
               "they score on each dataset.", className="page-desc"),
        ui.filter_row([
            ui.control("Model", dcc.Dropdown(
                id="md-pick", options=[], clearable=False,
                placeholder="scan first…", className="dash-dropdown"), 280),
        ]),
        ui.loading(html.Div(id="md-body")),
    ], className="page")


def _spec_table(mcard):
    rows = []
    for name, ps in mcard.inference_specs.items():
        rng = []
        if ps.values is not None:
            vals = ", ".join(str(v) for v in ps.values[:8])
            rng.append(f"∈ {{{vals}{', …' if len(ps.values) > 8 else ''}}}")
        if ps.min is not None:
            rng.append(f"≥ {fmt_num(ps.min)}")
        if ps.max is not None:
            rng.append(f"≤ {fmt_num(ps.max)}")
        # only context windows are tokenizer-dependent — show it inside Type
        type_cell = (html.Span([html.Code(ps.type),
                                html.Span(" · ", className="muted"),
                                html.Code(ps.tokenizer, className="small")])
                     if ps.tokenizer else html.Code(ps.type))
        rows.append([html.Code(name), type_cell, " · ".join(rng) or "—",
                     fmt_num(ps.default) if ps.default is not None else "—"])
    if not rows:
        return html.Div("no inference schema declared", className="muted small")
    return ui.table(["Parameter", "Type", "Constraints", "Default"], rows)


def _runs_table(idx, model):
    """The runs of one model: only the parameters that *vary* between them
    are chips (the constants are said once, above), each run is named by
    its label (the hash is its tooltip), costs get a column per unit."""
    from ..naming import (distinct_params, group_key, param_value_str,
                          run_labels, run_rows, short_run)
    rows = db.q("""SELECT dataset, arch, run_id, resolved, violations, n_docs,
                          expected_docs, coverage, costs, tags
                   FROM runs WHERE model=?""", model)
    if not rows:
        return html.Div("no runs found for this model", className="muted small")
    labels = run_labels(idx, [r for r in run_rows() if r["model"] == model])
    # run ids are content hashes, so ordering by them is arbitrary noise: order
    # by what the reader sees — dataset, then the run's own label, naturally
    # (num_beams=1, 4, 10 — not 1, 10, 4)
    rows = sorted(rows, key=lambda r: (
        natural_key(r[0]),
        natural_key(labels.get(group_key(model, r[1], r[2]), r[2]))))
    resolved = [json.loads(r[3] or "{}") for r in rows]
    varying = set(distinct_params(resolved))
    given = [{k: v["value"] for k, v in res.items()
              if isinstance(v, dict) and v.get("source") == "given"}
             for res in resolved]
    # the same given value in every run: said once, not on every row
    common = {k: v for k, v in (given[0] if given else {}).items()
              if k not in varying and all(g.get(k) == v for g in given)}
    units = [u for u in ("usd", "kwh", "time")
             if any((json.loads(r[8] or "{}").get(u) or {}).get("known")
                    for r in rows)]
    trs = []
    for (ds, arch, rid, _resj, vj, nd, ed, cov, cj, tj), res in zip(rows, resolved):
        vary = sorted((p for p in varying if (res.get(p) or {}).get("value") is not None),
                      key=natural_key)
        # on a model's own page a run is what sets it apart from the others
        # (its varying parameters); the content hash is the tooltip
        run_cell = html.Span([ui.chip(f"{p}={param_value_str(res[p]['value'])}", "mono")
                              for p in vary] or [html.Code(short_run(rid))],
                             className="chips", title=f"run id {rid}")
        viol = json.loads(vj or "[]")
        tags = json.loads(tj or "[]")
        if viol:
            check = ui.badge(f"{len(viol)} illegal", "bad")
        elif "missing:run" in tags or "unreadable:run card" in tags:
            check = html.Span(ui.badge("defaults assumed", "warn"),
                              title="no readable run_*.json: the model card's "
                                    "defaults stand in for the parameters")
        else:
            check = html.Span(ui.badge("matches card", "ok"),
                              title="every parameter is one the model card "
                                    "declares, within its constraints")
        costs = json.loads(cj or "{}")
        docs = html.Span([human_count(nd),
                          html.Span(f" ({ui.pct(cov, 0)})", className="muted")
                          if cov is not None else ""],
                         title=(f"{nd:,} of {ed:,} documents of the evaluation "
                                "split" if ed else f"{nd:,} documents"))
        trs.append([
            ds, run_cell, arch_label(idx, arch), check, docs,
            *[human_cost(u, (costs.get(u) or {}).get("total"))
              if (costs.get(u) or {}).get("known") else "—" for u in units],
        ])
    n_bad = sum(1 for r in rows if json.loads(r[4] or "[]"))
    n_ds = len({r[0] for r in rows})
    heads = ["Dataset", "Run", "Architecture", "Check", "Documents"] + \
        [UNIT_HEAD[u] for u in units]
    table = ui.table(heads, trs, num_cols={4, *range(5, 5 + len(units))},
                     nowrap_cols={3, 4, *range(5, 5 + len(units))})
    constants = (html.Div([html.Span("Same in every run: ", className="muted small"),
                           *[ui.chip(f"{k}={param_value_str(v)}", "mono")
                             for k, v in sorted(common.items(),
                                                key=lambda kv: natural_key(kv[0]))]],
                          className="chips", style={"marginBottom": "8px"})
                 if common else None)
    summary = (f"{len(trs)} runs on {n_ds} dataset{'s' if n_ds > 1 else ''}"
               + (f" · {n_bad} with illegal parameters" if n_bad else
                  " · all match the card"))
    return html.Div([html.Div(summary, className="muted small",
                              style={"marginBottom": "6px"}),
                     constants, ui.fold(table, len(trs), "Show the runs", limit=12)])



def _quality_glance(model):
    from ..naming import group_key, parse_group_key, run_labels, run_rows
    runs = db.q("SELECT DISTINCT dataset, arch, run_id FROM runs WHERE model=?",
                model)
    if not runs:
        return None
    idx = scanner.cards()
    name = idx.model(model).name
    labels = run_labels(idx, [r for r in run_rows() if r["model"] == model])

    def short_label(arch, rid):
        lab = labels.get(group_key(model, arch, rid), rid[:10])
        if lab.startswith(name):
            lab = lab[len(name):].strip(" ·")
        return lab.strip("()") or rid[:10]

    from ..naming import encode_runs
    enc = encode_runs(idx, [r for r in run_rows() if r["model"] == model])
    by_run: dict[str, dict] = {}
    datasets = sorted({r[0] for r in runs}, key=natural_key)
    for ds in datasets:
        anns = [r[0] for r in db.q(
            "SELECT DISTINCT ann_key FROM gold_agg WHERE dataset=? ORDER BY 1", ds)]
        ann = "@combined" if "@combined" in anns else (anns[0] if anns else None)
        if not ann:
            continue
        keys = [(model, a, rid) for d2, a, rid in runs if d2 == ds]
        sc = run_scores(ds, keys, ann, "f1", "O")
        for (m, a, rid), v in sc.items():
            if v["mean"] is not None:
                by_run.setdefault(group_key(m, a, rid), {})[ds] = round(v["mean"], 3)
    if not by_run:
        return None
    # a dot per run and dataset, dodged sideways inside the dataset's slot:
    # position carries the score, the model's hue (lighter = another run)
    # and the architecture's shape carry identity — no rainbow of bars
    ordered = sorted(by_run, key=lambda k: natural_key(enc.get(k, {}).get("label", k)))
    n = len(ordered)
    width = min(0.7, 0.09 * n)
    series = []
    for i, k in enumerate(ordered):
        e = enc.get(k, {})
        per_ds = by_run[k]
        off = (i - (n - 1) / 2) * (width / max(1, n - 1)) if n > 1 else 0.0
        lab = short_label(*parse_group_key(k)[1:])
        xs = [j + off for j, d in enumerate(datasets) if d in per_ds]
        ys = [per_ds[d] for d in datasets if d in per_ds]
        series.append({"name": lab, "x": xs, "y": ys, "mode": "markers",
                       "color": e.get("color"), "shape": e.get("shape", "circle"),
                       "mpl_marker": e.get("mpl_marker", "o"), "size": 10,
                       "edge": e.get("base"),
                       "hover": [f"{lab}<br>{d}: F1@O = {per_ds[d]:.3f}"
                                 for d in datasets if d in per_ds]})
    top = max(max(s_["y"]) for s_ in series)
    spec = {"kind": "scatter", "xlabel": "dataset",
            "ylabel": "F1@O", "series": series, "size": "2col",
            # from zero, so a 0.03 gap looks like 0.03 on every model's page
            "yrange": [0, min(1.0, round(top * 1.15 + 0.005, 2)) or 1.0],
            "xticks": {"vals": list(range(len(datasets))), "text": datasets},
            "xrange": [-0.6, len(datasets) - 0.4],
            "legend": "right" if n > 6 else "top",
            "name": f"quality-{model}",
            "caption": (f"F1@O of every run of {name} per dataset, against the "
                        "union of the dataset's annotation sets (or its only "
                        "one), macro-averaged over documents.")}
    n_runs = len(ordered)
    return ui.exportable(
        "md-quality", spec,
        html.Div([html.Div(f"{n_runs} run{'s' if n_runs > 1 else ''} of {name} on "
                           f"{len(datasets)} dataset{'s' if len(datasets) > 1 else ''} · "
                           "macro-averaged over documents · colour = run, "
                           "shape = architecture", className="rq-head"),
                  ui.graph("md-quality", to_plotly(spec), 320)],
                 role="figure", **{"aria-label": f"F1@O of every run of {name}"}),
        title="How it scores — F1@O per dataset")


def _body(model):
    idx = scanner.cards()
    mcard = idx.model(model)
    refs = mcard.references
    caps = mcard.capabilities
    link = refs.get("url") or refs.get("huggingface_id")
    header = ui.card([
        html.Div([html.B(mcard.name),
                  html.Span(f"  ·  {mcard.model_id}", className="muted small")],
                 style={"marginBottom": "9px"}),
        ui.meta_row([
            *[ui.meta_chip("family", " › ".join(path)) for path in mcard.family],
            ui.meta_chip("backend", mcard.backend) if mcard.backend else None,
            ui.meta_chip("params", human_count(mcard.n_parameters))
            if mcard.n_parameters else None,
            _joined("can", [c for c, on in caps.items() if on]),
            _joined("lang", mcard.languages),
            _joined("trained on", mcard.supervision),
            _joined("domain", [d.get("domain") for d in mcard.domains]),
            _joined("sub-domain", [d.get("sub-domain") for d in mcard.domains]),
            *[ui.meta_chip("context", window_str(ps.tokenizer, ps.default, True))
              for ps in mcard.context_params() if "input" in ps.name
              and ps.default],
            ui.meta_chip("cutoff", (mcard.raw.get("knowledge_cutoff") or "")[:10])
            if mcard.raw.get("knowledge_cutoff") else None,
            ui.meta_chip("license", refs.get("license")) if refs.get("license")
            else None,
        ]),
        html.Div([html.A("model page ↗", href=link, target="_blank",
                         style={"fontSize": "12.5px"})] if link else [],
                 style={"marginTop": "8px"}),
    ] if mcard.raw else [html.Div([
        ui.badge("no model card", "warn"),
        html.Span(f" folder token “{model}” has no models/model.*.json — runs "
                  "are kept but nothing can be validated or defaulted.",
                  className="muted small")])])

    bibtex = refs.get("bibtex")
    bib_card = ui.card([
        dcc.Clipboard(content=bibtex, title="copy the BibTeX entry",
                      className="btn small clip-btn clip-bib",
                      style={"display": "inline-flex"}),
        html.Details([html.Summary("Show the entry"),
                      html.Div(bibtex, className="mono-block")],
                     className="fold"),
    ], title="Reference") if bibtex else None

    # what the reader came for first: who it is, how it scores, which runs
    # say so; the schema and the reference are for checking
    return html.Div([
        header,
        _quality_glance(model),
        ui.card(_runs_table(idx, model), title="Runs"),
        ui.card(_spec_table(mcard), title="Inference parameters (from the card)"),
        bib_card,
    ])


def _joined(key, values):
    """One chip per kind of fact: "lang en · fr · de", not three chips."""
    vals = list(dict.fromkeys(str(v) for v in values if v))
    return ui.meta_chip(key, " · ".join(vals)) if vals else None


def register(app):
    @app.callback(Output("md-pick", "options"), Output("md-pick", "value"),
                  State("vis-models", "data"), Input("shown-models", "data"),
                  Input("catalog-version", "data"),
                  State("md-pick", "options"), State("md-pick", "value"),
                  prevent_initial_call=True)
    def refresh_models(visible, _shown, _v, cur_opts, current):
        if not visible:
            raise PreventUpdate
        opts = _model_options()
        if cur_opts == opts and current:
            raise PreventUpdate
        vals = {o["value"] for o in opts}
        value = current if current in vals else (opts[0]["value"] if opts else None)
        return opts, value

    @app.callback(Output("md-body", "children"), Input("md-pick", "value"),
                  prevent_initial_call=True)
    def body(model):
        if not model:
            return ui.empty_state("No models found — add model cards and scan.")
        return _body(model)
