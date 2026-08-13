"""Models explorer — cards, inference specs, runs, quick quality glance."""
from __future__ import annotations

import json

from dash import Input, Output, State, dcc, html

from .. import db, scanner, ui
from ..figures import to_plotly, PALETTE
from ..metrics import run_scores
from ..naming import natural_key, slot_color, window_str
from ..util import fmt_num, human_cost, human_count


def _model_options():
    idx = scanner.cards()
    models = {r[0] for r in db.q("SELECT DISTINCT model FROM runs")} | set(idx.models)
    return [{"label": idx.model(m).name, "value": m}
            for m in sorted(models, key=lambda m: natural_key(idx.model(m).name))]


def layout():
    return html.Div([
        html.H2("Models", className="page-title"),
        html.P("Model cards with their declared inference-parameter schema, "
               "every run found in the tree (validated against the card), and "
               "a quality glance across datasets.", className="page-desc"),
        ui.filter_row([
            ui.control("Model", dcc.Dropdown(
                id="md-pick", options=[], clearable=False,
                placeholder="scan first…", className="dash-dropdown"), 280),
        ]),
        html.Div(id="md-body"),
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
    from ..naming import group_key, run_labels, run_rows
    rows = db.q("""SELECT dataset, arch, run_id, resolved, violations, n_docs,
                          expected_docs, coverage, costs, wall_s
                   FROM runs WHERE model=?""", model)
    if not rows:
        return html.Div("no runs found for this model", className="muted small")
    # run ids are content hashes, so ordering by them is arbitrary noise: order
    # by what the reader sees — dataset, then the run's own label, naturally
    # (num_beams=1, 4, 10 — not 1, 10, 4)
    labels = run_labels(idx, [r for r in run_rows() if r["model"] == model])
    rows = sorted(rows, key=lambda r: (
        natural_key(r[0]),
        natural_key(labels.get(group_key(model, r[1], r[2]), r[2]))))
    trs = []
    for ds, arch, rid, resj, vj, nd, ed, cov, cj, wall in rows:
        res = json.loads(resj or "{}")
        given = {k: v["value"] for k, v in res.items()
                 if isinstance(v, dict) and v.get("source") == "given"}
        given = dict(sorted(given.items(), key=lambda kv: natural_key(kv[0])))
        chips = [ui.chip(f"{k}={fmt_num(v) if not isinstance(v, (list, str)) else (v if len(str(v)) < 14 else str(v)[:12] + '…')}",
                         "mono") for k, v in list(given.items())[:4]]
        if len(given) > 4:
            chips.append(ui.chip(f"+{len(given) - 4}"))
        viol = json.loads(vj or "[]")
        costs = json.loads(cj or "{}")
        cost_txt = [human_cost(unit, (costs.get(unit) or {}).get("total"))
                    for unit in ("usd", "kwh", "time")
                    if (costs.get(unit) or {}).get("known")]
        # documents and coverage answer the same question — how much of the
        # collection this run actually produced — so they share one cell
        docs = html.Span([human_count(nd),
                          html.Span(f" ({ui.pct(cov, 0)})", className="muted")
                          if cov is not None else ""])
        trs.append([
            ds, arch or "—", html.Code(rid[:12]),
            html.Span(chips),
            ui.badge(f"{len(viol)} illegal", "bad") if viol else ui.badge("valid", "ok"),
            docs,
            " · ".join(cost_txt) or "—",
        ])
    return ui.table(["Dataset", "Architecture", "Run", "Parameters",
                     "Quality check", "Documents", "Total cost"],
                    trs, num_cols={5})


def _quality_glance(model):
    from ..naming import group_key, run_labels, run_rows
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

    series_by_run: dict[str, dict] = {}
    datasets = sorted({r[0] for r in runs})
    for ds in datasets:
        anns = [r[0] for r in db.q(
            "SELECT DISTINCT ann_key FROM gold_agg WHERE dataset=? ORDER BY 1", ds)]
        ann = "@combined" if "@combined" in anns else (anns[0] if anns else None)
        if not ann:
            continue
        keys = [(model, a, rid) for d2, a, rid in runs if d2 == ds]
        sc = run_scores(ds, keys, ann, "f1", "O")
        for (m, a, rid), v in sc.items():
            series_by_run.setdefault(short_label(a, rid), {})[ds] = \
                (round(v["mean"], 3) if v["mean"] is not None else None)
    if not series_by_run:
        return None
    series = []
    for i, (lab, per_ds) in enumerate(
            sorted(series_by_run.items(), key=lambda kv: natural_key(kv[0]))):
        series.append({"name": lab, "x": datasets,
                       "y": [per_ds.get(d) for d in datasets],
                       "color": slot_color(i)})
    spec = {"kind": "bar", "xlabel": "dataset", "ylabel": "F1@O (combined gold)",
            "series": series}
    return ui.card([ui.graph("md-quality", to_plotly(spec), 300)],
                   title="Quality glance — F1@O per dataset and run")


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
            *[ui.meta_chip("can", c) for c, on in caps.items() if on],
            *[ui.meta_chip("lang", l) for l in mcard.languages],
            *[ui.meta_chip("trained on", sup) for sup in mcard.supervision],
            *[ui.meta_chip("domain", d.get("domain")) for d in mcard.domains
              if d.get("domain")],
            *[ui.meta_chip("sub-domain", d.get("sub-domain"))
              for d in mcard.domains if d.get("sub-domain")],
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
        html.Div(bibtex, className="mono-block"),
        html.Div([dcc.Clipboard(content=bibtex, title="copy BibTeX",
                                className="btn small",
                                style={"display": "inline-flex"}),
                  html.Span("copy BibTeX", className="small muted")],
                 className="flex", style={"marginTop": "8px"}),
    ], title="Reference") if bibtex else None

    return html.Div([
        header,
        ui.card(_spec_table(mcard), title="Inference parameter schema"),
        ui.card(_runs_table(scanner.cards(), model),
                title="Runs found in the tree"),
        _quality_glance(model),
        bib_card,
    ])


def register(app):
    @app.callback(Output("md-pick", "options"), Output("md-pick", "value"),
                  Input("catalog-version", "data"), State("md-pick", "value"))
    def refresh_models(_v, current):
        opts = _model_options()
        vals = {o["value"] for o in opts}
        value = current if current in vals else (opts[0]["value"] if opts else None)
        return opts, value

    @app.callback(Output("md-body", "children"), Input("md-pick", "value"))
    def body(model):
        if not model:
            return ui.empty_state("No models found — add model cards and scan.")
        return _body(model)
