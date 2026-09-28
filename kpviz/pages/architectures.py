"""Architectures explorer — hardware, cost variables, rates, spend."""
from __future__ import annotations

import json

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..costs import cost_formula
from ..naming import group_key, run_labels, run_rows
from ..textproc import tokenizer_inner
from ..util import UNIT_HEAD, human_cost, human_count


def _canon(idx, token: str) -> str:
    """Canonical identity of an arch token (card key when it resolves)."""
    card = idx.arch(token)
    return f"card:{card.arch_key}" if card.known else f"raw:{token}"


def _tokens_of(idx, token: str) -> list[str]:
    """Every run-table token that resolves to the same architecture."""
    mine = _canon(idx, token)
    toks = {token} | {r[0] for r in db.q("SELECT DISTINCT arch FROM runs")
                      if _canon(idx, r[0]) == mine}
    return sorted(toks)


def _arch_options():
    idx = scanner.cards()
    tokens = sorted(set(list(idx.archs) +
                        [r[0] for r in db.q("SELECT DISTINCT arch FROM runs")]))
    # a run may carry "n.a" (declared *absent*, on purpose) or a token with no
    # card; neither is an architecture, so neither belongs in this picker. Such
    # runs stay first-class everywhere else and are tagged on the Overview.
    seen, opts = set(), []
    for t in tokens:
        if not idx.arch(t).known:
            continue
        c = _canon(idx, t)
        if c in seen:
            continue
        seen.add(c)
        opts.append({"label": idx.arch(t).name, "value": t})
    return opts


def layout():
    return html.Div([
        html.H1("Architectures", className="page-title"),
        html.P("Where inference ran: the cost variables each card declares, "
               "the rates that price them, and what its runs cost.",
               className="page-desc"),
        ui.filter_row([
            ui.control("Architecture", dcc.Dropdown(
                id="ar-pick", options=[], clearable=False,
                placeholder="scan first…", className="dash-dropdown"), 320),
        ]),
        html.Div(id="ar-body"),
    ], className="page")


def _body(token):
    idx = scanner.cards()
    ac = idx.arch(token)
    if not ac.known:
        runs = db.q("""SELECT dataset, model, run_id FROM runs WHERE arch=?
                       ORDER BY dataset""", token)
        return html.Div([
            ui.card([ui.badge("unresolved architecture", "warn"),
                     html.Span(f"  “{token}” matches no architectures/architecture.*.json "
                               "card. Runs under it stay fully usable for quality "
                               "analysis; every cost is unknown (Insights → Quality "
                               "vs. cost lists them beside the figure).",
                               className="muted small")]),
            ui.card(ui.table(["Dataset", "Model", "Run"],
                             [[d, idx.model(m).name, html.Code(r[:12], title=r)]
                              for d, m, r in runs]),
                    title=f"Runs on this token · {len(runs)}") if runs else None,
        ])

    sw = ac.raw.get("software") or {}
    hw = ac.raw.get("hardware") or {}
    cpu, gpu = hw.get("cpu") or {}, hw.get("gpu") or {}
    header = ui.card([
        html.Div([html.B(ac.name),
                  html.Span(f"  ·  {ac.raw.get('arch_id')}", className="muted small")],
                 style={"marginBottom": "9px"}),
        ui.meta_row([
            ui.meta_chip("kind", ac.kind) if ac.kind else None,
            ui.meta_chip("cpu", " ".join(str(x) for x in
                                         (cpu.get("vendor"), cpu.get("model")) if x))
            if cpu else None,
            ui.meta_chip("cpu", f"{cpu['cores']} cores") if cpu.get("cores") else None,
            ui.meta_chip("ram", f"{cpu['ram_gb']} GB") if cpu.get("ram_gb") else None,
            ui.meta_chip("gpu", f"{gpu.get('count', 1)}× "
                                + " ".join(str(x) for x in
                                           (gpu.get("vendor"), gpu.get("model")) if x))
            if gpu else None,
            ui.meta_chip("gpu", f"{gpu['vram_gb']} GB VRAM")
            if gpu.get("vram_gb") else None,
            *[ui.meta_chip(k, v) for k, v in sw.items()
              if not isinstance(v, dict)],
            *[ui.meta_chip("pkg", f"{k} {v}")
              for k, v in (sw.get("packages") or {}).items()],
        ]),
        html.Div(ac.raw.get("_note", ""), className="muted small",
                 style={"marginTop": "9px"}),
    ])

    var_rows = []
    for var, spec in ac.variables.items():
        unit = spec.get("unit", "—")
        if spec.get("tokenizer"):
            # only token counts are tokenizer-dependent: unit -> token[o200k_base]
            unit = html.Code(f"{unit}[{tokenizer_inner(spec['tokenizer'])}]")
        var_rows.append([html.Code(var), unit,
                         ui.badge(spec.get("level", "?"),
                                  "info" if spec.get("level") == "document" else "gray"),
                         spec.get("description", "—")])
    vars_card = ui.card(ui.table(
        ["Variable", "Unit", "Level", "Description"], var_rows),
        title="Raw cost variables (declare-before-use)")

    toks = _tokens_of(idx, token)
    runs = db.q(f"""SELECT dataset, model, arch, run_id, n_docs, costs
                    FROM runs WHERE arch IN ({','.join('?' * len(toks))})
                    ORDER BY dataset, model""", *toks)
    labels = run_labels(idx, [r for r in run_rows() if r["arch"] in toks])
    totals: dict[str, float] = {}
    trs = []
    example = None             # the largest priced run, as a worked example
    for ds, m, a, rid, nd, cj in runs:
        costs = json.loads(cj or "{}")
        cells = []
        for unit in ac.cost_units:
            c = costs.get(unit)
            if c and c.get("known"):
                totals[unit] = totals.get(unit, 0.0) + (c["total"] or 0)
                cells.append(human_cost(unit, c["total"]))
                if nd and (example is None or nd > example[2]):
                    example = (labels.get(group_key(m, a, rid), idx.model(m).name),
                               ds, nd, unit, c["total"])
            else:
                cells.append("—")
        trs.append([ds, html.Span(labels.get(group_key(m, a, rid), idx.model(m).name),
                                  title=f"run id {rid}"),
                    human_count(nd), *cells])

    rate_rows = [[UNIT_HEAD.get(unit, unit), html.Code(cost_formula(ac, unit))]
                 for unit in ac.cost_units]
    worked = None
    if example:
        lab, ds, nd, unit, tot = example
        worked = html.Div([
            html.Span("Worked example: ", className="muted"),
            f"{lab} on {ds} predicted {nd:,} documents → "
            f"{human_cost(unit, tot)} in all, "
            f"{human_cost(unit, tot * 1000 / nd)} per 1,000 documents."],
            className="small worked")
    rates_card = ui.card([ui.table(["Cost", "Linear model"], rate_rows), worked],
                         title="Rates: how a run is priced")

    tiles = []
    for unit, tot in totals.items():
        label = {"usd": "Total spend", "kwh": "Total energy",
                 "time": "Total wall time"}.get(unit, unit)
        tiles.append(ui.stat_tile(label, human_cost(unit, tot),
                                  f"across {len(runs)} run{'s' if len(runs) > 1 else ''}"))
    n_m = len({m for _d, m, *_r in runs})
    n_d = len({d for d, *_r in runs})
    runs_card = ui.card([
        html.Div(f"{len(trs)} run{'s' if len(trs) > 1 else ''} of {n_m} "
                 f"model{'s' if n_m > 1 else ''} on {n_d} "
                 f"dataset{'s' if n_d > 1 else ''}", className="muted small",
                 style={"marginBottom": "6px"}),
        ui.fold(ui.table(["Dataset", "Run", "Documents",
                          *[UNIT_HEAD.get(u, u) for u in ac.cost_units]], trs,
                         num_cols=set(range(2, 3 + len(ac.cost_units)))),
                len(trs), "Show the runs", limit=12)],
        title="Runs on this architecture") if runs else None

    return html.Div([header, ui.kpi_row(tiles) if tiles else None,
                     runs_card,
                     html.Div([rates_card, vars_card], className="grid-2")])


def register(app):
    @app.callback(Output("ar-pick", "options"), Output("ar-pick", "value"),
                  State("vis-architectures", "data"), Input("shown-architectures", "data"),
                  Input("catalog-version", "data"), State("ar-pick", "value"),
                  prevent_initial_call=True)
    def refresh_archs(visible, _shown, _v, current):
        if not visible:
            raise PreventUpdate
        opts = _arch_options()
        vals = {o["value"] for o in opts}
        value = current if current in vals else (opts[0]["value"] if opts else None)
        return opts, value

    @app.callback(Output("ar-body", "children"), Input("ar-pick", "value"),
                  prevent_initial_call=True)
    def body(token):
        if not token:
            return ui.empty_state("No architectures found — add cards and scan.")
        return _body(token)
