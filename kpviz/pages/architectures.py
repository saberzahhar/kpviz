"""Architectures explorer — hardware, cost variables, rates, spend."""
from __future__ import annotations

import json

from dash import Input, Output, State, dcc, html

from .. import db, scanner, ui
from ..costs import cost_formula
from ..textproc import tokenizer_inner
from ..util import fmt_num, human_cost, human_count, human_duration


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
        html.H2("Architectures", className="page-title"),
        html.P("Where inference physically ran. Each card declares raw cost "
               "variables (with a document or batch level) and linear rate "
               "models per cost unit; run costs are resolved against these — "
               "never invented.", className="page-desc"),
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
                               "analysis; every cost is treated as unknown (drawn "
                               "as dashed performance-only lines).",
                               className="muted small")]),
            ui.card(ui.table(["Dataset", "Model", "Run"],
                             [[d, idx.model(m).name, html.Code(r[:12])]
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
                                  "info" if spec.get("level") == "document" else "warn"),
                         spec.get("description", "—")])
    vars_card = ui.card(ui.table(
        ["Variable", "Unit", "Level", "Description"], var_rows),
        title="Raw cost variables (declare-before-use)")

    rate_rows = [[html.Code(unit), html.Code(cost_formula(ac, unit))]
                 for unit in ac.cost_units]
    rates_card = ui.card(ui.table(["Cost unit", "Linear model"], rate_rows),
                         title="Rates — cost models")

    toks = _tokens_of(idx, token)
    runs = db.q(f"""SELECT dataset, model, run_id, n_docs, costs, wall_s
                    FROM runs WHERE arch IN ({','.join('?' * len(toks))})
                    ORDER BY dataset, model""", *toks)
    totals: dict[str, float] = {}
    trs = []
    for ds, m, rid, nd, cj, wall in runs:
        costs = json.loads(cj or "{}")
        cells = []
        for unit in ac.cost_units:
            c = costs.get(unit)
            if c and c.get("known"):
                totals[unit] = totals.get(unit, 0.0) + (c["total"] or 0)
                cells.append(human_cost(unit, c["total"]))
            else:
                cells.append("—")
        trs.append([ds, idx.model(m).name, html.Code(rid[:12]),
                    human_count(nd), *cells])
    tiles = []
    for unit, tot in totals.items():
        label = {"usd": "Total spend (USD)", "kwh": "Total energy (kWh)",
                 "time": "Total wall time"}.get(unit, unit)
        val = human_cost(unit, tot)
        tiles.append(ui.stat_tile(label, val, f"across {len(runs)} runs"))
    runs_card = ui.card(ui.table(
        ["Dataset", "Model", "Run", "Documents", *ac.cost_units], trs,
        num_cols=set(range(3, 4 + len(ac.cost_units)))),
        title=f"Runs on this architecture · {len(runs)}") if runs else None

    return html.Div([header, ui.kpi_row(tiles) if tiles else None,
                     html.Div([vars_card, rates_card], className="grid-2"),
                     runs_card])


def register(app):
    @app.callback(Output("ar-pick", "options"), Output("ar-pick", "value"),
                  Input("catalog-version", "data"), State("ar-pick", "value"))
    def refresh_archs(_v, current):
        opts = _arch_options()
        vals = {o["value"] for o in opts}
        value = current if current in vals else (opts[0]["value"] if opts else None)
        return opts, value

    @app.callback(Output("ar-body", "children"), Input("ar-pick", "value"))
    def body(token):
        if not token:
            return ui.empty_state("No architectures found — add cards and scan.")
        return _body(token)
