"""Home — scan control, live progress with ETA, inventory and issues."""
from __future__ import annotations

import json
import time

from dash import Input, Output, State, dcc, html, no_update

from .. import db, scanner, ui
from ..config import settings
from ..util import human_bytes, human_count, human_duration


def layout():
    st = settings()
    return html.Div([
        html.H2("Overview", className="page-title"),
        html.P(["Data root ", html.Code(str(st.data_root)),
                " — scans detect new, modified and deleted components by "
                "size/mtime + BLAKE2 content hash, then re-derive only what "
                "changed, in parallel across all CPU threads. Exact per-step "
                "timings of every scan are archived in .kpviz/scan_stats/."],
               className="page-desc"),
        html.Div([
            html.Button("Scan for changes", id="btn-scan",
                        className="btn primary", n_clicks=0),
            html.Button("Full re-scan (re-hash everything)", id="btn-rescan",
                        className="btn", n_clicks=0),
            html.Button("Cancel", id="btn-cancel", className="btn small",
                        n_clicks=0, style={"display": "none"}),
            html.Span(id="scan-headline", className="muted small"),
        ], className="flex", style={"marginBottom": "14px"}),
        html.Div(id="scan-progress"),
        html.Div(id="home-inventory"),
        html.Div(id="home-backends"),
        html.Div([
            html.H3("Issues & integrity", className="section-title"),
            ui.filter_row([
                ui.control("Group runs by", dcc.RadioItems(
                    id="issues-groupby",
                    options=[{"label": " dataset", "value": "dataset"},
                             {"label": " model", "value": "model"}],
                    value="dataset", inline=True,
                    className="kp-check kp-inline"), 240),
            ]),
            html.Div(id="home-issues"),
        ]),
        dcc.Interval(id="home-poll", interval=1000, n_intervals=0),
    ], className="page")


# ---------------------------------------------------------------------------

def _progress_panel(snap: dict):
    if not snap["running"] and not snap["finished_at"]:
        return ui.card(html.Div("No scan in this session yet — the catalog "
                                "below reflects the stored database.",
                                className="muted small"),
                       title="Scan")
    icons = {"pending": "○", "running": "◐", "done": "●"}
    rows = []
    for s in snap["steps"]:
        dur = ""
        if s.get("t_start") and s.get("t_end"):
            dur = human_duration(s["t_end"] - s["t_start"])
        rows.append(html.Div([
            html.Span(icons.get(s["status"], "○"),
                      className=f"step-ico {s['status']}"),
            html.Span(s["label"], className="step-label"),
            html.Span(dur, className="step-detail",
                      style={"minWidth": "64px"}),
            html.Span(s["detail"] or "", className="step-detail"),
        ], className="step-row"))
    n_steps = len(snap["steps"])
    n_done = sum(1 for s in snap["steps"] if s["status"] == "done")
    running = [s for s in snap["steps"] if s["status"] == "running"]
    frac_run = 0.0
    if running and running[0]["total"]:
        frac_run = min(1.0, running[0]["done"] / running[0]["total"])
    frac = (n_done + frac_run) / n_steps if n_steps else 0
    if snap["running"]:
        status_line = [
            html.B(f"{100 * frac:.0f}% "),
            html.Span(f"— ETA {human_duration(snap.get('eta_s'))} · "
                      f"elapsed {human_duration(snap.get('elapsed_s'))}",
                      className="muted")]
    elif snap.get("error"):
        status_line = [ui.badge("scan failed", "bad"),
                       html.Span(" see log below", className="muted small")]
    else:
        dur = (snap.get("finished_at") or 0) - (snap.get("started_at") or 0)
        status_line = [ui.badge("scan complete", "ok"),
                       html.Span(f" in {human_duration(dur)}",
                                 className="muted small")]
    ch = snap["changes"]
    chips = html.Div([
        ui.chip(f"{ch.get('new', 0)} new", "blue"),
        ui.chip(f"{ch.get('modified', 0)} modified"),
        ui.chip(f"{ch.get('deleted', 0)} deleted"),
        ui.chip(f"{ch.get('unchanged', 0)} unchanged"),
    ])
    return ui.card([
        html.Div(status_line, style={"marginBottom": "8px"}),
        html.Div(html.Div(className="progress-fill",
                          style={"width": f"{100 * frac:.1f}%"}),
                 className="progress-track", style={"marginBottom": "10px"}),
        chips,
        html.Div(rows, style={"margin": "8px 0 10px"}),
        html.Div("\n".join(snap["log"]), className="scan-log")
        if snap["log"] else None,
    ], title="Scan progress")


def _inventory():
    if not db.q1("SELECT 1 FROM files LIMIT 1"):
        return ui.empty_state("Nothing in the catalog yet — run a scan.")
    counts = {k: v for k, v in db.q(
        "SELECT kind, count(*) FROM files GROUP BY kind")}
    n_docs = db.q1("SELECT count(*), sum(n_words) FROM documents")
    n_gold = db.q1("SELECT sum(n) FROM gold_agg WHERE ann_key <> '@combined'")
    n_kp = db.q1("SELECT count(*), count(pos) FROM keyphrases")
    n_runs = db.q1("SELECT count(*) FROM runs")
    n_preds = db.q1("SELECT count(*), sum(n_preds) FROM preds")
    n_ds = db.q1("SELECT count(DISTINCT dataset) FROM documents")
    size = db.q1("SELECT sum(size) FROM files")
    tiles = [
        ui.stat_tile("Datasets", n_ds[0] if n_ds else 0,
                     f"{human_bytes(size[0] or 0)} on disk" if size else ""),
        ui.stat_tile("Documents", human_count(n_docs[0] if n_docs else 0),
                     f"{human_count(n_docs[1] or 0)} words" if n_docs else ""),
        ui.stat_tile("Gold keyphrases", human_count(n_gold[0] if n_gold else 0),
                     f"{human_count(n_kp[0])} unique · {human_count(n_kp[1])} POS-tagged"
                     if n_kp else ""),
        ui.stat_tile("Models", counts.get("model_card", 0),
                     f"{counts.get('arch_card', 0)} architectures"),
        ui.stat_tile("Runs", n_runs[0] if n_runs else 0,
                     f"{counts.get('batch_preds', 0)} batch files"),
        ui.stat_tile("Predictions", human_count(n_preds[1] or 0) if n_preds else 0,
                     f"over {human_count(n_preds[0] or 0)} documents" if n_preds else ""),
    ]
    return html.Div([html.H3("Catalog", className="section-title"),
                     ui.kpi_row(tiles)])


def _issues(groupby: str):
    if not db.q1("SELECT 1 FROM runs LIMIT 1"):
        return None
    idx = scanner.cards()
    rows = db.q("""SELECT dataset, model, arch, run_id, tags, coverage
                   FROM runs WHERE tags IS NOT NULL AND tags <> '[]'
                   ORDER BY dataset, model, run_id""")
    blocks = []

    if rows:
        groups: dict[str, list] = {}
        for ds, m, a, rid, tj, cov in rows:
            gk = ds if groupby == "dataset" else idx.model(m).name
            groups.setdefault(gk, []).append((ds, m, a, rid, json.loads(tj), cov))
        content = []
        for gname in sorted(groups):
            runs = groups[gname]
            content.append(html.Div([
                html.Span(gname),
                html.Span(f"{len(runs)} run(s) flagged", className="muted"),
            ], className="group-head"))
            trs = []
            for ds, m, a, rid, tags, cov in runs:
                other = idx.model(m).name if groupby == "dataset" else ds
                trs.append([
                    other, html.Code(rid[:12]), a or "—",
                    html.Span([ui.tag_chip(t) for t in tags]),
                ])
            content.append(ui.table(
                ["Model" if groupby == "dataset" else "Dataset", "Run",
                 "Architecture", "Issue tags"], trs))
        n_flag = len(rows)
        n_all = db.q1("SELECT count(*) FROM runs")[0]
        blocks.append(ui.card(
            [html.Div(f"{n_flag} of {n_all} runs carry at least one issue tag.",
                      className="muted small", style={"marginBottom": "4px"}),
             *content],
            title="Run issues"))
    else:
        blocks.append(ui.card(html.Div([
            ui.badge("all clear", "ok"),
            html.Span(" every run is complete, valid and fully linked",
                      className="muted small")]), title="Run issues"))

    # share of the dataset's own documents, in the same cell as the count —
    # "157" means nothing until you know whether the collection has 200 or 20k
    fl = db.q("""SELECT d.dataset, f.fl, count(*),
                        (SELECT count(*) FROM documents t
                          WHERE t.dataset = d.dataset)
                 FROM documents d, UNNEST(d.flags) AS f(fl)
                 GROUP BY 1, 2 ORDER BY 1, 3 DESC""")
    if fl:
        trs = [[ds, ui.tag_chip(flag),
                html.Span([f"{n:,}".replace(",", " "),
                           html.Span(f" ({100.0 * n / tot:.1f}%)"
                                     if tot else " (—)",
                                     className="muted")])]
               for ds, flag, n, tot in fl]
        blocks.append(ui.card(ui.table(
            ["Dataset", "Quality flag", "Documents"], trs, num_cols={2}),
            title="Data quality"))
    return html.Div(blocks)


def _backends_panel():
    """Which fast paths are live. A silent fallback here is the difference
    between minutes and hours, or between exact and approximate token
    counts — it must never be invisible."""
    from ..diag import backends
    from ..config import settings as _settings
    b = backends()
    rows = []
    for name, info in b.items():
        kind = {"ok": "ok", "warn": "warn", "info": "gray"}[info["level"]]
        rows.append([name, ui.badge(info["value"], kind),
                     info["note"] or ""])
    d = _settings().describe()
    tiles = [
        ui.stat_tile("Scan workers", d["workers"],
                     f"of {d['usable_cpus']} usable CPUs"),
        ui.stat_tile("DuckDB threads", d["db_threads"],
                     f"{d['duckdb_memory_gb']} GB of {d['usable_ram_gb']} GB"),
        ui.stat_tile("Chunk target", f"{d['chunk_bytes'] // 2**20} MB",
                     f"hash mode: {d['hash_mode']}"),
        ui.stat_tile("Derivation scope",
                     f"gold {d['gold_scope']} · tokens {d['token_scope']}",
                     "distribution charts always cover every split"),
    ]
    warn = [n for n, i in b.items() if i["level"] == "warn"]
    return html.Div([
        html.H3("Engine", className="section-title"),
        html.Div("Fast paths in use on this machine"
                 + (" — some are falling back, see below"
                    if warn else " — all optimal"),
                 className="section-note"),
        ui.kpi_row(tiles),
        ui.card(ui.table(["Component", "Active", "Note"], rows)),
    ])


def register(app):
    @app.callback(
        Output("scan-progress", "children"),
        Output("home-inventory", "children"),
        Output("home-backends", "children"),
        Output("home-issues", "children"),
        Output("btn-scan", "disabled"),
        Output("btn-rescan", "disabled"),
        Output("btn-cancel", "style"),
        Output("scan-headline", "children"),
        Output("catalog-version", "data"),
        Input("home-poll", "n_intervals"),
        Input("btn-scan", "n_clicks"),
        Input("btn-rescan", "n_clicks"),
        Input("btn-cancel", "n_clicks"),
        Input("issues-groupby", "value"),
        State("catalog-version", "data"),
    )
    def poll(_n, s_clicks, r_clicks, c_clicks, groupby, known_version):
        from dash import ctx
        trig = ctx.triggered_id
        if trig == "btn-scan":
            scanner.start_scan(False)
        elif trig == "btn-rescan":
            scanner.start_scan(True)
        elif trig == "btn-cancel":
            scanner.request_cancel()
        snap = scanner.STATE.snapshot()
        running = snap["running"]
        v = db.scan_version()
        version_out = v if v != known_version else no_update
        # inventory/issues only when idle (and when stale or explicitly asked)
        refresh_static = (not running) and (
            v != known_version or trig in ("btn-scan", "btn-rescan",
                                           "issues-groupby", None))
        inv = _inventory() if refresh_static else no_update
        iss = _issues(groupby or "dataset") if refresh_static else no_update
        eng = _backends_panel() if refresh_static else no_update
        headline = f"catalog version {v}" if (v and not running) else ""
        return (_progress_panel(snap), inv, eng, iss,
                running, running,
                {"display": "inline-flex"} if running else {"display": "none"},
                headline, version_out)
