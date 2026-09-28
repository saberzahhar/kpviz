"""Home — scan control, live progress with ETA, inventory and issues."""
from __future__ import annotations

import json

from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..config import settings
from ..util import human_bytes, human_count, human_duration, stable_hash


def layout():
    st = settings()
    return html.Div([
        html.H2("Overview", className="page-title"),
        html.P(["Data root ", html.Code(str(st.data_root)),
                ". A scan re-derives only the files that changed."],
               className="page-desc"),
        html.Div([
            html.Button("Scan for changes", id="btn-scan",
                        className="btn primary", n_clicks=0),
            html.Button("Full re-scan", id="btn-rescan",
                        className="btn", n_clicks=0,
                        title="Re-hash every file and re-derive every "
                              "collection and run"),
            html.Button("Retry tokenizer downloads", id="btn-retry-tok",
                        className="btn small ghost", n_clicks=0,
                        title="Forget which tokenizer assets were unavailable "
                              "and try again on the next scan"),
            html.Button("Cancel", id="btn-cancel", className="btn small",
                        n_clicks=0, style={"display": "none"}),
            html.Span(id="scan-headline", className="muted small"),
        ], className="flex", style={"marginBottom": "14px"}),
        html.Div(id="scan-progress", children=_progress_panel(
            scanner.STATE.snapshot())),
        html.Div(id="home-inventory"),
        html.Div(id="home-issues"),
        html.Div(id="home-backends"),
        dcc.Store(id="scan-panel-sig"),
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
    if snap.get("skipped") and not snap["running"]:
        return ui.card(html.Div([
            ui.badge("up to date", "ok"),
            html.Span(" nothing changed since the last scan — the catalog was "
                      "left as it was (no re-derivation, no cache invalidation)",
                      className="muted small")]), title="Scan")
    if snap["running"]:
        status_line = [
            html.B(f"{100 * frac:.0f}% "),
            html.Span(f"— ETA {human_duration(snap.get('eta_s'))} · "
                      f"elapsed {human_duration(snap.get('elapsed_s'))}",
                      className="muted")]
    elif snap.get("error"):
        status_line = [ui.badge("scan failed", "bad"),
                       html.Span(" see log below", className="muted small")]
    elif snap.get("cancelled"):
        status_line = [ui.badge("scan cancelled", "warn"),
                       html.Span(" the next scan redoes the unfinished work",
                                 className="muted small")]
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
    n_kp = db.q1("SELECT count(*), count(nullif(pos, '')) FROM keyphrases")
    n_runs = db.q1("SELECT count(*) FROM runs")
    n_preds = db.q1("SELECT count(*), sum(n_preds), "
                    "count(DISTINCT dataset || chr(0) || doc_id) FROM preds")
    n_ds = db.q1("SELECT count(DISTINCT dataset) FROM documents")
    size = db.q1("SELECT sum(size) FROM files")
    tiles = [
        ui.stat_tile("Datasets", n_ds[0] if n_ds else 0,
                     f"{human_bytes(size[0] or 0)} on disk" if size else ""),
        ui.stat_tile("Documents", human_count(n_docs[0] if n_docs else 0),
                     f"{human_count(n_docs[1] or 0)} words" if n_docs else ""),
        ui.stat_tile("Gold keyphrases", human_count(n_gold[0] if n_gold else 0),
                     f"{human_count(n_kp[0])} distinct phrases · "
                     f"{human_count(n_kp[1])} POS-tagged"
                     if n_kp else ""),
        ui.stat_tile("Models", counts.get("model_card", 0),
                     f"{counts.get('arch_card', 0)} architectures"),
        ui.stat_tile("Runs", n_runs[0] if n_runs else 0,
                     f"{counts.get('batch_preds', 0)} batch files"),
        ui.stat_tile("Predictions", human_count(n_preds[1] or 0) if n_preds else 0,
                     f"{human_count(n_preds[0] or 0)} (run, document) lines · "
                     f"{human_count(n_preds[2] or 0)} documents" if n_preds else ""),
    ]
    return html.Div([html.H3("Catalog", className="section-title"),
                     ui.kpi_row(tiles)])


_TAG_KEEP_VALUE = ("illegal parameter", "missing", "unresolved")


def _tag_group(tag: str) -> tuple[str, str]:
    """(group, value): a categorical tag keeps its value in the group
    ("missing:architecture"), a count does not ("unscored:5" -> "unscored")."""
    key, _, val = str(tag).partition(":")
    if key.startswith(_TAG_KEEP_VALUE):
        return tag, ""
    return key, val


def _few(names, n: int = 3) -> str:
    names = sorted(set(names), key=str)
    return ", ".join(names[:n]) + (f" +{len(names) - n}" if len(names) > n else "")


def _issues():
    """Everything that needs attention, one row per kind of issue: what, how
    widespread, what it means and what KPViz did about it. The per-run list
    is one click away."""
    idx = scanner.cards()
    rows_out = []            # (severity, chip, where, meaning)

    runs = db.q("""SELECT dataset, model, arch, run_id, tags FROM runs
                   WHERE tags IS NOT NULL AND tags <> '[]'
                   ORDER BY dataset, model, run_id""")
    groups: dict[str, list] = {}
    for ds, m, a, rid, tj in runs:
        for t in json.loads(tj):
            g, val = _tag_group(t)
            groups.setdefault(g, []).append((ds, idx.model(m).name, val))
    for g, hits in groups.items():
        n_runs = len(hits)
        vals = [v for *_x, v in hits if v]
        span = ""
        if vals:
            try:
                nums = sorted(float(v.rstrip("%")) for v in vals)
                unit = "%" if vals[0].endswith("%") else ""
                lo, hi = nums[0], nums[-1]
                span = (f" · {lo:g}{unit}" if lo == hi
                        else f" · {lo:g}–{hi:g}{unit}")
            except ValueError:
                span = ""
        where = (f"{n_runs} run{'s' if n_runs > 1 else ''}{span} · "
                 f"{_few(h[1] for h in hits)} · on {_few(h[0] for h in hits)}")
        sev = 0 if g.startswith(("illegal", "missing", "unresolved")) else 1
        rows_out.append((sev, g, where, _meaning(g)))

    for e in db.kv_get("card_errors", []) or []:
        rows_out.append((0, "unreadable card", html.Code(e["file"]),
                         e["error"] + " — treated as absent until fixed"))

    coll = db.kv_get("collection_issues", {}) or {}
    by_key: dict[str, list] = {}
    for ds, d in sorted(coll.items()):
        for k, v in d.items():
            if k != "first_malformed_byte" and v:
                by_key.setdefault(k, []).append(f"{ds} {human_count(v)}")
    for k, parts in by_key.items():
        rows_out.append((1, k, " · ".join(parts), _meaning(k)))

    fl = db.q("""SELECT f.fl, d.dataset, count(*),
                        (SELECT count(*) FROM documents t
                          WHERE t.dataset = d.dataset)
                 FROM documents d, UNNEST(d.flags) AS f(fl)
                 GROUP BY 1, 2 ORDER BY 1, 3 DESC""")
    # one row per kind of flag: "lang_mismatch" lists where (dataset ·
    # section or annotation set) instead of one near-identical row each
    by_flag: dict[str, list] = {}
    for flag, ds, n, tot in fl:
        kind, _, where = flag.partition(":")
        share = f" {100.0 * n / tot:.1f}%" if tot else ""
        by_flag.setdefault(kind, []).append(
            f"{ds}{' ' + where if where else ''} {human_count(n)}{share}")
    for flag, parts in by_flag.items():
        rows_out.append((2, flag, " · ".join(parts[:4])
                         + (f" +{len(parts) - 4}" if len(parts) > 4 else ""),
                         _meaning(flag)))

    toks = db.kv_get("tokenizers", {}) or {}
    for spec, info in sorted(toks.items()):
        if info.get("status") != "exact":
            rows_out.append((1, "approximate tokens", html.Code(spec),
                             (info.get("why") or "asset unavailable")
                             + " — counts and window positions are estimated"))

    if not rows_out:
        return html.Div([html.H3("Needs attention", className="section-title"),
                         ui.card(html.Div([ui.badge("all clear", "ok"), html.Span(
                             " every run is complete, valid and linked; every "
                             "collection and tokenizer is clean",
                             className="muted small")]))])
    rows_out.sort(key=lambda r: (r[0], str(r[1])))
    n_all = (db.q1("SELECT count(*) FROM runs") or [0])[0]
    trs = [[ui.tag_chip(g) if isinstance(g, str) else g, where,
            html.Span(meaning, className="muted small")]
           for _sev, g, where, meaning in rows_out]
    detail = None
    if runs:
        det = [[ds, idx.model(m).name, html.Code(rid[:12]), a or "—",
                html.Span([ui.tag_chip(t) for t in json.loads(tj)])]
               for ds, m, a, rid, tj in runs]
        detail = html.Details([
            html.Summary(f"Every flagged run ({len(runs)} of {n_all})"),
            ui.table(["Dataset", "Model", "Run", "Architecture", "Issue tags"],
                     det)], className="fold")
    return html.Div([
        html.H3("Needs attention", className="section-title"),
        ui.card([ui.table(["Issue", "Where", "What it means"], trs), detail])])


def _meaning(key: str) -> str:
    """What an issue means and what was done about it (one line)."""
    base = str(key).split(":", 1)[0]
    if base.startswith("illegal parameter"):
        return ("a run's parameter value is outside what its model card "
                "declares (or undeclared); the run is kept and flagged")
    what = str(key).split(":", 1)[1] if ":" in key else ""
    if base == "missing" and what == "run":
        return ("the run folder has no run_*.json: its parameters are unknown, "
                "so the model card's defaults are assumed")
    if base == "missing" and what == "dataset":
        return ("predictions for a dataset with no document collection; they "
                "enter no score")
    if base == "missing":
        return (f"no {what or 'matching'} card for this run's folder name; the "
                "run is kept, and what needs the card (costs, validation) is "
                "unknown")
    if base.startswith("lang_mismatch"):
        return ("the detected language of a section or annotation set "
                "differs from the declared one; the documents stay in, and "
                "Insights → data quality measures their effect")
    return ui.TAG_HELP.get(base, "")


def _backends_panel():
    """Which fast paths are live. A silent fallback here is the difference
    between minutes and hours, or between exact and approximate token
    counts — it must never be invisible."""
    from ..diag import backends
    from ..config import settings as _settings
    b = backends()
    rows = []
    for name, info in b.items():
        if name == "TeX":        # the probed engine is listed below instead
            continue
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
    # tokenizer *assets*, not just packages: an approximate token count is
    # exactly the silent degradation this panel exists to expose
    toks = db.kv_get("tokenizers", {}) or {}
    for spec, info in sorted(toks.items()):
        exact = info.get("status") == "exact"
        rows.append([html.Code(spec),
                     ui.badge("exact" if exact else "approximate",
                              "ok" if exact else "warn"),
                     "" if exact else (info.get("why") or "asset unavailable")
                     + " — token counts and window positions are estimated"])
    from ..export import tex_status
    tstat, teng = tex_status()
    rows.append(["TeX (probed)", ui.badge(teng or ("checking…" if tstat == "probing"
                                                     else "none"),
                                          "ok" if teng else "gray"),
                 "PGF typesetting" if teng else "PDF/PNG via Matplotlib"])
    warn = [n for n, i in b.items() if i["level"] == "warn" and n != "TeX"]
    n_exact = sum(1 for i in toks.values() if i.get("status") == "exact")
    summary = (f"{d['workers']} scan workers · DuckDB threads {d['db_threads']} · "
               f"tokenizers {n_exact}/{len(toks)} exact · TeX "
               f"{teng or 'none'}"
               + (f" · {len(warn)} fallback(s)" if warn else ""))
    return html.Details([
        html.Summary(["Engine", html.Span(" · " + summary, className="stats-sum")]),
        ui.kpi_row(tiles),
        ui.card(ui.table(["Component", "Active", "Note"], rows)),
    ], className="more-opts engine-fold", open=bool(warn))


def _panel_sig(snap: dict) -> str:
    """What the progress panel shows, minus the clock (elapsed/ETA change on
    every tick): an unchanged panel is not re-sent."""
    s = {k: v for k, v in snap.items() if k not in ("elapsed_s", "eta_s")}
    s["eta_bucket"] = round((snap.get("eta_s") or 0) / 5)
    s["elapsed_bucket"] = int(snap.get("elapsed_s") or 0)
    return stable_hash(s)


def register(app):
    from ..appfactory import _sidebar_status

    @app.callback(
        Output("scan-progress", "children"),
        Output("btn-scan", "disabled"),
        Output("btn-rescan", "disabled"),
        Output("btn-cancel", "style"),
        Output("scan-headline", "children"),
        Output("catalog-version", "data"),
        Output("scan-poll", "disabled"),
        Output("sidebar-scan", "children"),
        Output("scan-panel-sig", "data"),
        Input("scan-poll", "n_intervals"),
        Input("btn-scan", "n_clicks"),
        Input("btn-rescan", "n_clicks"),
        Input("btn-cancel", "n_clicks"),
        Input("btn-retry-tok", "n_clicks"),
        State("catalog-version", "data"),
        State("scan-panel-sig", "data"),
        prevent_initial_call=True)
    def poll(_n, s_clicks, r_clicks, c_clicks, t_clicks, known_version, last_sig):
        """The only periodic callback. It runs while a scan runs (the
        interval is disabled otherwise) and returns only what changed."""
        trig = ctx.triggered_id
        if trig == "btn-scan":
            scanner.start_scan(False)
        elif trig == "btn-rescan":
            scanner.start_scan(True)
        elif trig == "btn-cancel":
            scanner.request_cancel()
        elif trig == "btn-retry-tok":
            from ..textproc import forget_tokenizers
            forget_tokenizers(settings().tokenizer_cache)
        snap = scanner.STATE.snapshot()
        running = snap["running"]
        v = db.scan_version()
        sig = _panel_sig(snap)
        panel = _progress_panel(snap) if sig != last_sig else no_update
        headline = (f"catalog version {v}" if (v and not running) else "")
        if trig == "btn-retry-tok":
            headline = "tokenizer downloads will be retried on the next scan"
        return (panel, running, running,
                {"display": "inline-flex"} if running else {"display": "none"},
                headline, v if v != known_version else no_update,
                not running, _sidebar_status(),
                sig if sig != last_sig else no_update)

    @app.callback(
        Output("home-inventory", "children"),
        Output("home-backends", "children"),
        Output("home-issues", "children"),
        State("vis-home", "data"), Input("shown-home", "data"),
        Input("catalog-version", "data"),
        prevent_initial_call=True)
    def static(visible, _shown, _v):
        """Catalog, engine and issues: on show and after a scan publishes —
        never on a timer."""
        if not visible:
            raise PreventUpdate
        has = db.q1("SELECT 1 FROM files LIMIT 1")
        return (_inventory(), _backends_panel(), _issues() if has else None)
