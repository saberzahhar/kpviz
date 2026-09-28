"""Home — scan control, live progress with ETA, inventory and issues."""
from __future__ import annotations

import json

from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..config import settings
from ..naming import arch_label, group_key, natural_key, run_labels, run_rows
from ..util import human_bytes, human_count, human_duration, stable_hash


def layout():
    root = settings().data_root
    return html.Div([
        html.H1("Overview", className="page-title"),
        html.P(["KPViz reads the collections, model cards and runs under ",
                html.Code(root.name or str(root), title=str(root)),
                ", scores every run the same way and turns the results into "
                "figures for a paper. A scan re-derives only what changed."],
               className="page-desc"),
        html.Div([
            # "Explore results" is the page's one primary action once there
            # is a catalog; before that, scanning is (see static())
            dcc.Link(["Explore results", html.Span(" →", **{"aria-hidden": "true"})],
                     href="/insights#rq4", id="home-explore",
                     className="btn primary", style={"display": "none"}),
            html.Button("Scan for changes", id="btn-scan",
                        className="btn primary", n_clicks=0,
                        title="Re-derive the files that were added, changed "
                              "or removed since the last scan"),
            html.Button("Cancel", id="btn-cancel", className="btn",
                        n_clicks=0, style={"display": "none"}),
            html.Details([
                html.Summary("Maintenance"),
                html.Div([
                    html.Button("Full re-scan", id="btn-rescan",
                                className="btn small", n_clicks=0,
                                title="Re-hash every file and re-derive every "
                                      "collection and run"),
                    html.Button("Retry tokenizer downloads", id="btn-retry-tok",
                                className="btn small", n_clicks=0,
                                title="Forget which tokenizer assets were "
                                      "unavailable and try again on the next "
                                      "scan"),
                ], className="maint-body"),
            ], className="maint"),
            html.Span(id="scan-headline", className="muted small",
                      role="status", **{"aria-live": "polite"}),
        ], className="flex action-row"),
        html.Div(id="scan-progress", children=_progress_panel(
            scanner.STATE.snapshot())),
        html.Div(id="home-inventory"),
        html.Div(id="home-coverage"),
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
    log = None
    if snap["log"]:
        log = html.Details([
            html.Summary(f"Scan log ({len(snap['log'])} lines)"),
            html.Div("\n".join(snap["log"]), className="scan-log", role="log")],
            className="fold", open=bool(snap.get("error")))
    return ui.card([
        html.Div(status_line, style={"marginBottom": "8px"}),
        html.Div(html.Div(className="progress-fill",
                          style={"width": f"{100 * frac:.1f}%"}),
                 className="progress-track", role="progressbar",
                 style={"marginBottom": "10px"},
                 **{"aria-valuenow": round(100 * frac), "aria-valuemin": 0,
                    "aria-valuemax": 100, "aria-label": "Scan progress"}),
        chips,
        html.Div(rows, style={"margin": "8px 0 10px"}),
        log,
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
    return html.Div([html.H2("Catalog", className="section-title"),
                     ui.kpi_row(tiles)])


def _coverage():
    """What was evaluated, at a glance: models down, datasets across; a cell
    holds the number of runs, tinted by how much of the evaluation split
    they cover, marked when one of them carries an issue."""
    rows = db.q("""SELECT model, dataset, count(*), min(coverage),
                          max(CASE WHEN tags IS NOT NULL AND tags <> '[]'
                                   THEN 1 ELSE 0 END)
                   FROM runs GROUP BY 1, 2""")
    if not rows:
        return None
    idx = scanner.cards()
    cell = {(m, d): (n, cov, bool(fl)) for m, d, n, cov, fl in rows}
    models = sorted({r[0] for r in rows}, key=lambda m: natural_key(idx.model(m).name))
    datasets = sorted({r[1] for r in rows}, key=natural_key)

    def td(m, d):
        hit = cell.get((m, d))
        if not hit:
            return html.Td("·", className="cov-cell cov-none",
                           title=f"{idx.model(m).name} was not run on {d}")
        n, cov, flagged = hit
        share = 1.0 if cov is None else float(cov)
        step = 4 if share >= 0.999 else 3 if share >= 0.75 else 2 if share >= 0.25 else 1
        what = (f"{n} run{'s' if n > 1 else ''} · "
                + ("coverage unknown (no evaluation split)" if cov is None
                   else f"{100 * share:.0f}% of the evaluation split"
                        + (" (least-covered run)" if n > 1 else ""))
                + (" · has issues (see Needs attention)" if flagged else ""))
        return html.Td([str(n), html.Span("!", className="cov-flag",
                                          **{"aria-hidden": "true"})
                        if flagged else None],
                       className=f"cov-cell cov-{step}", title=what,
                       **{"aria-label": what})
    head = html.Thead(html.Tr([html.Th("Model")] +
                              [html.Th(d, scope="col") for d in datasets]))
    body = html.Tbody([html.Tr([html.Th(idx.model(m).name, scope="row")]
                               + [td(m, d) for d in datasets])
                       for m in models])
    legend = html.Div([
        html.Span("runs per model and dataset; shade = share of the evaluation "
                  "split predicted:", className="muted small"),
        *[html.Span([html.Span(className=f"cov-swatch cov-{k}"), lab],
                    className="cov-key")
          for k, lab in ((4, "all"), (3, "≥ 75 %"), (2, "≥ 25 %"), (1, "< 25 %"))],
        html.Span([html.Span("!", className="cov-flag"), " issue"],
                  className="cov-key"),
    ], className="cov-legend")
    return html.Div([
        html.H2("Coverage", className="section-title"),
        ui.card([html.Div(html.Table([head, body], className="cov-table"),
                          className="table-wrap"), legend])])


# ---- needs attention ------------------------------------------------------
# One row per *kind* of issue, whatever the number of runs, parameters or
# tokenizers it touches; the meaning is printed once, the severity has an
# icon and a word (never colour alone).
SEVERITY = [("✕", "error", "results are missing or unusable"),
            ("!", "warning", "results are kept, with a caveat"),
            ("i", "note", "for information")]

_ERRORS = {"unreadable card", "unreadable:run card", "missing:dataset"}
_WARNINGS = ("illegal parameter", "missing", "unresolved_ids", "incomplete",
             "approximate tokens", "malformed_lines", "missing_id",
             "duplicate_docs", "duplicate_doc_ids")


def severity(kind: str) -> int:
    """0 error, 1 warning, 2 note."""
    if kind in _ERRORS:
        return 0
    return 1 if str(kind).split(":", 1)[0] in _WARNINGS else 2


def _tag_group(tag: str) -> tuple[str, str]:
    """(kind, value). "missing:*" and "unreadable:*" keep their value in the
    kind (a missing model and a missing run mean different things); every
    other tag is grouped by its key, so six illegal parameters are one row
    listing the six names, and counts ("unscored:5") become a range."""
    key, _, val = str(tag).partition(":")
    if key in ("missing", "unreadable"):
        return str(tag), ""
    return key, val


def _few(names, n: int = 3) -> str:
    names = sorted(set(names), key=natural_key)
    return ", ".join(names[:n]) + (f" +{len(names) - n}" if len(names) > n else "")


# what the number after a count tag counts ("unscored:20" -> 20 documents)
_COUNTS = {"unscored": "documents", "unresolved_ids": "unknown ids",
           "duplicate_docs": "repeated documents", "multi_split": "splits",
           "malformed_lines": "lines", "missing_id": "lines",
           "incomplete": "coverage"}


def _span(vals: list[str], kind: str = "") -> str:
    """Numeric values as a range with what they count ("3–20 documents",
    "coverage 20–54%"), names as a short list."""
    if not vals:
        return ""
    try:
        nums = sorted(float(v.rstrip("%")) for v in vals)
    except ValueError:
        return _few(vals, 6)
    unit = "%" if vals[0].endswith("%") else ""
    lo, hi = nums[0], nums[-1]
    rng = f"{lo:g}{unit}" if lo == hi else f"{lo:g}–{hi:g}{unit}"
    what = _COUNTS.get(kind, "")
    if not what:
        return rng
    return f"{what} {rng}" if unit else f"{rng} {what}"


def issue_rows() -> list[tuple]:
    """[(severity, kind, where, meaning)] — one per kind of issue."""
    idx = scanner.cards()
    out = []
    runs = db.q("""SELECT dataset, model, arch, run_id, tags FROM runs
                   WHERE tags IS NOT NULL AND tags <> '[]'""")
    groups: dict[str, list] = {}
    for ds, m, a, rid, tj in runs:
        for t in json.loads(tj):
            kind, val = _tag_group(t)
            groups.setdefault(kind, []).append(((ds, m, a, rid), ds,
                                                idx.model(m).name, val))
    for kind, hits in groups.items():
        n_runs = len({h[0] for h in hits})
        span = _span([h[3] for h in hits if h[3]], kind)
        where = (f"{n_runs} run{'s' if n_runs > 1 else ''}"
                 + (f" · {span}" if span else "")
                 + f" · {_few(h[2] for h in hits)} · on {_few(h[1] for h in hits)}")
        out.append((severity(kind), kind, where, meaning(kind)))

    for e in db.kv_get("card_errors", []) or []:
        out.append((0, "unreadable card", html.Code(e["file"]),
                    e["error"] + " — treated as absent until fixed"))

    coll = db.kv_get("collection_issues", {}) or {}
    by_key: dict[str, list] = {}
    for ds, d in sorted(coll.items()):
        for k, v in d.items():
            if k != "first_malformed_byte" and v:
                by_key.setdefault(k, []).append(f"{ds} {human_count(v)}")
    for k, parts in by_key.items():
        out.append((severity(k), k, " · ".join(parts), meaning(k)))

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
        out.append((severity(flag), flag, " · ".join(parts[:4])
                    + (f" +{len(parts) - 4}" if len(parts) > 4 else ""),
                    meaning(flag)))

    toks = db.kv_get("tokenizers", {}) or {}
    approx = {spec: info for spec, info in toks.items()
              if info.get("status") != "exact"}
    if approx:
        whys = sorted({info.get("why") or "asset unavailable"
                       for info in approx.values()})
        out.append((severity("approximate tokens"), "approximate tokens",
                    html.Span([html.Code(s) for s in sorted(approx)],
                              className="code-list"),
                    "token counts and window positions are estimated ("
                    + "; ".join(whys[:2]) + (" …" if len(whys) > 2 else "")
                    + "); Overview → Maintenance → Retry tokenizer downloads"))
    out.sort(key=lambda r: (r[0], str(r[1])))
    return out


def _sev_cell(sev: int):
    ico, word, _ = SEVERITY[sev]
    return html.Span([html.Span(ico, className="sev-ico", **{"aria-hidden": "true"}),
                      word], className=f"sev sev-{sev}")


def _issues():
    """Everything that needs attention, one row per kind of issue: how bad,
    what, how widespread, and what it means (what KPViz did about it). The
    per-run list is one click away."""
    rows = issue_rows()
    if not rows:
        return html.Div([html.H2("Needs attention", className="section-title"),
                         ui.card(html.Div([ui.badge("no issues", "ok"), html.Span(
                             " none detected by the current checks: every run "
                             "is complete, valid and linked; every collection "
                             "and tokenizer is clean", className="muted small")]))])
    n_sev = [sum(1 for r in rows if r[0] == k) for k in range(3)]
    counts = html.Div([
        html.Span([html.Span(SEVERITY[k][0], className="sev-ico",
                             **{"aria-hidden": "true"}),
                   f"{n} {SEVERITY[k][1]}{'s' if n > 1 else ''}"],
                  className=f"sev sev-count sev-{k}", title=SEVERITY[k][2])
        for k, n in enumerate(n_sev) if n], className="sev-counts")
    trs = [[_sev_cell(sev), ui.tag_chip(kind) if isinstance(kind, str) else kind,
            where, html.Span(mean, className="muted small")]
           for sev, kind, where, mean in rows]
    runs = db.q("""SELECT dataset, model, arch, run_id, tags FROM runs
                   WHERE tags IS NOT NULL AND tags <> '[]'
                   ORDER BY dataset, model, run_id""")
    detail = None
    if runs:
        idx = scanner.cards()
        labels = run_labels(idx, run_rows())
        n_all = (db.q1("SELECT count(*) FROM runs") or [0])[0]
        det = [[ds, html.Span(labels.get(group_key(m, a, rid), idx.model(m).name),
                              title=f"run {rid}"),
                arch_label(idx, a),
                html.Span([ui.tag_chip(t) for t in json.loads(tj)], className="chips")]
               for ds, m, a, rid, tj in runs]
        detail = html.Details([
            html.Summary(f"Every run with an issue ({len(runs)} of {n_all})"),
            ui.table(["Dataset", "Run", "Architecture", "Issues"], det)],
            className="fold")
    return html.Div([
        html.H2("Needs attention", className="section-title"),
        ui.card([counts,
                 ui.table(["Severity", "Issue", "Where", "What it means"], trs,
                          nowrap_cols={0, 1}),
                 detail])])


_MEANINGS = {
    "approximate tokens": "a tokenizer's assets are unavailable: token counts "
                          "and window positions are estimated",
    "unreadable card": "a card that cannot be parsed is treated as absent "
                       "until fixed",
    "illegal parameter": "a run's parameter value is outside what its model "
                         "card declares (or is undeclared); the run is kept "
                         "and flagged",
    "missing:run": "the run folder has no run_*.json: its parameters are "
                   "unknown, so the model card's defaults are assumed",
    "missing:dataset": "predictions for a dataset with no document "
                       "collection; they enter no score",
    "unreadable:run card": "the run_*.json cannot be parsed: its parameters "
                           "are unknown, so the model card's defaults are "
                           "assumed",
    "lang_mismatch": "the detected language of a section or annotation set "
                     "differs from the declared one; the documents stay in, "
                     "and Insights → Data quality measures their effect",
    "missing_section": "a section the dataset card declares is absent from "
                       "these documents; they are scored on the sections "
                       "they have, and Insights → Data quality measures the "
                       "effect",
}


def meaning(key: str) -> str:
    """What an issue means and what was done about it (one line)."""
    key = str(key)
    if key in _MEANINGS:
        return _MEANINGS[key]
    base, _, what = key.partition(":")
    if base in _MEANINGS:
        return _MEANINGS[base]
    if base == "missing":
        return (f"no {what or 'matching'} card for this run's folder name; the "
                "run is kept, and what needs the card (costs, validation) is "
                "unknown")
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
        headline = ""        # the catalog version is in the sidebar
        if trig == "btn-retry-tok":
            headline = "tokenizer downloads will be retried on the next scan"
        return (panel, running, running,
                {"display": "inline-flex"} if running else {"display": "none"},
                headline, v if v != known_version else no_update,
                not running, _sidebar_status(),
                sig if sig != last_sig else no_update)

    @app.callback(
        Output("home-inventory", "children"),
        Output("home-coverage", "children"),
        Output("home-backends", "children"),
        Output("home-issues", "children"),
        Output("home-explore", "style"),
        Output("btn-scan", "className"),
        State("vis-home", "data"), Input("shown-home", "data"),
        Input("catalog-version", "data"),
        prevent_initial_call=True)
    def static(visible, _shown, _v):
        """Catalog, coverage, engine and issues: on show and after a scan
        publishes — never on a timer. With a catalog, exploring the results
        is the page's primary action; without one, scanning is."""
        if not visible:
            raise PreventUpdate
        has = bool(db.q1("SELECT 1 FROM runs LIMIT 1"))
        return (_inventory(), _coverage() if has else None, _backends_panel(),
                _issues() if db.q1("SELECT 1 FROM files LIMIT 1") else None,
                {} if has else {"display": "none"},
                "btn" if has else "btn primary")
