"""Datasets explorer — statistics, annotations, quality, document browser.

Distribution charts read the tiny `gold_agg` aggregates (every split);
per-keyphrase panels (POS, browser drill-down) read gold instance rows
(eval splits under the default gold scope) joined against the global
keyphrase cache.
"""
from __future__ import annotations

import json

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from .. import db, scanner, ui
from ..figures import MUTED, PALETTE, PATTERNS, to_plotly
from ..naming import (natural_key, order_splits, slot_color, split_color,
                      split_rank, tokenizer_label)
from ..util import fmt_num, human_count, mean_sd

# P green · R yellow · M orange · U red
PRMU_COLORS = {"P": "#008300", "R": "#eda100", "M": "#eb6834", "U": "#e34948"}
PRMU_NAMES = {"P": "Present", "R": "Reordered", "M": "Mixed", "U": "Unseen"}
WORDS_TOK = "words"
COMBINED = "@combined"
# splits carry their own fixed hue everywhere in the app (naming.SPLIT_COLORS);
# annotation sets are told apart by the label, never by a second colour scale,
# so nothing on this page needs transparency to be read.
POS_TOP_N = 4          # top-4 patterns, everything else folded into "Other"
PANEL_H = 330          # every distribution panel gets the same box


def _datasets():
    return [r[0] for r in db.q("SELECT DISTINCT dataset FROM documents ORDER BY 1")]


def _annotators(ds: str) -> list[str]:
    """Real annotation sets, never the synthetic union."""
    return [r[0] for r in db.q(
        "SELECT DISTINCT ann_key FROM gold_agg WHERE dataset=? AND ann_key<>? "
        "ORDER BY 1", ds, COMBINED)]


def _splits(ds: str) -> list[str]:
    return [r[0] for r in db.q(
        "SELECT DISTINCT coalesce(split,'?') FROM documents WHERE dataset=? "
        "ORDER BY 1", ds)]


def _group_series(data: dict[tuple[str, str], dict], cats: list,
                  anns: list[str], splits: list[str], as_pct: bool = True,
                  horizontal: bool = False):
    """One series per (annotator, split), coloured by split.

    Splits keep the app-wide hue (training teal · validation violet · testing
    blue) and always appear in pipeline order; the annotation set is named in
    the series label. Values are shares within their own group, so every panel
    sits on the same 0–100 axis whatever the collection size."""
    series = []
    for a in anns:
        for s in order_splits(splits):
            counts = data.get((a, s))
            if not counts:
                continue
            tot = sum(counts.values()) or 1
            ys, hover = [], []
            for c in cats:
                n = counts.get(c, 0)
                ys.append(100.0 * n / tot if as_pct else n)
                hover.append(f"{a} · {s}<br>{c}: {n} "
                             f"({100.0 * n / tot:.1f}% of {tot})")
            name = f"{a} · {s}" if len(anns) > 1 else str(s)
            # hue is the split, everywhere in the app; when several annotation
            # sets share a panel the *pattern* separates them, so the colour
            # never has to mean two things at once
            pat = PATTERNS[anns.index(a) % len(PATTERNS)] if len(anns) > 1 else ""
            ser = {"name": name, "color": split_color(s), "hover": hover,
                   "pattern": pat}
            if horizontal:
                ser.update(y=[str(c) for c in cats], x=ys)
            else:
                ser.update(x=[str(c) for c in cats], y=ys)
            series.append(ser)
    return series


def layout():
    return html.Div([
        html.H2("Datasets", className="page-title"),
        html.P("Collection statistics derived at scan time — split sizes, "
               "length distributions against model context windows, PRMU "
               "portions, POS patterns and per-document quality flags. The "
               "browser reads documents straight from your files through the "
               "byte-offset index.", className="page-desc"),
        ui.filter_row([
            ui.control("Dataset", dcc.Dropdown(
                id="ds-pick", options=[], clearable=False,
                placeholder="scan first…", className="dash-dropdown"), 210),
            ui.control("Annotation sets", dcc.Dropdown(
                id="ds-ann", clearable=False, className="dash-dropdown"), 210),
            ui.control("Split", dcc.Dropdown(
                id="ds-split", clearable=False, className="dash-dropdown"), 150),
            ui.control("Tokenizer", dcc.Dropdown(
                id="ds-tok", clearable=False, className="dash-dropdown"), 260),
        ]),
        html.Div(id="ds-body"),
        html.Div(id="ds-doc-view"),
    ], className="page")


# ---------------------------------------------------------------------------

def _hist_bins(rows, nbins=24):
    vals = [r[0] for r in rows if r[0] is not None]
    if not vals:
        return [], {}
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1
    w = (hi - lo) / nbins
    groups: dict[str, list[int]] = {}
    for v, g in rows:
        if v is None:
            continue
        b = min(nbins - 1, int((v - lo) / w))
        groups.setdefault(str(g), [0] * nbins)[b] += 1
    centers = [round(lo + (i + 0.5) * w, 1) for i in range(nbins)]
    return centers, groups


def _body(ds: str, ann: str, split: str | None, tok: str | None):
    idx = scanner.cards()
    card = idx.dataset(ds)
    use_split = split and split != "(all)"
    all_anns = _annotators(ds)
    # "(each)" keeps every annotation set separate — the synthetic @combined
    # union is never plotted, so no panel silently merges two annotators
    anns = all_anns if (not ann or ann == "(each)") else [ann]
    splits = [split] if use_split else _splits(ds)
    ann_sql = ",".join("?" * len(anns)) or "NULL"
    agg_where = (f"dataset=? AND ann_key IN ({ann_sql})"
                 + (" AND split=?" if use_split else ""))
    agg_args = [ds, *anns] + ([split] if use_split else [])

    split_rows = db.q("SELECT coalesce(split,'?'), count(*), avg(n_words) "
                      "FROM documents WHERE dataset=? GROUP BY 1 ORDER BY 1", ds)
    n_docs_total = sum(r[1] for r in split_rows)
    n_docs_sel = (db.q1("SELECT count(*) FROM documents WHERE dataset=? AND "
                        "coalesce(split,'?')=?", ds, split)[0]
                  if use_split else n_docs_total)

    per_ann = {r[0]: r[1:] for r in db.q(
        f"""SELECT ann_key, sum(n), sum(words_sum),
                   sum(CASE WHEN prmu='P' THEN n ELSE 0 END)
            FROM gold_agg WHERE {agg_where} GROUP BY 1""", *agg_args)}
    n_gold = sum(v[0] or 0 for v in per_ann.values())
    words_sum = sum(v[1] or 0 for v in per_ann.values())
    n_present = sum(v[2] or 0 for v in per_ann.values())

    def _by_ann(fmt):
        return " · ".join(f"{a} {fmt(per_ann[a])}" for a in anns if a in per_ann)

    header = ui.card([
        html.Div([html.B(card.description or ds)], style={"marginBottom": "9px"}),
        ui.meta_row([
            ui.meta_chip("domain", card.domain) if card.domain else None,
            ui.meta_chip("sub-domain", card.subdomain) if card.subdomain else None,
            *[ui.meta_chip("lang", l) for l in card.languages],
            *[ui.meta_chip("section", sec) for sec in card.sections],
            *[ui.meta_chip("annotation", a) for a in card.annotations],
            *[ui.meta_chip("metadata", m) for m in card.metadata_spec
              if m != "split"],
        ]),
    ])

    # Three of the four tiles are per-document facts, so they are computed
    # from gold *instances*: the union of the selected annotation sets (never
    # their sum — one phrase annotated by two people is one keyphrase), and a
    # mean ± sd across documents rather than a bare ratio. The basis is named
    # in the sub-line because instance rows follow --gold-scope.
    doc_where = "d.dataset=?" + (" AND coalesce(d.split,'?')=?" if use_split else "")
    doc_args = [ds] + ([split] if use_split else [])
    per_doc = db.q(
        f"""SELECT g.doc_id, count(DISTINCT list_aggregate(g.stems, 'string_agg', ' ')),
                   count(DISTINCT list_aggregate(g.stems, 'string_agg', ' '))
                       FILTER (WHERE g.prmu = 'P')
              FROM gold g JOIN documents d
                ON d.dataset = g.dataset AND d.doc_id = g.doc_id
             WHERE g.dataset=? AND g.ann_key IN ({ann_sql}) AND {doc_where}
             GROUP BY 1""", ds, *anns, *doc_args)
    n_kp_doc = [r[1] for r in per_doc]
    p_share = [r[2] / r[1] for r in per_doc if r[1]]
    n_unique = (db.q1(
        f"""SELECT count(DISTINCT list_aggregate(g.stems, 'string_agg', ' '))
              FROM gold g JOIN documents d
              ON d.dataset = g.dataset AND d.doc_id = g.doc_id
             WHERE g.dataset=? AND g.ann_key IN ({ann_sql}) AND {doc_where}""",
        ds, *anns, *doc_args) or [0])[0]
    basis = f"over {human_count(len(per_doc))} documents with stored gold"

    kpis = ui.kpi_row([
        ui.stat_tile("Documents", human_count(n_docs_sel),
                     " · ".join(f"{s}: {human_count(n)}" for s, n in
                                sorted(((r[0], r[1]) for r in split_rows),
                                       key=lambda t: split_rank(t[0])))),
        ui.stat_tile("Unique keyphrases", human_count(n_unique),
                     ("union of " + ", ".join(anns)) if len(anns) > 1
                     else "distinct surface forms"),
        ui.stat_tile("Keyphrases per document", mean_sd(n_kp_doc, 1), basis),
        ui.stat_tile("Present (P) per document",
                     mean_sd(p_share, 1, pct=True),
                     "share of a document's gold occurring in order"),
    ])

    charts = []
    # ---- document length panel — "words" (default) or a model tokenizer ----
    tok = tok or WORDS_TOK
    if tok == WORDS_TOK:
        rows = db.q("SELECT n_words, coalesce(split,'?') FROM documents WHERE dataset=?", ds)
        xlabel, title, uv = "document length (words)", "Document length", []
    else:
        rows = db.q("""SELECT t.n_tokens, coalesce(d.split,'?')
                       FROM doc_tokens t JOIN documents d USING (dataset, doc_id)
                       WHERE t.dataset=? AND t.tokenizer=?""", ds, tok)
        approx = db.q1("SELECT bool_or(approx) FROM doc_tokens WHERE dataset=? AND tokenizer=?",
                       ds, tok)
        xlabel = (f"document length ({tokenizer_label(tok)}"
                  + (" tokens, approximate)" if approx and approx[0]
                     else " tokens)"))
        title = "Document length — against the context limits of the models that ran here"
        vlines = []
        for m, rj in db.q("SELECT DISTINCT model, resolved FROM runs WHERE dataset=?", ds):
            mc = idx.model(m)
            for p in mc.context_params():
                if p.tokenizer != tok or "input" not in p.name:
                    continue
                res = json.loads(rj or "{}").get(p.name) or {}
                v = res.get("value") or p.default
                if v:
                    vlines.append({"x": v, "label": f"{mc.name} · {v}",
                                   "color": MUTED, "dash": True})
        seen, uv = set(), []
        for v in vlines:
            if v["x"] not in seen:
                seen.add(v["x"])
                uv.append(v)
    centers, groups = _hist_bins(rows)
    if centers:
        span = max(centers)
        inside = [v for v in uv if v["x"] <= span * 1.35][:4]
        beyond = [v for v in uv if v["x"] > span * 1.35]
        # each split is normalised to its own 100%: collections are wildly
        # unbalanced (1.3 M training vs 100 k testing), and the question here
        # is whether the *shapes* differ, not which split is bigger
        series = []
        for sp in order_splits(groups):
            ys = groups[sp]
            tot = sum(ys) or 1
            series.append({
                "name": sp, "x": centers,
                "y": [100.0 * y / tot for y in ys],
                "color": split_color(sp),
                "hover": [f"{sp}<br>~{c:g} · {y} docs ({100.0 * y / tot:.1f}%)"
                          for c, y in zip(centers, ys)]})
        spec = {"kind": "bar", "xlabel": xlabel,
                "ylabel": "% of the split's documents", "barmode": "group",
                "series": series, "vlines": inside}
        note = (html.Div("beyond the axis: " + " · ".join(
            v["label"] for v in beyond), className="muted small")
            if beyond else None)
        charts.append(ui.card([ui.graph("ds-g-len", to_plotly(spec), PANEL_H),
                               note], title=title))

    # ---- PRMU pies: one row per annotator, one pie per split ---------------
    rows = db.q(f"""SELECT ann_key, coalesce(split,'?'), prmu, sum(n)
                    FROM gold_agg WHERE {agg_where} GROUP BY 1,2,3""", *agg_args)
    if rows:
        cell: dict[tuple[str, str], dict] = {}
        for a, sp, pr, n in rows:
            cell.setdefault((a, sp), {})[pr or "U"] = n
        rows_used = [a for a in anns if any((a, sp) in cell for sp in splits)]
        cols_used = [sp for sp in order_splits(splits)
                     if any((a, sp) in cell for a in rows_used)]
        panels, order = [], ["P", "R", "M", "U"]
        for a in rows_used:
            for sp in cols_used:
                c = cell.get((a, sp), {})
                tot = sum(c.values())
                panels.append({
                    "title": (f"{sp} · {human_count(tot)}" if tot
                              else f"{sp} · none"),
                    "labels": [f"{pr} — {PRMU_NAMES[pr]}" for pr in order],
                    "values": [c.get(pr, 0) for pr in order],
                    "colors": [PRMU_COLORS[pr] for pr in order]})
        spec = {"kind": "pie_grid", "panels": panels, "ncols": len(cols_used),
                "row_titles": rows_used, "size": "2col"}
        charts.append(ui.card(
            [ui.graph("ds-g-prmu", to_plotly(spec),
                      max(250, 200 * len(rows_used) + 76))],
            title="Keyphrase PRMU distribution"))

    # ---- keyphrase length + POS tags, split-coloured ------------------------
    rows = db.q(f"""SELECT ann_key, coalesce(split,'?'), n_words_b, sum(n)
                    FROM gold_agg WHERE {agg_where} GROUP BY 1,2,3""", *agg_args)
    c1 = None
    if rows:
        data: dict[tuple[str, str], dict] = {}
        for a, sp, b, n in rows:
            key = "6+" if b >= 6 else str(b)
            d = data.setdefault((a, sp), {})
            d[key] = d.get(key, 0) + n
        cats = sorted({k for d in data.values() for k in d}, key=natural_key)
        # keyphrase length lives in spaCy word tokens; the tokenizer selector
        # drives the *document* length panel, so name the unit honestly
        spec = {"kind": "bar", "barmode": "group",
                "xlabel": f"length ({WORDS_TOK})",
                "ylabel": "% of the group's gold", "yrange": [0, 100],
                "series": _group_series(data, cats, anns, splits)}
        c1 = ui.card([ui.graph("ds-g-kplen", to_plotly(spec), PANEL_H)],
                     title="Keyphrase length", style={"minWidth": 0})
    rows = db.q(f"""SELECT g.ann_key, coalesce(d.split,'?'), k.pos, count(*)
                    FROM gold g
                    JOIN documents d USING (dataset, doc_id)
                    JOIN keyphrases k ON k.kp = g.display
                    WHERE g.dataset=? AND g.ann_key IN ({ann_sql})
                      {'AND d.split=?' if use_split else ''}
                      AND k.pos IS NOT NULL
                    GROUP BY 1,2,3""", *agg_args)
    if rows:
        data = {}
        totals: dict[str, int] = {}
        for a, sp, pos, n in rows:
            data.setdefault((a, sp), {})[pos] = n
            totals[pos] = totals.get(pos, 0) + n
        # one shared set of categories across every group: the top-4 patterns
        # overall plus "Other", so the same row means the same thing whether
        # you read it on training or on testing
        top = [pos for pos, _ in sorted(totals.items(),
                                        key=lambda kv: (-kv[1], kv[0]))][:POS_TOP_N]
        cats = top + (["Other"] if len(totals) > len(top) else [])
        folded: dict[tuple[str, str], dict] = {}
        for key, counts in data.items():
            d = folded.setdefault(key, {})
            for pos, n in counts.items():
                d.setdefault(pos if pos in top else "Other", 0)
                d[pos if pos in top else "Other"] += n
        spec = {"kind": "bar", "barmode": "group", "orientation": "h",
                "xlabel": "% of the group's tagged gold",
                "series": _group_series(folded, list(reversed(cats)), anns,
                                        splits, horizontal=True)}
        c2 = ui.card([ui.graph("ds-g-pos", to_plotly(spec), PANEL_H)],
                     title="Keyphrase POS tags", style={"minWidth": 0})
    else:
        c2 = ui.card(html.Div("No POS-tagged keyphrases here yet — install the "
                              "spaCy model for this language and re-scan, or "
                              "widen --gold-scope.", className="muted small"),
                     title="Keyphrase POS tags", style={"minWidth": 0})
    grid1 = html.Div([c for c in (c1, c2) if c], className="grid-2")

    browser = ui.card([
        html.Div([
            dcc.Input(id="ds-search", type="text", debounce=True,
                      placeholder="search document ids…",
                      style={"border": "1px solid var(--border)",
                             "borderRadius": "8px", "padding": "7px 11px",
                             "fontSize": "13px", "width": "260px"}),
            dcc.Dropdown(id="ds-flagged", multi=True, options=[],
                         placeholder="data quality flags (all documents)",
                         className="dash-dropdown",
                         style={"minWidth": "420px", "flex": "1"}),
        ], className="flex", style={"marginBottom": "8px"}),
        html.Div(id="ds-doc-list"),
    ], title="Document browser")

    return html.Div([header, kpis, *charts, grid1, browser])


# ---------------------------------------------------------------------------

def register(app):
    @app.callback(Output("ds-pick", "options"), Output("ds-pick", "value"),
                  Input("catalog-version", "data"), State("ds-pick", "value"))
    def refresh_datasets(_v, current):
        ds = _datasets()
        value = current if current in ds else (ds[0] if ds else None)
        return ds, value

    @app.callback(Output("ds-ann", "options"), Output("ds-ann", "value"),
                  Output("ds-split", "options"), Output("ds-split", "value"),
                  Output("ds-tok", "options"), Output("ds-tok", "value"),
                  Input("ds-pick", "value"))
    def set_options(ds):
        if not ds:
            return [], None, [], None, [], None
        # "(each)" is the default: every annotation set gets its own row/series
        anns = [{"label": "(each, kept separate)", "value": "(each)"}] + [
            {"label": a, "value": a} for a in _annotators(ds)]
        splits = ["(all)"] + _splits(ds)
        toks = [WORDS_TOK] + [r[0] for r in db.q(
            "SELECT DISTINCT tokenizer FROM doc_tokens WHERE dataset=? ORDER BY 1", ds)]
        return (anns, "(each)", splits, "(all)", toks, WORDS_TOK)

    @app.callback(Output("ds-body", "children"),
                  Input("ds-pick", "value"), Input("ds-ann", "value"),
                  Input("ds-split", "value"), Input("ds-tok", "value"))
    def body(ds, ann, split, tok):
        if not ds:
            return ui.empty_state("No datasets in the catalog — run a scan on "
                                  "the Overview page.")
        if not _annotators(ds):
            return ui.empty_state("No annotations derived for this dataset yet.")
        return _body(ds, ann, split, tok)

    @app.callback(Output("ds-flagged", "options"), Input("ds-pick", "value"))
    def flag_options(ds):
        if not ds:
            return []
        return [{"label": f"{fl}  ({n})", "value": fl} for fl, n in db.q(
            """SELECT f.fl, count(*) FROM documents, UNNEST(flags) AS f(fl)
                WHERE dataset=? GROUP BY 1 ORDER BY 2 DESC""", ds)]

    @app.callback(Output("ds-doc-list", "children"),
                  Input("ds-pick", "value"), Input("ds-search", "value"),
                  Input("ds-flagged", "value"), Input("ds-split", "value"))
    def doc_list(ds, q, flagged, split):
        if not ds:
            return None
        where, args = "dataset=?", [ds]
        if q:
            where += " AND doc_id ILIKE ?"
            args.append(f"%{q}%")
        # intersection, not union: picking two flags asks for the documents
        # that carry *both*, which is how you find the compounded cases
        for fl in (flagged or []):
            where += " AND list_contains(flags, ?)"
            args.append(fl)
        if split and split != "(all)":
            where += " AND coalesce(split,'?')=?"
            args.append(split)
        rows = db.q(f"""SELECT doc_id, coalesce(split,'?'), n_words, flags
                        FROM documents WHERE {where} ORDER BY doc_id LIMIT 15""",
                    *args)
        if not rows:
            return html.Div("no matching documents", className="muted small")
        table_rows, ids = [], []
        for doc_id, sp, nw, flags in rows:
            ids.append(doc_id)
            table_rows.append([
                html.Code(doc_id), sp, human_count(nw),
                html.Span([ui.tag_chip(f) for f in (flags or [])[:3]])
                if flags else html.Span("—", className="muted")])
        return ui.table(["Document", "Split", "Words", "Flags"], table_rows,
                        num_cols={2}, row_ids=ids, table_id="ds-doc")

    @app.callback(Output("ds-doc-view", "children"),
                  Input({"type": "ds-doc-row", "key": ALL}, "n_clicks"),
                  State("ds-pick", "value"),
                  prevent_initial_call=True)
    def doc_view(clicks, ds):
        if not any(c for c in clicks if c):
            return no_update
        doc_id = ctx.triggered_id["key"]
        row = db.q1("""SELECT file_id, byte_off, byte_len, split, flags, ann_counts
                       FROM documents WHERE dataset=? AND doc_id=?""", ds, doc_id)
        if not row:
            return None
        obj = db.read_line(int(row[0]), row[1], row[2]) or {}
        secs = []
        for s in obj.get("sections", []):
            secs.append(html.Div([
                html.Div(f"{s.get('field')} · {','.join(s.get('language') or [])}",
                         className="doc-field"),
                html.Div(s.get("content", ""), className="doc-text"),
            ], className="doc-section"))
        gold_rows = db.q("""SELECT g.ann_key, g.display, g.prmu, k.pos, g.n_words
                            FROM gold g LEFT JOIN keyphrases k ON k.kp=g.display
                            WHERE g.dataset=? AND g.doc_id=?
                              AND g.ann_key <> '@combined'
                            ORDER BY g.ann_key, g.kp_idx""", ds, doc_id)
        by_ann: dict[str, list] = {}
        for ak, disp, prmu, pos, nw in gold_rows:
            by_ann.setdefault(ak, []).append(
                html.Span([html.Span(className=f"prmu-dot prmu-{prmu or 'U'}"),
                           disp, html.Span(pos or "", className="muted small")],
                          className="kp-gold",
                          title=f"{PRMU_NAMES.get(prmu, '?')}"))
        ann_blocks = [html.Div([html.Div(ak, className="doc-field"),
                                html.Div(chips)], className="doc-section")
                      for ak, chips in by_ann.items()]
        if not ann_blocks:
            counts = json.loads(row[5] or "{}")
            note = ("no annotations on this document" if not counts else
                    "gold instances are stored for eval splits and flagged "
                    "documents (run with --gold-scope all to browse all "
                    "training gold) — counts: "
                    + ", ".join(f"{k}: {v}" for k, v in counts.items()))
            ann_blocks = [html.Div(note, className="muted small")]
        flags = row[4] or []
        return ui.card([
            html.Div([html.B(doc_id), html.Span(f"  ·  split {row[3]}",
                                                className="muted small"),
                      html.Span([ui.tag_chip(f) for f in flags],
                                style={"marginLeft": "10px"})],
                     style={"marginBottom": "10px"}),
            html.Div([html.Div(secs, className="grow"),
                      html.Div(ann_blocks, style={"width": "360px",
                                                  "flexShrink": "0"})],
                     className="flex", style={"alignItems": "flex-start"}),
        ], title="Document")
