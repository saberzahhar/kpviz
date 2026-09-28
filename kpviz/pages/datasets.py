"""Datasets explorer — statistics, annotations, quality, document browser.

Distribution charts read the tiny `gold_agg` aggregates (every split);
per-keyphrase panels (POS, browser drill-down) read gold instance rows
(eval splits under the default gold scope) joined against the global
keyphrase cache.
"""
from __future__ import annotations

import json

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..figures import MUTED, PATTERNS, to_plotly
from ..metrics import memo
from ..textproc import fix_text
from ..naming import (group_key, limit_str, natural_key, order_splits,
                      parse_group_key, run_labels, run_rows, split_color,
                      split_rank, tokenizer_label)
from ..util import declared_langs, human_count, mean_sd

# P green · R yellow · M orange · U red
# one hue, ordered by how much of the phrase the document contains: dark =
# verbatim (P) → light = none of it (U); no traffic-light judgement
PRMU_COLORS = {"P": "#0b4a6f", "R": "#3b7fa8", "M": "#8ab6d3", "U": "#cfe0ec"}
PRMU_NAMES = {"P": "Present", "R": "Reordered", "M": "Mixed", "U": "Unseen"}
WORDS_TOK = "words"
COMBINED = "@combined"
# splits carry their own fixed hue everywhere in the app (naming.SPLIT_COLORS);
# annotation sets are told apart by the label, never by a second colour scale,
# so nothing on this page needs transparency to be read.
POS_TOP_N = 4          # top-4 patterns, everything else folded into "Other"
PANEL_H = 330          # every distribution panel gets the same box


def _datasets():
    return memo("ds_datasets", lambda: [r[0] for r in db.q(
        "SELECT DISTINCT dataset FROM documents ORDER BY 1")])


def _annotators(ds: str) -> list[str]:
    """Real annotation sets, never the synthetic union."""
    return memo(("ds_annotators", ds), lambda: [r[0] for r in db.q(
        "SELECT DISTINCT ann_key FROM gold_agg WHERE dataset=? AND ann_key<>? "
        "ORDER BY 1", ds, COMBINED)])


def _splits(ds: str) -> list[str]:
    return memo(("ds_splits", ds), lambda: [r[0] for r in db.q(
        "SELECT DISTINCT coalesce(split,'?') FROM documents WHERE dataset=? "
        "ORDER BY 1", ds)])


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
        html.P("What each collection contains: splits, lengths, how much of "
               "the gold occurs in the text, and every document, read from "
               "your files.", className="page-desc"),
        ui.filter_row([
            ui.control("Dataset", dcc.Dropdown(
                id="ds-pick", options=[], clearable=False,
                placeholder="scan first…", className="dash-dropdown"), 210),
            ui.control("Split", dcc.Dropdown(
                id="ds-split", clearable=False, className="dash-dropdown"), 150),
            ui.control("Annotation sets", dcc.Dropdown(
                id="ds-ann", clearable=False, className="dash-dropdown"), 200),
            ui.control("Length in", dcc.Dropdown(
                id="ds-tok", clearable=False, className="dash-dropdown"), 230),
        ]),
        ui.loading(html.Div(id="ds-body")),
        # the browser lives outside ds-body: changing the split, annotation or
        # tokenizer re-renders the statistics, not the search you typed
        ui.card([
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
            dcc.Store(id="ds-page", data=0),
            html.Div(id="ds-doc-list"),
        ], title="Document browser"),
        html.Div(id="ds-doc-view"),
        dcc.Store(id="ds-doc-id"),
        # why a run scored what it did on the open document
        html.Div(ui.card([
            ui.filter_row([
                ui.control("Run", dcc.Dropdown(
                    id="ds-explain-run", options=[], placeholder="open a document, "
                    "then pick one of the runs that predicted it",
                    className="dash-dropdown"), 460),
                ui.control("Annotation set", dcc.Dropdown(
                    id="ds-explain-ann", options=[], clearable=False,
                    className="dash-dropdown"), 200),
            ]),
            ui.loading(html.Div(id="ds-explain")),
        ], title="Explain a run's score on this document"),
            id="ds-explain-card", style={"display": "none"}),
    ], className="page")


# ---------------------------------------------------------------------------

def _hist_bins(inner_sql: str, args: list, nbins: int = 24):
    """Per-group histogram of `inner_sql`'s (v, g) rows, binned in DuckDB —
    a training split of millions of documents is never fetched row by row.
    The range stops at the 99.5th percentile (the last bin holds the longest
    0.5 %), so one 40 000-word outlier does not squeeze every other document
    into the first bar. Returns (centers, {group: counts}, clipped?)."""
    got = db.q1(f"""SELECT min(v), quantile_cont(v, 0.995), max(v)
                    FROM ({inner_sql}) WHERE v IS NOT NULL""", *args)
    if not got or got[0] is None:
        return [], {}, False
    # lengths are whole numbers: bins are whole numbers of words or tokens
    # (a 0.6-word bin is empty every other time and draws a comb)
    import math
    lo, top = int(got[0]), int(got[2])
    hi = max(lo, int(math.ceil(float(got[1]))))
    w = max(1, math.ceil((hi - lo + 1) / nbins))
    nbins = max(1, math.ceil((hi - lo + 1) / w))
    groups: dict[str, list[int]] = {}
    for g, b, n in db.q(f"""SELECT g, least({nbins - 1}, greatest(0,
                                   floor((v - ?) / ?)))::INTEGER AS b, count(*)
                            FROM ({inner_sql}) WHERE v IS NOT NULL
                            GROUP BY 1, 2""", lo, w, *args):
        groups.setdefault(str(g), [0] * nbins)[b] += int(n)
    centers = [round(lo + (i + 0.5) * w - 0.5, 1) for i in range(nbins)]
    return centers, groups, top > lo + nbins * w - 1


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
                     ("distinct stemmed forms, union of " + ", ".join(anns))
                     if len(anns) > 1 else "distinct stemmed forms"),
        ui.stat_tile("Keyphrases per document", mean_sd(n_kp_doc, 1), basis),
        ui.stat_tile("Present (P) per document",
                     mean_sd(p_share, 1, pct=True),
                     "share of a document's gold occurring verbatim (contiguous, stemmed)"),
    ])

    charts = []
    # ---- document length panel — "words" (default) or a model tokenizer ----
    tok = tok or WORDS_TOK
    approx = None
    if tok == WORDS_TOK:
        inner = ("SELECT n_words AS v, coalesce(split,'?') AS g FROM documents "
                 "WHERE dataset=?")
        inner_args = [ds]
        xlabel, title, uv = "document length (words)", "Document length", []
    else:
        inner = ("""SELECT t.n_tokens AS v, coalesce(d.split,'?') AS g
                    FROM doc_tokens t JOIN documents d USING (dataset, doc_id)
                    WHERE t.dataset=? AND t.tokenizer=?""")
        inner_args = [ds, tok]
        approx = db.q1("SELECT bool_or(approx) FROM doc_tokens WHERE dataset=? AND tokenizer=?",
                       ds, tok)
        xlabel = (f"document length ({tokenizer_label(tok)}"
                  + (" tokens, approximate)" if approx and approx[0]
                     else " tokens)"))
        title = "Document length and the models' input windows"
        # the input window of every model that counts in this tokenizer —
        # its card default, and any other value its runs used here — whether
        # or not the model ran on this dataset
        wins: dict[int, list[str]] = {}
        resolved = {}
        for m, rj in db.q("SELECT DISTINCT model, resolved FROM runs WHERE dataset=?", ds):
            resolved.setdefault(m, []).append(json.loads(rj or "{}"))
        for m in sorted(set(idx.models) | set(resolved)):
            mc = idx.model(m)
            for p in mc.context_params():
                if p.tokenizer != tok or "input" not in p.name:
                    continue
                vals = {p.default} | {(r.get(p.name) or {}).get("value")
                                      for r in resolved.get(m, [])}
                for v in vals:
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        continue
                    if v > 0 and mc.name not in wins.setdefault(v, []):
                        wins[v].append(mc.name)
        uv = [{"x": v, "label": f"{limit_str(v)} · " + ", ".join(names[:2])
               + (f" +{len(names) - 2}" if len(names) > 2 else ""),
               "color": MUTED, "dash": True} for v, names in sorted(wins.items())]
    centers, groups, clipped = _hist_bins(inner, inner_args)
    if clipped:
        xlabel += " (the last bin also holds the longest 0.5 %)"
    if centers:
        span = max(centers)
        inside = [v for v in uv if v["x"] <= span * 1.35][:4]
        beyond = [v for v in uv if v["x"] > span * 1.35]
        # each split is normalised to its own 100%: collections are wildly
        # unbalanced (1.3 M training vs 100 k testing), and the question here
        # is whether the *shapes* differ, not which split is bigger
        # one outline per split (a frequency polygon), not 24 × 3 bars: the
        # shapes are compared, and three thin lines read at a glance
        series = []
        for sp in order_splits(groups):
            ys = groups[sp]
            tot = sum(ys) or 1
            series.append({
                "name": sp, "x": centers, "mode": "lines", "width": 1.8,
                "y": [round(100.0 * y / tot, 2) for y in ys],
                "color": split_color(sp),
                "hover": [f"{sp}<br>~{c:g} · {y} docs ({100.0 * y / tot:.1f}%)"
                          for c, y in zip(centers, ys)]})
        spec = {"kind": "line", "xlabel": xlabel,
                "ylabel": "% of the split's documents",
                "series": series, "vlines": inside, "size": "2col",
                "name": f"doc-length-{ds}",
                "caption": (f"Document length distribution of {ds} per split, "
                            + ("in words" if tok == WORDS_TOK
                               else f"in {tokenizer_label(tok)} tokens")
                            + (" (approximate)" if approx and approx[0] else "")
                            + ("; dashed lines mark the input windows of the "
                               "models that count in this tokenizer"
                               if tok != WORDS_TOK else "")
                            + (" (beyond the axis: " + "; ".join(
                                v["label"] for v in beyond) + ")" if beyond else "")
                            + ".")}
        note = (html.Div("beyond the axis: " + " · ".join(
            v["label"] for v in beyond), className="muted small")
            if beyond else None)
        charts.append(ui.exportable(
            "ds-len", spec, html.Div([ui.graph("ds-g-len", to_plotly(spec), PANEL_H),
                                      note]), title=title))

    # ---- PRMU: one aligned 100 % bar per (annotation set, split) -----------
    # (aligned bars, not pies: the P share of two splits is compared on a
    # common baseline instead of by angle)
    rows = db.q(f"""SELECT ann_key, coalesce(split,'?'), prmu, sum(n)
                    FROM gold_agg WHERE {agg_where} GROUP BY 1,2,3""", *agg_args)
    if rows:
        cell: dict[tuple[str, str], dict] = {}
        for a, sp, pr, n in rows:
            cell.setdefault((a, sp), {})[pr or "U"] = n
        rows_used = [a for a in anns if any((a, sp) in cell for sp in splits)]
        cols_used = [sp for sp in order_splits(splits)
                     if any((a, sp) in cell for a in rows_used)]
        order = ["P", "R", "M", "U"]
        cats, tots = [], {}
        for a in rows_used:
            for sp in cols_used:
                c = cell.get((a, sp), {})
                if not sum(c.values()):
                    continue
                lab = f"{a} · {sp}" if len(rows_used) > 1 else sp
                cats.append(lab)
                tots[lab] = c
        series = []
        for pr in order:
            ys, hv = [], []
            for lab in cats:
                c = tots[lab]
                tot = sum(c.values())
                ys.append(round(100.0 * c.get(pr, 0) / tot, 2))
                hv.append(f"{lab}<br>{pr} — {PRMU_NAMES[pr]}: "
                          f"{human_count(c.get(pr, 0))} of {human_count(tot)} "
                          f"({100.0 * c.get(pr, 0) / tot:.1f}%)")
            series.append({"name": f"{pr} — {PRMU_NAMES[pr]}",
                           "x": ys, "y": list(cats), "hover": hv,
                           "color": PRMU_COLORS[pr]})
        spec = {"kind": "bar", "barmode": "stack", "orientation": "h",
                "xlabel": "% of the gold keyphrases", "xrange": [0, 100],
                "series": series, "size": "2col",
                "aspect": min(1.2, 0.18 + 0.07 * len(cats)),
                "name": f"prmu-{ds}",
                "caption": (f"PRMU classes of the gold keyphrases of {ds} per "
                            "annotation set and split (Boudin & Gallina, 2021, "
                            "on stemmed tokens: P = the keyphrase occurs "
                            "contiguously in the document, R = all its words "
                            "occur but not as that sequence, M = some occur, "
                            "U = none do).")}
        charts.append(ui.exportable(
            "ds-prmu", spec,
            ui.graph("ds-g-prmu", to_plotly(spec), max(200, 44 * len(cats) + 110)),
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
        # keyphrase length is in words; the tokenizer selector drives the
        # *document* length panel only
        spec = {"kind": "bar", "barmode": "group",
                "xlabel": f"length ({WORDS_TOK})",
                "ylabel": "% of the group's gold", "yrange": [0, 100],
                "series": _group_series(data, cats, anns, splits),
                "size": "1col", "name": f"kp-length-{ds}",
                "caption": (f"Length of the gold keyphrases of {ds} in words, per annotation set and split.")}
        c1 = ui.exportable("ds-kplen", spec,
                           ui.graph("ds-g-kplen", to_plotly(spec), PANEL_H),
                           title="Keyphrase length", style={"minWidth": 0})
    rows = db.q(f"""SELECT g.ann_key, coalesce(d.split,'?'), k.pos, count(*)
                    FROM gold g
                    JOIN documents d USING (dataset, doc_id)
                    JOIN keyphrases k ON k.kp = g.display
                     AND k.lang = left(coalesce(g.lang, 'en'), 2)
                    WHERE g.dataset=? AND g.ann_key IN ({ann_sql})
                      {'AND d.split=?' if use_split else ''}
                      AND k.pos IS NOT NULL AND k.pos <> ''
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
                                        splits, horizontal=True),
                "size": "1col", "name": f"kp-pos-{ds}",
                "caption": (f"Part-of-speech patterns of the gold keyphrases of "
                            f"{ds} (spaCy tagger; the {POS_TOP_N} most frequent "
                            "patterns, the rest as Other), per annotation set "
                            "and split.")}
        c2 = ui.exportable("ds-pos", spec,
                           ui.graph("ds-g-pos", to_plotly(spec), PANEL_H),
                           title="Keyphrase POS tags", style={"minWidth": 0})
    else:
        c2 = ui.card(html.Div("No POS-tagged keyphrases here yet — install the "
                              "spaCy model for this language and re-scan, or "
                              "widen --gold-scope.", className="muted small"),
                     title="Keyphrase POS tags", style={"minWidth": 0})
    grid1 = html.Div([c for c in (c1, c2) if c], className="grid-2")
    return html.Div([header, kpis, *charts, grid1])


# ---------------------------------------------------------------------------

def register(app):
    @app.callback(Output("ds-pick", "options"), Output("ds-pick", "value"),
                  State("vis-datasets", "data"), Input("shown-datasets", "data"),
                  Input("catalog-version", "data"),
                  State("ds-pick", "options"), State("ds-pick", "value"),
                  prevent_initial_call=True)
    def refresh_datasets(visible, _shown, _v, cur_opts, current):
        """Only while the page is on screen; a hidden page catches up when
        shown (the catalog version is an input)."""
        if not visible:
            raise PreventUpdate
        ds = _datasets()
        if cur_opts == ds and current in ds:
            raise PreventUpdate
        value = current if current in ds else (ds[0] if ds else None)
        return ds, value

    @app.callback(Output("ds-ann", "options"), Output("ds-ann", "value"),
                  Output("ds-split", "options"), Output("ds-split", "value"),
                  Output("ds-tok", "options"), Output("ds-tok", "value"),
                  Input("ds-pick", "value"), prevent_initial_call=True)
    def set_options(ds):
        if not ds:
            return [], None, [], None, [], None
        # "(each)" is the default: every annotation set gets its own row/series
        anns = [{"label": "each set", "value": "(each)"}] + [
            {"label": a, "value": a} for a in _annotators(ds)]
        splits = ([{"label": "all splits", "value": "(all)"}]
                  + [{"label": s_, "value": s_} for s_ in order_splits(_splits(ds))])
        toks = [{"label": "words", "value": WORDS_TOK}] + [
            {"label": f"{tokenizer_label(t)} tokens", "value": t}
            for t in (r[0] for r in db.q(
                "SELECT DISTINCT tokenizer FROM doc_tokens WHERE dataset=? "
                "ORDER BY 1", ds))]
        return (anns, "(each)", splits, "(all)", toks, WORDS_TOK)

    @app.callback(Output("ds-body", "children"),
                  Input("ds-pick", "value"), Input("ds-ann", "value"),
                  Input("ds-split", "value"), Input("ds-tok", "value"),
                  prevent_initial_call=True)
    def body(ds, ann, split, tok):
        if not ds:
            return ui.empty_state("No datasets in the catalog — run a scan on "
                                  "the Overview page.")
        if not _annotators(ds):
            return ui.empty_state("No annotations derived for this dataset yet.")
        return _body(ds, ann, split, tok)

    @app.callback(Output("ds-flagged", "options"), Input("ds-pick", "value"),
                  prevent_initial_call=True)
    def flag_options(ds):
        if not ds:
            return []
        return [{"label": f"{fl}  ({n})", "value": fl} for fl, n in db.q(
            """SELECT f.fl, count(*) FROM documents, UNNEST(flags) AS f(fl)
                WHERE dataset=? GROUP BY 1 ORDER BY 2 DESC""", ds)]

    @app.callback(Output("ds-doc-list", "children"), Output("ds-page", "data"),
                  Input("ds-pick", "value"), Input("ds-search", "value"),
                  Input("ds-flagged", "value"), Input("ds-split", "value"),
                  Input({"type": "ds-pager", "dir": ALL}, "n_clicks"),
                  State("ds-page", "data"),
                  prevent_initial_call=True)
    def doc_list(ds, q, flagged, split, _pager, page):
        if not ds:
            return None, 0
        trig = ctx.triggered_id
        page = int(page or 0)
        if isinstance(trig, dict) and trig.get("type") == "ds-pager":
            page = max(0, page + (1 if trig.get("dir") == "next" else -1))
        else:
            page = 0                      # a new filter starts at the top
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
        # natural order (kp20k_testing_2 before _10) over *every* match, in
        # SQL, paged — sorting the first 50 000 unordered ids made the "first
        # 25" arbitrary on large collections
        per = 12
        total = db.q1(f"SELECT count(*) FROM documents WHERE {where}", *args)[0]
        if not total:
            return html.Div("no matching documents", className="muted small"), 0
        page = min(page, (total - 1) // per)
        rows = db.q(f"""SELECT doc_id, coalesce(split,'?'), n_words, flags
                        FROM documents WHERE {where}
                        ORDER BY regexp_replace(doc_id, '\\d+$', ''),
                                 TRY_CAST(regexp_extract(doc_id, '(\\d+)$', 1)
                                          AS BIGINT) NULLS FIRST,
                                 doc_id
                        LIMIT {per} OFFSET {page * per}""", *args)
        table_rows, ids = [], []
        for doc_id, sp, nw, flags in rows:
            ids.append(doc_id)
            table_rows.append([
                html.Code(doc_id), sp, human_count(nw),
                html.Span([ui.tag_chip(f) for f in (flags or [])[:3]])
                if flags else html.Span("—", className="muted")])
        first = page * per + 1
        pager = html.Span([
            html.Button("‹ Previous", id={"type": "ds-pager", "dir": "prev"},
                        className="btn small", disabled=page == 0),
            html.Button("Next ›", id={"type": "ds-pager", "dir": "next"},
                        className="btn small", disabled=first + len(rows) > total,
                        style={"marginLeft": "6px"}),
        ], style={"marginLeft": "10px"})
        return html.Div([
            html.Div([f"{first:,}–{first + len(rows) - 1:,} of {total:,} matching "
                      "documents, in natural order — click a row (or focus it "
                      "and press Enter) to open it", pager],
                     className="muted small", style={"marginBottom": "4px"}),
            ui.table(["Document", "Split", "Words", "Flags"], table_rows,
                     num_cols={2}, row_ids=ids, table_id="ds-doc")]), page

    @app.callback(Output("ds-explain-run", "options"), Output("ds-explain-run", "value"),
                  Output("ds-explain-ann", "options"), Output("ds-explain-ann", "value"),
                  Output("ds-explain-card", "style"),
                  Input("ds-doc-id", "data"), State("ds-pick", "value"),
                  prevent_initial_call=True)
    def explain_options(doc_id, ds):
        if not doc_id or not ds:
            return [], None, [], None, {"display": "none"}
        idx = scanner.cards()
        rows = [r for r in run_rows([ds])]
        labels = run_labels(idx, rows)
        have = {tuple(r) for r in db.q(
            """SELECT DISTINCT model, arch, run_id FROM matches
               WHERE dataset=? AND doc_id=?""", ds, doc_id)}
        opts = [{"label": labels.get(group_key(*k), k[0]), "value": group_key(*k)}
                for k in sorted(have, key=lambda k: natural_key(
                    labels.get(group_key(*k), k[0])))]
        anns = [r[0] for r in db.q(
            """SELECT DISTINCT ann_key FROM gold WHERE dataset=? AND doc_id=?
               ORDER BY 1""", ds, doc_id)]
        ann = "@combined" if "@combined" in anns else (anns[0] if anns else None)
        style = {"display": "block"} if opts else {"display": "none"}
        return (opts, opts[0]["value"] if opts else None, anns, ann, style)

    @app.callback(Output("ds-explain", "children"),
                  Input("ds-explain-run", "value"), Input("ds-explain-ann", "value"),
                  State("ds-doc-id", "data"), State("ds-pick", "value"),
                  prevent_initial_call=True)
    def explain(run_key, ann, doc_id, ds):
        if not run_key or not ann or not doc_id or not ds:
            return None
        return _explain(ds, doc_id, parse_group_key(run_key), ann)

    @app.callback(Output("ds-doc-view", "children"), Output("ds-doc-id", "data"),
                  Input({"type": "ds-doc-row", "key": ALL}, "n_clicks"),
                  State("ds-pick", "value"),
                  prevent_initial_call=True)
    def doc_view(clicks, ds):
        if not any(c for c in clicks if c):
            return no_update, no_update
        doc_id = ctx.triggered_id["key"]
        return _doc_card(ds, doc_id), doc_id

    def _doc_card(ds, doc_id):
        row = db.q1("""SELECT file_id, byte_off, byte_len, split, flags, ann_counts
                       FROM documents WHERE dataset=? AND doc_id=?""", ds, doc_id)
        if not row:
            return None
        obj = db.read_line(int(row[0]), row[1], row[2])
        src_note = None
        if obj is None:
            src_note = ("the source line cannot be read — the file is missing "
                        "or changed since the last scan; rescan to refresh")
            obj = {}
        elif str(obj.get("_id")) != str(doc_id) and not str(doc_id).startswith("@"):
            src_note = ("the source file changed since the last scan (this "
                        "offset now holds another document); rescan to refresh")
            obj = {}
        secs = []
        for s in obj.get("sections", []):
            secs.append(html.Div([
                html.Div(f"{s.get('field')} · "
                         f"{','.join(declared_langs(s)) or 'language from the card'}",
                         className="doc-field"),
                # Windows-1252 bytes read as Latin-1 ("d\x92analyse") shown
                # as the characters they were
                html.Div(fix_text(s.get("content", "")), className="doc-text"),
            ], className="doc-section"))
        gold_rows = db.q("""SELECT g.ann_key, coalesce(g.surface, g.display),
                                   g.prmu, k.pos, g.n_words
                            FROM gold g LEFT JOIN keyphrases k ON k.kp=g.display
                              AND k.lang = left(coalesce(g.lang, 'en'), 2)
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
            (html.Div(src_note, className="ins-banner", role="status")
             if src_note else None),
            html.Div([html.Div(secs, className="grow"),
                      html.Div(ann_blocks, className="doc-anns")],
                     className="flex doc-view", style={"alignItems": "flex-start"}),
        ], title="Document")


def _explain(ds: str, doc_id: str, key: tuple, ann: str):
    """A run's predictions on one document, as the scorer saw them: ranked,
    normalised and de-duplicated, each marked with the gold keyphrase it
    matched (or none), then the gold it missed — the evidence behind every
    P/R/F1 number for this (run, document)."""
    from ..derive import _uniq_stems
    from ..textproc import PhraseCache
    model, arch, run_id = key
    m = db.q1("""SELECT pred_ranks, gold_idxs, n_uniq, n_gold FROM matches
                 WHERE dataset=? AND model=? AND arch=? AND run_id=?
                   AND doc_id=? AND ann_key=?""", ds, model, arch, run_id,
              doc_id, ann)
    p = db.q1("""SELECT file_id, byte_off, byte_len FROM preds
                 WHERE dataset=? AND model=? AND arch=? AND run_id=? AND doc_id=?""",
              ds, model, arch, run_id, doc_id)
    if not m or not p:
        return html.Div("this run has no scored line for this document "
                        "and annotation set", className="muted small")
    line = db.read_line(int(p[0]), p[1], p[2])
    if line is None or str(line.get("_id")) != str(doc_id):
        return html.Div("the prediction file changed since the last scan — "
                        "rescan to explain this run", className="ins-banner")
    raw = [fix_text(str(x)) for x in (line.get("inferences") or [])]
    lang = (scanner.cards().dataset(ds).languages[:1] or ["en"])[0]
    cache = PhraseCache(max_size=4096)
    an = [cache.analyze(x, lang, persist=False) for x in raw]
    uniq, first = _uniq_stems(an)
    ranks, gidx = list(m[0] or []), list(m[1] or [])
    hit = dict(zip(ranks, gidx))
    gold = db.q("""SELECT kp_idx, coalesce(surface, display), prmu FROM gold
                   WHERE dataset=? AND doc_id=? AND ann_key=? ORDER BY kp_idx""",
                ds, doc_id, ann)
    gname = {g[0]: g[1] for g in gold}
    first_of = {}
    for i, a in enumerate(an):
        first_of.setdefault(a["sstr"], i)
    rank_of_raw = {fi: r for r, fi in enumerate(first)}
    items = []
    for i, (x, a) in enumerate(zip(raw, an)):
        r = rank_of_raw.get(i)
        if r is None:
            why = ("no word token" if not a["sstr"] else
                   f"duplicate of #{rank_of_raw.get(first_of[a['sstr']], 0) + 1}")
            items.append(html.Li([html.Span(x, className="muted"),
                                  html.Span(f"  — {why}", className="muted small")],
                                 className="pred dup"))
            continue
        g = hit.get(r)
        mark = "✓" if g is not None else "✗"
        items.append(html.Li([
            html.Span(f"#{r + 1} ", className="muted small"),
            html.Span(mark, className="ok" if g is not None else "bad",
                      title="matches a gold keyphrase" if g is not None
                      else "matches no gold keyphrase"),
            " ", html.Span(x),
            html.Span(f"  → {gname.get(g, g)}" if g is not None else "",
                      className="muted small"),
            html.Span(f"  [{a['sstr']}]", className="muted small mono"),
        ], className="pred"))
    missed = [g for g in gold if g[0] not in set(gidx)]
    n_uniq, n_gold = int(m[2] or 0), int(m[3] or 0)

    def score(k):
        cut = n_gold if k == "O" else (max(n_uniq, 1) if k == "M" else int(k))
        tp = sum(1 for r in ranks if r < cut)
        dp = min(cut, n_uniq) if n_uniq else 0
        p_ = tp / dp if dp else 0.0
        r_ = tp / n_gold if n_gold else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return f"@{k}: P {p_:.2f} · R {r_:.2f} · F1 {f_:.2f}"
    return html.Div([
        html.Div(f"{len(raw)} predictions → {n_uniq} unique after normalisation "
                 f"and stemming · {len(ranks)} of {n_gold} gold matched · "
                 + "  |  ".join(score(k) for k in ("5", "10", "O", "M")),
                 className="small", style={"marginBottom": "8px"}),
        html.Div([
            html.Div([html.Div("Ranked predictions", className="doc-field"),
                      html.Ol(items, className="pred-list")], className="grow"),
            html.Div([html.Div(f"Missed gold ({len(missed)})", className="doc-field"),
                      html.Div([html.Span([html.Span(className=f"prmu-dot prmu-{pr or 'U'}"),
                                           name], className="kp-gold",
                                          title=PRMU_NAMES.get(pr, "?"))
                                for _i, name, pr in missed]
                               or [html.Span("none", className="muted small")])],
                     className="doc-anns"),
        ], className="flex doc-view", style={"alignItems": "flex-start"}),
    ])
