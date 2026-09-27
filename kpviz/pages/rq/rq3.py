"""RQ3 — To what extent is a keyphrase extractable?

Only *present* gold keyphrases (class P) are meaningful here: the question
is whether the model could ever have seen them. Two conditions per run —
all gold present keyphrases, vs. gold present keyphrases whose first
occurrence survives truncation to the run's context window (measured with
the model's own tokenizer). The paired difference is tested per run with a
two-sided Wilcoxon signed-rank on per-document scores (dagger marks).
"""
from __future__ import annotations

import json

import numpy as np
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ... import db, scanner, ui
from ...metrics import memo, metric_label, paired, run_scores_many
from ...naming import run_labels, run_rows, window_str
from ...stats import fmt_effect, p_str, sig_mark
from ..insights_common import (ann_options, datasets_with_runs,
                               effective_runs, effect_cell, figure_block, gate,
                               metric_caption, metric_controls, models_control,
                               p_cells, p_headers, resolve_ann, rq_header,
                               runs_control, selected_runs, stats_cfg,
                               stats_inputs, stats_note, value_cell, vis)

RQ = "rq3"     # panel (a): truncation conditions
RQB = "rq3b"   # panel (b): length bins

COND_ALL = "full-document"
COND_WIN = "document truncated to model's context window"


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("To what extent is a keyphrase extractable?",
                  "Models with a bounded input window never see part of a "
                  "long document. Both conditions evaluate against *present* "
                  "gold only (class P): every present keyphrase, vs. only "
                  "those whose first occurrence fits inside the run's context "
                  "window under the model's own tokenizer. The gap is the "
                  "truncation penalty; a dagger marks runs where the paired "
                  "difference is significant."),
        ui.filter_row([
            ui.control("Dataset", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds,
                value=("semeval2010" if "semeval2010" in ds else (ds[0] if ds else None)),
                clearable=False, className="dash-dropdown"), 200),
            *metric_controls(RQ, include_prmu=False),
        ]),
        ui.filter_row([
            models_control(RQ),
            runs_control(RQ, 460),
        ]),
        figure_block(RQ, height=430),
        html.Div(style={"height": "10px"}),
        figure_block(RQB, height=430),
    ])


def _run_limits(ds: str) -> dict[tuple, tuple | None]:
    """{(model, arch, run_id): (tokenizer, input limit, from_default) | None}
    for every run *of this dataset* — one query per catalog version.

    Run ids are reused across datasets, so the window must come from this
    dataset's run (the old per-run lookup had no dataset filter and could
    apply another dataset's window). A run that never declares its window
    still *has* one — the card's default — and the table says so. A value
    that is not a positive integer yields no window instead of a crash."""
    def build():
        idx = scanner.cards()
        out = {}
        for model, arch, run_id, resj in db.q(
                "SELECT model, arch, run_id, resolved FROM runs WHERE dataset=?", ds):
            resolved = json.loads(resj or "{}")
            lim = None
            for ps in idx.model(model).context_params():
                if "input" not in ps.name:
                    continue
                info = resolved.get(ps.name) or {}
                given = info.get("value")
                v = given if given is not None else ps.default
                try:
                    v = int(v) if v is not None else None
                except (TypeError, ValueError):
                    v = None
                if v and v > 0 and ps.tokenizer:
                    lim = (ps.tokenizer, v,
                           given is None or info.get("source") == "default")
                    break
            out[(model, arch, run_id)] = lim
        return out
    return memo(("rq3_limits", ds), build)


def _run_limit(ds, model, arch, run_id):
    return _run_limits(ds).get((model, arch, run_id))


def register(app):
    from ..insights_common import (register_dataset_refresh,
                                   register_model_run_chain)
    register_dataset_refresh(app, f"{RQ}-ds", multi=False,
                             prefer=["semeval2010"])
    register_model_run_chain(app, RQ, multi_ds=False)

    @app.callback(Output(f"{RQ}-ann", "options"), Input(vis(RQ), "data"),
                  Input(f"{RQ}-ds", "value"), prevent_initial_call=True)
    def opts(visible, ds):
        if not visible:
            raise PreventUpdate
        return ann_options([ds] if ds else [])

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "caption", "rq": RQ}, "value"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Output({"type": "rq-graph", "rq": RQB}, "figure"),
        Output({"type": "fig-spec", "rq": RQB}, "data"),
        Output({"type": "caption", "rq": RQB}, "value"),
        Output({"type": "rq-table", "rq": RQB}, "children"),
        Output({"type": "fig-sig", "rq": RQ}, "data"),
        Input(vis(RQ), "data"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-ann", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        *stats_inputs(),
        State({"type": "fig-sig", "rq": RQ}, "data"),
        prevent_initial_call=True)
    def update(visible, *args):
        *inputs, last_sig = args
        sig = gate(visible, inputs, last_sig)
        return (*_update(*inputs), sig)

    def _update(ds, measure, k, ann_choice, models_sel, runs_sel, *stat_vals):
        from ...figures import MUTED, to_plotly
        from ...naming import encode_runs, group_key
        empty = to_plotly({"kind": "bar", "series": []})
        if not ds:
            return empty, None, "", None, empty, None, "", None
        idx = scanner.cards()
        cfg = stats_cfg(*stat_vals)
        alpha = cfg.alpha
        chosen = effective_runs([ds], models_sel, runs_sel)
        keys = selected_runs(chosen)
        ann = resolve_ann(ds, ann_choice)
        rows_meta = run_rows([ds])
        labels = run_labels(idx, rows_meta)
        enc = encode_runs(idx, rows_meta)
        mlab = metric_label(measure, k)

        # one call for the "present" condition across every run, and one per
        # distinct (tokenizer, limit) — the dataset-wide gold mask is the
        # expensive part and must not be rebuilt per run
        limits: dict[tuple, list] = {}
        for key in keys:
            lim = _run_limit(ds, *key)
            if lim:
                limits.setdefault(lim, []).append(key)
        tok_ok: dict[tuple, tuple] = {}
        calls = {
            "present": dict(dataset=ds, run_keys=keys, ann_key=ann,
                            measure=measure, k=k, require_position=True,
                            per_doc=True),
            # panel (b), all gold: scored in the same concurrent batch
            "plain": dict(dataset=ds, run_keys=keys, ann_key=ann,
                          measure=measure, k=k, per_doc=True),
        }
        for li, (lim, lim_keys) in enumerate(limits.items()):
            tokz, L, _dflt = lim
            have = _tok_coverage(ds, tokz)
            tok_ok[lim] = have
            if have[0]:
                calls[f"trunc{li}"] = dict(
                    dataset=ds, run_keys=lim_keys, ann_key=ann, measure=measure,
                    k=k, require_position=True, tok_limit=(tokz, L),
                    per_doc=True)
        got = run_scores_many(calls)
        present_all, plain_all = got.pop("present"), got.pop("plain")
        trunc_all: dict[tuple, dict] = {}
        for res in got.values():
            trunc_all.update(res)

        # ---- panel (a): present vs present-within-window -----------------
        # gather, test every run concurrently (NumPy releases the GIL), then
        # correct the whole family at once: the dagger follows the adjusted p
        approx_any = False
        runs_a = []
        for key in keys:
            present = present_all.get(key) or {"mean": None, "n": 0}
            if present["mean"] is None:
                continue
            lim = _run_limit(ds, *key)
            trunc = None
            if lim and tok_ok.get(lim, (0, False))[0]:
                trunc = trunc_all.get(key)
                approx_any = approx_any or bool(tok_ok[lim][1])
            runs_a.append((key, lim, present, trunc))

        def _stats_a(item):
            _key, _lim, present, trunc = item
            pa = present.get("per_doc")
            pb = (trunc or {}).get("per_doc")
            out = {"ci_all": cfg.mean_ci(pa.vals) if pa is not None else (None, None),
                   "ci_win": (cfg.mean_ci(pb.vals) if pb is not None and len(pb)
                              else (None, None)),
                   "test": None}
            if pa is not None and pb is not None:
                vb, va = paired(pb, pa)
                out["test"] = cfg.paired(vb, va)
            return out
        st_a = _pmap(_stats_a, runs_a)
        p_raw = [(s["test"] or {}).get("p") for s in st_a]
        p_adj = cfg.adjust_all(p_raw)
        n_tests_a = sum(p is not None for p in p_raw)

        xs, ys_all, ys_win, hv_all, hv_win, marks = [], [], [], [], [], []
        err_all, err_win = [], []
        table_rows, tex_rows = [], []
        n_all_seen, n_win_seen = [], []
        for (key, lim, present, trunc), st, p, pj in zip(runs_a, st_a, p_raw, p_adj):
            gk = group_key(*key)
            lab = labels.get(gk, key[0])
            win_txt = window_str(*lim) if lim else "—"
            t = st["test"] or {}
            mark = sig_mark(pj, alpha)
            tm = None if (trunc is None or trunc["mean"] is None) else trunc["mean"]
            xs.append(lab)
            ys_all.append(round(present["mean"], 4))
            ys_win.append(None if tm is None else round(tm, 4))
            err_all.append(st["ci_all"])
            err_win.append(st["ci_win"])
            hv_all.append(f"{lab}<br>{COND_ALL}: {present['mean']:.3f} "
                          + _ci_hover(st["ci_all"]) + f"(n={present['n']})")
            hv_win.append(f"{lab}<br>{COND_WIN}: "
                          + ("—" if tm is None else
                             f"{tm:.3f} " + _ci_hover(st["ci_win"])
                             + f"(n={trunc['n']})")
                          + f"<br>context window {win_txt}"
                          + (f"<br>Δ = {t['diff']:+.3f} "
                             + _ci_hover((t.get('lo'), t.get('hi')))
                             + f"<br>{cfg.test_name('paired')}: p={p_str(p)}"
                             + (f", adjusted {p_str(pj)}" if cfg.adjust != "none" else "")
                             + f" {mark}<br>{cfg.effect_name('paired')} = "
                             + fmt_effect(t.get("effect"))
                             if t.get("diff") is not None else ""))
            n_all_seen.append(present["n"])
            if trunc and trunc.get("n"):
                n_win_seen.append(trunc["n"])
            marks.append(mark)
            cells = [value_cell(present["mean"], present["n"], ci=st["ci_all"]),
                     value_cell(tm, trunc["n"] if trunc else None,
                                ci=st["ci_win"]),
                     value_cell(t.get("diff"), None, signed=True, mark=mark,
                                ci=(t.get("lo"), t.get("hi")))]
            cells += (p_cells(p, pj, cfg, applicable=bool(t))
                      + [effect_cell(t.get("effect"))])
            table_rows.append([lab, win_txt] + [c[0] for c in cells])
            tex_rows.append([lab, win_txt] + [c[1] for c in cells])

        def _n_txt(seen):
            if not seen:
                return ""
            lo, hi = min(seen), max(seen)
            return f" (n={lo})" if lo == hi else f" (n={lo}–{hi})"

        name_all = COND_ALL + _n_txt(n_all_seen)
        name_win = COND_WIN + _n_txt(n_win_seen)
        methods_a = cfg.method_text("paired", n_tests_a)
        ci_txt = cfg.ci_text()
        # many runs: horizontal bars, one readable row per run (rotated
        # labels ate more room than the plot); few runs: columns
        horiz = len(xs) > 8
        ser = [{"name": name_all, "x": xs, "y": ys_all, "hover": hv_all,
                "color": "#2a78d6", "err": err_all},
               {"name": name_win, "x": xs, "y": ys_win, "hover": hv_win,
                "color": "#1baf7a", "text": marks, "err": err_win}]
        if horiz:
            ser = [dict(sr, x=sr["y"], y=sr["x"]) for sr in ser]
        specA = {
            "kind": "bar", "size": "2col",
            **({"orientation": "h", "xlabel": mlab,
                "aspect": min(1.35, 0.22 + 0.034 * len(xs))} if horiz
               else {"ylabel": mlab}),
            "series": ser,
            "name": f"extractability-{ds}",
            "caption": (f"{mlab} on {ds} against present gold keyphrases "
                        "(class P, in-order occurrence) under two conditions: "
                        f"{COND_ALL} (every present keyphrase), vs. "
                        f"{COND_WIN} (only those whose first occurrence ends "
                        "inside the window, measured with the model's own "
                        "tokenizer"
                        + (", approximate token positions where the exact "
                           "tokenizer was unavailable" if approx_any else "")
                        + "). "
                        + (f"Error bars: {ci_txt}. " if ci_txt else "")
                        + f"Paired per-document comparison: {methods_a}. "
                        + metric_caption(measure, k, None, ann_choice, [ds])),
        }
        headers = (["Run", "Context window", "full document", "truncated",
                    "Δ (trunc − full)"]
                   + p_headers(cfg) + [cfg.effect_name("paired")])
        specA["table"] = {"headers": headers, "rows": tex_rows,
                          "label": f"extract-{ds}",
                          "notes": f"Statistics: {methods_a}"
                                   + (f"; {ci_txt}" if ci_txt else "") + "."}
        tableA = html.Div([
            ui.table(headers, table_rows,
                     num_cols=set(range(2, len(headers)))),
            stats_note(cfg, "paired", n_tests_a)])

        # ---- panel (b): length-binned curves ------------------------------
        # all gold here on purpose: this panel asks how the *whole* task degrades
        # with length, not how much present gold survives truncation
        seriesB, vlines = [], []
        used_tok: dict[int, int] = {}
        lens_cache: dict[str, "np.ndarray"] = {}
        runs_b = []
        default_tok = _default_tokenizer(ds)
        for key in keys:
            model, arch, run_id = key
            gk = group_key(*key)
            lim = _run_limit(ds, model, arch, run_id)
            tokz = lim[0] if lim else default_tok
            if tokz is None:
                continue
            per_doc = (plain_all.get(key) or {}).get("per_doc")
            if tokz not in lens_cache:
                lens_cache[tokz] = _doc_lengths(ds, tokz, per_doc)
            lens = lens_cache[tokz]
            # (length, score) per document, sorted by length then score
            PL = np.zeros(0, dtype=np.int64)
            PS = np.zeros(0, dtype=np.float64)
            if per_doc is not None and len(per_doc) and len(lens):
                L = lens[per_doc.ords]
                ok = L >= 0
                PL, PS = L[ok], per_doc.vals[ok]
                order = np.lexsort((PS, PL))
                PL, PS = PL[order], PS[order]
            runs_b.append((key, gk, lim, PL, PS))

        # documents that fit the window vs documents the model could not
        # have read in full: two independent groups
        def _stats_b(item):
            _key, _gk, lim, PL, PS = item
            if not lim:
                return None
            within, over = PS[PL <= lim[1]], PS[PL > lim[1]]
            return {"within": within, "over": over,
                    "ci_w": cfg.mean_ci(within) if len(within) > 1 else (None, None),
                    "ci_o": cfg.mean_ci(over) if len(over) > 1 else (None, None),
                    "test": cfg.indep(within, over) if len(within) and len(over)
                    else None}
        st_b = _pmap(_stats_b, runs_b)
        pb_raw = [((s or {}).get("test") or {}).get("p") for s in st_b]
        pb_adj = cfg.adjust_all(pb_raw)
        n_tests_b = sum(p is not None for p in pb_raw)
        splitB_rows, splitB_tex = [], []
        for (key, gk, lim, PL, PS), st, p, pj in zip(runs_b, st_b, pb_raw, pb_adj):
            model = key[0]
            lab = labels.get(gk, model)
            # a run without a declared window still gets a row — it is
            # stated, not dropped, so the table's run list matches the figure
            if not lim:
                blank = [value_cell(None)] * 3 + [("—", "—")] * (len(p_headers(cfg)) + 1)
                splitB_rows.append([lab, html.Span("no window", className="muted")]
                                   + [c[0] for c in blank])
                splitB_tex.append([lab, "no window"] + [c[1] for c in blank])
            else:
                t = st["test"] or {}
                m_split = sig_mark(pj, alpha)
                w, o = st["within"], st["over"]
                cells = [
                    value_cell(float(w.mean(dtype=float)) if len(w) else None,
                               len(w) or None, ci=st["ci_w"]),
                    value_cell(float(o.mean(dtype=float)) if len(o) else None,
                               len(o) or None, ci=st["ci_o"]),
                    value_cell(t.get("diff"), None, signed=True, mark=m_split,
                               ci=(t.get("lo"), t.get("hi")))]
                cells += (p_cells(p, pj, cfg, applicable=bool(t))
                          + [effect_cell(t.get("effect"))])
                splitB_rows.append([lab, window_str(*lim)] + [c[0] for c in cells])
                splitB_tex.append([lab, window_str(*lim)] + [c[1] for c in cells])
            n_pairs = len(PL)
            if n_pairs < 4:
                continue
            xs_b, ys_b, hv, eb = _length_bins(PL, PS, cfg)
            for i in range(len(xs_b)):
                hv[i] = (f"{lab}<br>~{xs_b[i]:.0f} tokens · n={hv[i]}<br>"
                         f"{mlab} = {ys_b[i]:.3f} " + _ci_hover(eb[i]))
            e = enc.get(gk, {})
            seriesB.append({"name": lab, "x": xs_b, "y": ys_b,
                            "hover": hv, "mode": "lines+markers",
                            "color": e.get("color", "#2a78d6"),
                            "mpl_marker": e.get("mpl_marker", "o"),
                            "shape": e.get("shape", "circle"), "width": 2,
                            "band": eb if cfg.ci != "none" and len(runs_b) <= 4
                            else None,
                            "legendgroup": model})
            if lim and lim[1] not in used_tok:
                used_tok[lim[1]] = 1
                vlines.append({"x": lim[1],
                               "label": f"{lab} limit {lim[1]}",
                               "color": e.get("color", MUTED), "dash": True,
                               "shade_beyond": len(keys) == 1})
        xmax = max((max(s["x"]) for s in seriesB if s["x"]), default=0)
        inside = [v for v in vlines if v["x"] <= xmax * 1.35]
        beyond = [v for v in vlines if v["x"] > xmax * 1.35]
        methods_b = cfg.method_text("indep", n_tests_b)
        specB = {
            "kind": "line", "size": "2col", "xlabel": "document length "
            "(model tokens, equal-count bins)"
            # the axis label only names the window sizes; which run has which
            # window is spelled out in the caption (a list of run labels
            # overflowed the figure width)
            + ("  ·  windows beyond axis: " + ", ".join(
                (f"{x / 1000:.0f}k" if x >= 1000 else f"{x:g}")
                for x in sorted({v["x"] for v in beyond}))
               if beyond else ""), "ylabel": mlab,
            "series": seriesB, "vlines": inside,
            "legend": "right" if len(seriesB) > 6 else "top",
            "name": f"length-curve-{ds}",
            "caption": (f"{mlab} on {ds} across document-length bins "
                        "(equal-count bins over the run's own tokenizer "
                        "counts), against all gold keyphrases. Dashed "
                        "verticals mark each run's input window; the drop past "
                        "the line is the truncation penalty. "
                        + ("Windows beyond the plotted lengths: "
                           + "; ".join(v["label"] for v in beyond) + ". "
                           if beyond else "")
                        + (f"Shaded bands: {ci_txt}. "
                           if ci_txt and len(runs_b) <= 4 else "")
                        + metric_caption(measure, k, None, ann_choice, [ds])),
        }
        headersB = (["Run", "Context window",
                     "docs within window", "docs beyond window",
                     "Δ (within − beyond)"]
                    + p_headers(cfg) + [cfg.effect_name("indep")])
        specB["table"] = {"headers": headersB, "rows": splitB_tex,
                          "label": f"length-split-{ds}",
                          "notes": f"Statistics: {methods_b}"
                                   + (f"; {ci_txt}" if ci_txt else "") + "."}
        tableB = (html.Div([ui.table(headersB, splitB_rows,
                                     num_cols=set(range(2, len(headersB)))),
                            stats_note(cfg, "indep", n_tests_b)])
                  if splitB_rows else
                  html.Div("no run on this dataset declares a context window, "
                           "so no document can be called truncated.",
                           className="muted small"))
        capB = specB["caption"] + (
            " The table splits each run's documents at its own window and "
            f"compares the two groups ({methods_b}).")
        specB["caption"] = capB
        return (to_plotly(specA), specA, specA["caption"], tableA,
                to_plotly(specB), specB, capB, tableB)


def _pmap(fn, items):
    """Per-run statistics in parallel: the heavy parts (ranking, resampling,
    matrix products) run in NumPy with the GIL released."""
    if len(items) <= 2:
        return [fn(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as ex:
        return list(ex.map(fn, items))


def _ci_hover(ci) -> str:
    lo, hi = ci if ci else (None, None)
    return "" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}] "


def _tok_coverage(ds: str, tokz: str) -> tuple:
    """(gold rows with a token position, any approximate?) per catalog
    version — asked once per distinct window, never per interaction."""
    return memo(("rq3_tokcov", ds, tokz), lambda: tuple(db.q1(
        """SELECT count(*), coalesce(bool_or(approx), false) FROM gold_tokpos
           WHERE dataset=? AND tokenizer=?""", ds, tokz) or (0, False)))


def _default_tokenizer(ds: str):
    def build():
        r = db.q1("SELECT min(tokenizer) FROM doc_tokens WHERE dataset=?", ds)
        return r[0] if r else None
    return memo(("rq3_deftok", ds), build)


def _doc_lengths(ds: str, tokz: str, per_doc) -> "np.ndarray":
    """Document lengths (model tokens) aligned on the shared document index,
    −1 where unknown; memoised per catalog version."""
    if per_doc is None:
        return np.zeros(0, dtype=np.int64)
    index = per_doc.index

    def build():
        arr = np.full(len(index.ids), -1, dtype=np.int64)
        got = db.qnp("""SELECT doc_id, n_tokens FROM doc_tokens
                        WHERE dataset=? AND tokenizer=?""", ds, tokz)
        pos = index.pos
        for d, n in zip(got["doc_id"].tolist(), got["n_tokens"].tolist()):
            o = pos.get(d)
            if o is not None and n is not None:
                arr[o] = n
        return arr, arr.nbytes
    return memo(("rq3_lens", ds, tokz, len(index.ids)), build, sized=True)


def _length_bins(PL, PS, cfg):
    """Equal-count length bins → (x = mean length, y = mean score, n per
    bin, interval per bin)."""
    n_pairs = len(PL)
    nb = min(8, max(3, n_pairs // 12))
    per_bin = max(1, n_pairs // nb)
    xs, ys, ns, eb = [], [], [], []
    for i in range(0, n_pairs, per_bin):
        cl, cs = PL[i:i + per_bin], PS[i:i + per_bin]
        if len(cl) < max(2, per_bin // 2) and xs:
            break
        xs.append(float(cl.mean()))
        ys.append(round(float(cs.mean(dtype=float)), 4))
        ns.append(len(cl))
        eb.append(cfg.mean_ci(cs))
    return xs, ys, ns, eb
