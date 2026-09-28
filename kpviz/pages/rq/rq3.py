"""RQ3 — To what extent is a keyphrase extractable?

Only *present* gold keyphrases (class P) are meaningful here: the question
is whether the model could ever have seen them. The same predictions are
scored against every present keyphrase and against those whose first
occurrence ends inside the run's usable context window (its own tokenizer,
minus reserved tokens) — gold eligibility, over one matched cohort of
documents, tested with the paired test of the Statistics bar. Panel (b) is
an observational length curve, binned independently of the scores.
"""
from __future__ import annotations

import json

import numpy as np
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ... import db, scanner, ui
from ...metrics import memo, metric_label, paired, run_scores_many
from ...naming import CONDITION, run_labels, run_rows, window_str
from ...stats import fmt_effect, p_str, sig_mark
from ..insights_common import (ann_options, best_per_model,
                               datasets_with_runs, effective_runs, effect_cell,
                               empty_result, figure_block, gate, gold_controls,
                               gold_phrase, scope_line,
                               metric_caption, metric_control, models_control,
                               p_cells, p_headers, resolve_ann, rq_header,
                               runs_control, selected_runs, split_metric,
                               stats_cfg, stats_inputs, stats_note,
                               systems_control, value_cell, vis, shown)

RQ = "rq3"     # panel (a): truncation conditions
RQB = "rq3b"   # panel (b): length bins

COND_ALL = "all present gold"
COND_WIN = "present gold inside the context window"


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("How does the input window change the eligible gold?",
                  "The predictions stay fixed; they are scored against every "
                  "present keyphrase and against those that end inside the "
                  "run's input window. The gap is how much gold a bounded "
                  "window puts out of reach — not a re-run of the model on "
                  "truncated input."),
        ui.filter_row([
            ui.control("Dataset", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds, value=_longest(ds),
                clearable=False, className="dash-dropdown"), 200),
            metric_control(RQ),
            systems_control(RQ),
        ]),
        ui.more([*gold_controls(RQ, include_prmu=False), models_control(RQ),
                 runs_control(RQ, 460)]),
        figure_block(RQ, height=340, title="A · Present gold inside the window",
                     label="Scores against all present gold and against the "
                           "gold inside each run's window"),
        html.Div(style={"height": "12px"}),
        figure_block(RQB, height=380, title="B · Scores along document length",
                     label="Scores by document length, with each run's "
                           "input window"),
    ])


def _longest(ds: list[str]):
    """Default dataset: the one with the longest documents (where a window
    can bite), else the first."""
    if not ds:
        return None
    try:
        got = dict(db.q("SELECT dataset, avg(n_words) FROM documents GROUP BY 1"))
    except Exception:
        got = {}
    return max(ds, key=lambda d: (got.get(d) or 0, -ds.index(d)))


def _as_count(v) -> int | None:
    """A positive token count, or None (never raises: ∞, NaN, "abc")."""
    try:
        n = int(v) if v is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    return n if n is not None and n > 0 else None


def _reserved(ps, resolved: dict) -> int:
    """Tokens of the window the document cannot use: special tokens, the
    prompt, few-shot examples, the output budget. The run's own
    `reserved_tokens` parameter wins over the card's `reserved_tokens` on
    the context-window parameter; neither means 0."""
    for v in ((resolved.get("reserved_tokens") or {}).get("value"),
              (ps.raw or {}).get("reserved_tokens")):
        try:
            n = int(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if n >= 0:
            return n
    return 0


def _run_limits(ds: str) -> dict[tuple, tuple | None]:
    """{(model, arch, run_id): (tokenizer, usable input limit, from_default,
    reserved) | None} for every run *of this dataset* — one query per
    catalog version. The usable limit is the window minus what the run
    reserves in it (N14).

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
                v = _as_count(given if given is not None else ps.default)
                if v and ps.tokenizer:
                    res = _reserved(ps, resolved)
                    if v - res > 0:
                        lim = (ps.tokenizer, v - res,
                               given is None or info.get("source") == "default",
                               res)
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
                             prefer=lambda: [_longest(datasets_with_runs())])
    register_model_run_chain(app, RQ, multi_ds=False)

    @app.callback(Output(f"{RQ}-ann", "options"), State(vis(RQ), "data"), Input(shown(RQ), "data"),
                  Input(f"{RQ}-ds", "value"), prevent_initial_call=True)
    def opts(visible, _shown, ds):
        if not visible:
            raise PreventUpdate
        return ann_options([ds] if ds else [])

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "rq-head", "rq": RQ}, "children"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Output({"type": "rq-graph", "rq": RQB}, "figure"),
        Output({"type": "fig-spec", "rq": RQB}, "data"),
        Output({"type": "rq-head", "rq": RQB}, "children"),
        Output({"type": "rq-table", "rq": RQB}, "children"),
        Output({"type": "fig-sig", "rq": RQ}, "data"),
        State(vis(RQ), "data"), Input(shown(RQ), "data"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-metric", "value"),
        Input(f"{RQ}-ann", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        Input(f"{RQ}-unit", "value"), *stats_inputs(),
        Input("catalog-version", "data"),
        State({"type": "fig-sig", "rq": RQ}, "data"),
        prevent_initial_call=True)
    def update(visible, _shown, *args):
        # the catalog version is an input so exactly the visible workbench
        # re-renders once when a scan publishes; gate() signs it itself
        *inputs, _catalog, last_sig = args
        sig = gate(visible, inputs, last_sig)
        return (*_update(*inputs), sig)

    def _update(ds, metric, ann_choice, models_sel, runs_sel, unit, *stat_vals):
        from ...figures import MUTED, empty_figure, to_plotly
        from ...naming import encode_runs, group_key, legend_items
        measure, k = split_metric(metric)
        if not ds:
            msg = empty_result("No dataset with runs yet — scan first.")
            return (empty_figure(), None, msg, None,
                    empty_figure(), None, None, None)
        idx = scanner.cards()
        cfg = stats_cfg(*stat_vals)
        alpha = cfg.alpha
        chosen = effective_runs([ds], models_sel, runs_sel)
        keys = selected_runs(chosen)
        ann = resolve_ann(ds, ann_choice)
        rows_meta = run_rows([ds])
        labels = run_labels(idx, rows_meta)
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
            tokz, L = lim[:2]
            have = _tok_coverage(ds, tokz)
            tok_ok[lim] = have
            if have[0]:
                calls[f"trunc{li}"] = dict(
                    dataset=ds, run_keys=lim_keys, ann_key=ann, measure=measure,
                    k=k, require_position=True, tok_limit=(tokz, L),
                    per_doc=True)
        got = run_scores_many(calls)
        present_all, plain_all = got.pop("present"), got.pop("plain")
        if unit != "run":
            # one system per model: its best run on present gold here
            keys = best_per_model(
                keys, lambda key: (present_all.get(key) or {}).get("mean"))
        shown_keys = {group_key(*key) for key in keys}
        # shades are assigned among the runs actually drawn
        enc = encode_runs(idx, [r for r in rows_meta if group_key(
            r["model"], r["arch"], r["run_id"]) in shown_keys])
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
            """Both conditions over the *same* documents — those with present
            gold inside the window — so the bars, their gap and the paired
            test describe one cohort. A run without a usable window keeps
            its all-present mean over its own documents."""
            _key, _lim, present, trunc = item
            pa = present.get("per_doc")
            pb = (trunc or {}).get("per_doc")
            out = {"ci_all": (None, None), "ci_win": (None, None), "test": None,
                   "m_all": present["mean"], "n_all": present["n"],
                   "m_win": None, "n_win": 0}
            if pa is not None and pb is not None and len(pb):
                vb, va = paired(pb, pa)
                out.update(m_all=float(va.mean()), n_all=len(va),
                           m_win=float(vb.mean()), n_win=len(vb),
                           ci_all=cfg.mean_ci(va), ci_win=cfg.mean_ci(vb),
                           test=cfg.paired(vb, va), n_own=present["n"])
            elif pa is not None:
                out["ci_all"] = cfg.mean_ci(pa.vals)
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
            win_txt = window_str(*lim) if lim else "no window"
            t = st["test"] or {}
            mark = sig_mark(pj, alpha)
            tm = st["m_win"]
            ma = st["m_all"]
            xs.append(lab)
            ys_all.append(round(ma, 4))
            ys_win.append(None if tm is None else round(tm, 4))
            err_all.append(st["ci_all"])
            err_win.append(st["ci_win"])
            own = st.get("n_own")
            hv_all.append(f"{lab}<br>{COND_ALL}: {ma:.3f} "
                          + _ci_hover(st["ci_all"]) + f"(n={st['n_all']}"
                          + (f" matched; {present['mean']:.3f} over all {own} "
                             "documents with present gold" if own and own != st["n_all"]
                             else "") + ")")
            hv_win.append(f"{lab}<br>{COND_WIN}: "
                          + ("—" if tm is None else
                             f"{tm:.3f} " + _ci_hover(st["ci_win"])
                             + f"(n={st['n_win']})")
                          + f"<br>context window {win_txt}"
                          + (f"<br>Δ = {t['diff']:+.3f} "
                             + _ci_hover((t.get('lo'), t.get('hi')))
                             + f"<br>{cfg.test_name('paired')}: p={p_str(p)}"
                             + (f", adjusted {p_str(pj)}" if cfg.adjust != "none" else "")
                             + f" {mark}<br>{cfg.effect_name('paired')} = "
                             + fmt_effect(t.get("effect"))
                             if t.get("diff") is not None else ""))
            n_all_seen.append(st["n_all"])
            if st["n_win"]:
                n_win_seen.append(st["n_win"])
            marks.append(mark)
            if unit != "run":
                # one bar per model: its name is the label (the run's
                # settings are in the hover and the table)
                xs[-1] = idx.model(key[0]).name
            cells = [value_cell(ma, st["n_all"], ci=st["ci_all"]),
                     value_cell(tm, st["n_win"] or None, ci=st["ci_win"]),
                     value_cell(t.get("diff"), None, signed=True, mark=mark,
                                ci=(t.get("lo"), t.get("hi")))]
            cells += (p_cells(p, pj, cfg, applicable=bool(t))
                      + [effect_cell(t.get("effect"))])
            tex_rows.append([lab, win_txt] + [c[1] for c in cells])
            if not lim:
                # a sentence, not five dashes
                table_rows.append([lab, html.Span(win_txt, className="muted"),
                                   cells[0][0], ui.Wide(
                                       "no window: every present keyphrase is "
                                       "within reach, nothing to compare")])
            else:
                table_rows.append([lab, win_txt] + [c[0] for c in cells])

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
        horiz = len(xs) > 8 or max((len(x) for x in xs), default=0) > 16
        # the reference condition recedes (grey), the window condition
        # carries the accent: the eye reads the gap, not two loud bars
        ser = [{"name": name_all, "x": xs, "y": ys_all, "hover": hv_all,
                "color": CONDITION["context"], "err": err_all},
               {"name": name_win, "x": xs, "y": ys_win, "hover": hv_win,
                "color": CONDITION["focus"], "text": marks, "err": err_win}]
        if horiz:
            ser = [dict(sr, x=sr["y"], y=sr["x"]) for sr in ser]
        specA = {
            "kind": "bar", "size": "2col",
            **({"orientation": "h", "xlabel": mlab,
                "aspect": min(1.35, 0.22 + 0.034 * len(xs))} if horiz
               else {"ylabel": mlab}),
            "series": ser,
            "name": f"extractability-{ds}",
            "caption": (f"{mlab} on {ds}: the same predictions scored against "
                        "two gold references — every present gold keyphrase "
                        "(class P, contiguous occurrence), and only those "
                        "whose first occurrence ends inside the run's usable "
                        "input window (the model's own tokenizer over the "
                        "concatenated document, minus any reserved tokens"
                        + ("; approximate token positions where the exact "
                           "tokenizer was unavailable" if approx_any else "")
                        + "). Both bars are over the documents with present "
                        "gold inside the window, so their gap is the paired "
                        "difference tested; this measures gold eligibility, "
                        "not a re-run of the model on truncated input"
                        + (" (at @O the cutoff also shrinks with the eligible "
                           "gold)" if k == "O" else "") + ". "
                        + (f"Error bars: {ci_txt}. " if ci_txt else "")
                        + f"Paired per-document comparison: {methods_a}. "
                        + metric_caption(measure, k, None, ann_choice, [ds])),
        }
        headers = (["Run", "Usable window", "all present gold", "gold in window",
                    "Δ (window − all)"]
                   + p_headers(cfg) + [cfg.effect_name("paired")])
        specA["table"] = {"headers": headers, "rows": tex_rows,
                          "label": f"extract-{ds}",
                          "notes": f"Statistics: {methods_a}"
                                   + (f"; {ci_txt}" if ci_txt else "") + "."}
        tableA = html.Div([
            ui.fold(ui.table(headers, table_rows,
                             num_cols=set(range(2, len(headers)))),
                    len(table_rows),
                    f"Table · {len(table_rows)} "
                    f"{'models' if unit != 'run' else 'runs'} · "
                    f"{sum(1 for m_ in marks if m_)} significant gaps"),
            stats_note(cfg, "paired", n_tests_a)])
        # every run with a window scored the same with and without it: say
        # so, or two identical bars per run read as a bug
        gaps = [st_["m_win"] - st_["m_all"] for st_ in st_a
                if st_["m_win"] is not None]
        no_gap = bool(gaps) and all(abs(g_) < 1e-12 for g_ in gaps)
        headA = scope_line(
            f"{mlab} on {ds}, present gold only",
            f"{len(runs_a)} {'models (best run each)' if unit != 'run' else 'runs'}",
            f"n = {_n_txt(n_win_seen).strip(' ()').replace('n=', '')} documents "
            "with present gold in the window" if n_win_seen else None,
            gold_phrase(ann_choice, [ds]),
            ("every document fits every window here: no gold is out of reach"
             if no_gap else None))
        if not runs_a:
            headA = empty_result(
                f"No selected run has present gold with a known position on {ds}.",
                "Pick another dataset, or clear the model and run filters.")

        # ---- panel (b): length-binned curves ------------------------------
        # all gold here on purpose: this panel asks how the *whole* task degrades
        # with length, not how much present gold survives truncation
        seriesB, vlines = [], []
        used_tok: dict[int, int] = {}
        lens_cache: dict[str, "np.ndarray"] = {}
        runs_b = []
        default_tok = _default_tokenizer(ds)
        words = None
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
            # (length, score) per document, sorted by length only (stable,
            # in document order): the score never decides which bin a
            # document falls in
            PL = np.zeros(0, dtype=np.int64)
            PS = np.zeros(0, dtype=np.float64)
            PW = np.zeros(0, dtype=np.int64)
            if per_doc is not None and len(per_doc) and len(lens):
                if words is None:
                    words = _doc_words(ds, per_doc.index)
                L = lens[per_doc.ords]
                W = words[per_doc.ords]
                ok = (L >= 0) & (W >= 0)
                PL, PS, PW = L[ok], per_doc.vals[ok], W[ok]
                order = np.argsort(PL, kind="stable")
                PL, PS, PW = PL[order], PS[order], PW[order]
            runs_b.append((key, gk, lim, PL, PS, PW))

        # documents that fit the window vs documents the model could not
        # have read in full: two independent groups
        def _stats_b(item):
            _key, _gk, lim, PL, PS, _PW = item
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
        for (key, gk, lim, PL, PS, PW), st, p, pj in zip(runs_b, st_b, pb_raw, pb_adj):
            model = key[0]
            lab = labels.get(gk, model)
            # a run without a declared window still gets a row — it is
            # stated, not dropped, so the table's run list matches the figure
            if not lim:
                blank = [value_cell(None)] * 3 + [("—", "—")] * (len(p_headers(cfg)) + 1)
                splitB_rows.append([lab, html.Span("no window", className="muted"),
                                    ui.Wide("reads every document whole: "
                                            "nothing to split")])
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
                splitB_tex.append([lab, window_str(*lim)] + [c[1] for c in cells])
                if len(w) and len(o):
                    splitB_rows.append([lab, window_str(*lim)] + [c[0] for c in cells])
                else:
                    # one side is empty: say which, instead of a row of dashes
                    splitB_rows.append([
                        lab, window_str(*lim),
                        cells[0][0] if len(w) else html.Span("none", className="muted"),
                        cells[1][0] if len(o) else html.Span("none", className="muted"),
                        ui.Wide("no test: every document is "
                                + ("longer" if len(o) else "shorter")
                                + " than the window")])
            n_pairs = len(PL)
            if n_pairs < 4:
                continue
            # one length unit for every curve: words. Each run's window is
            # converted with this dataset's measured tokens per word for its
            # tokenizer (the table above tests with the exact token counts)
            ow = np.argsort(PW, kind="stable")
            xs_b, ys_b, hv, eb = _length_bins(PW[ow], PS[ow], cfg)
            for i in range(len(xs_b)):
                hv[i] = (f"{lab}<br>~{xs_b[i]:.0f} words · n={hv[i]}<br>"
                         f"{mlab} = {ys_b[i]:.3f} " + _ci_hover(eb[i]))
            e = enc.get(gk, {})
            seriesB.append({"name": lab, "x": xs_b, "y": ys_b,
                            "hover": hv, "mode": "lines+markers",
                            "color": e.get("color", "#2a78d6"),
                            "mpl_marker": e.get("mpl_marker", "o"),
                            "shape": e.get("shape", "circle"), "width": 2,
                            "dash": e.get("dash", "solid"),
                            # bands for one or two curves only: more overlap
                            # into grey mud
                            "band": eb if cfg.ci != "none" and len(runs_b) <= 2
                            else None,
                            "legendgroup": model})
            ratio = _tokens_per_word(ds, lim[0]) if lim else None
            if lim and ratio and lim[1] not in used_tok:
                used_tok[lim[1]] = 1
                vlines.append({"x": lim[1] / ratio,
                               "label": f"≈{lim[1] / ratio:,.0f} words · "
                                        + window_str(*lim),
                               "color": MUTED, "dash": True,
                               "shade_beyond": len(keys) == 1})
        xmax = max((max(s["x"]) for s in seriesB if s["x"]), default=0)
        inside = [v for v in vlines if v["x"] <= xmax * 1.35]
        beyond = [v for v in vlines if v["x"] > xmax * 1.35]
        methods_b = cfg.method_text("indep", n_tests_b)
        specB = {
            "kind": "line", "size": "2col", "xlabel": "document length "
            "(words, bins of about equal size)"
            # the axis label only names the window sizes; which run has which
            # window is spelled out in the caption (a list of run labels
            # overflowed the figure width)
            + ("  ·  windows beyond the axis: " + ", ".join(
                (f"≈{x / 1000:.0f}k" if x >= 1000 else f"≈{x:,.0f}")
                for x in sorted({v["x"] for v in beyond})) + " words"
               if beyond else ""), "ylabel": mlab, "hovermode": "x",
            "series": seriesB, "vlines": inside,
            "legend_items": legend_items(enc, [group_key(*key) for key in keys],
                                         lines=True),
            "name": f"length-curve-{ds}",
            "caption": (f"{mlab} on {ds} across document-length bins in "
                        "words (about equal-count bins; tied lengths share a "
                        "bin and the longest documents are kept), against "
                        "all gold keyphrases. Dashed verticals mark each "
                        "run's usable input window, converted to words with "
                        f"the measured tokens per word of its tokenizer on "
                        f"{ds}. This is an observational "
                        "association: longer documents also differ in domain, "
                        "annotation density and difficulty, so the change "
                        "past a window is not by itself a truncation effect. "
                        + ("Windows beyond the plotted lengths: "
                           + "; ".join(v["label"] for v in beyond) + ". "
                           if beyond else "")
                        + (f"Shaded bands: {ci_txt}. "
                           if ci_txt and len(runs_b) <= 2 else "")
                        + metric_caption(measure, k, None, ann_choice, [ds])),
        }
        headersB = (["Run", "Usable window",
                     "docs within window", "docs beyond window",
                     "Δ (within − beyond)"]
                    + p_headers(cfg) + [cfg.effect_name("indep")])
        specB["table"] = {"headers": headersB, "rows": splitB_tex,
                          "label": f"length-split-{ds}",
                          "notes": f"Statistics: {methods_b}"
                                   + (f"; {ci_txt}" if ci_txt else "") + "."}
        tableB = (html.Div([ui.fold(ui.table(headersB, splitB_rows,
                                             num_cols=set(range(2, len(headersB)))),
                                    len(splitB_rows),
                                    f"Table · {len(splitB_rows)} runs split at "
                                    "their window"),
                            stats_note(cfg, "indep", n_tests_b)])
                  if splitB_rows else
                  html.Div("No run on this dataset declares an input window, "
                           "so no document can be called truncated.",
                           className="muted small"))
        capB = specB["caption"] + (
            " The table splits each run's documents at its own window, in "
            f"its own tokens, and compares the two groups ({methods_b}).")
        specB["caption"] = capB
        headB = scope_line(
            f"{mlab} on {ds}, all gold",
            f"{len(seriesB)} {'models (best run each)' if unit != 'run' else 'runs'}",
            "by document length in words",
            "dashed: each run's window" if vlines else "no run has a window",
            gold_phrase(ann_choice, [ds]))
        figA, figB = to_plotly(specA), to_plotly(specB)
        if not runs_a:
            figA, specA = empty_figure(), None
        if not seriesB:
            figB, specB = empty_figure(), None
            headB = empty_result(
                "Too few scored documents with a known length to draw curves.")
        return (figA, specA, headA, tableA, figB, specB, headB, tableB)


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


def _doc_words(ds: str, index) -> "np.ndarray":
    """Document lengths in words aligned on the shared document index, −1
    where unknown; memoised per catalog version."""
    def build():
        arr = np.full(len(index.ids), -1, dtype=np.int64)
        got = db.qnp("SELECT doc_id, n_words FROM documents WHERE dataset=?", ds)
        pos = index.pos
        for d, n in zip(got["doc_id"].tolist(), got["n_words"].tolist()):
            o = pos.get(d)
            if o is not None and n is not None:
                arr[o] = n
        return arr, arr.nbytes
    return memo(("rq3_words", ds, len(index.ids)), build, sized=True)


def _tokens_per_word(ds: str, tokz: str) -> float | None:
    """Measured tokens per word of `tokz` on this dataset (to draw a window
    given in tokens on a word axis)."""
    def build():
        r = db.q1("""SELECT sum(t.n_tokens)::DOUBLE / nullif(sum(d.n_words), 0)
                     FROM doc_tokens t JOIN documents d USING (dataset, doc_id)
                     WHERE t.dataset=? AND t.tokenizer=? AND d.n_words > 0""",
                  ds, tokz)
        return float(r[0]) if r and r[0] else None
    return memo(("rq3_tpw", ds, tokz), build)


def _length_bins(PL, PS, cfg):
    """About-equal-count length bins → (x = mean length, y = mean score,
    n per bin, interval per bin). PL must be sorted.

    Bin edges are length values, not positions: documents of the same
    length always share a bin (the score never decides the split), and
    every document is kept — a short last bin is merged into the one
    before it instead of being dropped (it holds the longest documents,
    the ones a truncation analysis is about)."""
    n_pairs = len(PL)
    if not n_pairs:
        return [], [], [], []
    nb = min(8, max(3, n_pairs // 12))
    target = n_pairs / nb
    # cut after position i only where the length changes
    cuts = [0]
    for b in range(1, nb):
        i = int(round(b * target))
        i = int(np.searchsorted(PL, PL[min(i, n_pairs - 1)], side="left"))
        if cuts[-1] < i < n_pairs:
            cuts.append(i)
    cuts.append(n_pairs)
    bounds = list(zip(cuts[:-1], cuts[1:]))
    min_n = max(2, int(target // 2))
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_n:
        a, _b = bounds[-2]
        bounds = bounds[:-2] + [(a, n_pairs)]
    xs, ys, ns, eb = [], [], [], []
    for a, b in bounds:
        cl, cs = PL[a:b], PS[a:b]
        xs.append(float(cl.mean()))
        ys.append(round(float(cs.mean(dtype=float)), 4))
        ns.append(len(cl))
        eb.append(cfg.mean_ci(cs))
    return xs, ys, ns, eb
