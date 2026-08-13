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

from dash import Input, Output, dcc, html

from ... import db, scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import run_labels, run_rows, window_str
from ...stats import (mann_whitney_u, p_str, sig_caption, sig_mark,
                      wilcoxon_signed_rank)
from ...util import fmt_num
from ..insights_common import (alpha_of, ann_options, datasets_with_runs,
                               effective_runs, figure_block, metric_caption,
                               metric_controls, models_control, resolve_ann,
                               rq_header, runs_control, selected_runs,
                               value_cell)

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


def _run_limit(idx, model, arch, run_id) -> tuple[str, int, bool] | None:
    """(tokenizer, effective input limit, came_from_default) for a run.

    A run that never declares its window still *has* one — the card's default —
    and the table says so rather than showing a blank."""
    mcard = idx.model(model)
    row = db.q1("""SELECT resolved FROM runs
                   WHERE model=? AND arch=? AND run_id=? LIMIT 1""",
                model, arch, run_id)
    resolved = json.loads(row[0] or "{}") if row else {}
    for ps in mcard.context_params():
        if "input" not in ps.name:
            continue
        info = resolved.get(ps.name) or {}
        given = info.get("value")
        v = given if given is not None else ps.default
        if v and ps.tokenizer:
            return (ps.tokenizer, int(v),
                    given is None or info.get("source") == "default")
    return None


def register(app):
    from ..insights_common import (register_dataset_refresh,
                                   register_model_run_chain)
    register_dataset_refresh(app, f"{RQ}-ds", multi=False,
                             prefer=["semeval2010"])
    register_model_run_chain(app, RQ, multi_ds=False)

    @app.callback(Output(f"{RQ}-ann", "options"), Input(f"{RQ}-ds", "value"))
    def opts(ds):
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
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-ann", "value"),
        Input(f"{RQ}-models", "value"), Input(f"{RQ}-runs", "value"),
        Input("ins-alpha", "value"))
    def update(ds, measure, k, ann_choice, models_sel, runs_sel, alpha_ix):
        from ...figures import MUTED, to_plotly
        from ...naming import encode_runs, group_key
        empty = to_plotly({"kind": "bar", "series": []})
        if not ds:
            return empty, None, "", None, empty, None, "", None
        idx = scanner.cards()
        alpha = alpha_of(alpha_ix)
        chosen = effective_runs([ds], models_sel, runs_sel)
        keys = selected_runs(chosen)
        ann = resolve_ann(ds, ann_choice)
        rows_meta = run_rows([ds])
        labels = run_labels(idx, rows_meta)
        enc = encode_runs(idx, rows_meta)
        mlab = metric_label(measure, k)

        # ---- panel (a): present vs present-within-window -----------------
        xs_all, ys_all, hv_all = [], [], []
        xs_win, ys_win, hv_win, marks = [], [], [], []
        table_rows, tex_rows = [], []
        n_all_seen, n_win_seen = [], []
        approx_any = False

        # one call for the "present" condition across every run, and one per
        # distinct (tokenizer, limit) — the dataset-wide gold mask is the
        # expensive part and must not be rebuilt per run
        present_all = run_scores(ds, keys, ann, measure, k,
                                 require_position=True, per_doc=True)
        limits: dict[tuple, list] = {}
        for key in keys:
            lim = _run_limit(idx, *key)
            if lim:
                limits.setdefault(lim, []).append(key)
        trunc_all: dict[tuple, dict] = {}
        tok_ok: dict[tuple, tuple] = {}
        for lim, lim_keys in limits.items():
            tokz, L, _dflt = lim
            have = db.q1("""SELECT count(*), bool_or(approx) FROM gold_tokpos
                            WHERE dataset=? AND tokenizer=?""", ds, tokz)
            tok_ok[lim] = have or (0, False)
            if have and have[0]:
                trunc_all.update(run_scores(ds, lim_keys, ann, measure, k,
                                            require_position=True,
                                            tok_limit=(tokz, L), per_doc=True))

        for key in keys:
            model, arch, run_id = key
            gk = group_key(*key)
            lab = labels.get(gk, model)
            present = present_all.get(key) or {"mean": None, "n": 0}
            if present["mean"] is None:
                continue
            lim = _run_limit(idx, model, arch, run_id)
            trunc, p_val = None, None
            win_txt = window_str(*lim) if lim else "—"
            if lim:
                have = tok_ok.get(lim, (0, False))
                if have[0]:
                    trunc = trunc_all.get(key)
                    approx_any = approx_any or bool(have[1])
                    pa = present.get("per_doc", {})
                    pb = (trunc or {}).get("per_doc", {})
                    common = sorted(set(pa) & set(pb))
                    _w, p_val = wilcoxon_signed_rank(
                        [pb[d] for d in common], [pa[d] for d in common])
            mark = sig_mark(p_val, alpha)
            xs_all.append(lab)
            ys_all.append(round(present["mean"], 3))
            hv_all.append(f"{lab}<br>{COND_ALL}: {present['mean']:.3f} "
                          f"(n={present['n']})")
            n_all_seen.append(present["n"])
            xs_win.append(lab)
            ys_win.append(None if (trunc is None or trunc["mean"] is None)
                          else round(trunc["mean"], 3))
            hv_win.append(f"{lab}<br>{COND_WIN}: "
                          + ("—" if trunc is None or trunc["mean"] is None
                             else f"{trunc['mean']:.3f} (n={trunc['n']})")
                          + f"<br>context window {win_txt}"
                          + f"<br>Wilcoxon signed-rank: p={p_str(p_val)} {mark}")
            if trunc and trunc.get("n"):
                n_win_seen.append(trunc["n"])
            marks.append(mark)
            delta = (None if (trunc is None or trunc["mean"] is None)
                     else trunc["mean"] - present["mean"])
            cells = [value_cell(present["mean"], present["n"]),
                     value_cell(None if trunc is None else trunc["mean"],
                                trunc["n"] if trunc else None),
                     value_cell(delta, None, signed=True, mark=mark)]
            table_rows.append([lab, win_txt] + [c[0] for c in cells])
            tex_rows.append([lab, win_txt] + [c[1] for c in cells])

        def _n_txt(seen):
            if not seen:
                return ""
            lo, hi = min(seen), max(seen)
            return f" (n={lo})" if lo == hi else f" (n={lo}–{hi})"

        name_all = COND_ALL + _n_txt(n_all_seen)
        name_win = COND_WIN + _n_txt(n_win_seen)
        specA = {
            "kind": "bar", "size": "2col", "ylabel": mlab,
            "series": [
                {"name": name_all, "x": xs_all, "y": ys_all, "hover": hv_all,
                 "color": "#2a78d6"},
                {"name": name_win, "x": xs_win, "y": ys_win, "hover": hv_win,
                 "color": "#1baf7a", "text": marks},
            ],
            "name": f"extractability-{ds}",
            "caption": (f"{mlab} on {ds} against present gold keyphrases "
                        "(class P, in-order occurrence) under two conditions: "
                        f"{COND_ALL} (every present keyphrase), vs. "
                        f"{COND_WIN} (only those whose first occurrence ends "
                        "inside the window, measured with the model's own "
                        "tokenizer"
                        + (", approximate token positions where the exact "
                           "tokenizer was unavailable" if approx_any else "")
                        + "). Daggers mark a significant paired difference "
                        "(two-sided Wilcoxon signed-rank on per-document "
                        f"scores; {sig_caption(alpha)}). "
                        + metric_caption(measure, k, None, ann_choice)),
        }
        headers = ["Run", "Context window", name_all, name_win, "Δ"]
        specA["table"] = {"headers": headers, "rows": tex_rows,
                          "label": f"extract-{ds}"}
        tableA = ui.table(headers, table_rows, num_cols={2, 3, 4})

        # ---- panel (b): length-binned curves ------------------------------
        # all gold here on purpose: this panel asks how the *whole* task degrades
        # with length, not how much present gold survives truncation
        seriesB, vlines = [], []
        used_tok: dict[int, int] = {}
        plain_all = run_scores(ds, keys, ann, measure, k, per_doc=True)
        lens_cache: dict[str, dict] = {}
        splitB_rows, splitB_tex = [], []
        default_tok = [r[0] for r in db.q(
            "SELECT DISTINCT tokenizer FROM doc_tokens WHERE dataset=? LIMIT 1", ds)]
        for key in keys:
            model, arch, run_id = key
            gk = group_key(*key)
            lim = _run_limit(idx, model, arch, run_id)
            tokz = lim[0] if lim else (default_tok[0] if default_tok else None)
            if tokz is None:
                continue
            if tokz not in lens_cache:
                lens_cache[tokz] = dict(db.q(
                    """SELECT doc_id, n_tokens FROM doc_tokens
                       WHERE dataset=? AND tokenizer=?""", ds, tokz))
            lens = lens_cache[tokz]
            per = plain_all.get(key) or {}
            pairs = sorted((lens.get(d), s) for d, s in per.get("per_doc", {}).items()
                           if lens.get(d) is not None)
            # documents that fit the window vs documents the model could not
            # have read in full: two independent groups, so Mann–Whitney U.
            # A run without a declared window still gets a row — it is stated,
            # not dropped, so the table's run list matches the figure's.
            if not lim:
                splitB_rows.append([labels.get(gk, model),
                                    html.Span("no window", className="muted"),
                                    *[value_cell(None)[0]] * 3])
                splitB_tex.append([labels.get(gk, model), "no window",
                                   "—", "—", "—"])
            if lim:
                within = [s for L, s in pairs if L <= lim[1]]
                over = [s for L, s in pairs if L > lim[1]]
                p_split = None
                if within and over:
                    _u, p_split = mann_whitney_u(within, over)
                m_split = sig_mark(p_split, alpha)
                d_split = ((sum(within) / len(within) - sum(over) / len(over))
                           if within and over else None)
                cells = [
                    value_cell(sum(within) / len(within) if within else None,
                               len(within) or None),
                    value_cell(sum(over) / len(over) if over else None,
                               len(over) or None),
                    value_cell(d_split, None, signed=True, mark=m_split)]
                splitB_rows.append([labels.get(gk, model), window_str(*lim)]
                                   + [c[0] for c in cells])
                splitB_tex.append([labels.get(gk, model), window_str(*lim)]
                                  + [c[1] for c in cells])
            if len(pairs) < 4:
                continue
            nb = min(8, max(3, len(pairs) // 12))
            per_bin = max(1, len(pairs) // nb)
            xs, ys, hv = [], [], []
            for i in range(0, len(pairs), per_bin):
                chunk = pairs[i:i + per_bin]
                if len(chunk) < max(2, per_bin // 2) and xs:
                    break
                xs.append(sum(p[0] for p in chunk) / len(chunk))
                ys.append(round(sum(p[1] for p in chunk) / len(chunk), 3))
                hv.append(f"{labels.get(gk, model)}<br>"
                          f"~{xs[-1]:.0f} tokens · n={len(chunk)}<br>"
                          f"{mlab} = {ys[-1]:.3f}")
            e = enc.get(gk, {})
            seriesB.append({"name": labels.get(gk, model), "x": xs, "y": ys,
                            "hover": hv, "mode": "lines+markers",
                            "color": e.get("color", "#2a78d6"),
                            "mpl_marker": e.get("mpl_marker", "o"),
                            "shape": e.get("shape", "circle"), "width": 2})
            if lim and lim[1] not in used_tok:
                used_tok[lim[1]] = 1
                vlines.append({"x": lim[1],
                               "label": f"{labels.get(gk, model)} limit {lim[1]}",
                               "color": e.get("color", MUTED), "dash": True,
                               "shade_beyond": len(keys) == 1})
        xmax = max((max(s["x"]) for s in seriesB if s["x"]), default=0)
        inside = [v for v in vlines if v["x"] <= xmax * 1.35]
        beyond = [v for v in vlines if v["x"] > xmax * 1.35]
        specB = {
            "kind": "line", "size": "2col", "xlabel": "document length "
            "(model tokens, equal-count bins)"
            + ("  ·  beyond axis: " + "; ".join(v["label"] for v in beyond)
               if beyond else ""), "ylabel": mlab,
            "series": seriesB, "vlines": inside,
            "name": f"length-curve-{ds}",
            "caption": (f"{mlab} on {ds} across document-length bins "
                        "(equal-count bins over the run's own tokenizer "
                        "counts), against all gold keyphrases. Dashed "
                        "verticals mark each run's input window; the drop past "
                        "the line is the truncation penalty. "
                        + metric_caption(measure, k, None, ann_choice)),
        }
        headersB = ["Run", "Context window",
                    "documents within model's context window",
                    "documents surpassing model's context window",
                    "Δ (within − surpassing)"]
        specB["table"] = {"headers": headersB, "rows": splitB_tex,
                          "label": f"length-split-{ds}"}
        tableB = (ui.table(headersB, splitB_rows, num_cols={2, 3, 4})
                  if splitB_rows else
                  html.Div("no run on this dataset declares a context window, "
                           "so no document can be called truncated.",
                           className="muted small"))
        capB = specB["caption"] + (
            " The table splits each run's documents at its own window and tests "
            "the two groups against each other (two-sided Mann–Whitney U; "
            + sig_caption(alpha) + ").")
        specB["caption"] = capB
        return (to_plotly(specA), specA, specA["caption"], tableA,
                to_plotly(specB), specB, capB, tableB)
