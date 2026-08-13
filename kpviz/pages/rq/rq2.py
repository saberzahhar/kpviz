"""RQ2 — To what extent can poor input data quality affect evaluation?

Quality criteria (language mismatches, train→test similarity pairs against
a model's own supervision data, or any high-similarity pair) flag documents;
the metric is recomputed with and without them and the delta reported with
sample sizes."""
from __future__ import annotations

import json

from dash import Input, Output, dcc, html

from ... import db, scanner, ui
from ...metrics import metric_label, run_scores
from ...naming import run_labels, run_rows, parse_group_key
from ...stats import mann_whitney_u, p_str, sig_caption, sig_mark
from ..insights_common import (alpha_of, ann_options, datasets_with_runs,
                               effective_runs, figure_block, metric_caption,
                               metric_controls, models_control, prmu_arg,
                               resolve_ann, rq_header, runs_control,
                               selected_runs, value_cell)

RQ = "rq2"

CRITERIA = [
    {"label": " language mismatch (any section/annotation)", "value": "lang"},
    {"label": " similar to the model's own training data", "value": "sup_leak"},
    {"label": " similar to any other document (any pair)", "value": "any_leak"},
]


def layout():
    ds = datasets_with_runs()
    return html.Div([
        rq_header("To what extent can poor input data quality affect evaluation?",
                  "Documents are flagged by data-quality checks — detected "
                  "language disagreeing with the declared one, or testing "
                  "documents whose similarity pairs point back to a model's "
                  "own supervision data (train→test leakage). The metric is "
                  "recomputed excluding the flagged documents; the delta is "
                  "the bias those documents inject."),
        ui.filter_row([
            ui.control("Dataset", dcc.Dropdown(
                id=f"{RQ}-ds", options=ds,
                value=("kpbiomed" if "kpbiomed" in ds else (ds[0] if ds else None)),
                clearable=False, className="dash-dropdown"), 200),
            *metric_controls(RQ),
        ]),
        ui.filter_row([
            models_control(RQ),
            runs_control(RQ, 460),
        ]),
        ui.filter_row([
            ui.control("Quality criteria", dcc.Checklist(
                id=f"{RQ}-crit", options=CRITERIA,
                value=["lang", "sup_leak"], className="kp-check"), 380),
            ui.control("Similarity threshold", dcc.Slider(
                id=f"{RQ}-thr", min=0.5, max=1.0, step=0.01, value=0.8,
                marks={0.5: "0.5", 0.75: "0.75", 1.0: "1.0"},
                tooltip={"placement": "bottom", "always_visible": True}), 300),
            ui.control("Pair label", dcc.Dropdown(
                id=f"{RQ}-label", options=["(any)"], value="(any)",
                clearable=False, className="dash-dropdown"), 190),
        ]),
        figure_block(RQ, height=430),
    ])


def _leak_docs(ds: str, thr: float, label: str | None,
               sup_datasets: set[str] | None) -> set[str]:
    """Documents of `ds` flagged through similarity pairs — one query.

    sup_datasets: restrict the counterpart side to these datasets' *training*
    documents (None = any counterpart). The previous version issued one point
    lookup per pair to resolve the counterpart's split, which is thousands of
    statements for a few thousand pairs."""
    args: list = [ds, ds, float(thr)]
    sql = """
        WITH sided AS (
            SELECT doc_id_a AS mine, dataset_b AS other_ds, doc_id_b AS other_id,
                   score, label FROM leakage WHERE dataset_a = ?
            UNION ALL
            SELECT doc_id_b AS mine, dataset_a AS other_ds, doc_id_a AS other_id,
                   score, label FROM leakage WHERE dataset_b = ?
        )
        SELECT DISTINCT s.mine FROM sided s"""
    if sup_datasets is not None:
        ph = ",".join("?" * len(sup_datasets))
        sql += f"""
        JOIN documents d ON d.dataset = s.other_ds AND d.doc_id = s.other_id
        WHERE s.score >= ? AND s.other_ds IN ({ph})
          AND lower(coalesce(d.split, '')) IN ('train', 'training')"""
        args += sorted(sup_datasets)
    else:
        sql += " WHERE s.score >= ?"
    if label and label != "(any)":
        sql += " AND coalesce(s.label, '(unlabelled)') = ?"
        args.append(label)
    return {r[0] for r in db.q(sql, *args)}


def register(app):
    from ..insights_common import (register_dataset_refresh,
                                   register_model_run_chain)
    register_dataset_refresh(app, f"{RQ}-ds", multi=False,
                             prefer=["kpbiomed"])
    register_model_run_chain(app, RQ, multi_ds=False)

    @app.callback(Output(f"{RQ}-ann", "options"),
                  Output(f"{RQ}-label", "options"),
                  Input(f"{RQ}-ds", "value"))
    def opts(ds):
        labels = ["(any)"] + [r[0] for r in db.q(
            """SELECT DISTINCT coalesce(label,'(unlabelled)') FROM leakage
               WHERE dataset_a=? OR dataset_b=? ORDER BY 1""", ds, ds)]
        return ann_options([ds] if ds else []), labels

    @app.callback(
        Output({"type": "rq-graph", "rq": RQ}, "figure"),
        Output({"type": "fig-spec", "rq": RQ}, "data"),
        Output({"type": "caption", "rq": RQ}, "value"),
        Output({"type": "rq-table", "rq": RQ}, "children"),
        Input(f"{RQ}-ds", "value"), Input(f"{RQ}-measure", "value"),
        Input(f"{RQ}-k", "value"), Input(f"{RQ}-prmu", "value"),
        Input(f"{RQ}-ann", "value"), Input(f"{RQ}-models", "value"),
        Input(f"{RQ}-runs", "value"),
        Input(f"{RQ}-crit", "value"), Input(f"{RQ}-thr", "value"),
        Input(f"{RQ}-label", "value"), Input("ins-alpha", "value"))
    def update(ds, measure, k, prmu_sel, ann_choice, models_sel, runs_sel,
               crit, thr, label, alpha_ix):
        from ...figures import to_plotly
        if not ds:
            return (to_plotly({"kind": "bar", "series": []}), None, "",
                    html.Div("no dataset", className="muted small"))
        idx = scanner.cards()
        alpha = alpha_of(alpha_ix)
        chosen = effective_runs([ds], models_sel, runs_sel)
        keys = selected_runs(chosen)
        ann = resolve_ann(ds, ann_choice)
        prmu = prmu_arg(prmu_sel)
        crit = crit or []

        lang_docs: set[str] = set()
        if "lang" in crit:
            lang_docs = {r[0] for r in db.q(
                """SELECT doc_id FROM documents, UNNEST(flags) AS f(fl)
                   WHERE dataset=? AND fl LIKE 'lang_mismatch%'""", ds)}
        any_leak = (_leak_docs(ds, thr or 0.8, label, None)
                    if "any_leak" in crit else set())

        labels = run_labels(idx, run_rows([ds]))
        rows_out, table_rows, tex_rows = [], [], []

        # one scoring pass for every run: per-document scores answer all four
        # questions (all / excluding flagged / flagged only / the test) in
        # Python arithmetic, so the gold mask is built once instead of 4×N times
        per_all = run_scores(ds, keys, ann, measure, k, prmu=prmu, per_doc=True)
        # the flagged set only varies with the model's supervision datasets
        sup_cache: dict[frozenset, set[str]] = {}
        for key in keys:
            model, arch, run_id = key
            flagged = set(lang_docs) | set(any_leak)
            if "sup_leak" in crit:
                sup = frozenset(idx.model(model).supervision)
                if sup:
                    if sup not in sup_cache:
                        sup_cache[sup] = _leak_docs(ds, thr or 0.8, label,
                                                    set(sup))
                    flagged |= sup_cache[sup]
            per = per_all.get(key, {}).get("per_doc", {}) or {}
            vals_all = list(per.values())
            fl = [s for d, s in per.items() if d in flagged]
            cl = [s for d, s in per.items() if d not in flagged]
            if not vals_all:
                continue
            base = {"mean": sum(vals_all) / len(vals_all), "n": len(vals_all)}
            excl = ({"mean": sum(cl) / len(cl), "n": len(cl)} if cl
                    else {"mean": None, "n": 0})
            only = ({"mean": sum(fl) / len(fl), "n": len(fl)} if fl else None)
            lab = labels.get("||".join(key), model)
            # Mann-Whitney U: are flagged documents scored differently from
            # clean ones under this run? That is exactly the tested difference,
            # so the dagger sits on the (w/o − w/) delta.
            p = None
            if fl and cl:
                _u, p = mann_whitney_u(fl, cl)
            mark = sig_mark(p, alpha)
            delta = ((excl["mean"] - only["mean"])
                     if (only and excl["mean"] is not None
                         and only["mean"] is not None) else None)
            shift = ((excl["mean"] - base["mean"])
                     if excl["mean"] is not None else None)
            rows_out.append({
                "label": lab + (f" {mark}" if mark else ""),
                "x0": base["mean"], "x1": excl["mean"],
                "x2": only["mean"] if only else None,
                "hover": [f"{lab}<br>all documents: {base['mean']:.3f} (n={base['n']})",
                          f"{lab}<br>without flagged: "
                          + (f"{excl['mean']:.3f} (n={excl['n']})"
                             if excl["mean"] is not None else "—")
                          + (f"<br>reported score moves {shift:+.3f}"
                             if shift is not None else "")
                          + f"<br>with flagged: "
                          + (f"{only['mean']:.3f} (n={only['n']})"
                             if only and only["mean"] is not None else "—")
                          + f"<br>MWU w/o vs w/: p={p_str(p)} {mark}",
                          f"{lab}<br>flagged documents only: "
                          + (f"{only['mean']:.3f} (n={only['n']})"
                             if only and only["mean"] is not None else "—")],
            })
            cells = [value_cell(base["mean"], base["n"]),
                     value_cell(excl["mean"], excl["n"] or None),
                     value_cell(only["mean"] if only else None,
                                only["n"] if only else None),
                     value_cell(delta, None, signed=True, mark=mark)]
            table_rows.append([lab] + [c[0] for c in cells])
            tex_rows.append([lab] + [c[1] for c in cells])

        mlab = metric_label(measure, k, prmu)
        crit_txt = []
        if "lang" in crit:
            crit_txt.append(f"language mismatch ({len(lang_docs)} docs)")
        if "sup_leak" in crit:
            crit_txt.append(f"similarity ≥ {thr:.2f} to the model's own "
                            "training data")
        if "any_leak" in crit:
            crit_txt.append(f"similarity ≥ {thr:.2f} to any document "
                            f"({len(any_leak)} docs)")
        spec = {
            "kind": "dumbbell", "size": "2col",
            "rows": rows_out,
            # the three points are a verdict, not a palette: gold = what you
            # would report, green = the clean subset, red = the documents the
            # criteria flagged. Distance between red and green is the bias.
            "points": [
                {"key": "x0", "name": "all documents", "color": "#C9971C"},
                {"key": "x1", "name": "without flagged", "color": "#0f7a3d"},
                {"key": "x2", "name": "flagged only", "color": "#d03b3b"},
            ],
            "xlabel": mlab, "name": f"quality-impact-{ds}",
            "caption": (f"Impact of data-quality filtering on {mlab} for {ds}: "
                        f"score over all evaluated documents vs. excluding "
                        f"documents flagged by {'; '.join(crit_txt) or 'no criterion'}. "
                        "The dagger marks a significant difference between the "
                        "flagged and the clean documents themselves (two-sided "
                        "Mann–Whitney U on per-document scores; "
                        + sig_caption(alpha) + "). "
                        + metric_caption(measure, k, prmu_sel, ann_choice)),
        }
        headers = ["Run", "All", "w/o flag", "w/ flag", "Δ (w/o − w/)"]
        spec["table"] = {"headers": headers, "rows": tex_rows,
                         "label": f"quality-{ds}"}
        table = ui.table(headers, table_rows, num_cols={1, 2, 3, 4})
        return to_plotly(spec), spec, spec["caption"], table
