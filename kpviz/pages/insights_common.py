"""Shared machinery for the five research-question workbenches."""
from __future__ import annotations


from dash import dcc, html
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..metrics import PRMU, memo, metric_label
from ..naming import (group_key, natural_key, parse_group_key, run_labels,
                      run_rows)
from ..stats import ALPHAS, ALPHA_DEFAULT
from ..util import stable_hash


def vis(rq: str) -> str:
    """Id of the store that is True while this workbench is on screen."""
    return f"vis-{rq}"


def gate(visible, inputs, last_sig) -> str:
    """Compute only what is on screen, and only when something changed.

    Returns the signature of this view (its inputs + the catalog version);
    raises PreventUpdate when the workbench is hidden, or when it is shown
    again with nothing changed. The signature lives in a per-client store,
    so a second tab or a reload still renders."""
    if not visible:
        raise PreventUpdate
    sig = stable_hash([inputs, db.scan_version()])
    if sig == last_sig:
        raise PreventUpdate
    return sig


def alpha_of(slider_value) -> float:
    """Significance level from the shared Insights slider (index -> alpha)."""
    try:
        return ALPHAS[int(slider_value)]
    except (TypeError, ValueError, IndexError):
        return ALPHA_DEFAULT


def value_cell(value, n=None, digits: int = 3, signed: bool = False,
               mark: str = "") -> tuple:
    """(html cell, plain-text cell) for "0.123 (n=140)" / "+0.045†".

    The sample size belongs next to the number it qualifies, not in a column of
    its own, and a significance mark is a superscript. Returns both renderings
    so the HTML table and the exported LaTeX row stay in step."""
    if value is None:
        return html.Span("—", className="muted"), "—"
    num = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    kids = [num]
    txt = num
    if mark:
        kids.append(html.Sup(mark))
        txt += f" {mark}"
    if n is not None:
        kids.append(html.Span(f" (n={n})", className="muted"))
        txt += f" (n={n})"
    return html.Span(kids), txt


def datasets_with_runs() -> list[str]:
    """Datasets that have runs AND a document collection (runs whose dataset
    was removed stay visible on Overview as missing:dataset tags)."""
    return memo("datasets_with_runs", lambda: [r[0] for r in db.q(
        """SELECT DISTINCT r.dataset FROM runs r
           WHERE EXISTS (SELECT 1 FROM documents d WHERE d.dataset = r.dataset)
           ORDER BY 1""")])


def _ann_keys(ds: str) -> list[str]:
    return memo(("ann_keys", ds), lambda: [r[0] for r in db.q(
        "SELECT DISTINCT ann_key FROM gold_agg WHERE dataset=? ORDER BY 1", ds)])


def ann_for(ds: str) -> str | None:
    anns = _ann_keys(ds)
    if not anns:
        return None
    return "@combined" if "@combined" in anns else anns[0]


def ann_options(datasets: list[str]) -> list[str]:
    if not datasets:
        return []
    common = None
    for ds in datasets:
        anns = set(_ann_keys(ds))
        common = anns if common is None else (common & anns)
    return ["auto"] + sorted(common or [])


def register_dataset_refresh(app, dropdown_id: str, multi: bool,
                             prefer: list[str] | None = None,
                             rq: str | None = None):
    """Keep an RQ dataset selector in sync with the catalog — when the
    workbench is on screen (a hidden one catches up when shown)."""
    from dash import Input, Output, State
    rq = rq or dropdown_id.split("-")[0]

    @app.callback(Output(dropdown_id, "options"),
                  Output(dropdown_id, "value"),
                  Input(vis(rq), "data"),
                  Input("catalog-version", "data"),
                  State(dropdown_id, "options"),
                  State(dropdown_id, "value"),
                  prevent_initial_call=True)
    def _refresh(visible, _v, cur_opts, current):
        if not visible:
            raise PreventUpdate
        ds = datasets_with_runs()
        if cur_opts == ds and current:
            raise PreventUpdate
        if multi:
            kept = [d for d in (current or []) if d in ds]
            if not kept:
                kept = [d for d in (prefer or []) if d in ds] or ds[:3]
            return ds, kept
        value = current if current in ds else (
            next((d for d in (prefer or []) if d in ds), None) or
            (ds[0] if ds else None))
        return ds, value


def resolve_ann(ds: str, choice: str | None) -> str | None:
    if not choice or choice == "auto":
        return ann_for(ds)
    return choice


def model_options(datasets: list[str]):
    """Models that have runs on the selected datasets."""
    def build():
        idx = scanner.cards()
        models = sorted({r["model"] for r in run_rows(datasets or None)})
        return [{"label": idx.model(m).name, "value": m} for m in models]
    return memo(("model_options", tuple(datasets or ())), build)


def run_options(datasets: list[str], require_all: bool = False,
                models: list[str] | None = None):
    """Options for a run picker — models first, then runs (memoised per
    catalog version: the update callback and the picker chain both ask)."""
    return memo(("run_options", tuple(datasets or ()), require_all,
                 tuple(models or ())),
                lambda: _run_options(datasets, require_all, models), 65536)


def _run_options(datasets: list[str], require_all: bool = False,
                 models: list[str] | None = None):
    """require_all -> only (model, arch, run_id) triples present on every
    selected dataset. models -> restrict to the selected models."""
    rows = run_rows(datasets or None)
    if models:
        mset = set(models)
        rows = [r for r in rows if r["model"] in mset]
    idx = scanner.cards()
    labels = run_labels(idx, rows)
    seen: dict[str, set] = {}
    for r in rows:
        seen.setdefault(group_key(r["model"], r["arch"], r["run_id"]),
                        set()).add(r["dataset"])
    opts = []
    for k in sorted(seen, key=lambda k: natural_key(labels.get(k, k))):
        n = len(seen[k])
        ok = (n == len(datasets)) if (require_all and datasets) else True
        lab = labels.get(k, k)
        if datasets and n < len(datasets):
            lab += f"  — on {n}/{len(datasets)} datasets"
        opts.append({"label": lab, "value": k, "disabled": require_all and not ok})
    return opts


def register_model_run_chain(app, prefix: str, multi_ds: bool,
                             require_all: bool = False):
    """Wire dataset -> models -> runs selection for one workbench (only
    while it is on screen)."""
    from dash import Input, Output, State

    @app.callback(Output(f"{prefix}-models", "options"),
                  Output(f"{prefix}-models", "value"),
                  Input(vis(prefix), "data"),
                  Input(f"{prefix}-ds", "value"),
                  State(f"{prefix}-models", "value"),
                  prevent_initial_call=True)
    def _models(visible, ds_sel, current):
        if not visible:
            raise PreventUpdate
        ds = (ds_sel or []) if multi_ds else ([ds_sel] if ds_sel else [])
        opts = model_options(ds)
        vals = {o["value"] for o in opts}
        kept = [m for m in (current or []) if m in vals]
        return opts, kept  # empty selection = all models

    @app.callback(Output(f"{prefix}-runs", "options"),
                  Output(f"{prefix}-runs", "value"),
                  Input(vis(prefix), "data"),
                  Input(f"{prefix}-ds", "value"),
                  Input(f"{prefix}-models", "value"),
                  State(f"{prefix}-runs", "value"),
                  prevent_initial_call=True)
    def _runs(visible, ds_sel, models_sel, current):
        if not visible:
            raise PreventUpdate
        ds = (ds_sel or []) if multi_ds else ([ds_sel] if ds_sel else [])
        opts = run_options(ds, require_all=require_all,
                           models=models_sel or None)
        vals = {o["value"] for o in opts if not o.get("disabled")}
        kept = [v for v in (current or []) if v in vals]
        return opts, kept  # empty selection = all listed runs


def effective_runs(datasets: list[str], models_sel, runs_sel,
                   require_all: bool = False) -> list[str]:
    """Selected runs, or every eligible run of the selected models."""
    opts = run_options(datasets, require_all=require_all,
                       models=models_sel or None)
    eligible = [o["value"] for o in opts if not o.get("disabled")]
    chosen = [v for v in (runs_sel or []) if v in eligible]
    return chosen or eligible


def metric_controls(prefix: str, default_k: str = "O",
                    include_prmu: bool = True):
    out = [
        ui.control("Measure", dcc.Dropdown(
            id=f"{prefix}-measure",
            options=[{"label": m.upper(), "value": m} for m in ("f1", "p", "r")],
            value="f1", clearable=False, className="dash-dropdown"), 110),
        ui.control("@k", dcc.Dropdown(
            id=f"{prefix}-k",
            options=[{"label": f"@{k}", "value": k} for k in ("5", "10", "O", "M")],
            value=default_k, clearable=False, className="dash-dropdown"), 100),
    ]
    if include_prmu:
        out.append(ui.control("PRMU classes", dcc.Checklist(
            id=f"{prefix}-prmu",
            options=[{"label": f" {p}", "value": p} for p in PRMU],
            value=list(PRMU), inline=True, className="kp-check kp-inline"), 210))
    out.append(ui.control("Annotation", dcc.Dropdown(
        id=f"{prefix}-ann", options=["auto"], value="auto",
        clearable=False, className="dash-dropdown"), 150))
    return out


def models_control(prefix: str, width: int = 280):
    return ui.control("Models", dcc.Dropdown(
        id=f"{prefix}-models", multi=True, placeholder="all models",
        className="dash-dropdown"), width)


def runs_control(prefix: str, width: int = 380,
                 placeholder: str = "all runs of the selected models"):
    return ui.control("Runs", dcc.Dropdown(
        id=f"{prefix}-runs", multi=True, placeholder=placeholder,
        className="dash-dropdown"), width)


def gold_phrase(ann_choice: str | None, datasets: list[str]) -> str:
    """The gold actually used, per dataset — never assumed.

    "auto" resolves to the union of every annotation set when a dataset has
    several (@combined) and to its only set otherwise, so the caption names
    what was resolved: "author gold (kp20k, kpbiomed); editor gold (kptimes)"
    or "author+reader combined gold (semeval2010)"."""
    groups: dict[str, list[str]] = {}
    for ds in datasets or []:
        ann = resolve_ann(ds, ann_choice)
        if ann is None:
            continue
        if ann == "@combined":
            parts = [a for a in _ann_keys(ds) if a != "@combined"]
            name = "+".join(parts) + " combined gold"
        else:
            name = f"{ann} gold"
        groups.setdefault(name, []).append(ds)
    if not groups:
        return "gold keyphrases"
    return "; ".join(f"{g} ({', '.join(d)})" for g, d in groups.items())


CONVENTIONS = ("predictions and gold lowercased, spaCy-tokenised, "
               "Snowball (Porter2) stemmed and de-duplicated keeping rank "
               "order; P@k = tp / min(k, #predictions) (no padding); "
               "documents with no gold after filtering excluded")


def metric_caption(measure: str, k: str, prmu: list[str], ann: str | None,
                   datasets: list[str] | None = None) -> str:
    lab = metric_label(measure, k, prmu)
    gold = gold_phrase(ann, datasets or [])
    extra = ""
    if prmu and sorted(prmu) != sorted(PRMU):
        extra = (", restricted to the "
                 + "/".join({"P": "Present", "R": "Reordered", "M": "Mixed",
                             "U": "Unseen"}[p] for p in prmu)
                 + " classes (in-order PRMU)")
    return (f"{lab}, macro-averaged over documents, against {gold}{extra}; "
            f"{CONVENTIONS}.")


def prmu_arg(prmu_sel: list[str]):
    return None if (not prmu_sel or sorted(prmu_sel) == sorted(PRMU)) else prmu_sel


def selected_runs(values: list[str]) -> list[tuple[str, str, str]]:
    return [parse_group_key(v) for v in (values or [])]


def rq_header(question: str, method: str):
    return html.Div([
        html.Div(question, className="rq-question"),
        html.Div(method, className="rq-method"),
    ])


def figure_block(rq: str, height: int = 470, with_table: bool = True):
    """Graph + caption editor + export bar + optional table container.

    The graph and table sit under a loading overlay that appears only after
    300 ms (fast updates never flash a spinner) and keeps the previous
    figure readable underneath; the fig-sig store remembers, per browser
    tab, which view is already on screen."""
    kids = [
        dcc.Store(id={"type": "fig-sig", "rq": rq}),
        ui.loading(ui.graph({"type": "rq-graph", "rq": rq}, height=height)),
        ui.caption_editor(rq),
        ui.export_bar(rq),
    ]
    if with_table:
        kids.append(ui.loading(html.Div(id={"type": "rq-table", "rq": rq},
                                        style={"marginTop": "14px"})))
    return ui.card(kids)
