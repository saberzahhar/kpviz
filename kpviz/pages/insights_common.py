"""Shared machinery for the five research-question workbenches."""
from __future__ import annotations


from dash import dcc, html
from dash.exceptions import PreventUpdate

from .. import db, scanner, ui
from ..metrics import PRMU, memo, metric_label
from ..naming import (group_key, natural_key, parse_group_key, run_labels,
                      run_rows)
from ..stats import (ALPHAS, ALPHA_DEFAULT, RESAMPLES, StatsCfg,
                     fmt_effect, p_str)
from ..util import stable_hash


def vis(rq: str) -> str:
    """Id of the store that is True while this workbench is on screen. Read
    as a State: hiding a page writes False here without waking anything."""
    return f"vis-{rq}"


def shown(rq: str) -> str:
    """Id of the counter the router bumps each time this workbench is
    *shown* — the Input that (re)renders it. Nothing listens to hiding."""
    return f"shown-{rq}"


# scan steps that delete or insert derived rows: while one runs, the tables
# a workbench reads are half-written (a purged collection, a run being
# re-matched), so nothing is computed — or cached — from them
_WRITE_STEPS = {"documents", "inferences", "keyphrases", "scores", "finalize"}


def catalog_unavailable() -> str | None:
    """Why analyses are withheld right now, or None.

    "scanning": a scan is writing the store. "incomplete": the last scan
    failed or was cancelled after it had started deleting — the store may
    mix old and new rows until a scan completes. Workbenches and exports
    stay on what was shown before, and the Insights banner says why."""
    snap = scanner.STATE.snapshot()
    if snap["running"] and any(s["status"] == "running" and s["key"] in _WRITE_STEPS
                               for s in snap["steps"]):
        return "scanning"
    if db.scan_version() and memo("last_scan_ok",
                                  lambda: bool(db.kv_get("last_scan_ok", True)),
                                  64) is False:
        return "incomplete"
    return None


def gate(visible, inputs, last_sig) -> str:
    """Compute only what is on screen, only when something changed, and
    only from a coherent catalog.

    Returns the signature of this view (its inputs + the catalog version);
    raises PreventUpdate when the workbench is hidden, when a scan is
    writing the store (or the last one left it incomplete), or when it is
    shown again with nothing changed. The signature lives in a per-client
    store, so a second tab or a reload still renders."""
    if not visible or catalog_unavailable():
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
               mark: str = "", ci: tuple | None = None) -> tuple:
    """(html cell, LaTeX cell) for "0.123 [0.110, 0.137] (n=140)" / "+0.045†".

    The sample size and the interval belong next to the number they qualify,
    not in columns of their own, and a significance mark is a superscript.
    The LaTeX cell is structured ({"v", "lo", "hi", "n", "mark", ...}), so
    the exported table can choose how much of it to print (export options)
    while the HTML table and the LaTeX row stay in step."""
    if value is None:
        return html.Span("—", className="muted"), "—"
    num = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    kids = [num]
    if mark:
        kids.append(html.Sup(mark))
    lo, hi = ci if ci else (None, None)
    has_ci = lo is not None and hi is not None
    if has_ci:
        f = (lambda v: f"{v:+.{digits}f}") if signed else (lambda v: f"{v:.{digits}f}")
        kids.append(html.Span(f" [{f(lo)}, {f(hi)}]", className="ci"))
    title = None
    if n is not None:
        # with an interval the cell is already two numbers wide: n moves to
        # the tooltip (and stays in the exported table if asked for)
        if has_ci:
            title = f"n = {n} documents"
        else:
            kids.append(html.Span(f" (n={n})", className="muted"))
    return html.Span(kids, title=title, className="vcell"), {"v": value, "lo": lo, "hi": hi, "n": n,
                             "mark": mark, "signed": signed, "digits": digits}


# ---- statistics settings (Insights → Statistics) ---------------------------

STATS_IDS = ("ins-alpha", "ins-family", "ins-adjust", "ins-ci", "ins-resamples")


def stats_inputs():
    """Every workbench that tests something listens to all five settings."""
    from dash import Input
    return [Input(i, "value") for i in STATS_IDS]


def stats_cfg(alpha_ix, family=None, adjust=None, ci=None, resamples=None) -> StatsCfg:
    return StatsCfg(alpha=alpha_of(alpha_ix), family=family or "rank",
                    adjust=adjust or "holm", ci=ci or "t",
                    resamples=resamples or RESAMPLES)


_ADJ_SHORT = {"holm": "Holm", "bonferroni": "Bonf.", "bh": "BH"}


def p_cells(p, p_adj, cfg: StatsCfg, applicable: bool = True) -> list[tuple]:
    """[(html, tex)] for the raw p and — when a correction is on — the
    adjusted p that the dagger actually follows. `applicable=False` (no
    second condition to compare with) prints a dash, not "n<6"."""
    def one(v, strong=False):
        if not applicable:
            return html.Span("—", className="muted"), "—"
        txt = p_str(v, "n<6")
        node = html.B(txt) if (strong and v is not None and v < cfg.alpha) else txt
        return html.Span(node, className="" if v is not None else "muted"), txt
    out = [one(p, strong=cfg.adjust == "none")]
    if cfg.adjust != "none":
        out.append(one(p_adj, strong=True))
    return out


def p_headers(cfg: StatsCfg) -> list[str]:
    return ["p"] + ([f"p ({_ADJ_SHORT[cfg.adjust]})"]
                    if cfg.adjust != "none" else [])


def effect_cell(e) -> tuple:
    txt = fmt_effect(e)
    return (html.Span(txt, className="muted" if txt == "—" else ""), txt)


def stats_note(cfg: StatsCfg, design: str, n_tests: int) -> html.Div:
    """The methods sentence under a table, as it will read in the caption."""
    return html.Div(["Statistics: ", cfg.method_text(design, n_tests)
                     + (f"; {cfg.ci_text()}" if cfg.ci_text() else "") + "."],
                    className="muted small stats-note")


def datasets_with_runs() -> list[str]:
    """Datasets that have runs AND a document collection (runs whose dataset
    was removed stay visible on Overview as missing:dataset tags)."""
    return memo("datasets_with_runs", lambda: [r[0] for r in db.q(
        """SELECT DISTINCT r.dataset FROM runs r
           WHERE EXISTS (SELECT 1 FROM documents d WHERE d.dataset = r.dataset)
           ORDER BY 1""")])


def default_datasets(n: int) -> list[str]:
    """The n datasets most runs were evaluated on (in name order): a
    workbench opens on the slice with the most to compare, whatever the
    collections are called."""
    def build():
        return [r[0] for r in db.q(
            """SELECT r.dataset, count(DISTINCT (r.model, r.arch, r.run_id)) AS n
               FROM runs r
               WHERE EXISTS (SELECT 1 FROM documents d WHERE d.dataset = r.dataset)
               GROUP BY 1 ORDER BY n DESC, 1""")]
    ranked = memo("datasets_by_runs", build)
    return sorted(ranked[:n])


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
                             prefer=None,
                             rq: str | None = None):
    """Keep an RQ dataset selector in sync with the catalog — when the
    workbench is on screen (a hidden one catches up when shown)."""
    from dash import Input, Output, State
    rq = rq or dropdown_id.split("-")[0]

    @app.callback(Output(dropdown_id, "options"),
                  Output(dropdown_id, "value"),
                  State(vis(rq), "data"), Input(shown(rq), "data"),
                  Input("catalog-version", "data"),
                  State(dropdown_id, "options"),
                  State(dropdown_id, "value"),
                  prevent_initial_call=True)
    def _refresh(visible, _shown, _v, cur_opts, current):
        if not visible:
            raise PreventUpdate
        ds = datasets_with_runs()
        if cur_opts == ds and current:
            raise PreventUpdate
        pref = prefer() if callable(prefer) else prefer
        if multi:
            kept = [d for d in (current or []) if d in ds]
            if not kept:
                kept = ([d for d in (pref or []) if d in ds]
                        or default_datasets(3))
            return ds, kept
        value = current if current in ds else (
            next((d for d in (pref or []) if d in ds), None) or
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
                  State(vis(prefix), "data"), Input(shown(prefix), "data"),
                  Input(f"{prefix}-ds", "value"),
                  State(f"{prefix}-models", "value"),
                  prevent_initial_call=True)
    def _models(visible, _shown, ds_sel, current):
        if not visible:
            raise PreventUpdate
        ds = (ds_sel or []) if multi_ds else ([ds_sel] if ds_sel else [])
        opts = model_options(ds)
        vals = {o["value"] for o in opts}
        kept = [m for m in (current or []) if m in vals]
        return opts, kept  # empty selection = all models

    @app.callback(Output(f"{prefix}-runs", "options"),
                  Output(f"{prefix}-runs", "value"),
                  State(vis(prefix), "data"), Input(shown(prefix), "data"),
                  Input(f"{prefix}-ds", "value"),
                  Input(f"{prefix}-models", "value"),
                  State(f"{prefix}-runs", "value"),
                  prevent_initial_call=True)
    def _runs(visible, _shown, ds_sel, models_sel, current):
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


METRIC_OPTIONS = [{"label": f"{m.upper()}@{k}", "value": f"{m}|{k}"}
                  for m in ("f1", "p", "r") for k in ("5", "10", "O", "M")]


def split_metric(value) -> tuple[str, str]:
    """"f1|O" -> ("f1", "O"); anything unexpected -> F1@O."""
    m, _, k = str(value or "").partition("|")
    if m not in ("f1", "p", "r") or k not in ("5", "10", "O", "M"):
        return "f1", "O"
    return m, k


def metric_control(prefix: str, default_k: str = "O"):
    """One control for the measure and its cutoff (F1@5 … R@M)."""
    return ui.control("Metric", dcc.Dropdown(
        id=f"{prefix}-metric", options=METRIC_OPTIONS, value=f"f1|{default_k}",
        clearable=False, searchable=False, className="dash-dropdown"), 110)


def gold_controls(prefix: str, include_prmu: bool = True):
    """Which gold counts: PRMU classes and annotation set (secondary)."""
    out = []
    if include_prmu:
        out.append(ui.control("PRMU classes", dcc.Checklist(
            id=f"{prefix}-prmu",
            options=[{"label": f" {p}", "value": p} for p in PRMU],
            value=list(PRMU), inline=True, className="kp-check kp-inline"), 210))
    out.append(ui.control("Annotation", dcc.Dropdown(
        id=f"{prefix}-ann", options=["auto"], value="auto",
        clearable=False, className="dash-dropdown"), 150))
    return out


def systems_control(prefix: str, default: str = "model"):
    """One point per model (its best run on the selection) or every run."""
    return ui.control("Systems", dcc.RadioItems(
        id=f"{prefix}-unit", value=default, className="segmented",
        inline=True,
        options=[{"label": "one per model (best run)", "value": "model",
                  "title": "each model's best-scoring run on this selection"},
                 {"label": "every run", "value": "run"}]), 280)


def best_per_model(keys: list[tuple], score) -> list[tuple]:
    """Each model's best run by `score(key)` (None scores lose), in the
    order the models first appear."""
    best: dict[str, tuple] = {}
    for key in keys:
        v = score(key)
        cur = best.get(key[0])
        if cur is None or (v is not None and (cur[1] is None or v > cur[1])):
            best[key[0]] = (key, v)
    return [kv[0] for kv in best.values()]


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


CONVENTIONS = ("predictions and gold NFKC-normalised, lowercased, split "
               "into Unicode words and Snowball-stemmed (Porter2 for English); predictions de-duplicated keeping "
               "rank order, gold de-duplicated per annotation set; PRMU "
               "contiguous (Boudin & Gallina, 2021); "
               "P@k = tp / min(k, #predictions) (no padding); "
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
                 + " classes; every prediction still counts in P@k")
    return (f"{lab}, macro-averaged over documents, against {gold}{extra}; "
            f"{CONVENTIONS}.")


def prmu_arg(prmu_sel: list[str] | None):
    """None (no filter) for every class or an uninitialised control; the
    selection otherwise. An *empty* selection is not "all": callers check
    `prmu_empty` first and say so instead of scoring."""
    if prmu_sel is None or sorted(prmu_sel) == sorted(PRMU):
        return None
    return list(prmu_sel)


def prmu_empty(prmu_sel) -> bool:
    return prmu_sel is not None and len(prmu_sel) == 0


NO_PRMU = ("No keyphrase class is selected. Tick at least one of P, R, M, U "
           "under Filters & settings.")


def selected_runs(values: list[str]) -> list[tuple[str, str, str]]:
    return [parse_group_key(v) for v in (values or [])]


def rq_header(question: str, method: str):
    """The question in plain words and one sentence of method; details live
    in the caption and the Methods fold."""
    return html.Div([
        html.H2(question, className="rq-question"),
        html.P(method, className="rq-method"),
    ])


def figure_block(rq: str, height: int = 470, with_table: bool = True,
                 title: str | None = None, label: str | None = None):
    """Scope line + graph + export bar + optional table.

    The scope line above the graph names what the figure shows (metric,
    datasets, how many systems, the filters in force), so a screenshot or a
    projected figure identifies itself; with nothing to show it carries the
    reason and the graph steps aside. The graph and table sit under a
    loading overlay that appears only after 400 ms and keeps the previous
    figure readable underneath; the fig-sig store remembers, per browser
    tab, which view is already on screen."""
    kids = [
        dcc.Store(id={"type": "fig-sig", "rq": rq}),
        # a titled panel: title and controls, then the scope line; untitled:
        # the scope line itself sits beside the controls
        *([html.Div([html.H3(title, className="panel-title"), ui.fig_controls(rq)],
                    className="card-head"),
           html.Div(id={"type": "rq-head", "rq": rq}, className="rq-head")]
          if title else
          [html.Div([html.Div(id={"type": "rq-head", "rq": rq}, className="rq-head"),
                     ui.fig_controls(rq)], className="card-head card-head-scope")]),
        ui.figure_frame(rq, html.Div(
            ui.loading(ui.graph({"type": "rq-graph", "rq": rq},
                                height=height, grow=True)),
            id={"type": "rq-gwrap", "rq": rq}, role="figure",
            **{"aria-label": label or title or "figure"})),
        ui.export_bar(rq),
    ]
    if with_table:
        kids.append(ui.loading(html.Div(id={"type": "rq-table", "rq": rq},
                                        className="rq-table")))
    return ui.card([k for k in kids if k is not None])


def scope_line(lead: str, *parts) -> html.Div:
    """"F1@O · 3 datasets · 12 runs on all of them · author gold": what the
    figure below shows, from the result itself (never from the controls)."""
    rest = [p for p in parts if p]
    return html.Div([html.B(lead)] + [html.Span(" · " + str(p)) for p in rest],
                    className="scope")


def empty_result(msg: str, hint: str | None = None) -> html.Div:
    """Why there is no figure, and what to change — in place of the figure."""
    return html.Div([html.Div(msg, className="empty-msg"),
                     html.Div(hint, className="muted small") if hint else None],
                    className="empty-result", role="status")
