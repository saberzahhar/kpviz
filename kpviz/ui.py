"""Shared Dash building blocks (cards, tiles, tables, export bar)."""
from __future__ import annotations

from dash import dcc, html

from .util import fmt_num


def card(children, title: str | None = None, **kw):
    kids = ([html.Div(title, className="card-title")] if title else []) + (
        children if isinstance(children, list) else [children])
    return html.Div(kids, className="card " + kw.pop("className", ""), **kw)


def stat_tile(label: str, value, sub: str | None = None):
    return html.Div([
        html.Div(label, className="stat-label"),
        html.Div(str(value), className="stat-value"),
        html.Div(sub or "", className="stat-sub"),
    ], className="stat-tile")


def kpi_row(tiles: list):
    return html.Div(tiles, className="kpi-row")


def chip(text, kind: str = ""):
    return html.Span(str(text), className=f"chip {kind}")


def badge(text, kind: str = "info"):
    return html.Span(str(text), className=f"badge {kind}")


# issue-tag chips: "key:value" -> a two-part chip colored by key
_TAG_KINDS = [("illegal parameter", "t-illegal"), ("missing", "t-missing"),
              ("incomplete", "t-incomplete"), ("unresolved", "t-unresolved"),
              ("lang_mismatch", "t-lang")]


# what each issue tag means and what KPViz did about it (chip tooltip)
TAG_HELP = {
    "duplicate_docs": "the run predicts some documents more than once; the "
                      "first line (lowest batch, then byte offset) is scored, "
                      "the others ignored",
    "duplicate_doc_ids": "the collection repeats document ids; the first "
                         "line is kept, the others ignored",
    "gold_duplicates": "gold keyphrases identical to an earlier one of the "
                       "same annotation set after stemming; counted once",
    "gold_empty": "gold keyphrases with no word token; dropped (they can "
                  "never be matched)",
    "unscored": "predicted documents with no stored gold (a training split, "
                "or no annotation); they enter no score",
    "unresolved_ids": "predicted document ids absent from the collection",
    "incomplete": "share of the evaluation split this run predicts; its "
                  "means are over the predicted documents only",
    "multi_split": "the run predicts documents of several splits",
    "malformed_lines": "lines that are not valid JSON (skipped)",
    "missing_id": "lines without an _id",
}


def tag_chip(tag: str):
    key, _, value = str(tag).partition(":")
    kind = next((cls for prefix, cls in _TAG_KINDS
                 if key.startswith(prefix)), "t-other")
    parts = [html.Span(key, className="tag-k")]
    if value:
        parts.append(html.Span(value, className="tag-v"))
    help_ = TAG_HELP.get(key)
    return html.Span(parts, className=f"tag {kind}",
                     title=f"{tag} — {help_}" if help_ else tag)


# ---- metadata chips ------------------------------------------------------
# Card metadata reads as "key | value": the key carries the darker weight of
# the pair's hue, the value the lighter one. One hue per *kind* of fact, held
# across Datasets, Models and Architectures, so "lang" is the same grey
# everywhere and a domain never looks like a language.
META_TONES = {
    "domain": "m-blue", "sub-domain": "m-blue-soft",
    "lang": "m-grey", "section": "m-teal", "annotation": "m-violet",
    "taxonomy": "m-slate", "metadata": "m-slate", "split": "m-slate",
    "family": "m-indigo", "backend": "m-amber", "can": "m-green",
    "params": "m-grey", "trained on": "m-rose", "cutoff": "m-slate",
    "kind": "m-amber", "cpu": "m-teal", "gpu": "m-indigo", "ram": "m-grey",
    "os": "m-slate", "python": "m-slate", "pkg": "m-grey",
    "api": "m-amber", "provider": "m-rose", "variable": "m-violet",
    "license": "m-slate", "context": "m-teal",
}


def meta_chip(key: str, value=None, tone: str | None = None, title=None):
    tone = tone or META_TONES.get(key, "m-slate")
    kids = [html.Span(str(key), className="mtag-k")]
    if value is not None and str(value) != "":
        kids.append(html.Span(str(value), className="mtag-v"))
    return html.Span(kids, className=f"mtag {tone}",
                     title=title or (f"{key}: {value}" if value is not None
                                     else str(key)))


def meta_row(chips: list):
    return html.Div([c for c in chips if c is not None], className="mtag-row")


class Wide:
    """A table cell that fills the rest of its row: one sentence ("no
    window: nothing to split") instead of a run of dashes."""

    def __init__(self, child):
        self.child = child


def table(headers: list, rows: list[list], num_cols: set[int] | None = None,
          row_ids: list | None = None, table_id: str | None = None,
          nowrap_cols: set[int] | None = None):
    num_cols = num_cols or set()
    nowrap = nowrap_cols or set()
    head = html.Thead(html.Tr([
        html.Th(h, className="num" if i in num_cols else "")
        for i, h in enumerate(headers)]))
    body_rows = []
    clickable = row_ids is not None and table_id is not None
    for ri, row in enumerate(rows):
        cells = list(row)
        if clickable and cells:
            # the first cell holds a real button (keyboard, screen readers);
            # the row stays a table row, and a click anywhere on it presses
            # that button (assets/keys.js)
            cells[0] = html.Button(cells[0], className="row-open", n_clicks=0,
                                   id={"type": f"{table_id}-row",
                                       "key": str(row_ids[ri])})
        tds = []
        for ci, c in enumerate(cells):
            if isinstance(c, Wide):
                tds.append(html.Td(c.child, colSpan=len(headers) - ci,
                                   className="wide"))
                break
            tds.append(html.Td(c, className=("num" if ci in num_cols else "")
                               + (" nowrap" if ci in nowrap else "")))
        body_rows.append(html.Tr(tds, className="row-click" if clickable else None))
    # headers never wrap (they read as one phrase), so a wide table scrolls
    # inside its card instead of stretching the page
    return html.Div(html.Table([head, html.Tbody(body_rows)],
                               className="kp-table"), className="table-wrap")


def control(label: str, component, width: int | None = None):
    """A labelled control. The label is a real <label> bound to the input
    when the component has a plain id (screen readers announce it)."""
    style = {"minWidth": f"{width}px"} if width else {}
    cid = getattr(component, "id", None)
    lab = (html.Label(label, htmlFor=cid, className="control-label")
           if isinstance(cid, str) else
           html.Div(label, className="control-label"))
    return html.Div([lab, component], className="control", style=style)


def filter_row(controls: list):
    return html.Div(controls, className="filter-row")


def section(title: str, note: str | None = None):
    out = [html.H3(title, className="section-title")]
    if note:
        out.append(html.Div(note, className="section-note"))
    return out


def loading(child):
    """Spinner overlay for real work only: shown after 400 ms, over the
    previous content (kept visible, dimmed) rather than instead of it."""
    return dcc.Loading(child, delay_show=400, type="dot", color="#2a78d6",
                       overlay_style={"visibility": "visible",
                                      "opacity": 0.55})


def graph(id, figure=None, height: int = 420, config: dict | None = None,
          grow: bool = False):
    """A graph `height` px tall; with grow=True that is a minimum, and a
    figure that sets its own height (a dumbbell with 22 runs) gets it."""
    from .figures import plotly_config
    return dcc.Graph(id=id, figure=figure or {},
                     config=config or plotly_config(),
                     style=({"minHeight": f"{height}px"} if grow
                            else {"height": f"{height}px"}))


VIEWS = [("interactive", "Interactive"), ("paper", "Paper"), ("both", "Both")]


def segmented(id, options, value, persist: bool = True):
    """A small segmented control (RadioItems drawn as one row of buttons),
    remembered per browser."""
    return dcc.RadioItems(id=id, value=value, inline=True,
                          options=[{"label": lab, "value": v} for v, lab in options],
                          className="segmented seg-sm",
                          **({"persistence": True, "persistence_type": "local"}
                             if persist else {}))


def fig_controls(key: str, formats: list | None = None,
                 default: str | None = None):
    """Top-right of a figure card: the figure's format (when it has more
    than one) and the view — interactive, as printed (Paper), or both side
    by side."""
    kids = []
    if formats and len(formats) > 1:
        kids.append(html.Div(
            segmented({"type": "fig-format", "rq": key}, formats,
                      default or formats[0][0]),
            className="seg-group", role="group", title="Figure format",
            **{"aria-label": "Figure format"}))
    kids.append(html.Div(
        segmented({"type": "fig-view", "rq": key}, VIEWS, "interactive"),
        className="seg-group", role="group",
        title="Interactive figure, the figure as printed, or both",
        **{"aria-label": "View"}))
    return html.Div(kids, className="fig-controls")


def figure_frame(key: str, screen):
    """The interactive figure and, beside or instead of it, the figure as it
    will print (appfactory renders it when the view asks for it, and while
    a new rendering is on its way the previous one stays, faded)."""
    paper = dcc.Loading(
        html.Div(html.Div("Typesetting the figure…", className="muted small"),
                 id={"type": "exp-preview", "rq": key}, className="paper-view"),
        delay_show=0, type="dot", color="#2a78d6",
        overlay_style={"visibility": "visible", "opacity": 0.35})
    return html.Div([
        dcc.Store(id={"type": "exp-want", "rq": key}),
        html.Div(screen, className="fig-screen"),
        html.Div(paper, className="fig-paper"),
    ], id={"type": "fig-frame", "rq": key}, className="fig-frame view-interactive")


def export_bar(rq: str, spec: dict | None = None, caption: str | None = None):
    """One Export menu per figure, folded: open it, then one click on the
    format. Inside: downloads (PDF, PNG, PGF, TeX, everything as a zip),
    the LaTeX figure and table to copy, the paper and layout options, and
    the caption. A status line answers every click at once.

    Exports use the figure as computed (the publication specification),
    not the on-screen zoom or hidden legend entries. The caption box is the
    generated caption until someone edits it; an edited caption is never
    overwritten by a recomputed figure (appfactory keeps it and offers the
    new one). The spec is written to {'type':'fig-spec','rq':rq} by the
    workbench callback — or given here for a figure drawn once."""
    clip_fig = clip_tab = hint = ""
    if spec:
        from .export import snippets
        clip_fig, clip_tab, hint = snippets(spec)

    def b(what, label, primary=False, title=None):
        return html.Button(label, id={"type": "exp-btn", "rq": rq, "what": what},
                           className="btn small" + (" primary" if primary else ""),
                           n_clicks=0, title=title)

    def opt(kind, label, options, value, width):
        # remembered per browser: a paper keeps its venue across sessions
        return html.Div([
            html.Span(label, className="exp-opt-label"),
            dcc.Dropdown(id={"type": f"exp-{kind}", "rq": rq}, options=options,
                         value=value, clearable=False, searchable=False,
                         persistence=True, persistence_type="local",
                         className="dash-dropdown exp-dd")],
            className="exp-opt", style={"minWidth": f"{width}px"})

    def clip(kind, title):
        # the label is drawn by CSS on the clipboard itself (::after), so a
        # click anywhere on the button copies
        return dcc.Clipboard(title=title, id={"type": f"exp-clip-{kind}", "rq": rq},
                             content=clip_fig if kind == "fig" else clip_tab,
                             className=f"btn small clip-btn clip-{kind}",
                             style={"display": "inline-flex"})

    from .figures import VENUES
    return html.Div([
        # data only when there is a spec: an explicit data=None makes Dash
        # fire the export callbacks of every workbench at page load
        dcc.Store(id={"type": "fig-spec", "rq": rq},
                  **({"data": spec} if spec else {})),
        dcc.Store(id={"type": "cap-base", "rq": rq}),
        dcc.Download(id={"type": "exp-dl", "rq": rq}),
        html.Details([
            html.Summary("Export", className="exp-summary"),
            html.Div([
                html.Div([
                    html.Span("Download", className="exp-group-label"),
                    b("pdf", "PDF", primary=True,
                      title="vector PDF at the paper's size, from the figure "
                            "as computed (not the on-screen zoom)"),
                    b("png", "PNG", title="300 dpi PNG"),
                    b("pgf", "PGF", title="PGF picture, typeset by your paper"),
                    b("tex", "TeX", title="the LaTeX figure environment (and "
                                          "the table) as a .tex file"),
                    b("zip", "All (.zip)",
                      title="PDF + PGF + PNG + .tex + data + provenance"),
                    html.Span("Copy", className="exp-group-label"),
                    clip("fig", "copy the LaTeX figure environment"),
                    clip("tab", "copy the booktabs table"),
                ], className="export-bar"),
                html.Span(id={"type": "exp-status", "rq": rq}, className="exp-status",
                          role="status", **{"aria-live": "polite"}),
                html.Div([
                    opt("venue", "Paper", [{"label": v["label"], "value": k}
                                           for k, v in VENUES.items()], "generic", 250),
                    opt("span", "Width", [{"label": "as designed", "value": "auto"},
                                          {"label": "one column", "value": "col"},
                                          {"label": "full text width", "value": "full"}],
                        "auto", 150),
                    opt("height", "Height", [{"label": "compact", "value": "compact"},
                                             {"label": "standard", "value": "std"},
                                             {"label": "tall", "value": "tall"}],
                        "std", 115),
                    opt("legend", "Legend", [{"label": "auto", "value": "auto"},
                                             {"label": "above", "value": "top"},
                                             {"label": "right", "value": "right"},
                                             {"label": "none (in caption)", "value": "none"}],
                        "auto", 150),
                    opt("cells", "Table cells", [
                        {"label": "value", "value": "value"},
                        {"label": "value [CI]", "value": "ci"},
                        {"label": "value [CI] (n)", "value": "ci_n"}], "ci", 140),
                ], className="exp-opts"),
                html.Div([
                    html.Span("Caption", className="exp-opt-label"),
                    html.Span(id={"type": "cap-note", "rq": rq}, className="cap-note",
                              role="status", **{"aria-live": "polite"}),
                    html.Button("Restore the generated caption",
                                id={"type": "cap-reset", "rq": rq},
                                className="btn small ghost", n_clicks=0),
                ], className="cap-head"),
                caption_editor(rq, caption or ""),
                html.Div(hint, id={"type": "exp-hint", "rq": rq}, className="hint"),
            ], className="exp-body"),
        ], className="exp-menu"),
    ], id={"type": "exp-wrap", "rq": rq}, className="exp-wrap")


def exportable(key: str, spec: dict, title: str | None = None,
               height: int = 420, variants: dict | None = None,
               formats: list | None = None, note=None, lead=None, **card_kw):
    """A card holding a figure drawn once (not by a workbench callback):
    its title and controls, an optional scope line (`lead`), the figure
    (interactive, as printed, or both), an optional note, and its Export
    menu.

    `variants` maps a format to its spec ({"overlay": spec, "group": …});
    `formats` lists (format, label) in display order, the first being the
    default. Switching format happens in the browser (every variant is
    computed with the page), and the export follows the format shown."""
    from .figures import to_plotly

    def fig(sp):
        f = to_plotly(sp)
        if sp.get("height"):
            f.update_layout(height=sp["height"])
        return f.to_plotly_json()
    store = None
    if variants and formats:
        spec = variants[formats[0][0]]
        drawn = {k: {"figure": fig(sp), "spec": sp} for k, sp in variants.items()}
        store = dcc.Store(id={"type": "fig-variants", "rq": key}, data=drawn)
        first = drawn[formats[0][0]]["figure"]
    else:
        first = fig(spec)
    screen = graph({"type": "fig-graph", "rq": key}, first,
                   spec.get("height") or height, grow=True)
    head = html.Div([html.Div(title or "", className="card-title"),
                     fig_controls(key, formats if variants else None)],
                    className="card-head")
    return card([store, head, lead, figure_frame(key, screen), note,
                 export_bar(key, spec, caption=spec.get("caption", ""))],
                **card_kw)


def caption_editor(rq: str, initial: str = ""):
    """Editable caption. The LaTeX snippets are rebuilt when it loses focus
    (n_blur), not on every keystroke; downloads read its current value."""
    return html.Div(dcc.Textarea(
        id={"type": "caption", "rq": rq}, value=initial,
        placeholder="Caption used in exports…"),
        className="caption-box")


def more(children, label: str = "Filters & settings", open_: bool = False):
    """Secondary controls, folded: the page shows what most people change."""
    return html.Details([html.Summary(label),
                         html.Div(children, className="more-body")],
                        className="more-opts", open=open_)


def fold(child, n_rows: int, label: str, limit: int = 8):
    """A long table (or list) folded behind a one-line summary; short ones
    are shown as they are."""
    if n_rows <= limit:
        return child
    return html.Details([html.Summary(label), child], className="fold")


def empty_state(msg: str):
    return card(html.Div([
        html.Div("◌", style={"fontSize": "34px", "color": "var(--baseline)"}),
        html.Div(msg, className="muted", style={"marginTop": "6px"}),
    ], style={"textAlign": "center", "padding": "30px 0"}))


def pct(x, digits=0):
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=3):
    return fmt_num(x, digits)
