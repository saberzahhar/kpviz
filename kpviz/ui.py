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


def tag_chip(tag: str):
    key, _, value = str(tag).partition(":")
    kind = next((cls for prefix, cls in _TAG_KINDS
                 if key.startswith(prefix)), "t-other")
    parts = [html.Span(key, className="tag-k")]
    if value:
        parts.append(html.Span(value, className="tag-v"))
    return html.Span(parts, className=f"tag {kind}", title=tag)


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


def table(headers: list, rows: list[list], num_cols: set[int] | None = None,
          row_ids: list | None = None, table_id: str | None = None):
    num_cols = num_cols or set()
    head = html.Thead(html.Tr([
        html.Th(h, className="num" if i in num_cols else "")
        for i, h in enumerate(headers)]))
    body_rows = []
    for ri, row in enumerate(rows):
        tds = [html.Td(c if isinstance(c, (str, int, float)) or c is None
                       else c, className="num" if ci in num_cols else "")
               for ci, c in enumerate(row)]
        kw = {}
        if row_ids is not None and table_id is not None:
            kw = {"id": {"type": f"{table_id}-row", "key": str(row_ids[ri])},
                  "className": "row-click", "n_clicks": 0}
        body_rows.append(html.Tr(tds, **kw))
    # headers never wrap (they read as one phrase), so a wide table scrolls
    # inside its card instead of stretching the page
    return html.Div(html.Table([head, html.Tbody(body_rows)],
                               className="kp-table"), className="table-wrap")


def control(label: str, component, width: int | None = None):
    style = {"minWidth": f"{width}px"} if width else {}
    return html.Div([html.Div(label, className="control-label"), component],
                    className="control", style=style)


def filter_row(controls: list):
    return html.Div(controls, className="filter-row")


def section(title: str, note: str | None = None):
    out = [html.H3(title, className="section-title")]
    if note:
        out.append(html.Div(note, className="section-note"))
    return out


def graph(id, figure=None, height: int = 420, config: dict | None = None):
    from .figures import plotly_config
    return dcc.Graph(id=id, figure=figure or {},
                     config=config or plotly_config(),
                     style={"height": f"{height}px"})


def export_bar(rq: str):
    """One export bar per insight: stores + buttons + clipboards.

    The figure/table spec is written to {'type':'fig-spec','rq':rq} by the
    RQ callback; global callbacks in appfactory handle rendering/downloads."""
    def b(what, label, primary=False):
        return html.Button(label, id={"type": "exp-btn", "rq": rq, "what": what},
                           className="btn small" + (" primary" if primary else ""),
                           n_clicks=0)
    return html.Div([
        dcc.Store(id={"type": "fig-spec", "rq": rq}),
        dcc.Download(id={"type": "exp-dl", "rq": rq}),
        html.Div([
            dcc.Clipboard(title="copy LaTeX (figure environment)",
                          id={"type": "exp-clip-fig", "rq": rq},
                          content="", className="btn small",
                          style={"display": "inline-flex"}),
            html.Span("LaTeX figure", className="small muted",
                      style={"marginRight": "10px", "marginLeft": "-2px"}),
            dcc.Clipboard(title="copy LaTeX (booktabs table)",
                          id={"type": "exp-clip-tab", "rq": rq},
                          content="", className="btn small",
                          style={"display": "inline-flex"}),
            html.Span("LaTeX table", className="small muted",
                      style={"marginRight": "10px", "marginLeft": "-2px"}),
            b("pdf", "PDF", primary=True), b("png", "PNG"),
            b("pgf", ".pgf"), b("zip", "Bundle"),
            html.Span(id={"type": "exp-hint", "rq": rq}, className="hint"),
        ], className="export-bar"),
    ])


def caption_editor(rq: str, initial: str = ""):
    """dcc.Textarea has no debounce; the callback it drives only builds two
    LaTeX strings (the TeX engine probe is cached and no figure is rendered),
    so per-keystroke cost is negligible. Figure rendering happens only on an
    explicit download click, behind a cache."""
    return html.Div(dcc.Textarea(
        id={"type": "caption", "rq": rq}, value=initial,
        placeholder="Caption used in exports…"), className="caption-box")


def empty_state(msg: str):
    return card(html.Div([
        html.Div("◌", style={"fontSize": "34px", "color": "var(--baseline)"}),
        html.Div(msg, className="muted", style={"marginTop": "6px"}),
    ], style={"textAlign": "center", "padding": "30px 0"}))


def pct(x, digits=0):
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=3):
    return fmt_num(x, digits)
