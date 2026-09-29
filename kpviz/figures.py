"""FigureSpec — one declarative chart description, two renderers.

The UI renders a spec with Plotly (interactive layer); exports re-render
the *same spec* with Matplotlib (PGF/PDF/PNG), so what you publish is what
you saw. Specs are plain JSON-able dicts and live in dcc.Store.

Theme = the validated reference palette (light, publication-grade):
categorical slots in fixed order, hairline solid grid, thin marks,
legend for >= 2 series, selective direct labels.
"""
from __future__ import annotations

import io
import math
import re

import plotly.graph_objects as go

# Plotly's JSON serialiser lazily imports PIL inside whatever thread happens
# to serialise a figure first. Two concurrent callbacks then race on a
# half-initialised module ("partially initialized module 'PIL.Image'").
# Importing it here, at single-threaded app-build time, removes the race.
try:  # pragma: no cover - optional dependency of plotly
    import importlib as _il
    _il.import_module("PIL.Image")
except Exception:
    pass

# ---- palette / chrome (light mode) ---------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"        # marks (rules, the frontier); never text
TICK = "#64635f"         # axis and colour-bar text: 5.9:1 on the surface
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
# diverging pair: agreement (positive) reads calm blue, disagreement red
DIV_LOW, DIV_MID, DIV_HIGH = "#e34948", "#f0efec", "#2a78d6"
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# a second categorical dimension on bars without touching the hue: Plotly
# pattern shapes and their Matplotlib hatch equivalents
PATTERNS = ["", "/", "x", "."]
_MPL_HATCH = {"": None, "/": "///", "x": "xxx", ".": "..."}

SIZES = {"1col": (3.35, 2.5), "2col": (7.0, 3.1), "slide": (7.5, 4.3)}

# Where the figure goes. Widths are the templates' own \columnwidth and
# \textwidth (in inches), so the exported figure is drawn at the size it is
# printed and never rescaled — the labels stay at `font` pt. `preamble` is
# what the venue typesets text with, used when KPViz compiles the PDF itself
# (a .pgf always takes the fonts of the document it is \input into).
VENUES = {
    "generic": {"label": "Generic (article, two-column)", "col": 3.18,
                "text": 6.50, "font": 8, "twocol": True,
                "preamble": []},
    "acl": {"label": "*ACL (ACL · EMNLP · NAACL)", "col": 3.03, "text": 6.30,
            "font": 9, "twocol": True,
            "preamble": [r"\usepackage{times}", r"\usepackage{latexsym}"]},
    "acm": {"label": "ACM (acmart sigconf)", "col": 3.33, "text": 7.00,
            "font": 8, "twocol": True,
            "preamble": [r"\usepackage{libertine}",
                         r"\usepackage[libertine]{newtxmath}"],
            "fallback": [r"\usepackage{mathptmx}"]},
    "ieee": {"label": "IEEE (IEEEtran conference)", "col": 3.49, "text": 7.12,
             "font": 8, "twocol": True,
             "preamble": [r"\usepackage{mathptmx}"]},
    "lncs": {"label": "Springer LNCS (single column)", "col": 4.80,
             "text": 4.80, "font": 8, "twocol": False, "preamble": []},
    "neurips": {"label": "NeurIPS · ICLR · ICML (single column)",
                "col": 5.50, "text": 5.50, "font": 9, "twocol": False,
                "preamble": [r"\usepackage{times}"]},
}
HEIGHTS = {"compact": 0.8, "std": 1.0, "tall": 1.3}


def geometry(spec: dict) -> tuple[float, float, float]:
    """(width in, height in, base font pt) of the exported figure.

    Without export options this is the historical size (1col/2col/slide at
    8 pt). With them: the venue's column or text width, an aspect that suits
    the span (column figures are taller relative to their width), and the
    venue's figure font."""
    size = spec.get("size", "2col")
    exp = spec.get("export") or {}
    if size == "slide" or not exp:
        w, h = SIZES.get(size, SIZES["2col"])
        return w, h, 8 if size in ("1col", "2col") else 11
    v = VENUES.get(exp.get("venue") or "generic", VENUES["generic"])
    span = exp.get("span") or "auto"
    if span == "auto":
        span = "col" if size == "1col" else "full"
    if not v["twocol"]:
        span_w = v["text"]
        aspect = 0.62 if size != "1col" else 0.75
    elif span == "col":
        span_w, aspect = v["col"], 0.75
    else:
        span_w, aspect = v["text"], 0.44
    if spec.get("aspect"):
        aspect = spec["aspect"]
    aspect *= HEIGHTS.get(exp.get("height") or "std", 1.0)
    return span_w, round(span_w * aspect, 3), v["font"]
FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def seq_color(t: float) -> str:
    t = min(1.0, max(0.0, t))
    return SEQ_RAMP[int(round(t * (len(SEQ_RAMP) - 1)))]


# ---- direct label placement ----------------------------------------------
# Scatter labels ("bart-base-kp20k (num_beams=10)") collide with each other,
# with other markers and with the dashed no-cost lines when every one of them
# is pinned "middle right". This pass picks one of nine anchors per label by
# greedy first-fit in normalised figure space, longest label first (hardest to
# place). Both renderers call it with their own text metrics, so each output is
# decluttered against its real geometry. Cost is O(n^2) over labelled points.

_ANCHORS = ("middle right", "middle left", "top center", "bottom center",
            "top right", "bottom right", "top left", "bottom left")
# anchor -> matplotlib (ha, va, dx_pt, dy_pt)
MPL_ANCHOR = {
    "middle right": ("left", "center_baseline", 5.0, 0.0),
    "middle left": ("right", "center_baseline", -5.0, 0.0),
    "top center": ("center", "bottom", 0.0, 4.5),
    "bottom center": ("center", "top", 0.0, -4.5),
    "top right": ("left", "bottom", 3.5, 3.0),
    "bottom right": ("left", "top", 3.5, -3.0),
    "top left": ("right", "bottom", -3.5, 3.0),
    "bottom left": ("right", "top", -3.5, -3.0),
    "middle center": ("center", "center_baseline", 0.0, 0.0),
}
_MAX_LABELS = 60          # beyond this, direct labels are not readable anyway
# (char width, line height, pad x, pad y) in fractions of the plot area. The
# two renderers have genuinely different geometry — 11px text over a ~1080px
# web canvas vs 7pt text over a 453pt figure — so each is placed against its
# own metrics rather than one compromise that declutters neither.
GEOM = {
    "plotly": (5.7 / 1080.0, 15.0 / 470.0, 6.0 / 1080.0, 5.0 / 470.0),
    # pads are >= the offsets in MPL_ANCHOR, so the box reserved during
    # placement is never smaller than the text actually drawn
    "mpl": (3.95 / 453.0, 8.6 / 173.0, 5.5 / 453.0, 4.5 / 173.0),
    "mpl1col": (3.4 / 200.0, 7.4 / 132.0, 5.0 / 200.0, 4.0 / 132.0),
}
_CHAR_W, _LINE_H, _PAD_X, _PAD_Y = GEOM["plotly"]


def _rect(anchor, px, py, w, h, pad_x=_PAD_X, pad_y=_PAD_Y):
    """Label box for `anchor`, in normalised coordinates."""
    if anchor.endswith("right"):
        x0, x1 = px + pad_x, px + pad_x + w
    elif anchor.endswith("left"):
        x0, x1 = px - pad_x - w, px - pad_x
    else:
        x0, x1 = px - w / 2, px + w / 2
    if anchor.startswith("top"):
        y0, y1 = py + pad_y, py + pad_y + h
    elif anchor.startswith("bottom"):
        y0, y1 = py - pad_y - h, py - pad_y
    else:
        y0, y1 = py - h / 2, py + h / 2
    return x0, y0, x1, y1


def _overlap(a, b) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _norm_axis(vals, scale):
    """(project, lo, hi) mapping data values into [0, 1]."""
    if scale == "log":
        vals = [v for v in vals if v is not None and v > 0]
        vals = [math.log10(v) for v in vals]
        proj = lambda v: math.log10(v) if v and v > 0 else None  # noqa: E731
    else:
        vals = [v for v in vals if v is not None]
        proj = lambda v: v  # noqa: E731
    if not vals:
        return proj, 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.06
    return proj, lo - pad, hi + pad


def place_labels(spec: dict, geom: str = "plotly") -> dict:
    """{(series_index, point_index): anchor} for every shown text label."""
    char_w, line_h, pad_x, pad_y = GEOM.get(geom, GEOM["plotly"])
    series = spec.get("series") or []
    items, xs, ys = [], [], []
    for si, s in enumerate(series):
        sx, sy = s.get("x") or [], s.get("y") or []
        xs.extend(sx)
        ys.extend(sy)
        if not (s.get("text") and s.get("show_text", True)):
            continue
        for pi, t in enumerate(s["text"]):
            if t and pi < len(sx) and pi < len(sy):
                items.append((si, pi, sx[pi], sy[pi], str(t)))
    if not items or len(items) > _MAX_LABELS:
        return {}
    fr = spec.get("frontier") or {}
    xs.extend(fr.get("x") or [])
    ys.extend(fr.get("y") or [])
    hl_ys = [h["y"] for h in spec.get("hlines") or [] if h.get("y") is not None]
    ys.extend(hl_ys)
    px_of, x0, x1 = _norm_axis(xs, spec.get("xscale"))
    py_of, y0, y1 = _norm_axis(ys, spec.get("yscale"))
    sx_ = lambda v: None if px_of(v) is None else (px_of(v) - x0) / (x1 - x0)  # noqa: E731
    sy_ = lambda v: None if py_of(v) is None else (py_of(v) - y0) / (y1 - y0)  # noqa: E731

    # Obstacles a label must not land on: every marker, the full width of each
    # dashed rule (plus a taller band at both ends, since Plotly writes the
    # rule's own label on the left and Matplotlib on the right), and each
    # segment of the Pareto staircase.
    blocked = []
    for s in series:
        for xv, yv in zip(s.get("x") or [], s.get("y") or []):
            nx, ny = sx_(xv), sy_(yv)
            if nx is None or ny is None:
                continue
            # at least half a line box tall, so no label can run through a mark
            blocked.append((nx - 0.008, ny - line_h * 0.62,
                            nx + 0.008, ny + line_h * 0.62))
        # the interval whiskers too: a label across one hides it
        for xv, e in zip(s.get("x") or [], s.get("err") or []):
            lo, hi = (e or (None, None))[:2] if e else (None, None)
            nx, nlo, nhi = sx_(xv), sy_(lo), sy_(hi)
            if None not in (nx, nlo, nhi):
                blocked.append((nx - 0.006, min(nlo, nhi), nx + 0.006, max(nlo, nhi)))
    for h in spec.get("hlines") or []:
        ny = sy_(h.get("y"))
        if ny is None:
            continue
        blocked.append((0.0, ny - line_h * 0.5, 1.0, ny + line_h * 0.5))
        w = min(0.98, len(str(h.get("label") or "")) * char_w + 0.02)
        blocked.append((0.0, ny - line_h * 0.8, w, ny + line_h * 0.8))
        blocked.append((1.0 - w, ny - line_h * 0.8, 1.0, ny + line_h * 0.8))
    fx, fy = [sx_(v) for v in (fr.get("x") or [])], \
             [sy_(v) for v in (fr.get("y") or [])]
    for i in range(len(fx) - 1):
        a, b, ya, yb = fx[i], fx[i + 1], fy[i], fy[i + 1]
        if None in (a, b, ya, yb):
            continue
        blocked.append((min(a, b), ya - 0.004, max(a, b), ya + 0.004))   # tread
        blocked.append((b - 0.004, min(ya, yb), b + 0.004, max(ya, yb)))  # riser

    placed, out = [], {}
    for si, pi, xv, yv, txt in sorted(items, key=lambda it: -len(it[4])):
        nx, ny = sx_(xv), sy_(yv)
        if nx is None or ny is None:
            continue
        w, h = len(txt) * char_w, line_h
        best, best_cost = "middle right", None
        for k, anchor in enumerate(_ANCHORS):
            r = _rect(anchor, nx, ny, w, h, pad_x, pad_y)
            # leaving the frame is the worst outcome: the label is clipped.
            out_of_frame = (max(0.0, -r[0]) + max(0.0, r[2] - 1.0)
                            + max(0.0, -r[1]) + max(0.0, r[3] - 1.0))
            cost = out_of_frame * 24.0 + k * 1e-4
            for other in placed:
                cost += _overlap(r, other) * 60.0
            for b in blocked:
                cost += _overlap(r, b) * 30.0
            if best_cost is None or cost < best_cost:
                best, best_cost = anchor, cost
                if cost <= k * 1e-4 + 1e-9:   # perfect fit, stop searching
                    break
        out[(si, pi)] = best
        placed.append(_rect(best, nx, ny, w, h, pad_x, pad_y))
    return out


# ===========================================================================
# Plotly renderer (interactive layer)
# ===========================================================================

def _err_arrays(ys, err):
    """(plus, minus) distances for Plotly error bars from (lo, hi) pairs;
    None where there is no interval."""
    plus, minus = [], []
    for y, e in zip(ys, err or []):
        lo, hi = (e or (None, None))[:2] if e else (None, None)
        if y is None or lo is None or hi is None:
            plus.append(None)
            minus.append(None)
        else:
            plus.append(max(0.0, hi - y))
            minus.append(max(0.0, y - lo))
    return plus, minus


def _has_err(err) -> bool:
    return bool(err) and any(e and e[0] is not None for e in err)


def _bar_gap(spec: dict) -> float:
    """A gap that keeps each bar at most ~26 px wide on the plot's usual
    width: three categories across a 1 000 px card otherwise draw 100 px
    slabs that outweigh the data they carry."""
    horiz = spec.get("orientation") == "h"
    key = "y" if horiz else "x"
    cats = {c for s_ in spec.get("series", []) for c in s_.get(key, [])}
    n_cat = max(1, len(cats))
    n_ser = (1 if spec.get("barmode") in ("stack", "overlay")
             else max(1, len(spec.get("series", []))))
    width = 480.0 if spec.get("size") == "1col" else 1000.0
    if horiz:
        width = 26.0 * n_cat * n_ser / 0.65      # rows get the height they need
    slot = width / n_cat
    return round(min(0.85, max(0.3, 1 - 26.0 * n_ser / slot)), 3)


def _plotly_layout(spec: dict) -> dict:
    items = spec.get("legend_items") or []
    n_series = len(items or spec.get("series", []))
    kind = spec.get("kind")
    show_legend = spec.get("show_legend",
                           (n_series >= 2 or kind == "dumbbell")
                           and kind != "heatmap")
    # a legend with an architecture block says what each channel encodes
    legend_title = spec.get("legend_title") or (
        "colour = model · shape = architecture"
        if any(i.get("block", "").startswith("architecture") for i in items)
        else "")
    horiz_bar = kind == "bar" and spec.get("orientation") == "h"
    # gridlines on the value axis only: a line between two categories (or at
    # every minor tick of a log axis) turns a plot into graph paper
    x_grid = kind == "scatter" and not spec.get("xticks") or horiz_bar
    y_grid = not horiz_bar and kind != "dumbbell"
    lay = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, size=13.5, color=INK),
        margin=dict(l=60, r=24, t=30 if spec.get("title") else 12, b=52),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12.5, color=INK2),
                    # no title on screen: Plotly indents a horizontal row of
                    # items by the title's width and the row overflows a
                    # narrow card; the scope line above the figure says what
                    # colour and shape encode (the exports keep the title)
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        xaxis=dict(title=dict(text=spec.get("xlabel") or "", font=dict(size=13, color=INK2)),
                   showgrid=x_grid, gridcolor=GRID, gridwidth=1, zeroline=False,
                   linecolor=BASELINE, linewidth=1,
                   tickfont=dict(size=12, color=TICK)),
        yaxis=dict(title=dict(text=spec.get("ylabel") or "", font=dict(size=13, color=INK2)),
                   showgrid=y_grid, gridcolor=GRID, gridwidth=1, zeroline=False,
                   linecolor=BASELINE, linewidth=1,
                   tickfont=dict(size=12, color=TICK)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(family=FONT_STACK, size=12.5, color=INK)),
    )
    if spec.get("hovermode") == "x":
        lay["hovermode"] = "x unified"
    if spec.get("legend") == "right" and show_legend:
        # many series: a column beside the plot reads better than a block of
        # rows above it that eats half the height
        lay["legend"] = dict(orientation="v", yanchor="top", y=1, x=1.01,
                             xanchor="left", font=dict(size=12, color=INK2),
                             title=dict(text=legend_title,
                                        font=dict(size=11.5, color=TICK)),
                             groupclick="toggleitem")
        lay["margin"]["r"] = 12
    # axis scale only when explicitly requested (never force 'linear' onto
    # categorical axes — string categories would coerce to NaN)
    if spec.get("xtickangle"):
        lay["xaxis"]["tickangle"] = spec["xtickangle"]
        lay["margin"]["b"] = 84
    if spec.get("xscale"):
        lay["xaxis"]["type"] = spec["xscale"]
    if spec.get("yscale"):
        lay["yaxis"]["type"] = spec["yscale"]
    for ax in ("xaxis", "yaxis"):
        if lay[ax].get("type") == "log":
            # one tick and one gridline per decade; plain decimals when the
            # values were scaled to a readable unit ("per 1,000 documents"),
            # 10^n otherwise — the Matplotlib export does the same
            lay[ax].update(dtick=1, minor=dict(showgrid=False))
            if spec.get(ax[0] + "plain"):
                lay[ax].update(exponentformat="none", tickformat=",~g")
            else:
                lay[ax].update(exponentformat="power", showexponent="all")
    # keep the user's zoom and legend toggles across updates of the same view
    lay["uirevision"] = spec.get("name") or spec.get("kind")
    # a changed metric, cost unit or filter moves the marks to their new
    # place instead of redrawing the plot (same traces only; Plotly skips it
    # otherwise)
    lay["transition"] = dict(duration=350, easing="cubic-in-out")
    if len(spec.get("series", [])) >= 2 and spec.get("barmode") == "stack":
        lay["legend"]["traceorder"] = "normal"
    if spec.get("title"):
        lay["title"] = dict(text=spec["title"], font=dict(size=14, color=INK),
                            x=0, xanchor="left")
    if spec.get("xticks"):
        lay["xaxis"].update(tickmode="array", tickvals=spec["xticks"]["vals"],
                            ticktext=spec["xticks"]["text"])
    if spec.get("xrange"):
        lay["xaxis"]["range"] = spec["xrange"]
    if spec.get("yrange"):
        lay["yaxis"]["range"] = spec["yrange"]
    if kind == "hist":
        # bins touch when overlaid (one shape per split, seen through each
        # other); side by side, each bin holds one thin bar per split
        overlay = spec.get("histmode", "overlay") == "overlay"
        lay["barmode"] = "overlay" if overlay else "group"
        lay["bargap"] = 0.04 if overlay else 0.16
        lay["bargroupgap"] = 0.0 if overlay else 0.08
    if kind == "bar":
        lay["barmode"] = spec.get("barmode", "group")
        lay["bargap"] = _bar_gap(spec)
        lay["bargroupgap"] = 0.1
        if horiz_bar:
            cats = {c for s_ in spec.get("series", []) for c in s_.get("y", [])}
            lay["yaxis"]["autorange"] = "reversed"      # first run on top
            lay["yaxis"]["tickfont"] = dict(size=12.5, color=INK2)
            n_s = 1 if spec.get("barmode") == "stack" else max(1, len(spec.get("series", [])))
            if len(cats) > 6:
                lay["height"] = len(cats) * (12 * n_s + 12) + 150
    return lay


def _pie_grid_layout(spec: dict) -> tuple[int, int, list[dict]]:
    """(ncols, nrows, panels) for a pie grid — rows are groups, columns the
    subgroups inside them, so the geometry itself carries the grouping."""
    panels = spec.get("panels") or []
    ncols = spec.get("ncols") or max(1, len({p.get("col") for p in panels}))
    nrows = max(1, -(-len(panels) // ncols))
    return ncols, nrows, panels


def _dumbbell_points(spec: dict) -> list[tuple[str, str, str]]:
    """[(row key, legend name, colour)] for a dumbbell.

    Two points is the classic before/after; a third (here: the flagged
    documents themselves) turns the same row into a small distribution, so the
    reader sees what was removed and not only that something was."""
    pts = spec.get("points")
    if pts:
        return [(p["key"], p["name"], p.get("color", MUTED)) for p in pts]
    return [("x0", spec.get("x0_name", "A"), MUTED),
            ("x1", spec.get("x1_name", "B"), spec.get("accent", "#2a78d6"))]


def to_plotly(spec: dict) -> go.Figure:
    kind = spec.get("kind", "scatter")

    if kind == "pie_grid":
        from plotly.subplots import make_subplots
        ncols, nrows, panels = _pie_grid_layout(spec)
        row_titles = spec.get("row_titles") or []
        fig = make_subplots(
            rows=nrows, cols=ncols,
            specs=[[{"type": "domain"}] * ncols for _ in range(nrows)],
            subplot_titles=[p.get("title", "") for p in panels],
            horizontal_spacing=0.04,
            # enough for a subplot title, not the 25% gap an even split gives
            vertical_spacing=min(0.2, 0.34 / max(1, nrows)))
        for i, p in enumerate(panels):
            fig.add_trace(go.Pie(
                labels=p["labels"], values=p["values"],
                marker=dict(colors=p.get("colors"),
                            line=dict(color=SURFACE, width=1.5)),
                sort=False, direction="clockwise", hole=p.get("hole", 0.38),
                textinfo="percent", textposition="inside",
                insidetextorientation="horizontal",
                texttemplate="%{percent:.0%}",
                textfont=dict(size=10),
                hovertemplate=("%{label}: %{value} (%{percent})"
                               f"<extra>{p.get('title', '')}</extra>"),
                showlegend=(i == 0)),
                row=1 + i // ncols, col=1 + i % ncols)
        lay = _plotly_layout(spec)
        lay.pop("xaxis", None)
        lay.pop("yaxis", None)
        # a pie grid has no axes to steal room from, so the margins are the only
        # thing standing between a subplot title (or the legend) and a crop
        lay["margin"] = dict(l=96, r=16, t=26, b=40)
        lay["showlegend"] = True
        lay["legend"] = dict(orientation="h", yanchor="top", y=-0.02,
                             xanchor="center", x=0.5,
                             font=dict(size=11.5, color=INK2))
        fig.update_layout(**lay)
        fig.update_annotations(font=dict(size=11, color=INK2), yshift=4)
        for r, t in enumerate(row_titles):
            fig.add_annotation(x=0, y=1 - (r + 0.5) / max(1, nrows),
                               xref="paper", yref="paper", text=f"<b>{t}</b>",
                               showarrow=False, xanchor="left",
                               font=dict(size=11.5, color=INK))
        return fig

    fig = go.Figure(layout=_plotly_layout(spec))

    if kind == "heatmap":
        h = spec["heat"]
        stops = ([DIV_LOW, DIV_MID, DIV_HIGH] if h.get("diverging") else SEQ_RAMP)
        colorscale = [[i / (len(stops) - 1), c] for i, c in enumerate(stops)]
        fig.add_trace(go.Heatmap(
            z=h["z"], x=h["x"], y=h["y"],
            zmin=h.get("zmin"), zmax=h.get("zmax"), zmid=h.get("zmid"),
            colorscale=colorscale, xgap=3, ygap=3,
            customdata=h.get("customdata"),
            hovertemplate=h.get("hover", "%{y} × %{x}: %{z:.3f}<extra></extra>"),
            # the colour bar sits under the matrix, where the eye already is
            colorbar=dict(orientation="h", thickness=10, len=0.4, x=0.5,
                          xanchor="center", y=-0.2, yanchor="top",
                          outlinewidth=0, tickfont=dict(size=11, color=TICK))))
        # cell text in white on dark cells, ink on light ones (a dark-blue
        # cell with ink text is 4:1 at best)
        zmin, zmax = h.get("zmin", 0), h.get("zmax", 1)
        for i, row in enumerate(h.get("text") or []):
            for j, t in enumerate(row):
                if t in (None, ""):
                    continue
                v = (h["z"][i][j] if i < len(h["z"]) and j < len(h["z"][i])
                     else None)
                if v is None:
                    col, frac = TICK, None
                else:
                    frac = (v - zmin) / (zmax - zmin) if zmax > zmin else 0.5
                    col = "#ffffff" if _hex_dark(_ramp_at(stops, frac)) else INK
                fig.add_annotation(x=h["x"][j], y=h["y"][i], text=str(t),
                                   showarrow=False,
                                   font=dict(size=14, color=col))
        # square cells: a 3 × 3 matrix is not a banner
        fig.update_yaxes(autorange="reversed", showgrid=False,
                         scaleanchor="x", scaleratio=1, constrain="domain",
                         tickfont=dict(size=12.5, color=INK2))
        fig.update_xaxes(showgrid=False, constrain="domain",
                         tickfont=dict(size=12.5, color=INK2))
        fig.update_layout(margin=dict(l=110, r=24, t=12, b=90))
        return fig

    if kind == "dumbbell":
        rows = spec.get("rows", [])
        ylabels = [r["label"] for r in rows]
        points = _dumbbell_points(spec)
        for i, r in enumerate(rows):
            xs = [r.get(k) for k, _n, _c in points if r.get(k) is not None]
            if len(xs) > 1:
                fig.add_trace(go.Scatter(
                    x=[min(xs), max(xs)], y=[i, i], mode="lines",
                    line=dict(color=BASELINE, width=2),
                    showlegend=False, hoverinfo="skip"))
        # the subsets of one row sit just above and below its line, so the
        # three points and their whiskers never draw over each other
        n_pts = len(points)
        for j, (kx, name, color) in enumerate(points):
            off = (j - (n_pts - 1) / 2) * 0.16
            fig.add_trace(go.Scatter(
                x=[r.get(kx) for r in rows], y=[i + off for i in range(len(rows))],
                mode="markers", name=name,
                marker=dict(size=10, color=color,
                            line=dict(color=SURFACE, width=1.5)),
                **({"error_x": dict(
                    type="data", symmetric=False, visible=True,
                    array=_err_arrays([r.get(kx) for r in rows],
                                      [(r.get("err") or {}).get(kx) for r in rows])[0],
                    arrayminus=_err_arrays([r.get(kx) for r in rows],
                                           [(r.get("err") or {}).get(kx) for r in rows])[1],
                    thickness=1.1, width=2, color=color)}
                   if _has_err([(r.get("err") or {}).get(kx) for r in rows])
                   else {}),
                customdata=[(r.get("hover") or [])[j]
                            if isinstance(r.get("hover"), list)
                            and len(r["hover"]) > j else "" for r in rows],
                hovertemplate="%{customdata}<extra>" + name + "</extra>"))
        fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=ylabels,
                         showgrid=False, autorange="reversed",
                         range=[len(rows) - 0.5, -0.5],
                         tickfont=dict(size=12.5, color=INK2))
        fig.update_xaxes(showgrid=True)
        # 44 px per row: two rows no longer sit 300 px apart
        fig.update_layout(height=max(180, 44 * len(rows) + 120))
        return fig

    # scatter / line / bar families ----------------------------------------
    anchors = place_labels(spec) if kind != "bar" else {}
    items = spec.get("legend_items") or []
    for it in items:
        # a legend entry per model (hue) and per architecture (shape); the
        # marks themselves stay out of the legend and toggle with their model
        fig.add_trace(go.Scatter(
            x=[None], y=[None], name=it["name"], legendgroup=it.get("group"),
            mode="lines+markers" if it.get("line") else "markers",
            marker=dict(size=9, color=it["color"],
                        symbol=it.get("shape", "circle"),
                        line=dict(color=SURFACE, width=1)),
            line=dict(color=it["color"], width=2), hoverinfo="skip",
            showlegend=True))
    for si, s in enumerate(spec.get("series", [])):
        common = dict(name=s.get("name", ""),
                      showlegend=bool(s.get("in_legend", True)) and not items)
        hover = s.get("hover")
        if kind == "hist":
            col = s.get("color", "#2a78d6")
            overlay = spec.get("histmode", "overlay") == "overlay"
            bw = spec.get("bin_width")
            fig.add_trace(go.Bar(
                x=s["x"], y=s["y"],
                width=(bw if overlay and bw else None),
                marker=dict(color=_rgba(col, 0.38) if overlay else col,
                            line=dict(color=col, width=1.4 if overlay else 0)),
                customdata=hover,
                hovertemplate=("%{customdata}<extra></extra>" if hover
                               else "%{x}: %{y:.2f}<extra>"
                               + s.get("name", "") + "</extra>"),
                **common))
            continue
        if kind == "bar":
            horiz = spec.get("orientation") == "h"
            fig.add_trace(go.Bar(
                x=s["x"], y=s["y"], orientation="h" if horiz else "v",
                marker=dict(color=s.get("color", "#2a78d6"),
                            opacity=s.get("alpha", 1.0),
                            cornerradius=4,
                            pattern=dict(shape=s.get("pattern", ""),
                                         size=5, solidity=0.32,
                                         fgcolor=SURFACE),
                            line=dict(width=0)),
                text=s.get("text"),
                textposition="outside" if s.get("text") else None,
                textfont=dict(size=12, color=INK2),
                **({("error_x" if horiz else "error_y"): dict(
                    type="data", symmetric=False, visible=True,
                    array=_err_arrays(s[("x" if horiz else "y")], s["err"])[0],
                    arrayminus=_err_arrays(s[("x" if horiz else "y")], s["err"])[1],
                    thickness=1.1, width=2.5, color=INK2)}
                   if _has_err(s.get("err")) else {}),
                customdata=hover,
                hovertemplate=("%{customdata}<extra></extra>" if hover
                               else ("%{y}: %{x:.3f}" if horiz
                                     else "%{x}: %{y:.3f}")
                               + "<extra>" + s.get("name", "") + "</extra>"),
                **common))
            continue
        mode = s.get("mode", "markers")
        if s.get("text") and s.get("show_text", True) and "text" not in mode:
            mode = mode + "+text"
        # a light run of a model still reads: its mark is outlined in the
        # model's full hue; a provisional mark is drawn open
        edge = s.get("edge") or s.get("color", "#2a78d6")
        opened = bool(s.get("open"))
        marker = dict(size=s.get("size", 9) + (1 if opened else 0),
                      color=s.get("color", "#2a78d6"),
                      symbol=s.get("shape", "circle") + ("-open" if opened else ""),
                      opacity=s.get("alpha", 1.0),
                      line=dict(color=edge, width=2 if opened else 1))
        line = dict(color=s.get("color", "#2a78d6"),
                    width=s.get("width", 1.8),
                    dash=_plotly_dash(s.get("dash")))
        pos = [anchors.get((si, pi), "middle right")
               for pi in range(len(s.get("text") or []))] or "middle right"
        band = s.get("band")
        if band and _has_err(band):
            pts = [(x, e) for x, e in zip(s["x"], band) if e and e[0] is not None]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[x for x, _e in pts] + [x for x, _e in reversed(pts)],
                    y=[e[1] for _x, e in pts] + [e[0] for _x, e in reversed(pts)],
                    fill="toself", fillcolor=s.get("color", "#2a78d6"),
                    opacity=0.13, line=dict(width=0), hoverinfo="skip",
                    showlegend=False, legendgroup=s.get("legendgroup")
                    or s.get("name")))
        if _has_err(s.get("err")):
            plus, minus = _err_arrays(s["y"], s["err"])
            common["error_y"] = dict(type="data", symmetric=False, array=plus,
                                     arrayminus=minus, thickness=1.1, width=2.5,
                                     color=s.get("color", INK2))
        if s.get("legendgroup"):
            common["legendgroup"] = s["legendgroup"]
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode=mode, marker=marker, line=line,
            text=s.get("text"), textposition=pos,
            textfont=dict(size=11, color=INK2),
            customdata=hover,
            hovertemplate=("%{customdata}<extra></extra>" if hover else None),
            connectgaps=False, **common))

    # frontier (step line under the points)
    fr = spec.get("frontier")
    if fr and fr.get("x"):
        fig.add_trace(go.Scatter(
            x=fr["x"], y=fr["y"], mode="lines", name="Pareto frontier",
            line=dict(color="#6f6e69", width=2, shape="hv"),
            showlegend=True, hoverinfo="skip"))

    for hl in spec.get("hlines", []):
        fig.add_hline(y=hl["y"], line_color=hl.get("color", MUTED),
                      line_dash="dash" if hl.get("dash", True) else "solid",
                      line_width=1.4, opacity=hl.get("alpha", 0.9),
                      annotation_text=hl.get("label", ""),
                      annotation_position="top left",
                      annotation_font=dict(size=11, color=INK2))
    for vl in spec.get("vlines", []):
        # the label sits inside the plot, at the bottom: at the top it
        # collided with the legend row
        fig.add_vline(x=vl["x"], line_color=vl.get("color", MUTED),
                      line_dash="dash" if vl.get("dash", True) else "solid",
                      line_width=1.4, opacity=vl.get("alpha", 0.9),
                      annotation_text=vl.get("label", ""),
                      annotation_position="bottom right",
                      annotation_font=dict(size=11.5, color=INK2))
        if vl.get("shade_beyond"):
            xr = spec.get("xrange")
            x1 = xr[1] if xr else None
            if x1 is None:
                xs = [max(s["x"]) for s in spec.get("series", []) if s.get("x")]
                x1 = max(xs) * 1.05 if xs else vl["x"] * 2
            fig.add_vrect(x0=vl["x"], x1=x1, fillcolor=MUTED,
                          opacity=0.06, line_width=0)
    return fig


def empty_figure() -> go.Figure:
    """No figure: no axes, no grid, nothing to misread as data. The
    workbench hides the graph and shows the reason in its scope line."""
    fig = go.Figure(layout=dict(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                                meta="empty", xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                margin=dict(l=0, r=0, t=0, b=0)))
    return fig


def plotly_config(name: str = "kpviz") -> dict:
    """No mode bar: KPViz exports the publication figure itself (the camera
    button would save the on-screen zoom instead, and the bar covered the
    legend on hover). Double-click still resets a zoom."""
    return {"displaylogo": False, "displayModeBar": False,
            "scrollZoom": False, "doubleClick": "reset"}


def _rgba(h: str, a: float) -> str:
    """"#2a78d6", 0.4 -> "rgba(42,120,214,0.4)" (a fill that shows what is
    behind it, under an opaque outline of the same hue)."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a:g})"


def _ramp_at(stops: list[str], t: float) -> str:
    """The colour at t ∈ [0, 1] of a ramp of stops (linear in sRGB)."""
    t = min(1.0, max(0.0, t))
    pos = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(pos))
    f = pos - i

    def rgb(h):
        h = h.lstrip("#")
        return [int(h[k:k + 2], 16) for k in (0, 2, 4)]
    a, b = rgb(stops[i]), rgb(stops[i + 1])
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


def _hex_dark(h: str) -> bool:
    """Would white text read better than ink on this colour?"""
    h = h.lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) / 255 for k in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return lum < 0.2


# ===========================================================================
# Matplotlib renderer (export layer — also used for PGF/PDF/PNG)
# ===========================================================================

# Matplotlib's PGF backend escapes LaTeX-special *ASCII* itself (so "num_beams"
# is safe), but a non-ASCII glyph missing from the TeX font is dropped without
# warning — which silently deleted every em dash and, worse, every significance
# dagger from exported figures. Translate them to TeX before handing over.
_TEX_CHARS = {
    "—": "---", "–": "--", "‑": "-", "−": "$-$", "’": "'", "‘": "`",
    "“": "``", "”": "''", "…": r"\ldots{}", "·": r"$\cdot$",
    "×": r"$\times$", "±": r"$\pm$", "≈": r"$\approx$", "≃": r"$\simeq$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≠": r"$\neq$", "→": r"$\rightarrow$",
    "←": r"$\leftarrow$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    "†": r"$\dagger$", "‡": r"$\ddagger$", "°": r"$^{\circ}$",
    "µ": r"$\mu$", "μ": r"$\mu$", "α": r"$\alpha$", "β": r"$\beta$",
    "Δ": r"$\Delta$", "σ": r"$\sigma$", "ρ": r"$\rho$", "λ": r"$\lambda$",
    "∞": r"$\infty$", "√": r"$\surd$", "⚠": "", "◌": "", "◆": r"$\blacklozenge$",
    "★": r"$\star$", "•": r"$\bullet$",
    "τ": r"$\tau$", "η": r"$\eta$", "χ": r"$\chi$", "κ": r"$\kappa$",
    "δ": r"$\delta$", "ε": r"$\varepsilon$", "θ": r"$\theta$",
    "π": r"$\pi$", "ν": r"$\nu$", "ω": r"$\omega$", "γ": r"$\gamma$",
    "²": r"$^{2}$", "³": r"$^{3}$", "¹": r"$^{1}$", "½": r"$\frac{1}{2}$",
    "∈": r"$\in$", "∑": r"$\sum$", "≪": r"$\ll$", "≫": r"$\gg$",
    "∼": r"$\sim$", "∝": r"$\propto$", "⇒": r"$\Rightarrow$",
    "✓": r"$\checkmark$", "✗": r"$\times$", " ": "~", " ": r"\,",
    " ": r"\,",
}
_TEX_TABLE = str.maketrans(_TEX_CHARS)
# Matplotlib 3.10's PGF backend neutralises only "^" and "%", so a label like
# "bart-base-kp20k (num_beams=4)" writes a raw underscore into the .pgf and the
# figure fails to compile the moment it is \input into a real paper. Escape the
# rest ourselves — in text-bearing fields only, because "#2a78d6" is a colour.
# every TeX special Matplotlib's PGF backend does not handle itself (it
# escapes % and ^): a literal backslash (a prompt "\\d+", a Windows path) or
# a lone "$" (a price) used to fail the render — and take TeX down for the
# session. Each is escaped exactly once, before the non-ASCII translation
# (whose output is itself TeX).
# ("$" becomes \textdollar, not \$: Matplotlib turns "\$" back into a bare
# "$" in non-math text before the PGF backend writes it)
_TEX_ASCII = {"\\": r"\textbackslash{}", "$": r"\textdollar{}", "_": r"\_", "&": r"\&",
              "#": r"\#", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}"}
_TEX_ASCII_RE = re.compile(r"[\\$_&#{}~]")
_TEXT_KEYS = frozenset((
    "text", "xlabel", "ylabel", "title", "name", "label", "labels", "ylabels",
    "x0_name", "x1_name", "legend_title", "annot", "caption",
    # heat["x"]/["y"] and categorical bar x values are axis text; numeric x/y
    # entries are left alone by the isinstance check below
    "x", "y",
))


def _tex_str(s: str) -> str:
    s = _TEX_ASCII_RE.sub(lambda m: _TEX_ASCII[m.group(0)], s)
    return s if s.isascii() else s.translate(_TEX_TABLE)


# a series' "dash": True / False (legacy) or a Plotly dash name
_MPL_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-.",
             "longdash": (0, (8, 3)), "longdashdot": (0, (8, 3, 1, 3))}


def _plotly_dash(d) -> str:
    if isinstance(d, str) and d in _MPL_DASH:
        return d
    return "dash" if d else "solid"


def _mpl_dash(d):
    return _MPL_DASH.get(_plotly_dash(d), "-")


def tex_sanitize(obj, key: str | None = None):
    """Copy of `obj` with TeX-hostile characters translated (PGF export only).

    Escaping is applied to strings reached through a text-bearing key, so
    colours, dash styles and marker codes pass through untouched.
    """
    if isinstance(obj, str):
        return _tex_str(obj) if key in _TEXT_KEYS else obj
    if isinstance(obj, dict):
        return {k: tex_sanitize(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tex_sanitize(v, key) for v in obj]
    return obj


def _mpl_rc(size, pgf: bool) -> dict:
    """rc for one export; `size` is a spec (venue-aware) or a size name."""
    if isinstance(size, dict):
        base = geometry(size)[2]
    else:
        base = 8 if size in ("1col", "2col") else 11
    rc = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": base, "axes.titlesize": base + 1,
        "axes.labelsize": base, "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1, "legend.fontsize": base - 1,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": BASELINE, "ytick.color": BASELINE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.labelcolor": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4, "lines.markersize": 5,
        "savefig.dpi": 300, "figure.dpi": 130,
        "pdf.fonttype": 42,
    }
    if pgf:
        rc.update({"pgf.rcfonts": False, "font.family": "serif"})
    else:
        rc.update({"font.family": "serif",
                   "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
                   "mathtext.fontset": "dejavuserif"})
    return rc


def _mpl_pie_grid(spec: dict, plt, size: str, figsize, pgf: bool):
    """Pie grid: one row per group, one pie per subgroup — same geometry the
    interactive figure uses, so the exported version reads identically."""
    ncols, nrows, panels = _pie_grid_layout(spec)
    w, h = figsize
    with plt.rc_context(_mpl_rc(size, pgf)):
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(w, max(1.5, 1.45 * nrows + 0.45)),
                                 layout="constrained")
        for ax in axes.ravel():
            ax.axis("off")
        first = None
        for i, p in enumerate(panels):
            ax = axes[i // ncols][i % ncols]
            vals = [v for v in p["values"]]
            if not any(vals):
                ax.set_title(p.get("title", ""), fontsize=7, color=INK2)
                continue
            wedges, _t, _a = ax.pie(
                vals, colors=p.get("colors"),
                startangle=90, counterclock=False,
                wedgeprops=dict(width=1 - p.get("hole", 0.38),
                                edgecolor="white", linewidth=0.7),
                autopct=lambda pc: f"{pc:.0f}%" if pc >= 6 else "",
                pctdistance=0.72, textprops=dict(fontsize=6, color=INK))
            ax.set_title(p.get("title", ""), fontsize=7, color=INK2, pad=2)
            if first is None:
                first = (wedges, p["labels"])
        if first:
            fig.legend(first[0], first[1], loc="outside lower center",
                       ncols=min(4, len(first[1])), fontsize=7,
                       frameon=False)
        for r, t in enumerate(spec.get("row_titles") or []):
            if r < nrows:
                axes[r][0].text(-0.28, 0.5, t, transform=axes[r][0].transAxes,
                                rotation=90, va="center", ha="center",
                                fontsize=7.5, color=INK, fontweight="bold")
    return fig


def _num(v):
    """None/NaN-safe number for Matplotlib: missing is NaN (drawn as a gap),
    never zero. Plotly tolerates None; Matplotlib raised TypeError on it, so
    any figure with a missing value (an RQ3 run without a window, an
    undefined correlation) failed to export."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def render(spec: dict, pgf: bool, fmt: str, **savefig_kw) -> bytes:
    """Build *and save* a figure inside the same rc context.

    Saving after `to_mpl`'s context had exited dropped `pdf.fonttype: 42`
    (and `pgf.rcfonts`), so Matplotlib PDFs embedded Type 3 fonts — the
    classic camera-ready rejection. Every export path goes through here."""
    import matplotlib
    if matplotlib.get_backend().lower() not in (
            "agg", "pdf", "ps", "svg", "cairo", "pgf", "template"):
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    with plt.rc_context(_mpl_rc(spec, pgf)):
        fig = to_mpl(spec, pgf=pgf)
        try:
            fig.savefig(buf, format=fmt, **savefig_kw)
        finally:
            plt.close(fig)
    return buf.getvalue()


def to_mpl(spec: dict, pgf: bool = False):
    import matplotlib
    # every render here goes to an in-memory buffer, so a non-file backend
    # (inherited from MPLBACKEND or a notebook) would try to open a display
    if matplotlib.get_backend().lower() not in (
            "agg", "pdf", "ps", "svg", "cairo", "pgf", "template"):
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # placement is measured on the original strings; TeX macros like
    # "$\ddagger$" are 11 characters wide on paper only if you can't read
    raw_spec = spec
    if pgf:
        spec = tex_sanitize(spec)
    size = spec.get("size", "2col")
    gw, gh, _font = geometry(spec)
    figsize = (gw, gh)
    # the legend option from the export controls ("modulation" of the figure)
    leg = (spec.get("export") or {}).get("legend")
    if leg and leg != "auto":
        spec = dict(spec, legend=leg)
    if gw < 3.6 and size != "1col":
        size = "1col"          # a column-wide render uses the compact styling
    kind = spec.get("kind", "scatter")

    if kind == "pie_grid":
        return _mpl_pie_grid(spec, plt, size, figsize, pgf)

    with plt.rc_context(_mpl_rc(spec, pgf)):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        if kind == "heatmap":
            h = spec["heat"]
            import numpy as _np
            z = _np.array([[_num(v) for v in row] for row in h["z"]], dtype=float)
            if h.get("diverging"):
                cmap = LinearSegmentedColormap.from_list(
                    "kpdiv", [DIV_LOW, DIV_MID, DIV_HIGH])
                vmin = h.get("zmin", -1)
                vmax = h.get("zmax", 1)
                mid = h.get("zmid", 0)
                mid = min(max(mid, vmin + 1e-9), vmax - 1e-9)
                norm = TwoSlopeNorm(vcenter=mid, vmin=vmin, vmax=vmax)
                mesh_kw = dict(cmap=cmap, norm=norm)
            else:
                cmap = LinearSegmentedColormap.from_list("kpseq", SEQ_RAMP)
                mesh_kw = dict(cmap=cmap, vmin=h.get("zmin"), vmax=h.get("zmax"))
            # vector cells (pcolormesh), not an image: the PGF backend cannot
            # stream raster graphics, and vector cells stay sharp in a PDF
            ny, nx = z.shape
            im = ax.pcolormesh(_np.arange(nx + 1) - 0.5, _np.arange(ny + 1) - 0.5,
                               _np.ma.masked_invalid(z), shading="flat",
                               edgecolors="white", linewidth=0.8, **mesh_kw)
            ax.set_xlim(-0.5, nx - 0.5)
            ax.set_ylim(ny - 0.5, -0.5)
            ax.set_xticks(range(len(h["x"])), h["x"], rotation=30,
                          ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(h["y"])), h["y"])
            ax.grid(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
            if h.get("text"):
                for i, row in enumerate(h["text"]):
                    for j, t in enumerate(row):
                        if t not in (None, "") and z[i][j] == z[i][j]:
                            val = z[i][j]
                            dark = _is_dark(cmap, norm(val) if h.get("diverging")
                                            else _seq_t(val, h))
                            ax.text(j, i, str(t), ha="center", va="center",
                                    fontsize=max(5.5, (6 if size == "1col" else 7)),
                                    color="white" if dark else INK)
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            # Matplotlib rasterises a colour bar with ≥ 50 levels; PGF cannot
            # carry rasters, and a vector gradient costs little here
            if getattr(cb, "solids", None) is not None:
                cb.solids.set_rasterized(False)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=6 if size == "1col" else 7, color=MUTED)
        elif kind == "dumbbell":
            rows = spec.get("rows", [])
            ys = range(len(rows))
            points = _dumbbell_points(spec)
            for i, r in enumerate(rows):
                xs = [r.get(k) for k, _n, _c in points if r.get(k) is not None]
                if len(xs) > 1:
                    ax.plot([min(xs), max(xs)], [i, i], color=BASELINE,
                            lw=1.4, zorder=1)
            n_pts = len(points)
            for j, (kx, name, color) in enumerate(points):
                off = (j - (n_pts - 1) / 2) * 0.16      # as on screen
                yj = [y + off for y in ys]
                errs = [(r.get("err") or {}).get(kx) for r in rows]
                if _has_err(errs):
                    plus, minus = _err_arrays([r.get(kx) for r in rows], errs)
                    ax.errorbar([_num(r.get(kx)) for r in rows], yj,
                                xerr=[[_num(m) for m in minus],
                                      [_num(p) for p in plus]],
                                fmt="none", ecolor=color, elinewidth=0.7,
                                capsize=1.4, zorder=1 + j)
                ax.scatter([_num(r.get(kx)) for r in rows], yj, s=36,
                           color=color, zorder=2 + j, label=name,
                           edgecolors="white", linewidths=1.0)
            ax.set_yticks(list(ys), [r["label"] for r in rows])
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            _legend(ax, spec, len(points), (0, 1.12), min(3, len(points)))
        elif kind == "hist":
            series = spec.get("series", [])
            bw = float(spec.get("bin_width") or 1.0)
            overlay = spec.get("histmode", "overlay") == "overlay"
            n = max(1, len(series))
            for si, s in enumerate(series):
                xs = [_num(v) for v in s["x"]]
                ys = [_num(v) for v in s["y"]]
                col = s.get("color", "#2a78d6")
                if overlay:
                    ax.bar(xs, ys, width=bw, color=col, alpha=0.38,
                           linewidth=0, label=s.get("name", ""))
                    edges = [x - bw / 2 for x in xs] + [xs[-1] + bw / 2] if xs else []
                    if xs:
                        ax.stairs(ys, edges, color=col, lw=0.9)
                else:
                    w = bw * 0.84 / n
                    offs = [x - bw * 0.42 + w * (si + 0.5) for x in xs]
                    ax.bar(offs, ys, width=w * 0.9, color=col, linewidth=0,
                           label=s.get("name", ""))
            ax.grid(axis="x", visible=False)
            if len(series) >= 2:
                _legend(ax, spec, len(series), (0, 1.16), min(4, len(series)))
        elif kind == "bar":
            horiz = spec.get("orientation") == "h"
            cat_key, val_key = ("y", "x") if horiz else ("x", "y")
            cats: list = []
            for s in spec.get("series", []):
                for xv in s[cat_key]:
                    if xv not in cats:
                        cats.append(xv)
            n = max(1, len(spec.get("series", [])))
            stacked = spec.get("barmode") == "stack"
            overlaid = spec.get("barmode") == "overlay"
            width = 0.72 / (1 if stacked or overlaid else n)
            bottoms = {c: 0.0 for c in cats}
            for si, s in enumerate(spec.get("series", [])):
                pos = [cats.index(xv) for xv in s[cat_key]]
                offs = ([p for p in pos] if stacked or overlaid else
                        [p - 0.36 + width * (si + 0.5) for p in pos])
                bots = [bottoms[xv] for xv in s[cat_key]] if stacked else None
                hatch = _MPL_HATCH.get(s.get("pattern", ""), None)
                bar_kw = dict(color=s.get("color", "#2a78d6"),
                              alpha=s.get("alpha", 1.0),
                              label=s.get("name", ""), linewidth=0,
                              hatch=hatch, edgecolor=SURFACE)
                vals = [_num(v) for v in s[val_key]]
                if _has_err(s.get("err")) and not stacked:
                    plus, minus = _err_arrays(s[val_key], s["err"])
                    ekey = "xerr" if horiz else "yerr"
                    bar_kw[ekey] = [[_num(m) for m in minus], [_num(p) for p in plus]]
                    bar_kw["error_kw"] = dict(ecolor=INK2, elinewidth=0.6,
                                              capsize=1.4, capthick=0.6)
                if horiz:
                    ax.barh(offs, vals, height=width * 0.92, left=bots,
                            **bar_kw)
                else:
                    ax.bar(offs, vals, width=width * 0.92, bottom=bots,
                           **bar_kw)
                if s.get("text") and not stacked:
                    for xv, yv, t in zip(offs, vals, s["text"]):
                        if t and yv == yv:
                            ax.annotate(str(t), (yv, xv) if horiz else (xv, yv),
                                        xytext=(3, 0) if horiz else (0, 2),
                                        textcoords="offset points",
                                        ha="left" if horiz else "center",
                                        va="center" if horiz else "bottom",
                                        fontsize=7, color=INK2)
                if stacked:
                    for xv, yv in zip(s[cat_key], s[val_key]):
                        bottoms[xv] += yv or 0
            if horiz:
                ax.set_yticks(range(len(cats)), [str(c) for c in cats])
                ax.grid(axis="y", visible=False)
                ax.set_ylim(len(cats) - 0.5, -0.5)          # first on top
            else:
                long_cat = any(len(str(c)) > 8 for c in cats)
                ax.set_xticks(range(len(cats)), [str(c) for c in cats],
                              rotation=(20 if long_cat else 0),
                              ha="right" if long_cat else "center")
                ax.grid(axis="x", visible=False)
            if len(spec.get("series", [])) >= 2:
                _legend(ax, spec, len(spec["series"]), (0, 1.16),
                        min(2 if size == "1col" else 4, len(spec["series"])))
        else:  # scatter / line
            anchors = place_labels(raw_spec,
                                   "mpl1col" if size == "1col" else "mpl")
            for si, s in enumerate(spec.get("series", [])):
                mode = s.get("mode", "markers")
                y_raw = s["y"]
                s = dict(s, y=[_num(v) for v in s["y"]])
                band = s.get("band")
                if band and _has_err(band):
                    bx = [x for x, e in zip(s["x"], band) if e and e[0] is not None]
                    blo = [e[0] for e in band if e and e[0] is not None]
                    bhi = [e[1] for e in band if e and e[0] is not None]
                    ax.fill_between(bx, blo, bhi, color=s.get("color", "#2a78d6"),
                                    alpha=0.14, lw=0, zorder=1)
                if _has_err(s.get("err")):
                    plus, minus = _err_arrays(y_raw, s["err"])
                    ax.errorbar(s["x"], s["y"],
                                yerr=[[_num(m) for m in minus], [_num(p) for p in plus]],
                                fmt="none", ecolor=s.get("color", INK2),
                                elinewidth=0.6, capsize=1.4, capthick=0.6,
                                zorder=2)
                if "lines" in mode:
                    ax.plot(s["x"], s["y"], color=s.get("color", "#2a78d6"),
                            lw=s.get("width", 1.6),
                            ls=_mpl_dash(s.get("dash")),
                            marker=(s.get("mpl_marker", "o")
                                    if "markers" in mode else None),
                            ms=4.5, markeredgecolor="white",
                            markeredgewidth=0.8,
                            alpha=s.get("alpha", 1.0),
                            label=s.get("name", "") if s.get("in_legend", True) else None)
                else:
                    # outlined in the model's full hue; provisional = open
                    opened = bool(s.get("open"))
                    col = s.get("color", "#2a78d6")
                    ax.scatter(s["x"], s["y"],
                               s=(s.get("size", 9) ** 2) * 0.55,
                               c="none" if opened else col,
                               marker=s.get("mpl_marker", "o"),
                               alpha=s.get("alpha", 1.0),
                               edgecolors=s.get("edge") or col,
                               linewidths=1.3 if opened else 0.8,
                               label=s.get("name", "") if s.get("in_legend", True) else None,
                               zorder=3)
                if s.get("text") and s.get("show_text", True):
                    for pi, (xv, yv, t) in enumerate(
                            zip(s["x"], s["y"], s["text"])):
                        if not t:
                            continue
                        ha, va, dx, dy = MPL_ANCHOR[
                            anchors.get((si, pi), "middle right")]
                        ax.annotate(str(t), (xv, yv), xytext=(dx, dy),
                                    textcoords="offset points", ha=ha, va=va,
                                    fontsize=max(5.5, (6 if size == "1col" else 7)),
                                    color=INK2)
            fr = spec.get("frontier")
            if fr and fr.get("x"):
                ax.plot(fr["x"], fr["y"], drawstyle="steps-post", color="#6f6e69",
                        lw=1.3, label="Pareto frontier", zorder=2)
            if spec.get("xscale") == "log":
                ax.set_xscale("log")
                if spec.get("xplain"):
                    from matplotlib.ticker import FuncFormatter, NullFormatter
                    ax.xaxis.set_major_formatter(
                        FuncFormatter(lambda v, _p: f"{v:,g}"))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.xaxis.set_major_formatter(_log_fmt())
            if spec.get("yscale") == "log":
                ax.set_yscale("log")
            # gridlines on the value axis only (as on screen)
            if kind == "line" or spec.get("xticks"):
                ax.grid(axis="x", visible=False)
            if spec.get("xticks"):
                ax.set_xticks(spec["xticks"]["vals"], spec["xticks"]["text"])
            items = spec.get("legend_items") or []
            if items:
                from matplotlib.lines import Line2D
                hs = [Line2D([], [], color=it["color"],
                             ls="-" if it.get("line") else "none", lw=1.4,
                             marker=it.get("mpl_marker", "o"), ms=4.5,
                             markeredgecolor="white", markeredgewidth=0.6)
                      for it in items]
                if fr and fr.get("x"):
                    hs.append(Line2D([], [], color=MUTED, lw=1.0,
                                     drawstyle="steps-post"))
                    items = items + [{"name": "Pareto frontier"}]
                if len(hs) >= 2:
                    title = ("colour = model · shape = architecture"
                             if any(str(it.get("block", "")).startswith("arch")
                                    for it in items) else None)
                    _legend(ax, spec, len(hs), (0, 1.22),
                            2 if size == "1col" else 4,
                            handles=hs, labels=[it["name"] for it in items],
                            title=title)
            else:
                handles, labels_ = ax.get_legend_handles_labels()
                if len(labels_) >= 2:
                    _legend(ax, spec, len(labels_), (0, 1.22),
                            2 if size == "1col" else 3)

        for hl in spec.get("hlines", []):
            ax.axhline(hl["y"], color=hl.get("color", MUTED),
                       ls="--" if hl.get("dash", True) else "-",
                       lw=1.0, alpha=hl.get("alpha", 0.9))
            if hl.get("label"):
                ax.annotate(hl["label"], (1.0, hl["y"]),
                            xycoords=("axes fraction", "data"),
                            xytext=(-2, 3), textcoords="offset points",
                            ha="right", fontsize=max(5.5, 6.5), color=INK2)
        for vl in spec.get("vlines", []):
            ax.axvline(vl["x"], color=vl.get("color", MUTED),
                       ls="--" if vl.get("dash", True) else "-",
                       lw=1.0, alpha=vl.get("alpha", 0.9))
            if vl.get("label"):
                ax.annotate(vl["label"], (vl["x"], 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(3, -8), textcoords="offset points",
                            fontsize=max(5.5, 6.5), color=INK2)

        if spec.get("xlabel"):
            ax.set_xlabel(spec["xlabel"])
        if spec.get("ylabel"):
            ax.set_ylabel(spec["ylabel"])
        if spec.get("xrange"):
            ax.set_xlim(spec["xrange"])
        if spec.get("yrange"):
            ax.set_ylim(spec["yrange"])
        # shade after the limits are final, or the band stops short of the
        # axis edge whenever xrange changes them
        for vl in spec.get("vlines", []):
            if vl.get("shade_beyond"):
                lo, hi = ax.get_xlim()
                ax.axvspan(vl["x"], hi, color=MUTED, alpha=0.06, lw=0)
                ax.set_xlim(lo, hi)
        if spec.get("title") and size == "slide":
            ax.set_title(spec["title"], loc="left")
        return fig


def _legend(ax, spec: dict, n: int, anchor, ncols: int, handles=None,
            labels=None, title=None):
    """Legend placement: above the axes (default), in a column to the right
    (many entries), or none (the caption carries it)."""
    where = spec.get("legend") or "auto"
    if where == "none":
        return
    hl = (handles, labels) if handles is not None else ()
    tkw = dict(title=title, title_fontsize="x-small", alignment="left") if title else {}
    if where == "right":
        ax.legend(*hl, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncols=1,
                  fontsize="x-small", handlelength=1.6, borderaxespad=0, **tkw)
        return
    rows = -(-n // max(1, ncols))
    ax.legend(*hl, loc="lower left", bbox_to_anchor=(0, 1.01), ncols=ncols,
              borderaxespad=0.2, handlelength=1.6, columnspacing=1.0,
              fontsize="small" if rows > 2 else None, **tkw)


def _log_fmt():
    """10^n tick labels on log axes, as Plotly shows them (exponentformat
    'power'), instead of SI prefixes in one renderer and powers in the other."""
    from matplotlib.ticker import LogFormatterMathtext
    return LogFormatterMathtext()


def _seq_t(val, h):
    zmin = h.get("zmin", 0) or 0
    zmax = h.get("zmax", 1) or 1
    return (val - zmin) / (zmax - zmin) if zmax > zmin else 0.5


def _is_dark(cmap, t) -> bool:
    try:
        r, g, b, _ = cmap(float(t))
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45
    except Exception:
        return False
