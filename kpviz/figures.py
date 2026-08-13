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
    import PIL.Image  # noqa: F401
except Exception:
    pass

# ---- palette / chrome (light mode) ---------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
DIV_LOW, DIV_MID, DIV_HIGH = "#2a78d6", "#f0efec", "#e34948"
from .naming import PALETTE  # categorical slots (fixed order)  # noqa: E402
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# a second categorical dimension on bars without touching the hue: Plotly
# pattern shapes and their Matplotlib hatch equivalents
PATTERNS = ["", "/", "x", "."]
_MPL_HATCH = {"": None, "/": "///", "x": "xxx", ".": "..."}

SIZES = {"1col": (3.35, 2.5), "2col": (7.0, 3.1), "slide": (7.5, 4.3)}
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

def _plotly_layout(spec: dict) -> dict:
    n_series = len(spec.get("series", []))
    show_legend = spec.get("show_legend",
                           (n_series >= 2 or spec.get("kind") == "dumbbell")
                           and spec.get("kind") != "heatmap")
    lay = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, size=13, color=INK),
        margin=dict(l=56, r=24, t=30 if spec.get("title") else 12, b=48),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12, color=INK2),
                    title=dict(text=spec.get("legend_title") or "")),
        xaxis=dict(title=dict(text=spec.get("xlabel") or "", font=dict(size=12, color=INK2)),
                   gridcolor=GRID, gridwidth=1, zeroline=False,
                   linecolor=BASELINE, linewidth=1,
                   tickfont=dict(size=11, color=MUTED)),
        yaxis=dict(title=dict(text=spec.get("ylabel") or "", font=dict(size=12, color=INK2)),
                   gridcolor=GRID, gridwidth=1, zeroline=False,
                   linecolor=BASELINE, linewidth=1,
                   tickfont=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(family=FONT_STACK, size=12, color=INK)),
    )
    # axis scale only when explicitly requested (never force 'linear' onto
    # categorical axes — string categories would coerce to NaN)
    if spec.get("xtickangle"):
        lay["xaxis"]["tickangle"] = spec["xtickangle"]
        lay["margin"]["b"] = 84
    if spec.get("xscale"):
        lay["xaxis"]["type"] = spec["xscale"]
    if spec.get("yscale"):
        lay["yaxis"]["type"] = spec["yscale"]
    if len(spec.get("series", [])) >= 2 and spec.get("barmode") == "stack":
        lay["legend"]["traceorder"] = "normal"
    if spec.get("title"):
        lay["title"] = dict(text=spec["title"], font=dict(size=14, color=INK),
                            x=0, xanchor="left")
    if spec.get("xrange"):
        lay["xaxis"]["range"] = spec["xrange"]
    if spec.get("yrange"):
        lay["yaxis"]["range"] = spec["yrange"]
    if spec.get("kind") == "bar":
        lay["barmode"] = spec.get("barmode", "group")
        lay["bargap"] = 0.35
        lay["bargroupgap"] = 0.12
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
        colorscale = ([[0, DIV_LOW], [0.5, DIV_MID], [1, DIV_HIGH]]
                      if h.get("diverging")
                      else [[i / (len(SEQ_RAMP) - 1), c] for i, c in enumerate(SEQ_RAMP)])
        text = h.get("text")
        fig.add_trace(go.Heatmap(
            z=h["z"], x=h["x"], y=h["y"],
            zmin=h.get("zmin"), zmax=h.get("zmax"), zmid=h.get("zmid"),
            colorscale=colorscale, xgap=2, ygap=2,
            text=text, texttemplate="%{text}" if text else None,
            textfont=dict(size=11, color=INK),
            hovertemplate=h.get("hover", "%{y} × %{x}: %{z:.3f}<extra></extra>"),
            colorbar=dict(thickness=10, outlinewidth=0,
                          tickfont=dict(size=10, color=MUTED))))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False)
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
        for j, (kx, name, color) in enumerate(points):
            fig.add_trace(go.Scatter(
                x=[r.get(kx) for r in rows], y=list(range(len(rows))),
                mode="markers", name=name,
                marker=dict(size=11, color=color,
                            line=dict(color=SURFACE, width=2)),
                customdata=[(r.get("hover") or [])[j]
                            if isinstance(r.get("hover"), list)
                            and len(r["hover"]) > j else "" for r in rows],
                hovertemplate="%{customdata}<extra>" + name + "</extra>"))
        fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=ylabels,
                         showgrid=False, autorange="reversed")
        return fig

    # scatter / line / bar families ----------------------------------------
    anchors = place_labels(spec) if kind != "bar" else {}
    for si, s in enumerate(spec.get("series", [])):
        common = dict(name=s.get("name", ""),
                      showlegend=bool(s.get("in_legend", True)))
        hover = s.get("hover")
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
        marker = dict(size=s.get("size", 10),
                      color=s.get("color", "#2a78d6"),
                      symbol=s.get("shape", "circle"),
                      opacity=s.get("alpha", 1.0),
                      line=dict(color=SURFACE, width=2))
        line = dict(color=s.get("color", "#2a78d6"),
                    width=s.get("width", 2),
                    dash="dash" if s.get("dash") else "solid")
        pos = [anchors.get((si, pi), "middle right")
               for pi in range(len(s.get("text") or []))] or "middle right"
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
            line=dict(color=MUTED, width=1.4, shape="hv"),
            opacity=0.85, showlegend=True, hoverinfo="skip"))

    for hl in spec.get("hlines", []):
        fig.add_hline(y=hl["y"], line_color=hl.get("color", MUTED),
                      line_dash="dash" if hl.get("dash", True) else "solid",
                      line_width=1.4, opacity=hl.get("alpha", 0.9),
                      annotation_text=hl.get("label", ""),
                      annotation_position="top left",
                      annotation_font=dict(size=11, color=INK2))
    for vl in spec.get("vlines", []):
        fig.add_vline(x=vl["x"], line_color=vl.get("color", MUTED),
                      line_dash="dash" if vl.get("dash", True) else "solid",
                      line_width=1.4, opacity=vl.get("alpha", 0.9),
                      annotation_text=vl.get("label", ""),
                      annotation_position="top",
                      annotation_font=dict(size=11, color=INK2))
        if vl.get("shade_beyond"):
            xr = spec.get("xrange")
            x1 = xr[1] if xr else None
            if x1 is None:
                xs = [max(s["x"]) for s in spec.get("series", []) if s.get("x")]
                x1 = max(xs) * 1.05 if xs else vl["x"] * 2
            fig.add_vrect(x0=vl["x"], x1=x1, fillcolor=MUTED,
                          opacity=0.06, line_width=0)
    return fig


def plotly_config(name: str = "kpviz") -> dict:
    return {"displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 3,
                                     "filename": name},
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


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
    "★": r"$\star$", "•": r"$\bullet$", " ": "~", " ": r"\,",
    " ": r"\,",
}
_TEX_TABLE = str.maketrans(_TEX_CHARS)
# Matplotlib 3.10's PGF backend neutralises only "^" and "%", so a label like
# "bart-base-kp20k (num_beams=4)" writes a raw underscore into the .pgf and the
# figure fails to compile the moment it is \input into a real paper. Escape the
# rest ourselves — in text-bearing fields only, because "#2a78d6" is a colour.
_TEX_ASCII = {"_": r"\_", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}",
              "~": r"\textasciitilde{}"}
_TEX_ASCII_RE = re.compile(r"[_&#{}~]")
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


def _mpl_rc(size: str, pgf: bool) -> dict:
    base = 8 if size in ("1col", "2col") else 11
    rc = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": base, "axes.titlesize": base + 1,
        "axes.labelsize": base, "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1, "legend.fontsize": base - 1,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
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
    figsize = SIZES.get(size, SIZES["2col"])
    kind = spec.get("kind", "scatter")

    if kind == "pie_grid":
        return _mpl_pie_grid(spec, plt, size, figsize, pgf)

    with plt.rc_context(_mpl_rc(size, pgf)):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        if kind == "heatmap":
            h = spec["heat"]
            z = h["z"]
            if h.get("diverging"):
                cmap = LinearSegmentedColormap.from_list(
                    "kpdiv", [DIV_LOW, DIV_MID, DIV_HIGH])
                vmin = h.get("zmin", -1)
                vmax = h.get("zmax", 1)
                mid = h.get("zmid", 0)
                mid = min(max(mid, vmin + 1e-9), vmax - 1e-9)
                norm = TwoSlopeNorm(vcenter=mid, vmin=vmin, vmax=vmax)
                im = ax.imshow(z, cmap=cmap, norm=norm, aspect="auto")
            else:
                cmap = LinearSegmentedColormap.from_list("kpseq", SEQ_RAMP)
                im = ax.imshow(z, cmap=cmap, vmin=h.get("zmin"),
                               vmax=h.get("zmax"), aspect="auto")
            ax.set_xticks(range(len(h["x"])), h["x"], rotation=30,
                          ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(h["y"])), h["y"])
            ax.grid(False)
            if h.get("text"):
                for i, row in enumerate(h["text"]):
                    for j, t in enumerate(row):
                        if t not in (None, ""):
                            val = z[i][j]
                            dark = _is_dark(cmap, norm(val) if h.get("diverging")
                                            else _seq_t(val, h))
                            ax.text(j, i, str(t), ha="center", va="center",
                                    fontsize=max(5.5, (6 if size == "1col" else 7)),
                                    color="white" if dark else INK)
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
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
            for j, (kx, name, color) in enumerate(points):
                ax.scatter([r.get(kx) for r in rows], list(ys), s=42,
                           color=color, zorder=2 + j, label=name,
                           edgecolors="white", linewidths=1.2)
            ax.set_yticks(list(ys), [r["label"] for r in rows])
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12),
                      ncols=min(3, len(points)))
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
            width = 0.72 / (1 if stacked else n)
            bottoms = {c: 0.0 for c in cats}
            for si, s in enumerate(spec.get("series", [])):
                pos = [cats.index(xv) for xv in s[cat_key]]
                offs = ([p for p in pos] if stacked else
                        [p - 0.36 + width * (si + 0.5) for p in pos])
                bots = [bottoms[xv] for xv in s[cat_key]] if stacked else None
                hatch = _MPL_HATCH.get(s.get("pattern", ""), None)
                bar_kw = dict(color=s.get("color", "#2a78d6"),
                              alpha=s.get("alpha", 1.0),
                              label=s.get("name", ""), linewidth=0,
                              hatch=hatch, edgecolor=SURFACE)
                if horiz:
                    ax.barh(offs, s[val_key], height=width * 0.92, left=bots,
                            **bar_kw)
                else:
                    ax.bar(offs, s[val_key], width=width * 0.92, bottom=bots,
                           **bar_kw)
                if s.get("text") and not stacked:
                    for xv, yv, t in zip(offs, s[val_key], s["text"]):
                        if t and yv is not None:
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
            else:
                long_cat = any(len(str(c)) > 8 for c in cats)
                ax.set_xticks(range(len(cats)), [str(c) for c in cats],
                              rotation=(20 if long_cat else 0),
                              ha="right" if long_cat else "center")
                ax.grid(axis="x", visible=False)
            if len(spec.get("series", [])) >= 2:
                ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16),
                          ncols=min(4, len(spec["series"])))
        else:  # scatter / line
            anchors = place_labels(raw_spec,
                                   "mpl1col" if size == "1col" else "mpl")
            for si, s in enumerate(spec.get("series", [])):
                mode = s.get("mode", "markers")
                if "lines" in mode:
                    ax.plot(s["x"], s["y"], color=s.get("color", "#2a78d6"),
                            lw=s.get("width", 1.6),
                            ls="--" if s.get("dash") else "-",
                            marker=(s.get("mpl_marker", "o")
                                    if "markers" in mode else None),
                            ms=4.5, markeredgecolor="white",
                            markeredgewidth=0.8,
                            alpha=s.get("alpha", 1.0),
                            label=s.get("name", "") if s.get("in_legend", True) else None)
                else:
                    ax.scatter(s["x"], s["y"],
                               s=(s.get("size", 10) ** 2) * 0.55,
                               c=s.get("color", "#2a78d6"),
                               marker=s.get("mpl_marker", "o"),
                               alpha=s.get("alpha", 1.0),
                               edgecolors="white", linewidths=1.0,
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
                ax.plot(fr["x"], fr["y"], drawstyle="steps-post", color=MUTED,
                        lw=1.0, alpha=0.9, label="Pareto frontier", zorder=2)
            if spec.get("xscale") == "log":
                ax.set_xscale("log")
            if spec.get("yscale") == "log":
                ax.set_yscale("log")
            handles, labels_ = ax.get_legend_handles_labels()
            if len(labels_) >= 2:
                ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22),
                          ncols=2 if size == "1col" else 3)

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
            if vl.get("shade_beyond"):
                ax.axvspan(vl["x"], ax.get_xlim()[1], color=MUTED, alpha=0.06,
                           lw=0)

        if spec.get("xlabel"):
            ax.set_xlabel(spec["xlabel"])
        if spec.get("ylabel"):
            ax.set_ylabel(spec["ylabel"])
        if spec.get("xrange"):
            ax.set_xlim(spec["xrange"])
        if spec.get("yrange"):
            ax.set_ylim(spec["yrange"])
        if spec.get("title") and size == "slide":
            ax.set_title(spec["title"], loc="left")
        return fig


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
