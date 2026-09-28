"""The interface contract: what a reader sees must say what the scorer did.

These tests need no scanned store. They pin the rules the pages rely on —
gold highlighted exactly where the scorer finds it, costs in readable
units, issues grouped by kind with a severity and a meaning, one colour per
PRMU class in Python and CSS alike, deep links to every workbench — so a
later edit cannot quietly break them.
"""
from __future__ import annotations

import re
import sys

from conftest import REPO

sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------- gold marks
def test_present_spans_follow_the_scorer():
    """Contiguous stems in order, every occurrence, merged when they touch,
    never across a separating mark."""
    from kpviz.highlight import present_spans
    text = "Neural networks learn. A neural, network is not one."
    spans = present_spans(text, ["neural network"], "en")
    assert [text[a:b] for a, b, _ in spans] == ["Neural networks"]
    # overlapping phrases become one span listing both
    t2 = "deep neural network models"
    sp2 = present_spans(t2, ["deep neural", "neural network"], "en")
    assert len(sp2) == 1 and sorted(sp2[0][2]) == [0, 1]
    assert t2[sp2[0][0]:sp2[0][1]] == "deep neural network"
    # nothing to mark, nothing marked
    assert present_spans("", ["x"], "en") == []
    assert present_spans("text", [], "en") == []


def test_present_spans_keep_original_offsets():
    """Offsets index the original text, ligatures and accents included."""
    from kpviz.highlight import present_spans
    text = "Une ﬁne étude: résumé automatique."
    spans = present_spans(text, ["résumé automatique", "fine"], "fr")
    got = sorted(text[a:b] for a, b, _ in spans)
    assert got == ["résumé automatique", "ﬁne"]


# ---------------------------------------------------------------- PRMU filter
def test_empty_prmu_selection_is_not_all():
    from kpviz.pages.insights_common import prmu_arg, prmu_empty
    assert prmu_arg(None) is None
    assert prmu_arg(["P", "R", "M", "U"]) is None
    assert prmu_arg(["P"]) == ["P"]
    assert prmu_empty([]) and not prmu_empty(None) and not prmu_empty(["U"])


# ---------------------------------------------------------------- costs
def test_cost_scale_picks_a_readable_unit():
    from kpviz.pages.rq.rq4 import cost_scale
    assert cost_scale([0.2, 0.5], "per_doc") == (1.0, "per document")
    assert cost_scale([9e-6, 2e-5, 5e-4], "per_doc") == (1e3, "per 1,000 documents")
    assert cost_scale([1e-9, 2e-9], "per_doc") == (1e6, "per million documents")
    assert cost_scale([3.0], "total") == (1.0, "per run")
    assert cost_scale([0, None], "per_doc") == (1.0, "per document")


def test_fmt_cost_never_prints_e_notation():
    from kpviz.pages.rq.rq4 import fmt_cost
    for v in (0.00913, 9.13e-06, 1.2e-9, 0.0, 123456.0):
        assert "e" not in fmt_cost(v, "usd").lower(), v
    assert fmt_cost(0.00913, "usd") == "$0.00913"
    assert fmt_cost(None, "usd") == "—"


def test_human_count_billions():
    from kpviz.util import human_count
    assert human_count(70_600_000_000) == "70.6 B"
    assert human_count(139_000_000) == "139 M"
    assert human_count(1_500_000) == "1.50 M"
    assert human_count(12_345) == "12.3 k"
    assert human_count(None) == "—"


# ---------------------------------------------------------------- issues
def test_issue_tags_group_by_kind():
    """Six illegal parameters are one row listing six names; a missing run
    and a missing model stay two rows (they mean different things)."""
    from kpviz.pages.home import _span, _tag_group
    assert _tag_group("illegal parameter:alpha") == ("illegal parameter", "alpha")
    assert _tag_group("missing:run") == ("missing:run", "")
    assert _tag_group("missing:model") == ("missing:model", "")
    assert _tag_group("unscored:20") == ("unscored", "20")
    assert _span(["alpha", "pos", "method"]) == "alpha, method, pos"
    assert _span(["3", "20"], "unscored") == "3–20 documents"
    assert _span(["20%", "54%"], "incomplete") == "coverage 20–54%"


def test_every_issue_kind_has_a_severity_and_a_meaning():
    """Each tag the scanner writes (and each document flag) is explained on
    the Overview — an empty 'what it means' cell is a bug."""
    from kpviz.pages.home import SEVERITY, meaning, severity
    src = (REPO / "kpviz" / "scanner.py").read_text() + \
        (REPO / "kpviz" / "derive.py").read_text()
    kinds = set(re.findall(r'tags\.append\(f?"([a-z_ ]+(?::[a-z_ ]+)?)', src))
    kinds |= {k.split(":")[0] for k in
              re.findall(r'flags\.append\(f"([a-z_]+):', src)}
    kinds = {k if k.startswith(("missing:", "unreadable:")) else k.split(":")[0]
             for k in kinds}
    # written through variables, not literals: per-parameter, per-tokenizer,
    # per-card and the scanner's per-run and per-collection counters
    kinds |= {"illegal parameter", "approximate tokens", "unreadable card",
              "duplicate_docs", "duplicate_doc_ids", "gold_duplicates",
              "gold_empty", "malformed_lines", "missing_id"}
    assert {"illegal parameter", "missing:run", "missing_section",
            "lang_mismatch", "unscored"} <= kinds
    for k in kinds:
        assert meaning(k), f"no meaning for {k!r}"
        assert 0 <= severity(k) < len(SEVERITY)
    assert severity("missing:dataset") == 0
    assert severity("illegal parameter") == 1
    assert severity("lang_mismatch") == 2


# ---------------------------------------------------------------- one palette
def test_prmu_colours_match_between_python_and_css():
    from kpviz.naming import PRMU_COLORS
    css = (REPO / "kpviz" / "assets" / "kpviz.css").read_text()
    for c, hexv in PRMU_COLORS.items():
        m = re.search(rf"--prmu-{c}:\s*(#[0-9a-fA-F]{{6}})", css)
        assert m and m.group(1).lower() == hexv.lower(), c


def test_split_colours_are_the_validated_set():
    """The split hues were validated as a categorical palette (lightness,
    chroma, colour-vision separation, contrast); aliases share them."""
    from kpviz.naming import split_color
    assert split_color("train") == split_color("training") == "#1a9e8f"
    assert split_color("dev") == split_color("validation") == "#c2761c"
    assert split_color("test") == split_color("testing") == "#5b50c8"


# ---------------------------------------------------------------- navigation
def test_workbench_lives_in_the_url_hash(app_ctx):
    """/insights#rq3 opens RQ3; a tab click writes the hash, so reload, Back
    and pasted links all land on the same workbench."""
    js = (REPO / "kpviz" / "assets" / "route.js").read_text()
    assert "route: function (pathname, hash, lastRq)" in js
    assert 'return "#" +' in js
    from kpviz.appfactory import build_app
    app = build_app()
    routes = [cb for cb in app.callback_map if "page-" in cb and "tab-rq1" in cb]
    assert routes, "route() must drive the tab classes"
    inputs = {i["id"] + "." + i["property"]
              for cb in routes for i in app.callback_map[cb]["inputs"]}
    assert "url.hash" in inputs and "url.pathname" in inputs


# ---------------------------------------------------------------- tables
def test_clickable_rows_have_a_real_button():
    from dash import html
    from kpviz import ui
    t = ui.table(["Doc", "Words"], [[html.Code("d1"), 3]], row_ids=["d1"],
                 table_id="ds-doc")
    row = t.children.children[1].children[0]
    assert row.className == "row-click"
    btn = row.children[0].children
    assert isinstance(btn, html.Button) and btn.id == {"type": "ds-doc-row", "key": "d1"}


def test_wide_cell_spans_the_rest_of_the_row():
    from kpviz import ui
    t = ui.table(["Run", "Window", "a", "b", "c"],
                 [["r", "no window", ui.Wide("nothing to split")]])
    tds = t.children.children[1].children[0].children
    assert len(tds) == 3 and tds[2].colSpan == 3


# ---------------------------------------------------------------- labels
def test_point_labels_avoid_interval_whiskers():
    """A label never runs across another point's interval (the RQ4 case:
    a frontier label to the right of its point, a whisker just beyond)."""
    from kpviz.figures import GEOM, _norm_axis, _overlap, _rect, place_labels
    label = "bart-base-kp20k (num_beams=10)"
    spec = {"kind": "scatter", "xscale": "log", "series": [
        {"x": [0.0002, 0.5], "y": [0.42, 0.6]},
        {"x": [0.03], "y": [0.55], "text": [label]},
        {"x": [0.05], "y": [0.56], "err": [(0.50, 0.60)]},
    ]}
    char_w, line_h, pad_x, pad_y = GEOM["plotly"]
    px, x0, x1 = _norm_axis([0.0002, 0.5, 0.03, 0.05], "log")
    py, y0, y1 = _norm_axis([0.42, 0.6, 0.55, 0.56], None)
    nx = lambda v: (px(v) - x0) / (x1 - x0)          # noqa: E731
    ny = lambda v: (py(v) - y0) / (y1 - y0)          # noqa: E731
    whisker = (nx(0.05) - 0.006, ny(0.50), nx(0.05) + 0.006, ny(0.60))

    def box(anchor):
        return _rect(anchor, nx(0.03), ny(0.55), len(label) * char_w, line_h,
                     pad_x, pad_y)
    assert _overlap(box("middle right"), whisker) > 0      # the trap is real
    chosen = place_labels(spec)[(1, 0)]
    assert _overlap(box(chosen), whisker) == 0, chosen
