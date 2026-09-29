"""The app builds, serves, and every workbench renders on the sample store.

Callbacks are called directly (as Dash would, with the workbench visible);
the browser-level checks (callback counts, idle traffic, console errors)
live in tools/bench/probe_ui.py.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def app(app_ctx):
    from kpviz.appfactory import build_app
    return build_app()


def _cb(app, needle):
    return [v for k, v in app.callback_map.items() if needle in k][0]["callback"].__wrapped__


def test_layout_and_assets_serve(app):
    c = app.server.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/_dash-layout").status_code == 200
    assert c.get("/_dash-dependencies").status_code == 200
    assert c.get("/kpviz-perf").status_code == 200


def test_hidden_workbench_does_nothing(app):
    from dash.exceptions import PreventUpdate
    rq4 = _cb(app, '"rq":"rq4","type":"rq-graph"')
    with pytest.raises(PreventUpdate):
        rq4(False, 1, ["kp20k"], "f1|O", ["P", "R", "M", "U"], "auto", "usd",
            "per_doc", "log", None, None, ["y"], 2, "rank", "holm", "t", 1000,
            1, None)


ALLS = ["P", "R", "M", "U"]
THREE = ["kp20k", "kpbiomed", "kptimes"]


def _calls(stats, v=1):
    # the last two arguments: the catalog version (an input) and the
    # workbench's last signature (a state)
    return {
        "rq1": (True, 1, THREE, "f1|O", ALLS, "auto", "kendall", None, None,
                "model", *stats, v, None),
        "rq2": (True, 1, "kpbiomed", "f1|O", ALLS, "auto", None, None,
                ["lang", "sup_leak"], 0.8, "(any)", "run", *stats, v, None),
        "rq3": (True, 1, "semeval2010", "f1|O", "auto", None, None, "run",
                *stats, v, None),
        "rq4": (True, 1, THREE, "r|M", ["R", "M", "U"], "auto", "usd",
                "per_doc", "log", None, None, ["y"], *stats, v, None),
        "rq5": (True, 1, "num_beams", None, THREE, "controlled", "f1|O", ALLS,
                "auto", *stats, v, None),
    }


@pytest.mark.parametrize("stats", [
    (2, "rank", "holm", "t", 1000),
    (2, "mean", "bh", "bootstrap", 1000),
    (1, "resample", "none", "none", 1000),
])
def test_every_workbench_renders(app, stats):
    from kpviz.export import fig_pdf, fig_png, latex_table
    for rq, args in _calls(stats).items():
        out = _cb(app, f'"rq":"{rq}","type":"rq-graph"')(*args)
        spec = out[1]
        assert spec and spec.get("caption"), rq
        assert fig_png(spec)[:4] == b"\x89PNG", rq       # exports render too
        if stats[1] == "rank":
            assert fig_pdf(spec)[0][:5] == b"%PDF-", rq
        tab = spec.get("table")
        if tab:
            for cells in ("value", "ci", "ci_n"):
                tex = latex_table(tab["headers"], tab["rows"], spec["caption"],
                                  tab.get("label", rq), cells=cells,
                                  notes=tab.get("notes"))
                assert "\\toprule" in tex and "\\bottomrule" in tex


def test_statistics_reach_captions_and_tables(app):
    out = _cb(app, '"rq":"rq3","type":"rq-graph"')(*_calls((2, "rank", "holm", "t", 1000))["rq3"])
    spec = out[1]
    assert "Wilcoxon" in spec["caption"] and "Student-t" in spec["caption"]
    assert "p (Holm" in " ".join(spec["table"]["headers"])
    assert any(isinstance(c, dict) and c.get("lo") is not None
               for row in spec["table"]["rows"] for c in row)
    assert any(s.get("err") for s in spec["series"])
    out = _cb(app, '"rq":"rq1","type":"rq-graph"')(*_calls((2, "rank", "holm", "t", 1000))["rq1"])
    assert "Kendall" in out[1]["caption"] and "Fisher-z" in out[1]["caption"]


def test_one_system_per_model_and_one_identity(app):
    """Best run per model keeps one row per model; the model's hue is the
    same in every workbench and the legend lists models, not runs."""
    cb = _cb(app, '"rq":"rq3","type":"rq-graph"')
    args = list(_calls((2, "rank", "holm", "t", 1000))["rq3"])
    args[7] = "model"
    out = cb(*args)
    specA, specB = out[1], out[5]
    rows = specA["series"][0]["x"] if specA.get("orientation") != "h" \
        else specA["series"][0]["y"]
    from kpviz import db
    n_models = db.q1("SELECT count(DISTINCT model) FROM runs "
                     "WHERE dataset='semeval2010'")[0]
    assert len(rows) <= n_models
    from kpviz.naming import model_color
    items = specB["legend_items"]
    assert items and all(it["color"] == model_color(it["group"]) for it in items)
    rq4 = _cb(app, '"rq":"rq4","type":"rq-graph"')(
        *_calls((2, "rank", "holm", "t", 1000))["rq4"])[1]
    colors4 = {it["group"]: it["color"] for it in rq4["legend_items"]
               if not it["group"].startswith("arch:")}
    for it in items:
        if it["group"] in colors4:
            assert colors4[it["group"]] == it["color"]


def test_caption_names_the_gold_actually_used(app_ctx):
    from kpviz.pages.insights_common import metric_caption
    cap = metric_caption("f1", "O", ["P", "R", "M", "U"], "auto",
                         ["kp20k", "kptimes", "semeval2010"])
    assert "author gold (kp20k)" in cap
    assert "editor gold (kptimes)" in cap
    assert "author+reader combined gold (semeval2010)" in cap
    assert "Porter2" in cap and "no padding" in cap


def test_identical_conditions_are_not_untestable():
    from kpviz.stats import StatsCfg
    r = StatsCfg().paired([0.5] * 50, [0.5] * 50)
    assert r["p"] == 1.0 and r["effect"] == 0.0


def test_rq5_controlled_sweep_and_envelope(app):
    """Controlled: one series per fixed configuration, labelled with what is
    fixed; envelope: one series per model, and the caption says it is the
    best observed run per value."""
    cb = _cb(app, '"rq":"rq5","type":"rq-graph"')
    args = list(_calls((2, "rank", "holm", "t", 1000))["rq5"])
    ctrl = cb(*args)[1]
    assert "configuration that varies it alone" in ctrl["caption"]
    args[5] = "envelope"
    env = cb(*args)[1]
    assert "best observed run at each value" in env["caption"]
    heads = env["table"]["headers"]
    # the test columns come right after the row label (N11)
    assert heads[2] == "p" and "Effect size" in heads[:6]


def test_scan_in_progress_pauses_workbenches(app, monkeypatch):
    """N1: while a scan writes the store, no workbench computes (or caches)
    anything from the half-written tables."""
    from dash.exceptions import PreventUpdate
    from kpviz import scanner
    snap = scanner.STATE.snapshot()
    fake = dict(snap, running=True,
                steps=[{"key": "inferences", "status": "running"}])
    monkeypatch.setattr(scanner.STATE, "snapshot", lambda: fake)
    with pytest.raises(PreventUpdate):
        _cb(app, '"rq":"rq4","type":"rq-graph"')(
            *_calls((2, "rank", "holm", "t", 1000), v=2)["rq4"])


def test_inspector_explains_a_runs_score(app_ctx):
    """U9: the ranked predictions of a run on one document, with what each
    matched, agree with the stored match primitives."""
    from kpviz import db
    from kpviz.pages.datasets import _explain
    m, a, r, d, ranks = db.q1("""SELECT model, arch, run_id, doc_id, pred_ranks
                                 FROM matches WHERE dataset='kp20k'
                                   AND ann_key='author' AND len(pred_ranks) > 0
                                 LIMIT 1""")
    out = str(_explain("kp20k", d, (m, a, r), "author").to_plotly_json())
    assert out.count("'✓'") == len(ranks)
    assert f"{len(ranks)} of" in out and "Missed gold" in out


def _find(node, pred, out=None):
    """Every component in a Dash tree for which pred(component) holds."""
    out = [] if out is None else out
    if pred(node):
        out.append(node)
    kids = getattr(node, "children", None)
    for k in (kids if isinstance(kids, (list, tuple)) else [kids]):
        if hasattr(k, "to_plotly_json"):
            _find(k, pred, out)
    return out


def test_dataset_figures_offer_formats(app_ctx):
    """Document length (overlaid, side by side, outlines), PRMU (bars,
    pies) and keyphrase length (side by side, overlaid) each carry every
    variant, and the export follows the one on screen."""
    from kpviz.pages.datasets import _body
    body = _body("kp20k", "(each)", None, None)
    stores = {n.id["rq"]: n.data for n in _find(
        body, lambda n: isinstance(getattr(n, "id", None), dict)
        and n.id.get("type") == "fig-variants")}
    assert set(stores["ds-len"]) == {"overlay", "group", "lines"}
    assert set(stores["ds-prmu"]) == {"bars", "pies"}
    assert set(stores["ds-kplen"]) == {"group", "overlay"}
    assert stores["ds-len"]["overlay"]["spec"]["kind"] == "hist"
    assert stores["ds-prmu"]["pies"]["spec"]["kind"] == "pie_grid"
    for variants in stores.values():
        for v in variants.values():
            assert v["figure"]["data"] and v["spec"]["caption"]


def test_paper_preview_and_tex_download(app):
    """The printed figure is rendered at the venue's width with its caption,
    and the TeX download is the figure environment the snippet copies."""
    from kpviz.pages.datasets import _body
    body = _body("kp20k", "(each)", None, None)
    spec = [n.data for n in _find(
        body, lambda n: getattr(n, "id", None) == {"type": "fig-spec", "rq": "ds-len"})][0]
    prev = _cb(app, '"type":"exp-preview"')
    out = prev(1, spec, "My caption.", "acl", "col", "std", "auto", "ci")
    js = str(out.to_plotly_json())
    assert "data:image/png;base64," in js and "My caption." in js
    from kpviz.figures import geometry
    w = geometry(dict(spec, export={"venue": "acl", "span": "col"}))[0]
    assert "ACL" in js and f"{w:.2f} ×" in js and f"{w * 96:.0f}px" in js
    dl = _cb(app, '"type":"exp-dl"')
    from contextvars import copy_context
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    def run():
        context_value.set(AttributeDict(**{"triggered_inputs": [
            {"prop_id": '{"rq":"ds-len","type":"exp-btn","what":"tex"}.n_clicks',
             "value": 1}]}))
        return dl([0, 0, 0, 1, 0], spec, "My caption.", None, None, None, None, None)
    got, status = copy_context().run(run)
    assert got["filename"].endswith(".tex") and "\\begin{figure" in got["content"]
    assert "My caption." in got["content"] and status.startswith("Downloaded")
