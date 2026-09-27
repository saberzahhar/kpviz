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
        rq4(False, ["kp20k"], "f1", "O", ["P", "R", "M", "U"], "auto", "usd",
            "per_doc", "log", None, None, ["y"], 2, "rank", "holm", "t", 1000,
            None)


ALLS = ["P", "R", "M", "U"]
THREE = ["kp20k", "kpbiomed", "kptimes"]


def _calls(stats):
    return {
        "rq1": (True, THREE, "f1", "O", ALLS, "auto", "kendall", None, None,
                *stats, None),
        "rq2": (True, "kpbiomed", "f1", "O", ALLS, "auto", None, None,
                ["lang", "sup_leak"], 0.8, "(any)", *stats, None),
        "rq3": (True, "semeval2010", "f1", "O", "auto", None, None, *stats, None),
        "rq4": (True, THREE, "r", "M", ["R", "M", "U"], "auto", "usd",
                "per_doc", "log", None, None, ["y"], *stats, None),
        "rq5": (True, "num_beams", None, THREE, "f1", "O", ALLS, "auto",
                *stats, None),
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
