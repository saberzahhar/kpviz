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
            "per_doc", "log", None, None, ["y"], None)


def test_every_workbench_renders(app):
    alls = ["P", "R", "M", "U"]
    calls = {
        "rq1": (True, ["kp20k", "kpbiomed", "kptimes"], "f1", "O", alls, "auto",
                "pearson", None, None, None),
        "rq2": (True, "kpbiomed", "f1", "O", alls, "auto", None, None,
                ["lang", "sup_leak"], 0.8, "(any)", 2, None),
        "rq3": (True, "semeval2010", "f1", "O", "auto", None, None, 2, None),
        "rq4": (True, ["kp20k", "kpbiomed", "kptimes"], "r", "M", ["R", "M", "U"],
                "auto", "usd", "per_doc", "log", None, None, ["y"], None),
        "rq5": (True, "num_beams", None, ["kp20k", "kpbiomed", "kptimes"], "f1",
                "O", alls, "auto", 2, None),
    }
    from kpviz.export import fig_png
    for rq, args in calls.items():
        out = _cb(app, f'"rq":"{rq}","type":"rq-graph"')(*args)
        spec = out[1]
        assert spec and spec.get("caption"), rq
        assert fig_png(spec)[:4] == b"\x89PNG", rq       # exports render too


def test_caption_names_the_gold_actually_used(app_ctx):
    from kpviz.pages.insights_common import metric_caption
    cap = metric_caption("f1", "O", ["P", "R", "M", "U"], "auto",
                         ["kp20k", "kptimes", "semeval2010"])
    assert "author gold (kp20k)" in cap
    assert "editor gold (kptimes)" in cap
    assert "author+reader combined gold (semeval2010)" in cap
    assert "Porter2" in cap and "no padding" in cap
