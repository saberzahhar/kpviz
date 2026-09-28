"""Exports compile inside real paper templates, at their printed size.

Every workbench's figure is rendered as PGF for a venue and \\input into a
document of that venue's class, together with its booktabs table; the
build must succeed and nothing may run into the margin (an Overfull \\hbox
of more than 1 pt is a failure). Skipped without a TeX installation."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(not shutil.which("pdflatex"),
                                reason="no TeX installation")

CLASSES = {
    "generic": r"\documentclass[twocolumn]{article}",
    "acl": r"\documentclass[11pt,twocolumn]{article}"
           "\n\\usepackage[a4paper,margin=2.5cm,columnsep=0.6cm]{geometry}"
           "\n\\usepackage{times}",
    "ieee": r"\documentclass[conference]{IEEEtran}",
    "lncs": r"\documentclass{llncs}",
    "acm": r"\documentclass[sigconf]{acmart}",
}
# classes that need a title block before any content
TITLE = {"acm": "\\title{KPViz export test}\\author{A. Author}\\maketitle\n"}


def _has_class(venue):
    cls = re.search(r"\{(\w+)\}$", CLASSES[venue].splitlines()[0]).group(1)
    return bool(subprocess.run(["kpsewhich", cls + ".cls"], capture_output=True,
                               text=True).stdout.strip())


def _compile(tmp, body, venue, engine="pdflatex"):
    doc = (CLASSES[venue] + "\n\\usepackage{pgf}\n\\usepackage{booktabs}"
           "\n\\usepackage{graphicx}\n\\usepackage{adjustbox}\n\\begin{document}\n"
           + TITLE.get(venue, "") + "\\section{Results}\nText.\n" + body + "\n\\end{document}\n")
    (tmp / "main.tex").write_text(doc)
    r = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                        "main.tex"], cwd=tmp, capture_output=True, text=True,
                       timeout=300)
    log = (tmp / "main.log").read_text(errors="replace")
    return r.returncode, log


@pytest.fixture(scope="module")
def specs(app_ctx):
    """One spec per workbench panel, rendered by the real callbacks."""
    from kpviz.appfactory import build_app
    from test_ui import _cb, _calls
    app = build_app()
    out = {}
    for rq, args in _calls((2, "rank", "holm", "t", 1000)).items():
        res = _cb(app, f'"rq":"{rq}","type":"rq-graph"')(*args)
        out[rq] = res[1]
        if rq == "rq3":
            out["rq3b"] = res[5]
    return out


def _body(spec, name, tmp, venue, span, legend="auto"):
    from kpviz.export import (fig_pgf, figure_env, latex_figure, latex_table)
    from kpviz.figures import geometry
    spec = dict(spec, export={"venue": venue, "span": span, "legend": legend,
                              "cells": "ci"})
    pgf, err = fig_pgf(spec)
    assert pgf, err
    (tmp / f"{name}.pgf").write_text(pgf)
    body = latex_figure(f"{name}.pgf", spec["caption"], name, pgf=True,
                        env=figure_env(spec), dims=geometry(spec))
    tab = spec.get("table")
    if tab:
        from kpviz.export import table_caption
        body += "\n" + latex_table(tab["headers"], tab["rows"],
                                   table_caption(spec),
                                   name, cells="ci", notes=tab.get("notes"),
                                   venue=venue)
    return body


def _overfull(log):
    return [float(m) for m in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide", log)]


@pytest.mark.parametrize("venue", ["generic", "acl", "ieee", "lncs", "acm"])
def test_every_figure_compiles_in_venue(specs, tmp_path, venue):
    from kpviz.export import tex_engine
    if not tex_engine():
        pytest.skip("TeX present but no engine passed the probe")
    if not _has_class(venue):
        pytest.skip(f"{venue} class not installed")
    if venue == "acm":
        code, log = _compile(tmp_path, "", venue)
        if code:
            pytest.skip("acmart installed without its fonts")
    body = "\n\n".join(_body(spec, name, tmp_path, venue, "auto")
                       for name, spec in specs.items())
    # a column-wide variant with the legend moved to the right
    body += "\n\n" + _body(specs["rq3b"], "rq3b-col", tmp_path, venue, "col", "right")
    code, log = _compile(tmp_path, body, venue)
    assert code == 0, log[-3000:]
    assert (tmp_path / "main.pdf").stat().st_size > 10_000
    over = [w for w in _overfull(log) if w > 1.0]
    assert not over, f"content runs into the margin by {over} pt"


@pytest.mark.parametrize("engine", ["xelatex", "lualatex"])
def test_other_engines(specs, tmp_path, engine):
    if not shutil.which(engine):
        pytest.skip(engine)
    body = _body(specs["rq2"], "rq2", tmp_path, "generic", "auto")
    code, log = _compile(tmp_path, body, "generic", engine)
    assert code == 0, log[-3000:]


# ---- export robustness (second review round: N2, N5, N15) --------------------
def _bar(label, name="t"):
    return {"kind": "bar", "size": "1col", "name": name, "xlabel": "x",
            "ylabel": "F1", "series": [{"name": "s", "x": [label, "b"],
                                        "y": [0.5, 0.6], "color": "#2a78d6"}]}


@pytest.mark.parametrize("label", [r"prompt=Extract\nkeyphrases", "cost $ per doc",
                                   "50% of it", "x^2", "a_b & c #1 {d} ~e"])
def test_tex_specials_in_labels_typeset(label):
    from kpviz import export
    pdf, method, note = export.fig_pdf(_bar(label, "spec-" + label))
    assert pdf[:5] == b"%PDF-" and method.startswith("TeX"), (method, note)
    assert not export._demoted


def test_figure_failure_does_not_demote_the_engine(monkeypatch):
    """A figure TeX cannot set falls back for *that* figure; the engine stays
    offered as long as the probe figure still compiles."""
    from kpviz import export, figures
    real = figures.render
    calls = {"n": 0}

    def flaky(spec, tex, fmt, **kw):
        if spec.get("name") == "unsettable" and kw.get("backend") == "pgf":
            calls["n"] += 1
            raise RuntimeError("LaTeX was not able to process\n! Undefined control sequence.")
        return real(spec, tex, fmt, **kw)
    monkeypatch.setattr(export, "render", flaky)
    pdf, method, note = export.fig_pdf(_bar("fine", "unsettable"))
    assert pdf[:5] == b"%PDF-" and method == "Matplotlib"
    assert "could not typeset this figure" in note
    assert not export._demoted
    ok_pdf, ok_method, _ = export.fig_pdf(_bar("fine", "settable"))
    assert ok_method.startswith("TeX")


def test_pdf_during_probe_never_waits_on_the_render_lock(tmp_path, monkeypatch):
    """N15: the engine is resolved before the render lock; a PDF asked for
    while the probe runs returns as soon as the probe does."""
    import time
    from kpviz import export
    monkeypatch.setattr(export, "_probe_file", lambda: tmp_path / "probe.json")
    monkeypatch.setitem(export._probe, "status", "idle")
    export.start_tex_probe()
    t0 = time.time()
    _pdf, method, _note = export.fig_pdf(_bar("race", "race"))
    assert time.time() - t0 < 60 and method.startswith("TeX")


def test_snippet_download_and_bundle_agree_on_names():
    import io
    import zipfile
    from kpviz import export
    spec = dict(_bar("a", "pareto usd"), export={"venue": "acl"}, caption="c")
    stem, ref = export.export_name(spec)
    assert (stem, ref) == ("pareto-usd-acl", "figures/pareto-usd-acl")
    z = zipfile.ZipFile(io.BytesIO(export.export_bundle(spec)))
    names = set(z.namelist())
    assert {f"{stem}.pdf", f"{stem}.png", f"{stem}.tex", "provenance.json"} <= names
    tex = z.read(f"{stem}.tex").decode()
    assert f"{ref}.pgf" in tex or f"{ref}.pdf" in tex
    readme = z.read("README.txt").decode()
    assert "Verified to compile" not in readme and "PDF rendered via: TeX" in readme
