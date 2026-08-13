#!/usr/bin/env python3
"""Compile a real KPViz export with every installed TeX engine.

    python tools/test_latex_compile.py [--data sample_data] [--state DIR]

An exported figure is only useful if it typesets inside the user's paper, so
this drives the actual export functions on deliberately hostile text
(underscores from hyperparameter names, em dashes, significance daggers, %, #,
&, >=, Greek) and then compiles the emitted .tex + .pgf + table with pdflatex,
xelatex and lualatex. Exit code is non-zero if any available engine fails.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from kpviz.config import init_settings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="sample_data")
ap.add_argument("--state", default="/tmp/kpviz_state_test")
ap.add_argument("--keep", action="store_true", help="keep the build directory")
args = ap.parse_args()
init_settings(args.data, state_dir=args.state)

from kpviz.export import (fig_pgf, latex_figure,  # noqa: E402
                          latex_table, tex_engine)

HOSTILE = {
    "kind": "scatter", "size": "2col", "xscale": "log",
    "xlabel": "cost (USD) — per document, log scale",
    "ylabel": "F1@O (macro over 3 datasets)",
    "series": [
        {"name": "bart_base — 100 % coverage", "x": [9.31e-06], "y": [0.421],
         "color": "#2a78d6", "mpl_marker": "o", "in_legend": False,
         "text": ["bart-base-kp20k (num_beams=1) ‡"], "show_text": True},
        {"name": "gpt-4o & friends", "x": [4.7e-04], "y": [0.603],
         "color": "#eb6834", "mpl_marker": "s", "in_legend": False,
         "text": ["gpt-4o — #1 (100 % of docs)"], "show_text": True},
        {"name": "MultipartiteRank", "x": [2.34e-07], "y": [0.428],
         "color": "#0f9b8e", "mpl_marker": "^", "in_legend": False,
         "text": ["MultipartiteRank (α=1.1, threshold≥0.74, pos={ADJ,NOUN})"],
         "show_text": True},
    ],
    "hlines": [{"y": 0.552, "dash": True,
                "label": "Llama-3.3-70B-Instruct — no usd"}],
    "frontier": {"x": [2.34e-07, 4.7e-04], "y": [0.428, 0.603]},
    "name": "hostile",
}
CAPTION = ("Cost–performance over kp20k — daggers: † p<0.05, ‡ p<0.01; "
           "similarity ≥ 0.80 · train→test leakage; α=1.1 ± 0.02; "
           "100 % coverage; size = #parameters; A & B.")
# the shapes the insight tables really emit: inline sample sizes, a signed
# delta carrying a significance mark, an em-dash for "not applicable"
TABLE = (["Run", "Context window", "full-document (n=29)",
          "document truncated to model's context window", "Δ (w/o − w/)"],
         [["bart-base-kp20k (num_beams=4)", "512 (bart-base)",
           "0.415 (n=29)", "0.550 (n=29)", "+0.135 †"],
          ["gpt-4o — 100 %", "128k (o200k, default)", "0.451 (n=29)",
           "0.451 (n=29)", "+0.000"],
          ["MultipartiteRank (α=1.1)", "no window", "0.257 (n=29)", "—", "—"]],
         "Impact of data-quality filtering — 100 % of documents & runs; "
         "daggers mark p<0.05.")


def main() -> int:
    print("tex engine reported by KPViz:", tex_engine())
    pgf = fig_pgf(HOSTILE)
    if not pgf:
        print("no TeX distribution — nothing to compile")
        return 0
    d = Path(tempfile.mkdtemp(prefix="kpviz-tex-"))
    (d / "hostile.pgf").write_text(pgf)
    body = (latex_figure("hostile.pgf", CAPTION, "hostile", pgf=True) + "\n\n"
            + latex_table(TABLE[0], TABLE[1], TABLE[2], "hostile"))
    (d / "figure.tex").write_text(body)
    (d / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage{pgf}\n"
        "\\begin{document}\n\\input{figure.tex}\n\\end{document}\n")
    ok = True
    for eng in ("pdflatex", "xelatex", "lualatex"):
        if not shutil.which(eng):
            print(f"{eng:<10} not installed — skipped")
            continue
        pdf = d / "main.pdf"
        pdf.unlink(missing_ok=True)
        r = subprocess.run([eng, "-interaction=nonstopmode", "-halt-on-error",
                            "main.tex"], cwd=d, capture_output=True, text=True)
        if pdf.exists() and pdf.stat().st_size > 1000:
            print(f"{eng:<10} OK   {pdf.stat().st_size:>7} bytes")
            pdf.rename(d / f"{eng}.pdf")
        else:
            ok = False
            err = [l for l in (r.stdout or "").splitlines()
                   if l.startswith("!")][:3]
            print(f"{eng:<10} FAIL {' / '.join(err) or r.returncode}")
    print(("PASS — every engine typeset the export" if ok else "FAIL")
          + f"\nbuild dir: {d}" if args.keep or not ok else "")
    if not (args.keep or not ok):
        shutil.rmtree(d, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
