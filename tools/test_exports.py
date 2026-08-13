#!/usr/bin/env python3
"""Headless verification of the export engine + RQ2 leakage delta.

    python tools/test_exports.py [--data sample_data] [--state DIR]

The state directory must already hold a scanned catalog (tools/scan_once.py).
"""
import argparse
import sys

sys.path.insert(0, ".")
from kpviz.config import init_settings

_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default="sample_data")
_ap.add_argument("--state", default="/tmp/kpviz_state_test")
_args = _ap.parse_args()
init_settings(_args.data, state_dir=_args.state)

from kpviz import db  # noqa: E402
from kpviz.export import (export_bundle, fig_pdf, fig_pgf, fig_png,  # noqa: E402
                          latex_figure, latex_table, tex_engine)
from kpviz.metrics import run_scores  # noqa: E402

spec = {
    "kind": "scatter", "size": "1col", "xscale": "log",
    "xlabel": "cost (USD) — per document", "ylabel": "F1@O",
    "series": [
        {"name": "bart", "x": [2.6e-5], "y": [0.53], "color": "#2a78d6",
         "mpl_marker": "o", "size": 11, "alpha": 1.0,
         "text": ["bart-base-kp20k (num\\_beams=4)"]},
        {"name": "gpt", "x": [4.7e-4], "y": [0.64], "color": "#eb6834",
         "mpl_marker": "s", "size": 13, "alpha": 0.56, "text": ["gpt-4o"]},
    ],
    "hlines": [{"y": 0.586, "label": "Llama-3.3 — no usd", "color": "#898781"}],
    "frontier": {"x": [2.6e-5, 4.7e-4], "y": [0.53, 0.64]},
    "caption": "Cost–performance test export.",
    "name": "pareto-test",
}

print("tex engine:", tex_engine())
png = fig_png(spec)
print("png bytes:", len(png), png[:4] == b"\x89PNG")
pdf, method = fig_pdf(spec)
print("pdf bytes:", len(pdf), "method:", method, pdf[:5] == b"%PDF-")
pgf = fig_pgf(spec)
print("pgf chars:", len(pgf) if pgf else None,
      (pgf or "")[:40].replace("\n", " "))
z = export_bundle(spec, "pareto test")
print("zip bytes:", len(z), z[:2] == b"PK")
open("/tmp/export_test.pdf", "wb").write(pdf)
open("/tmp/export_test.png", "wb").write(png)

tab = latex_table(["Run", "F1@O", "USD"],
                  [["bart-base-kp20k (num_beams=4)", 0.5285, 2.6e-5],
                   ["gpt-4o", 0.6432, 4.7e-4]],
                  "Test table caption with % and _ escapes.", "test")
print("--- latex table ---")
print(tab)
print("--- latex figure ---")
print(latex_figure("figures/pareto-test.pgf", "A caption.", "pareto-test",
                   pgf=True))

# RQ2 leakage sanity: pick a run that actually has matches on the dataset that
# receives leaked documents, so the tool works on any tree, not just the sample.
row = db.q(
    """SELECT l.dataset_a, l.dataset_b, m.model, m.arch, m.run_id
         FROM leakage l
         JOIN matches m ON m.dataset = l.dataset_b
        WHERE l.label='near-duplicate' AND l.score>=0.8
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY count(*) DESC LIMIT 1""")
if not row:
    print("RQ2 leakage: no (leakage, run) pair in this catalog — skipped")
    sys.exit(0)
ds_a, ds_b, *key = row[0]
key = tuple(key)
leak = {r[0] for r in db.q(
    """SELECT doc_id_b FROM leakage
        WHERE dataset_a=? AND dataset_b=?
          AND label='near-duplicate' AND score>=0.8""", ds_a, ds_b)}
sc = run_scores(ds_b, [key], "author", "f1", "O")[key]
excl = run_scores(ds_b, [key], "author", "f1", "O",
                 exclude_doc_ids=leak)[key]
only = run_scores(ds_b, [key], "author", "f1", "O", doc_ids=leak)[key]


def _f(d):
    return "n/a" if d["mean"] is None else f"{d['mean']:.4f}"


delta = ("n/a" if excl["mean"] is None or sc["mean"] is None
         else f"{excl['mean'] - sc['mean']:+.4f}")
print(f"RQ2 {ds_b} ← {ds_a} leak, {key[0]}/{key[2]}: "
      f"all={_f(sc)} (n={sc['n']}) excl={_f(excl)} (n={excl['n']}) "
      f"leaked-only={_f(only)} (n={only['n']}) delta={delta}")
