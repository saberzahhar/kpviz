#!/usr/bin/env python3
"""Assert the precomputed SQL metrics equal the Python reference exactly.

`run_metrics` exists purely for speed; if it ever disagreed with the
document-by-document Python path the tool would be quietly wrong. This
compares every (run, annotation set, measure, k) combination in the catalog.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOL = 1e-12


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data")
    ap.add_argument("--state", default="/tmp/kpviz_v13")
    a = ap.parse_args()

    from kpviz.config import init_settings
    init_settings(a.data, state_dir=a.state)
    from kpviz import db
    from kpviz.metrics import KS, MEASURES, run_scores

    runs = db.q("SELECT DISTINCT dataset, model, arch, run_id FROM matches")
    anns = {r[0]: r[1] for r in db.q(
        "SELECT dataset, ann_key FROM matches GROUP BY 1,2")}
    ann_by_ds: dict[str, list[str]] = {}
    for ds, ann in db.q("SELECT dataset, ann_key FROM matches GROUP BY 1,2"):
        ann_by_ds.setdefault(ds, []).append(ann)

    by_ds: dict[str, list[tuple]] = {}
    for ds, model, arch, run_id in runs:
        by_ds.setdefault(ds, []).append((model, arch, run_id))

    checked = 0
    worst = 0.0
    failures = []
    for ds, keys in sorted(by_ds.items()):
        for ann in sorted(ann_by_ds.get(ds, [])):
            for measure in MEASURES:
                for k in KS:
                    fast = run_scores(ds, keys, ann, measure, k)
                    slow = run_scores(ds, keys, ann, measure, k,
                                      use_cache=False)
                    for key in keys:
                        f, s = fast.get(key, {}), slow.get(key, {})
                        checked += 1
                        if (f.get("mean") is None) != (s.get("mean") is None):
                            failures.append((ds, ann, measure, k, key,
                                             f.get("mean"), s.get("mean")))
                            continue
                        if f.get("mean") is None:
                            continue
                        d = abs(f["mean"] - s["mean"])
                        worst = max(worst, d)
                        if d > TOL or f["n"] != s["n"]:
                            failures.append((ds, ann, measure, k, key,
                                             f, s))
    print(f"compared {checked} (run, ann, measure, k) cells")
    print(f"worst absolute difference: {worst:.3e}")
    if failures:
        print(f"FAIL — {len(failures)} mismatches, first 5:")
        for x in failures[:5]:
            print("   ", x)
        return 1
    print("PASS — SQL fast path is identical to the Python reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
