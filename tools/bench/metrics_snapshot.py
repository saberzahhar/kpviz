#!/usr/bin/env python3
"""Snapshot every score the workbenches can ask for, for parity checks.

    python tools/bench/metrics_snapshot.py --data D --state S --out snap.json
    python tools/bench/metrics_snapshot.py --compare old.json new.json

The grid covers every dataset × annotation set × measure × k × gold filter
(none, P, R+M+U, present-with-position, present-within-512-tokens) × run,
with per-document scores. It only uses the public `metrics.run_scores` API, so
the same script snapshots old and new code and `--compare` reports every
difference above 1e-9 — the gate every performance change must pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FILTERS = {
    "none": {},
    "P": {"prmu": ["P"]},
    "RMU": {"prmu": ["R", "M", "U"]},
    "P+pos": {"prmu": ["P"], "require_position": True},
    "pos+512": {"require_position": True, "tok_limit": "512"},
}


def snapshot(data: str, state: str, repo: str) -> dict:
    sys.path.insert(0, repo)
    from kpviz.config import init_settings
    init_settings(data, state_dir=state)
    from kpviz import db, metrics
    out: dict[str, list] = {}
    for (ds,) in db.q("SELECT DISTINCT dataset FROM matches ORDER BY 1"):
        keys = [tuple(r) for r in db.q(
            "SELECT DISTINCT model, arch, run_id FROM matches WHERE dataset=? "
            "ORDER BY 1,2,3", ds)]
        toks = [r[0] for r in db.q(
            "SELECT DISTINCT tokenizer FROM gold_tokpos WHERE dataset=? ORDER BY 1", ds)]
        for (ann,) in db.q("SELECT DISTINCT ann_key FROM matches WHERE dataset=? "
                           "ORDER BY 1", ds):
            for fname, f in FILTERS.items():
                kw = dict(f)
                if "tok_limit" in kw:
                    if not toks:
                        continue
                    kw["tok_limit"] = (toks[0], 512)
                for measure in ("f1", "p", "r"):
                    for k in ("5", "10", "O", "M"):
                        res = metrics.run_scores(ds, keys, ann, measure, k,
                                                 per_doc=True, use_cache=False, **kw)
                        for key in keys:
                            r = res.get(key) or {}
                            pd = r.get("per_doc") or {}
                            if hasattr(pd, "as_dict"):
                                pd = pd.as_dict()
                            out["|".join([ds, ann, fname, measure, k, *key])] = [
                                r.get("mean"), r.get("n"),
                                {d: round(float(v), 12) for d, v in dict(pd).items()}]
    return out


def compare(a: dict, b: dict, tol: float = 1e-9) -> int:
    bad = 0
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            print(f"missing on one side: {key}")
            bad += 1
            continue
        (ma, na, pa), (mb, nb, pb) = va, vb
        if na != nb or (ma is None) != (mb is None) or \
                (ma is not None and abs(ma - mb) > tol):
            print(f"{key}: mean {ma} (n={na}) vs {mb} (n={nb})")
            bad += 1
            continue
        if set(pa) != set(pb) or any(abs(pa[d] - pb[d]) > tol for d in pa):
            print(f"{key}: per-document scores differ")
            bad += 1
    print(f"{len(a)} vs {len(b)} cells compared, {bad} differ")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--state")
    ap.add_argument("--out")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        return compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
    snap = snapshot(a.data, a.state, a.repo)
    Path(a.out).write_text(json.dumps(snap))
    print(f"{len(snap)} cells written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
