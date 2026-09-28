#!/usr/bin/env python3
"""Content fingerprint of every derived table (order-independent).

    python tools/bench/fingerprint.py --state STATE [--json out.json]

Two stores built from the same data tree must print the same fingerprints,
whatever the worker count, the scheduling, or whether they were built
incrementally or from scratch. Floats are rounded to 12 significant digits
(parallel aggregation may differ in the last bits); scan bookkeeping
(file ids, byte offsets, mtimes, timestamps of the scan itself) is excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TABLES = {
    "documents": ["file_id"],
    "gold": [], "gold_agg": [], "doc_tokens": [], "gold_tokpos": [],
    "preds": ["file_id"], "matches": [], "runs": [], "run_metrics": [],
    "keyphrases": [], "leakage": [], "batches": [],
}


def _norm(v):
    if isinstance(v, float):
        return float(f"{v:.12g}")
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def fingerprints(db_path: str) -> dict:
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    out = {}
    for table, skip in TABLES.items():
        cols = [r[0] for r in con.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name='{table}' ORDER BY ordinal_position").fetchall()]
        keep = [c for c in cols if c not in skip]
        if not keep:
            continue
        rows = con.execute(f"SELECT {', '.join(keep)} FROM {table}").fetchall()
        lines = sorted(json.dumps([_norm(v) for v in r], default=str) for r in rows)
        h = hashlib.blake2b(digest_size=10)
        for ln in lines:
            h.update(ln.encode())
            h.update(b"\n")
        out[table] = f"{len(rows)}:{h.hexdigest()}"
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fp = fingerprints(str(Path(a.state) / "kpviz.duckdb"))
    print(json.dumps(fp, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(fp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
