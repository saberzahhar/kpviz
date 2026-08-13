#!/usr/bin/env python3
"""Pool efficiency from the archived scan stats — the number to cite.

    python tools/pool_efficiency.py [.kpviz/scan_stats]

efficiency = Σ worker seconds / (wall seconds × workers). Below ~0.5 the scan
thread is doing work the pool is waiting on; near 1.0 the phase is genuinely
CPU-bound across every core.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else ".kpviz/scan_stats"
    files = sorted(glob.glob(str(Path(root) / "scan-*.json")))
    if not files:
        print(f"no scan stats under {root}")
        return 1
    s = json.load(open(files[-1]))
    workers = (s.get("budget") or {}).get("workers") or s.get("workers") or 1
    t = s.get("timing", {})
    print(f"{Path(files[-1]).name}   {s.get('duration_s')}s total   "
          f"{workers} workers")
    print(f"{'step':<14}{'wall':>9}{'worker':>10}{'wait':>9}{'ingest':>9}"
          f"{'efficiency':>12}")
    print("-" * 63)
    for step in s["steps"]:
        wall = step.get("duration_s")
        if not wall:
            continue
        key = step["key"]
        ws = t.get(f"{key}_worker_s", 0.0)
        wait = t.get(f"{key}_pool_wait_s", 0.0)
        ing = t.get(f"{key}_ingest_s", 0.0)
        eff = ws / (wall * workers) if wall and workers else 0.0
        print(f"{key:<14}{wall:9.2f}{ws:10.2f}{wait:9.2f}{ing:9.2f}"
              f"{eff:11.1%}")
    print("\nserial phases (no pool):")
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        if k.endswith(("_worker_s", "_pool_wait_s", "_ingest_s")):
            continue
        print(f"  {k:<28}{v:8.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
