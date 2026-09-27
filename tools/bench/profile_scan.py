#!/usr/bin/env python3
"""Profile a real scan, function by function, workers included.

    python tools/bench/profile_scan.py --data TREE --state STATE [--top 25]

Runs one cold scan with KPVIZ_PROFILE_DIR set, so every worker task
(documents chunk, predictions task, POS chunk) writes its own cProfile
stats; the parent (discover, ingest, finalize) is profiled here. Prints,
per task kind, the functions by own time summed over all tasks — the
place to look before optimising anything.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--keep", default=None,
                    help="keep the per-task .prof files in this folder")
    a = ap.parse_args()
    prof_dir = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="kpviz-prof-"))
    if a.keep:
        shutil.rmtree(prof_dir, ignore_errors=True)
        prof_dir.mkdir(parents=True)
    os.environ["KPVIZ_PROFILE_DIR"] = str(prof_dir)
    if Path(a.state).exists():
        shutil.rmtree(a.state)
    from kpviz.config import init_settings
    kw = {"state_dir": a.state}
    if a.workers:
        kw["workers"] = a.workers
    init_settings(a.data, **kw)
    from kpviz import db, scanner
    db.connect()
    pr = cProfile.Profile()
    pr.enable()
    scanner.STATE.start()
    scanner._scan_main(False)
    pr.disable()
    snap = scanner.STATE.snapshot()
    if snap.get("error"):
        print(snap["error"])
        return 1
    for s in snap["steps"]:
        if s["t_start"] and s["t_end"]:
            print(f"  {s['key']:<12} {s['t_end'] - s['t_start']:7.2f}s")

    def show(title, stats):
        out = io.StringIO()
        stats.stream = out
        stats.sort_stats("tottime").print_stats(a.top)
        body = out.getvalue()
        start = body.find("ncalls")
        print(f"\n===== {title} =====\n" + body[start:])

    groups = defaultdict(list)
    for f in prof_dir.glob("*.prof"):
        groups[f.name.split("-")[0]].append(str(f))
    for kind, files in sorted(groups.items()):
        st = pstats.Stats(files[0])
        for f in files[1:]:
            st.add(f)
        show(f"{kind} — {len(files)} task(s), own time summed over workers", st)
    show("parent (scan thread + ingest)", pstats.Stats(pr))
    if not a.keep:
        shutil.rmtree(prof_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
