#!/usr/bin/env python3
"""Run one scan while sampling the memory of the parent and every worker.

    python tools/bench/scan_profile.py --data sample_data --state /tmp/st [--full]
                                       [--workers N] [--json out.json]

Prints per-phase wall time, the scan's internal timings, and peak memory per
phase: parent RSS, workers' summed RSS / PSS / USS, the largest worker, and
mean CPU. PSS counts shared pages once, so parent RSS + workers PSS is the
honest "how much RAM did this scan need" number. Needs `psutil`.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MB = 2 ** 20


def _mem(p):
    try:
        mi = p.memory_full_info()
        return mi.rss, getattr(mi, "pss", mi.rss), mi.uss
    except Exception:
        return 0, 0, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data")
    ap.add_argument("--state", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json", default=None, help="also write results here")
    a = ap.parse_args()

    import psutil
    sys.path.insert(0, str(REPO))
    from kpviz.config import init_settings
    kw = {"state_dir": a.state}
    if a.workers:
        kw["workers"] = a.workers
    st = init_settings(a.data, **kw)
    from kpviz import db, scanner
    db.connect()

    me = psutil.Process()
    samples: list[dict] = []
    stop = threading.Event()

    def sampler():
        psutil.cpu_percent(interval=None)
        while not stop.is_set():
            snap = scanner.STATE.snapshot()
            step = next((s["key"] for s in snap["steps"]
                         if s["status"] == "running"), "-")
            prss, _ppss, _puss = _mem(me)
            kids = [_mem(k) for k in me.children(recursive=True)]
            samples.append({
                "step": step, "parent_rss": prss, "n_kids": len(kids),
                "kids_rss": sum(k[0] for k in kids),
                "kids_pss": sum(k[1] for k in kids),
                "kids_uss": sum(k[2] for k in kids),
                "max_kid_rss": max((k[0] for k in kids), default=0),
                "cpu": psutil.cpu_percent(interval=None)})
            time.sleep(0.25)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    v0 = db.scan_version()
    t0 = time.time()
    if not scanner.start_scan(a.full):
        print("a scan is already running")
        return 1
    scanner.wait_scan()
    wall = time.time() - t0
    stop.set()
    th.join()
    snap = scanner.STATE.snapshot()

    print(f"machine: {platform.platform()} · {psutil.cpu_count()} cpus · "
          f"{psutil.virtual_memory().total / 2**30:.1f} GB · Python "
          f"{platform.python_version()}")
    print(f"budget: {json.dumps(st.describe())}")
    print(f"scan wall {wall:.2f}s · error={bool(snap['error'])} · "
          f"catalog version {v0} -> {db.scan_version()}")
    if snap["error"]:
        print(snap["error"])
    steps = {}
    for s in snap["steps"]:
        if s["t_start"] and s["t_end"]:
            steps[s["key"]] = round(s["t_end"] - s["t_start"], 3)
            print(f"  {s['key']:<12} {steps[s['key']]:8.2f}s  {s['detail'][:70]}")
    if getattr(scanner.STATE, "counters", None):
        print("counters:", json.dumps(scanner.STATE.counters))
    print("internal timings:")
    for k, v in sorted(scanner.STATE.timing.items(), key=lambda kv: -kv[1]):
        if v >= 0.005:
            print(f"  {k:<30} {v:8.2f}s")
    by_step: dict[str, dict] = {}
    for smp in samples:
        d = by_step.setdefault(smp["step"], {"cpu": []})
        for k in ("parent_rss", "kids_rss", "kids_pss", "kids_uss",
                  "max_kid_rss", "n_kids"):
            d[k] = max(d.get(k, 0), smp[k])
        d["cpu"].append(smp["cpu"])
    print("peak memory per phase (MB): parent_rss kids_rss kids_pss "
          "kids_uss max_kid n_procs mean_cpu%")
    for step, d in by_step.items():
        cpu = d.pop("cpu")
        d["mean_cpu"] = round(sum(cpu) / max(1, len(cpu)), 1)
        print(f"  {step:<12} {d['parent_rss'] / MB:9.0f} {d['kids_rss'] / MB:8.0f} "
              f"{d['kids_pss'] / MB:8.0f} {d['kids_uss'] / MB:8.0f} "
              f"{d['max_kid_rss'] / MB:7.0f} {d['n_kids']:7d} {d['mean_cpu']:9.0f}")
    peak = max((s["parent_rss"] + s["kids_pss"] for s in samples), default=0)
    size = Path(st.db_path).stat().st_size
    print(f"peak parent RSS + workers PSS: {peak / MB:.0f} MB · "
          f"DuckDB file {size / MB:.1f} MB")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "wall_s": wall, "steps": steps, "timing": scanner.STATE.timing,
            "memory": by_step, "peak_mb": peak / MB, "db_mb": size / MB,
            "version_before": v0, "version_after": db.scan_version(),
            "budget": st.describe()}, indent=1, default=str))
    return 1 if snap["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
