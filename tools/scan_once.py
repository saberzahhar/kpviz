#!/usr/bin/env python3
"""Run one scan without the server, print the step breakdown, exit.

    python tools/scan_once.py [--data DIR] [--state DIR] [--workers N]
                              [--full] [--gold-scope all] [--token-scope all]

This is also the recommended way to derive a large corpus before serving it:
the scan gets the whole machine instead of sharing it with the UI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data")
    ap.add_argument("--state", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-scope", choices=["eval", "all"], default="eval")
    ap.add_argument("--token-scope", choices=["eval", "all"], default="eval")
    ap.add_argument("--hash", choices=["auto", "always"], default="auto")
    a = ap.parse_args()

    from kpviz.config import init_settings
    kw = {}
    if a.state:
        kw["state_dir"] = a.state
    if a.workers:
        kw["workers"] = a.workers
    st = init_settings(a.data, gold_scope=a.gold_scope,
                       token_scope=a.token_scope, hash_mode=a.hash, **kw)

    from kpviz import db, diag, scanner
    diag.print_banner(st)
    db.connect()
    t0 = time.time()
    scanner.STATE.start()
    scanner._scan_main(a.full)
    snap = scanner.STATE.snapshot()
    if snap["error"]:
        print("\nSCAN FAILED\n" + snap["error"])
        return 1
    print(f"\nscan complete in {time.time() - t0:.1f}s")
    for s in snap["steps"]:
        if s["t_start"] and s["t_end"]:
            print(f"  {s['key']:<12} {s['t_end'] - s['t_start']:8.2f}s  "
                  f"{s['detail'][:64]}")
    print("\ninternal timings:")
    for k, v in sorted(scanner.STATE.timing.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28} {v:8.2f}s")
    counts = {t: db.q1(f"SELECT count(*) FROM {t}")[0] for t in
              ("files", "documents", "gold", "gold_agg", "doc_tokens",
               "gold_tokpos", "keyphrases", "kp_stage", "runs", "batches",
               "preds", "matches", "run_metrics", "leakage")}
    print("\ntable rows:", json.dumps(counts))
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
