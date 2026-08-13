#!/usr/bin/env python3
"""KPViz — keyphrase evaluation cockpit.

    python app.py                 # serves ./data (falls back to ./sample_data)
    python app.py --data /path/to/data --port 8050

First launch triggers a scan automatically when the catalog is empty; after
that, use “Scan for changes” on the Overview page — only new/modified files
are re-derived.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help="data root (default: ./data, else ./sample_data)")
    ap.add_argument("--state", default=None,
                    help="state directory for the DuckDB store "
                         "(default: .kpviz next to the data root)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--workers", type=int, default=None,
                    help="scan worker processes (default: all cores)")
    ap.add_argument("--pos-scope", choices=["eval", "all"], default="eval",
                    help="(POS now tags unique phrases once; kept for compat)")
    ap.add_argument("--token-scope", choices=["eval", "all"], default="eval",
                    help="tokenizer-count documents for eval splits only (default) or all")
    ap.add_argument("--gold-scope", choices=["eval", "all"], default="eval",
                    help="store per-keyphrase gold rows for eval splits only "
                         "(default; quality-flagged documents are always kept, "
                         "and distribution charts always cover every split) or all")
    ap.add_argument("--db-threads", type=int, default=None,
                    help="DuckDB threads during a scan (default: cores left "
                         "over from the worker pool)")
    ap.add_argument("--io-workers", type=int, default=None,
                    help="hashing/IO concurrency (default: 2x workers, max 64)")
    ap.add_argument("--hash", choices=["auto", "always"], default="auto",
                    dest="hash_mode",
                    help="auto: content-hash only files whose size/mtime moved "
                         "(default); always: hash everything every scan")
    ap.add_argument("--no-autoscan", action="store_true",
                    help="do not scan automatically when the catalog is empty")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    data = args.data
    if data is None:
        if (here / "data").is_dir():
            data = here / "data"
        elif (here / "sample_data").is_dir():
            data = here / "sample_data"
            print("· no ./data directory — serving the bundled sample_data")
        else:
            print("error: no data directory found. Pass --data /path/to/data",
                  file=sys.stderr)
            return 2
    data = Path(data)
    if not data.is_dir():
        print(f"error: {data} is not a directory", file=sys.stderr)
        return 2

    from kpviz.config import init_settings
    kw = {}
    if args.state:
        kw["state_dir"] = args.state
    if args.workers:
        kw["workers"] = args.workers
    if args.db_threads:
        kw["db_threads"] = args.db_threads
    if args.io_workers:
        kw["io_workers"] = args.io_workers
    st = init_settings(data, pos_scope=args.pos_scope,
                       token_scope=args.token_scope,
                       gold_scope=args.gold_scope,
                       hash_mode=args.hash_mode, **kw)

    from kpviz import db, diag, scanner
    db.connect()
    print(f"· data root   {st.data_root}")
    print(f"· state store {st.db_path}")
    diag.print_banner(st)

    if not args.no_autoscan and not db.q1("SELECT 1 FROM files LIMIT 1"):
        print("· empty catalog — starting the first scan in the background")
        scanner.start_scan(False)

    from kpviz.appfactory import build_app
    app = build_app()

    url = f"http://{args.host}:{args.port}"
    print(f"· serving     {url}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
