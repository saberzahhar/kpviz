"""KPViz — evaluate keyphrase models and put the figures in your paper.

    kpviz                          # serves ./data (else ./sample_data)
    kpviz --data /path/to/data --port 8050

The first launch scans the tree when the catalog is empty; after that,
"Scan for changes" on the Overview page re-derives only what changed.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None, home: Path | None = None) -> int:
    """Parse the command line, prepare the store, scan if empty, serve.

    `home` is where ./data and ./sample_data are looked for when --data is
    not given: the current directory for the `kpviz` command, the
    repository for `python app.py`."""
    ap = argparse.ArgumentParser(prog="kpviz", description=__doc__,
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
    ap.add_argument("--token-scope", choices=["eval", "all"], default="all",
                    help="document lengths in model tokens for every split "
                         "(default) or for evaluation splits only (faster)")
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
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network (tokenizer assets must "
                         "already be cached under the state directory)")
    ap.add_argument("--ui-cache-mb", type=int, default=192,
                    help="memory for cached per-document scores (default 192)")
    ap.add_argument("--no-autoscan", action="store_true",
                    help="do not scan automatically when the catalog is empty")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    here = Path(home) if home is not None else Path.cwd()
    data = args.data
    if data is None:
        if (here / "data").is_dir():
            data = here / "data"
        elif (here / "sample_data").is_dir():
            data = here / "sample_data"
            print("· no ./data directory — serving ./sample_data")
        else:
            print("error: no data directory found. Pass --data /path/to/data",
                  file=sys.stderr)
            return 2
    data = Path(data)
    if not data.is_dir():
        print(f"error: {data} is not a directory", file=sys.stderr)
        return 2

    from .config import init_settings
    kw = {}
    if args.state:
        kw["state_dir"] = args.state
    if args.workers:
        kw["workers"] = args.workers
    if args.db_threads:
        kw["db_threads"] = args.db_threads
    if args.io_workers:
        kw["io_workers"] = args.io_workers
    if args.offline:
        os.environ["KPVIZ_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    st = init_settings(data,
                       token_scope=args.token_scope,
                       gold_scope=args.gold_scope,
                       hash_mode=args.hash_mode, **kw)

    from . import db, diag, metrics, scanner
    db.connect()
    metrics.set_cache_budget(args.ui_cache_mb * 2**20)
    print(f"· data root   {st.data_root}")
    print(f"· state store {st.db_path}")
    diag.print_banner(st)

    if not args.no_autoscan and not db.q1("SELECT 1 FROM files LIMIT 1"):
        print("· empty catalog — starting the first scan in the background")
        scanner.start_scan(False)

    from .appfactory import build_app
    app = build_app()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"! listening on {args.host}: anyone who can reach this port can "
              "browse your documents (there is no authentication). KPViz is "
              "a single-user, single-process tool.")
    url = f"http://{args.host}:{args.port}"
    print(f"· serving     {url}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


def run() -> None:
    """Console-script entry point (`kpviz`)."""
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
