"""Asynchronous spill ingestion.

Worker results are NDJSON spill files. Feeding them to DuckDB one file at a
time, on the thread that also drives the pool, is the classic Amdahl trap:
the pool idles for the length of every ingest burst, and each statement pays
a full parse/bind/plan cycle for a few thousand rows.

This module fixes both halves. A single writer thread batches spills per
table and issues **one** `read_json` over a *list* of files, so DuckDB plans
one parallel scan instead of N serial ones; the scan thread only enqueues.
Bounded queue + a cap on unflushed files keep disk and RAM in check.
"""
from __future__ import annotations

import os
import queue
import threading
import time

from . import db

FLUSH_FILES = 48            # files per table per statement
FLUSH_BYTES = 192 << 20     # or this much accumulated spill
FLUSH_SECS = 2.0            # or this old
MAX_PENDING_FILES = 4096    # backpressure: stop workers running away from us


class Ingestor:
    """Drains worker spills into DuckDB on one dedicated thread.

    `tables` maps a result-dict key (== target table) to
    (columns, mode) where mode is 'insert' | 'replace' | 'ignore'.
    """

    def __init__(self, tables: dict[str, tuple[dict, str]],
                 on_error=None):
        self.tables = tables
        self._q: queue.Queue = queue.Queue(maxsize=MAX_PENDING_FILES)
        self._pending: dict[str, list[str]] = {t: [] for t in tables}
        self._bytes: dict[str, int] = {t: 0 for t in tables}
        self._deadline: dict[str, float] = {}
        self._stop = threading.Event()
        self._err = None
        self._on_error = on_error
        self.ingest_s = 0.0
        self.rows = 0
        self.statements = 0
        self.n_pending = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="kpviz-ingest")
        self._thread.start()

    # -- producer side -----------------------------------------------------
    def submit(self, res: dict) -> None:
        """Enqueue one worker result. Blocks only under real backpressure."""
        paths = {t: res[t] for t in self.tables if res.get(t)}
        if not paths:
            return
        self._q.put(paths)
        with self._lock:
            self.n_pending += len(paths)

    def drain(self) -> None:
        """Flush everything queued and wait for it to land."""
        self._q.put(("__flush__",))
        while True:
            with self._lock:
                if self.n_pending == 0 and self._q.empty():
                    break
            if self._err:
                break
            time.sleep(0.02)
        self.check()

    def close(self) -> None:
        self.drain()
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=30)
        self.check()

    def check(self) -> None:
        if self._err:
            err, self._err = self._err, None
            raise err

    # -- writer thread -----------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                self._flush_due()
                continue
            if item is None:
                break
            try:
                if isinstance(item, tuple):          # explicit flush marker
                    self._flush_all()
                    continue
                for table, path in item.items():
                    self._pending[table].append(path)
                    try:
                        self._bytes[table] += os.path.getsize(path)
                    except OSError:
                        pass
                    self._deadline.setdefault(table, time.monotonic() + FLUSH_SECS)
                    with self._lock:
                        self.n_pending -= 1
                self._flush_due()
            except Exception as exc:                 # surface to the scanner
                self._err = exc
                if self._on_error:
                    self._on_error(exc)
                with self._lock:
                    self.n_pending = 0
        self._flush_all()

    def _flush_due(self) -> None:
        now = time.monotonic()
        for table, paths in list(self._pending.items()):
            if not paths:
                continue
            if (len(paths) >= FLUSH_FILES or self._bytes[table] >= FLUSH_BYTES
                    or now >= self._deadline.get(table, 0)):
                self._flush(table)

    def _flush_all(self) -> None:
        for table in list(self._pending):
            if self._pending[table]:
                self._flush(table)

    def _flush(self, table: str) -> None:
        paths = self._pending[table]
        if not paths:
            return
        self._pending[table] = []
        self._bytes[table] = 0
        self._deadline.pop(table, None)
        cols, mode = self.tables[table]
        t0 = time.perf_counter()
        try:
            self.rows += db.ingest_ndjson(table, paths, cols, mode=mode)
        finally:
            self.ingest_s += time.perf_counter() - t0
            self.statements += 1

    # -- telemetry ---------------------------------------------------------
    def stats(self) -> dict:
        return {"ingest_s": round(self.ingest_s, 3),
                "ingest_rows": self.rows,
                "ingest_statements": self.statements}
