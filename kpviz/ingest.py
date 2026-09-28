"""Asynchronous spill ingestion.

Worker results are NDJSON spill files. Feeding them to DuckDB one file at a
time, on the thread that also drives the pool, is the classic Amdahl trap:
the pool idles for the length of every ingest burst, and each statement pays
a full parse/bind/plan cycle for a few thousand rows.

A single writer thread batches spills per table and issues **one**
`read_json` over a *list* of files, so DuckDB plans one parallel scan instead
of N serial ones; the scan thread only enqueues.

`drain()` is a real barrier: it enqueues a flush request carrying an Event
that the writer sets only *after* every earlier spill has been written, so
signatures are never recorded for rows that have not landed. A write error
is sticky — it fails that drain and every later submit/drain/close, instead
of being cleared by the first reader.
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
MAX_PENDING_BYTES = 1 << 30  # backpressure: queued spill bytes, not files


class IngestError(RuntimeError):
    pass


class Ingestor:
    """Drains worker spills into DuckDB on one dedicated thread.

    `tables` maps a result-dict key (== target table) to
    (columns, mode) where mode is 'insert' | 'replace' | 'ignore'.
    """

    def __init__(self, tables: dict[str, tuple[dict, str]],
                 on_error=None):
        self.tables = tables
        self._q: queue.Queue = queue.Queue()
        self._pending: dict[str, list[str]] = {t: [] for t in tables}
        self._bytes: dict[str, int] = {t: 0 for t in tables}
        self._deadline: dict[str, float] = {}
        self._err: BaseException | None = None
        self._on_error = on_error
        self.ingest_s = 0.0
        self.rows = 0
        self.statements = 0
        self.files = 0
        self._queued_bytes = 0
        self._space = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="kpviz-ingest")
        self._thread.start()

    # -- producer side -----------------------------------------------------
    def submit(self, res: dict) -> None:
        """Enqueue one worker result. Blocks only under real backpressure
        (more than MAX_PENDING_BYTES of spill waiting)."""
        self.check()
        paths = {t: res[t] for t in self.tables if res.get(t)}
        if not paths:
            return
        size = 0
        for p in paths.values():
            try:
                size += os.path.getsize(p)
            except OSError:
                pass
        with self._space:
            while (self._queued_bytes > MAX_PENDING_BYTES and self._err is None
                   and self._thread.is_alive()):
                self._space.wait(timeout=0.5)
            self._queued_bytes += size
        self._q.put(("spill", paths, size))

    def drain(self, timeout: float | None = None) -> None:
        """Return only once everything submitted so far is in DuckDB."""
        self.check()
        done = threading.Event()
        self._q.put(("flush", done))
        t0 = time.monotonic()
        while not done.wait(0.05):
            if self._err is not None:
                break
            if not self._thread.is_alive():
                raise IngestError("ingest thread died before the flush completed")
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise IngestError("timed out waiting for ingest")
        self.check()

    def close(self) -> None:
        try:
            if self._err is None and self._thread.is_alive():
                self.drain()
        finally:
            self._q.put(None)
            self._thread.join(timeout=60)
        self.check()

    def check(self) -> None:
        if self._err is not None:
            raise IngestError(f"spill ingest failed: {self._err}") from self._err

    # -- writer thread -----------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                item = ("tick",)
            if item is None:
                break
            try:
                if self._err is None:
                    if item[0] == "spill":
                        _kind, paths, size = item
                        for table, path in paths.items():
                            self._pending[table].append(path)
                            try:
                                self._bytes[table] += os.path.getsize(path)
                            except OSError:
                                pass
                            self._deadline.setdefault(
                                table, time.monotonic() + FLUSH_SECS)
                        self._release(size)
                        self._flush_due()
                    elif item[0] == "flush":
                        self._flush_all()
                    else:
                        self._flush_due()
                elif item[0] == "spill":
                    self._release(item[2])
            except BaseException as exc:             # surface to the scanner
                self._err = exc
                if self._on_error:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass
                with self._space:
                    self._space.notify_all()
            finally:
                if item[0] == "flush":
                    item[1].set()

    def _release(self, size: int) -> None:
        with self._space:
            self._queued_bytes = max(0, self._queued_bytes - size)
            self._space.notify_all()

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
            self.files += len(paths)
        finally:
            self.ingest_s += time.perf_counter() - t0
            self.statements += 1

    # -- telemetry ---------------------------------------------------------
    def stats(self) -> dict:
        return {"ingest_s": round(self.ingest_s, 3),
                "ingest_rows": self.rows,
                "ingest_files": self.files,
                "ingest_statements": self.statements}
