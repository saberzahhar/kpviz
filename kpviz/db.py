"""DuckDB analytical store (schema v4).

Holds *derived* data and indices only — the corpus and inference files are
addressed by (file, byte offset, length), never copied.

Two design notes that matter for throughput:

* **Reads do not take the write lock.** DuckDB is MVCC internally and a
  cursor per thread is the documented concurrency pattern, so UI queries no
  longer queue behind a bulk COPY. Only writers serialise.
* **UI callbacks never write here.** Result caches live in process memory
  (metrics.py) and colour assignments in a small JSON file next to the store,
  so a click can never queue behind the scan's writer lock.
* **The high-volume derived tables carry no primary key.** They are
  purge-then-insert, and a repeated document id inside the new input is
  resolved right after ingest (first line wins, the count becomes an issue
  tag) — `scanner._dedup_*`. Dropping the
  ART index turns `INSERT OR REPLACE` into a plain `INSERT` (~6× cheaper)
  and stops `DELETE` from paying index maintenance (measured: the dominant
  cost of re-deriving a run). `keyphrases` keeps its key — that key *is* the
  global dedup mechanism — but is fed through a staging table so the index
  is paid once per scan instead of once per chunk.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import duckdb

from .config import settings

MARKED_MISSING: list[str] = []


def mark_missing_optional_modules() -> list[str]:
    """DuckDB's Python client tries `import pandas` for every bound
    parameter (2–8 times per query). When pandas is not installed each
    attempt walks all of sys.path again — failed imports are not cached —
    which cost 0.21 s of a 0.24 s workbench callback. Recording the
    module as absent (sys.modules[name] = None, the import system's own
    "known missing" marker) makes that check free. Only modules that
    genuinely cannot be found are marked; an installed pandas is untouched.

    A process-wide side effect, so it is opt-in: the server and the scan
    call it at start-up (and the start-up banner says so) — importing
    kpviz.db from a notebook or a tool changes nothing. Installing pandas
    later in such a process needs a restart."""
    import importlib.util
    import sys
    for name in ("pandas", "polars", "pyarrow"):
        if name in sys.modules:
            continue
        try:
            missing = importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            missing = True
        if missing:
            sys.modules[name] = None
            if name not in MARKED_MISSING:
                MARKED_MISSING.append(name)
    return MARKED_MISSING

SCHEMA_VERSION = 6

# Rebuilt on a schema change. `keyphrases` is deliberately absent: its shape
# is tracked separately (KP_SCHEMA_VERSION) so a schema bump elsewhere never
# throws away the phrase cache — the thing that makes later scans cheap.
_DERIVED_TABLES = [
    "documents", "doc_tokens", "gold", "gold_agg", "gold_tokpos",
    "runs", "batches", "preds", "matches", "leakage", "run_metrics",
    "kp_stage", "agg_cache", "phrase_pos", "color_assign",
]
# v2: keyed per normalised phrase with the POS tag only (the token/stem
# columns were never read); '' marks "tagged, no pattern"
# v3: keyed per (language, phrase)
KP_SCHEMA_VERSION = 3

_DDL = """
CREATE SEQUENCE IF NOT EXISTS seq_file_id START 1;

CREATE TABLE IF NOT EXISTS files(
    relpath    VARCHAR PRIMARY KEY,
    file_id    BIGINT NOT NULL,
    kind       VARCHAR NOT NULL,
    dataset    VARCHAR, model VARCHAR, arch VARCHAR, run_id VARCHAR,
    batch_idx  INTEGER,
    size       BIGINT, mtime DOUBLE, hash VARCHAR,
    sig        VARCHAR,
    scanned_at TIMESTAMP DEFAULT now()
);

-- gold keyphrases for POS tagging: one row per (language, normalised
-- phrase) — French "chat" and English "chat" are two phrases, tagged by two
-- models, whatever order they were ingested in; the surface form is picked
-- deterministically at merge time
CREATE TABLE IF NOT EXISTS keyphrases(
    lang     VARCHAR NOT NULL,
    kp       VARCHAR NOT NULL,
    raw      VARCHAR,
    pos      VARCHAR,
    PRIMARY KEY (lang, kp)
);
-- worker spills land here first, then one anti-join merges them
CREATE TABLE IF NOT EXISTS kp_stage(
    kp VARCHAR, raw VARCHAR, lang VARCHAR
);

CREATE TABLE IF NOT EXISTS documents(
    dataset    VARCHAR NOT NULL,
    doc_id     VARCHAR NOT NULL,
    split      VARCHAR,
    file_id    BIGINT,
    byte_off   BIGINT, byte_len BIGINT,
    n_sections INTEGER,
    n_chars    BIGINT, n_words  BIGINT,
    langs      VARCHAR[],
    detected_langs VARCHAR[],
    sections   VARCHAR,
    metadata   VARCHAR,
    ann_counts VARCHAR,
    flags      VARCHAR[]
);

-- src_off (gold, doc_tokens, gold_tokpos) is the byte offset of the source
-- document line: when a collection repeats a document id, the first line is
-- kept and the rows of the others are removed by it after ingest
CREATE TABLE IF NOT EXISTS doc_tokens(
    dataset VARCHAR NOT NULL, doc_id VARCHAR NOT NULL, src_off BIGINT,
    tokenizer VARCHAR NOT NULL,
    n_tokens INTEGER, approx BOOLEAN
);

-- gold *instances*: eval-scope splits plus every quality-flagged document;
-- one row per distinct stemmed keyphrase of an annotation set (duplicates
-- and token-less keyphrases are counted per collection, not stored)
CREATE TABLE IF NOT EXISTS gold(
    dataset VARCHAR NOT NULL, doc_id VARCHAR NOT NULL,
    ann_key VARCHAR NOT NULL, kp_idx INTEGER NOT NULL,
    src_off BIGINT,
    display VARCHAR,         -- normal form (joins keyphrases.kp)
    surface VARCHAR,         -- as annotated (first variant, spacing collapsed)
    stems   VARCHAR[],
    lang    VARCHAR,
    n_words INTEGER,
    prmu    VARCHAR,
    -- where the earliest contiguous occurrence ENDS: character offset in the
    -- original document text, and stemmed-token index (-1 when not present)
    end_char INTEGER,
    end_word INTEGER
);

-- distribution aggregates for EVERY split (tiny, chart-ready)
CREATE TABLE IF NOT EXISTS gold_agg(
    dataset VARCHAR NOT NULL, split VARCHAR, ann_key VARCHAR NOT NULL,
    prmu VARCHAR, n_words_b INTEGER,
    n BIGINT, words_sum BIGINT
);

CREATE TABLE IF NOT EXISTS gold_tokpos(
    dataset VARCHAR NOT NULL, doc_id VARCHAR NOT NULL, src_off BIGINT,
    ann_key VARCHAR NOT NULL, kp_idx INTEGER NOT NULL,
    tokenizer VARCHAR NOT NULL,
    tok_end INTEGER, approx BOOLEAN
);

CREATE TABLE IF NOT EXISTS runs(
    dataset VARCHAR NOT NULL, model VARCHAR NOT NULL,
    arch VARCHAR NOT NULL, run_id VARCHAR NOT NULL,
    params    VARCHAR,
    resolved  VARCHAR,
    violations VARCHAR,
    tags      VARCHAR,
    arch_known BOOLEAN,
    n_batches INTEGER, n_docs INTEGER, n_dup_docs INTEGER,
    expected_docs INTEGER, coverage DOUBLE,
    var_totals VARCHAR,
    costs      VARCHAR,
    t_start TIMESTAMP, t_end TIMESTAMP, wall_s DOUBLE,
    PRIMARY KEY (dataset, model, arch, run_id)
);

CREATE TABLE IF NOT EXISTS batches(
    dataset VARCHAR NOT NULL, model VARCHAR NOT NULL,
    arch VARCHAR NOT NULL, run_id VARCHAR NOT NULL, batch_idx INTEGER NOT NULL,
    costs VARCHAR,
    t_start TIMESTAMP, t_end TIMESTAMP, wall_s DOUBLE
);

CREATE TABLE IF NOT EXISTS preds(
    dataset VARCHAR NOT NULL, model VARCHAR NOT NULL,
    arch VARCHAR NOT NULL, run_id VARCHAR NOT NULL,
    doc_id VARCHAR NOT NULL,
    batch_idx INTEGER,
    file_id BIGINT, byte_off BIGINT, byte_len BIGINT,
    n_preds INTEGER, n_uniq INTEGER,
    costs VARCHAR
);

CREATE TABLE IF NOT EXISTS matches(
    dataset VARCHAR NOT NULL, model VARCHAR NOT NULL,
    arch VARCHAR NOT NULL, run_id VARCHAR NOT NULL,
    doc_id VARCHAR NOT NULL, ann_key VARCHAR NOT NULL,
    batch_idx INTEGER, byte_off BIGINT,   -- source line (duplicate policy)
    n_uniq INTEGER, n_gold INTEGER,
    pred_ranks INTEGER[],
    gold_idxs  INTEGER[]
);

-- exact unfiltered scores, precomputed in SQL at finalize (see metrics.py)
CREATE TABLE IF NOT EXISTS run_metrics(
    dataset VARCHAR, model VARCHAR, arch VARCHAR, run_id VARCHAR,
    ann_key VARCHAR, k VARCHAR,
    f1_mean DOUBLE, p_mean DOUBLE, r_mean DOUBLE, n BIGINT
);

CREATE TABLE IF NOT EXISTS leakage(
    dataset_a VARCHAR, doc_id_a VARCHAR,
    dataset_b VARCHAR, doc_id_b VARCHAR,
    score DOUBLE, label VARCHAR
);

CREATE TABLE IF NOT EXISTS kv(k VARCHAR PRIMARY KEY, v VARCHAR);
"""

_wlock = threading.RLock()          # writers only
_init_lock = threading.Lock()
_con: duckdb.DuckDBPyConnection | None = None
_local = threading.local()


def connect() -> duckdb.DuckDBPyConnection:
    """The process-wide connection. Fast path takes no lock at all."""
    global _con
    if _con is not None:
        return _con
    with _init_lock:
        if _con is not None:
            return _con
        st = settings()
        con = duckdb.connect(str(st.db_path))
        tmp = str(st.tmp_dir / "duckdb").replace("'", "''")
        con.execute(f"SET temp_directory='{tmp}'")
        for stmt in ("SET checkpoint_threshold='1GB'",   # no checkpoint thrash
                     "SET allocator_flush_threshold='64MB'",
                     "SET allocator_background_threads=true"):
            try:           # return freed memory to the OS (DuckDB >= 1.1)
                con.execute(stmt)
            except Exception:
                pass
        _apply_budget(con, scanning=False)
        con.execute("CREATE TABLE IF NOT EXISTS kv(k VARCHAR PRIMARY KEY, v VARCHAR)")
        _migrate(con)
        # strip line comments before splitting: a ';' inside a comment would
        # otherwise cut a statement in half
        ddl = "\n".join(l for l in _DDL.splitlines()
                        if not l.lstrip().startswith("--"))
        for stmt in ddl.split(";"):
            if stmt.strip():
                con.execute(stmt)
        con.execute("INSERT OR REPLACE INTO kv VALUES ('schema_version', ?)",
                    [json.dumps(SCHEMA_VERSION)])
        con.execute("INSERT OR REPLACE INTO kv VALUES ('kp_schema_version', ?)",
                    [json.dumps(KP_SCHEMA_VERSION)])
        _con = con
        return _con


def _migrate(con) -> None:
    """A schema change drops the derived tables (file hashes survive, so the
    next scan re-derives without re-reading anything it doesn't have to) and
    clears derivation signatures. The phrase cache survives unless its own
    shape changed."""
    def _kv(key):
        row = con.execute("SELECT v FROM kv WHERE k=?", [key]).fetchall()
        return json.loads(row[0][0]) if row else None

    if _kv("kp_schema_version") not in (None, KP_SCHEMA_VERSION):
        con.execute("DROP TABLE IF EXISTS keyphrases")
    if _kv("schema_version") == SCHEMA_VERSION:
        return
    for t in _DERIVED_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("DELETE FROM kv WHERE k LIKE 'run_sig:%' OR k LIKE 'scores_sig:%'")
    try:
        con.execute("UPDATE files SET sig = NULL")
    except Exception:
        pass


def _apply_budget(con, scanning: bool) -> None:
    """Threads and memory for the current phase.

    During a scan the process pool owns the cores and DuckDB gets the RAM the
    workers leave; while serving, DuckDB gets every core (UI queries are
    short and parallel) but a small memory budget, so the server's footprint
    after a scan shrinks back instead of keeping a multi-GB buffer pool."""
    st = settings()
    from .hostinfo import usable_cpus
    threads = st.db_threads if scanning else max(2, usable_cpus())
    mem = st.duckdb_memory_bytes if scanning else st.duckdb_serve_bytes
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{max(64, mem // 2**20)}MB'")


def set_scan_mode(on: bool) -> None:
    try:
        with _wlock:
            _apply_budget(connect(), scanning=on)
    except Exception:
        pass


def cursor() -> duckdb.DuckDBPyConnection:
    """A per-thread cursor. Reads through this are lock-free."""
    cur = getattr(_local, "cur", None)
    if cur is None:
        cur = connect().cursor()
        _local.cur = cur
    return cur


def _retryable(exc: Exception) -> bool:
    """Only a cursor/connection invalidated underneath us is worth a retry;
    a parser, binder or catalog error would fail identically a second time
    and its second traceback would hide the first."""
    if isinstance(exc, (duckdb.ConnectionException,)):
        return True
    msg = str(exc).lower()
    return "closed" in msg or "invalidated" in msg


def q(sql: str, *params):
    try:
        return cursor().execute(sql, list(params) if params else None).fetchall()
    except duckdb.Error as e:
        if not _retryable(e):
            raise
        _local.cur = None
        return cursor().execute(sql, list(params) if params else None).fetchall()


def qnp(sql: str, *params) -> dict:
    """Columnar fetch ({column: numpy array}) — for anything large. Building
    Python row tuples of list columns was 72 % of the old scoring path."""
    try:
        return cursor().execute(sql, list(params) if params else None).fetchnumpy()
    except duckdb.Error as e:
        if not _retryable(e):
            raise
        _local.cur = None
        return cursor().execute(sql, list(params) if params else None).fetchnumpy()


def q1(sql: str, *params):
    rows = q(sql, *params)
    return rows[0] if rows else None


def qdict(sql: str, *params) -> list[dict]:
    try:
        res = cursor().execute(sql, list(params) if params else None)
    except duckdb.Error as e:
        if not _retryable(e):
            raise
        _local.cur = None
        res = cursor().execute(sql, list(params) if params else None)
    cols = [d[0] for d in res.description]
    return [dict(zip(cols, r)) for r in res.fetchall()]


def execute(sql: str, *params):
    with _wlock:
        return connect().execute(sql, list(params) if params else None)


def executemany(sql: str, rows):
    with _wlock:
        return connect().executemany(sql, rows)


# ---------------------------------------------------------------------------
# Bulk ingestion. `paths` may be a single path or a list — DuckDB reads a
# file list in one statement, which is the difference between one planned,
# parallel scan and one statement per spill file.
# ---------------------------------------------------------------------------

def ingest_ndjson(table: str, paths, columns: dict[str, str],
                  mode: str = "insert", delete_after: bool = True) -> int:
    if not table.isidentifier():
        raise ValueError(f"bad table name {table!r}")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [str(p) for p in paths]
    if not paths:
        return 0
    cols = ", ".join(f"'{c}': '{t}'" for c, t in columns.items())
    collist = ", ".join(columns)
    verb = {"insert": "INSERT", "replace": "INSERT OR REPLACE",
            "ignore": "INSERT OR IGNORE"}[mode]
    with _wlock:
        con = connect()
        n = con.execute(
            f"{verb} INTO {table} ({collist}) "
            f"SELECT {collist} FROM read_json(?, format='newline_delimited', "
            f"columns={{{cols}}})", [paths]).fetchall()
    if delete_after:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
    return n[0][0] if n else 0


# ---------------------------------------------------------------------------
# kv + cache helpers
# ---------------------------------------------------------------------------

def kv_get(key: str, default=None):
    row = q1("SELECT v FROM kv WHERE k=?", key)
    return json.loads(row[0]) if row else default


def kv_set(key: str, value) -> None:
    execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", key, json.dumps(value))


def kv_set_many(pairs: dict) -> None:
    """One statement for any number of keys (executemany paid ~1.5 ms a
    row: 137 run signatures took 0.2 s)."""
    if not pairs:
        return
    execute("INSERT OR REPLACE INTO kv SELECT unnest(?), unnest(?)",
            list(pairs), [json.dumps(v) for v in pairs.values()])


def kv_get_prefix(prefix: str) -> dict:
    return {r[0]: json.loads(r[1])
            for r in q("SELECT k, v FROM kv WHERE k LIKE ?", prefix + "%")}


_scan_version_cache: int | None = None


def scan_version() -> int:
    """Cached in memory — the UI polls this every second and it must not
    touch the database while a scan holds the writer busy."""
    global _scan_version_cache
    if _scan_version_cache is None:
        _scan_version_cache = int(kv_get("scan_version", 0))
    return _scan_version_cache


def bump_scan_version() -> int:
    global _scan_version_cache
    v = int(kv_get("scan_version", 0)) + 1
    kv_set("scan_version", v)
    _scan_version_cache = v
    return v


# ---------------------------------------------------------------------------
# Stable colour slots — persisted in a small JSON file, never in DuckDB, so
# rendering a figure is a pure read of the store
# ---------------------------------------------------------------------------
_color_lock = threading.Lock()
_colors: dict[str, dict[str, int]] | None = None


def _colors_path() -> Path:
    return settings().state_dir / "colors.json"


def color_seq(scope: str, entities: list[str]) -> dict[str, int]:
    """Stable slot per entity: colour follows the entity, never its rank.

    Every scan assigns the whole catalog's models, groups and runs at once,
    in sorted order (`naming.assign_all_slots`), so pages only read slots and
    a fresh store gets the same colours whatever order pages are opened in.
    An entity this has not seen yet takes the next free slot."""
    global _colors
    with _color_lock:
        if _colors is None:
            try:
                _colors = json.loads(_colors_path().read_text(encoding="utf-8"))
            except Exception:
                _colors = {}
        rows = _colors.setdefault(scope, {})
        missing = sorted(e for e in set(entities) if e not in rows)
        if missing:
            nxt = max(rows.values(), default=-1) + 1
            for i, e in enumerate(missing):
                rows[e] = nxt + i
            try:
                tmp = _colors_path().with_suffix(".json.tmp")
                tmp.write_text(json.dumps(_colors, ensure_ascii=False),
                               encoding="utf-8")
                os.replace(tmp, _colors_path())
            except OSError:
                pass
        return {e: rows[e] for e in entities if e in rows}


# ---------------------------------------------------------------------------
# Raw-line retrieval through the offset index (the "never copy" contract)
# ---------------------------------------------------------------------------
_relpath_cache: dict[int, tuple] = {}


def read_line(relpath_or_fileid, byte_off: int, byte_len: int) -> dict | None:
    """Read one JSON line in place. Returns None when the file is gone or has
    changed since the scan — by size *or* modification time, so a same-size
    edit is caught too (the offsets would then point into other data).
    Callers also compare the line's `_id` with the one they asked for."""
    st = settings()
    size = mtime = None
    if isinstance(relpath_or_fileid, int):
        hit = _relpath_cache.get(relpath_or_fileid)
        if hit is None:
            row = q1("SELECT relpath, size, mtime FROM files WHERE file_id=?",
                     relpath_or_fileid)
            if not row:
                return None
            hit = (row[0], row[1], row[2])
            if len(_relpath_cache) > 4096:
                _relpath_cache.clear()
            _relpath_cache[relpath_or_fileid] = hit
        rel, size, mtime = hit
    else:
        rel = relpath_or_fileid
    p = st.data_root / rel
    try:
        stt = p.stat()
        if size is not None and stt.st_size != size:
            return None
        if mtime is not None and abs(stt.st_mtime - float(mtime)) > 1e-3:
            return None
        with open(p, "rb") as f:
            f.seek(byte_off)
            return json.loads(f.read(byte_len).decode("utf-8", "replace"))
    except Exception:
        return None


def clear_read_cache() -> None:
    _relpath_cache.clear()
