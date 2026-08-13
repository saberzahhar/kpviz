"""DuckDB analytical store (schema v4).

Holds *derived* data and indices only — the corpus and inference files are
addressed by (file, byte offset, length), never copied.

Two design notes that matter for throughput:

* **Reads do not take the write lock.** DuckDB is MVCC internally and a
  cursor per thread is the documented concurrency pattern, so UI queries no
  longer queue behind a bulk COPY. Only writers serialise.
* **The high-volume derived tables carry no primary key.** They are always
  purge-then-insert, so uniqueness holds by construction, and dropping the
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
from .util import stable_hash

SCHEMA_VERSION = 4

# Rebuilt on a schema change. `keyphrases` is deliberately absent: its shape
# is tracked separately (KP_SCHEMA_VERSION) so a schema bump elsewhere never
# throws away the phrase cache — the thing that makes later scans cheap.
_DERIVED_TABLES = [
    "documents", "doc_tokens", "gold", "gold_agg", "gold_tokpos",
    "runs", "batches", "preds", "matches", "leakage", "run_metrics",
    "kp_stage", "agg_cache", "phrase_pos",
]
KP_SCHEMA_VERSION = 1

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

-- the phrase cache: one row per unique normalised phrase, first seen wins
CREATE TABLE IF NOT EXISTS keyphrases(
    kp       VARCHAR PRIMARY KEY,
    raw      VARCHAR,
    lang     VARCHAR,
    tokens   VARCHAR[],
    stems    VARCHAR[],
    n_tokens INTEGER,
    pos      VARCHAR
);
-- worker spills land here first, then one anti-join merges them
CREATE TABLE IF NOT EXISTS kp_stage(
    kp VARCHAR, raw VARCHAR, lang VARCHAR,
    tokens VARCHAR[], stems VARCHAR[], n_tokens INTEGER
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

CREATE TABLE IF NOT EXISTS doc_tokens(
    dataset VARCHAR NOT NULL, doc_id VARCHAR NOT NULL,
    tokenizer VARCHAR NOT NULL,
    n_tokens INTEGER, approx BOOLEAN
);

-- gold *instances*: eval-scope splits plus every quality-flagged document
CREATE TABLE IF NOT EXISTS gold(
    dataset VARCHAR NOT NULL, doc_id VARCHAR NOT NULL,
    ann_key VARCHAR NOT NULL, kp_idx INTEGER NOT NULL,
    display VARCHAR,
    stems   VARCHAR[],
    lang    VARCHAR,
    n_words INTEGER,
    prmu    VARCHAR,
    first_char INTEGER,
    first_word INTEGER
);

-- distribution aggregates for EVERY split (tiny, chart-ready)
CREATE TABLE IF NOT EXISTS gold_agg(
    dataset VARCHAR NOT NULL, split VARCHAR, ann_key VARCHAR NOT NULL,
    prmu VARCHAR, n_words_b INTEGER,
    n BIGINT, words_sum BIGINT
);

CREATE TABLE IF NOT EXISTS gold_tokpos(
    dataset VARCHAR NOT NULL, doc_id VARCHAR NOT NULL,
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
    n_docs INTEGER,
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
    costs VARCHAR,
    known_doc BOOLEAN
);

CREATE TABLE IF NOT EXISTS matches(
    dataset VARCHAR NOT NULL, model VARCHAR NOT NULL,
    arch VARCHAR NOT NULL, run_id VARCHAR NOT NULL,
    doc_id VARCHAR NOT NULL, ann_key VARCHAR NOT NULL,
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

CREATE TABLE IF NOT EXISTS agg_cache(
    key VARCHAR PRIMARY KEY, scan_version INTEGER, payload VARCHAR,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS color_assign(
    scope VARCHAR NOT NULL, entity VARCHAR NOT NULL, seq INTEGER,
    PRIMARY KEY (scope, entity)
);
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
        con.execute(f"SET memory_limit='{max(1, st.duckdb_memory_bytes // 2**20)}MB'")
        con.execute(f"SET threads={st.db_threads}")
        con.execute(f"SET temp_directory='{st.tmp_dir / 'duckdb'}'")
        try:                       # keep bulk ingest from checkpoint-thrashing
            con.execute("SET checkpoint_threshold='1GB'")
        except Exception:
            pass
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


def set_scan_mode(on: bool) -> None:
    """During a scan the process pool owns the cores; afterwards the UI wants
    them for its aggregate queries."""
    st = settings()
    from .hostinfo import usable_cpus
    threads = st.db_threads if on else max(2, usable_cpus())
    try:
        with _wlock:
            connect().execute(f"SET threads={threads}")
    except Exception:
        pass


def cursor() -> duckdb.DuckDBPyConnection:
    """A per-thread cursor. Reads through this are lock-free."""
    cur = getattr(_local, "cur", None)
    if cur is None:
        cur = connect().cursor()
        _local.cur = cur
    return cur


def q(sql: str, *params):
    try:
        return cursor().execute(sql, list(params) if params else None).fetchall()
    except duckdb.Error:
        # a cursor can be invalidated by DDL on another thread; retry once
        _local.cur = None
        return cursor().execute(sql, list(params) if params else None).fetchall()


def q1(sql: str, *params):
    rows = q(sql, *params)
    return rows[0] if rows else None


def qdict(sql: str, *params) -> list[dict]:
    cur = cursor()
    res = cur.execute(sql, list(params) if params else None)
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
    if not pairs:
        return
    executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)",
                [[k, json.dumps(v)] for k, v in pairs.items()])


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


def cache_get(payload_key: dict):
    key = stable_hash(payload_key)
    row = q1("SELECT payload, scan_version FROM agg_cache WHERE key=?", key)
    if row and row[1] == scan_version():
        return json.loads(row[0])
    return None


def cache_put(payload_key: dict, payload) -> None:
    key = stable_hash(payload_key)
    execute("INSERT OR REPLACE INTO agg_cache VALUES (?, ?, ?, now())",
            key, scan_version(), json.dumps(payload))


def cache_clear_stale() -> None:
    execute("DELETE FROM agg_cache WHERE scan_version <> ?", scan_version())


# ---------------------------------------------------------------------------
# Stable color sequencing (one round trip, not one per entity)
# ---------------------------------------------------------------------------

def color_seq(scope: str, entities: list[str]) -> dict[str, int]:
    rows = dict(q("SELECT entity, seq FROM color_assign WHERE scope=?", scope))
    missing = [e for e in entities if e not in rows]
    if missing:
        nxt = max(rows.values(), default=-1) + 1
        new = [[scope, e, nxt + i] for i, e in enumerate(missing)]
        executemany("INSERT OR REPLACE INTO color_assign VALUES (?,?,?)", new)
        rows.update({e: nxt + i for i, e in enumerate(missing)})
    return {e: rows[e] for e in entities if e in rows}


# ---------------------------------------------------------------------------
# Raw-line retrieval through the offset index (the "never copy" contract)
# ---------------------------------------------------------------------------
_relpath_cache: dict[int, str] = {}


def read_line(relpath_or_fileid, byte_off: int, byte_len: int) -> dict | None:
    st = settings()
    if isinstance(relpath_or_fileid, int):
        rel = _relpath_cache.get(relpath_or_fileid)
        if rel is None:
            row = q1("SELECT relpath FROM files WHERE file_id=?",
                     relpath_or_fileid)
            if not row:
                return None
            rel = row[0]
            if len(_relpath_cache) > 4096:
                _relpath_cache.clear()
            _relpath_cache[relpath_or_fileid] = rel
    else:
        rel = relpath_or_fileid
    p = st.data_root / rel
    try:
        with open(p, "rb") as f:
            f.seek(byte_off)
            return json.loads(f.read(byte_len).decode("utf-8", "replace"))
    except Exception:
        return None
