"""Incremental scan orchestration.

Design rules, each one learned from a real trace:

1. **One pool for the whole scan**, created with an initializer that pins
   every nested thread pool to one thread. Tearing pools down between phases
   discarded warm phrase caches, spaCy pipelines and tokenizers, and let
   rayon/BLAS oversubscribe the machine by a factor of the core count.
2. **Every chunk of every unit of work shares that pool** — all collections
   together, all runs together. 75 small runs must occupy 32 workers, not one.
3. **Nothing serial on the critical path.** Spills are ingested by a writer
   thread batching one `read_json` per *list* of files; job planning is a
   generator; the corpus is never read twice; per-file and per-run SQL loops
   are set-based.
4. **Bookkeeping is deferred.** Signatures are written only after the rows
   they describe have landed, so an interrupted scan re-derives rather than
   silently skipping.

Every scan writes exact per-step / per-job timings to
`.kpviz/scan_stats/scan-*.json`. Nothing prints to the console.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import deque
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                ThreadPoolExecutor, as_completed, wait)
from datetime import datetime
from pathlib import Path

from . import db, derive
from .cards import CardIndex, classify_path, is_unknown_token
from .config import settings
from .costs import resolve_costs, resolve_var_totals
from .ingest import Ingestor
from .textproc import pos_available
from .util import RateEMA, file_hash, human_count, human_duration, \
    line_chunks, stable_hash

CODE_VERSION = 11  # bump to force re-derivation after algorithm changes

STEPS = [
    ("discover", "Discover & diff files"),
    ("cards", "Cards & run configurations"),
    ("documents", "Documents: stats, gold analysis, tokens"),
    ("inferences", "Inferences: matching & costs"),
    ("keyphrases", "Keyphrases: POS-tag new unique phrases"),
    ("scores", "Similarity scores (leakage)"),
    ("finalize", "Aggregates, metrics & cache"),
]

_TS_FORMATS = ("%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")
_EVAL_SPLIT_SQL = "lower(coalesce(d.split,'')) NOT IN ('train','training')"
# stand-in for a NULL split so equality joins work (no split may equal it)
_NULL_SPLIT = "@@kpviz_null_split@@"


def _parse_ts(s):
    if not s:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(str(s)[:19], fmt)
        except ValueError:
            continue
    return None


def _mp_context():
    """Avoid plain `fork`: the scan runs in a thread of the server process,
    which holds an open DuckDB connection with its own background threads, and
    forking a multi-threaded process copies mutexes mid-flight (DuckDB warns
    about it; CPython 3.14 changes the Linux default for the same reason).
    The workers are genuinely spawn-safe — they import only `textproc` and
    receive every path through the args dict.

    Exception: `spawn`/`forkserver` children re-import the main module, which
    does not exist in an interactive session (`python -c`, a notebook). There
    we fall back to `fork` and say so in the scan log."""
    import __main__
    main_ok = bool(getattr(__main__, "__file__", None))
    methods = mp.get_all_start_methods()
    if main_ok:
        for name in ("forkserver", "spawn"):
            if name in methods:
                return mp.get_context(name)
    elif "fork" in methods:
        STATE.log_line("no importable __main__ (interactive session) — "
                       "using fork for workers")
        return mp.get_context("fork")
    return mp.get_context()


# ===========================================================================
# progress state
# ===========================================================================
class ScanState:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "_lock", threading.Lock()):
            self.running = False
            self.cancel = False
            self.started_at = None
            self.finished_at = None
            self.error = None
            self.steps = {k: {"key": k, "label": lab, "status": "pending",
                              "done": 0, "total": 0, "unit": "", "detail": "",
                              "t_start": None, "t_end": None}
                          for k, lab in STEPS}
            self.changes = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0}
            self.activity = ""
            self.log: deque = deque(maxlen=200)
            self.jobs: list[dict] = []
            self.timing: dict[str, float] = {}
            self._ema: dict[str, RateEMA] = {}
            self._step_work: dict[str, tuple[float, float]] = {}

    def start(self):
        self.reset()
        with self._lock:
            self.running = True
            self.started_at = time.time()

    def finish(self, error: str | None = None):
        with self._lock:
            self.running = False
            self.error = error
            self.finished_at = time.time()

    def step(self, key: str, **kw):
        with self._lock:
            self.steps[key].update(kw)

    def step_status(self, key: str, status: str):
        with self._lock:
            s = self.steps[key]
            s["status"] = status
            if status == "running" and s["t_start"] is None:
                s["t_start"] = time.time()
            if status == "done":
                s["t_end"] = time.time()

    def add_work(self, key: str, done: float = 0, total: float = 0):
        with self._lock:
            d, t = self._step_work.get(key, (0.0, 0.0))
            self._step_work[key] = (d + done, t + total)
            if done:
                self._ema.setdefault(key, RateEMA()).add(done)

    def add_job(self, **job):
        with self._lock:
            self.jobs.append(job)

    def add_timing(self, key: str, secs: float):
        with self._lock:
            self.timing[key] = round(self.timing.get(key, 0.0) + secs, 4)

    def log_line(self, msg: str):
        with self._lock:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.activity = msg

    _DEFAULT_RATES = {"discover": 2e9, "documents": 40e6,
                      "inferences": 80e6, "keyphrases": 12000.0,
                      "scores": 60e6}

    def snapshot(self) -> dict:
        with self._lock:
            eta, have_eta = 0.0, False
            for key, (d, t) in self._step_work.items():
                rem = max(0.0, t - d)
                if rem <= 0:
                    continue
                ema = self._ema.get(key)
                rate = (ema.rate if ema and ema.rate else None) or \
                    self._DEFAULT_RATES.get(key, 10e6)
                eta += rem / rate
                have_eta = True
            return {
                "running": self.running, "cancel": self.cancel,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "error": self.error,
                "steps": [dict(v) for v in self.steps.values()],
                "changes": dict(self.changes), "activity": self.activity,
                "log": list(self.log)[-12:],
                "eta_s": eta if (have_eta and self.running) else None,
                "elapsed_s": (time.time() - self.started_at) if self.started_at else None,
            }

    def stats_payload(self, extra: dict) -> dict:
        with self._lock:
            steps = []
            for s in self.steps.values():
                d = dict(s)
                if d["t_start"] and d["t_end"]:
                    d["duration_s"] = round(d["t_end"] - d["t_start"], 4)
                steps.append(d)
            return {
                **extra,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "duration_s": (round(self.finished_at - self.started_at, 3)
                               if self.started_at and self.finished_at else None),
                "error": self.error, "changes": dict(self.changes),
                "steps": steps, "jobs": list(self.jobs),
                "timing": dict(self.timing), "log": list(self.log),
            }


STATE = ScanState()
_scan_thread: threading.Thread | None = None
_cards_lock = threading.Lock()
_cards: CardIndex | None = None


def cards() -> CardIndex:
    global _cards
    with _cards_lock:
        if _cards is None:
            _cards = CardIndex.load(settings().data_root)
        return _cards


def refresh_cards() -> CardIndex:
    global _cards
    with _cards_lock:
        _cards = CardIndex.load(settings().data_root)
        return _cards


def start_scan(full_rehash: bool = False) -> bool:
    global _scan_thread
    if STATE.running:
        return False
    STATE.start()
    _scan_thread = threading.Thread(target=_scan_main, args=(full_rehash,),
                                    daemon=True, name="kpviz-scan")
    _scan_thread.start()
    return True


def request_cancel():
    STATE.cancel = True


# ===========================================================================

def _scan_main(full_rehash: bool):
    error = None
    db.set_scan_mode(True)
    try:
        _do_scan(full_rehash)
    except Exception:
        error = traceback.format_exc(limit=8)
        STATE.log_line("scan failed: " + error.strip().splitlines()[-1])
    finally:
        db.set_scan_mode(False)
    STATE.finish(error=error)
    _write_scan_stats(full_rehash)


def _write_scan_stats(full_rehash: bool):
    try:
        st = settings()
        st.stats_dir.mkdir(parents=True, exist_ok=True)
        payload = STATE.stats_payload({
            "scan_version": db.scan_version(),
            "code_version": CODE_VERSION,
            "schema_version": db.SCHEMA_VERSION,
            "full_rehash": full_rehash,
            "budget": st.describe(),
            "backends": _backends_snapshot(),
            "data_root": str(st.data_root),
            "wall_started": (datetime.fromtimestamp(STATE.started_at)
                             .isoformat(timespec="milliseconds")
                             if STATE.started_at else None),
            "wall_finished": (datetime.fromtimestamp(STATE.finished_at)
                              .isoformat(timespec="milliseconds")
                              if STATE.finished_at else None),
        })
        ts = time.strftime("%Y%m%d-%H%M%S",
                           time.localtime(STATE.started_at or time.time()))
        with open(st.stats_dir / f"scan-{ts}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def _backends_snapshot():
    try:
        from .diag import backends
        return {k: v["value"] for k, v in backends().items()}
    except Exception:
        return {}


def _eff_chunk(size: int) -> int:
    """Chunk target: big enough that ingest overhead stays amortised, small
    enough that one file still fans out across the pool."""
    st = settings()
    per_worker = max(1, size // max(1, st.workers * 3))
    return max(4 * 1024 * 1024, min(st.chunk_bytes, per_worker))


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------
_DOC_TABLES = {
    "documents": ({"dataset": "VARCHAR", "doc_id": "VARCHAR", "split": "VARCHAR",
                   "file_id": "BIGINT", "byte_off": "BIGINT", "byte_len": "BIGINT",
                   "n_sections": "INTEGER", "n_chars": "BIGINT", "n_words": "BIGINT",
                   "langs": "VARCHAR[]", "detected_langs": "VARCHAR[]",
                   "sections": "VARCHAR", "metadata": "VARCHAR",
                   "ann_counts": "VARCHAR", "flags": "VARCHAR[]"}, "insert"),
    "gold": ({"dataset": "VARCHAR", "doc_id": "VARCHAR", "ann_key": "VARCHAR",
              "kp_idx": "INTEGER", "display": "VARCHAR", "stems": "VARCHAR[]",
              "lang": "VARCHAR", "n_words": "INTEGER", "prmu": "VARCHAR",
              "first_char": "INTEGER", "first_word": "INTEGER"}, "insert"),
    "gold_agg": ({"dataset": "VARCHAR", "split": "VARCHAR", "ann_key": "VARCHAR",
                  "prmu": "VARCHAR", "n_words_b": "INTEGER", "n": "BIGINT",
                  "words_sum": "BIGINT"}, "insert"),
    "doc_tokens": ({"dataset": "VARCHAR", "doc_id": "VARCHAR",
                    "tokenizer": "VARCHAR", "n_tokens": "INTEGER",
                    "approx": "BOOLEAN"}, "insert"),
    "gold_tokpos": ({"dataset": "VARCHAR", "doc_id": "VARCHAR",
                     "ann_key": "VARCHAR", "kp_idx": "INTEGER",
                     "tokenizer": "VARCHAR", "tok_end": "INTEGER",
                     "approx": "BOOLEAN"}, "insert"),
    "kp_stage": ({"kp": "VARCHAR", "raw": "VARCHAR", "lang": "VARCHAR",
                  "tokens": "VARCHAR[]", "stems": "VARCHAR[]",
                  "n_tokens": "INTEGER"}, "insert"),
}
_PRED_TABLES = {
    "preds": ({"dataset": "VARCHAR", "model": "VARCHAR", "arch": "VARCHAR",
               "run_id": "VARCHAR", "doc_id": "VARCHAR", "batch_idx": "INTEGER",
               "file_id": "BIGINT", "byte_off": "BIGINT", "byte_len": "BIGINT",
               "n_preds": "INTEGER", "n_uniq": "INTEGER",
               "costs": "VARCHAR", "known_doc": "BOOLEAN"}, "insert"),
    "matches": ({"dataset": "VARCHAR", "model": "VARCHAR", "arch": "VARCHAR",
                 "run_id": "VARCHAR", "doc_id": "VARCHAR", "ann_key": "VARCHAR",
                 "n_uniq": "INTEGER", "n_gold": "INTEGER",
                 "pred_ranks": "INTEGER[]", "gold_idxs": "INTEGER[]"}, "insert"),
    "kp_stage": _DOC_TABLES["kp_stage"],
}


def _do_scan(full_rehash: bool):
    st = settings()
    con = db.connect()
    idx = refresh_cards()

    # ---- step 1: discover -------------------------------------------------
    STATE.step_status("discover", "running")
    found, known, deleted = _discover(con, full_rehash)
    STATE.step_status("discover", "done")

    # ---- step 2: cards ----------------------------------------------------
    STATE.step_status("cards", "running")
    run_dirs = _collect_runs(found)
    STATE.step("cards", total=len(run_dirs) or 1, unit="runs",
               done=len(run_dirs),
               detail=f"{len(idx.datasets)} datasets · {len(idx.models)} models · "
                      f"{len(idx.archs)} architectures · {len(run_dirs)} runs")
    STATE.step_status("cards", "done")

    needed_tokenizers = _tokenizers_by_dataset(idx, run_dirs)
    _warm_tokenizers(needed_tokenizers)

    doc_jobs = _plan_doc_jobs(idx, found, known, needed_tokenizers)

    ctx = _mp_context()
    pool = ProcessPoolExecutor(max_workers=st.workers, mp_context=ctx,
                               initializer=derive.init_worker,
                               initargs=(st.phrase_cache_max,))
    try:
        # ---- step 3: documents --------------------------------------------
        STATE.step_status("documents", "running")
        if doc_jobs:
            _derive_documents(con, pool, doc_jobs, needed_tokenizers)
        STATE.step("documents",
                   detail=f"{len(doc_jobs)} collection(s) derived" if doc_jobs
                          else "everything up to date")
        STATE.step_status("documents", "done")

        # ---- step 4: inferences -------------------------------------------
        # planned only now: a run's signature embeds the signature of the
        # collection it was matched against, which the documents phase has
        # just written. Planning earlier would record a stale value and
        # re-derive every run of a fresh collection on the *next* scan.
        STATE.step_status("inferences", "running")
        pred_jobs = _plan_pred_jobs(idx, run_dirs, full_rehash)
        if pred_jobs:
            _derive_all_runs(con, pool, idx, pred_jobs)
        _purge_orphan_runs(con, run_dirs)
        STATE.step("inferences",
                   detail=f"{len(pred_jobs)} run(s) rederived · "
                          f"{len(run_dirs)} runs total")
        STATE.step_status("inferences", "done")

        # ---- merge the phrase cache before POS ----------------------------
        _merge_keyphrases(con)

        # ---- step 5: keyphrases (POS) -------------------------------------
        STATE.step_status("keyphrases", "running")
        _pos_phase(con, pool)
        STATE.step_status("keyphrases", "done")
    finally:
        pool.shutdown(wait=True)

    # ---- step 6: scores ---------------------------------------------------
    STATE.step_status("scores", "running")
    _load_scores(con, found)
    STATE.step_status("scores", "done")

    # ---- step 7: finalize -------------------------------------------------
    STATE.step_status("finalize", "running")
    t0 = time.perf_counter()
    STATE.log_line("aggregating runs…")
    _aggregate_runs_sql(con, idx, run_dirs, found)
    STATE.add_timing("aggregate_runs_s", time.perf_counter() - t0)
    t0 = time.perf_counter()
    STATE.log_line("precomputing unfiltered metrics…")
    from .metrics import rebuild_run_metrics
    n_metrics = rebuild_run_metrics(con)
    STATE.add_timing("run_metrics_s", time.perf_counter() - t0)
    db.bump_scan_version()
    db.cache_clear_stale()
    with db._wlock:
        con.execute("CHECKPOINT")
    STATE.step("finalize", detail=f"catalog version {db.scan_version()} · "
                                  f"{human_count(n_metrics)} metric rows")
    STATE.step_status("finalize", "done")
    STATE.log_line("scan complete in " +
                   human_duration(time.time() - (STATE.started_at or time.time())))


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------

def _discover(con, full_rehash: bool):
    st = settings()
    root = st.data_root
    STATE.log_line("walking the data tree…")
    t0 = time.perf_counter()
    found: dict[str, dict] = {}

    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    name = entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if not name.startswith("."):
                            stack.append(entry.path)
                        continue
                    p = Path(entry.path)
                    info = classify_path(root, p)
                    if info is None:
                        continue
                    stat = entry.stat()          # cached by the directory read
                    rel = str(p.relative_to(root)).replace(os.sep, "/")
                    found[rel] = {"kind": info["kind"], "info": info, "path": p,
                                  "size": stat.st_size, "mtime": stat.st_mtime}
        except OSError:
            continue
    STATE.add_timing("walk_s", time.perf_counter() - t0)

    known = {r[0]: {"file_id": r[1], "size": r[2], "mtime": r[3],
                    "hash": r[4], "sig": r[5]}
             for r in db.q("SELECT relpath, file_id, size, mtime, hash, sig FROM files")}

    # A file only needs hashing when size/mtime moved (or when asked). On a
    # first scan there is nothing to compare a hash against, so hashing every
    # byte of the corpus before deriving it is pure cost.
    to_hash = []
    for rel, f in found.items():
        old = known.get(rel)
        same_stat = (old and old["size"] == f["size"]
                     and abs((old["mtime"] or 0) - f["mtime"]) < 1e-6)
        if same_stat and not full_rehash and st.hash_mode != "always":
            f["hash"], f["file_id"], f["status"] = old["hash"], old["file_id"], "unchanged"
        elif same_stat and old and old.get("hash") and st.hash_mode == "always":
            to_hash.append(rel)
        elif old is None and st.hash_mode == "auto" and not full_rehash:
            f["hash"], f["file_id"], f["status"] = None, None, "new"
        else:
            to_hash.append(rel)

    STATE.step("discover", total=len(found), unit="files")
    hash_bytes = sum(found[r]["size"] for r in to_hash)
    STATE.add_work("discover", total=hash_bytes)
    t0 = time.perf_counter()
    if to_hash:
        STATE.log_line(f"hashing {human_count(len(to_hash))} changed file(s) "
                       f"({human_count(hash_bytes)} bytes)…")
        done_n = 0
        with ThreadPoolExecutor(max_workers=st.io_workers) as hp:
            futs = {hp.submit(file_hash, found[r]["path"]): r for r in to_hash}
            for fut in as_completed(futs):       # no head-of-line blocking
                if STATE.cancel:
                    raise RuntimeError("cancelled")
                rel = futs[fut]
                f = found[rel]
                f["hash"] = fut.result()
                old = known.get(rel)
                if old is None:
                    f["status"], f["file_id"] = "new", None
                elif old["hash"] == f["hash"]:
                    f["status"], f["file_id"] = "unchanged", old["file_id"]
                else:
                    f["status"], f["file_id"] = "modified", old["file_id"]
                done_n += 1
                STATE.add_work("discover", done=f["size"])
                if done_n % 64 == 0 or done_n == len(to_hash):
                    STATE.step("discover", done=done_n,
                               detail=f"hashed {done_n}/{len(to_hash)}")
    STATE.add_timing("hash_s", time.perf_counter() - t0)

    deleted = [rel for rel in known if rel not in found]
    for rel, f in found.items():
        STATE.changes[f["status"]] = STATE.changes.get(f["status"], 0) + 1
    STATE.changes["deleted"] = len(deleted)

    # bulk-register: one statement, not two per file
    t0 = time.perf_counter()
    _register_files(con, found, known)
    if deleted:
        STATE.log_line(f"purging {len(deleted)} deleted file(s)…")
        _purge_deleted(con, deleted, known)
    STATE.add_timing("register_files_s", time.perf_counter() - t0)
    STATE.step("discover", done=len(found),
               detail=f"{len(found)} files · {len(to_hash)} hashed")
    return found, known, deleted


def _register_files(con, found: dict, known: dict):
    st = settings()
    new_rels = [rel for rel, f in found.items() if f.get("file_id") is None]
    if new_rels:
        # one round trip for the whole block: a nextval() per new file is
        # 10k statements on a first scan
        row = db.q1("SELECT coalesce(max(file_id), 0) FROM files")
        start = int(row[0]) + 1
        for i, rel in enumerate(sorted(new_rels)):
            found[rel]["file_id"] = start + i

    rows = []
    for rel, f in found.items():
        info = f["info"]
        rows.append({"relpath": rel, "file_id": int(f["file_id"]),
                     "kind": f["kind"], "dataset": info.get("dataset"),
                     "model": info.get("model"), "arch": info.get("arch"),
                     "run_id": info.get("run"), "batch_idx": info.get("batch_idx"),
                     "size": f["size"], "mtime": f["mtime"], "hash": f["hash"],
                     "sig": (known.get(rel) or {}).get("sig")})
    if not rows:
        return
    spill = st.tmp_dir / "files_register.ndjson"
    derive._write_ndjson(spill, rows)
    db.ingest_ndjson("files", spill, {
        "relpath": "VARCHAR", "file_id": "BIGINT", "kind": "VARCHAR",
        "dataset": "VARCHAR", "model": "VARCHAR", "arch": "VARCHAR",
        "run_id": "VARCHAR", "batch_idx": "INTEGER", "size": "BIGINT",
        "mtime": "DOUBLE", "hash": "VARCHAR", "sig": "VARCHAR"},
        mode="replace")


def _purge_deleted(con, deleted: list[str], known: dict):
    datasets = set()
    for rel in deleted:
        row = db.q1("SELECT kind, dataset FROM files WHERE relpath=?", rel)
        if row and row[0] == "dataset_docs" and row[1]:
            datasets.add(row[1])
    for ds in datasets:
        _purge_dataset_docs(con, [ds])
    if deleted:
        ph = ",".join("?" * len(deleted))
        db.execute(f"DELETE FROM files WHERE relpath IN ({ph})", *deleted)


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _collect_runs(found: dict) -> dict[tuple, dict]:
    runs: dict[tuple, dict] = {}
    for rel, f in found.items():
        info = f["info"]
        if f["kind"] not in ("run_card", "batch_meta", "batch_preds"):
            continue
        key = (info["dataset"], info["model"], info["arch"], info["run"])
        r = runs.setdefault(key, {"run_card": None, "metas": {}, "preds": {}})
        if f["kind"] == "run_card":
            r["run_card"] = (rel, f)
        elif f["kind"] == "batch_meta":
            r["metas"][info["batch_idx"]] = (rel, f)
        else:
            r["preds"][info["batch_idx"]] = (rel, f)
    return runs


def _tokenizers_by_dataset(idx: CardIndex, run_dirs: dict) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for (ds, model, arch, _run) in run_dirs:
        specs = out.setdefault(ds, set())
        for spec in idx.model(model).tokenizer_specs:
            specs.add(spec)
        for var, vspec in idx.arch(arch).variables.items():
            if vspec.get("tokenizer"):
                specs.add(vspec["tokenizer"])
    return {k: sorted(v) for k, v in out.items()}


def _warm_tokenizers(needed: dict[str, list[str]]):
    """Resolve every tokenizer once, here, in the parent. Otherwise N workers
    race on a cold cache: N concurrent downloads of the same asset and N
    non-atomic writes to one file."""
    from .textproc import get_tokenizer
    st = settings()
    specs = sorted({s for v in needed.values() for s in v})
    if not specs:
        return
    t0 = time.perf_counter()
    for spec in specs:
        tk = get_tokenizer(spec, st.tokenizer_cache)
        exact = tk.exact
        STATE.log_line(f"tokenizer {spec}: "
                       + ("exact" if exact else "APPROXIMATE (asset unavailable)"))
    STATE.add_timing("warm_tokenizers_s", time.perf_counter() - t0)


def _doc_sig(f: dict, card, tokenizers: list[str]) -> str:
    st = settings()
    return stable_hash({"hash": f["hash"], "size": f["size"],
                        "mtime": None if f["hash"] else round(f["mtime"], 3),
                        "code": CODE_VERSION, "card": card.raw,
                        "tok": sorted(tokenizers), "tokscope": st.token_scope,
                        "gold": st.gold_scope})


def _plan_doc_jobs(idx, found, known, needed_tokenizers) -> list[dict]:
    jobs = []
    for rel, f in sorted(found.items()):
        if f["kind"] != "dataset_docs":
            continue
        ds = f["info"]["dataset"]
        card = idx.dataset(ds)
        sig = _doc_sig(f, card, needed_tokenizers.get(ds, []))
        if f["status"] == "unchanged" and known.get(rel, {}).get("sig") == sig:
            continue
        jobs.append({"rel": rel, "f": f, "ds": ds, "card": card, "sig": sig})
    return jobs


def _plan_pred_jobs(idx: CardIndex, run_dirs: dict,
                    full_rehash: bool) -> list[dict]:
    """One kv read for every stored signature instead of one per run."""
    stored = db.kv_get_prefix("run_sig:")
    have_costs = {tuple(r) for r in
                  db.q("SELECT dataset, model, arch, run_id FROM runs")}
    doc_sigs = {r[0]: r[1] for r in db.q(
        "SELECT dataset, sig FROM files WHERE kind='dataset_docs'")}
    jobs = []
    for key, parts in sorted(run_dirs.items()):
        ds, model, arch, run_id = key
        sig = stable_hash({
            "code": CODE_VERSION,
            "docs": doc_sigs.get(ds),
            "model_card": idx.model(model).raw.get("languages"),
            "preds": {i: f["hash"] or (f["size"], round(f["mtime"], 3))
                      for i, (rel, f) in sorted(parts["preds"].items())},
            "metas": {i: f["hash"] or (f["size"], round(f["mtime"], 3))
                      for i, (rel, f) in sorted(parts["metas"].items())},
        })
        if (not full_rehash and stored.get(f"run_sig:{'/'.join(key)}") == sig
                and key in have_costs):
            continue
        jobs.append({"key": key, "parts": parts, "sig": sig,
                     "bytes": sum(f["size"] for rel, f in parts["preds"].values())})
    return jobs


# ---------------------------------------------------------------------------
# pool driver
# ---------------------------------------------------------------------------

def _run_pool(pool, fn, args_iter, step_key, ingestor: Ingestor | None,
              on_result=None, max_inflight: int | None = None) -> dict:
    """Submit from an *iterator* with backpressure; hand results straight to
    the ingest thread so the pool never waits on DuckDB."""
    st = settings()
    max_inflight = max_inflight or max(4, st.workers * 3)
    it = iter(args_iter)
    pending = set()
    agg = {"n_docs": 0, "secs": 0.0, "items": 0, "wait_s": 0.0}
    exhausted = False

    def _fill():
        nonlocal exhausted
        while not exhausted and len(pending) < max_inflight:
            try:
                pending.add(pool.submit(fn, next(it)))
            except StopIteration:
                exhausted = True

    _fill()
    while pending:
        if STATE.cancel:
            for f in pending:
                f.cancel()
            raise RuntimeError("cancelled")
        t0 = time.perf_counter()
        done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
        agg["wait_s"] += time.perf_counter() - t0
        for fut in done:
            res = fut.result()
            if ingestor is not None:
                ingestor.submit(res)
                ingestor.check()
            if on_result:
                on_result(res)
            agg["n_docs"] += res.get("n_docs", 0)
            agg["items"] += res.get("items", 0)
            agg["secs"] += res.get("secs", 0.0)
            work = res.get("bytes", res.get("items", 0))
            STATE.add_work(step_key, done=work)
            snap = STATE.steps[step_key]
            STATE.step(step_key, done=snap["done"] + work)
        _fill()
    STATE.add_timing(f"{step_key}_pool_wait_s", agg["wait_s"])
    STATE.add_timing(f"{step_key}_worker_s", agg["secs"])
    return agg


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def _derive_documents(con, pool, doc_jobs: list[dict],
                      needed_tokenizers: dict[str, list[str]]):
    st = settings()
    total_bytes = sum(j["f"]["size"] for j in doc_jobs)
    STATE.step("documents", total=total_bytes, unit="bytes")
    STATE.add_work("documents", total=total_bytes)

    _purge_dataset_docs(con, sorted({j["ds"] for j in doc_jobs}))

    per_rel = {j["rel"]: {"remaining": 0, "ds": j["ds"], "sig": j["sig"],
                          "bytes": j["f"]["size"], "n_docs": 0,
                          "worker_s": 0.0, "t0": time.perf_counter()}
               for j in doc_jobs}

    def jobs_iter():
        for j in doc_jobs:
            f, card, ds = j["f"], j["card"], j["ds"]
            STATE.log_line(f"deriving documents of “{ds}”")
            combined = len(card.annotations) > 1
            for i, (a, b) in enumerate(line_chunks(f["path"], _eff_chunk(f["size"]))):
                per_rel[j["rel"]]["remaining"] += 1
                yield {
                    "path": str(f["path"]), "start": a, "end": b,
                    "dataset": ds, "file_id": f["file_id"],
                    "out_dir": str(st.tmp_dir), "tag": f"{ds}_{i}",
                    "rel": j["rel"],
                    "card": {"sections": {k: v.get("languages", [])
                                          for k, v in card.sections.items()},
                             "anns": {k: v.get("languages", [])
                                      for k, v in card.annotations.items()},
                             "combined": combined},
                    "tokenizers": sorted(needed_tokenizers.get(ds, [])),
                    "tok_cache": str(st.tokenizer_cache),
                    "token_scope": st.token_scope, "gold_scope": st.gold_scope,
                }

    def on_result(res):
        info = per_rel[res["rel"]]
        info["remaining"] -= 1
        info["n_docs"] += res.get("n_docs", 0)
        info["worker_s"] += res.get("secs", 0.0)

    ing = Ingestor(_DOC_TABLES)
    try:
        _run_pool(pool, derive.derive_doc_chunk, jobs_iter(), "documents",
                  ing, on_result=on_result)
        ing.drain()
    finally:
        ing.close()
    STATE.add_timing("documents_ingest_s", ing.ingest_s)

    # signatures only now that every row has landed
    sigs = [[info["sig"], rel] for rel, info in per_rel.items()
            if info["remaining"] == 0]
    if sigs:
        db.executemany("UPDATE files SET sig=? WHERE relpath=?", sigs)
    for rel, info in per_rel.items():
        STATE.add_job(step="documents", label=info["ds"], bytes=info["bytes"],
                      n_docs=info["n_docs"],
                      worker_s=round(info["worker_s"], 3),
                      duration_s=round(time.perf_counter() - info["t0"], 3))
        STATE.log_line(f"“{info['ds']}” derived — "
                       f"{human_count(info['n_docs'])} documents")


# ---------------------------------------------------------------------------
# inferences
# ---------------------------------------------------------------------------

def _build_gold_pack(con, ds: str, path: Path) -> int:
    """One NDJSON pack per dataset, parsed lazily per document in the workers.

    Scope: every document in an eval split, plus every quality-flagged
    document whatever its split (so the data-quality workbench can score
    them). Predictions only ever reference those, so nothing is lost."""
    rows = db.q(f"""
        SELECT g.doc_id, g.ann_key, g.stems, g.lang
        FROM gold g JOIN documents d
          ON d.dataset = g.dataset AND d.doc_id = g.doc_id
        WHERE g.dataset = ? AND ({_EVAL_SPLIT_SQL} OR len(d.flags) > 0)
        ORDER BY g.doc_id, g.ann_key, g.kp_idx""", ds)
    packed: dict[str, dict] = {}
    for doc_id, ann_key, stems, lang in rows:
        e = packed.setdefault(doc_id, {"_id": doc_id, "langs": {}, "gold": {}})
        e["langs"].setdefault(ann_key, lang)
        e["gold"].setdefault(ann_key, []).append(list(stems or []))
    path.write_bytes(b"")                 # keep it readable even when empty
    derive._write_ndjson(path, list(packed.values()))
    return len(packed)


def _derive_all_runs(con, pool, idx: CardIndex, pred_jobs: list[dict]):
    st = settings()
    total_bytes = sum(j["bytes"] for j in pred_jobs)
    STATE.step("inferences", total=total_bytes, unit="bytes")
    STATE.add_work("inferences", total=total_bytes)

    t0 = time.perf_counter()
    packs: dict[str, Path] = {}
    for ds in sorted({j["key"][0] for j in pred_jobs}):
        p = st.tmp_dir / ("goldpack_" + ds.replace("/", "_") + ".ndjson")
        n = _build_gold_pack(con, ds, p)
        packs[ds] = p
        STATE.log_line(f"gold pack “{ds}”: {human_count(n)} documents")
    STATE.add_timing("gold_packs_s", time.perf_counter() - t0)

    # one DELETE for every changed run instead of one per run
    _purge_runs(con, [j["key"] for j in pred_jobs])

    # batch metadata: bulk, not one statement per batch file
    t0 = time.perf_counter()
    brows = []
    for job in pred_jobs:
        ds, model, arch, run_id = job["key"]
        parts = job["parts"]
        for bidx in sorted(set(parts["metas"]) | set(parts["preds"])):
            meta = {}
            if bidx in parts["metas"]:
                try:
                    with open(parts["metas"][bidx][1]["path"], encoding="utf-8") as fh:
                        meta = json.load(fh) or {}
                except Exception:
                    meta = {}
            t_s, t_e = (_parse_ts(meta.get("start_timestamp")),
                        _parse_ts(meta.get("end_timestamp")))
            brows.append({
                "dataset": ds, "model": model, "arch": arch, "run_id": run_id,
                "batch_idx": bidx, "n_docs": None,
                "costs": json.dumps(meta.get("costs") or {}),
                "t_start": t_s.isoformat() if t_s else None,
                "t_end": t_e.isoformat() if t_e else None,
                "wall_s": ((t_e - t_s).total_seconds() if (t_s and t_e) else None)})
    if brows:
        spill = st.tmp_dir / "batches_register.ndjson"
        derive._write_ndjson(spill, brows)
        db.ingest_ndjson("batches", spill, {
            "dataset": "VARCHAR", "model": "VARCHAR", "arch": "VARCHAR",
            "run_id": "VARCHAR", "batch_idx": "INTEGER", "n_docs": "INTEGER",
            "costs": "VARCHAR", "t_start": "TIMESTAMP", "t_end": "TIMESTAMP",
            "wall_s": "DOUBLE"})
    STATE.add_timing("batch_meta_s", time.perf_counter() - t0)

    per_run = {j["key"]: {"remaining": 0, "sig": j["sig"], "bytes": j["bytes"],
                          "n_docs": 0, "worker_s": 0.0}
               for j in pred_jobs}
    STATE.log_line(f"matching {len(pred_jobs)} run(s) across the pool…")

    def jobs_iter():
        for job in pred_jobs:
            ds, model, arch, run_id = job["key"]
            primary = (idx.dataset(ds).languages[:1] or ["en"])[0]
            for bidx, (rel, f) in sorted(job["parts"]["preds"].items()):
                for ci, (a, b) in enumerate(line_chunks(f["path"],
                                                        _eff_chunk(f["size"]))):
                    per_run[job["key"]]["remaining"] += 1
                    yield {
                        "path": str(f["path"]), "start": a, "end": b,
                        "dataset": ds, "model": model, "arch": arch,
                        "run_id": run_id, "batch_idx": bidx,
                        "file_id": f["file_id"], "out_dir": str(st.tmp_dir),
                        "tag": f"{ds}_{model}_{arch}_{run_id}_{bidx}_{ci}".replace("/", "_"),
                        "gold_pack": str(packs[ds]), "primary_lang": primary,
                        "run_key": list(job["key"]),
                    }

    def on_result(res):
        info = per_run[tuple(res["run_key"])]
        info["remaining"] -= 1
        info["n_docs"] += res.get("n_docs", 0)
        info["worker_s"] += res.get("secs", 0.0)

    ing = Ingestor(_PRED_TABLES)
    try:
        _run_pool(pool, derive.derive_preds_chunk, jobs_iter(), "inferences",
                  ing, on_result=on_result)
        ing.drain()
    finally:
        ing.close()
    STATE.add_timing("inferences_ingest_s", ing.ingest_s)

    # ---- one global pass for everything that used to be per-run ----------
    t0 = time.perf_counter()
    datasets = sorted(packs)
    ph = ",".join("?" * len(datasets))
    db.execute(f"""UPDATE preds SET known_doc = TRUE FROM documents d
                   WHERE preds.known_doc IS NULL AND preds.dataset IN ({ph})
                     AND d.dataset = preds.dataset AND d.doc_id = preds.doc_id""",
               *datasets)
    db.execute(f"""UPDATE preds SET known_doc = FALSE
                   WHERE known_doc IS NULL AND dataset IN ({ph})""", *datasets)
    db.execute("""UPDATE batches SET n_docs = sub.n FROM (
                    SELECT dataset, model, arch, run_id, batch_idx, count(*) AS n
                    FROM preds GROUP BY 1,2,3,4,5) sub
                  WHERE batches.dataset=sub.dataset AND batches.model=sub.model
                    AND batches.arch=sub.arch AND batches.run_id=sub.run_id
                    AND batches.batch_idx=sub.batch_idx""")
    STATE.add_timing("known_doc_s", time.perf_counter() - t0)

    db.kv_set_many({f"run_sig:{'/'.join(k)}": info["sig"]
                    for k, info in per_run.items() if info["remaining"] == 0})
    for k, info in per_run.items():
        STATE.add_job(step="inferences", label="/".join(k), bytes=info["bytes"],
                      n_docs=info["n_docs"],
                      worker_s=round(info["worker_s"], 3))
    for p in packs.values():
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# purges (set-based)
# ---------------------------------------------------------------------------

def _purge_dataset_docs(con, datasets: list[str]):
    if not datasets:
        return
    ph = ",".join("?" * len(datasets))
    for t in ("documents", "gold", "gold_agg", "doc_tokens", "gold_tokpos"):
        total = db.q1(f"SELECT count(*) FROM {t}")[0]
        mine = db.q1(f"SELECT count(*) FROM {t} WHERE dataset IN ({ph})",
                     *datasets)[0]
        if mine and mine == total:
            db.execute(f"DELETE FROM {t}")
        elif mine:
            db.execute(f"DELETE FROM {t} WHERE dataset IN ({ph})", *datasets)


def _purge_runs(con, keys: list[tuple]):
    if not keys:
        return
    live = {tuple(k) for k in keys}
    for t in ("preds", "matches", "batches", "runs", "run_metrics"):
        rows = db.q1(f"SELECT count(*) FROM {t}")
        if not rows or not rows[0]:
            continue
        present = {tuple(r) for r in
                   db.q(f"SELECT DISTINCT dataset, model, arch, run_id FROM {t}")}
        if present and present <= live:
            db.execute(f"DELETE FROM {t}")
            continue
        vals = ",".join("(?,?,?,?)" for _ in keys)
        args = [x for k in keys for x in k]
        db.execute(f"""DELETE FROM {t} WHERE (dataset, model, arch, run_id)
                       IN (VALUES {vals})""", *args)


def _purge_orphan_runs(con, run_dirs: dict):
    live = set(run_dirs.keys())
    gone = [tuple(k) for k in
            db.q("SELECT DISTINCT dataset, model, arch, run_id FROM runs")
            if tuple(k) not in live]
    if gone:
        _purge_runs(con, gone)
        db.kv_set_many({f"run_sig:{'/'.join(k)}": None for k in gone})


# ---------------------------------------------------------------------------
# keyphrase cache merge
# ---------------------------------------------------------------------------

def _merge_keyphrases(con):
    """One anti-join merges the staged phrases into the global cache.

    Feeding 2 M phrases through `INSERT OR IGNORE` chunk by chunk pays an ART
    probe per row on the scan thread; grouping and anti-joining once is a
    single parallel set operation."""
    n = db.q1("SELECT count(*) FROM kp_stage")
    if not n or not n[0]:
        return
    t0 = time.perf_counter()
    STATE.log_line(f"merging {human_count(n[0])} staged phrase analyses…")
    with db._wlock:
        con.execute("""
            INSERT INTO keyphrases (kp, raw, lang, tokens, stems, n_tokens)
            SELECT s.kp, any_value(s.raw), any_value(s.lang),
                   any_value(s.tokens), any_value(s.stems), any_value(s.n_tokens)
            FROM kp_stage s
            WHERE NOT EXISTS (SELECT 1 FROM keyphrases k WHERE k.kp = s.kp)
            GROUP BY s.kp""")
        con.execute("DELETE FROM kp_stage")
    STATE.add_timing("merge_keyphrases_s", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# run aggregation — one set of queries for every run
# ---------------------------------------------------------------------------

def _cost_var_sums(table: str) -> dict[tuple, dict[str, tuple[float, int]]]:
    """{(ds,model,arch,run): {var: (sum, n_observations)}} computed in SQL.

    The previous version pulled every `costs` JSON string of every prediction
    row into Python and parsed it one at a time — millions of `json.loads`
    calls on the scan thread, on every scan."""
    keys = [r[0] for r in db.q(
        f"SELECT DISTINCT unnest(json_keys(costs)) FROM {table} "
        f"WHERE costs IS NOT NULL AND costs <> '{{}}'")]
    if not keys:
        return {}
    parts = []
    for i, k in enumerate(keys):
        expr = f"TRY_CAST(json_extract_string(costs, '$.\"{k}\"') AS DOUBLE)"
        parts.append(f"sum({expr}) AS s{i}, count({expr}) AS n{i}")
    rows = db.q(f"""SELECT dataset, model, arch, run_id, {', '.join(parts)}
                    FROM {table} WHERE costs IS NOT NULL GROUP BY 1,2,3,4""")
    out: dict[tuple, dict[str, tuple[float, int]]] = {}
    for r in rows:
        key = tuple(r[:4])
        d = {}
        for i, k in enumerate(keys):
            s, n = r[4 + 2 * i], r[5 + 2 * i]
            if n:
                d[k] = (float(s or 0.0), int(n))
        out[key] = d
    return out


def _aggregate_runs_sql(con, idx: CardIndex, run_dirs: dict, found: dict):
    datasets_present = {f["info"]["dataset"] for f in found.values()
                        if f["kind"] == "dataset_docs"}

    counts = {tuple(r[:4]): r[4:] for r in db.q(
        """SELECT dataset, model, arch, run_id, count(*), count(DISTINCT doc_id),
                  sum(CASE WHEN known_doc THEN 0 ELSE 1 END)
           FROM preds GROUP BY 1,2,3,4""")}
    coverage = {tuple(r[:4]): (r[4], r[5]) for r in db.q(f"""
        WITH pd AS (
          SELECT p.dataset, p.model, p.arch, p.run_id,
                 coalesce(d.split, '{_NULL_SPLIT}') AS split, count(*) AS c
          FROM preds p JOIN documents d
            ON d.dataset = p.dataset AND d.doc_id = p.doc_id
          GROUP BY 1,2,3,4,5),
        maj AS (SELECT dataset, model, arch, run_id,
                       arg_max(split, c) AS split FROM pd GROUP BY 1,2,3,4),
        dsz AS (SELECT dataset, coalesce(split, '{_NULL_SPLIT}') AS split,
                       count(*) AS n FROM documents GROUP BY 1,2)
        SELECT maj.dataset, maj.model, maj.arch, maj.run_id, maj.split, dsz.n
        FROM maj LEFT JOIN dsz
          ON dsz.dataset = maj.dataset AND dsz.split = maj.split""")}
    doc_sums_all = _cost_var_sums("preds")
    batch_sums_all = _cost_var_sums("batches")
    batch_meta = {tuple(r[:4]): r[4:] for r in db.q(
        """SELECT dataset, model, arch, run_id, count(*), sum(wall_s),
                  min(t_start), max(t_end)
           FROM batches GROUP BY 1,2,3,4""")}

    rows = []
    for key, parts in sorted(run_dirs.items()):
        ds, model, arch, run_id = key
        mcard, acard = idx.model(model), idx.arch(arch)

        params = {}
        if parts["run_card"]:
            try:
                with open(parts["run_card"][1]["path"], encoding="utf-8") as fh:
                    params = (json.load(fh) or {}).get("parameters", {}) or {}
            except Exception:
                params = {}
        resolved, violations = mcard.validate_params(params) if mcard.raw else \
            ({k: {"value": v, "source": "given"} for k, v in params.items()}, [])

        n_rows, n_docs, n_unknown = counts.get(key, (0, 0, 0))
        n_unknown = int(n_unknown or 0)
        _split, expected = coverage.get(key, (None, None))
        cov = (n_docs / expected) if (expected or 0) > 0 else None
        n_batches, wall_sum, t_start, t_end = batch_meta.get(key, (0, None, None, None))

        var_totals = resolve_var_totals(acard, doc_sums_all.get(key, {}),
                                        batch_sums_all.get(key, {}),
                                        n_docs or 0, n_batches or 0,
                                        wall_sum or None)
        costs = resolve_costs(acard, var_totals, n_docs or 0)

        tags: list[str] = []
        if ds not in datasets_present:
            tags.append("missing:dataset")
        if is_unknown_token(arch) or not acard.known:
            tags.append("missing:architecture")
        if parts["run_card"] is None:
            tags.append("missing:run")
        if not mcard.raw:
            tags.append("missing:model")
        for v in violations:
            t = f"illegal parameter:{v['param']}"
            if t not in tags:
                tags.append(t)
        if cov is not None and cov < 0.999:
            tags.append(f"incomplete:{100 * cov:.0f}%")
        if n_unknown:
            tags.append(f"unresolved_ids:{n_unknown}")

        rows.append({
            "dataset": ds, "model": model, "arch": arch, "run_id": run_id,
            "params": json.dumps(params, ensure_ascii=False),
            "resolved": json.dumps(resolved, ensure_ascii=False, default=str),
            "violations": json.dumps(violations, ensure_ascii=False, default=str),
            "tags": json.dumps(tags, ensure_ascii=False),
            "arch_known": acard.known,
            "n_batches": n_batches or 0, "n_docs": n_docs or 0,
            "n_dup_docs": (n_rows or 0) - (n_docs or 0),
            "expected_docs": expected, "coverage": cov,
            "var_totals": json.dumps(var_totals), "costs": json.dumps(costs),
            "t_start": t_start.isoformat() if t_start else None,
            "t_end": t_end.isoformat() if t_end else None,
            "wall_s": wall_sum})

    if not rows:
        return
    st = settings()
    spill = st.tmp_dir / "runs_register.ndjson"
    derive._write_ndjson(spill, rows)
    db.ingest_ndjson("runs", spill, {
        "dataset": "VARCHAR", "model": "VARCHAR", "arch": "VARCHAR",
        "run_id": "VARCHAR", "params": "VARCHAR", "resolved": "VARCHAR",
        "violations": "VARCHAR", "tags": "VARCHAR", "arch_known": "BOOLEAN",
        "n_batches": "INTEGER", "n_docs": "INTEGER", "n_dup_docs": "INTEGER",
        "expected_docs": "INTEGER", "coverage": "DOUBLE",
        "var_totals": "VARCHAR", "costs": "VARCHAR",
        "t_start": "TIMESTAMP", "t_end": "TIMESTAMP", "wall_s": "DOUBLE"},
        mode="replace")


# ---------------------------------------------------------------------------
# POS phase
# ---------------------------------------------------------------------------

def _pos_chunk_size(total: int, workers: int) -> int:
    """Fan the phase across the pool: a fixed 20 k chunk left 29 of 32
    workers idle whenever fewer than ~600 k phrases were new."""
    return int(max(500, min(20_000, total // max(1, workers * 4) or 1)))


def _pos_phase(con, pool):
    st = settings()
    db.execute("UPDATE keyphrases SET pos=NULL WHERE trim(coalesce(pos,'')) = ''")
    per_lang = {r[0]: r[1] for r in db.q(
        """SELECT coalesce(lang,'en'), count(*) FROM keyphrases
           WHERE pos IS NULL GROUP BY 1""")}
    langs_ok = {l for l in per_lang if pos_available(l)}
    total = sum(per_lang[l] for l in langs_ok)
    skipped = sum(n for l, n in per_lang.items() if l not in langs_ok)
    STATE.step("keyphrases", total=total or 1, done=0, unit="phrases",
               detail=(f"{human_count(total)} new phrase(s) to tag"
                       + (f" · {human_count(skipped)} without a spaCy model"
                          if skipped else "")) if (total or skipped)
               else "phrase cache up to date")
    STATE.add_work("keyphrases", total=total)
    if not total:
        return
    chunk = _pos_chunk_size(total, st.workers)
    STATE.log_line(f"POS-tagging {human_count(total)} new unique keyphrases "
                   f"({chunk} per task)…")

    def task_iter():
        for lang in sorted(langs_ok):
            cur = db.connect().cursor()
            try:
                cur.execute("""SELECT kp, coalesce(raw, kp) FROM keyphrases
                               WHERE pos IS NULL AND coalesce(lang,'en')=?""",
                            [lang])
                i = 0
                while True:
                    batch = cur.fetchmany(chunk)
                    if not batch:
                        break
                    yield {"phrases": [[kp, raw] for kp, raw in batch],
                           "lang": lang, "out_dir": str(st.tmp_dir),
                           "tag": f"{lang}_{i}"}
                    i += 1
            finally:
                cur.close()

    pos_files: list[str] = []
    t_ing = [0.0]

    def flush(paths):
        if not paths:
            return
        t0 = time.perf_counter()
        with db._wlock:
            con.execute("CREATE OR REPLACE TEMP TABLE tmp_pos AS "
                        "SELECT * FROM read_json(?, format='newline_delimited',"
                        " columns={'kp':'VARCHAR','pos':'VARCHAR'})", [paths])
            con.execute("UPDATE keyphrases SET pos = tmp_pos.pos FROM tmp_pos "
                        "WHERE keyphrases.kp = tmp_pos.kp")
            con.execute("DROP TABLE IF EXISTS tmp_pos")
        t_ing[0] += time.perf_counter() - t0
        for p in paths:
            try:
                Path(p).unlink()
            except OSError:
                pass

    def on_result(res):
        if res.get("pos"):
            pos_files.append(res["pos"])
            if len(pos_files) >= 64:      # spread the lock instead of one
                flush(list(pos_files))    # stop-the-world update at the end
                pos_files.clear()

    _run_pool(pool, derive.pos_chunk, task_iter(), "keyphrases", None,
              on_result=on_result)
    flush(list(pos_files))
    STATE.add_timing("pos_ingest_s", t_ing[0])


def _load_scores(con, found: dict):
    score_files = {rel: f for rel, f in found.items() if f["kind"] == "scores"}
    sigs = db.kv_get_prefix("scores_sig:")
    changed = [rel for rel, f in score_files.items()
               if f["status"] != "unchanged"
               or sigs.get(f"scores_sig:{rel}") != (f["hash"] or f["size"])]
    if not changed and score_files:
        STATE.step("scores", detail="up to date")
        return
    db.execute("DELETE FROM leakage")
    total = sum(f["size"] for f in score_files.values())
    STATE.step("scores", total=max(total, 1), unit="bytes")
    STATE.add_work("scores", total=total)
    done = 0
    new_sigs = {}
    for rel, f in sorted(score_files.items()):
        STATE.log_line(f"loading similarity scores ({rel})")
        db.execute("""
            INSERT INTO leakage
            SELECT dataset_A, CAST(doc_id_A AS VARCHAR),
                   dataset_B, CAST(doc_id_B AS VARCHAR),
                   CAST(score AS DOUBLE), label
            FROM read_json(?, format='newline_delimited', columns={
                'dataset_A':'VARCHAR','doc_id_A':'VARCHAR',
                'dataset_B':'VARCHAR','doc_id_B':'VARCHAR',
                'score':'DOUBLE','label':'VARCHAR'})""", str(f["path"]))
        done += f["size"]
        STATE.add_work("scores", done=f["size"])
        STATE.step("scores", done=done)
        new_sigs[f"scores_sig:{rel}"] = f["hash"] or f["size"]
    db.kv_set_many(new_sigs)
