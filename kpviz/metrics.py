"""Evaluation metrics over the match primitives.

The DB stores, per (run, document, annotation set), the ranks of the correct
predictions and which gold keyphrase each one hit. Any P/R/F1 at any cutoff
under any gold-side filter (PRMU class, within-context-window,
leakage-excluded documents, …) reduces to counting pairs — no re-matching, no
re-reading files.

Conventions (documented in every export caption):
  * predictions are lowercased, spaCy-tokenised, stemmed and de-duplicated,
    keeping first occurrence order; gold '+'-variants all count as the same
    keyphrase;
  * P@k = tp / min(k, #unique predictions)   (no padding penalty),
  * R@k = tp / #gold-after-filter,
  * @O uses #gold-after-filter as the cutoff, @M uses all predictions;
  * documents with zero gold after filtering are excluded (n reported);
  * scores are macro-averaged over documents.

Execution. Every score — filtered or not, aggregate or per document — is one
DuckDB statement over `matches`, vectorised, parallel and outside the GIL;
results come back as NumPy columns. (The previous path fetched every match
row into Python lists and looped: 2.7–3.2 s per call at 22 runs × 20 k
documents, of which 1.9 s was building the Python rows; one statement takes
~0.13 s.) `rebuild_run_metrics` still precomputes the unfiltered aggregate
table at the end of a scan, the common first view.

Memory. Per-document results are `PerDoc` objects: a sorted int32 array of
document ordinals into a per-dataset index shared by every result, plus a
float64 score array — about 12 bytes per document instead of ~100 for a
dict entry, and without a private copy of every id string. They sit in a
cache bounded by *bytes* (not entries), with single-flight: concurrent
callbacks asking for the same scores wait for one computation.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from . import db

KS = ["5", "10", "O", "M"]
MEASURES = ["f1", "p", "r"]
PRMU = ["P", "R", "M", "U"]
CACHE_BYTES = 192 * 2**20       # overridable via set_cache_budget()
def _max_concurrent() -> int:
    """Scoring statements allowed at once: DuckDB runs them outside the GIL,
    so up to one per core overlaps well; beyond that they only queue."""
    try:
        from .hostinfo import usable_cpus
        return max(2, usable_cpus())
    except Exception:
        return 2


def metric_label(measure: str, k: str, prmu: list[str] | None = None) -> str:
    lab = f"{measure.upper()}@{k}"
    if prmu and sorted(prmu) != sorted(PRMU):
        lab += " [" + "".join(sorted(prmu, key=PRMU.index)) + "]"
    return lab


# ---------------------------------------------------------------------------
# Precomputed unfiltered scores (exact, computed in SQL)
# ---------------------------------------------------------------------------
_RUN_METRICS_SQL = """
INSERT INTO run_metrics
WITH ks(k, kv) AS (VALUES ('5', 5), ('10', 10), ('O', -1), ('M', -2)),
d AS (
  SELECT m.dataset, m.model, m.arch, m.run_id, m.ann_key,
         ks.k,
         CASE ks.k WHEN 'O' THEN m.n_gold
                   WHEN 'M' THEN greatest(m.n_uniq, 1)
                   ELSE ks.kv END AS cut,
         m.n_uniq, m.n_gold, m.pred_ranks
  FROM matches m CROSS JOIN ks
  WHERE m.n_gold > 0),
s AS (
  SELECT dataset, model, arch, run_id, ann_key, k,
         len(list_filter(pred_ranks, x -> x < cut))::DOUBLE AS tp,
         CASE WHEN n_uniq > 0 THEN least(cut, n_uniq) ELSE 0 END AS dp,
         n_gold
  FROM d),
sc AS (
  SELECT dataset, model, arch, run_id, ann_key, k,
         CASE WHEN dp > 0 THEN tp / dp ELSE 0.0 END AS p,
         tp / n_gold AS r
  FROM s)
SELECT dataset, model, arch, run_id, ann_key, k,
       avg(CASE WHEN p + r > 0 THEN 2 * p * r / (p + r) ELSE 0.0 END) AS f1_mean,
       avg(p) AS p_mean, avg(r) AS r_mean, count(*) AS n
FROM sc GROUP BY 1,2,3,4,5,6
"""


def rebuild_run_metrics(con) -> int:
    """Recompute the unfiltered metric table. Returns the row count."""
    with db._wlock:
        con.execute("DELETE FROM run_metrics")
        con.execute(_RUN_METRICS_SQL)
    row = db.q1("SELECT count(*) FROM run_metrics")
    return row[0] if row else 0


def _from_run_metrics(dataset: str, run_keys: list[tuple], ann_key: str,
                      measure: str, k: str) -> dict:
    if not run_keys:
        return {}
    col = {"f1": "f1_mean", "p": "p_mean", "r": "r_mean"}[measure]
    vals = ",".join("(?,?,?)" for _ in run_keys)
    args = [dataset, ann_key, k] + [x for key in run_keys for x in key]
    rows = db.q(f"""SELECT model, arch, run_id, {col}, n FROM run_metrics
                    WHERE dataset=? AND ann_key=? AND k=?
                      AND (model, arch, run_id) IN (VALUES {vals})""", *args)
    got = {(r[0], r[1], r[2]): {"mean": r[3], "n": int(r[4]), "n_skipped": 0}
           for r in rows}
    # a run with no rows here has no scoreable document; report it as such
    for key in run_keys:
        got.setdefault(tuple(key), {"mean": None, "n": 0, "n_skipped": 0})
    return got


# ---------------------------------------------------------------------------
# Per-dataset document index and compact per-document results
# ---------------------------------------------------------------------------
class DocIndex:
    """Sorted document ids of one dataset; position = ordinal."""

    __slots__ = ("ids", "pos")

    def __init__(self, ids: list[str]):
        self.ids = ids
        self.pos = {d: i for i, d in enumerate(ids)}

    def ords(self, doc_ids) -> np.ndarray:
        pos = self.pos
        return np.fromiter(sorted({pos[d] for d in doc_ids if d in pos}),
                           dtype=np.int32)


class PerDoc(Mapping):
    """Per-document scores of one run: a read-only {doc_id: score} mapping
    backed by two arrays (sorted ordinals, scores) over a shared DocIndex."""

    __slots__ = ("ords", "vals", "index")

    def __init__(self, ords: np.ndarray, vals: np.ndarray, index: DocIndex):
        self.ords, self.vals, self.index = ords, vals, index

    # Mapping protocol (keeps every caller that used a dict working)
    def __len__(self):
        return len(self.ords)

    def __iter__(self):
        ids = self.index.ids
        return (ids[o] for o in self.ords.tolist())

    def __getitem__(self, doc_id):
        o = self.index.pos.get(doc_id)
        if o is not None:
            i = int(np.searchsorted(self.ords, o))
            if i < len(self.ords) and self.ords[i] == o:
                return float(self.vals[i])
        raise KeyError(doc_id)

    def __contains__(self, doc_id):
        try:
            self[doc_id]
            return True
        except KeyError:
            return False

    # vectorised helpers
    def values_array(self) -> np.ndarray:
        return self.vals

    def as_dict(self) -> dict:
        ids = self.index.ids
        return {ids[o]: float(v) for o, v in zip(self.ords.tolist(),
                                                  self.vals.tolist())}

    def select(self, doc_ids, inside: bool = True) -> np.ndarray:
        """Scores of the documents in (or not in) `doc_ids`."""
        mask = np.isin(self.ords, self.index.ords(doc_ids))
        return self.vals[mask if inside else ~mask]

    @property
    def nbytes(self) -> int:
        return int(self.ords.nbytes + self.vals.nbytes)


def paired(a: PerDoc, b: PerDoc) -> tuple[np.ndarray, np.ndarray]:
    """Scores of the documents both results share, aligned (sorted by id)."""
    common, ia, ib = np.intersect1d(a.ords, b.ords, assume_unique=True,
                                    return_indices=True)
    return a.vals[ia], b.vals[ib]


def common_ords(results: list[PerDoc]) -> np.ndarray:
    out = None
    for r in results:
        out = r.ords if out is None else np.intersect1d(out, r.ords,
                                                        assume_unique=True)
    return out if out is not None else np.zeros(0, dtype=np.int32)


def values_at(r: PerDoc, ords: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(r.ords, ords)
    return r.vals[idx]


def doc_index(dataset: str) -> DocIndex:
    """The dataset's document index for the current catalog (cached)."""
    key = ("docindex", dataset, db.scan_version())

    def build():
        ids = [r[0] for r in db.q("""SELECT DISTINCT doc_id FROM documents
                                     WHERE dataset=? ORDER BY doc_id""", dataset)]
        idx = DocIndex(ids)
        return idx, sum(len(d) + 120 for d in ids)   # ~ bytes of str + dict slot
    return _CACHE.get_or_compute(key, build)


# ---------------------------------------------------------------------------
# The scoring statement
# ---------------------------------------------------------------------------

def _scores_sql(n_runs: int, prmu: bool, need_pos: bool, tok: bool,
                doc_filter: str | None) -> str:
    """One statement: per-document P, R, F1 for every selected run.

    Parameters are named: $ds, $ann, $k, $r{i}_{0..2} (run keys) and, when
    the corresponding filter is on, $prmu, $tok, $tok_limit. Filters that are
    off are left out of the SQL text entirely."""
    runs = ",".join(f"({i}, $r{i}_0, $r{i}_1, $r{i}_2)" for i in range(n_runs))
    filtered = prmu or need_pos or tok
    if filtered:
        conds = ["g.dataset = $ds", "g.ann_key = $ann"]
        if prmu:
            conds.append("list_contains($prmu, coalesce(g.prmu, 'U'))")
        if need_pos:
            conds.append("coalesce(g.end_char, -1) >= 0")
        tok_join = ""
        if tok:
            tok_join = ("LEFT JOIN gold_tokpos t ON t.dataset = g.dataset "
                        "AND t.doc_id = g.doc_id AND t.ann_key = g.ann_key "
                        "AND t.kp_idx = g.kp_idx AND t.tokenizer = $tok")
            conds.append("t.tok_end IS NOT NULL AND t.tok_end <= $tok_limit")
        gold_cte = f"""
    ok AS (
      SELECT g.doc_id, list(g.kp_idx) AS allowed, count(*)::INTEGER AS n_ok
      FROM gold g {tok_join}
      WHERE {" AND ".join(conds)}
      GROUP BY 1),"""
        join = "LEFT JOIN ok ON ok.doc_id = m.doc_id"
        n_allowed = "coalesce(ok.n_ok, 0)"
        allowed = ", ok.allowed"
        # pairs (rank, gold index) with rank inside the cutoff and the gold
        # keyphrase allowed by the filter
        tp = ("len(list_filter(range(len(c.pred_ranks)), i -> c.pred_ranks[i + 1] < c.cut "
              "AND list_contains(c.allowed, c.gold_idxs[i + 1])))")
    else:
        gold_cte, join, n_allowed, allowed = "", "", "m.n_gold", ""
        tp = "len(list_filter(c.pred_ranks, x -> x < c.cut))"
    dfilter = ""
    if doc_filter == "include":
        dfilter = "AND m.doc_id IN (SELECT doc_id FROM kp_docfilter)"
    elif doc_filter == "exclude":
        dfilter = "AND m.doc_id NOT IN (SELECT doc_id FROM kp_docfilter)"
    return f"""
    WITH sel(ri, model, arch, run_id) AS (VALUES {runs}),
    di AS (SELECT doc_id, (row_number() OVER (ORDER BY doc_id) - 1)::INTEGER AS ord
           FROM (SELECT DISTINCT doc_id FROM documents WHERE dataset = $ds)),{gold_cte}
    base AS (
      SELECT sel.ri, m.doc_id, m.n_uniq,
             {n_allowed} AS n_al, m.pred_ranks, m.gold_idxs{allowed}
      FROM matches m {join}
      JOIN sel ON sel.model = m.model AND sel.arch = m.arch AND sel.run_id = m.run_id
      WHERE m.dataset = $ds AND m.ann_key = $ann {dfilter}),
    c AS (
      SELECT *, CASE $k WHEN 'O' THEN n_al WHEN 'M' THEN greatest(n_uniq, 1)
                        ELSE TRY_CAST($k AS INTEGER) END AS cut
      FROM base WHERE n_al > 0),
    t AS (
      SELECT c.ri, c.doc_id, c.n_al,
             {tp}::DOUBLE AS tp,
             CASE WHEN c.n_uniq > 0 THEN least(c.cut, c.n_uniq) ELSE 0 END AS dp
      FROM c),
    s AS (
      SELECT ri, doc_id,
             CASE WHEN dp > 0 THEN tp / dp ELSE 0.0 END AS p,
             tp / n_al AS r
      FROM t),
    skipped AS (
      SELECT ri, count(*) AS n FROM base WHERE n_al <= 0 GROUP BY 1)
    -- the run is returned as its position in `sel` (a small integer): a
    -- concatenated key per row was 440 k Python strings to fetch and sort
    SELECT s.ri AS run, di.ord,
           s.p, s.r,
           CASE WHEN s.p + s.r > 0 THEN 2 * s.p * s.r / (s.p + s.r) ELSE 0.0 END AS f1,
           -1::BIGINT AS n_skipped
    FROM s JOIN di ON di.doc_id = s.doc_id
    UNION ALL      -- one row per run with its count of gold-less documents
    SELECT ri, -1, 0.0, 0.0, 0.0, n FROM skipped
    ORDER BY run, ord
    """


def _compute(dataset: str, run_keys: list[tuple], ann_key: str, k: str,
             prmu, tok_limit, require_position, doc_ids, exclude_doc_ids):
    """{run_key: {"p","r","f1": PerDoc, "n_skipped"}} — one statement."""
    doc_filter = ("include" if doc_ids is not None
                  else "exclude" if exclude_doc_ids is not None else None)
    sql = _scores_sql(len(run_keys), prmu is not None, bool(require_position),
                      tok_limit is not None, doc_filter)
    params = {"ds": dataset, "ann": ann_key, "k": str(k)}
    if prmu is not None:
        params["prmu"] = sorted(prmu)
    if tok_limit is not None:
        params.update(tok=tok_limit[0], tok_limit=int(tok_limit[1]))
    for i, key in enumerate(run_keys):
        for j, part in enumerate(key):
            params[f"r{i}_{j}"] = part

    index = doc_index(dataset)
    with _SEM:
        cur = db.connect().cursor()
        try:
            if doc_filter:
                ids = doc_ids if doc_filter == "include" else exclude_doc_ids
                cur.execute("CREATE OR REPLACE TEMP TABLE kp_docfilter(doc_id VARCHAR)")
                if ids:
                    cur.execute("INSERT INTO kp_docfilter SELECT unnest(?)",
                                [sorted(str(d) for d in ids)])
            cols = cur.execute(sql, params).fetchnumpy()
        finally:
            cur.close()

    runs = np.asarray(cols["run"], dtype=np.int64)
    n_sk = np.asarray(cols["n_skipped"], dtype=np.int64)
    skip_rows = n_sk >= 0
    skipped = {runs[i]: int(n_sk[i]) for i in np.flatnonzero(skip_rows)}
    keep = ~skip_rows
    runs = runs[keep]
    ords = np.asarray(cols["ord"], dtype=np.int32)[keep]
    vals = {m: np.asarray(cols[m], dtype=np.float64)[keep]
            for m in ("p", "r", "f1")}
    bounds = {}
    if len(runs):
        change = np.flatnonzero(runs[1:] != runs[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(runs)]))
        for a, b in zip(starts.tolist(), ends.tolist()):
            bounds[runs[a]] = (a, b)
    out = {}
    for ri, key in enumerate(run_keys):
        name = ri
        a, b = bounds.get(name, (0, 0))
        o = ords[a:b].copy()
        out[tuple(key)] = {m: PerDoc(o, vals[m][a:b].copy(), index)
                           for m in ("p", "r", "f1")}
        out[tuple(key)]["n_skipped"] = skipped.get(name, 0)
    return out


def run_scores(dataset: str, run_keys: list[tuple[str, str, str]],
               ann_key: str, measure: str = "f1", k: str = "O",
               prmu: list[str] | None = None,
               tok_limit: tuple[str, int] | None = None,
               require_position: bool = False,
               doc_ids: set[str] | None = None,
               exclude_doc_ids: set[str] | None = None,
               per_doc: bool = False,
               use_cache: bool = True) -> dict:
    """Macro-averaged scores for runs of one dataset.

    run_keys: [(model, arch, run_id), ...] — pass them all at once: one
    statement scores every run. `doc_ids=None` means no document filter;
    an empty set means "no documents" (it is never confused with None).
    Returns {run_key: {"mean", "n", "n_skipped"[, "per_doc": PerDoc]}}.
    """
    run_keys = [tuple(x) for x in run_keys]
    filtered = (prmu is not None or tok_limit is not None or require_position
                or doc_ids is not None or exclude_doc_ids is not None)
    if not run_keys:
        return {}
    if not filtered and not per_doc:
        return _from_run_metrics(dataset, run_keys, ann_key, measure, k)

    def compute():
        res = _compute(dataset, sorted(run_keys), ann_key, k, prmu, tok_limit,
                       require_position, doc_ids, exclude_doc_ids)
        size = sum(pd.nbytes for r in res.values()
                   for m, pd in r.items() if m != "n_skipped")
        return res, size

    if use_cache and doc_ids is None and exclude_doc_ids is None:
        _note_recent(dataset, run_keys, ann_key, k, prmu, tok_limit,
                     require_position)
    if use_cache:
        key = ("scores", dataset, tuple(sorted(run_keys)), ann_key, str(k),
               tuple(sorted(prmu)) if prmu is not None else None,
               tuple(tok_limit) if tok_limit is not None else None,
               bool(require_position),
               None if doc_ids is None else frozenset(doc_ids),
               None if exclude_doc_ids is None else frozenset(exclude_doc_ids),
               db.scan_version())
        res = _CACHE.get_or_compute(key, compute)
    else:
        res = compute()[0]

    out: dict[tuple, dict] = {}
    for key in run_keys:
        r = res[tuple(key)]
        pd = r[measure]
        n = len(pd)
        item = {"mean": float(pd.vals.mean()) if n else None, "n": n,
                "n_skipped": r["n_skipped"]}
        if per_doc:
            item["per_doc"] = pd
        out[key] = item
    return out


# ---------------------------------------------------------------------------
# Warm-up: the views a person actually looked at are recomputed in the
# background after the server starts and after every scan, so the first
# click after either is a cache hit (the demo's "warm start").
# ---------------------------------------------------------------------------
_RECENT: "OrderedDict[str, dict]" = OrderedDict()
_RECENT_MAX = 48
_recent_lock = threading.Lock()
_warm_state = {"running": False, "done": 0, "total": 0, "generation": 0}


def _recent_path():
    try:
        from .config import settings
        return settings().state_dir / "recent_views.json"
    except Exception:
        return None


def _load_recent() -> None:
    import json
    p = _recent_path()
    if not p or not p.exists() or _RECENT:
        return
    try:
        for kw in json.loads(p.read_text(encoding="utf-8")):
            _RECENT[json.dumps(kw, sort_keys=True)] = kw
    except Exception:
        pass


def _note_recent(dataset, run_keys, ann_key, k, prmu, tok_limit,
                 require_position) -> None:
    import json
    import os
    kw = {"dataset": dataset, "run_keys": sorted([list(r) for r in run_keys]),
          "ann_key": ann_key, "k": str(k),
          "prmu": sorted(prmu) if prmu is not None else None,
          "tok_limit": list(tok_limit) if tok_limit is not None else None,
          "require_position": bool(require_position)}
    sig = json.dumps(kw, sort_keys=True)
    with _recent_lock:
        _load_recent()
        new = sig not in _RECENT
        _RECENT[sig] = kw
        _RECENT.move_to_end(sig)
        while len(_RECENT) > _RECENT_MAX:
            _RECENT.popitem(last=False)
        items = list(_RECENT.values()) if new else None
    if items is None:
        return
    p = _recent_path()
    if p:
        try:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(items), encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            pass


def _default_views() -> list[dict]:
    """Before anything was viewed: every dataset's full-run per-document
    scores (RQ2, RQ3 panel b, RQ4 intervals, RQ5 all start from these)."""
    out = []
    try:
        for (ds,) in db.q("SELECT DISTINCT dataset FROM run_metrics ORDER BY 1"):
            keys = [list(r) for r in db.q(
                "SELECT DISTINCT model, arch, run_id FROM run_metrics WHERE dataset=?",
                ds)]
            anns = [r[0] for r in db.q(
                "SELECT DISTINCT ann_key FROM run_metrics WHERE dataset=? ORDER BY 1", ds)]
            if not keys or not anns:
                continue
            ann = "@combined" if "@combined" in anns else anns[0]
            out.append({"dataset": ds, "run_keys": sorted(keys), "ann_key": ann,
                        "k": "O", "prmu": None, "tok_limit": None,
                        "require_position": False})
    except Exception:
        pass
    return out


_warm_enabled = [False]
_warm_thread = [None]


def enable_warm() -> None:
    """Only a serving process warms (a command-line scan exiting while a
    daemon thread sits inside DuckDB aborts the interpreter). The exit hook
    stops a running warm-up and waits for it."""
    import atexit
    if not _warm_enabled[0]:
        _warm_enabled[0] = True
        atexit.register(_stop_warm)


def _stop_warm() -> None:
    _warm_state["generation"] += 1
    th = _warm_thread[0]
    if th is not None and th.is_alive():
        th.join(timeout=5)


def warm_async() -> None:
    """Recompute recent views (most recent first) in one background thread.
    A newer call or a catalog change supersedes a running warm-up."""
    if not _warm_enabled[0]:
        return
    with _recent_lock:
        _load_recent()
        views = list(reversed(_RECENT.values())) or _default_views()
        _warm_state["generation"] += 1
        gen = _warm_state["generation"]
        _warm_state.update(running=True, done=0, total=len(views))

    def run():
        version = db.scan_version()
        for kw in views:
            if _warm_state["generation"] != gen or db.scan_version() != version:
                return
            try:
                run_scores(per_doc=True, **dict(kw, run_keys=[tuple(r) for r in kw["run_keys"]],
                                                tok_limit=tuple(kw["tok_limit"])
                                                if kw.get("tok_limit") else None))
            except Exception:
                pass
            _warm_state["done"] += 1
        if _warm_state["generation"] == gen:
            _warm_state["running"] = False
    th = threading.Thread(target=run, daemon=True, name="kpviz-warm")
    _warm_thread[0] = th
    th.start()


def warm_status() -> dict:
    return dict(_warm_state)


# ---------------------------------------------------------------------------
# Result cache: bounded in bytes, single-flight
# ---------------------------------------------------------------------------
_SEM = threading.BoundedSemaphore(_max_concurrent())
_POOL = None


def run_scores_many(calls: dict) -> dict:
    """Several `run_scores` calls at once: {name: kwargs} -> {name: result}.

    A workbench that needs a few independent scorings (RQ3: present gold,
    every distinct context window, all gold) runs them concurrently — each
    is one DuckDB statement executed outside the GIL."""
    global _POOL
    if len(calls) <= 1:
        return {k: run_scores(**kw) for k, kw in calls.items()}
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=_max_concurrent(),
                                   thread_name_prefix="kpviz-score")
    futs = {k: _POOL.submit(run_scores, **kw) for k, kw in calls.items()}
    return {k: f.result() for k, f in futs.items()}


class ByteLRU:
    """LRU cache bounded by the summed size of its values.

    `get_or_compute(key, fn)`: fn() -> (value, nbytes). Concurrent callers of
    the same missing key wait for the first one's computation instead of
    repeating it — a page load fires several callbacks that need the same
    per-document scores."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._d: OrderedDict = OrderedDict()
        self._sizes: dict = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._inflight: dict = {}
        self.hits = self.misses = 0

    def get_or_compute(self, key, fn):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            ev = self._inflight.get(key)
            owner = ev is None
            if owner:
                ev = self._inflight[key] = [threading.Event(), None, None]
                self.misses += 1
        if not owner:
            ev[0].wait()
            if ev[2] is not None:
                raise ev[2]
            return ev[1]
        try:
            value, size = fn()
        except BaseException as e:
            ev[2] = e
            with self._lock:
                self._inflight.pop(key, None)
            ev[0].set()
            raise
        with self._lock:
            ev[1] = value
            self._inflight.pop(key, None)
            if size <= self.max_bytes:
                self._d[key] = value
                self._sizes[key] = size
                self._bytes += size
                while self._bytes > self.max_bytes and self._d:
                    old, _v = self._d.popitem(last=False)
                    self._bytes -= self._sizes.pop(old, 0)
        ev[0].set()
        return value

    def clear(self):
        with self._lock:
            self._d.clear()
            self._sizes.clear()
            self._bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._d), "mb": round(self._bytes / 2**20, 1),
                    "budget_mb": round(self.max_bytes / 2**20),
                    "hits": self.hits, "misses": self.misses}


_CACHE = ByteLRU(CACHE_BYTES)


def set_cache_budget(nbytes: int) -> None:
    _CACHE.max_bytes = int(nbytes)


def cache_stats() -> dict:
    return _CACHE.stats()


def memo(key, fn, size: int = 4096, sized: bool = False):
    """Cache a derived value for the current catalog (UI helpers). With
    sized=True, fn returns (value, nbytes) and the cache charges that."""
    if sized:
        return _CACHE.get_or_compute((key, db.scan_version()), fn)
    return _CACHE.get_or_compute((key, db.scan_version()), lambda: (fn(), size))
