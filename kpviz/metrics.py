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

`rebuild_run_metrics` precomputes the *unfiltered* case in one SQL statement
at the end of a scan. `run_scores` uses it whenever no gold-side filter is
active, which is the common first view — the arithmetic is the same, so the
numbers are identical (tools/test_metric_parity.py asserts it).
"""
from __future__ import annotations

import json
from collections import defaultdict

from . import db

KS = ["5", "10", "O", "M"]
MEASURES = ["f1", "p", "r"]
PRMU = ["P", "R", "M", "U"]
_MEMO_MAX = 4096

# _doc_scores returns (precision, recall, f1) in that order. Selecting by
# MEASURES.index() silently returned precision for "f1" — the SQL parity test
# (tools/test_metric_parity.py) is what surfaced it. Never index by position.
_MEASURE_SLOT = {"p": 0, "r": 1, "f1": 2}


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
                      measure: str, k: str) -> dict | None:
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
# Gold-side filter masks
# ---------------------------------------------------------------------------

def gold_masks(dataset: str, ann_key: str,
               prmu: list[str] | None = None,
               tok_limit: tuple[str, int] | None = None,
               require_position: bool = False) -> dict[str, set[int]]:
    """{doc_id: set(kp_idx allowed)} under the given filters.

    tok_limit=(tokenizer, L): keep gold whose first occurrence ends within
    the first L tokens (i.e. survives input truncation at L).
    require_position: drop gold with no in-text occurrence."""
    if tok_limit:
        sql = """SELECT g.doc_id, g.kp_idx, g.prmu, g.first_char, t.tok_end
                 FROM gold g LEFT JOIN gold_tokpos t
                   ON t.dataset=g.dataset AND t.doc_id=g.doc_id
                  AND t.ann_key=g.ann_key AND t.kp_idx=g.kp_idx
                  AND t.tokenizer=?
                 WHERE g.dataset=? AND g.ann_key=?"""
        params = [tok_limit[0], dataset, ann_key]
    else:
        sql = """SELECT g.doc_id, g.kp_idx, g.prmu, g.first_char
                 FROM gold g WHERE g.dataset=? AND g.ann_key=?"""
        params = [dataset, ann_key]

    allowed: dict[str, set[int]] = defaultdict(set)
    prmu_set = set(prmu) if prmu else None
    for row in db.q(sql, *params):
        doc_id, kp_idx, cat, first_char = row[0], row[1], row[2], row[3]
        if prmu_set is not None and (cat or "U") not in prmu_set:
            continue
        if require_position and (first_char is None or first_char < 0):
            continue
        if tok_limit:
            tok_end = row[4]
            if tok_end is None or tok_end > tok_limit[1]:
                continue
        allowed[doc_id].add(kp_idx)
    return allowed


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _doc_scores(n_uniq: int, n_gold_allowed: int, tp_ranks: list[int],
                k: str) -> tuple[float, float, float]:
    """(precision, recall, f1) for one document — index via _MEASURE_SLOT."""
    if n_gold_allowed <= 0:
        return (float("nan"),) * 3
    cut = {"O": n_gold_allowed, "M": max(n_uniq, 1)}.get(k)
    if cut is None:
        cut = int(k)
    tp = sum(1 for r in tp_ranks if r < cut)
    denom_p = min(cut, n_uniq) if n_uniq else 0
    p = tp / denom_p if denom_p else 0.0
    r = tp / n_gold_allowed
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


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

    run_keys: [(model, arch, run_id), ...] — pass them all at once; the
    dataset-wide gold mask is then built once for the whole list.
    """
    run_keys = [tuple(x) for x in run_keys]
    filtered = (prmu is not None or tok_limit is not None or require_position
                or doc_ids is not None or exclude_doc_ids is not None)

    cache_key = memo_key = None
    if use_cache:
        cache_key = {"fn": "run_scores", "dataset": dataset,
                     "runs": sorted(run_keys), "ann": ann_key, "m": measure,
                     "k": k, "prmu": sorted(prmu) if prmu else None,
                     "tok": tok_limit, "pos": require_position, "pd": per_doc,
                     "docs": sorted(doc_ids) if doc_ids else None,
                     "xdocs": sorted(exclude_doc_ids) if exclude_doc_ids else None}
        memo_key = _memo_key(cache_key)
        hit = _MEMO.get(memo_key)
        if hit is not None:
            return hit
        if not per_doc:
            payload = db.cache_get(cache_key)
            if payload is not None:
                out = {tuple(json.loads(kk)): vv for kk, vv in payload.items()}
                _memo_put(memo_key, out)
                return out
        # exact precomputed table for the unfiltered case
        if not filtered and not per_doc:
            fast = _from_run_metrics(dataset, run_keys, ann_key, measure, k)
            if fast is not None:
                _memo_put(memo_key, fast)
                return fast

    masks = gold_masks(dataset, ann_key, prmu, tok_limit,
                       require_position) if (prmu is not None
                                             or tok_limit is not None
                                             or require_position) else None

    out: dict[tuple, dict] = {}
    mi = _MEASURE_SLOT[measure]
    if run_keys:
        vals = ",".join("(?,?,?)" for _ in run_keys)
        args = [dataset, ann_key] + [x for key in run_keys for x in key]
        rows = db.q(f"""SELECT model, arch, run_id, doc_id, n_uniq, n_gold,
                               pred_ranks, gold_idxs
                        FROM matches
                        WHERE dataset=? AND ann_key=?
                          AND (model, arch, run_id) IN (VALUES {vals})""", *args)
    else:
        rows = []

    acc: dict[tuple, list] = {key: [0.0, 0, 0, {}] for key in run_keys}
    for model, arch, run_id, doc_id, n_uniq, n_gold, ranks, idxs in rows:
        key = (model, arch, run_id)
        slot = acc.get(key)
        if slot is None:
            continue
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        if exclude_doc_ids is not None and doc_id in exclude_doc_ids:
            continue
        ranks = ranks or []
        if masks is not None:
            allowed = masks.get(doc_id, set())
            tp_ranks = [r for r, gi in zip(ranks, idxs or []) if gi in allowed]
            n_allowed = len(allowed)
        else:
            tp_ranks = list(ranks)
            n_allowed = n_gold
        s = _doc_scores(n_uniq or 0, n_allowed, tp_ranks, k)[mi]
        if s != s:                     # NaN -> no gold after filtering
            slot[2] += 1
            continue
        slot[0] += s
        slot[1] += 1
        if per_doc:
            slot[3][doc_id] = s

    for key in run_keys:
        total, n, n_skipped, per_map = acc[key]
        res = {"mean": (total / n) if n else None, "n": n,
               "n_skipped": n_skipped}
        if per_doc:
            res["per_doc"] = per_map
        out[key] = res

    if cache_key is not None:
        if not per_doc:
            db.cache_put(cache_key, {json.dumps(list(kk)): vv
                                     for kk, vv in out.items()})
        _memo_put(memo_key, out)
    return out


# ---------------------------------------------------------------------------
# in-process memo on top of the DuckDB cache (invalidated by scan version)
# ---------------------------------------------------------------------------
_MEMO: dict = {}
_MEMO_ORDER: list = []
_MEMO_VERSION: int | None = None


def _memo_key(cache_key: dict):
    global _MEMO_VERSION, _MEMO, _MEMO_ORDER
    from .util import stable_hash
    v = db.scan_version()
    if v != _MEMO_VERSION:
        _MEMO, _MEMO_ORDER, _MEMO_VERSION = {}, [], v
    return stable_hash(cache_key)


def _memo_put(key, value) -> None:
    if key is None:
        return
    if key not in _MEMO:
        _MEMO_ORDER.append(key)
        while len(_MEMO_ORDER) > _MEMO_MAX:      # bounded: a long-lived server
            _MEMO.pop(_MEMO_ORDER.pop(0), None)  # must not grow without limit
    _MEMO[key] = value
