"""The one-statement SQL scoring path against an independent reference.

The reference below is written from the documented conventions (metrics.py
docstring), in plain Python over the stored match primitives and gold rows —
it shares no code with the SQL path. The grid covers every dataset,
annotation set, measure, cutoff and gold filter, and must not be empty.
"""
from __future__ import annotations

import math

import pytest

KS = ("5", "10", "O", "M")


def _reference(db, ds, ann, k, prmu=None, need_pos=False, tok=None):
    allowed = None
    if prmu is not None or need_pos or tok is not None:
        allowed = {}
        sql = """SELECT g.doc_id, g.kp_idx, g.prmu, g.end_char, t.tok_end
                 FROM gold g LEFT JOIN gold_tokpos t
                   ON t.dataset=g.dataset AND t.doc_id=g.doc_id AND t.ann_key=g.ann_key
                  AND t.kp_idx=g.kp_idx AND t.tokenizer=?
                 WHERE g.dataset=? AND g.ann_key=?"""
        for doc, kp, cat, end, tok_end in db.q(sql, tok[0] if tok else "", ds, ann):
            if prmu is not None and (cat or "U") not in prmu:
                continue
            if need_pos and (end is None or end < 0):
                continue
            if tok is not None and (tok_end is None or tok_end > tok[1]):
                continue
            allowed.setdefault(doc, set()).add(kp)
    out = {}
    for model, arch, run, doc, n_uniq, n_gold, ranks, idxs in db.q(
            """SELECT model, arch, run_id, doc_id, n_uniq, n_gold, pred_ranks,
                      gold_idxs FROM matches WHERE dataset=? AND ann_key=?""", ds, ann):
        if allowed is None:
            n_al, hits = n_gold, list(ranks)
        else:
            a = allowed.get(doc, set())
            n_al, hits = len(a), [r for r, gi in zip(ranks, idxs) if gi in a]
        if n_al <= 0:
            continue
        cut = {"O": n_al, "M": max(n_uniq, 1)}.get(k) or int(k)
        tp = sum(1 for r in hits if r < cut)
        dp = min(cut, n_uniq) if n_uniq else 0
        p = tp / dp if dp else 0.0
        r = tp / n_al
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.setdefault((model, arch, run), []).append((doc, p, r, f))
    return out


def test_sql_scoring_matches_reference(app_ctx):
    db = app_ctx
    from kpviz import metrics
    compared = 0
    for (ds,) in db.q("SELECT DISTINCT dataset FROM matches ORDER BY 1"):
        keys = [tuple(r) for r in db.q(
            "SELECT DISTINCT model, arch, run_id FROM matches WHERE dataset=?", ds)]
        tok = db.q1("SELECT tokenizer FROM gold_tokpos WHERE dataset=? LIMIT 1", ds)
        filters = [dict(), dict(prmu=["P"]), dict(prmu=["R", "M", "U"]),
                   dict(prmu=["P"], require_position=True)]
        if tok:
            filters.append(dict(require_position=True, tok_limit=(tok[0], 512)))
        for (ann,) in db.q("SELECT DISTINCT ann_key FROM matches WHERE dataset=?", ds):
            for f in filters:
                for k in KS:
                    ref = _reference(db, ds, ann, k, f.get("prmu"),
                                     f.get("require_position", False),
                                     f.get("tok_limit"))
                    for mi, measure in enumerate(("p", "r", "f1")):
                        got = metrics.run_scores(ds, keys, ann, measure, k,
                                                 per_doc=True, **f)
                        for key in keys:
                            rows = ref.get(key, [])
                            g = got[key]
                            assert g["n"] == len(rows), (ds, ann, f, k, key)
                            if not rows:
                                assert g["mean"] is None
                                continue
                            want = sum(r[1 + mi] for r in rows) / len(rows)
                            assert math.isclose(g["mean"], want, abs_tol=1e-12)
                            pd = g["per_doc"]
                            for doc, *vals in rows:
                                assert math.isclose(pd[doc], vals[mi], abs_tol=1e-12)
                            compared += 1
    assert compared > 500, "parity grid unexpectedly small"


def test_unfiltered_fast_path_equals_sql(app_ctx):
    from kpviz import metrics
    db = app_ctx
    ds = "kp20k"
    keys = [tuple(r) for r in db.q(
        "SELECT DISTINCT model, arch, run_id FROM matches WHERE dataset=?", ds)]
    for k in KS:
        fast = metrics.run_scores(ds, keys, "author", "f1", k)
        slow = metrics.run_scores(ds, keys, "author", "f1", k, per_doc=True)
        for key in keys:
            assert fast[key]["n"] == slow[key]["n"]
            assert math.isclose(fast[key]["mean"], slow[key]["mean"], abs_tol=1e-12)


def test_empty_document_filter_is_not_no_filter(app_ctx):
    """An empty selection means "no documents", even after the unfiltered
    result for the same runs was cached (review B, E06)."""
    from kpviz import metrics
    db = app_ctx
    keys = [tuple(r) for r in db.q(
        "SELECT DISTINCT model, arch, run_id FROM matches WHERE dataset='kp20k'")]
    full = metrics.run_scores("kp20k", keys, "author", "f1", "O", per_doc=True)
    assert any(v["n"] for v in full.values())
    empty = metrics.run_scores("kp20k", keys, "author", "f1", "O", doc_ids=set())
    assert all(v["n"] == 0 and v["mean"] is None for v in empty.values())
    one = next(iter(full[keys[0]]["per_doc"]))
    only = metrics.run_scores("kp20k", keys[:1], "author", "f1", "O", doc_ids={one})
    assert only[keys[0]]["n"] == 1
    excl = metrics.run_scores("kp20k", keys[:1], "author", "f1", "O",
                              exclude_doc_ids={one})
    assert excl[keys[0]]["n"] == full[keys[0]]["n"] - 1


def test_cache_is_bounded_in_bytes(app_ctx):
    from kpviz import metrics
    c = metrics.ByteLRU(1000)
    for i in range(50):
        c.get_or_compute(i, lambda: ("x", 100))
    assert c.stats()["entries"] == 10


def test_single_flight(app_ctx):
    import threading
    import time
    from kpviz import metrics
    c = metrics.ByteLRU(10**6)
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return "v", 1
    ts = [threading.Thread(target=lambda: c.get_or_compute("k", slow)) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1


def _ref_rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, tie, i = [0.0] * len(values), 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie += t ** 3 - t
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks, tie


@pytest.mark.parametrize("n", [7, 300, 5000])
def test_rank_tests_vector_path_is_exact(n):
    """Large samples take the NumPy path; it must give the same average
    ranks and tie term as the reference loop, hence identical p-values."""
    import random
    from kpviz import stats
    R = random.Random(n)
    x = [R.choice([0, .25, .5, 1 / 3, 1]) for _ in range(n)]
    ranks, tie = stats._rankdata(x)
    rr, rt = _ref_rankdata(x)
    assert [float(v) for v in ranks] == rr and tie == rt
    y = [R.choice([0, .25, .5, 1]) for _ in range(n)]
    _u, p = stats.mann_whitney_u(x, y)
    _w, pw = stats.wilcoxon_signed_rank(x, y)
    for v in (p, pw):           # None below the documented minimum sample size
        assert v is None or 0 <= v <= 1
    if n >= 300:
        assert p is not None and pw is not None
