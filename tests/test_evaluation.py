"""Raw JSON in, final scores out, with every expected number written by hand.

The metric grid in test_metrics.py starts from the stored match primitives,
so it cannot catch an error those primitives share (PRMU, de-duplication,
language of matching, repeated lines). This oracle can: a two-document,
bilingual collection whose PRMU classes, gold identities and P/R values are
worked out below from the documented conventions alone.
"""
from __future__ import annotations

import json
import sys

import pytest

from conftest import REPO, scan

sys.path.insert(0, str(REPO))

TEXT = ("Graph based ranking of candidate keyphrases. We study neural "
        "networks for extraction, and a neural, network view.")
AUTHOR = [
    "graph ranking",          # R: both stems present, never adjacent
    "neural network",         # P: "neural networks" (stems: neural network)
    "Neural Networks",        # duplicate of the previous after stemming
    "...",                    # no word token: dropped
    "keyphrase extraction",   # R
    "deep learning",          # U
    "graph mining",           # M
]
READER_FR = ["réseaux de neurones"]      # French gold on an English text: U
PREDS = ["neural networks", "graph ranking", "réseaux de neurones", "unrelated"]


def _tree(root):
    docs = root / "documents"
    docs.mkdir(parents=True)
    (docs / "document.minibi.json").write_text(json.dumps({
        "description": "bilingual oracle",
        "metadata": {"split": {"type": "split"}},
        "document": {"title+abstract": {"type": ["title", "abstract"],
                                        "languages": ["en"]}},
        "annotations": {"author": {"type": "author", "languages": ["en"]},
                        "reader": {"type": "reader", "languages": ["fr"]}}}))
    d1 = {"_id": "d1", "metadata": {"split": "test"},
          "sections": [{"field": "title+abstract", "content": TEXT}],
          "annotations": [{"annotator": "author", "keyphrases": AUTHOR},
                          {"annotator": "reader", "keyphrases": READER_FR}]}
    d1_again = dict(d1, annotations=[{"annotator": "author",
                                      "keyphrases": ["something else"]}])
    d2 = {"_id": "d2", "metadata": {"split": "train"},
          "sections": [{"field": "title+abstract", "content": TEXT}],
          "annotations": [{"annotator": "author", "keyphrases": AUTHOR}]}
    (docs / "document.minibi.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in (d1, d2, d1_again)))
    (root / "models").mkdir()
    (root / "models" / "model.oracle.json").write_text(json.dumps({
        "name": "oracle", "backend": "none",
        "capabilities": {"extractive": True, "abstractive": True}}))
    (root / "architectures").mkdir()
    (root / "architectures" / "architecture.local.json").write_text(json.dumps({
        "arch_id": "local", "kind": "local",
        "variables": {"time": {"unit": "s", "level": "document"}},
        "rates": {"time": {"time": 1.0}}}))
    run = root / "inferences" / "minibi" / "oracle" / "local" / "r1"
    run.mkdir(parents=True)
    (run / "run_r1.json").write_text(json.dumps({"parameters": {}}))
    (run / "batch_00000.json").write_text(json.dumps({"batch_idx": 0}))
    lines = [{"_id": "d1", "inferences": PREDS, "costs": {"time": 1.0}},
             {"_id": "d2", "inferences": PREDS, "costs": {"time": 1.0}},
             # a repeated prediction line: the first one is the one scored
             {"_id": "d1", "inferences": ["unrelated"], "costs": {"time": 1.0}}]
    (run / "batch_00000.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines))
    return root


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    root = _tree(tmp_path_factory.mktemp("oracle") / "data")
    state = root.parent / "state"
    scan(root, state, workers=2)
    import duckdb
    return duckdb.connect(str(state / "kpviz.duckdb"), read_only=True)


def test_gold_is_deduplicated_and_classified(oracle):
    rows = oracle.execute("""SELECT kp_idx, surface, prmu FROM gold
                             WHERE doc_id='d1' AND ann_key='author'
                             ORDER BY kp_idx""").fetchall()
    assert [(r[1], r[2]) for r in rows] == [
        ("graph ranking", "R"), ("neural network", "P"),
        ("keyphrase extraction", "R"), ("deep learning", "U"),
        ("graph mining", "M")]
    assert [r[0] for r in rows] == list(range(5))          # renumbered
    issues = json.loads(oracle.execute(
        "SELECT v FROM kv WHERE k='collection_issues'").fetchone()[0])["minibi"]
    # d1 and d2 each carry one duplicate and one empty keyphrase
    assert issues["gold_duplicates"] == 2 and issues["gold_empty"] == 2
    assert issues["duplicate_doc_ids"] == 1
    # the repeated id's later line left nothing behind
    assert oracle.execute("SELECT count(*) FROM documents WHERE doc_id='d1'"
                          ).fetchone()[0] == 1
    assert oracle.execute("SELECT count(*) FROM gold WHERE surface='something else'"
                          ).fetchone()[0] == 0


def test_present_occurrence_is_where_the_phrase_ends(oracle):
    end_char, = oracle.execute("""SELECT end_char FROM gold WHERE doc_id='d1'
                                  AND ann_key='author' AND prmu='P'""").fetchone()
    assert TEXT[:end_char].endswith("neural networks")


def test_scores_by_hand(oracle):
    got = {r[0]: r[1:] for r in oracle.execute(
        """SELECT ann_key, p_mean, r_mean, n FROM run_metrics WHERE k='M'""").fetchall()}
    # author: 2 of 5 gold matched by 4 unique predictions
    assert got["author"] == pytest.approx((0.5, 0.4, 1))
    # reader (French): the French prediction, analysed in French, matches
    assert got["reader"] == pytest.approx((0.25, 1.0, 1))
    # combined: 6 distinct gold, 3 matched — each in its own language
    assert got["@combined"] == pytest.approx((0.75, 0.5, 1))


def test_run_tags_count_repeats_and_unscored(oracle):
    tags = json.loads(oracle.execute("SELECT tags FROM runs").fetchone()[0])
    assert "duplicate_docs:1" in tags and "unscored:1" in tags
    assert not any(t.startswith("incomplete") for t in tags)
    assert oracle.execute("SELECT count(*) FROM preds WHERE doc_id='d1'"
                          ).fetchone()[0] == 1


def test_contiguous_prmu_unit_cases():
    from kpviz import textproc as tp

    def cls(kp, *sections, lang="en"):
        words = []
        for s in sections:                       # a sentinel between sections
            words += ([tp.SENT] if words else []) + tp.tokens(tp.norm_text(s))
        doc = tp.StemmedDoc(tp.stem_tokens(words, lang))
        var, marked = [], []
        for v in tp.split_variants(kp):
            w, m = tp.phrase_tokens(tp.norm_phrase(v))
            var.append(tp.stem_tokens(w, lang))
            if m is not w:
                marked.append(tp.stem_tokens(m, lang))
        return tp.prmu_classify(var, doc, marked)[0]
    assert cls("neural network", "we train neural networks") == "P"
    assert cls("neural network", "neural very large network") == "R"   # gap
    assert cls("neural network", "network of neural units") == "R"     # reversed
    assert cls("neural network", "a neural, network") == "R"           # separator
    assert cls("neural network", "a neural (network)") == "R"
    assert cls("e-commerce site", "an e-commerce site") == "P"         # infix mark
    assert cls("neural network", "neural", "network") == "R"           # sections
    assert cls("model model", "one model model") == "P"                # repeats
    assert cls("neural network", "neural nets") == "M"
    assert cls("deep learning", "nothing here") == "U"
    assert cls("deep+neural network", "neural network, deep") == "P"   # variants
    assert cls("U.S. army", "the U.S. army moved") == "P"              # own marks
    # French elision and accents: the apostrophe is word-internal
    assert cls("analyse syntaxique", "L'analyse syntaxique de l'arabe",
               lang="fr") == "P"
    assert cls("système", "Le Système MASPAR", lang="fr") == "P"
    # Windows-1252 apostrophe decoded as Latin-1 (U+0092) is repaired
    assert cls("d'analyse", "outil d\x92analyse", lang="fr") == "P"
    # Arabic words stay whole with their harakat (the stemmer drops them);
    # Chinese is one token per character
    assert cls("تحليل نحوي", "نظام تحليل نحوي للغة", lang="ar") == "P"
    assert cls("تحليل", "تَحْلِيل النص", lang="ar") == "P"
    assert cls("学习", "深度学习模型", lang="zh") == "P"
    assert cls("学模", "深度学习模型", lang="zh") == "R"


def test_variant_separator_keeps_plus_signs_in_names():
    from kpviz import textproc as tp
    assert tp.split_variants("C++") == ["C++"]
    assert tp.split_variants("A+ grading") == ["A+ grading"]
    assert tp.split_variants("a+b") == ["a", "b"]


def test_overlapping_alternatives_get_a_maximum_matching():
    """E08: gold {a, b} and {a}, predictions a, b: both can match."""
    from kpviz.derive import _match
    assert _match(["a", "b"], [["a", "b"], ["a"]]) == ([0, 1], [1, 0])
    # the normal, non-overlapping case is unchanged (rank order, first gold)
    assert _match(["x", "a", "b"], [["a"], ["b"], ["c"]]) == ([1, 2], [0, 1])
    # a later prediction never displaces an earlier one out of the matching
    pr, _g = _match(["a", "a2", "b"], [["a", "a2"], ["b"]])
    assert pr == [0, 2]


def test_section_boundaries_in_the_derivation(tmp_path):
    """A phrase inside a later section is present; one spanning two
    sections is not (and its end offset points into the original text)."""
    import os
    from kpviz import derive
    title = "Oncologic disease and therapy"
    abstract = "Recent progress on cardiovascular disease motivates this study."
    doc = {"_id": "x", "metadata": {"split": "test"},
           "sections": [{"field": "title", "content": title},
                        {"field": "abstract", "content": abstract}],
           "annotations": [{"annotator": "author", "keyphrases": [
               "cardiovascular disease", "therapy recent", "oncologic disease"]}]}
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(doc) + "\n")
    derive.derive_doc_chunk({
        "dataset": "t", "path": str(p), "start": 0, "end": os.path.getsize(p),
        "card": {"sections": {"title": ["en"], "abstract": ["en"]},
                 "anns": {"author": ["en"]}},
        "out_dir": str(tmp_path), "tag": "t", "file_id": 0, "tokenizers": []})
    rows = [json.loads(l) for l in (tmp_path / "gold_t.ndjson").read_text().splitlines()]
    got = {r["display"]: (r["prmu"], r["end_char"]) for r in rows}
    full = derive.SECTION_JOIN.join([title, abstract])
    assert got["therapy recent"][0] == "R"
    assert got["oncologic disease"][0] == "P"
    cat, end = got["cardiovascular disease"]
    assert cat == "P" and full[:end].endswith("cardiovascular disease")
