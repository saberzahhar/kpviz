"""The data contract as users write it (README, paper Fig. 2).

The sample tree deliberately uses every accepted spelling: a JSONC
architecture card, `languages` and `language`, `dataset_a` and `dataset_A`,
slash and ISO-8601 timestamps, a run without its run card, an undeclared
architecture (`n.a`) and illegal parameters.
"""
from __future__ import annotations

import json

from conftest import REPO, run


def test_jsonc_card_resolves(app_ctx):
    db = app_ctx
    rows = db.q("SELECT DISTINCT arch_known FROM runs WHERE model='gpt4o'")
    assert rows == [(True,)], "the README-style (commented) api card must load"
    costs = json.loads(db.q1("SELECT costs FROM runs WHERE model='gpt4o' "
                             "AND dataset='kp20k'")[0])
    assert costs["usd"]["known"] and costs["usd"]["total"] > 0


def test_both_language_spellings(app_ctx):
    db = app_ctx
    # talnarchives declares annotation languages as "languages"
    assert db.q("SELECT DISTINCT lang FROM gold WHERE dataset='talnarchives'") == [("fr",)]


def test_both_similarity_key_spellings(app_ctx, sample_tree):
    db = app_ctx
    n_file = sum(1 for _ in open(sample_tree / "insights" / "scores.jsonl"))
    n = db.q1("SELECT count(*) FROM leakage WHERE dataset_a IS NOT NULL "
              "AND doc_id_b IS NOT NULL")[0]
    assert n == n_file


def test_iso_timestamps_keep_fractions_and_offsets(app_ctx):
    db = app_ctx
    got = db.q1("""SELECT count(*), count(t_start) FROM batches
                   WHERE model='gpt4o'""")
    assert got[0] and got[0] == got[1]


def test_issue_tags(app_ctx):
    db = app_ctx
    tags = {r[0]: json.loads(r[1]) for r in db.q(
        "SELECT dataset || '/' || model || '/' || run_id, tags FROM runs")}
    assert "missing:run" in tags["kptimes/llama3.370BInstruct/ab1a2c4366c3"]
    assert "missing:architecture" in tags["kp20k/llama3.370BInstruct/ab1a2c4366c3"]
    assert any(t.startswith("illegal parameter") for t in tags["kp20k/multipartiterank/mprank-bad"])
    # coverage is computed within the majority split and never exceeds 100 %
    assert db.q1("SELECT max(coverage) FROM runs")[0] <= 1.0


def test_broken_inputs_are_reported_not_dropped(sample_tree, tmp_path):
    """A malformed card and malformed JSONL lines surface on the Overview
    (kv entries) instead of silently disappearing."""
    import shutil
    tree = tmp_path / "tree"
    shutil.copytree(sample_tree, tree)
    (tree / "models" / "model.broken.json").write_text('{"name": "x",,}')
    coll = tree / "documents" / "document.kptimes.jsonl"
    coll.write_text(coll.read_text() + "{not json}\n")
    batch = sorted((tree / "inferences" / "kp20k").rglob("batch_00000.jsonl"))[0]
    batch.write_text(batch.read_text() + "{\"_id\": \n")
    # a cost variable name that used to abort the scan (SQL built from data)
    batch.write_text(batch.read_text() + json.dumps(
        {"_id": "kp20k_testing_0", "inferences": ["graph"],
         "costs": {'gpu\'s "time"': 1.5}}) + "\n")
    state = tmp_path / "st"
    run(REPO / "tools" / "scan_once.py", "--data", tree, "--state", state)
    code = ("import duckdb,sys,json;c=duckdb.connect(sys.argv[1],read_only=True);"
            "kv=dict(c.execute('select k,v from kv').fetchall());"
            "print(json.dumps({k: json.loads(v) for k,v in kv.items() "
            "if k in ('card_errors','collection_issues','run_issues')}))")
    kv = json.loads(run("-c", code, state / "kpviz.duckdb").stdout)
    assert any("model.broken.json" in e["file"] for e in kv["card_errors"])
    assert kv["collection_issues"]["kptimes"]["malformed_lines"] == 1
    assert any(v.get("malformed_lines") for v in kv["run_issues"].values())
