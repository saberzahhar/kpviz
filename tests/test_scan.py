"""Scan invariants: every derived number is a function of the input folder.

* the worker count does not change any derived table (determinism);
* a no-op scan derives nothing and publishes nothing;
* an incremental re-scan after an edit equals a clean rebuild;
* an edit that cannot change derived rows (a card description) re-derives
  nothing.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from conftest import REPO, run, scan


def _fp(state: Path) -> dict:
    out = run(REPO / "tools" / "bench" / "fingerprint.py", "--state", state).stdout
    return json.loads(out)


def test_worker_count_does_not_change_results(sample_tree, tmp_path):
    # own stores: the session store is held open by the read-side tests
    four, one = tmp_path / "w4", tmp_path / "w1"
    scan(sample_tree, four, workers=4)
    scan(sample_tree, one, workers=1)
    a, b = _fp(four), _fp(one)
    assert a.keys() == b.keys()
    diff = {t: (a[t], b[t]) for t in a if a[t] != b[t]}
    assert not diff, f"tables differ between 4 workers and 1 worker: {diff}"


def test_noop_scan_publishes_nothing(sample_tree, tmp_path):
    state = tmp_path / "st"
    scan(sample_tree, state)
    before = _fp(state)
    out = scan(sample_tree, state)
    assert "nothing changed" in out or "up to date" in out
    assert _fp(state) == before
    ver = json.loads(run("-c", (
        "import duckdb,sys;c=duckdb.connect(sys.argv[1],read_only=True);"
        "print(c.execute(\"select v from kv where k='scan_version'\").fetchone()[0])"),
        state / "kpviz.duckdb").stdout)
    assert ver == 1, "a no-op scan must not bump the catalog version"


def test_incremental_equals_clean_rebuild(sample_tree, tmp_path):
    tree = tmp_path / "tree"
    shutil.copytree(sample_tree, tree)
    inc = tmp_path / "inc"
    scan(tree, inc)
    # edit: drop the last prediction line of one batch, change one gold list
    batch = sorted((tree / "inferences" / "kp20k").rglob("batch_00000.jsonl"))[0]
    lines = batch.read_text(encoding="utf-8").splitlines()
    batch.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    coll = tree / "documents" / "document.kptimes.jsonl"
    docs = coll.read_text(encoding="utf-8").splitlines()
    d = json.loads(docs[-1])
    d["annotations"][0]["keyphrases"].append("interest rates")
    docs[-1] = json.dumps(d, ensure_ascii=False)
    coll.write_text("\n".join(docs) + "\n", encoding="utf-8")
    scan(tree, inc)
    clean = tmp_path / "clean"
    scan(tree, clean)
    a, b = _fp(inc), _fp(clean)
    diff = {t: (a[t], b[t]) for t in a if a[t] != b[t]}
    assert not diff, f"incremental and clean rebuild differ: {diff}"


def test_description_edit_rederives_nothing(sample_tree, tmp_path):
    tree = tmp_path / "tree"
    shutil.copytree(sample_tree, tree)
    state = tmp_path / "st"
    scan(tree, state)
    card = tree / "documents" / "document.kp20k.json"
    obj = json.loads(card.read_text(encoding="utf-8"))
    obj["description"] = obj.get("description", "") + " (edited)"
    card.write_text(json.dumps(obj), encoding="utf-8")
    out = scan(tree, state)
    assert "everything up to date" in out          # documents step
    assert "0 run(s) rederived" in out             # inferences step


def test_concurrent_start_starts_one_scan(sample_tree, tmp_path):
    code = f"""
import sys, threading
sys.path.insert(0, {str(REPO)!r})
from kpviz.config import init_settings
init_settings({str(sample_tree)!r}, state_dir={str(tmp_path / 'st')!r}, workers=2)
from kpviz import db, scanner
db.connect()
if __name__ == "__main__":
    got = []
    ts = [threading.Thread(target=lambda: got.append(scanner.start_scan())) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    scanner.wait_scan()
    print(sum(got))
"""
    script = tmp_path / "race.py"
    script.write_text(code)
    assert run(script).stdout.strip() == "1"


def test_phrase_cache_retags_on_tagger_change(sample_tree, tmp_path):
    """A new spaCy model (or a store that predates the version record) keeps
    the phrase keys and re-tags; the phrase cache is never left empty."""
    state = tmp_path / "st"
    scan(sample_tree, state)
    count = ("import duckdb,sys;c=duckdb.connect(sys.argv[1],read_only=True);"
             "print(c.execute('select count(*), count(pos) from keyphrases').fetchone())")
    before = run("-c", count, state / "kpviz.duckdb").stdout
    for edit in ("UPDATE kv SET v = json_object('code', json_extract(v, '$.code'), "
                 "'taggers', json_object('en_core_web_sm', '0.0')) "
                 "WHERE k = 'phrase_cache_version'",
                 "DELETE FROM kv WHERE k = 'phrase_cache_version'"):
        run("-c", "import duckdb,sys;duckdb.connect(sys.argv[1]).execute(sys.argv[2])",
            state / "kpviz.duckdb", edit)
        out = scan(sample_tree, state)
        assert "nothing changed" not in out        # the re-tag is real work
        assert run("-c", count, state / "kpviz.duckdb").stdout == before
