"""Cards written the way real users write them, end to end.

The schema features below all come from a real tree (the JCDL demo cards):
dotted and hyphenated file tokens, hyphenated dataset names, upper-case
similarity keys without labels, an API architecture with a batch-level
`requests` variable at rate 0, a rented architecture billed only per batch,
`set` and `str`-with-values parameters, a tiktoken encoding tiktoken does
not ship, and a size-named training split. tools/synth_tree.py builds the
documents and runs around them; the scan must resolve every card, cost every
run, flag exactly the planted problems, and find the leakage."""
from __future__ import annotations

import json

import pytest

from conftest import REPO, run

ARCHS = {
    "api": {"arch_id": "openai_api", "name": "OpenAI API", "kind": "api",
            "variables": {"input_tokens": {"unit": "token", "tokenizer": "tiktoken[o200k_base]",
                                           "level": "document"},
                          "output_tokens": {"unit": "token", "tokenizer": "tiktoken[o200k_base]",
                                            "level": "document"},
                          "time": {"unit": "s", "level": "document"},
                          "requests": {"unit": "count", "level": "batch"}},
            "rates": {"usd": {"input_tokens": 2.5e-06, "output_tokens": 1e-05, "requests": 0.0},
                      "time": {"time": 1.0}}},
    "local": {"arch_id": "workstation-3090", "name": "Local workstation", "kind": "local",
              "variables": {"time": {"unit": "s", "level": "document"}},
              "rates": {"usd": {"time": 2.25e-05}, "kwh": {"time": 9.7e-05},
                        "time": {"time": 1.0}}},
    "rented": {"arch_id": "gcp-n1s4-2xT4", "name": "GCP rented", "kind": "rented",
               "variables": {"time": {"unit": "s", "level": "batch"}},
               "rates": {"usd": {"time": 0.000172222}, "time": {"time": 1.0}}},
}
BART = {"type": "context_window", "tokenizer": "transformers[bart-base]",
        "min": 1, "max": 1024, "default": 1024}
MODELS = {
    "bart-base-kp20k": {"name": "bart-base-kp20k", "backend": "transformers",
                        "capabilities": {"extractive": True, "abstractive": True},
                        "parameters": {"total": 139420416}, "supervision": ["kp20k"],
                        "languages": ["en"],
                        "inference": {"num_beams": {"type": "int", "min": 1},
                                      "input_max_size": BART, "output_max_size": BART}},
    "biobart-small": {"name": "biobart-small", "backend": "transformers",
                      "capabilities": {"extractive": True, "abstractive": True},
                      "supervision": ["kpbiomed"], "languages": ["en"],
                      "inference": {"num_beams": {"type": "int", "min": 1},
                                    "input_max_size": BART},
                      "parameters": {"total": 139420416}},
    "gpt-4o": {"model_id": "gpt-4o", "name": "GPT-4o", "backend": "openai",
               "capabilities": {"extractive": True, "abstractive": True},
               "inference": {"prompt": {"type": "str"},
                             "temperature": {"type": "float", "min": 0.0},
                             "few_shot": {"type": "int", "min": 0},
                             "input_max_size": {"type": "context_window",
                                                "tokenizer": "tiktoken[o200k_base]",
                                                "min": 1, "max": 128000, "default": 128000}}},
    "llama-3.3-70B-Instruct": {
        "name": "Llama-3.3-70B-Instruct", "backend": "transformers",
        "capabilities": {"extractive": True, "abstractive": True, "multilingual": True},
        "parameters": {"total": 70553706496},
        "inference": {"prompt": {"type": "str"}, "temperature": {"type": "float", "min": 0.0},
                      "input_max_size": {"type": "context_window", "tokenizer": "tiktoken[llama3]",
                                         "min": 1, "max": 131072, "default": 131072}}},
    "multipartiterank": {
        "model_id": "MultipartiteRank", "name": "MultipartiteRank", "backend": "pke",
        "capabilities": {"extractive": True, "abstractive": False},
        "inference": {"pos": {"type": "set", "values": ["ADJ", "NOUN", "PROPN", "VERB"]},
                      "threshold": {"type": "float", "min": 0.0, "max": 1.0},
                      "method": {"type": "str", "values": ["single", "average", "ward"]},
                      "n_extract": {"type": "int", "min": 1}}},
}


@pytest.fixture(scope="module")
def user_tree(tmp_path_factory):
    cards = tmp_path_factory.mktemp("cards")
    for sub, items, pre in (("architectures", ARCHS, "architecture"),
                            ("models", MODELS, "model")):
        (cards / sub).mkdir()
        for tok, obj in items.items():
            (cards / sub / f"{pre}.{tok}.json").write_text(json.dumps(obj, indent=2))
    (cards / "insights").mkdir()
    pairs = []
    for i in range(60):          # kp20k training <-> kpbiomed: leakage for kp20k models
        pairs.append({"dataset_A": "kp20k", "dataset_B": "kpbiomed",
                      "doc_id_A": f"kp20k_training_{i}", "doc_id_B": str(30000000 + i),
                      "score": 0.9})
    for i in range(30):          # kp20k testing <-> kpbiomed: leakage for biobart
        pairs.append({"dataset_A": "kp20k", "dataset_B": "kpbiomed",
                      "doc_id_A": f"kp20k_testing_{i}", "doc_id_B": str(31000000 + i),
                      "score": 0.85})
    for i in range(10):
        pairs.append({"dataset_A": "kp20k", "dataset_B": "semeval-2010",
                      "doc_id_A": f"kp20k_training_{100 + i}", "doc_id_B": f"C-{i}",
                      "score": 0.95})
    (cards / "insights" / "scores.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pairs))
    out = tmp_path_factory.mktemp("utree") / "tree"
    run(REPO / "tools" / "synth_tree.py", "--cards", cards, "--out", out,
        "--test-docs", "150", "--extra-train", "60")
    state = out.parent / "state"
    run(REPO / "tools" / "scan_once.py", "--data", out, "--state", state)
    return out, state


def _db(state):
    import duckdb
    return duckdb.connect(str(state / "kpviz.duckdb"), read_only=True)


def test_every_card_resolves_and_every_run_is_costed(user_tree):
    _tree, state = user_tree
    con = _db(state)
    kv = dict(con.execute("SELECT k, v FROM kv").fetchall())
    assert json.loads(kv["card_errors"]) == []
    rows = con.execute("SELECT model, arch, arch_known, costs FROM runs").fetchall()
    assert {r[0] for r in rows} == set(MODELS)
    for model, arch, known, costs in rows:
        if arch == "n.a":
            assert not known
            continue
        assert known, (model, arch)          # declared id or file token
        usd = json.loads(costs)["usd"]
        assert usd["known"] and usd["total"] > 0, (model, arch)
    # the folder named by declared id resolved to architecture.api.json
    assert ("gpt-4o", "openai_api") in {(r[0], r[1]) for r in rows}


def test_planted_problems_are_flagged_exactly(user_tree):
    _tree, state = user_tree
    con = _db(state)
    tags = [t for (j,) in con.execute("SELECT tags FROM runs").fetchall()
            for t in json.loads(j or "[]")]
    assert any(t.startswith("illegal parameter:pos") for t in tags)       # set member
    assert any(t.startswith("illegal parameter:threshold") for t in tags)  # > max
    assert "missing:run" in tags and "missing:architecture" in tags
    assert any(t.startswith("incomplete") for t in tags)


def test_leakage_through_size_named_training_split(user_tree):
    """kpbiomed's training documents are `train_large`: they must count as
    training data on the leakage side and never be evaluated."""
    tree, state = user_tree
    code = f"""
import sys; sys.path.insert(0, {str(REPO)!r})
from kpviz.config import init_settings
init_settings({str(tree)!r}, state_dir={str(state)!r})
from kpviz import db; db.connect()
from kpviz.pages.rq.rq2 import _leak_docs
print(len(_leak_docs("kp20k", 0.8, None, {{"kpbiomed"}})),
      len(_leak_docs("kpbiomed", 0.8, None, {{"kp20k"}})),
      len(_leak_docs("semeval-2010", 0.8, None, {{"kp20k"}})))
print(db.q1("SELECT count(*) FROM gold g JOIN documents d USING (dataset, doc_id) "
            "WHERE d.split = 'train_large' AND len(d.flags) = 0")[0])
"""
    out = run("-c", code).stdout.split()
    to_biobart, to_kp20k_models, semeval = map(int, out[:3])
    assert to_biobart > 0 and to_kp20k_models > 0 and semeval > 0
    assert int(out[3]) == 0                  # training gold is never derived


def test_unknown_tiktoken_encoding_is_explained(user_tree):
    _tree, state = user_tree
    con = _db(state)
    kv = dict(con.execute("SELECT k, v FROM kv").fetchall())
    tok = json.loads(kv["tokenizers"])
    llama = tok["tiktoken[llama3]"]
    assert llama["status"] == "approx"
    assert "no 'llama3' encoding" in llama.get("why", "")


def test_exact_tokenizer_from_local_file(tmp_path):
    """The exact path, without a download: a BPE tokenizer trained here and
    declared as transformers[file:…]. Counts and gold token positions must
    equal what the tokenizer itself says."""
    tk_lib = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    corpus = ["neural keyphrase generation with transformers",
              "keyphrase extraction from scientific documents",
              "graph based ranking of candidate keyphrases"] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=200,
                                                        special_tokens=["[UNK]"]))
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    _ = tk_lib
    from kpviz import textproc as tp
    mt = tp.ModelTokenizer(f"transformers[file:{path}]", allow_network=False,
                           expect="exact")
    assert mt.exact, mt.why
    text = "Keyphrase generation with graph based ranking of scientific documents."
    enc = mt.encode_cached(text)
    ids = tok.encode(text)
    assert enc.get("n") == len(ids.ids)
    for end in (0, 9, 20, 40, len(text)):
        want = sum(1 for (a, b) in ids.offsets if b <= end)
        got, approx = mt.char_to_token(text, end, enc)
        assert not approx and abs(got - want) <= 1, (end, got, want)
