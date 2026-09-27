#!/usr/bin/env python3
"""Scale the KPViz sample tree to demo-like volumes (for performance probing):
three 'mini' collections become N test documents each and every model gets
more hyper-parameter variants, giving ~70 runs and ~1M prediction lines.

    python tools/make_sample_data.py                       # builds sample_data/ first
    python tools/make_scaled_data.py . sample_src scaled_data 20000

N_TEST=20000 gives the performance baseline of docs/PERFORMANCE_PLAN.md
(5 datasets, 66 runs, 75,190 documents, 1.32 M prediction lines, ~300 MB).
Originally written for the external code review of KPViz (kpviz_review_tools).
"""
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

repo, src, out, n_test = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("msd", repo / "tools" / "make_sample_data.py")
msd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msd)
R = msd.RNG

if out.exists():
    shutil.rmtree(out)
for d in ("documents", "architectures", "models", "insights"):
    (out / d).mkdir(parents=True)
# cards: reuse the small tree's cards verbatim
small = repo / "sample_data"
for sub, pat in (("documents", "*.json"), ("architectures", "*.json"), ("models", "*.json")):
    for f in (small / sub).glob(pat):
        shutil.copy(f, out / sub / f.name)
for name in ("semeval2010", "talnarchives"):
    shutil.copy(small / "documents" / f"document.{name}.jsonl", out / "documents")

CS_PHR = [" ".join(R.sample(msd.ABSENT_POOL_CS + ["graph", "network", "learning", "model",
                                                  "query", "index", "protocol", "security",
                                                  "cache", "compiler", "parallel"], R.choice([1, 2, 2, 3])))
          for _ in range(6000)]
sets = {}
for ds, dom, pool, topics in (("kp20k", "cs", msd.ABSENT_POOL_CS, None),
                              ("kpbiomed", "bio", msd.ABSENT_POOL_BIO, msd.BIO_TOPICS),
                              ("kptimes", "news", msd.ABSENT_POOL_NEWS, msd.NEWS_TOPICS)):
    docs = []
    for i in range(n_test):
        phr = (R.sample(CS_PHR, 6) if topics is None
               else R.sample(topics[i % len(topics)], 4) + R.sample(CS_PHR, 2))
        docs.append(msd.make_doc(f"{ds}_testing_{i}", "testing", phr, dom, pool))
    train = []
    for i in range(n_test // 4):
        phr = R.sample(CS_PHR, 6)
        train.append(msd.make_doc(f"{ds}_training_{i}", "training", phr, dom, pool))
    msd.write_jsonl(out / "documents" / f"document.{ds}.jsonl", train + docs)
    sets[ds] = (docs, pool + ["experimental results", "case study", "framework"])
    print("wrote", ds, len(docs), "test +", len(train), "train")

pairs = []
for i in range(2000):
    pairs.append({"dataset_a": "kp20k", "doc_id_a": f"kp20k_training_{i % (n_test // 4)}",
                  "dataset_b": "kpbiomed", "doc_id_b": f"kpbiomed_testing_{i}",
                  "score": round(R.uniform(0.5, 0.99), 4),
                  "label": "near-duplicate" if i % 7 == 0 else None})
with open(out / "insights" / "scores.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

t0 = datetime(2026, 7, 30, 11, 13, 10)
n_runs = 0
fn = lambda q, dist: (lambda d: msd.predict(d, recall=q, extra=3, noise=0.5, distractors=dist))
for ds, (docs, dist) in sets.items():
    for beams in (1, 2, 4, 6, 8, 10):
        for inp in (256, 512):
            rid = f"bb{beams:02d}{inp}"
            msd.emit_run(out, ds, "bartbasekp20k", "gcp-n1s4-2xT4", rid,
                         {"num_beams": beams, "input_max_size": inp, "output_max_size": 128},
                         docs, fn(0.35 + 0.03 * beams, dist), "rented", t0, batch_size=512,
                         speed=1.0 / (0.4 + 0.15 * beams))
            n_runs += 1
    for beams in (1, 4, 10):
        msd.emit_run(out, ds, "bartlargekp20k", "gcp-n1s4-2xT4", f"bl{beams:02d}",
                     {"num_beams": beams, "input_max_size": 512, "output_max_size": 128},
                     docs, fn(0.45 + 0.02 * beams, dist), "rented", t0, batch_size=512, speed=0.5)
        n_runs += 1
    for method, thr in (("average", 0.74), ("ward", 0.62), ("single", 0.5)):
        msd.emit_run(out, ds, "multipartiterank", "workstation-3090", f"mp-{method}",
                     {"pos": ["ADJ", "NOUN", "PROPN"], "alpha": 1.1, "threshold": thr,
                      "method": method, "n_extract": 50},
                     docs, fn(0.3, dist), "local", t0, batch_size=512, speed=8.0)
        n_runs += 1
    for rep in range(3):
        msd.emit_run(out, ds, "gpt4o", "openai_api", f"gpt-t{rep}",
                     {"prompt": "…", "temperature": [0.0, 0.7, 1.0][rep], "top_p": 1.0},
                     docs, fn(0.55, dist), "api", t0 + timedelta(days=2), batch_size=512)
        n_runs += 1
    msd.emit_run(out, ds, "llama3.370BInstruct", "n.a", "llama-1",
                 {"prompt": "…", "temperature": 0.2, "top_p": 0.9},
                 docs, fn(0.5, dist), "none", t0 + timedelta(days=3), batch_size=512)
    n_runs += 1
n_files = sum(1 for _ in out.rglob("*") if _.is_file())
size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
print(f"scaled tree: {n_runs} runs, {n_files} files, {size / 1e6:.1f} MB")
