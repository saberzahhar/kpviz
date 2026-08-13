#!/usr/bin/env python3
"""Build the sample_data/ tree for KPViz demos and tests.

Uses the *real* cards and real files where available (semeval2010 + taln
subsets, the bart-base-kp20k beams=4 run on kp20k), and synthesises the
rest deterministically (seed 7): kp20k-mini / kpbiomed-mini / kptimes-mini
documents consistent with the real inference ids, extra model cards in the
same schema, multiple runs per model with controlled quality/cost knobs,
and a scores.jsonl with labelled leakage pairs.

Usage: python tools/make_sample_data.py [--src sample_src] [--out sample_data]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

RNG = random.Random(7)

# ---------------------------------------------------------------------------
FILLER_CS = [
    "In this paper we investigate the problem in depth and report empirical findings",
    "Extensive experiments demonstrate the effectiveness of the proposed approach",
    "We formalise the task and derive an efficient algorithm with provable guarantees",
    "The method is evaluated on several public benchmarks against strong baselines",
    "Our analysis highlights practical trade-offs between accuracy and efficiency",
    "Results show consistent improvements over the state of the art",
    "We further conduct an ablation study to quantify each component's contribution",
    "A detailed error analysis reveals systematic failure modes of prior methods",
]
FILLER_BIO = [
    "Patients were enrolled in a multicentre prospective cohort over two years",
    "Expression levels were quantified using standard laboratory assays",
    "Statistical significance was assessed with corrected non-parametric tests",
    "The findings suggest a potential pathway for targeted intervention",
    "A systematic review of the literature was conducted following PRISMA",
    "Clinical outcomes were followed up at six and twelve months",
]
FILLER_NEWS = [
    "Officials announced the decision at a press conference on Monday",
    "Analysts said the move could reshape the market in the coming months",
    "The government has faced mounting pressure over the policy",
    "Local residents expressed mixed reactions to the announcement",
    "The company declined to comment on the ongoing negotiations",
    "Observers noted the timing coincided with international talks",
]
ABSENT_POOL_CS = ["computational complexity", "machine learning", "distributed systems",
                  "information retrieval", "neural networks", "optimization",
                  "software engineering", "formal verification"]
ABSENT_POOL_BIO = ["public health", "clinical trial", "gene expression",
                   "risk factors", "immune response", "epidemiology"]
ABSENT_POOL_NEWS = ["foreign policy", "economic outlook", "trade agreement",
                    "climate policy", "energy markets", "public opinion"]

BIO_TOPICS = [
    ["insulin resistance", "type 2 diabetes", "metabolic syndrome", "glucose tolerance", "obesity"],
    ["tumor microenvironment", "immunotherapy", "checkpoint inhibitors", "t cells", "melanoma"],
    ["gut microbiome", "probiotics", "inflammatory bowel disease", "short chain fatty acids", "dysbiosis"],
    ["alzheimer disease", "amyloid beta", "cognitive decline", "neuroinflammation", "tau protein"],
    ["antibiotic resistance", "gram negative bacteria", "efflux pumps", "horizontal gene transfer", "biofilm"],
    ["cardiovascular disease", "hypertension", "statins", "lipid profile", "atherosclerosis"],
    ["breast cancer", "her2", "targeted therapy", "biomarkers", "chemotherapy"],
    ["chronic kidney disease", "dialysis", "glomerular filtration rate", "proteinuria", "renal fibrosis"],
]
NEWS_TOPICS = [
    ["interest rates", "central bank", "inflation", "monetary policy", "bond yields"],
    ["general election", "opposition party", "coalition talks", "voter turnout", "exit polls"],
    ["semiconductor industry", "chip manufacturing", "export controls", "supply chain", "foundries"],
    ["olympic games", "host city", "athletics", "doping controls", "opening ceremony"],
    ["renewable energy", "solar power", "grid capacity", "carbon emissions", "wind farms"],
    ["housing market", "mortgage rates", "property prices", "first-time buyers", "construction"],
]


def sentence_with(phrase: str, domain: str) -> str:
    pool = {"cs": FILLER_CS, "bio": FILLER_BIO, "news": FILLER_NEWS}[domain]
    templates = [
        f"This work focuses on {phrase} as a central concern.",
        f"We study {phrase} and its implications in detail.",
        f"The role of {phrase} has attracted considerable attention.",
        f"Particular emphasis is placed on {phrase} throughout.",
        f"Recent progress on {phrase} motivates our study.",
    ]
    return RNG.choice(templates) + " " + RNG.choice(pool) + "."


def make_doc(doc_id: str, split: str, phrases: list[str], domain: str,
             absent_pool: list[str], sections=("title", "abstract")) -> dict:
    phrases = [p.lower() for p in phrases if p]
    present = [p for p in phrases if RNG.random() < 0.78][:8] or phrases[:3]
    absent = RNG.sample(absent_pool, k=min(2, len(absent_pool)))
    title = f"On {present[0]}" + (f" and {present[1]}" if len(present) > 1 else "")
    body_sents = [sentence_with(p, domain) for p in present[1:]]
    RNG.shuffle(body_sents)
    abstract = " ".join(body_sents) or sentence_with(present[0], domain)
    hi = max(3, min(6, len(present)))
    gold = present[: RNG.randint(min(3, len(present)), hi)] + \
        RNG.sample(absent, k=RNG.randint(1, 2))
    RNG.shuffle(gold)
    return {
        "_id": doc_id,
        "metadata": {"split": split},
        "sections": [
            {"field": sections[0], "language": ["en"], "content": title},
            {"field": sections[1], "language": ["en"], "content": abstract},
        ],
        "annotations": [
            {"annotator": "author" if domain != "news" else "editor",
             "language": ["en"], "keyphrases": gold}
        ],
        "_gold": gold, "_present": present,   # stripped before writing
    }


# ---------------------------------------------------------------------------
# Model behaviour knobs: (recall of gold, precision-ish, rank noise)
# ---------------------------------------------------------------------------
def predict(doc: dict, recall: float, extra: int, noise: float,
            distractors: list[str], casing: str = "lower") -> list[str]:
    gold, present = doc["_gold"], doc["_present"]
    hits = [g for g in gold if RNG.random() < recall]
    others = [p for p in present if p not in gold and RNG.random() < 0.35]
    junk = RNG.sample(distractors, k=min(extra, len(distractors)))
    preds = hits + others + junk
    # rank noise: hits mostly first
    preds.sort(key=lambda p: (p not in hits) + RNG.random() * noise)
    if casing == "title":
        preds = [p.title() if RNG.random() < 0.5 else p for p in preds]
    seen, out = set(), []
    for p in preds:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out[: RNG.randint(5, 9)]


def token_len(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k not in ("_gold", "_present")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def emit_run(out: Path, ds: str, model: str, arch: str, run_id: str,
             params: dict, docs: list[dict], pred_fn, cost_mode: str,
             t0: datetime, batch_size: int = 64, coverage: float = 1.0,
             speed: float = 1.0):
    """cost_mode: rented (batch time) | local (doc time) | api (tokens) | none"""
    rd = out / "inferences" / ds / model / arch / run_id
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / f"run_{run_id}.json", {"parameters": params})
    kept = [d for d in docs if RNG.random() < coverage]
    t = t0
    for bi in range(0, max(1, (len(kept) + batch_size - 1) // batch_size)):
        chunk = kept[bi * batch_size:(bi + 1) * batch_size]
        lines, doc_times, in_toks, out_toks = [], [], [], []
        for d in chunk:
            preds = pred_fn(d)
            row = {"_id": d["_id"], "inferences": preds}
            dt = RNG.uniform(0.05, 0.12) / speed
            itok = token_len(" ".join(s["content"] for s in d["sections"]))
            otok = token_len(", ".join(preds)) + 8
            if cost_mode == "local":
                row["costs"] = {"time": round(dt, 6)}
            elif cost_mode == "api":
                row["costs"] = {"input_tokens": itok, "output_tokens": otok}
            lines.append(row)
            doc_times.append(dt)
            in_toks.append(itok)
            out_toks.append(otok)
        wall = sum(doc_times) * (1.15 if cost_mode != "api" else 4.0)
        t1 = t + timedelta(seconds=wall)
        meta = {"batch_idx": bi,
                "start_timestamp": t.strftime("%Y/%m/%d %H:%M:%S"),
                "end_timestamp": t1.strftime("%Y/%m/%d %H:%M:%S")}
        if cost_mode == "rented":
            meta["costs"] = {"time": round(wall, 4)}
        elif cost_mode == "local":
            meta["costs"] = {"time": round(wall, 4)}
        elif cost_mode == "api":
            meta["costs"] = {"input_tokens": sum(in_toks),
                             "output_tokens": sum(out_toks),
                             "requests": len(chunk)}
        write_json(rd / f"batch_{bi:05d}.json", meta)
        write_jsonl(rd / f"batch_{bi:05d}.jsonl", lines)
        t = t1 + timedelta(seconds=30)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sample_src")
    ap.add_argument("--out", default="sample_data")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "documents").mkdir(parents=True)
    (out / "architectures").mkdir()
    (out / "models").mkdir()
    (out / "insights").mkdir()

    # ---- cards (real ones straight through) ------------------------------
    for f in ["document.semeval2010.json", "document.talnarchives.json",
              "document.kp20k.json", "document.kpbiomed.json",
              "document.kptimes.json"]:
        shutil.copy(src / f, out / "documents" / f)
    for f in ["architecture.api.json", "architecture.local.json",
              "architecture.rented.json"]:
        shutil.copy(src / f, out / "architectures" / f)
    for f in ["model.multipartiterank.json", "model.bartbasekp20k.json",
              "model.bartkptimes.json"]:
        shutil.copy(src / f, out / "models" / f)

    # extra model cards in the same schema (the ones not uploaded)
    write_json(out / "models" / "model.gpt4o.json", {
        "model_id": "gpt-4o-2024-08-06", "name": "gpt-4o",
        "family": [["Neural model", "LLM", "Instruction-tuned"],
                   ["Proprietary", "OpenAI"]],
        "backend": "openai", "capabilities": {"extractive": True, "abstractive": True},
        "languages": ["en", "fr", "de", "es"],
        "domains": [],
        "references": {"url": "https://platform.openai.com/docs/models/gpt-4o"},
        "inference": {
            "prompt": {"type": "str"},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
            "top_p": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
            "frequency_penalty": {"type": "float", "min": -2.0, "max": 2.0, "default": 0.0},
            "presence_penalty": {"type": "float", "min": -2.0, "max": 2.0, "default": 0.0},
            "n_repeats": {"type": "int", "min": 1, "default": 1},
            "few_shot": {"type": "int", "min": 0, "default": 0},
            "input_max_size": {"type": "context_window",
                               "tokenizer": "tiktoken[o200k_base]",
                               "min": 1, "max": 128000, "default": 128000},
            "output_max_size": {"type": "context_window",
                                "tokenizer": "tiktoken[o200k_base]",
                                "min": 1, "max": 16384, "default": 4096}}})
    write_json(out / "models" / "model.llama3.370BInstruct.json", {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct", "name": "Llama-3.3-70B-Instruct",
        "family": [["Neural model", "LLM", "Instruction-tuned"],
                   ["Open weights", "Llama"]],
        "backend": "vllm", "capabilities": {"extractive": True, "abstractive": True},
        "parameters": {"total": 70600000000},
        "languages": ["en", "fr", "de", "es", "it", "pt"],
        "references": {"huggingface_id": "meta-llama/Llama-3.3-70B-Instruct",
                       "url": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"},
        "inference": {
            "prompt": {"type": "str"},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
            "top_p": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
            "n_repeats": {"type": "int", "min": 1, "default": 1},
            "few_shot": {"type": "int", "min": 0, "default": 0},
            "input_max_size": {"type": "context_window",
                               "tokenizer": "transformers[llama-3.3]",
                               "min": 1, "max": 131072, "default": 131072},
            "output_max_size": {"type": "context_window",
                                "tokenizer": "transformers[llama-3.3]",
                                "min": 1, "max": 131072, "default": 256}}})
    write_json(out / "models" / "model.bartlargekp20k.json", {
        "model_id": "https://huggingface.co/taln-ls2n/bart-large-kp20k",
        "name": "bart-large-kp20k",
        "family": [["Neural model", "Paradigms", "One2Seq"],
                   ["Pre-trained models", "BART"]],
        "backend": "transformers",
        "capabilities": {"extractive": True, "abstractive": True},
        "parameters": {"total": 406290432}, "supervision": ["kp20k"],
        "domains": [{"domain": "Academic", "sub-domain": "computer-sciences"}],
        "languages": ["en"],
        "references": {"url": "https://huggingface.co/taln-ls2n/bart-large-kp20k",
                       "parent_model": "https://huggingface.co/facebook/bart-large"},
        "inference": {
            "num_beams": {"type": "int", "min": 1},
            "input_max_size": {"type": "context_window",
                               "tokenizer": "transformers[bart-base]",
                               "min": 1, "max": 1024, "default": 1024},
            "output_max_size": {"type": "context_window",
                                "tokenizer": "transformers[bart-base]",
                                "min": 1, "max": 1024, "default": 1024}}})

    # ---- real document subsets -------------------------------------------
    def subset_jsonl(name: str, keep_train: int, keep_test: int):
        rows, ktr, kte = [], 0, 0
        for line in open(src / f"document.{name}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            sp = (d.get("metadata") or {}).get("split")
            if sp == "training" and ktr < keep_train:
                rows.append(line.rstrip("\n"))
                ktr += 1
            elif sp != "training" and kte < keep_test:
                rows.append(line.rstrip("\n"))
                kte += 1
            if ktr >= keep_train and kte >= keep_test:
                break
        with open(out / "documents" / f"document.{name}.jsonl", "w",
                  encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        return ktr, kte

    subset_jsonl("semeval2010", keep_train=10, keep_test=30)
    subset_jsonl("talnarchives", keep_train=0, keep_test=160)

    # ---- kp20k-mini from the REAL bart batch file -------------------------
    real_preds = [json.loads(l) for l in open(src / "kp20k_bart_batch_00000.jsonl",
                                              encoding="utf-8")]
    kp20k_docs = []
    for rp in real_preds:
        d = make_doc(rp["_id"], "testing", rp["inferences"], "cs", ABSENT_POOL_CS)
        kp20k_docs.append(d)
    kp20k_train = [make_doc(f"kp20k_training_{i}", "training",
                            RNG.sample(sum((t["inferences"] for t in
                                            RNG.sample(real_preds, 3)), []), 6),
                            "cs", ABSENT_POOL_CS) for i in range(60)]
    write_jsonl(out / "documents" / "document.kp20k.jsonl",
                kp20k_train + kp20k_docs)

    # ---- kpbiomed-mini & kptimes-mini --------------------------------------
    bio_docs, bio_train = [], []
    for i in range(140):
        topic = BIO_TOPICS[i % len(BIO_TOPICS)]
        phr = RNG.sample(topic, 4) + [RNG.choice(ABSENT_POOL_BIO)]
        bio_docs.append(make_doc(f"{31548000 + i}", "testing", phr, "bio",
                                 ABSENT_POOL_BIO))
    # leakage: 12 kpbiomed *testing* docs are near-copies of kp20k *training* docs
    leak_pairs = []
    for i in range(12):
        srcdoc = kp20k_train[i]
        clone = json.loads(json.dumps(srcdoc))
        clone["_id"] = f"{31549900 + i}"
        clone["metadata"]["split"] = "testing"
        clone["_gold"] = srcdoc["_gold"]
        clone["_present"] = srcdoc["_present"]
        bio_docs[i] = clone
        leak_pairs.append({"dataset_A": "kp20k", "dataset_B": "kpbiomed",
                           "doc_id_A": srcdoc["_id"], "doc_id_B": clone["_id"],
                           "score": round(RNG.uniform(0.88, 0.99), 5),
                           "label": "near-duplicate"})
    for i in range(50):
        topic = BIO_TOPICS[(i * 3) % len(BIO_TOPICS)]
        bio_train.append(make_doc(f"{31400000 + i}", "training",
                                  RNG.sample(topic, 4), "bio", ABSENT_POOL_BIO))
    write_jsonl(out / "documents" / "document.kpbiomed.jsonl", bio_train + bio_docs)

    news_docs = []
    for i in range(120):
        topic = NEWS_TOPICS[i % len(NEWS_TOPICS)]
        news_docs.append(make_doc(f"jt_{2026000 + i}", "testing",
                                  RNG.sample(topic, 4) + [RNG.choice(ABSENT_POOL_NEWS)],
                                  "news", ABSENT_POOL_NEWS))
    news_train = [make_doc(f"nyt_{i}", "training",
                           RNG.sample(NEWS_TOPICS[i % len(NEWS_TOPICS)], 4),
                           "news", ABSENT_POOL_NEWS) for i in range(40)]
    write_jsonl(out / "documents" / "document.kptimes.jsonl", news_train + news_docs)

    # intra-dataset train/test leakage inside kp20k + unlabeled pairs
    for i in range(6):
        leak_pairs.append({"dataset_A": "kp20k", "dataset_B": "kp20k",
                           "doc_id_A": kp20k_train[20 + i]["_id"],
                           "doc_id_B": kp20k_docs[i]["_id"],
                           "score": round(RNG.uniform(0.82, 0.97), 5),
                           "label": "train-test"})
    for i in range(25):
        a = RNG.choice(kp20k_docs)
        b = RNG.choice(bio_docs)
        leak_pairs.append({"dataset_A": "kp20k", "dataset_B": "kpbiomed",
                           "doc_id_A": a["_id"], "doc_id_B": b["_id"],
                           "score": round(RNG.uniform(0.55, 0.85), 5)})
    with open(out / "insights" / "scores.jsonl", "w", encoding="utf-8") as f:
        for p in leak_pairs:
            f.write(json.dumps(p) + "\n")

    # ---- runs ---------------------------------------------------------------
    DIST_CS = ABSENT_POOL_CS + ["experimental results", "case study", "framework"]
    DIST_BIO = ABSENT_POOL_BIO + ["patients", "treatment", "meta analysis"]
    DIST_NEWS = ABSENT_POOL_NEWS + ["government", "spokesperson", "quarterly report"]
    t0 = datetime(2026, 7, 30, 11, 13, 10)

    def bart_fn(q):        # quality by beams
        return lambda d: predict(d, recall=q, extra=2, noise=0.4, distractors=DIST_CS)

    ds_sets = {"kp20k": (kp20k_docs, DIST_CS), "kpbiomed": (bio_docs, DIST_BIO),
               "kptimes": (news_docs, DIST_NEWS)}

    # ids of kpbiomed testing docs that are near-copies of kp20k training
    # docs — kp20k-supervised models "remember" these (leakage inflation)
    clone_ids = {p["doc_id_B"] for p in leak_pairs
                 if p.get("label") == "near-duplicate"}

    # bart-base-kp20k: strong in-domain, weaker out-of-domain — except on
    # the leaked clones, which it aces
    bart_params = {"04b902a7dc7b": {"num_beams": 4, "input_max_size": 512, "output_max_size": 128},
                   "866b5b70aefb": {"num_beams": 1, "input_max_size": 512, "output_max_size": 128},
                   "11f084544590": {"num_beams": 10, "input_max_size": 512, "output_max_size": 128}}
    bart_quality = {"04b902a7dc7b": 0.62, "866b5b70aefb": 0.48, "11f084544590": 0.66}

    def bart_ood(q, dist):
        def fn(d):
            recall = 0.9 if d["_id"] in clone_ids else q
            return predict(d, recall=recall, extra=3, noise=0.6,
                           distractors=dist)
        return fn

    for run_id, params in bart_params.items():
        for ds, (docs, dist) in ds_sets.items():
            q = bart_quality[run_id] * (1.0 if ds == "kp20k" else 0.55)
            emit_run(out, ds, "bartbasekp20k", "gcp-n1s4-2xT4", run_id, params,
                     docs, bart_fn(q) if ds == "kp20k" else bart_ood(q, dist),
                     "rented", t0, speed=1.0 / (0.4 + 0.15 * params["num_beams"]))
    # overwrite kp20k beams=4 with the REAL files
    rd = out / "inferences" / "kp20k" / "bartbasekp20k" / "gcp-n1s4-2xT4" / "04b902a7dc7b"
    shutil.copy(src / "run_04b902a7dc7b.json", rd / "run_04b902a7dc7b.json")
    shutil.copy(src / "kp20k_bart_batch_00000.json", rd / "batch_00000.json")
    shutil.copy(src / "kp20k_bart_batch_00000.jsonl", rd / "batch_00000.jsonl")
    for extra in rd.glob("batch_0000[1-9].*"):
        extra.unlink()

    # bart-large-kp20k: better, slower (only on kp20k + kpbiomed)
    for ds in ("kp20k", "kpbiomed"):
        docs, dist = ds_sets[ds]
        q = 0.72 if ds == "kp20k" else 0.6 * 0.55 + 0.3
        emit_run(out, ds, "bartlargekp20k", "gcp-n1s4-2xT4", "3fe2a1c09d44",
                 {"num_beams": 4, "input_max_size": 512, "output_max_size": 128},
                 docs, lambda d, _q=q, _dist=dist: predict(
                     d, recall=(0.92 if d["_id"] in clone_ids else _q),
                     extra=2, noise=0.35, distractors=_dist),
                 "rented", t0 + timedelta(hours=2), speed=0.45)

    # MultipartiteRank: cheap, moderate, runs everywhere incl. french
    mp_params = {"parameters": {"pos": ["ADJ", "NOUN", "PROPN"], "alpha": 1.1,
                                "threshold": 0.74, "method": "average",
                                "n_extract": 50}}
    for ds, (docs, dist) in ds_sets.items():
        emit_run(out, ds, "multipartiterank", "workstation-3090", "mprank-def",
                 mp_params["parameters"], docs,
                 lambda d, _dist=dist: predict(d, recall=0.34, extra=2, noise=0.8,
                                               distractors=_dist),
                 "local", t0 + timedelta(hours=1), speed=8.0)
    # an alternative clustering method for the hyperparameter RQ
    for ds in ("kp20k", "kpbiomed"):
        docs, dist = ds_sets[ds]
        emit_run(out, ds, "multipartiterank", "workstation-3090", "mprank-ward",
                 dict(mp_params["parameters"], method="ward", threshold=0.62),
                 docs, lambda d, _dist=dist: predict(d, recall=0.30, extra=2,
                                                     noise=0.85, distractors=_dist),
                 "local", t0 + timedelta(hours=1, minutes=30), speed=7.0)
    # a run with ILLEGAL parameters (validation demo)
    emit_run(out, "kp20k", "multipartiterank", "workstation-3090", "mprank-bad",
             {"pos": ["NOUN", "VERBZ"], "alpha": -0.5, "threshold": 1.4,
              "method": "centroidal", "n_extract": 0, "mystery_knob": 3},
             kp20k_docs[:100],
             lambda d: predict(d, recall=0.22, extra=3, noise=1.0,
                               distractors=DIST_CS),
             "local", t0 + timedelta(hours=3), speed=8.0)

    # gpt-4o via API: strong, expensive; INCOMPLETE on kpbiomed (~55 %)
    gpt_prompt = ("Predict keyphrases in a comma-separated list from the following "
                  "document. ONLY OUTPUT THE COMMA-SEPARATED LIST NOTHING ELSE. "
                  "From most important to least important:\nInput: {document}\nOutput:")
    gpt_params = {"prompt": gpt_prompt, "top_p": 1.0, "output_max_size": 256,
                  "n_repeats": 3, "few_shot": 0, "temperature": 1.0,
                  "frequency_penalty": 0.0, "presence_penalty": 0.0}
    for ds, cov in (("kp20k", 1.0), ("kpbiomed", 0.55), ("kptimes", 1.0)):
        docs, dist = ds_sets[ds]
        emit_run(out, ds, "gpt4o", "openai_api", "a1f2e3d4c5b6", gpt_params,
                 docs, lambda d, _dist=dist: predict(d, recall=0.55, extra=2,
                                                     noise=0.5, distractors=_dist,
                                                     casing="title"),
                 "api", t0 + timedelta(days=5, hours=2), coverage=cov, speed=0.9)

    # Llama on an undisclosed architecture ("n.a") -> no cost model
    ll_params = {"prompt": gpt_prompt, "temperature": 0.2, "top_p": 0.9}
    for ds in ("kp20k", "kpbiomed", "kptimes"):
        docs, dist = ds_sets[ds]
        emit_run(out, ds, "llama3.370BInstruct", "n.a", "ab1a2c4366c3", ll_params,
                 docs, lambda d, _dist=dist: predict(d, recall=0.48, extra=2,
                                                     noise=0.55, distractors=_dist),
                 "none", t0 + timedelta(days=6), speed=0.5)

    # semeval2010 (long docs -> truncation RQ) + talnarchives (french)
    # stem-aware window membership (same matching the evaluator uses)
    sys_path_hack = str(Path(__file__).resolve().parent.parent)
    if sys_path_hack not in sys.path:
        sys.path.insert(0, sys_path_hack)
    from kpviz.textproc import stem_tokens, tokenize

    def in_window(gold_kp: str, hay_stems: list[str]) -> bool:
        for variant in gold_kp.split("+"):
            needle = stem_tokens(tokenize(variant), "en")
            m = len(needle)
            if m and any(hay_stems[i:i + m] == needle
                         for i in range(len(hay_stems) - m + 1)):
                return True
        return False

    sem_docs, taln_docs = [], []
    for line in open(out / "documents" / "document.semeval2010.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if (d.get("metadata") or {}).get("split") != "training":
            gold = [k for a in (d.get("annotations") or []) for k in a["keyphrases"]]
            text = "\n\n".join(s.get("content", "") for s in d["sections"])
            hay = stem_tokens(tokenize(text)[:400], "en")   # ≈ 512 bart tokens
            sem_docs.append({"_id": d["_id"], "sections": d["sections"],
                             "_gold": gold[:10],
                             "_present": [g.split("+")[0] for g in gold[:10]],
                             "_zone": [g.split("+")[0] for g in gold[:10]
                                       if in_window(g, hay)]})
    for line in open(out / "documents" / "document.talnarchives.jsonl", encoding="utf-8"):
        d = json.loads(line)
        gold = [k for a in d["annotations"] for k in a["keyphrases"]]
        taln_docs.append({"_id": d["_id"], "sections": d["sections"],
                          "_gold": gold[:10],
                          "_present": [g.split("+")[0] for g in gold[:10]]})

    def bart_truncated(d):
        """bart only sees the first ~512 tokens: strong recall on gold inside
        that zone, near-zero beyond it — makes the RQ3 rise visible."""
        zone = set(d.get("_zone") or [])
        hits = [g.split("+")[0] for g in d["_gold"]
                if RNG.random() < (0.75 if g.split("+")[0] in zone else 0.05)]
        junk = RNG.sample(DIST_CS, k=min(5, len(DIST_CS)))
        preds = hits + junk
        preds.sort(key=lambda p: (p not in hits) + RNG.random() * 0.5)
        return preds[: RNG.randint(9, 12)]

    emit_run(out, "semeval2010", "bartbasekp20k", "gcp-n1s4-2xT4", "04b902a7dc7b",
             bart_params["04b902a7dc7b"], sem_docs, bart_truncated,
             "rented", t0 + timedelta(days=1), speed=0.3)
    emit_run(out, "semeval2010", "gpt4o", "openai_api", "a1f2e3d4c5b6", gpt_params,
             sem_docs, lambda d: predict(d, recall=0.5, extra=3, noise=0.5,
                                         distractors=DIST_CS, casing="title"),
             "api", t0 + timedelta(days=5, hours=6), speed=0.4)
    emit_run(out, "semeval2010", "multipartiterank", "workstation-3090", "mprank-def",
             mp_params["parameters"], sem_docs,
             lambda d: predict(d, recall=0.3, extra=3, noise=0.9,
                               distractors=DIST_CS),
             "local", t0 + timedelta(days=1, hours=2), speed=2.0)
    emit_run(out, "talnarchives", "multipartiterank", "workstation-3090", "mprank-def",
             mp_params["parameters"], taln_docs,
             lambda d: predict(d, recall=0.3, extra=3, noise=0.8,
                               distractors=["analyse", "corpus", "évaluation",
                                            "méthode", "apprentissage"]),
             "local", t0 + timedelta(days=1, hours=4), speed=6.0)
    emit_run(out, "talnarchives", "bartbasekp20k", "gcp-n1s4-2xT4", "04b902a7dc7b",
             bart_params["04b902a7dc7b"], taln_docs,
             lambda d: predict(d, recall=0.12, extra=4, noise=1.2,
                               distractors=DIST_CS),
             "rented", t0 + timedelta(days=1, hours=6), speed=1.0)

    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"sample_data ready: {n_files} files, {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
