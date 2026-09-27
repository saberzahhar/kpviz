#!/usr/bin/env python3
"""Synthesise the `sample_src/` inputs that tools/make_sample_data.py expects.

The real corpora live on Zenodo (14.5 GB); this script writes a small,
deterministic stand-in so a fresh clone can build `sample_data/` and run every
page. Cards follow the README / paper Fig. 2 contract *as users write it*:
one architecture card carries `//` comments (the README's own JSONC style) and
one collection declares annotation languages with the card-schema spelling
`languages`, so both spellings stay exercised.

Originally written for the external code review of KPViz (kpviz_review_tools).

    python tools/make_sample_src.py OUT_DIR [--kp20k N]
"""
import argparse
import json
import random
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("out")
_ap.add_argument("--kp20k", type=int, default=500)
_args = _ap.parse_args()
R = random.Random(11)
out = Path(_args.out)
N_KP20K = _args.kp20k
out.mkdir(parents=True, exist_ok=True)


def wj(name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def wl(name, rows):
    with open(out / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- cards
def ds_card(desc, dom, sub, sections, anns, extra_meta=None):
    meta = {"split": {"type": "split"}}
    meta.update(extra_meta or {})
    return {"description": desc, "domain": dom, "sub-domain": sub,
            "metadata": meta,
            "document": {k: {"type": [k], "modality": "text", "languages": v}
                         for k, v in sections.items()},
            "annotations": {k: {"type": k, "expertise": k, "languages": v}
                            for k, v in anns.items()}}


wj("document.kp20k.json", ds_card("KP20k: computer-science abstracts.", "Academic",
                                  "computer-sciences",
                                  {"title": ["en"], "abstract": ["en"]}, {"author": ["en"]}))
wj("document.kpbiomed.json", ds_card("KPBioMed: PubMed abstracts.", "Academic",
                                     "biomedical", {"title": ["en"], "abstract": ["en"]},
                                     {"author": ["en"]}))
wj("document.kptimes.json", ds_card("KPTimes: news articles.", "News", "general",
                                    {"title": ["en"], "abstract": ["en"]},
                                    {"editor": ["en"]}))
wj("document.semeval2010.json", {
    "description": "SemEval-2010 scholarly documents (title + full text).",
    "domain": "Academic", "sub-domain": "computer-sciences",
    "metadata": {"split": {"type": "split"},
                 "category": {"type": "classification", "taxonomy": "ACM CCS (1998)", "depth": 3}},
    "document": {"title+full-text": {"type": ["title", "body"], "modality": "text",
                                     "languages": ["en"]}},
    "annotations": {"author": {"type": "author", "expertise": "author", "languages": ["en"]},
                    "reader": {"type": "reader", "expertise": "student", "languages": ["en"],
                               "post_annotation": {"type": "expert"}}}})
wj("document.talnarchives.json", ds_card(
    "TALN-archives: French NLP papers (title + abstract, FR and EN).", "Academic",
    "computational-linguistics",
    {"title": ["fr"], "abstract": ["fr"], "title_en": ["en"], "abstract_en": ["en"]},
    {"author": ["fr"]}))

wj("architecture.rented.json", {
    "arch_id": "gcp-n1s4-2xT4", "name": "GCP n1-standard-4 + 2x T4", "kind": "rented",
    "hardware": {"cpu": {"vendor": "Intel", "model": "Xeon", "cores": 4, "ram_gb": 15},
                 "gpu": {"vendor": "NVIDIA", "model": "T4", "count": 2, "vram_gb": 16}},
    "software": {"os": "Ubuntu 22.04", "python": "3.10",
                 "packages": {"torch": "2.3", "transformers": "4.41"}},
    "variables": {"time": {"unit": "s", "level": "batch", "description": "batch wall-clock"}},
    "rates": {"usd": {"time": 1.72e-4}, "time": {"time": 1.0}}})
wj("architecture.local.json", {
    "arch_id": "workstation-3090", "name": "Workstation RTX 3090", "kind": "local",
    "hardware": {"cpu": {"vendor": "AMD", "model": "Ryzen 9", "cores": 16, "ram_gb": 64},
                 "gpu": {"vendor": "NVIDIA", "model": "RTX 3090", "count": 1, "vram_gb": 24}},
    "variables": {"time": {"unit": "s", "level": "document"}},
    "rates": {"usd": {"time": 2.0e-5}, "kwh": {"time": 1.0e-4}, "time": {"time": 1.0}}})
(out / "architecture.api.json").write_text("""{
  // README style: architecture cards may carry // and /* */ comments (JSONC)
  "arch_id": "openai_api",
  "name": "OpenAI API",
  "kind": "api",
  "variables": {
    "input_tokens":  { "unit": "token", "tokenizer": "tiktoken[o200k_base]", "level": "document" },
    "output_tokens": { "unit": "token", "tokenizer": "tiktoken[o200k_base]", "level": "document" },
    "time":          { "unit": "s", "level": "document" }
  },
  "rates": {                                  // usd = 2.5e-6*input + 1e-5*output
    "usd":  { "input_tokens": 2.5e-6, "output_tokens": 1e-5 },
    "time": { "time": 1.0 }
  }
}
""", encoding="utf-8")

BART_INF = {"num_beams": {"type": "int", "min": 1},
            "input_max_size": {"type": "context_window", "tokenizer": "transformers[bart-base]",
                               "min": 1, "max": 1024, "default": 1024},
            "output_max_size": {"type": "context_window", "tokenizer": "transformers[bart-base]",
                                "min": 1, "max": 1024, "default": 1024}}
wj("model.bartbasekp20k.json", {
    "model_id": "https://huggingface.co/taln-ls2n/bart-base-kp20k", "name": "bart-base-kp20k",
    "family": [["Neural model", "Paradigms", "One2Seq"], ["Pre-trained models", "BART"]],
    "backend": "transformers", "capabilities": {"extractive": True, "abstractive": True},
    "parameters": {"total": 139420416}, "supervision": ["kp20k"],
    "domains": [{"domain": "Academic", "sub-domain": "computer-sciences"}],
    "languages": ["en"],
    "references": {"url": "https://huggingface.co/taln-ls2n/bart-base-kp20k",
                   "bibtex": "@misc{bartbasekp20k, title={bart-base-kp20k}}"},
    "inference": BART_INF})
wj("model.bartkptimes.json", {
    "model_id": "https://huggingface.co/taln-ls2n/bart-kptimes", "name": "bart-kptimes",
    "family": [["Neural model", "Paradigms", "One2Seq"], ["Pre-trained models", "BART"]],
    "backend": "transformers", "capabilities": {"extractive": True, "abstractive": True},
    "parameters": {"total": 406290432}, "supervision": ["kptimes"],
    "domains": [{"domain": "News", "sub-domain": "general"}], "languages": ["en"],
    "references": {"url": "https://huggingface.co/taln-ls2n/bart-kptimes"},
    "inference": BART_INF})
wj("model.multipartiterank.json", {
    "model_id": "https://github.com/boudinfl/pke", "name": "MultipartiteRank",
    "family": [["Statistical model", "Graph-based", "Unsupervised"]],
    "backend": "pke", "capabilities": {"extractive": True, "abstractive": False},
    "languages": ["en", "fr"],
    "references": {"url": "https://aclanthology.org/N18-2105/"},
    "inference": {
        "pos": {"type": "set", "values": ["ADJ", "NOUN", "PROPN", "VERB", "ADV"],
                "default": ["ADJ", "NOUN", "PROPN"]},
        "alpha": {"type": "float", "min": 0.0, "default": 1.1},
        "threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.74},
        "method": {"type": "str", "values": ["average", "ward", "single", "complete"],
                   "default": "average"},
        "n_extract": {"type": "int", "min": 1, "default": 10}}})

# ---------------------------------------------------------------- text
CS = ("graph algorithm network protocol distributed system database query index "
      "optimization compiler parallel memory cache scheduling security encryption "
      "learning classifier neural model training inference retrieval ranking search "
      "semantic ontology agent planning logic verification program analysis software "
      "testing requirement architecture service cloud wireless sensor routing mobile "
      "multimedia image video signal compression coding error correction channel "
      "estimation clustering feature selection kernel regression bayesian probabilistic "
      "markov decision process reinforcement reward policy convergence complexity "
      "approximation heuristic evolutionary genetic swarm fuzzy control robot "
      "vision recognition detection tracking segmentation language parsing").split()
STOP = "the of and to in a is that for with as by on we this are be an from which".split()


def phrase(pool, n=None):
    n = n or R.choice([1, 2, 2, 2, 3, 3])
    return " ".join(R.sample(pool, n))


def cs_text(n_words, insert):
    words = [R.choice(CS) if R.random() < 0.55 else R.choice(STOP) for _ in range(n_words)]
    for p, pos in insert:
        toks = p.split()
        words[pos:pos] = toks
    return " ".join(words)


# semeval2010: long full texts; author + reader gold (some '+' variants)
sem = []
for split, n in (("train", 20), ("test", 40)):
    for i in range(n):
        gold_a = [phrase(CS) for _ in range(R.randint(3, 6))]
        gold_r = [phrase(CS) for _ in range(R.randint(6, 12))]
        gold_r = [g + "+" + g + "s" if R.random() < 0.15 else g for g in gold_r]
        L = R.randint(3000, 7000)
        ins = [(g.split("+")[0], R.randint(0, L - 1)) for g in gold_a + gold_r
               if R.random() < 0.7]
        body = cs_text(L, ins)
        sem.append({"_id": f"semeval2010_{split}_{i}",
                    "metadata": {"split": split, "category": "C.2.4"},
                    "sections": [{"field": "title+full-text", "language": ["en"],
                                  "content": f"On {gold_a[0]}. " + body}],
                    "annotations": [{"annotator": "author", "keyphrases": gold_a},
                                    {"annotator": "reader", "keyphrases": gold_r}]})
wl("document.semeval2010.jsonl", sem)

# talnarchives: French docs; some flawed on purpose
FR = ("analyse syntaxique corpus annotation traduction automatique lexique "
      "morphologie sémantique dialogue système extraction information entités "
      "nommées apprentissage modèle langue évaluation grammaire arbre dépendances "
      "résumé texte discours terminologie alignement segmentation phonétique").split()
FR_STOP = "le la les de des du un une et en dans que qui pour sur est par avec".split()


def fr_text(n):
    return " ".join(R.choice(FR) if R.random() < 0.5 else R.choice(FR_STOP) for _ in range(n))


taln = []
for i in range(200):
    kps = [" ".join(R.sample(FR, R.choice([1, 2, 2, 3]))) for _ in range(R.randint(3, 6))]
    abstract = fr_text(R.randint(80, 160))
    for k in kps:
        if R.random() < 0.6:
            abstract += " " + k + " " + fr_text(8)
    secs = [{"field": "title", "content": "Une étude de la " + kps[0] + " pour le " + fr_text(4)},
            {"field": "abstract", "content": abstract}]
    if i % 9 == 0:   # title declared English but written in French
        secs.append({"field": "title_en", "content": "Une approche de la " + kps[0] + " dans les corpus"})
    else:
        secs.append({"field": "title_en", "content": "A study of " + phrase(CS) + " for the analysis of text"})
    if i % 13 != 0:  # sometimes no English abstract
        secs.append({"field": "abstract_en", "content": cs_text(R.randint(60, 120), [])})
    taln.append({"_id": f"taln_{i}", "metadata": {"split": "test"}, "sections": secs,
                 # card-schema spelling ("languages") on purpose; kp20k & co
                 # use the per-document "language" spelling
                 "annotations": [{"annotator": "author", "languages": ["fr"],
                                  "keyphrases": kps}]})
wl("document.talnarchives.jsonl", taln)

# the "real" bart-base-kp20k run on kp20k (one batch)
preds = []
for i in range(N_KP20K):
    preds.append({"_id": f"kp20k_testing_{i}",
                  "inferences": list(dict.fromkeys(phrase(CS) for _ in range(R.randint(6, 10))))})
wl("kp20k_bart_batch_00000.jsonl", preds)
wj("kp20k_bart_batch_00000.json", {"batch_idx": 0,
                                   "start_timestamp": "2025/07/30 11:13:10",
                                   "end_timestamp": "2025/07/30 11:24:40",
                                   "costs": {"time": 11.46 * N_KP20K / 10}})
wj("run_04b902a7dc7b.json", {"parameters": {"num_beams": 4, "input_max_size": 512,
                                            "output_max_size": 128}})
print("sample_src written to", out)
