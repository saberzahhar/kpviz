#!/usr/bin/env python3
"""Build a complete KPViz data tree around *your* cards.

    python tools/synth_tree.py --cards MY_TREE --out synth_data [--test-docs 1000]

MY_TREE holds any of architectures/, models/ and insights/ (your real
cards and similarity file); they are copied verbatim. What is usually too
big to share — the document collections and the inference runs — is
synthesised to match them:

* documents: every document your insights/scores.jsonl references exists,
  with the split its pairing implies (a pair that links a test document to
  training data is a leakage pair); documents linked by a pair share their
  keyphrases (near-duplicates), plus extra test/training documents so each
  dataset has an evaluation set; long full-text sections where the dataset
  is a full-text one; a few language-mismatched abstracts; French where the
  dataset is French;
* runs: for every model card, parameter settings drawn from its own
  `inference` schema (including one illegal value), on the datasets its
  languages allow, under the architecture its backend implies — named by
  file token or by declared id, so both resolution paths are exercised —
  with costs emitted exactly as each architecture card declares its
  variables (document- or batch-level), a missing run card, an incomplete
  run and an undeclared architecture (`n.a`);
* behaviour: supervision data helps in-domain, leaked documents score
  higher for the model that was trained on their duplicate, more beams
  help a little, and a model never predicts a present keyphrase that lies
  beyond its context window.

Deterministic for a given --seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kpviz.util import load_jsonc  # noqa: E402

# ---------------------------------------------------------------------------
# dataset profiles (known benchmarks by name; anything else gets "generic")
# ---------------------------------------------------------------------------
PROFILES = {
    "kp20k": dict(domain=("Academic", "computer-sciences"), lex="cs",
                  sections=("title", "abstract"), ann=["author"], lang="en",
                  desc="KP20k: computer-science article abstracts."),
    "kpbiomed": dict(domain=("Academic", "medical-sciences"), lex="bio",
                     sections=("title", "abstract"), ann=["author"], lang="en",
                     desc="KPBiomed: PubMed abstracts."),
    "kptimes": dict(domain=("Journalism", "print-news"), lex="news",
                    sections=("title", "abstract"), ann=["editor"], lang="en",
                    desc="KPTimes: news articles (NY Times, Japan Times)."),
    "semeval-2010": dict(domain=("Academic", "computer-sciences"), lex="cs",
                         sections=("title", "abstract", "body"),
                         ann=["author", "reader"], lang="en",
                         desc="SemEval-2010 Task 5: full scientific articles."),
    "taln-archives": dict(domain=("Academic", "linguistics"), lex="fr",
                          sections=("title", "abstract"), ann=["author"],
                          lang="fr", desc="TALN Archives: French NLP papers."),
}
GENERIC = dict(domain=("General", "general"), lex="cs",
               sections=("title", "abstract"), ann=["author"], lang="en",
               desc="Synthetic collection.")

LEX = {
    "cs": (["neural", "distributed", "probabilistic", "adaptive", "semantic",
            "parallel", "sparse", "robust", "incremental", "graph-based",
            "federated", "secure", "real-time", "approximate", "multi-agent"],
           ["network", "scheduling", "retrieval", "clustering", "caching",
            "compiler", "protocol", "query optimization", "classification",
            "embedding", "verification", "routing", "indexing", "inference",
            "segmentation", "load balancing", "access control", "ranking"]),
    "bio": (["chronic", "acute", "metabolic", "cardiovascular", "genetic",
             "pediatric", "inflammatory", "oncologic", "renal", "hepatic",
             "neonatal", "autoimmune", "viral", "bacterial", "hormonal"],
            ["disease", "therapy", "biomarker", "expression", "syndrome",
             "infection", "resistance", "inflammation", "dysfunction",
             "screening", "mortality", "outcome", "pathway", "response",
             "carcinoma", "insufficiency"]),
    "news": (["central", "regional", "municipal", "federal", "national",
              "coastal", "global", "domestic", "industrial", "rural"],
             ["bank", "election", "trade talks", "budget", "strike", "tariffs",
              "housing market", "energy policy", "summit", "court ruling",
              "tourism", "defense spending", "olympics", "earthquake"]),
    "fr": (["analyse", "extraction", "traduction", "annotation", "segmentation",
            "reconnaissance", "classification", "désambiguïsation", "génération",
            "évaluation"],
           ["syntaxique", "sémantique", "automatique", "lexicale",
            "morphologique", "de corpus", "de la parole", "terminologique",
            "multilingue", "d'entités nommées"]),
}
FILLER = {
    "cs": ["In this paper we investigate the problem in depth",
           "Extensive experiments demonstrate the effectiveness of the approach",
           "We formalise the task and derive an efficient algorithm",
           "The method is evaluated on public benchmarks against strong baselines",
           "Our analysis highlights trade-offs between accuracy and efficiency"],
    "bio": ["Patients were enrolled in a multicentre prospective cohort",
            "Expression levels were quantified using standard assays",
            "Statistical significance was assessed with corrected tests",
            "The findings suggest a potential pathway for intervention",
            "Clinical outcomes were followed up at six and twelve months"],
    "news": ["Officials announced the decision at a press conference",
             "Analysts said the move could reshape the market",
             "The government has faced mounting pressure over the policy",
             "Local residents expressed mixed reactions to the announcement"],
    "fr": ["Nous présentons une méthode originale pour cette tâche",
           "Les expériences menées sur plusieurs corpus confirment l'intérêt",
           "Cet article décrit une approche fondée sur des ressources libres",
           "Nous évaluons le système sur des données de référence"],
}
SPANISH = ["Los resultados muestran una mejora significativa en los pacientes",
           "Este estudio analiza la respuesta clínica durante dos años",
           "Se observó una reducción de la mortalidad en el grupo tratado"]
TEMPL = {
    "en": ["This work focuses on {p} as a central concern.",
           "We study {p} and its implications in detail.",
           "The role of {p} has attracted considerable attention.",
           "Recent progress on {p} motivates this study."],
    "fr": ["Ce travail porte sur {p} de manière approfondie.",
           "Nous étudions {p} et ses implications.",
           "Le rôle de {p} a suscité une attention considérable."],
}


def phrase_pool(lex: str) -> list[str]:
    adj, noun = LEX[lex]
    if lex == "fr":
        return [f"{a} {n}" for a in adj for n in noun]
    return [f"{a} {n}" for a in adj for n in noun] + list(noun)


# ---------------------------------------------------------------------------
# splits and ids
# ---------------------------------------------------------------------------
def split_of_ref(ds: str, doc_id: str, partners: set[tuple[str, str]]) -> str:
    """The split a referenced document most plausibly belongs to."""
    if ds == "kp20k":
        p = doc_id.rsplit("_", 1)[0]
        return {"kp20k_training": "train", "kp20k_testing": "test",
                "kp20k_validation": "validation"}.get(p, "test")
    if ds == "kptimes":
        return "test" if doc_id.startswith("jp") else "train"
    # a document paired with some *training* document elsewhere is a test
    # document (that is the leakage the pair records); paired only with
    # evaluation documents, it plays the training side
    if any(sp == "train" for _ods, sp in partners):
        return "test"
    return "train" if partners else "test"


class IdMaker:
    def __init__(self, ds: str, taken: set[str], rng: random.Random):
        self.ds, self.taken, self.rng, self.n = ds, taken, rng, 0

    def new(self, split: str) -> str:
        while True:
            self.n += 1
            ds, n = self.ds, self.n
            if ds == "kp20k":
                i = f"kp20k_{ {'train': 'training', 'test': 'testing'}.get(split, split)}_{900000 + n}"
            elif ds == "kpbiomed":
                i = str(40000000 + self.rng.randint(0, 9_999_999))
            elif ds == "kptimes":
                i = f"{'jp' if split == 'test' else 'ny'}{9000000 + n:07d}"
            elif ds == "semeval-2010":
                i = f"{'CHIJ'[n % 4]}-{200 + n}"
            elif ds == "taln-archives":
                i = f"taln-{2000 + n % 20}-{'long' if n % 3 else 'court'}-{n:03d}"
            else:
                i = f"{ds}-{split}-{n}"
            if i not in self.taken:
                self.taken.add(i)
                return i


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def make_doc(rng, ds, prof, doc_id, split, phrases, n_body_words, lang_mix):
    lang = prof["lang"]
    templ = TEMPL["fr" if lang == "fr" else "en"]
    filler = FILLER[prof["lex"]]
    present = phrases[:rng.randint(4, 7)]
    absent_pool = phrase_pool(prof["lex"])
    absent = [p for p in rng.sample(absent_pool, 6) if p not in present][:2]
    title = present[0][0].upper() + present[0][1:] + \
        (f" et {present[1]}" if lang == "fr" else f" and {present[1]}")
    sents = [templ[rng.randrange(len(templ))].format(p=p) for p in present[1:4]]
    sents += [rng.choice(filler) + "." for _ in range(3)]
    rng.shuffle(sents)
    abstract = " ".join(sents)
    if lang_mix:
        abstract = " ".join(rng.sample(SPANISH, 3))
    sections = [{"field": "title", "language": [lang], "content": title},
                {"field": "abstract", "language": [lang], "content": abstract}]
    if "body" in prof["sections"]:
        # a long body: later present keyphrases only appear deep inside it,
        # which is what a bounded context window cannot see
        body, late = [], present[4:]
        for i in range(n_body_words // 12):
            body.append(rng.choice(filler) + ".")
            if late and i > (n_body_words // 12) * 0.6 and rng.random() < 0.05:
                body.append(templ[0].format(p=late.pop(0)))
        body += [templ[1].format(p=p) for p in late]
        sections.append({"field": "body", "language": [lang],
                         "content": " ".join(body)})
    gold = present[:rng.randint(3, len(present))] + absent[:rng.randint(1, 2)]
    rng.shuffle(gold)
    anns = []
    for a in prof["ann"]:
        g = gold if a != "reader" else (
            [p for p in gold if rng.random() < 0.7] + [rng.choice(absent_pool)])
        anns.append({"annotator": a, "languages": [lang], "keyphrases": g})
    return {"_id": doc_id, "metadata": {"split": split},
            "sections": sections, "annotations": anns}


def dataset_card(ds: str, prof: dict) -> dict:
    lang = prof["lang"]
    card = {"description": prof["desc"], "domain": prof["domain"][0],
            "sub-domain": prof["domain"][1],
            "metadata": {"split": {"type": "split"}},
            "document": {}, "annotations": {}}
    # one entry per section field, as the documents carry them
    for fld in prof["sections"]:
        card["document"][fld] = {"type": [fld], "modality": "text",
                                 "languages": [lang]}
    for a in prof["ann"]:
        card["annotations"][a] = {
            "type": a, "expertise": {"author": "author", "editor": "editor",
                                     "reader": "student"}.get(a, a),
            "languages": [lang]}
    return card


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def run_id(params: dict) -> str:
    return hashlib.blake2b(json.dumps(params, sort_keys=True).encode(),
                           digest_size=6).hexdigest()


def param_grid(model: dict) -> list[dict]:
    """Settings a practitioner would try, from the card's own schema."""
    inf = model.get("inference") or {}
    big = (model.get("parameters") or {}).get("total", 0) > 1e9 or \
        model.get("backend") == "openai"
    grids: list[dict] = []
    if "num_beams" in inf:
        wins = [512, None] if "input_max_size" in inf else [None]
        for b in (1, 4, 10):
            for w in wins:
                p = {"num_beams": b}
                if w:
                    p["input_max_size"] = w
                grids.append(p)
    elif "temperature" in inf:
        base = {"prompt": "kp-extract-v1"} if "prompt" in inf else {}
        grids = [dict(base, temperature=0.0),
                 dict(base, temperature=0.7),
                 dict(base, temperature=0.0, few_shot=3)]
        if big and "input_max_size" in inf and "tiktoken[llama3]" in json.dumps(inf):
            grids.append(dict(base, temperature=0.0, input_max_size=4096))
    elif "method" in inf:
        vals = (inf["method"].get("values") or ["average"])
        pos = [v for v in ("NOUN", "PROPN", "ADJ")
               if v in (inf.get("pos", {}).get("values") or ["NOUN", "PROPN", "ADJ"])]
        for m in [v for v in ("average", "ward", "single") if v in vals] or vals[:2]:
            grids.append({"method": m, "threshold": 0.74, "pos": pos,
                          "n_extract": 10})
        # an illegal setting: flagged by KPViz, never dropped
        grids.append({"method": vals[0], "threshold": 1.5,
                      "pos": pos + ["NOUNS"], "n_extract": 10})
    else:
        grids = [{}]
    return grids


def arch_for(model: dict, archs: dict) -> tuple[str, dict | None]:
    """(folder token, card) — by backend; alternately named by declared id
    and by file token so both resolution paths are exercised."""
    by_kind = {a.get("kind"): (tok, a) for tok, a in archs.items()}
    backend = model.get("backend")
    big = (model.get("parameters") or {}).get("total", 0) > 1e9
    if backend == "openai" and "api" in by_kind:
        tok, a = by_kind["api"]
        return a.get("arch_id") or tok, a
    if big and "rented" in by_kind:
        tok, a = by_kind["rented"]
        return tok, a
    if "local" in by_kind:
        tok, a = by_kind["local"]
        return (a.get("arch_id") or tok) if backend == "transformers" else tok, a
    if archs:
        tok, a = next(iter(archs.items()))
        return tok, a
    return "n.a", None


def quality(model: dict, ds: str, params: dict) -> float:
    sup = set(model.get("supervision") or [])
    backend = model.get("backend")
    if backend == "pke":
        q = 0.30
    elif backend == "openai":
        q = 0.50
    elif (model.get("parameters") or {}).get("total", 0) > 1e9:
        q = 0.46
    else:
        q = 0.58 if ds in sup else 0.36
        q += 0.04 * ((model.get("parameters") or {}).get("total", 0) > 3e8)
    q += 0.015 * math.log2(params.get("num_beams", 1))
    q -= 0.06 * (params.get("temperature", 0) or 0)
    q += 0.03 * bool(params.get("few_shot"))
    return max(0.05, min(0.9, q))


def predict(rng, doc, model, params, q, window_tokens, leaked, pool):
    abstractive = (model.get("capabilities") or {}).get("abstractive", True)
    text_words = []
    first_pos = {}
    for s in doc["sections"]:
        for w in s["content"].lower().split():
            text_words.append(w.strip(".,;:"))
    joined = " ".join(text_words)
    gold = doc["annotations"][0]["keyphrases"]
    hits = []
    for g in gold:
        pos = joined.find(g.lower())
        present = pos >= 0
        if present:
            first_pos[g] = len(joined[:pos].split())
        if present and window_tokens and first_pos[g] * 1.3 > window_tokens:
            continue                    # the model never saw it
        if not present and not abstractive:
            continue
        p = q + (0.28 if leaked else 0.0) - (0.18 if not present else 0.0)
        if rng.random() < p:
            hits.append(g)
    junk = rng.sample(pool, 8)
    k = params.get("n_extract") or rng.randint(5, 10)
    preds = hits + [j for j in junk if j not in hits]
    # the hits mostly rank first
    preds.sort(key=lambda p: (p not in hits) + rng.random() * 0.9)
    return preds[:k]


def emit_run(rng, out, ds, model_tok, model, arch_tok, arch, params, docs,
             group_of, leak_groups, pool, t0, coverage=1.0, write_card=True):
    rid = run_id(params)
    rd = out / "inferences" / ds / model_tok / arch_tok / rid
    rd.mkdir(parents=True, exist_ok=True)
    if write_card:
        (rd / f"run_{rid}.json").write_text(json.dumps({"parameters": params},
                                                       indent=2))
    inf = model.get("inference") or {}
    win = params.get("input_max_size") or (inf.get("input_max_size") or {}).get("default")
    q = quality(model, ds, params)
    sup = set(model.get("supervision") or [])
    variables = (arch or {}).get("variables") or {}
    speed = 1.0 + (((model.get("parameters") or {}).get("total", 1e8)) / 1e9)
    kept = [d for d in docs if rng.random() < coverage]
    api = (arch or {}).get("kind") == "api"
    t = t0
    for bi in range(0, max(1, math.ceil(len(kept) / 64))):
        chunk = kept[bi * 64:(bi + 1) * 64]
        lines, sums = [], {}
        wall = 0.0
        for d in chunk:
            leaked = any(ld in sup for ld in leak_groups.get(group_of.get((ds, d["_id"])), ()))
            preds = predict(rng, d, model, params, q, win, leaked, pool)
            n_in = min(int(sum(len(s["content"].split()) for s in d["sections"]) * 1.3),
                       win or 10**9)
            dt = (0.02 + n_in / 4000) * speed * rng.uniform(0.8, 1.2)
            wall += dt
            doc_costs = {}
            for var, spec in variables.items():
                if (spec or {}).get("level") != "document":
                    continue
                unit = (spec or {}).get("unit")
                val = (n_in if var.startswith("input") else
                       int(len(", ".join(preds).split()) * 1.3) + 6
                       if unit == "token" else round(dt, 5) if unit == "s" else 1)
                doc_costs[var] = val
                sums[var] = sums.get(var, 0) + val
            row = {"_id": d["_id"], "inferences": preds}
            if doc_costs:
                row["costs"] = doc_costs
            lines.append(row)
        batch_costs = {}
        for var, spec in variables.items():
            if (spec or {}).get("level") == "batch":
                unit = (spec or {}).get("unit")
                batch_costs[var] = (len(chunk) if unit == "count"
                                    else round(wall * 1.1, 4) if unit == "s"
                                    else sum(sums.values()))
        t1 = t + timedelta(seconds=wall * (3.0 if api else 1.1))
        fmt = ("%Y-%m-%dT%H:%M:%S.%f+00:00" if api else "%Y/%m/%d %H:%M:%S")
        meta = {"batch_idx": bi, "start_timestamp": t.strftime(fmt),
                "end_timestamp": t1.strftime(fmt)}
        if batch_costs:
            meta["costs"] = batch_costs
        (rd / f"batch_{bi:05d}.json").write_text(json.dumps(meta, indent=2))
        with open(rd / f"batch_{bi:05d}.jsonl", "w", encoding="utf-8") as f:
            for r in lines:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        t = t1 + timedelta(seconds=20)
    return len(kept)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cards", required=True,
                    help="folder with architectures/, models/, insights/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-docs", type=int, default=1000,
                    help="evaluated documents per dataset (default 1000)")
    ap.add_argument("--extra-train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    src, out = Path(a.cards), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for sub in ("architectures", "models", "insights"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, out / sub)

    models = {p.name[len("model."):-len(".json")]: load_jsonc(p)
              for p in sorted((out / "models").glob("model.*.json"))}
    archs = {p.name[len("architecture."):-len(".json")]: load_jsonc(p)
             for p in sorted((out / "architectures").glob("architecture.*.json"))}

    # ---- the similarity graph: referenced ids, partners, groups ----------
    refs: dict[str, dict[str, set]] = {}
    edges: list[tuple] = []
    for f in sorted((out / "insights").glob("*.jsonl")) if (out / "insights").is_dir() else []:
        for ln in open(f, encoding="utf-8"):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            A = (d.get("dataset_a") or d.get("dataset_A"), d.get("doc_id_a") or d.get("doc_id_A"))
            B = (d.get("dataset_b") or d.get("dataset_B"), d.get("doc_id_b") or d.get("doc_id_B"))
            if None in A or None in B:
                continue
            edges.append((A, B))
            refs.setdefault(A[0], {}).setdefault(A[1], set())
            refs.setdefault(B[0], {}).setdefault(B[1], set())

    datasets = sorted(set(refs) | {s for m in models.values()
                                   for s in (m.get("supervision") or [])}
                      | {"kp20k", "kpbiomed", "kptimes"})
    # splits: kp20k/kptimes from their ids first, the rest from partners
    split: dict[tuple, str] = {}
    for ds, ids in refs.items():
        if ds in ("kp20k", "kptimes"):
            for i in ids:
                split[(ds, i)] = split_of_ref(ds, i, set())
    partners: dict[tuple, set] = {}
    for A, B in edges:
        partners.setdefault(A, set()).add(B)
        partners.setdefault(B, set()).add(A)
    for ds, ids in refs.items():
        if ds in ("kp20k", "kptimes"):
            continue
        for i in ids:
            ps = {(o[0], split.get(o, "test")) for o in partners.get((ds, i), ())}
            split[(ds, i)] = split_of_ref(ds, i, ps)

    # union-find groups of near-duplicates (share keyphrases)
    parent: dict[tuple, tuple] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for A, B in edges:
        ra, rb = find(A), find(B)
        if ra != rb:
            parent[ra] = rb
    group_of = {(ds, i): find((ds, i)) for ds, ids in refs.items() for i in ids}
    # which datasets' TRAINING data each group contains (leakage source)
    leak_groups: dict = {}
    for (ds, i), sp in split.items():
        if sp == "train":
            leak_groups.setdefault(group_of[(ds, i)], set()).add(ds)

    # ---- documents -----------------------------------------------------------
    t0 = datetime(2025, 3, 1, 9, 0, tzinfo=timezone.utc)
    (out / "documents").mkdir(exist_ok=True)
    group_phrases: dict = {}
    test_docs: dict[str, list[dict]] = {}
    for ds in datasets:
        prof = PROFILES.get(ds, GENERIC)
        pool = phrase_pool(prof["lex"])
        taken = set(refs.get(ds, {}))
        ids = [(i, split[(ds, i)]) for i in sorted(taken)]
        maker = IdMaker(ds, taken, rng)
        target = min(a.test_docs, 120) if ds in ("semeval-2010", "taln-archives") \
            else a.test_docs
        # the evaluated test set: at most 40 % referenced (mostly leaked)
        # documents, the rest clean — so "with" and "without flagged" both
        # exist; referenced test documents beyond it become validation
        # documents (unless their id names the split, as kp20k's do)
        quota = int(target * 0.4) if ds not in ("kp20k", "kptimes") else target
        kept_test = 0
        for j, (i, sp) in enumerate(ids):
            if sp == "test":
                if kept_test < quota:
                    kept_test += 1
                else:
                    ids[j] = (i, "validation")
                    split[(ds, i)] = "validation"
        n_test = kept_test
        ids += [(maker.new("test"), "test") for _ in range(max(0, target - n_test))]
        ids += [(maker.new("train"), "train") for _ in range(a.extra_train
                                                          if ds not in ("semeval-2010", "taln-archives")
                                                          else 60)]
        (out / "documents" / f"document.{ds}.json").write_text(
            json.dumps(dataset_card(ds, prof), indent=2, ensure_ascii=False))
        docs_test = []
        with open(out / "documents" / f"document.{ds}.jsonl", "w", encoding="utf-8") as f:
            for i, sp in ids:
                g = group_of.get((ds, i))
                if g is not None and g in group_phrases and group_phrases[g][0] == prof["lex"]:
                    base = list(group_phrases[g][1])
                    rng.shuffle(base)
                    phrases = base[:5] + rng.sample(pool, 2)
                else:
                    phrases = rng.sample(pool, 8)
                    if g is not None:
                        group_phrases[g] = (prof["lex"], phrases)
                # KPBiomed ships size-named training splits
                label = "train_large" if (sp == "train" and ds == "kpbiomed") else sp
                d = make_doc(rng, ds, prof, i, label, phrases,
                             2600 if "body" in prof["sections"] else 0,
                             lang_mix=(ds == "kpbiomed" and rng.random() < 0.015))
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                if sp == "test":
                    docs_test.append(d)
        # evaluate leaked documents first, then the rest, up to --test-docs
        docs_test.sort(key=lambda d: (group_of.get((ds, d["_id"])) not in leak_groups,
                                      d["_id"]))
        test_docs[ds] = docs_test[:target]
        print(f"documents {ds:14s} {len(ids):6d} ({n_test} referenced test) "
              f"-> {len(test_docs[ds])} evaluated")

    # ---- runs -----------------------------------------------------------------
    n_runs = n_lines = 0
    for mtok, model in models.items():
        langs = set(model.get("languages") or [])
        multi = (model.get("capabilities") or {}).get("multilingual") or \
            model.get("backend") in ("openai", "pke")
        arch_tok, arch = arch_for(model, archs)
        grid = param_grid(model)
        for di, ds in enumerate(datasets):
            prof = PROFILES.get(ds, GENERIC)
            if prof["lang"] != "en" and not multi and langs and prof["lang"] not in langs:
                continue
            if not test_docs.get(ds):
                continue
            # the big models are run on fewer settings where data is large
            g = grid if (len(test_docs[ds]) <= 200 or model.get("backend") != "transformers"
                         or (model.get("parameters") or {}).get("total", 0) < 3e8) \
                else grid[::2]
            pool = phrase_pool(prof["lex"])
            for pi, params in enumerate(g):
                write_card = not (model.get("backend") == "transformers"
                                  and (model.get("parameters") or {}).get("total", 0) > 1e9
                                  and ds == "kptimes" and pi == 0)
                cov = 0.93 if (model.get("backend") == "openai" and ds == "kpbiomed"
                               and pi == 1) else 1.0
                n_lines += emit_run(rng, out, ds, mtok, model, arch_tok, arch,
                                    params, test_docs[ds], group_of, leak_groups,
                                    pool, t0 + timedelta(days=di, hours=pi),
                                    coverage=cov, write_card=write_card)
                n_runs += 1
            if model.get("backend") == "pke" and ds == datasets[0]:
                n_lines += emit_run(rng, out, ds, mtok, model, "n.a", None,
                                    grid[0], test_docs[ds], group_of,
                                    leak_groups, pool, t0)
                n_runs += 1
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"runs {n_runs} · prediction lines {n_lines} · tree {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
