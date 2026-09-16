#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KPViz scaling benchmark — self-contained and reproducible.

Times the three normalisation/analysis components that dominate a keyphrase
evaluation, comparing the field's standard implementations against KPViz over
synthetic corpora parameterised by:

    N — number of documents
    V — shared vocabulary size (unique word types across the corpus)
    L — document length (tokens)

Component        SoTA (as used by the field)            Ours (KPViz)
---------------  -------------------------------------  ----------------------------
Tokenisation     word_tokenize (NLTK; pke, redefining-abs-kps) regex  [^\\W_]+  (NFKC+lower)
Stemming         PorterStemmer (NLTK; ~everyone)         PyStemmer (Snowball/Porter2, C)
PRMU             redefining-abs-kps (redefining-absent-kps)     indexed, in-order (bisect)

Each component is measured three ways:
    SoTA          the standard implementation, every occurrence
    Ours          KPViz's implementation, every occurrence
    Ours (cached) KPViz's implementation, each *unique* phrase analysed once
                  (the cache KPViz keeps across a corpus)

The SoTA blocks reproduce the cited code verbatim in spirit:
  * redefining-abs-kps stems with a fresh PorterStemmer() per word and tests
    presence with a contiguous-subsequence scan `contains()`;
  * the isolated stemming baseline reuses one PorterStemmer() (the fair case).
KPViz's blocks are copied from kpviz/textproc.py.

Dependencies:  nltk (punkt/punkt_tab)  and  PyStemmer.
    pip install nltk PyStemmer
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

Usage:
    python kpviz_scaling_benchmark.py                 # full sweep
    python kpviz_scaling_benchmark.py --quick         # smaller/faster
"""
from __future__ import annotations
import argparse, bisect, random, re, statistics, sys, time, unicodedata

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import Stemmer  # PyStemmer (C)

# ==========================================================================
# SoTA implementations (the field's standard code paths)
# ==========================================================================
def sota_tokenize(s: str) -> list[str]:
    return word_tokenize(s)

def sota_lc_stem_perword(words: list[str]) -> list[str]:
    # redefining-abs-kps: a fresh PorterStemmer() for every word (verbatim)
    return [PorterStemmer().stem(w.lower()) for w in words]

_PS = PorterStemmer()
def sota_stem_fair(words: list[str]) -> list[str]:
    # fair isolated stemming baseline: one reused instance
    return [_PS.stem(w) for w in words]

def contains(subseq, inseq) -> bool:
    n, m = len(inseq), len(subseq)
    return any(inseq[p:p + m] == subseq for p in range(0, n - m + 1))

def sota_prmu(tok_title, tok_text, tok_kps) -> list[str]:
    """redefining-abs-kps pmru_uw, returning a label per keyphrase."""
    out = []
    for kp in tok_kps:
        if contains(kp, tok_title) or contains(kp, tok_text):
            out.append("P")
        else:
            present = [w for w in kp if w in tok_title or w in tok_text]
            if len(present) == len(kp):
                out.append("R")
            elif present:
                out.append("M")
            else:
                out.append("U")
    return out

# ==========================================================================
# Ours — KPViz (kpviz/textproc.py, verbatim)
# ==========================================================================
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
def norm_text(t: str) -> str:
    return unicodedata.normalize("NFKC", t or "").lower()
def ours_tokenize(t: str) -> list[str]:
    return _WORD_RE.findall(norm_text(t))

_STEMMER = Stemmer.Stemmer("english")
def ours_stem(words: list[str]) -> list[str]:
    return _STEMMER.stemWords(words)

def position_index(stems: list[str]) -> dict:
    idx: dict = {}
    for i, s in enumerate(stems):
        idx.setdefault(s, []).append(i)
    return idx

def inorder_chain_end(kp_stems, index) -> int:
    pos = -1
    for s in kp_stems:
        lst = index.get(s)
        if not lst:
            return -1
        j = bisect.bisect_right(lst, pos)
        if j >= len(lst):
            return -1
        pos = lst[j]
    return pos

def ours_prmu(kp_stems_list, index) -> list[str]:
    out = []
    for stems in kp_stems_list:
        if not stems:
            out.append("U"); continue
        if inorder_chain_end(stems, index) >= 0:
            out.append("P"); continue
        uniq = set(stems)
        present = sum(1 for s in uniq if s in index)
        out.append("R" if present == len(uniq) else ("M" if present else "U"))
    return out

# ==========================================================================
# Synthetic corpus (documents share one vocabulary; keyphrases recur)
# ==========================================================================
ROOTS = ("system model data method result graph network learn train evaluate "
    "token phrase keyword document corpus vector search index rank score match "
    "cluster classify predict generate extract encode decode embed align parse "
    "compute optimize sample measure analyse compare report select filter merge "
    "detect recognise translate summarise annotate label validate benchmark "
    "process retrieve store query transform normalise stem tokenise segment "
    "represent estimate approximate simulate propagate aggregate distribute "
    "structure organise categorise associate correlate iterate converge derive "
    "language sentence semantic syntax lexical neural attention transformer "
    "gradient parameter feature signal channel latent hidden output input layer "
    "kernel matrix tensor probability entropy distance similarity frequency "
    "domain resource scenario architecture pipeline module component interface "
    "protocol dataset baseline metric accuracy precision recall coverage bias "
    "quality error penalty threshold window context inference training testing "
    "author reader expert indexer collection archive library citation abstract "
    "title section content passage summary topic concept relation entity mention"
    ).split()

_SFX = ["", "s", "ed", "ing", "es", "er", "ers", "ion", "ions", "al", "ally",
        "ment", "ments", "ive", "ized", "izing", "ness", "ly", "able", "ic"]

def build_vocab(V: int) -> list[str]:
    vocab = []
    for r in ROOTS:
        for s in _SFX:
            vocab.append(r + s)
            if len(vocab) >= V:
                return vocab[:V]
    i = 0
    while len(vocab) < V:                       # pad if needed
        vocab.append(f"term{i}"); i += 1
    return vocab[:V]

def make_corpus(N, V, L, K=25, seed=0):
    """Return list of docs: each = (doc_text:str, [kp_str, ...]).
    K keyphrases/doc stands in for gold + one run's predictions."""
    rng = random.Random(seed)
    vocab = build_vocab(V)
    weights = [1.0 / (i + 1) for i in range(len(vocab))]     # Zipfian
    poolsize = min(max(300, V), 20000)
    pool = []
    for _ in range(poolsize):
        n = rng.choice([1, 1, 2, 2, 2, 3])
        pool.append(rng.choices(vocab, weights=weights, k=n))
    docs = []
    for _ in range(N):
        toks = rng.choices(vocab, weights=weights, k=L)
        kps = [list(rng.choice(pool)) for _ in range(K)]
        for kp in kps:                                       # inject ~half -> Present
            if rng.random() < 0.5 and len(kp) <= L:
                p = rng.randint(0, L - len(kp))
                toks[p:p + len(kp)] = kp
        docs.append((" ".join(toks), [" ".join(kp) for kp in kps]))
    return docs

# ==========================================================================
# Benchmark drivers  (return seconds)
# ==========================================================================
def bench_tokenize(docs, mode):
    # keyphrases are what recur and get cached; documents are tokenised once
    # (inside PRMU) so they are excluded here.
    strings = [kp for _, kps in docs for kp in kps]
    t0 = time.perf_counter()
    if mode == "sota":
        for s in strings: sota_tokenize(s)
    elif mode == "ours":
        for s in strings: ours_tokenize(s)
    else:  # ours+cache: unique strings only
        cache = {}
        for s in strings:
            if s not in cache: cache[s] = ours_tokenize(s)
    return time.perf_counter() - t0

def bench_stem(docs, mode):
    # word lists from every keyphrase occurrence
    lists = [kp.split() for _, kps in docs for kp in kps]
    t0 = time.perf_counter()
    if mode == "sota":
        for ws in lists: sota_stem_fair(ws)
    elif mode == "ours":
        for ws in lists: ours_stem(ws)
    else:  # ours+cache: each unique word stemmed once
        cache = {}
        for ws in lists:
            for w in ws:
                if w not in cache: cache[w] = _STEMMER.stemWords([w])[0]
    return time.perf_counter() - t0

def bench_prmu(docs, mode):
    t0 = time.perf_counter()
    if mode == "sota":
        for text, kps in docs:
            tt = sota_lc_stem_perword(sota_tokenize(text))
            tk = [sota_lc_stem_perword(sota_tokenize(k)) for k in kps]
            sota_prmu([], tt, tk)
    elif mode == "ours":
        for text, kps in docs:
            idx = position_index(ours_stem(ours_tokenize(text)))
            kp_stems = [ours_stem(ours_tokenize(k)) for k in kps]
            ours_prmu(kp_stems, idx)
    else:  # ours+cache: unique keyphrase -> stems cached across the corpus
        cache = {}
        for text, kps in docs:
            idx = position_index(ours_stem(ours_tokenize(text)))
            kp_stems = []
            for k in kps:
                st = cache.get(k)
                if st is None:
                    st = ours_stem(ours_tokenize(k)); cache[k] = st
                kp_stems.append(st)
            ours_prmu(kp_stems, idx)
    return time.perf_counter() - t0

BENCH = {"Tokenise": bench_tokenize, "Stem": bench_stem, "PRMU": bench_prmu}

def timed(fn, docs, mode, repeats=3):
    return min(fn(docs, mode) for _ in range(repeats))

# ==========================================================================
# Sweeps
# ==========================================================================
def run_point(N, V, L, repeats):
    docs = make_corpus(N, V, L)
    n_occ = sum(len(kps) for _, kps in docs)
    row = {}
    for comp, fn in BENCH.items():
        row[comp] = {m: timed(fn, docs, m, repeats) for m in ("sota", "ours", "cache")}
    return n_occ, row

def fmt(s):
    return f"{s*1e3:8.1f}ms" if s < 1 else f"{s:8.2f}s "

def print_block(title, points, repeats):
    print(f"\n### {title}")
    for label, (N, V, L) in points:
        n_occ, row = run_point(N, V, L, repeats)
        print(f"\n  {label}   (N={N} docs, V={V} vocab, L={L} tok, "
              f"{n_occ} keyphrase occurrences)")
        print(f"    {'component':10s} {'SoTA':>10s} {'Ours':>10s} "
              f"{'Ours+cache':>11s} {'Ours×':>7s} {'cache×':>7s}")
        for comp in BENCH:
            s, o, c = (row[comp][m] for m in ("sota", "ours", "cache"))
            print(f"    {comp:10s} {fmt(s)} {fmt(o)} {fmt(c)} "
                  f"{s/o:6.1f}× {s/c:6.1f}×")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)  # matches Table 2 (best of 2)
    args = ap.parse_args()
    try:
        word_tokenize("warm up the punkt tokenizer .")
    except LookupError:
        import os, nltk
        os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
        try: nltk.pathsec.ALLOW_PROXIED_FETCH = True   # only needed behind a proxy
        except Exception: pass
        for p in ("punkt", "punkt_tab"):
            try: nltk.download(p, quiet=True)
            except Exception: pass
        word_tokenize("warm up .")

    if args.quick:
        N = [(f"N={n}", (n, 5000, 256)) for n in (250, 1000, 4000)]
        V = [(f"V={v}", (2000, v, 256)) for v in (1000, 8000)]
        L = [(f"L={l}", (2000, 5000, l)) for l in (128, 1024)]
    else:
        N = [(f"N={n}", (n, 5000, 256)) for n in (500, 2000, 8000)]
        V = [(f"V={v}", (2000, v, 256)) for v in (1000, 5000, 25000)]
        L = [(f"L={l}", (2000, 5000, l)) for l in (128, 512, 2048)]

    print("KPViz scaling benchmark  —  min of %d repeats" % args.repeats)
    print_block("Scaling N (documents); V=5000, L=256", N, args.repeats)
    print_block("Scaling V (shared vocabulary); N=2000, L=256", V, args.repeats)
    print_block("Scaling L (document length); N=2000, V=5000", L, args.repeats)

if __name__ == "__main__":
    sys.exit(main())
