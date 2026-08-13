"""Text processing, throughput edition.

Tokens come from spaCy (blank pipelines — pure tokenizers, no models
needed); stems from PyStemmer (C); every keyphrase is analysed once and
cached (worker-local dict + the global `keyphrases` table). PRMU is the
in-order definition computed on stemmed spaCy tokens:

    P — every keyphrase token appears in the document *in order*
    R — every token appears, but never in a single in-order chain
    M — some tokens appear
    U — none do

Model tokenizers (`transformers[...]`, `tiktoken[...]`) resolve exactly
when their backend + assets are available and fall back to a flagged
word-ratio heuristic otherwise. Nothing in this module raises past it.
"""
from __future__ import annotations

import bisect
import importlib.util
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Fast regex word tokenisation (language detection + section word counts)
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

_SNOWBALL_LANG = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "ru": "russian",
}


def norm_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(norm_text(text))


def norm_phrase(text: str) -> str:
    """Canonical keyphrase identity: NFKC, lowercased, whitespace-collapsed."""
    return " ".join(norm_text(text).split())


def split_variants(raw: str) -> list[str]:
    """Gold keyphrases may carry alternate forms joined by '+'."""
    parts = [p.strip() for p in (raw or "").split("+")]
    return [p for p in parts if p] or [raw]


# --------------------------------------------------------------------------
# Stemming: PyStemmer (C) -> snowballstemmer -> identity
# --------------------------------------------------------------------------
class _IdentityStemmer:
    def stemWords(self, tokens):
        return list(tokens)


@lru_cache(maxsize=16)
def get_stemmer(lang: str | None):
    name = _SNOWBALL_LANG.get((lang or "en")[:2])
    if not name:
        return _IdentityStemmer()
    try:
        import Stemmer  # PyStemmer — C bindings
        return Stemmer.Stemmer(name)
    except Exception:
        pass
    try:
        import snowballstemmer
        return snowballstemmer.stemmer(name)
    except Exception:
        return _IdentityStemmer()


def stem_tokens(tokens: list[str], lang: str | None) -> list[str]:
    return get_stemmer(lang).stemWords(tokens)


def stem_phrase(text: str, lang: str | None) -> str:
    """Stemmed representation of a short phrase (spaCy tokens + PyStemmer)."""
    return " ".join(stem_tokens(spacy_word_tokens(text, lang), lang))


# --------------------------------------------------------------------------
# spaCy tokenisation (blank pipelines; no downloads, C-speed)
# --------------------------------------------------------------------------
_BLANK_LANGS = {"en", "fr", "de", "es", "it", "pt", "nl", "ru"}


@lru_cache(maxsize=16)
def _blank(lang: str | None):
    import spacy
    code = (lang or "en")[:2]
    try:
        nlp = spacy.blank(code if code in _BLANK_LANGS else "xx")
    except Exception:
        nlp = spacy.blank("xx")
    nlp.max_length = 20_000_000
    return nlp


def spacy_word_tokens(text: str, lang: str | None) -> list[str]:
    """Lowercased word tokens of a short text (punct/space dropped)."""
    doc = _blank(lang).tokenizer(norm_text(text))
    return [t.text for t in doc if not (t.is_punct or t.is_space)]


def spacy_doc_tokens(text: str, lang: str | None) -> tuple[list[str], list[int]]:
    """(lowercased word tokens, char END offset of each) for a document.

    Uses Doc.to_array to stay in C — iterating Token objects in Python is
    ~4× slower on long documents. The text is normalised to lowercase
    before tokenising, so token strings are plain slices."""
    lowered = norm_text(text)
    doc = _blank(lang).tokenizer(lowered)
    try:
        from spacy.attrs import IDX, IS_PUNCT, IS_SPACE, LENGTH
        arr = doc.to_array([IDX, LENGTH, IS_PUNCT, IS_SPACE])
        toks, ends = [], []
        for i, l, p, s in arr.tolist():
            if p or s:
                continue
            toks.append(lowered[i:i + l])
            ends.append(i + l)
        return toks, ends
    except Exception:
        toks, ends = [], []
        for t in doc:
            if t.is_punct or t.is_space:
                continue
            toks.append(t.text)
            ends.append(t.idx + len(t.text))
        return toks, ends


# --------------------------------------------------------------------------
# Keyphrase analysis cache (worker-local; persisted in the keyphrases table)
# --------------------------------------------------------------------------
class PhraseCache:
    """Analyse each unique (normalised) phrase exactly once per worker.

    Memory-bounded: past `max_size` entries the lookaside cache resets
    (repeats are cheap to recompute; the global table dedups anyway).
    `persist` marks phrases that should reach the global table — the UI
    only ever shows phrases referenced by stored gold rows or predictions,
    so training-only phrases stay worker-local and are never POS-tagged.
    First occurrence wins for the language (INSERT OR IGNORE semantics)."""

    __slots__ = ("_cache", "fresh", "_persisted", "max_size")

    def __init__(self, max_size: int = 300_000):
        self._cache: dict[str, dict] = {}
        self.fresh: dict[str, dict] = {}      # to persist at next drain
        self._persisted: set[str] = set()     # already drained this process
        self.max_size = max_size

    def analyze(self, raw: str, lang: str | None, persist: bool = True) -> dict:
        norm = norm_phrase(raw)
        entry = self._cache.get(norm)
        if entry is None:
            lang2 = (lang or "en")[:2]
            tokens = spacy_word_tokens(norm, lang2)
            entry = {
                "kp": norm, "raw": " ".join(str(raw).split()), "lang": lang2,
                "tokens": tokens,
                "stems": get_stemmer(lang2).stemWords(tokens),
                "n_tokens": len(tokens),
            }
            if len(self._cache) >= self.max_size:      # RAM bound
                self._cache = dict(self.fresh)
            self._cache[norm] = entry
        if persist and norm not in self._persisted and norm not in self.fresh:
            self.fresh[norm] = entry
        return entry

    def drain(self) -> list[dict]:
        out = list(self.fresh.values())
        self._persisted.update(self.fresh)
        if len(self._persisted) > 2_000_000:           # RAM bound (set of str)
            self._persisted.clear()
        self.fresh = {}
        return out


# --------------------------------------------------------------------------
# In-order PRMU on stemmed tokens
# --------------------------------------------------------------------------

def position_index(stems: list[str]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, s in enumerate(stems):
        idx.setdefault(s, []).append(i)
    return idx


def inorder_chain_end(kp_stems: list[str], index: dict[str, list[int]]) -> int:
    """Earliest end position of an in-order chain of kp_stems, else -1.

    Greedy earliest-next is optimal for minimising the end position."""
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


def prmu_classify(variant_stems: list[list[str]],
                  index: dict[str, list[int]]) -> tuple[str, int]:
    """Best PRMU class over the '+'-variants and the earliest chain end.

    P: all tokens appear in order · R: all appear, never in order ·
    M: some appear · U: none."""
    best, best_end = "U", -1
    rank = {"P": 3, "R": 2, "M": 1, "U": 0}
    for stems in variant_stems:
        if not stems:
            continue
        end = inorder_chain_end(stems, index)
        if end >= 0:
            if best != "P" or best_end < 0 or end < best_end:
                best, best_end = "P", (end if best_end < 0 else min(best_end, end))
            continue
        present = sum(1 for s in set(stems) if s in index)
        cat = "R" if present == len(set(stems)) else ("M" if present else "U")
        if rank[cat] > rank[best]:
            best = cat
    return best, (best_end if best == "P" else -1)


# --------------------------------------------------------------------------
# Lightweight language identification (stopword voting)
# --------------------------------------------------------------------------
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the of and to in a is that for it as was with be by on not he this are or his from at which but have an they you were her she we there been their has will its".split()),
    "fr": frozenset("le la les de des du un une et en dans que qui pour sur pas au aux ce cette ces par plus avec ne se sont est été nous vous ils elle mais ou où donc car si leur".split()),
    "de": frozenset("der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach wird bei einer um".split()),
    "es": frozenset("de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta entre cuando".split()),
    "it": frozenset("di che e la il un a per in una sono mi si lo ma le ci con non del più questo al come da dei nel alla".split()),
    "pt": frozenset("de a o que e do da em um para é com não uma os no se na por mais as dos como mas foi ao ele das tem à seu sua".split()),
}
def detect_language(text: str, candidates: list[str] | None = None,
                    min_tokens: int = 5,
                    tokens: list[str] | None = None) -> tuple[str | None, float]:
    """Stopword vote over the *whole* text (no prefix sampling). Pass
    `tokens` to reuse a tokenisation the caller already has."""
    toks = tokens if tokens is not None else tokenize(text or "")
    if len(toks) < min_tokens:
        return None, 0.0
    langs = [l for l in (candidates or list(_STOPWORDS)) if l in _STOPWORDS]
    if not langs:
        return None, 0.0
    n = len(toks)
    scores = {l: sum(1 for t in toks if t in _STOPWORDS[l]) / n for l in langs}
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    if scores[best] < 0.08:
        return None, scores[best]
    return best, round(min(1.0, scores[best] * 2 + margin), 3)


# --------------------------------------------------------------------------
# Model tokenizer registry — exact when possible, flagged-approx otherwise
# --------------------------------------------------------------------------
_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*\[\s*([^\]]+)\s*\]\s*$")
_HF_ALIASES = {"bart-base": "facebook/bart-base",
               "bart-large": "facebook/bart-large"}
_FALLBACK_RATIO = {"transformers": 1.30, "tiktoken": 1.25, None: 1.30}


def tokenizer_inner(spec: str) -> str:
    """'tiktoken[o200k_base]' -> 'o200k_base' (display helper)."""
    m = _SPEC_RE.match(spec or "")
    return m.group(2) if m else (spec or "")


class ModelTokenizer:
    def __init__(self, spec: str, cache_dir: Path | None = None):
        self.spec = spec
        m = _SPEC_RE.match(spec or "")
        self.backend = m.group(1).lower() if m else None
        self.name = m.group(2) if m else (spec or "")
        self.cache_dir = cache_dir
        self._impl = None

    def _resolve(self):
        if self._impl is not None:
            return self._impl
        impl = None
        if self.backend == "transformers":
            impl = self._try_hf()
        elif self.backend == "tiktoken":
            impl = self._try_tiktoken()
        if impl is None:
            impl = ("approx", _FALLBACK_RATIO.get(self.backend, 1.30))
        self._impl = impl
        return impl

    def _try_hf(self):
        try:
            from tokenizers import Tokenizer
        except Exception:
            return None
        if self.cache_dir:
            local = self.cache_dir / self.name.replace("/", "__") / "tokenizer.json"
            if local.exists():
                try:
                    return ("hf", Tokenizer.from_file(str(local)))
                except Exception:
                    pass
        for repo in dict.fromkeys([_HF_ALIASES.get(self.name, self.name),
                                   self.name, f"facebook/{self.name}"]):
            try:
                tok = Tokenizer.from_pretrained(repo)
            except Exception:
                continue
            if self.cache_dir:
                # atomic: many workers may race on a cold cache and a torn
                # tokenizer.json would silently degrade every later scan
                try:
                    d = self.cache_dir / self.name.replace("/", "__")
                    d.mkdir(parents=True, exist_ok=True)
                    tmp = d / f"tokenizer.json.{os.getpid()}.tmp"
                    tok.save(str(tmp))
                    os.replace(tmp, d / "tokenizer.json")
                except Exception:
                    pass
            return ("hf", tok)
        return None

    def _try_tiktoken(self):
        try:
            import tiktoken
            return ("tiktoken", tiktoken.get_encoding(self.name))
        except Exception:
            return None

    @property
    def exact(self) -> bool:
        return self._resolve()[0] != "approx"

    def count_batch(self, texts: list[str]) -> tuple[list[int], bool]:
        kind, obj = self._resolve()
        if kind == "hf":
            return [len(e.ids) for e in obj.encode_batch(texts)], False
        if kind == "tiktoken":
            return [len(ids) for ids in
                    obj.encode_ordinary_batch(texts)], False
        return [int(round(len(_WORD_RE.findall(t)) * obj)) for t in texts], True

    def count(self, text: str) -> tuple[int, bool]:
        n, approx = self.count_batch([text])
        return n[0], approx

    # -- per-document encoding, reused across every gold phrase -------------
    def encode_cached(self, text: str):
        """A reusable per-document encoding.

        Returns a dict with an ascending array of token *end* positions plus
        the token count, so `char_to_token` is a bisect instead of a re-encode.
        For tiktoken the ends are byte positions (tokens partition the UTF-8
        bytes exactly), which keeps the answer exact — the previous code
        re-encoded the whole prefix once per keyphrase, i.e. tokenised long
        documents ~10× each."""
        kind, obj = self._resolve()
        if kind == "hf":
            enc = obj.encode(text)
            return {"kind": "hf", "ends": [o[1] for o in enc.offsets],
                    "n": len(enc.ids)}
        if kind == "tiktoken":
            ids = obj.encode_ordinary(text)
            try:
                ends, acc = [], 0
                for i in ids:
                    acc += len(obj.decode_single_token_bytes(i))
                    ends.append(acc)
                return {"kind": "tiktoken_bytes", "ends": ends, "n": len(ids)}
            except Exception:
                return {"kind": "tiktoken_slow", "n": len(ids)}
        return None

    def char_to_token(self, text: str, char_end: int,
                      encoding=None) -> tuple[int, bool]:
        """Tokens covering text[:char_end] (1-based count)."""
        kind, obj = self._resolve()
        if kind in ("hf", "tiktoken"):
            enc = encoding if encoding is not None else self.encode_cached(text)
            if enc and enc.get("kind") == "hf":
                return bisect.bisect_left(enc["ends"], char_end) + 1, False
            if enc and enc.get("kind") == "tiktoken_bytes":
                byte_end = len(text[:char_end].encode("utf-8"))
                return bisect.bisect_left(enc["ends"], byte_end) + 1, False
            # exact but slow fallback (unknown tiktoken internals)
            return len(obj.encode_ordinary(text[:char_end])), False
        words = len(_WORD_RE.findall(text[:char_end]))
        return int(round(words * obj)), True


_TOKENIZERS: dict[str, ModelTokenizer] = {}


def get_tokenizer(spec: str, cache_dir: Path | None = None) -> ModelTokenizer:
    key = spec or ""
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = ModelTokenizer(spec, cache_dir=cache_dir)
    return _TOKENIZERS[key]


# --------------------------------------------------------------------------
# spaCy POS tagging (dedicated scan phase; unique phrases only)
# --------------------------------------------------------------------------
_SPACY_MODELS = {"en": "en_core_web_sm", "fr": "fr_core_news_sm",
                 "de": "de_core_news_sm", "es": "es_core_news_sm",
                 "it": "it_core_news_sm", "pt": "pt_core_news_sm"}


def pos_available(lang: str | None) -> bool:
    name = _SPACY_MODELS.get((lang or "en")[:2])
    if not name:
        return False
    return importlib.util.find_spec(name) is not None


@lru_cache(maxsize=8)
def _spacy_tagger(lang: str | None):
    name = _SPACY_MODELS.get((lang or "en")[:2])
    if not name:
        return None
    try:
        import spacy
        # keep attribute_ruler: sm models map coarse pos_ through it
        return spacy.load(name, disable=["parser", "ner", "lemmatizer"])
    except Exception:
        return None


def pos_patterns(phrases: list[str], lang: str | None,
                 batch_size: int = 4096) -> list[str | None]:
    nlp = _spacy_tagger(lang)
    if nlp is None:
        return [None] * len(phrases)
    out: list[str | None] = []
    for doc in nlp.pipe(phrases, batch_size=batch_size):
        tags = [t.pos_ for t in doc if t.pos_ and not (t.is_space or t.is_punct)]
        out.append(" ".join(tags) or None)
    return out
