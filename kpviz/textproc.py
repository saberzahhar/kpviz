"""Text processing, throughput edition.

One tokenizer for everything that is matched — documents, gold keyphrases
and predictions: a Unicode-aware regular expression over NFKC-normalised,
lowercased text. Stems come from the Snowball stemmers (PyStemmer, C).
PRMU follows Boudin & Gallina (2021), computed on those stemmed tokens:

    P — the keyphrase's stemmed tokens occur in the stemmed document as a
        contiguous sequence, in order (within one section, not across a
        separating punctuation mark)
    R — every stemmed token occurs in the document, but never as that sequence
    M — some do
    U — none do

Tokens are maximal runs of letters, digits and combining marks (so Arabic
with harakat, Devanagari with vowel signs, and accented Latin stay whole);
Chinese and Japanese characters are one token each. Punctuation is dropped,
but a *separating* mark — one with whitespace (or a text boundary) beside
it — leaves a sentinel a contiguous match cannot cross; word-internal marks
("e-commerce", "and/or", "l'apprentissage") do not. A keyphrase carries the
same sentinels, so "U.S. army" still finds "U.S. army".

The documents phase used spaCy's blank tokenizers until revision 6: the
same token classes, five times slower, and the tokenizer benchmarked in the
paper's Table 2 was already this one.

Model tokenizers (`transformers[...]`, `tiktoken[...]`) resolve exactly
when their backend + assets are available and fall back to a flagged
word-ratio heuristic otherwise. Nothing in this module raises past it.
"""
from __future__ import annotations

import bisect
from collections import Counter
import importlib.util
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
# C1 control characters in real text are almost always Windows-1252 bytes
# decoded as Latin-1 ("d\x92analyse" for "d’analyse"): mapped back to the
# characters they were. Other control characters become spaces. One
# character each, so offsets are preserved.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_CP1252 = {}
for _b in range(0x80, 0xA0):
    try:
        _CP1252[_b] = bytes([_b]).decode("cp1252")
    except UnicodeDecodeError:
        _CP1252[_b] = " "
_CTRL_TABLE = {**{c: " " for c in range(0x20) if c not in (9, 10, 13)},
               0x7F: " ", **_CP1252}


def fix_text(text: str) -> str:
    """Repair mojibake control characters (C1 → Windows-1252, others →
    space); length-preserving, a no-op on clean text."""
    if text and _CTRL_RE.search(text):
        return text.translate(_CTRL_TABLE)
    return text or ""


def norm_text(text: str) -> str:
    """Repaired, NFKC, then str.lower() — deliberately not casefold():
    casefold maps German "ß" to "ss" and so would match "Strasse" against
    "Straße" in German but also change stems the Snowball stemmers expect;
    lower() keeps each language's own orthography. Stated in the
    conventions."""
    return unicodedata.normalize("NFKC", fix_text(text)).lower()


def norm_phrase(text: str) -> str:
    """Canonical keyphrase identity: NFKC, lowercased, whitespace-collapsed."""
    return " ".join(norm_text(text).split())


_VARIANT_SEP = re.compile(r"(?<=[^+\s])\+(?=[^+\s])")


def split_variants(raw: str) -> list[str]:
    """Gold keyphrases may carry alternate forms joined by '+' (the SemEval
    convention). Only a lone '+' between two other characters separates:
    "C++" and "A+ grading" stay whole."""
    raw = raw or ""
    if "+" not in raw:
        return [raw]
    parts = [p.strip() for p in _VARIANT_SEP.split(raw)]
    return [p for p in parts if p] or [raw]


# --------------------------------------------------------------------------
# Tokenisation (one regex tokenizer for documents, gold and predictions)
# --------------------------------------------------------------------------
SENT = "\x01"          # a separating mark: no contiguous match crosses it

# The common case — no combining mark, no CJK character — uses plain
# classes the regex engine checks in C; the Unicode-complete patterns
# (combining marks as word characters, one token per CJK character) are
# built once, on first need, and used only for texts that need them.
_FAST_TOK = re.compile(r"\x01|[^\W_]+")
_FAST_SEP = re.compile(r"(?:\s|\A)[^\w\s\x01]+|[^\w\s\x01]+(?:\s|\Z)")
_WORD_RE = re.compile(r"[^\W_]+")          # plain words (no sentinels)
_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿"
_CJK_RE = re.compile(f"[{_CJK}]")


@lru_cache(maxsize=1)
def _complex():
    """(mark-deletion table, token regex, separator regex, word regex) for
    text with combining marks or CJK characters."""
    ranges, start = [], None
    for cp in range(0x10000):                  # marks outside the BMP are
        is_m = unicodedata.category(chr(cp))[0] == "M"   # vanishingly rare
        if is_m and start is None:
            start = cp
        elif not is_m and start is not None:
            ranges.append((start, cp - 1))
            start = None
    marks = "".join(f"\\u{a:04x}" if a == b else f"\\u{a:04x}-\\u{b:04x}"
                    for a, b in ranges)
    delete = {cp: None for a, b in ranges for cp in range(a, b + 1)}
    word = rf"[{_CJK}]|(?:[^\W_{_CJK}]|[{marks}])+"
    p = rf"[^\w\s\x01{marks}]+"
    sep = (rf"(?:\s|\A){p}|{p}(?:\s|\Z)|{p}(?=[{_CJK}])|(?<=[{_CJK}]){p}")
    return (delete, re.compile(rf"\x01|{word}"), re.compile(sep),
            re.compile(word))


def _patterns(text: str):
    """(token regex, separator regex, word regex) suited to `text`."""
    if text.isascii():
        return _FAST_TOK, _FAST_SEP, _WORD_RE
    delete, tok, sep, word = _complex()
    if len(text.translate(delete)) != len(text) or _CJK_RE.search(text):
        return tok, sep, word
    return _FAST_TOK, _FAST_SEP, _WORD_RE


def tokens(lowered: str) -> list[str]:
    """Word tokens of normalised text, with SENT for each separating mark."""
    tok, sep, _w = _patterns(lowered)
    return tok.findall(sep.sub(" \x01 ", lowered))


def tokens_with_ends(lowered: str) -> tuple[list[str], list[int]]:
    """`tokens` plus the end offset (in `lowered`) of every token — the
    separators are replaced length-for-length, so offsets are preserved."""
    tok, sep, _w = _patterns(lowered)
    marked = sep.sub(lambda m: SENT + " " * (len(m.group()) - 1), lowered)
    toks, ends = [], []
    for m in tok.finditer(marked):
        toks.append(m.group())
        ends.append(m.end())
    return toks, ends


def tokenize(text: str) -> list[str]:
    """Plain word tokens of raw text (normalised; punctuation dropped)."""
    low = norm_text(text)
    return _patterns(low)[2].findall(low)


def phrase_tokens(norm: str) -> tuple[list[str], list[str]]:
    """(words, words with separator sentinels) of a normalised keyphrase."""
    toks = tokens(norm)
    if SENT not in toks:
        return toks, toks
    words = [t for t in toks if t != SENT]
    return words, toks


# --------------------------------------------------------------------------
# Stemming: PyStemmer (C) -> snowballstemmer -> identity, every Snowball
# language KPViz can name by its ISO 639-1 code
# --------------------------------------------------------------------------
_SNOWBALL_LANG = {
    "ar": "arabic", "hy": "armenian", "eu": "basque", "ca": "catalan",
    "da": "danish", "nl": "dutch", "en": "english", "fi": "finnish",
    "fr": "french", "de": "german", "el": "greek", "hi": "hindi",
    "hu": "hungarian", "id": "indonesian", "ga": "irish", "it": "italian",
    "lt": "lithuanian", "ne": "nepali", "no": "norwegian", "nb": "norwegian",
    "nn": "norwegian", "pt": "portuguese", "ro": "romanian", "ru": "russian",
    "sr": "serbian", "es": "spanish", "sv": "swedish", "ta": "tamil",
    "tr": "turkish", "yi": "yiddish",
}


class _IdentityStemmer:
    def stemWords(self, tokens):
        return list(tokens)


@lru_cache(maxsize=32)
def get_stemmer(lang: str | None):
    """The language's Snowball stemmer; identity when there is none. The C
    stemmer keeps a word cache — sized for a corpus vocabulary, not the
    default 10 000 words (a 20 % faster documents phase)."""
    name = _SNOWBALL_LANG.get((lang or "en")[:2])
    if not name:
        return _IdentityStemmer()
    try:
        import Stemmer  # PyStemmer — C bindings
        return Stemmer.Stemmer(name, 300_000)
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
    """Stemmed representation of a short phrase."""
    return " ".join(stem_tokens(phrase_tokens(norm_phrase(text))[0], lang))


# --------------------------------------------------------------------------
# Offsets between original and normalised text
# --------------------------------------------------------------------------
def _offsets_shared(text: str) -> bool:
    """True when normalising cannot move any offset: ASCII, or already NFKC
    with a length-preserving lowercase (both checks run in C)."""
    return text.isascii() or (unicodedata.is_normalized("NFKC", text)
                              and len(text.lower()) == len(text))


def norm_with_offsets(text: str) -> tuple[str, list[int] | None]:
    """(normalised text, offset map) with an *exact* alignment.

    The map m has m[j] = offset in `text` of normalised offset j (None when
    offsets are shared — the common case, decided in C). Otherwise the text
    is normalised cluster by cluster — a base character with the combining
    marks that follow it — and the pieces concatenated, so every boundary
    is known: composition ("e" + U+0301 -> "é"), compatibility expansion
    ("ﬁ" -> "fi") and changes that cancel out in total length all map
    exactly. A document normalised this way differs from whole-string NFKC
    only where a composition would span two clusters (not a case in scripts
    KPViz stems)."""
    text = fix_text(text)
    if _offsets_shared(text):
        return norm_text(text), None
    out, m = [], [0]
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        piece = norm_text(text[i:j])
        out.append(piece)
        # every normalised char of the cluster maps to the cluster's end
        m.extend([j] * len(piece))
        i = j
    return "".join(out), m


def norm_offsets(text: str, lowered: str) -> list[int] | None:
    """Map offsets in `lowered` back to `text` (see norm_with_offsets, which
    derives both together; kept for callers that already normalised)."""
    low2, m = norm_with_offsets(text)
    if m is None:
        return None
    if low2 != lowered:
        raise ValueError("norm_offsets: text was normalised differently; "
                         "use norm_with_offsets")
    return m


# --------------------------------------------------------------------------
# Keyphrase analysis cache (worker-local; persisted in the keyphrases table)
# --------------------------------------------------------------------------
class PhraseCache:
    """Analyse each unique (language, normalised phrase) once per worker.

    Keyed by language as well as text: the stemmer is per language, so the
    same string has different stems in two languages.

    Memory-bounded without cliffs: two generations. When the current one is
    full it becomes the old one; a hit in the old generation is promoted, so
    hot phrases survive every rollover and only cold ones are dropped.
    `persist` marks phrases that belong in the global `keyphrases` table
    (gold phrases — the only ones the UI shows or POS-tags)."""

    __slots__ = ("_new", "_old", "_raw", "fresh", "_persisted", "max_size",
                 "hits", "misses")

    def __init__(self, max_size: int = 300_000):
        # (lang, raw string) -> entry: a repeated prediction string skips
        # NFKC normalisation entirely (the dominant cost on cache hits)
        self._raw: dict[tuple, dict] = {}
        self._new: dict[tuple, dict] = {}
        self._old: dict[tuple, dict] = {}
        self.fresh: dict[tuple, dict] = {}    # to persist at next drain
        self._persisted: set[tuple] = set()   # already drained this process
        self.max_size = max(2, int(max_size))
        self.hits = 0
        self.misses = 0

    def analyze(self, raw: str, lang: str | None, persist: bool = True) -> dict:
        lang2 = (lang or "en")[:2]
        if not persist:
            hit = self._raw.get((lang2, raw))
            if hit is not None:
                self.hits += 1
                return hit
        norm = norm_phrase(raw)
        key = (lang2, norm)
        entry = self._new.get(key)
        if entry is None:
            entry = self._old.pop(key, None)
            if entry is None:
                self.misses += 1
                words, marked = phrase_tokens(norm)
                stemmer = get_stemmer(lang2)
                stems = stemmer.stemWords(words)
                entry = {"kp": norm, "raw": " ".join(fix_text(str(raw)).split()),
                         "lang": lang2, "tokens": words, "stems": stems,
                         "sstr": " ".join(stems), "n_tokens": len(words),
                         # the same stems with the keyphrase's own separating
                         # marks, for PRMU ("U.S. army" in "U.S. army")
                         "pstems": (stemmer.stemWords(marked)
                                    if marked is not words else None)}
            else:
                self.hits += 1
            if len(self._new) >= self.max_size // 2:     # generation rollover
                self._old, self._new = self._new, {}
            self._new[key] = entry
        else:
            self.hits += 1
        if persist:
            if key not in self._persisted and key not in self.fresh:
                self.fresh[key] = entry
        else:
            if len(self._raw) >= self.max_size // 2:
                self._raw = {}
            self._raw[(lang2, raw)] = entry
        return entry

    def drain(self) -> list[dict]:
        """Phrases to persist since the last drain (kp, raw, lang only)."""
        out = [{"kp": e["kp"], "raw": e["raw"], "lang": e["lang"]}
               for e in self.fresh.values()]
        self._persisted.update(self.fresh)
        if len(self._persisted) > 2_000_000:           # RAM bound (set of keys)
            self._persisted = set(list(self._persisted)[-1_000_000:])
        self.fresh = {}
        return out

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "size": len(self._new) + len(self._old), "raw": len(self._raw)}


# --------------------------------------------------------------------------
# PRMU on stemmed tokens (Boudin & Gallina, 2021)
# --------------------------------------------------------------------------

class StemmedDoc:
    """A document as the PRMU classifier sees it: its stemmed tokens (with
    SENT at every section boundary and separating mark) as one padded
    string, searched in C, and as a set.

    A contiguous occurrence is a substring " a b c " of the padded string;
    a sentinel between two stems makes that substring impossible, which is
    how a match is kept inside one section and off separating punctuation."""

    __slots__ = ("stems", "pad", "set")

    def __init__(self, stems: list[str]):
        self.stems = stems
        self.pad = " " + " ".join(stems) + " "
        self.set = set(stems)


def contiguous_end(kp_stems: list[str], doc: StemmedDoc) -> int:
    """Word position (sentinels not counted) where the earliest contiguous,
    in-order occurrence of kp_stems ends, else -1."""
    needle = " " + " ".join(kp_stems) + " "
    pad = doc.pad
    pos = pad.find(needle)
    if pos < 0:
        return -1
    end = pos + len(needle) - 1                 # the space after the match
    last = pad.count(" ", 0, end) - 1           # token index of its last stem
    return last - pad.count(SENT, 0, end)       # minus the sentinels before


_PRMU_RANK = {"P": 3, "R": 2, "M": 1, "U": 0}


def prmu_classify(variant_stems: list[list[str]], doc: StemmedDoc,
                  marked: list[list[str]] | None = None) -> tuple[str, int]:
    """Best PRMU class over the '+'-variants, and where the earliest present
    occurrence ends (word position; -1 unless P). `marked` holds variants
    with their own separating marks (tried too for P).

    P: the stemmed tokens occur contiguously, in order, in the stemmed
       document · R: every stemmed token occurs somewhere, never as that
       sequence · M: some do · U: none do."""
    best, best_end = "U", -1
    rank = _PRMU_RANK
    present_set = doc.set
    for stems in list(variant_stems) + list(marked or ()):
        if not stems:
            continue
        end = contiguous_end(stems, doc)
        if end >= 0:
            if best != "P" or end < best_end:
                best, best_end = "P", end
            continue
        if best == "P":
            continue
        distinct = set(stems)
        distinct.discard(SENT)
        present = len(distinct & present_set)
        cat = "R" if present == len(distinct) else ("M" if present else "U")
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
    "nl": frozenset("de en van het een in is dat op te zijn voor met die niet aan er om ook als bij door wordt worden maar dan of uit naar deze kan zij tot".split()),
    "ru": frozenset("и в не на что с по как это из за от для к о но же его а то все так было он мы она они при бы также или".split()),
    "ar": frozenset("في من على إلى أن عن مع هذا هذه التي الذي كان التى ما لا هو هي ثم أو قد كل بين بعد عند إن لم ذلك تلك منذ حتى كما أيضا وفي ومن".split()),
    "sv": frozenset("och att det som en på är av för med till den har de inte om ett han men var jag sig från vi så kan man när år".split()),
    "tr": frozenset("ve bir bu da de için ile olarak çok daha gibi olan ama en kadar sonra ne her mi göre ise ya veya şu".split()),
    "pl": frozenset("i w na z się nie do to że jest o jak ale po co tak za od przez przy dla jego oraz lub są być".split()),
}
# token -> languages whose stop-word list contains it, built once: one pass
# over the tokens instead of one generator per candidate language
_STOP_INDEX: dict[str, tuple[str, ...]] = {}
for _l, _ws in _STOPWORDS.items():
    for _w in _ws:
        _STOP_INDEX[_w] = _STOP_INDEX.get(_w, ()) + (_l,)


def language_scores(toks: list[str], n: int | None = None) -> dict[str, float]:
    """Share of the tokens that are each language's stop words."""
    n = n if n is not None else len(toks)
    if not n:
        return {}
    counts: dict[str, int] = {}
    # keep only stop words (a C-level filter), count each distinct one once,
    # then credit its languages
    for w, k in Counter(filter(_STOP_INDEX.__contains__, toks)).items():
        for l in _STOP_INDEX[w]:
            counts[l] = counts.get(l, 0) + k
    return {l: c / n for l, c in counts.items()}


def detect_language(text: str, candidates: list[str] | None = None,
                    min_tokens: int = 5,
                    tokens: list[str] | None = None,
                    n: int | None = None) -> tuple[str | None, float]:
    """Stopword vote over the *whole* text (no prefix sampling). Pass
    `tokens` (and their word count `n`) to reuse a tokenisation the caller
    already has."""
    toks = tokens if tokens is not None else tokenize(text or "")
    n = n if n is not None else len(toks)
    if n < min_tokens:
        return None, 0.0
    scores = language_scores(toks, n)
    if candidates:
        scores = {l: v for l, v in scores.items() if l in candidates}
    if not scores:
        return None, 0.0
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    if scores[best] < 0.08:
        return None, scores[best]
    return best, round(min(1.0, scores[best] * 2 + margin), 3)


def best_language(scores: dict[str, float]) -> str | None:
    """The stop-word language of `language_scores`, or None when no list
    reaches 8 % of the tokens (too short, or a language without a list)."""
    if not scores:
        return None
    best = max(scores, key=scores.get)
    return best if scores[best] >= 0.08 else None


def contradicts(scores: dict[str, float], declared: list[str]) -> bool:
    """Does the text's language contradict every declared one?

    Only when the best stop-word language is not declared and no declared
    language comes close to it (at least half its share): closely related
    languages sharing function words (Danish/Norwegian, Spanish/Portuguese)
    raise no false alarm, and a declared language without a stop-word list
    is never contradicted."""
    best = best_language(scores)
    if best is None or not declared:
        return False
    decl = [l[:2] for l in declared]
    if best in decl or not any(l in _STOPWORDS for l in decl):
        return False
    return max((scores.get(l, 0.0) for l in decl), default=0.0) < 0.5 * scores[best]


# --------------------------------------------------------------------------
# Model tokenizer registry — exact when possible, flagged-approx otherwise
# --------------------------------------------------------------------------
_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*\[\s*([^\]]+)\s*\]\s*$")
# short names cards use -> Hugging Face repositories holding the tokenizer
# (tried in order; Llama 3.x share one tokenizer, the Meta repositories are
# gated and need HF_TOKEN, the mirror after them is not)
_LLAMA3 = ("meta-llama/Llama-3.3-70B-Instruct", "unsloth/Llama-3.3-70B-Instruct")
_HF_ALIASES = {"bart-base": ("facebook/bart-base",),
               "bart-large": ("facebook/bart-large",),
               **{k: _LLAMA3 for k in ("llama3", "llama-3", "llama-3.1",
                                       "llama-3.2", "llama-3.3", "llama3.3")}}


def _hf_error(e: Exception, name: str) -> str:
    """One line a user can act on (no request ids, no tracebacks)."""
    kind, msg = type(e).__name__, str(e)
    if kind == "GatedRepoError" or "gated" in msg.lower():
        return (f"'{name}' is gated on Hugging Face: accept its licence on the "
                "model page and set HF_TOKEN")
    if (kind == "RepositoryNotFoundError" or "401" in msg or "404" in msg
            or "not found" in msg.lower()):
        return (f"no Hugging Face repository '{name}' (or it is private): "
                "declare transformers[<owner>/<repo>] (with HF_TOKEN if gated) "
                "or transformers[file:<path>/tokenizer.json]")
    if kind in ("ProxyError", "ConnectError", "ConnectionError", "ConnectTimeout",
                "Timeout", "ReadTimeout") or "403" in msg:
        return "Hugging Face is unreachable from this machine (network or proxy)"
    return f"{kind}: {msg.splitlines()[0][:120]}" if msg else kind
_FALLBACK_RATIO = {"transformers": 1.30, "tiktoken": 1.25, None: 1.30}


def tokenizer_inner(spec: str) -> str:
    """'tiktoken[o200k_base]' -> 'o200k_base' (display helper)."""
    m = _SPEC_RE.match(spec or "")
    return m.group(2) if m else (spec or "")


_NEG_TTL_S = 24 * 3600          # retry an unavailable asset once a day


def offline() -> bool:
    """True when the user asked for no network (demo halls, air-gapped HPC)."""
    return any(os.environ.get(v, "").lower() in ("1", "true", "yes")
               for v in ("KPVIZ_OFFLINE", "HF_HUB_OFFLINE",
                         "TRANSFORMERS_OFFLINE"))


class ModelTokenizer:
    """A model tokenizer, exact when its asset is available, otherwise a
    flagged word-ratio approximation.

    Resolution happens once per scan in the parent (`allow_network=True`),
    which caches assets under the state directory and records unavailable
    ones in a `.unavailable` marker (retried after a day, or on request).
    Workers are built with `allow_network=False` and the parent's verdict
    (`expect`), so they only ever read local files: N workers no longer each
    wait on DNS for the same missing asset."""

    def __init__(self, spec: str, cache_dir: Path | None = None,
                 allow_network: bool = True, expect: str | None = None):
        self.spec = spec
        m = _SPEC_RE.match(spec or "")
        self.backend = m.group(1).lower() if m else None
        self.name = m.group(2) if m else (spec or "")
        self.cache_dir = cache_dir
        self.allow_network = allow_network and not offline()
        self.expect = expect
        self.why = ""
        self._impl = None
        self._fp = None

    # -- asset locations -----------------------------------------------------
    def _dir(self) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / self.name.replace("/", "__")

    def _marker(self) -> Path | None:
        d = self._dir()
        return d / ".unavailable" if d else None

    def _known_unavailable(self) -> bool:
        mk = self._marker()
        try:
            return bool(mk and mk.exists()
                        and time.time() - mk.stat().st_mtime < _NEG_TTL_S)
        except OSError:
            return False

    def _mark_unavailable(self, why: str) -> None:
        mk = self._marker()
        if not mk:
            return
        try:
            mk.parent.mkdir(parents=True, exist_ok=True)
            mk.write_text(why[:500], encoding="utf-8")
        except OSError:
            pass

    # -- resolution ------------------------------------------------------------
    def _resolve(self):
        if self._impl is not None:
            return self._impl
        impl = None
        if self.expect != "approx":
            if self.backend == "transformers":
                impl = self._try_hf()
            elif self.backend == "tiktoken":
                # the Llama 3 tokenizer is tiktoken-*style* but not shipped
                # with tiktoken: it is published as a tokenizer.json
                impl = (self._try_hf() if self.name.lower() in _HF_ALIASES
                        else self._try_tiktoken())
            elif self.backend:
                self.why = f"unknown tokenizer backend {self.backend!r}"
        if impl is None:
            impl = ("approx", _FALLBACK_RATIO.get(self.backend, 1.30))
            self.why = self.why or "asset unavailable"
        self._impl = impl
        return impl

    def _try_hf(self):
        try:
            from tokenizers import Tokenizer
        except Exception:
            self.why = "the `tokenizers` package is not installed"
            return None
        d = self._dir()
        local = [d / "tokenizer.json"] if d else []
        # a local file path is also a valid spec: transformers[file:/x/tokenizer.json]
        if self.name.startswith("file:"):
            local.insert(0, Path(self.name[5:]))
        for f in local:
            if f.exists():
                try:
                    return ("hf", Tokenizer.from_file(str(f)))
                except Exception as e:
                    self.why = f"unreadable {f}: {str(e)[:120]}"
        if self.name.startswith("file:"):
            self.why = self.why or f"no file {self.name[5:]}"
            return None
        if not self.allow_network:
            src = _HF_ALIASES.get(self.name.lower(), (self.name,))[0]
            self.why = self.why or (
                ("offline" if offline() else "not cached locally")
                + f" — read from Hugging Face ({src}) when online")
            return None
        if self._known_unavailable():
            self.why = "unavailable at the last attempt (retried daily)"
            return None
        last = ""
        token = (os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)
        repos = list(_HF_ALIASES.get(self.name.lower(), ())) or [self.name]
        if "/" not in self.name and self.name.lower() not in _HF_ALIASES:
            repos.append(f"facebook/{self.name}")
        for repo in dict.fromkeys(repos):
            try:
                tok = Tokenizer.from_pretrained(repo, token=token)
            except TypeError:                   # tokenizers < 0.14
                try:
                    tok = Tokenizer.from_pretrained(repo)
                except Exception as e:
                    last = last or _hf_error(e, repo)
                    continue
            except Exception as e:
                # the first failure names the repository the user meant
                last = last or _hf_error(e, repo)
                continue
            if d:
                # atomic: a torn tokenizer.json would silently degrade every
                # later scan
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    tmp = d / f"tokenizer.json.{os.getpid()}.tmp"
                    tok.save(str(tmp))
                    os.replace(tmp, d / "tokenizer.json")
                except Exception:
                    pass
            return ("hf", tok)
        self.why = last or "download failed"
        self._mark_unavailable(self.why)
        return None

    def _try_tiktoken(self):
        try:
            import tiktoken
        except Exception:
            self.why = "the `tiktoken` package is not installed"
            return None
        if self.cache_dir and not os.environ.get("TIKTOKEN_CACHE_DIR"):
            os.environ["TIKTOKEN_CACHE_DIR"] = str(self.cache_dir / "tiktoken")
        try:
            known = set(tiktoken.list_encoding_names())
        except Exception:
            known = set()
        if known and self.name not in known:
            self.why = (f"tiktoken has no '{self.name}' encoding (it has "
                        f"{', '.join(sorted(known))}); declare "
                        "transformers[<owner>/<repo>] or "
                        "transformers[file:<path>/tokenizer.json]")
            return None
        if self.allow_network and self._known_unavailable():
            self.why = "unavailable at the last attempt (retried daily)"
            return None
        # local availability is separate from download permission: offline,
        # a cached encoding is read (tiktoken checks its cache before it
        # downloads) and the download itself is refused
        try:
            if self.allow_network:
                return ("tiktoken", tiktoken.get_encoding(self.name))
            import tiktoken.load as _tl
            real = _tl.read_file

            def no_download(blobpath):
                raise OSError(f"offline: {self.name} is not cached locally")
            _tl.read_file = no_download
            try:
                return ("tiktoken", tiktoken.get_encoding(self.name))
            finally:
                _tl.read_file = real
        except Exception as e:
            self.why = (f"not cached locally ({'offline' if offline() else 'no network in workers'})"
                        if isinstance(e, OSError) and "offline:" in str(e)
                        else f"{type(e).__name__}: {str(e)[:160]}")
            if self.allow_network:
                self._mark_unavailable(self.why)
            return None

    @property
    def exact(self) -> bool:
        return self._resolve()[0] != "approx"

    @property
    def status(self) -> str:
        """'exact' | 'approx' — what workers are told to expect."""
        return "exact" if self.exact else "approx"

    @property
    def fingerprint(self) -> str:
        """Identity of what counts tokens, for derivation signatures: the
        asset's *content* and the backend's version. A newly available exact
        asset, a tokenizer.json replaced at the same path, or a tokenizer
        library upgrade all re-derive the counts and positions."""
        kind, obj = self._resolve()
        if kind == "approx":
            return f"approx:{obj}"
        if self._fp is None:
            import hashlib
            h = hashlib.blake2b(digest_size=10)
            try:
                from importlib import metadata
                lib = "tokenizers" if kind == "hf" else "tiktoken"
                h.update(f"{lib}={metadata.version(lib)}".encode())
            except Exception:
                pass
            try:
                if kind == "hf":
                    h.update(obj.to_str().encode("utf-8"))
                else:
                    h.update(f"{obj.name}:{obj.n_vocab}".encode())
                    h.update(repr(sorted(obj._special_tokens.items())).encode())
                    h.update(str(obj._pat_str).encode())
            except Exception:
                h.update(self.name.encode())
            self._fp = f"exact:{kind}:{self.name}:{h.hexdigest()}"
        return self._fp

    # -- counting ------------------------------------------------------------
    def count_batch(self, texts: list[str]) -> tuple[list[int], bool]:
        kind, obj = self._resolve()
        if kind == "hf":
            # a count needs no offsets: the fast variant skips them (-25 %)
            enc = getattr(obj, "encode_batch_fast", obj.encode_batch)
            return [len(e.ids) for e in enc(texts)], False
        if kind == "tiktoken":
            return [len(ids) for ids in
                    obj.encode_ordinary_batch(texts)], False
        return [int(round(len(_WORD_RE.findall(t)) * obj)) for t in texts], True

    def count(self, text: str) -> tuple[int, bool]:
        n, approx = self.count_batch([text])
        return n[0], approx

    # -- per-document encoding, reused across every gold phrase -------------
    def encode_cached(self, text: str, word_starts: list[int] | None = None):
        """A reusable per-document encoding, so `char_to_token` is a bisect
        instead of a re-encode (or, when approximate, a re-scan of the prefix
        per keyphrase — quadratic on long documents).

        For tiktoken the ends are byte positions (tokens partition the UTF-8
        bytes exactly), which keeps the answer exact."""
        kind, obj = self._resolve()
        if kind == "hf":
            enc = obj.encode(text)
            # special tokens (BOS/EOS, separators) carry empty (0, 0)
            # offsets: they are counted in the position but never searched —
            # searching them made the ends unsorted and positions impossible
            ends, pos, top = [], [], 0
            for i, (a, b) in enumerate(enc.offsets):
                if b > a:
                    top = max(top, b)
                    ends.append(top)
                    pos.append(i + 1)
            return {"kind": "hf", "ends": ends, "pos": pos, "n": len(enc.ids)}
        if kind == "tiktoken":
            ids = obj.encode_ordinary(text)
            try:
                from itertools import accumulate
                ends = list(accumulate(map(len, obj.decode_tokens_bytes(ids))))
                # ASCII text: a char offset is a byte offset (no re-encode
                # of the prefix per keyphrase)
                return {"kind": "tiktoken_bytes", "ends": ends, "n": len(ids),
                        "ascii": text.isascii()}
            except Exception:
                return {"kind": "tiktoken_slow", "n": len(ids)}
        starts = (word_starts if word_starts is not None
                  else [mt.start() for mt in _WORD_RE.finditer(text)])
        return {"kind": "approx", "starts": starts}

    def char_to_token(self, text: str, char_end: int,
                      encoding=None) -> tuple[int, bool]:
        """Position (1-based, counting leading special tokens) of the token
        that contains the character just before `char_end` — i.e. how many
        tokens the model must read to have seen text[:char_end]. Boundary
        rule: a token ending exactly at char_end is that token (bisect_left
        on token ends); a token that straddles char_end counts, since the
        phrase is not complete before it."""
        kind, obj = self._resolve()
        enc = encoding if encoding is not None else self.encode_cached(text)
        if kind in ("hf", "tiktoken"):
            if enc and enc.get("kind") == "hf":
                ends, pos = enc["ends"], enc.get("pos")
                if not ends:
                    return enc.get("n", 0), False
                i = min(bisect.bisect_left(ends, char_end), len(ends) - 1)
                # 1-based position of that content token in the full encoding
                # (a leading BOS counts, as it does in the model's window)
                return (pos[i] if pos else i + 1), False
            if enc and enc.get("kind") == "tiktoken_bytes":
                byte_end = (char_end if enc.get("ascii")
                            else len(text[:char_end].encode("utf-8")))
                ends = enc["ends"]
                if not ends:
                    return 0, False
                return min(bisect.bisect_left(ends, byte_end), len(ends) - 1) + 1, False
            # exact but slow fallback (unknown tiktoken internals)
            return len(obj.encode_ordinary(text[:char_end])), False
        # regex words of text[:char_end] == words starting before char_end
        # (a word cut by the boundary still counts once, as findall did)
        words = bisect.bisect_left(enc["starts"], char_end)
        return int(round(words * obj)), True


_TOKENIZERS: dict[tuple, ModelTokenizer] = {}


def get_tokenizer(spec: str, cache_dir: Path | None = None,
                  allow_network: bool = True,
                  expect: str | None = None) -> ModelTokenizer:
    key = (spec or "", str(cache_dir or ""), allow_network, expect)
    tk = _TOKENIZERS.get(key)
    if tk is None:
        tk = _TOKENIZERS[key] = ModelTokenizer(spec, cache_dir=cache_dir,
                                               allow_network=allow_network,
                                               expect=expect)
    return tk


def forget_tokenizers(cache_dir: Path | None = None) -> None:
    """Drop resolved tokenizers and unavailability markers ("retry downloads")."""
    _TOKENIZERS.clear()
    if cache_dir and cache_dir.is_dir():
        for mk in cache_dir.glob("*/.unavailable"):
            try:
                mk.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------
# spaCy POS tagging (dedicated scan phase; unique phrases only)
# --------------------------------------------------------------------------
_SPACY_MODELS = {"en": "en_core_web_sm", "fr": "fr_core_news_sm",
                 "de": "de_core_news_sm", "es": "es_core_news_sm",
                 "it": "it_core_news_sm", "pt": "pt_core_news_sm",
                 "nl": "nl_core_news_sm", "ru": "ru_core_news_sm"}


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
