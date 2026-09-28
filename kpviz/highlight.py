"""Where a gold keyphrase occurs in a document's text — for the reader.

The document inspector marks every *present* (P) gold keyphrase in the
section text, found exactly as the scorer finds it: the same normalisation,
tokenizer and stemmer, contiguous stems in order, never across a separating
punctuation mark. Reordered, mixed and unseen keyphrases are not marked:
they do not occur as a phrase, and a highlight would pretend they do.
"""
from __future__ import annotations

from . import textproc as tp


def present_spans(text: str, phrases: list[str],
                  lang: str | None) -> list[tuple[int, int, list[int]]]:
    """Non-overlapping (start, end, [phrase indexes]) character spans of
    `text` where one of `phrases` occurs contiguously after stemming.

    Overlapping or touching occurrences are merged into one span that
    lists every phrase it covers."""
    if not text or not phrases:
        return []
    lowered, offmap = tp.norm_with_offsets(text)
    toks, ends = tp.tokens_with_ends(lowered)
    if not toks:
        return []
    stems = tp.get_stemmer(lang).stemWords(toks)
    starts = [e - len(t) for t, e in zip(toks, ends)]

    def orig(pos: int) -> int:
        return pos if offmap is None else offmap[pos]

    cache = tp.PhraseCache(max_size=4096)
    hits: list[tuple[int, int, int]] = []
    n = len(stems)
    for pi, phrase in enumerate(phrases):
        variants = []
        for v in tp.split_variants(str(phrase)):
            e = cache.analyze(v, lang, persist=False)
            if e["stems"]:
                variants.append(e["stems"])
            if e.get("pstems"):
                variants.append(e["pstems"])
        for pst in variants:
            m = len(pst)
            if not m or m > n:
                continue
            first = pst[0]
            for i in range(n - m + 1):
                if stems[i] == first and stems[i:i + m] == pst:
                    hits.append((orig(starts[i]), orig(ends[i + m - 1]), pi))
    if not hits:
        return []
    hits.sort()
    merged: list[list] = []
    for a, b, pi in hits:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            if pi not in merged[-1][2]:
                merged[-1][2].append(pi)
        else:
            merged.append([a, b, [pi]])
    return [(a, b, idx) for a, b, idx in merged]
