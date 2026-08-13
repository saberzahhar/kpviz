"""Worker-side derivations.

Per chunk of a document collection, one pass does everything except POS:
spaCy-blank tokens (C path), PyStemmer stems (one C call per document),
in-order PRMU via position indices, tokenizer counts batched in sub-batches
that release their text as they go, and a worker-local phrase cache so each
unique keyphrase is analysed once per process. POS tagging happens in its own
scan phase over globally unique untagged phrases.

Workers never touch DuckDB; they spill NDJSON the scanner COPYs. Everything
stays importable at module level (spawn/forkserver-safe).
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

try:
    import orjson as _fastjson

    def _loads(b):
        return _fastjson.loads(b)

    def _dumps(o) -> bytes:
        return _fastjson.dumps(o)

    JSON_BACKEND = "orjson"
except Exception:  # pragma: no cover - orjson is in requirements
    import json as _stdjson

    def _loads(b):
        return _stdjson.loads(b)

    def _dumps(o) -> bytes:
        return _stdjson.dumps(o, ensure_ascii=False, default=str).encode()

    JSON_BACKEND = "json"

import json

from . import textproc as tp

SECTION_JOIN = "\n\n"
TRAIN_SPLITS = {"train", "training"}
_TOK_SUBBATCH = 512          # documents per tokenizer encode batch


# ---------------------------------------------------------------------------
# Worker process initialisation
# ---------------------------------------------------------------------------
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")


def init_worker(phrase_cache_max: int = 300_000) -> None:
    """Pin every nested thread pool to one thread.

    HuggingFace `tokenizers` (rayon), tiktoken (rayon) and the BLAS under
    spaCy each size their internal pool to the whole machine. With one worker
    process per core that is cores² runnable threads — a context-switch storm
    that makes a 32-core box slower than a 4-core one. Process-level
    parallelism already saturates the machine; intra-worker threading can
    only take away. Must run before those libraries are imported."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _PHRASES
    _PHRASES = tp.PhraseCache(max_size=max(20_000, int(phrase_cache_max)))


# ---------------------------------------------------------------------------
# Worker-local state (persists across chunks and phases in the same process)
# ---------------------------------------------------------------------------
_PHRASES = tp.PhraseCache()
_MATCH_STEMS: dict[tuple[str, str], str] = {}   # (lang2, norm) -> stem string
_MATCH_STEMS_MAX = 600_000


def _match_stems(entry: dict, lang: str | None) -> str:
    """Language-consistent stem string for matching (memoised)."""
    lang2 = (lang or "en")[:2]
    key = (lang2, entry["kp"])
    hit = _MATCH_STEMS.get(key)
    if hit is None:
        if lang2 == entry["lang"]:
            hit = " ".join(entry["stems"])
        else:
            hit = " ".join(tp.get_stemmer(lang2).stemWords(entry["tokens"]))
        if len(_MATCH_STEMS) > _MATCH_STEMS_MAX:
            _MATCH_STEMS.clear()
        _MATCH_STEMS[key] = hit
    return hit


def _write_ndjson(path: Path, rows: list[dict]) -> str | None:
    """One buffered write instead of two syscalls per row."""
    if not rows:
        return None
    buf = io.BytesIO()
    for r in rows:
        buf.write(_dumps(r))
        buf.write(b"\n")
    with open(path, "wb") as f:
        f.write(buf.getbuffer())
    return str(path)


def _iter_lines(path: str, start: int, end: int):
    """Whole lines in [start, end). Byte ranges are newline-aligned by the
    planner, so no document is ever split."""
    with open(path, "rb") as f:
        f.seek(start)
        pos = start
        while pos < end:
            line = f.readline()
            if not line:
                break
            ln = len(line)
            s = line.strip()
            if s:
                try:
                    yield pos, ln, _loads(s)
                except Exception:
                    yield pos, ln, None
            pos += ln


def is_eval_split(split: str | None) -> bool:
    return (split or "").lower() not in TRAIN_SPLITS


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def derive_doc_chunk(args: dict) -> dict:
    """One byte-range of document.{ds}.jsonl -> spill files."""
    _t0 = time.perf_counter()
    ds = args["dataset"]
    card_sections: dict = args["card"].get("sections", {})
    card_anns: dict = args["card"].get("anns", {})
    emit_combined = args["card"].get("combined", False)
    token_scope = args.get("token_scope", "eval")
    gold_scope = args.get("gold_scope", "eval")
    cache_dir = Path(args["tok_cache"]) if args.get("tok_cache") else None
    toks = [tp.get_tokenizer(s, cache_dir) for s in args.get("tokenizers", [])]

    doc_rows: list[dict] = []
    gold_rows: list[dict] = []
    tok_rows: list[dict] = []
    tokpos_rows: list[dict] = []
    pending: list[tuple[str, str, list]] = []   # (doc_id, text, P-gold refs)
    agg: dict[tuple, list] = {}
    n_docs = 0

    def flush_tokens():
        """Tokenizer counts + gold token positions for the pending documents,
        then release their text (a chunk's worth of text held twice is the
        one place worker RSS could grow without bound)."""
        if not toks or not pending:
            pending.clear()
            return
        texts = [t for _i, t, _g in pending]
        for tk in toks:
            counts, approx = tk.count_batch(texts)
            for (doc_id, text, gold_p), n in zip(pending, counts):
                tok_rows.append({"dataset": ds, "doc_id": doc_id,
                                 "tokenizer": tk.spec, "n_tokens": n,
                                 "approx": approx})
                if not gold_p:
                    continue
                enc = tk.encode_cached(text)
                for g in gold_p:
                    te, ap = tk.char_to_token(text, g["first_char"], enc)
                    tokpos_rows.append({"dataset": ds, "doc_id": doc_id,
                                        "ann_key": g["ann_key"],
                                        "kp_idx": g["kp_idx"],
                                        "tokenizer": tk.spec,
                                        "tok_end": te, "approx": ap})
        pending.clear()

    for off, ln, obj in _iter_lines(args["path"], args["start"], args["end"]):
        if not isinstance(obj, dict):
            continue
        doc_id = str(obj.get("_id", f"@{off}"))
        meta = obj.get("metadata") or {}
        split = meta.get("split")
        eval_doc = is_eval_split(split)

        sections = []
        for s in (obj.get("sections") or []):
            fieldname = s.get("field") or "?"
            declared = s.get("language") or card_sections.get(fieldname) or []
            sections.append((fieldname, declared, s.get("content") or ""))
        full_text = SECTION_JOIN.join(c for _f, _l, c in sections)

        flags: list[str] = []
        sec_meta, detected, declared_union = [], [], []
        for fieldname, declared, content in sections:
            words = tp.tokenize(content)          # full text, never sampled
            det, _conf = tp.detect_language(content, tokens=words)
            detected.append(det)
            if det and declared and det not in [l[:2] for l in declared]:
                flags.append(f"lang_mismatch:section:{fieldname}")
            sec_meta.append({"field": fieldname, "langs": declared, "det": det,
                             "chars": len(content), "words": len(words)})
            for l in declared:
                if l not in declared_union:
                    declared_union.append(l)
        for fieldname in card_sections:
            if fieldname not in [s[0] for s in sections]:
                flags.append(f"missing_section:{fieldname}")

        streams: dict[str, tuple] = {}

        def stream(lang: str | None):
            lang2 = (lang or "en")[:2]
            got = streams.get(lang2)
            if got is None:
                words, ends = tp.spacy_doc_tokens(full_text, lang2)
                stems = tp.get_stemmer(lang2).stemWords(words)
                got = (tp.position_index(stems), ends)
                streams[lang2] = got
            return got

        anns = obj.get("annotations") or []
        ann_counts: dict[str, int] = {}
        combined: list[dict] = []
        combined_seen: set = set()
        doc_gold_p: list[dict] = []

        # annotation-level language flags first, so `keep_gold` sees every
        # quality issue (flagged documents always keep their gold instances)
        for a in anns:
            key = a.get("annotator") or "annotation"
            langs = a.get("language") or card_anns.get(key) or declared_union
            kps = a.get("keyphrases") or []
            ann_counts[key] = len(kps)
            if kps:
                joined = " ".join(str(k).split("+")[0] for k in kps)
                det, _ = tp.detect_language(joined, min_tokens=6)
                if det and langs and det not in [l[:2] for l in langs]:
                    flags.append(f"lang_mismatch:ann:{key}")
        keep_gold = (gold_scope == "all") or eval_doc or bool(flags)

        for a in anns:
            key = a.get("annotator") or "annotation"
            langs = a.get("language") or card_anns.get(key) or declared_union
            lang = (langs[0] if langs else None)
            kps = a.get("keyphrases") or []
            index, char_ends = stream(lang)
            lang2 = (lang or "en")[:2]
            stemmer = tp.get_stemmer(lang2)

            def agg_add(ann: str, cat: str, nw: int):
                k4 = (split, ann, cat, min(nw, 6) if nw else 0)
                slot = agg.get(k4)
                if slot is None:
                    agg[k4] = [1, nw]
                else:
                    slot[0] += 1
                    slot[1] += nw

            for i, kp in enumerate(kps):
                variants = tp.split_variants(str(kp))
                entries = [_PHRASES.analyze(v, lang, persist=keep_gold)
                           for v in variants]
                entries = [e for e in entries if e["tokens"]]
                var_stems = [stemmer.stemWords(e["tokens"]) for e in entries]
                cat, end_word = tp.prmu_classify(var_stems, index)
                first_char = char_ends[end_word] if end_word >= 0 else -1
                nw = entries[0]["n_tokens"] if entries else 0
                agg_add(key, cat, nw)
                stems_repr = [" ".join(vs) for vs in var_stems]
                row = None
                if keep_gold:
                    row = {"dataset": ds, "doc_id": doc_id, "ann_key": key,
                           "kp_idx": i,
                           "display": entries[0]["kp"] if entries else "",
                           "stems": stems_repr,
                           "lang": lang, "n_words": nw, "prmu": cat,
                           "first_char": first_char, "first_word": end_word}
                    gold_rows.append(row)
                    if cat == "P":
                        doc_gold_p.append(row)
                if emit_combined:
                    sig = frozenset(stems_repr)
                    if sig and sig not in combined_seen:
                        combined_seen.add(sig)
                        agg_add("@combined", cat, nw)
                        if keep_gold and row is not None:
                            crow = dict(row, ann_key="@combined",
                                        kp_idx=len(combined))
                            combined.append(crow)
                            if cat == "P":
                                doc_gold_p.append(crow)
        gold_rows.extend(combined)

        if toks and (token_scope == "all" or eval_doc):
            pending.append((doc_id, full_text, doc_gold_p))
            if len(pending) >= _TOK_SUBBATCH:
                flush_tokens()

        doc_rows.append({
            "dataset": ds, "doc_id": doc_id, "split": split,
            "file_id": args["file_id"], "byte_off": off, "byte_len": ln,
            "n_sections": len(sections),
            "n_chars": sum(m["chars"] for m in sec_meta),
            "n_words": sum(m["words"] for m in sec_meta),
            "langs": declared_union, "detected_langs": detected,
            "sections": json.dumps(sec_meta, ensure_ascii=False),
            "metadata": json.dumps(meta, ensure_ascii=False, default=str),
            "ann_counts": json.dumps(ann_counts, ensure_ascii=False),
            "flags": sorted(set(flags)),
        })
        n_docs += 1
    flush_tokens()

    agg_rows = [{"dataset": ds, "split": k[0], "ann_key": k[1], "prmu": k[2],
                 "n_words_b": k[3], "n": v[0], "words_sum": v[1]}
                for k, v in agg.items()]

    out = Path(args["out_dir"])
    tag = args["tag"]
    return {
        "documents": _write_ndjson(out / f"doc_{tag}.ndjson", doc_rows),
        "gold": _write_ndjson(out / f"gold_{tag}.ndjson", gold_rows),
        "gold_agg": _write_ndjson(out / f"agg_{tag}.ndjson", agg_rows),
        "doc_tokens": _write_ndjson(out / f"tok_{tag}.ndjson", tok_rows),
        "gold_tokpos": _write_ndjson(out / f"tokpos_{tag}.ndjson", tokpos_rows),
        "kp_stage": _write_ndjson(out / f"kp_{tag}.ndjson", _PHRASES.drain()),
        "n_docs": n_docs, "n_gold": len(gold_rows),
        "bytes": args["end"] - args["start"],
        "secs": round(time.perf_counter() - _t0, 4),
        "rel": args.get("rel"), "tag": tag,
    }


# ---------------------------------------------------------------------------
# Inference batches
# ---------------------------------------------------------------------------
_PACK_CACHE: dict[str, dict] = {}     # path -> {doc_id: raw json line}


def _load_pack(path: str) -> dict:
    """Per-dataset gold pack, loaded once per worker.

    Stored as NDJSON and kept as raw bytes per document, parsed only for the
    documents a chunk actually touches: a materialised dict of every gold
    keyphrase in a large collection is gigabytes per worker, the bytes are
    ~8× smaller, and the parse is amortised over the chunk."""
    pack = _PACK_CACHE.get(path)
    if pack is None:
        if len(_PACK_CACHE) > 4:
            _PACK_CACHE.clear()
        pack = {}
        try:
            with open(path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc_id = None
                    try:                       # fast path: _id is written first
                        i = line.index(b'"_id":"') + 7
                        j = line.index(b'"', i)
                        doc_id = line[i:j].decode("utf-8")
                    except ValueError:
                        try:
                            doc_id = str(_loads(line).get("_id"))
                        except Exception:
                            doc_id = None
                    if doc_id is not None:
                        pack[doc_id] = line
        except Exception:
            pack = {}
        _PACK_CACHE[path] = pack
    return pack


def _pack_entry(pack: dict, doc_id: str):
    raw = pack.get(doc_id)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    obj = _loads(raw)
    # intern the stem strings: the same stems recur across a corpus and
    # interning collapses millions of duplicate str objects
    gold = {}
    for ann, variants in (obj.get("gold") or {}).items():
        gold[sys.intern(ann)] = [[sys.intern(v) for v in vs] for vs in variants]
    entry = {"langs": obj.get("langs") or {}, "gold": gold}
    pack[doc_id] = entry
    return entry


def derive_preds_chunk(args: dict) -> dict:
    """Match one byte-range of a batch_XXXXX.jsonl against its dataset's gold
    pack (loaded once per worker, shared across every run of that dataset)."""
    _t0 = time.perf_counter()
    pack = _load_pack(args["gold_pack"])
    primary = (args.get("primary_lang") or "en")[:2]

    pred_rows, match_rows = [], []
    n_lines = 0
    for off, ln, obj in _iter_lines(args["path"], args["start"], args["end"]):
        if not isinstance(obj, dict):
            continue
        doc_id = str(obj.get("_id", f"@{off}"))
        raw_preds = [str(p) for p in (obj.get("inferences") or [])]
        entry = _pack_entry(pack, doc_id)
        n_lines += 1

        analyses = [_PHRASES.analyze(p, primary) for p in raw_preds]
        seen, uniq = set(), []
        for a in analyses:
            s = _match_stems(a, primary)
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)

        pred_rows.append({
            "dataset": args["dataset"], "model": args["model"],
            "arch": args["arch"], "run_id": args["run_id"],
            "doc_id": doc_id, "batch_idx": args["batch_idx"],
            "file_id": args["file_id"], "byte_off": off, "byte_len": ln,
            "n_preds": len(raw_preds), "n_uniq": len(uniq),
            "costs": json.dumps(obj["costs"]) if isinstance(obj.get("costs"), dict) else None,
            # resolved set-based in SQL after ingest
            "known_doc": None,
        })
        if entry is None:
            continue

        for ann_key, gold_variants in entry["gold"].items():
            lang = (entry["langs"].get(ann_key) or primary)[:2]
            if lang == primary:
                pstems = uniq
            else:
                seen2, pstems = set(), []
                for a in analyses:
                    s = _match_stems(a, lang)
                    if s and s not in seen2:
                        seen2.add(s)
                        pstems.append(s)
            taken = [False] * len(gold_variants)
            stem_to_golds: dict[str, list[int]] = {}
            for gi, variants in enumerate(gold_variants):
                for v in variants:
                    stem_to_golds.setdefault(v, []).append(gi)
            pr, gi_hit = [], []
            for rank, s in enumerate(pstems):
                for gi in stem_to_golds.get(s, ()):
                    if not taken[gi]:
                        taken[gi] = True
                        pr.append(rank)
                        gi_hit.append(gi)
                        break
            match_rows.append({
                "dataset": args["dataset"], "model": args["model"],
                "arch": args["arch"], "run_id": args["run_id"],
                "doc_id": doc_id, "ann_key": ann_key,
                "n_uniq": len(pstems), "n_gold": len(gold_variants),
                "pred_ranks": pr, "gold_idxs": gi_hit,
            })

    out = Path(args["out_dir"])
    tag = args["tag"]
    return {
        "preds": _write_ndjson(out / f"preds_{tag}.ndjson", pred_rows),
        "matches": _write_ndjson(out / f"match_{tag}.ndjson", match_rows),
        "kp_stage": _write_ndjson(out / f"kp_{tag}.ndjson", _PHRASES.drain()),
        "n_docs": n_lines,
        "bytes": args["end"] - args["start"],
        "secs": round(time.perf_counter() - _t0, 4),
        "run_key": args.get("run_key"), "tag": tag,
    }


# ---------------------------------------------------------------------------
# POS phase: unique untagged phrases only
# ---------------------------------------------------------------------------

def pos_chunk(args: dict) -> dict:
    """args: phrases [[kp, raw], ...], lang, out_dir, tag."""
    _t0 = time.perf_counter()
    raws = [r for _kp, r in args["phrases"]]
    pats = tp.pos_patterns(raws, args["lang"])
    rows = [{"kp": kp, "pos": p}
            for (kp, _r), p in zip(args["phrases"], pats) if p]
    out = Path(args["out_dir"])
    return {"pos": _write_ndjson(out / f"pos_{args['tag']}.ndjson", rows),
            "items": len(args["phrases"]),
            "secs": round(time.perf_counter() - _t0, 4)}
