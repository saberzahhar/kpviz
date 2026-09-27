"""Worker-side derivations.

Per chunk of a document collection, one pass does everything except POS:
spaCy-blank tokens (C path), PyStemmer stems (one C call per document),
in-order PRMU via position indices, tokenizer counts and positions from one
encoding per document, and a worker-local phrase cache so each unique
(language, keyphrase) is analysed once per process. POS tagging happens in
its own scan phase over globally unique untagged gold phrases.

Workers never touch DuckDB and never touch the network: they spill NDJSON the
scanner ingests, and they receive every tokenizer verdict from the parent.
Everything stays importable at module level (spawn/forkserver-safe).
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
        return _stdjson.dumps(o, ensure_ascii=False, default=str,
                              separators=(",", ":")).encode()

    JSON_BACKEND = "json"

from . import textproc as tp
from .util import declared_langs

SECTION_JOIN = "\n\n"
TRAIN_SPLITS = {"train", "training"}
_TOK_SUBBATCH = 512          # documents per tokenizer sub-batch


def _dumps_str(o) -> str:
    return _dumps(o).decode("utf-8")


# ---------------------------------------------------------------------------
# Worker process initialisation
# ---------------------------------------------------------------------------
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")


def init_worker(phrase_cache_max: int = 300_000,
                tiktoken_cache: str | None = None) -> None:
    """Pin every nested thread pool to one thread, and take workers offline.

    HuggingFace `tokenizers` (rayon), tiktoken (rayon) and the BLAS under
    spaCy each size their internal pool to the whole machine. With one worker
    process per core that is cores² runnable threads — a context-switch storm
    that makes a 32-core box slower than a 4-core one. Process-level
    parallelism already saturates the machine; intra-worker threading can
    only take away. Must run before those libraries are imported.

    The parent resolved every tokenizer asset before the pool started, so a
    worker has nothing to download: HF_HUB_OFFLINE keeps it from trying."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if tiktoken_cache:
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache
    global _PHRASES
    _PHRASES = tp.PhraseCache(max_size=max(20_000, int(phrase_cache_max)))


# ---------------------------------------------------------------------------
# Worker-local state (persists across chunks and phases in the same process)
# ---------------------------------------------------------------------------
_PHRASES = tp.PhraseCache()


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


def _iter_lines(path: str, start: int, end: int, bad: list | None = None):
    """Whole lines in [start, end). Byte ranges are newline-aligned by the
    planner, so no document is ever split. Unparseable lines are yielded as
    None and their byte offsets collected in `bad` (reported, never silent)."""
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
                    if bad is not None:
                        bad.append(pos)
                    yield pos, ln, None
            pos += ln


def is_eval_split(split: str | None) -> bool:
    return (split or "").lower() not in TRAIN_SPLITS


def _tokenizers(args: dict) -> list:
    """Model tokenizers exactly as the parent resolved them (local-only)."""
    cache_dir = Path(args["tok_cache"]) if args.get("tok_cache") else None
    expect = args.get("tok_expect") or {}
    return [tp.get_tokenizer(s, cache_dir, allow_network=False,
                             expect=expect.get(s))
            for s in args.get("tokenizers", [])]


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
    toks = _tokenizers(args)

    doc_rows: list[dict] = []
    gold_rows: list[dict] = []
    tok_rows: list[dict] = []
    tokpos_rows: list[dict] = []
    pending: list[tuple[str, str, list]] = []   # (doc_id, text, P-gold refs)
    agg: dict[tuple, list] = {}
    bad: list[int] = []
    n_docs = n_noid = 0

    def flush_tokens():
        """Tokenizer counts + gold token positions for the pending documents,
        then release their text (a chunk's worth of text held twice is the
        one place worker RSS could grow without bound).

        One encoding per document and tokenizer: documents with present gold
        get a full encoding (its length *is* the count); the rest are counted
        in one batch. Approximate tokenizers share one regex pass per
        document."""
        if not toks or not pending:
            pending.clear()
            return
        starts_cache: dict[int, list[int]] = {}

        def word_starts(i, text):
            got = starts_cache.get(i)
            if got is None:
                got = starts_cache[i] = [m.start() for m in
                                         tp._WORD_RE.finditer(text)]
            return got

        for tk in toks:
            exact = tk.exact
            ratio = None if exact else tk._resolve()[1]
            plain = [i for i, (_d, _t, g) in enumerate(pending) if not g or not exact]
            counts: dict[int, int] = {}
            if exact and plain:
                got, _ap = tk.count_batch([pending[i][1] for i in plain])
                counts.update(zip(plain, got))
            for i, (doc_id, text, gold_p) in enumerate(pending):
                enc = None
                if not exact:
                    ws = word_starts(i, text)
                    counts[i] = int(round(len(ws) * ratio))
                    if gold_p:
                        enc = {"kind": "approx", "starts": ws}
                elif gold_p:
                    enc = tk.encode_cached(text)
                    counts[i] = enc.get("n", 0)
                tok_rows.append({"dataset": ds, "doc_id": doc_id,
                                 "tokenizer": tk.spec, "n_tokens": counts[i],
                                 "approx": not exact})
                for g in gold_p:
                    te, ap = tk.char_to_token(text, g["end_char"], enc)
                    tokpos_rows.append({"dataset": ds, "doc_id": doc_id,
                                        "ann_key": g["ann_key"],
                                        "kp_idx": g["kp_idx"],
                                        "tokenizer": tk.spec,
                                        "tok_end": te, "approx": ap})
        pending.clear()

    def agg_add(split, ann: str, cat: str, nw: int):
        k4 = (split, ann, cat, min(nw, 6) if nw else 0)
        slot = agg.get(k4)
        if slot is None:
            agg[k4] = [1, nw]
        else:
            slot[0] += 1
            slot[1] += nw

    for off, ln, obj in _iter_lines(args["path"], args["start"], args["end"], bad):
        if not isinstance(obj, dict):
            continue
        if obj.get("_id") is None:
            n_noid += 1
        doc_id = str(obj.get("_id", f"@{off}"))
        meta = obj.get("metadata") or {}
        split = meta.get("split")
        eval_doc = is_eval_split(split)

        sections = []
        for s in (obj.get("sections") or []):
            fieldname = s.get("field") or "?"
            declared = declared_langs(s) or card_sections.get(fieldname) or []
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
        present_fields = {s[0] for s in sections}
        for fieldname in card_sections:
            if fieldname not in present_fields:
                flags.append(f"missing_section:{fieldname}")

        # one normalisation per document, shared by every language stream;
        # token ends are mapped back to original-text offsets when NFKC or
        # lowercasing changed the length
        lowered = tp.norm_text(full_text)
        offmap = tp.norm_offsets(full_text, lowered)
        streams: dict[str, tuple] = {}

        def stream(lang: str | None):
            lang2 = (lang or "en")[:2]
            got = streams.get(lang2)
            if got is None:
                words, ends = tp.spacy_doc_tokens(full_text, lang2, lowered)
                if offmap is not None:
                    ends = [offmap[e] for e in ends]
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
            langs = declared_langs(a) or card_anns.get(key) or declared_union
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
            langs = declared_langs(a) or card_anns.get(key) or declared_union
            lang = (langs[0] if langs else None)
            kps = a.get("keyphrases") or []
            index, char_ends = stream(lang)

            for i, kp in enumerate(kps):
                variants = tp.split_variants(str(kp))
                entries = [_PHRASES.analyze(v, lang, persist=keep_gold)
                           for v in variants]
                entries = [e for e in entries if e["tokens"]]
                var_stems = [e["stems"] for e in entries]
                cat, end_word = tp.prmu_classify(var_stems, index)
                end_char = char_ends[end_word] if end_word >= 0 else -1
                nw = entries[0]["n_tokens"] if entries else 0
                agg_add(split, key, cat, nw)
                stems_repr = [e["sstr"] for e in entries]
                row = None
                if keep_gold:
                    row = {"dataset": ds, "doc_id": doc_id, "ann_key": key,
                           "kp_idx": i,
                           "display": entries[0]["kp"] if entries else "",
                           "stems": stems_repr,
                           "lang": lang, "n_words": nw, "prmu": cat,
                           "end_char": end_char, "end_word": end_word}
                    gold_rows.append(row)
                    if cat == "P":
                        doc_gold_p.append(row)
                if emit_combined:
                    sig = frozenset(stems_repr)
                    if sig and sig not in combined_seen:
                        combined_seen.add(sig)
                        agg_add(split, "@combined", cat, nw)
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
            "sections": _dumps_str(sec_meta),
            "metadata": _meta_json(meta),
            "ann_counts": _dumps_str(ann_counts),
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
        "n_bad": len(bad), "first_bad": bad[0] if bad else None,
        "n_noid": n_noid,
        "bytes": args["end"] - args["start"],
        "secs": round(time.perf_counter() - _t0, 4),
        "cache": _PHRASES.stats(),
        "rel": args.get("rel"), "tag": tag,
    }


def _meta_json(meta: dict) -> str:
    """Document metadata as a JSON string (orjson; stdlib for exotic values)."""
    try:
        return _dumps_str(meta)
    except Exception:
        import json
        return json.dumps(meta, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Inference batches
# ---------------------------------------------------------------------------
_PACK_CACHE: dict[str, dict] = {}     # path -> {doc_id: raw json line}
# decoded entries, per pack, shared by every run of that dataset a worker
# matches (one run per task: without it each task decoded the same documents
# again). Bounded: cleared past _DECODED_MAX entries.
_DECODED: dict[str, dict] = {}
_DECODED_MAX = 200_000


def _load_pack(path: str) -> dict:
    """Per-dataset gold pack, loaded once per worker.

    Kept as raw bytes per document and decoded only for the documents a
    chunk touches (into a chunk-local dict — the long-lived pack never turns
    into a decoded copy). The id is taken from a real JSON parse: a manual
    scan for the closing quote mis-read ids containing escaped quotes."""
    pack = _PACK_CACHE.get(path)
    if pack is None:
        if len(_PACK_CACHE) > 4:
            _PACK_CACHE.clear()
            _DECODED.clear()
        pack = {}
        try:
            with open(path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        doc_id = str(_loads(line)["_id"])
                    except Exception:
                        continue
                    pack[doc_id] = line
        except OSError:
            pack = {}
        _PACK_CACHE[path] = pack
    return pack


def _pack_entry(pack: dict, doc_id: str, local: dict):
    hit = local.get(doc_id)
    if hit is not None:
        return hit
    raw = pack.get(doc_id)
    if raw is None:
        return None
    obj = _loads(raw)
    # intern the stem strings: the same stems recur across a corpus and
    # interning collapses millions of duplicate str objects
    gold = {}
    for ann, variants in (obj.get("gold") or {}).items():
        gold[sys.intern(ann)] = [[sys.intern(v) for v in vs] for vs in variants]
    entry = {"langs": obj.get("langs") or {}, "gold": gold}
    local[doc_id] = entry
    return entry


def _uniq_stems(analyses: list[dict]) -> tuple[list[str], list[int]]:
    """De-duplicated stem strings in rank order, and the index of the raw
    prediction each one came from."""
    seen, uniq, first = set(), [], []
    for i, a in enumerate(analyses):
        s = a["sstr"]
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
            first.append(i)
    return uniq, first


def _match(pstems: list[str], gold_variants: list[list[str]]):
    """Greedy rank-order matching: each prediction takes the first untaken
    gold keyphrase it equals (any '+'-variant)."""
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
    return pr, gi_hit


def derive_preds_chunk(args: dict) -> dict:
    """Match one or more byte-ranges of batch_XXXXX.jsonl files of one run
    against its dataset's gold pack (loaded once per worker, shared across
    every run of that dataset).

    Tiny batch files are coalesced into one task by the planner (`segments`):
    per-task overhead, not matching, dominated the phase when every 80 KB
    batch was its own task with its own three spill files."""
    _t0 = time.perf_counter()
    pack = _load_pack(args["gold_pack"])
    local = _DECODED.setdefault(args["gold_pack"], {})
    if sum(len(v) for v in _DECODED.values()) > _DECODED_MAX:
        _DECODED.clear()
        local = _DECODED.setdefault(args["gold_pack"], {})
    primary = (args.get("primary_lang") or "en")[:2]
    segments = args.get("segments") or [
        {k: args[k] for k in ("path", "start", "end", "batch_idx", "file_id")}]

    pred_rows, match_rows = [], []
    bad: list[int] = []
    n_lines = n_noid = n_bytes = 0
    for seg in segments:
        n_bytes += seg["end"] - seg["start"]
        for off, ln, obj in _iter_lines(seg["path"], seg["start"], seg["end"], bad):
            if not isinstance(obj, dict):
                continue
            if obj.get("_id") is None:
                n_noid += 1
            doc_id = str(obj.get("_id", f"@{off}"))
            raw_preds = [str(p) for p in (obj.get("inferences") or [])]
            entry = _pack_entry(pack, doc_id, local)
            n_lines += 1

            # predictions are analysed in each annotation's own language (the
            # gold was), never tokenised in one language and re-stemmed in
            # another; they are not persisted (the UI shows gold phrases only)
            by_lang: dict[str, tuple] = {}

            def uniq_in(lang2: str):
                got = by_lang.get(lang2)
                if got is None:
                    an = [_PHRASES.analyze(p, lang2, persist=False)
                          for p in raw_preds]
                    got = by_lang[lang2] = (an, *_uniq_stems(an))
                return got

            _an, uniq, _first = uniq_in(primary)
            costs = obj.get("costs")
            pred_rows.append({
                "dataset": args["dataset"], "model": args["model"],
                "arch": args["arch"], "run_id": args["run_id"],
                "doc_id": doc_id, "batch_idx": seg["batch_idx"],
                "file_id": seg["file_id"], "byte_off": off, "byte_len": ln,
                "n_preds": len(raw_preds), "n_uniq": len(uniq),
                "costs": _dumps_str(costs) if isinstance(costs, dict) else None,
            })
            if entry is None:
                continue

            for ann_key, gold_variants in entry["gold"].items():
                glang = entry["langs"].get(ann_key) or primary
                if isinstance(glang, list):
                    # a multilingual union (@combined): ranks follow the
                    # primary-language de-duplication; each gold keyphrase is
                    # compared with the prediction analysed in *its* language
                    an_p, uniq_p, first_p = uniq_in(primary)
                    taken = [False] * len(gold_variants)
                    gsets = [set(v) for v in gold_variants]
                    pr, gi_hit = [], []
                    for rank, ri in enumerate(first_p):
                        for gi, gl in enumerate(glang):
                            if taken[gi]:
                                continue
                            s = uniq_in((gl or primary)[:2])[0][ri]["sstr"]
                            if s and s in gsets[gi]:
                                taken[gi] = True
                                pr.append(rank)
                                gi_hit.append(gi)
                                break
                    n_uniq = len(uniq_p)
                else:
                    pstems = uniq_in(glang[:2])[1]
                    pr, gi_hit = _match(pstems, gold_variants)
                    n_uniq = len(pstems)
                match_rows.append({
                    "dataset": args["dataset"], "model": args["model"],
                    "arch": args["arch"], "run_id": args["run_id"],
                    "doc_id": doc_id, "ann_key": ann_key,
                    "n_uniq": n_uniq, "n_gold": len(gold_variants),
                    "pred_ranks": pr, "gold_idxs": gi_hit,
                })

    out = Path(args["out_dir"])
    tag = args["tag"]
    return {
        "preds": _write_ndjson(out / f"preds_{tag}.ndjson", pred_rows),
        "matches": _write_ndjson(out / f"match_{tag}.ndjson", match_rows),
        "n_docs": n_lines, "n_bad": len(bad), "n_noid": n_noid,
        "bytes": n_bytes,
        "secs": round(time.perf_counter() - _t0, 4),
        "cache": _PHRASES.stats(),
        "run_key": args.get("run_key"), "tag": tag,
    }


# ---------------------------------------------------------------------------
# POS phase: unique untagged gold phrases only
# ---------------------------------------------------------------------------

def pos_chunk(args: dict) -> dict:
    """args: phrases [[kp, raw], ...], lang, out_dir, tag.

    A phrase the tagger returns nothing for is stored as '' ("tagged, no
    pattern") so it is not re-tagged on every scan."""
    _t0 = time.perf_counter()
    # the matching phase is over: release its per-worker state before a
    # spaCy model (~300 MB) is loaded next to it
    if _PACK_CACHE or _DECODED:
        _PACK_CACHE.clear()
        _DECODED.clear()
        import gc
        gc.collect()
    rows = []
    if tp._spacy_tagger(args["lang"]) is not None:
        raws = [r for _kp, r in args["phrases"]]
        pats = tp.pos_patterns(raws, args["lang"])
        rows = [{"kp": kp, "pos": p or ""}
                for (kp, _r), p in zip(args["phrases"], pats)]
    out = Path(args["out_dir"])
    return {"pos": _write_ndjson(out / f"pos_{args['tag']}.ndjson", rows),
            "items": len(args["phrases"]),
            "secs": round(time.perf_counter() - _t0, 4)}
