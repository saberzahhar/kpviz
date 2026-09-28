"""Worker-side derivations.

Per chunk of a document collection, one pass does everything except POS:
regex word tokens with separator sentinels, PyStemmer stems (one C call
per document), contiguous PRMU as a C substring search, gold de-duplicated
per annotation set, tokenizer counts and positions from one
encoding per document, and a worker-local phrase cache so each unique
(language, keyphrase) is analysed once per process. POS tagging happens in
its own scan phase over globally unique untagged gold phrases.

Workers never touch DuckDB and never touch the network: they spill NDJSON the
scanner ingests, and they receive every tokenizer verdict from the parent.
Everything stays importable at module level (spawn/forkserver-safe).
"""
from __future__ import annotations

import bisect
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


def is_train_split(split: str | None) -> bool:
    """train, training, and the size-named variants datasets ship
    (KPBiomed's train_large / train_medium / train_small, train-2020 …)."""
    s = (split or "").strip().lower()
    return s in TRAIN_SPLITS or s.startswith(("train_", "train-", "training_",
                                              "training-"))


def is_eval_split(split: str | None) -> bool:
    return not is_train_split(split)


def _tokenizers(args: dict) -> list:
    """Model tokenizers exactly as the parent resolved them (local-only)."""
    cache_dir = Path(args["tok_cache"]) if args.get("tok_cache") else None
    expect = args.get("tok_expect") or {}
    out = []
    for s in args.get("tokenizers", []):
        tk = tp.get_tokenizer(s, cache_dir, allow_network=False,
                              expect=expect.get(s))
        if expect.get(s) == "exact" and not tk.exact:
            # the parent certified this asset: a worker that cannot load it
            # would silently write approximate counts under an exact
            # signature — fail the scan with the reason instead
            raise RuntimeError(f"tokenizer {s} was resolved exactly by the scan "
                               f"but a worker cannot load it: {tk.why}")
        out.append(tk)
    return out


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def _profiled(fn):
    """KPVIZ_PROFILE_DIR=<dir>: every task writes its cProfile stats there
    (tools/bench/profile_scan.py merges them). Off by default: one getenv."""
    import functools

    @functools.wraps(fn)
    def run(args):
        where = os.environ.get("KPVIZ_PROFILE_DIR")
        if not where:
            return fn(args)
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        try:
            return fn(args)
        finally:
            pr.disable()
            Path(where).mkdir(parents=True, exist_ok=True)
            pr.dump_stats(str(Path(where) / f"{fn.__name__}-{os.getpid()}-"
                              f"{time.perf_counter_ns()}.prof"))
    return run


@_profiled
def derive_doc_chunk(args: dict) -> dict:
    """One byte-range of document.{ds}.jsonl -> spill files."""
    _t0 = time.perf_counter()
    ds = args["dataset"]
    card_sections: dict = args["card"].get("sections", {})
    card_anns: dict = args["card"].get("anns", {})
    optional = set(args["card"].get("optional") or ())
    emit_combined = args["card"].get("combined", False)
    token_scope = args.get("token_scope", "all")
    gold_scope = args.get("gold_scope", "eval")
    toks = _tokenizers(args)

    doc_rows: list[dict] = []
    gold_rows: list[dict] = []
    tok_rows: list[dict] = []
    tokpos_rows: list[dict] = []
    # (doc_id, text or None, P-gold refs, byte offset, word count)
    pending: list[tuple] = []
    agg: dict[tuple, list] = {}
    bad: list[int] = []
    n_docs = n_noid = n_gold_empty = n_gold_dup = 0
    pos_specs = set(args.get("pos_tokenizers") or args.get("tokenizers") or ())
    need_text = any(tk.exact for tk in toks)
    SENT = tp.SENT

    def flush_tokens():
        """Tokenizer counts (every document in scope) and gold token
        positions (documents with present gold, for the tokenizers of the
        models evaluated on this dataset), then release the texts.

        One encoding per document and tokenizer: documents with positions
        to compute get a full encoding (its length *is* the count); the rest
        are counted in one batch. An approximate tokenizer needs no text at
        all for a count — it scales the document's word count."""
        if not toks or not pending:
            pending.clear()
            return
        for tk in toks:
            exact = tk.exact
            ratio = None if exact else tk._resolve()[1]
            with_pos = tk.spec in pos_specs
            plain = [i for i, (_d, _t, g, _o, _n) in enumerate(pending)
                     if not (g and with_pos)]
            counts: dict[int, int] = {}
            if exact and plain:
                got, _ap = tk.count_batch([pending[i][1] for i in plain])
                counts.update(zip(plain, got))
            for i, (doc_id, text, gold_p, src_off, n_words) in enumerate(pending):
                enc = None
                gold_p = gold_p if with_pos else ()
                if not exact:
                    counts[i] = int(round(n_words * ratio))
                    if gold_p:
                        enc = tk.encode_cached(text)
                elif gold_p:
                    enc = tk.encode_cached(text)
                    counts[i] = enc.get("n", 0)
                tok_rows.append({"dataset": ds, "doc_id": doc_id,
                                 "src_off": src_off,
                                 "tokenizer": tk.spec, "n_tokens": counts[i],
                                 "approx": not exact})
                for g in gold_p:
                    te, ap = tk.char_to_token(text, g["end_char"], enc)
                    tokpos_rows.append({"dataset": ds, "doc_id": doc_id,
                                        "src_off": src_off,
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

        # one tokenisation per section, reused for the word count, the
        # language check and (joined, with a sentinel between sections) the
        # document's token stream
        flags: list[str] = []
        sec_meta, detected, declared_union, sec_toks = [], [], [], []
        n_words = 0
        for fieldname, declared, content in sections:
            tk_ = tp.tokens(tp.norm_text(content))
            nw_ = len(tk_) - tk_.count(SENT)
            scores = tp.language_scores(tk_, nw_) if nw_ >= 5 else {}
            det = tp.best_language(scores)
            detected.append(det)
            if tp.contradicts(scores, declared):
                flags.append(f"lang_mismatch:section:{fieldname}")
            sec_meta.append({"field": fieldname, "langs": declared, "det": det,
                             "chars": len(content), "words": nw_})
            sec_toks.append(tk_)
            n_words += nw_
            for l in declared:
                if l not in declared_union:
                    declared_union.append(l)
        present_fields = {s[0] for s in sections}
        for fieldname in card_sections:
            if fieldname not in present_fields and fieldname not in optional:
                flags.append(f"missing_section:{fieldname}")

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
                words = tp.tokenize(" ".join(tp.split_variants(str(k))[0]
                                             for k in kps))
                scores = tp.language_scores(words) if len(words) >= 6 else {}
                if tp.contradicts(scores, langs):
                    flags.append(f"lang_mismatch:ann:{key}")
        keep_gold = (gold_scope == "all") or eval_doc or bool(flags)

        full_text = None
        if keep_gold or need_text:
            full_text = SECTION_JOIN.join(c for _f, _l, c in sections)
        if keep_gold:
            # stored gold needs where each present keyphrase ends in the
            # original text: tokenise the whole normalised document with
            # offsets, mapped back through the normalisation, with a
            # sentinel where a section starts
            lowered, offmap = tp.norm_with_offsets(full_text)
            starts, at = [], 0
            for _f, _l, c in sections[:-1]:      # where each later one starts
                at += len(c) + len(SECTION_JOIN)
                starts.append(at if offmap is None
                              else bisect.bisect_left(offmap, at))
            toks_all, ends_all = tp.tokens_with_ends(lowered)
            doc_toks, word_ends, si = [], [], 0
            for t_, e_ in zip(toks_all, ends_all):
                while si < len(starts) and e_ - len(t_) >= starts[si]:
                    doc_toks.append(SENT)
                    si += 1
                doc_toks.append(t_)
                if t_ != SENT:
                    word_ends.append(e_ if offmap is None else offmap[e_])
        else:
            doc_toks, word_ends = [], None
            for i_, t_ in enumerate(sec_toks):
                if i_:
                    doc_toks.append(SENT)
                doc_toks.extend(t_)
        streams: dict[str, tp.StemmedDoc] = {}

        def stream(lang: str | None) -> tp.StemmedDoc:
            lang2 = (lang or "en")[:2]
            got = streams.get(lang2)
            if got is None:
                got = streams[lang2] = tp.StemmedDoc(
                    tp.get_stemmer(lang2).stemWords(doc_toks))
            return got

        for a in anns:
            key = a.get("annotator") or "annotation"
            langs = declared_langs(a) or card_anns.get(key) or declared_union
            lang = (langs[0] if langs else None)
            kps = a.get("keyphrases") or []
            sdoc = stream(lang)

            # one gold keyphrase per stemmed identity: a keyphrase with no
            # word token is dropped (it can never be matched), and one whose
            # stemmed forms repeat an earlier one of the same annotation set
            # is a duplicate (a prediction can match only one of the two, so
            # keeping both would inflate |gold|). kp_idx is the kept order.
            seen_sig: set = set()
            kept = 0
            for kp in kps:
                variants = tp.split_variants(str(kp))
                entries = [_PHRASES.analyze(v, lang, persist=keep_gold)
                           for v in variants]
                entries = [e for e in entries if e["tokens"]]
                stems_repr = list(dict.fromkeys(e["sstr"] for e in entries))
                sig = frozenset(stems_repr)
                if not sig:
                    n_gold_empty += 1
                    continue
                if sig in seen_sig:
                    n_gold_dup += 1
                    continue
                seen_sig.add(sig)
                cat, end_word = tp.prmu_classify(
                    [e["stems"] for e in entries], sdoc,
                    [e["pstems"] for e in entries if e.get("pstems")])
                nw = entries[0]["n_tokens"]
                agg_add(split, key, cat, nw)
                row = None
                if keep_gold:
                    end_char = word_ends[end_word] if end_word >= 0 else -1
                    row = {"dataset": ds, "doc_id": doc_id, "ann_key": key,
                           "kp_idx": kept, "src_off": off,
                           "display": entries[0]["kp"],
                           "surface": entries[0]["raw"],
                           "stems": stems_repr,
                           "lang": lang, "n_words": nw, "prmu": cat,
                           "end_char": end_char, "end_word": end_word}
                    gold_rows.append(row)
                    if cat == "P":
                        doc_gold_p.append(row)
                kept += 1
                if emit_combined:
                    if sig not in combined_seen:
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
            pending.append((doc_id, full_text, doc_gold_p, off, n_words))
            if len(pending) >= _TOK_SUBBATCH:
                flush_tokens()

        doc_rows.append({
            "dataset": ds, "doc_id": doc_id, "split": split,
            "file_id": args["file_id"], "byte_off": off, "byte_len": ln,
            "n_sections": len(sections),
            "n_chars": sum(m["chars"] for m in sec_meta),
            "n_words": n_words,
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
        "n_noid": n_noid, "n_gold_empty": n_gold_empty,
        "n_gold_dup": n_gold_dup,
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
    """One-to-one matching of (unique) predictions to gold keyphrases, in
    rank order, maximum in size.

    When no stem belongs to two gold keyphrases — the normal case — each
    prediction simply takes the gold it equals (greedy is then optimal).
    When '+'-alternatives overlap ({a, b} and {a}), a greedy first pick can
    strand a later prediction (a takes {a, b}, b finds nothing although a →
    {a}, b → {a, b} matches both); predictions are then placed with
    augmenting paths, in rank order, which yields a maximum matching whose
    matched ranks are the earliest possible. The assignment is frozen here;
    gold-side filters later score against it."""
    stem_to_golds: dict[str, list[int]] = {}
    for gi, variants in enumerate(gold_variants):
        for v in variants:
            stem_to_golds.setdefault(v, []).append(gi)
    if all(len(g) == 1 for g in stem_to_golds.values()):
        taken = [False] * len(gold_variants)
        pr, gi_hit = [], []
        for rank, s in enumerate(pstems):
            g = stem_to_golds.get(s)
            if g and not taken[g[0]]:
                taken[g[0]] = True
                pr.append(rank)
                gi_hit.append(g[0])
        return pr, gi_hit
    owner = [-1] * len(gold_variants)          # gold -> prediction rank

    def augment(rank: int, seen: set) -> bool:
        for gi in stem_to_golds.get(pstems[rank], ()):
            if gi in seen:
                continue
            seen.add(gi)
            if owner[gi] < 0 or augment(owner[gi], seen):
                owner[gi] = rank
                return True
        return False
    for rank, s in enumerate(pstems):
        if s in stem_to_golds:
            augment(rank, set())
    pairs = sorted((r, gi) for gi, r in enumerate(owner) if r >= 0)
    return [r for r, _g in pairs], [g for _r, g in pairs]


@_profiled
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
                    "batch_idx": seg["batch_idx"], "byte_off": off,
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

@_profiled
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
        rows = [{"kp": kp, "lang": args["lang"], "pos": p or ""}
                for (kp, _r), p in zip(args["phrases"], pats)]
    out = Path(args["out_dir"])
    return {"pos": _write_ndjson(out / f"pos_{args['tag']}.ndjson", rows),
            "items": len(args["phrases"]),
            "secs": round(time.perf_counter() - _t0, 4)}
