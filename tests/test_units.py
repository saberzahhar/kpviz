"""Focused regressions for the fixes that do not need a scanned store."""
from __future__ import annotations

import sys
import time

from conftest import REPO

sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------- ingest
def test_drain_waits_for_the_write(monkeypatch, tmp_path):
    """drain() is a barrier: it returns after rows land, not when the flush
    request is dequeued (review B, D02)."""
    from kpviz import db, ingest
    landed = []

    def slow_ingest(table, paths, cols, mode="insert", delete_after=True):
        time.sleep(0.4)
        landed.append(table)
        return 1
    monkeypatch.setattr(db, "ingest_ndjson", slow_ingest)
    p = tmp_path / "x.ndjson"
    p.write_text("{}\n")
    ing = ingest.Ingestor({"t": ({"a": "INTEGER"}, "insert")})
    ing.submit({"t": str(p)})
    t0 = time.perf_counter()
    ing.drain()
    assert landed == ["t"] and time.perf_counter() - t0 >= 0.35
    ing.close()


def test_ingest_error_is_sticky(monkeypatch, tmp_path):
    from kpviz import db, ingest

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(db, "ingest_ndjson", boom)
    p = tmp_path / "x.ndjson"
    p.write_text("{}\n")
    ing = ingest.Ingestor({"t": ({"a": "INTEGER"}, "insert")})
    ing.submit({"t": str(p)})
    for _ in range(2):              # every later call fails too
        try:
            ing.drain()
            raise AssertionError("drain must raise")
        except ingest.IngestError:
            pass
    try:
        ing.close()
    except ingest.IngestError:
        pass


# ---------------------------------------------------------------- RQ4
def test_pareto_ties_and_non_finite():
    from kpviz.pages.rq.rq4 import _frontier
    xs, ys = _frontier([(1, .2), (1, .8), (2, .5), (3, .9), (float("nan"), 1.0)])
    assert list(zip(xs, ys)) == [(1, .8), (3, .9)]


# ---------------------------------------------------------------- exports
BAR_WITH_GAP = {"kind": "bar", "size": "2col", "ylabel": "F1@O",
                "series": [{"name": "a", "x": ["r1", "r2"], "y": [0.3, None],
                            "text": ["", "†"]},
                           {"name": "b", "x": ["r1", "r2"], "y": [None, 0.4]}],
                "vlines": [{"x": 0.5, "shade_beyond": True}], "xrange": [-1, 3]}
HEAT_WITH_GAP = {"kind": "heatmap", "size": "1col",
                 "heat": {"z": [[1, None], [None, 1]], "x": ["a", "b"], "y": ["a", "b"],
                          "zmin": -1, "zmax": 1, "zmid": 0, "diverging": True,
                          "text": [["1.000", ""], ["", "1.000"]]}}


def test_missing_values_export(tmp_path):
    from kpviz.export import fig_pdf, fig_png
    for spec in (BAR_WITH_GAP, HEAT_WITH_GAP):
        assert fig_png(spec)[:8] == b"\x89PNG\r\n\x1a\n"
        pdf, _method, _note = fig_pdf(spec)
        assert pdf[:5] == b"%PDF-"


def test_matplotlib_pdf_has_no_type3_fonts():
    """savefig inside the rc context: pdf.fonttype=42 applies (review A, X1)."""
    from kpviz.figures import render
    pdf = render(BAR_WITH_GAP, False, "pdf")
    # fonttype 42 embeds TrueType outlines as a Type0/CIDFontType2 font
    assert b"/Type3" not in pdf and b"/CIDFontType2" in pdf


def test_figure_environment_follows_size():
    from kpviz.export import latex_figure
    two = latex_figure("f.pgf", "c", "l", pgf=True, size="2col")
    one = latex_figure("f.pgf", "c", "l", pgf=True, size="1col")
    assert "\\begin{figure*}" in two and "\\resizebox" not in two
    assert "\\begin{figure}" in one and "figure*" not in one


def test_caption_edit_does_not_rerender():
    from kpviz import export
    spec = dict(BAR_WITH_GAP, caption="one")
    export.fig_png(spec)
    n = len(export._CACHE)
    export.fig_png(dict(spec, caption="two"))
    assert len(export._CACHE) == n


# ---------------------------------------------------------------- text
def test_language_keyed_phrase_cache():
    from kpviz import textproc as tp
    c = tp.PhraseCache()
    en = c.analyze("l'apprentissage", "en")
    fr = c.analyze("l'apprentissage", "fr")
    assert en["lang"] == "en" and fr["lang"] == "fr"
    assert fr["tokens"] == tp.PhraseCache().analyze("l'apprentissage", "fr")["tokens"]


def test_detect_language_matches_reference():
    from kpviz import textproc as tp
    texts = ["the model of the network is in a graph and the data",
             "le modèle de la langue est dans les données et le corpus",
             "der die das und in den von zu mit sich", "short"]
    for t in texts:
        toks = tp.tokenize(t)
        got = tp.detect_language(t, tokens=toks)
        if len(toks) < 5:
            assert got == (None, 0.0)
            continue
        n = len(toks)
        scores = {l: sum(1 for w in toks if w in s) / n for l, s in tp._STOPWORDS.items()}
        best = max(scores, key=scores.get)
        assert got[0] == (best if scores[best] >= 0.08 else None)


def test_approx_positions_equal_prefix_rescan():
    from kpviz import textproc as tp
    tk = tp.ModelTokenizer("nothing[here]", allow_network=False, expect="approx")
    text = "Alpha beta, gamma-delta épsilon; zeta_eta theta " * 20
    enc = tk.encode_cached(text)
    ratio = tk._resolve()[1]
    for end in range(0, len(text), 7):
        want = int(round(len(tp._WORD_RE.findall(text[:end])) * ratio))
        assert tk.char_to_token(text, end, enc) == (want, True)


def test_norm_offsets_map_back_to_source():
    from kpviz import textproc as tp
    text = "ﬃ cat Straße"
    low, m = tp.norm_with_offsets(text)
    toks, ends = tp.tokens_with_ends(low)
    assert [text[:m[e]].split()[-1] for e in ends][-1] == "Straße"


def test_normalisation_alignment_is_exact():
    """R05: composition, expansion and changes that cancel in total length
    all map exactly (the old map guessed proportionally, or not at all)."""
    from kpviz import textproc as tp
    for text, word, want_end in [
            ("e\u0301 alpha", "é", 2),                 # composition
            ("e\u0301 cat \ufb01", "cat", 6),         # net length unchanged
            ("\ufb01ne tuning", "fine", 3),            # expansion
    ]:
        low, m = tp.norm_with_offsets(text)
        assert m is not None and len(m) == len(low) + 1
        toks, ends = tp.tokens_with_ends(low)
        i = toks.index(word)
        assert m[ends[i]] == want_end, (text, word, m[ends[i]])
    assert tp.norm_with_offsets("plain ascii")[1] is None


def test_special_tokens_never_give_impossible_positions(tmp_path):
    """R05: a tokenizer with BOS/EOS gives (0, 0) offsets for them; the
    position of a content character counts BOS and never exceeds the
    encoding's length."""
    pytest = __import__("pytest")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)])
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    from kpviz import textproc as tp
    mt = tp.ModelTokenizer(f"transformers[file:{path}]", allow_network=False,
                           expect="exact")
    text = "a b c"
    enc = mt.encode_cached(text)
    assert enc["n"] == 5                      # <s> a b c </s>
    got = [mt.char_to_token(text, e, enc)[0] for e in (1, 3, 5)]
    assert got == [2, 3, 4]                   # a, b, c after <s>
    assert all(0 <= g <= enc["n"] for g in got)


def test_offline_tokenizer_never_downloads(tmp_path, monkeypatch):
    from kpviz import textproc as tp
    monkeypatch.setenv("KPVIZ_OFFLINE", "1")
    tk = tp.ModelTokenizer("transformers[some/unknown-model]", cache_dir=tmp_path)
    t0 = time.perf_counter()
    assert not tk.exact and time.perf_counter() - t0 < 1.0


def test_jsonc():
    import json
    from kpviz.util import strip_jsonc
    src = '{"a": "x // not a comment", /* c */ "b": 1 // trailing\n}'
    assert json.loads(strip_jsonc(src)) == {"a": "x // not a comment", "b": 1}


def test_training_split_names():
    """Size-named training splits (KPBiomed's train_large …) are training
    data in Python and in SQL alike — never evaluated, always leakage
    counterparts."""
    import duckdb
    from kpviz.derive import is_train_split
    from kpviz.scanner import TRAIN_SPLIT_SQL
    names = ["train", "Training", "train_large", "train-small", "training_2020",
             "test", "validation", "dev", "trainee", "", None]
    want = [True, True, True, True, True, False, False, False, False, False, False]
    assert [is_train_split(n) for n in names] == want
    con = duckdb.connect()
    got = [con.execute(f"SELECT {TRAIN_SPLIT_SQL.format(col='?')}", [n]).fetchone()[0]
           for n in names]
    assert got == want


def test_offline_tiktoken_reads_a_cached_encoding(monkeypatch, tmp_path):
    """R09: offline, a locally cached encoding is used (the old code gave up
    before asking tiktoken), while a download is refused."""
    pytest = __import__("pytest")
    tiktoken = pytest.importorskip("tiktoken")
    import tiktoken.load as tl
    from kpviz import textproc as tp

    class Enc:
        name, n_vocab, _special_tokens, _pat_str = "fake_base", 3, {}, "x"

    def cached(name):          # what a warm cache does: no read_file call
        return Enc()

    def uncached(name):        # what a cold cache does: a download
        return tl.read_file("https://example.invalid/" + name)
    monkeypatch.setattr(tiktoken, "list_encoding_names", lambda: ["fake_base"])
    monkeypatch.setattr(tiktoken, "get_encoding", cached)
    tk = tp.ModelTokenizer("tiktoken[fake_base]", cache_dir=tmp_path,
                           allow_network=False)
    assert tk.exact
    monkeypatch.setattr(tiktoken, "get_encoding", uncached)
    tk = tp.ModelTokenizer("tiktoken[fake_base]", cache_dir=tmp_path,
                           allow_network=False)
    assert not tk.exact and "not cached locally" in tk.why


def test_tokenizer_fingerprint_follows_the_asset_bytes(tmp_path):
    """R09: a tokenizer.json replaced at the same path changes the identity
    the derivation signatures use."""
    pytest = __import__("pytest")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from kpviz import textproc as tp
    path = tmp_path / "tokenizer.json"

    def save(vocab):
        t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        t.pre_tokenizer = pre_tokenizers.Whitespace()
        t.save(str(path))
    save({"[UNK]": 0, "a": 1})
    fp1 = tp.ModelTokenizer(f"transformers[file:{path}]", allow_network=False).fingerprint
    save({"[UNK]": 0, "a": 1, "b": 2})
    fp2 = tp.ModelTokenizer(f"transformers[file:{path}]", allow_network=False).fingerprint
    assert fp1.startswith("exact:") and fp1 != fp2


def test_card_validation_fails_closed():
    """RV-E10: defaults are checked, numeric value sets hold, an unknown type
    is flagged; KPViz's own run parameter needs no card declaration."""
    from kpviz.cards import ModelCard
    card = ModelCard("m", None, {"inference": {
        "num_beams": {"type": "int", "min": 1, "default": -1},
        "k": {"type": "int", "values": [1, 4, 8]},
        "mystery": {"type": "tensor"}}})
    _res, bad = card.validate_params({"k": 5, "mystery": 1, "reserved_tokens": 12})
    probs = {v["param"]: v["problem"] for v in bad}
    assert "card default" in probs["num_beams"]
    assert "not in allowed set" in probs["k"]
    assert "not one KPViz can check" in probs["mystery"]
    assert "reserved_tokens" not in probs


def test_cost_prefers_a_complete_level():
    """RV-R06: one document of ten reporting 2 s must not win over a complete
    batch total of 20 s."""
    from kpviz.cards import ArchCard
    from kpviz.costs import resolve_var_totals
    arch = ArchCard("a", None, {"variables": {"time": {"unit": "s",
                                                       "level": "document"}},
                                "rates": {"time": {"time": 1.0}}})
    got = resolve_var_totals(arch, {"time": (2.0, 1)}, {"time": (20.0, 3)},
                             n_docs=10, n_batches=3, wall_from_timestamps=None)
    assert got["time"]["total"] == 20.0 and got["time"]["level"] == "batch"
    assert "partial_coverage" not in got["time"]["flags"]


def test_window_minus_reserved_tokens():
    """R2-N14: the usable window is the declared one minus the reservation."""
    from kpviz.cards import ParamSpec
    from kpviz.pages.rq.rq3 import _as_count, _reserved
    ps = ParamSpec(name="input_max_size", type="context_window",
                   raw={"reserved_tokens": 2})
    assert _reserved(ps, {}) == 2
    assert _reserved(ps, {"reserved_tokens": {"value": 300}}) == 300
    assert _as_count(float("inf")) is None and _as_count("x") is None


def test_tokenizer_errors_are_one_actionable_line():
    from kpviz import textproc as tp

    class RepositoryNotFoundError(Exception):
        pass

    class GatedRepoError(Exception):
        pass
    raw = ("401 Client Error. (Request ID: Root=1-68d3-abc)\n\nRepository Not "
           "Found for url: https://huggingface.co/llama-3.3/resolve/main/tokenizer.json.")
    got = tp._hf_error(RepositoryNotFoundError(raw), "llama-3.3")
    assert "Request ID" not in got and "\n" not in got and "llama-3.3" in got
    assert "HF_TOKEN" in tp._hf_error(GatedRepoError("Cannot access gated repo"),
                                      "meta-llama/Llama-3.3-70B-Instruct")
    # the Llama 3 tokenizer is not a tiktoken encoding: it is read as the
    # published tokenizer.json (here offline, so approximate — and said why)
    tk = tp.ModelTokenizer("tiktoken[llama3]", allow_network=False)
    assert tk.status == "approx" and "tiktoken has no" not in tk.why


def test_multilingual_tokens():
    """Accents, elision, Arabic, Devanagari and CJK; mojibake repaired."""
    from kpviz import textproc as tp
    t = lambda s: [x for x in tp.tokens(tp.norm_text(s)) if x != tp.SENT]
    assert t("L'analyse d\x92un système") == ["l", "analyse", "d", "un", "système"]
    assert tp.fix_text("d\x92analyse") == "d’analyse"
    assert t("تَحْلِيل النص") == ["تَحْلِيل", "النص"]
    assert t("हिन्दी भाषा") == ["हिन्दी", "भाषा"]
    assert t("深度学习 model") == ["深", "度", "学", "习", "model"]
    # a separating mark leaves a sentinel; a word-internal one does not
    assert tp.tokens("a, b") == ["a", tp.SENT, "b"]
    assert tp.tokens("e-commerce") == ["e", "commerce"]
