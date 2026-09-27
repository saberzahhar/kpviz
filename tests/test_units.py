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
    low = tp.norm_text(text)
    m = tp.norm_offsets(text, low)
    toks, ends = tp.spacy_doc_tokens(text, "en", low)
    assert [text[:m[e]].split()[-1] for e in ends][-1] == "Straße"


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
