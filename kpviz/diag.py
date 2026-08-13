"""Which fast paths are actually active.

A silent fallback is the worst failure mode this tool has: PyStemmer missing
turns minutes into hours, a missing tokenizer asset turns exact token counts
into flagged approximations, and neither announces itself. Everything here is
cheap to probe and is reported at startup and on the Overview page.
"""
from __future__ import annotations

import importlib.util
import shutil
from functools import lru_cache

_LEVEL_OK, _LEVEL_WARN, _LEVEL_INFO = "ok", "warn", "info"


def _probe_stemmer():
    try:
        import Stemmer  # noqa: F401
        return "PyStemmer (C)", _LEVEL_OK, ""
    except Exception:
        pass
    if importlib.util.find_spec("snowballstemmer"):
        return ("snowballstemmer (pure Python)", _LEVEL_WARN,
                "orders of magnitude slower — pip install PyStemmer")
    return "none (identity)", _LEVEL_WARN, "stemming disabled — pip install PyStemmer"


def _probe_json():
    if importlib.util.find_spec("orjson"):
        return "orjson", _LEVEL_OK, ""
    return "stdlib json", _LEVEL_WARN, "pip install orjson for faster jsonl parsing"


def _probe_spacy():
    if not importlib.util.find_spec("spacy"):
        return "missing", _LEVEL_WARN, "tokenisation falls back to regex; no POS"
    from .textproc import _SPACY_MODELS
    have = [l for l, m in _SPACY_MODELS.items()
            if importlib.util.find_spec(m) is not None]
    if not have:
        return ("spaCy, no POS models", _LEVEL_WARN,
                "install e.g. en_core_web_sm for POS patterns")
    return "spaCy + " + ", ".join(sorted(have)), _LEVEL_OK, ""


def _probe_tokenizers():
    bits, level, note = [], _LEVEL_OK, ""
    if importlib.util.find_spec("tokenizers"):
        bits.append("transformers[…]")
    else:
        level, note = _LEVEL_WARN, "pip install tokenizers"
    if importlib.util.find_spec("tiktoken"):
        bits.append("tiktoken[…]")
    else:
        level, note = _LEVEL_WARN, (note + " / tiktoken").strip(" /")
    return (", ".join(bits) or "none", level,
            note or "exact counts when the asset is cached; flagged otherwise")


def _probe_tex():
    for eng in ("lualatex", "xelatex", "pdflatex"):
        if shutil.which(eng):
            return f"{eng} available", _LEVEL_OK, "probed lazily on first export"
    return ("none", _LEVEL_INFO,
            "PDF/PNG still export via Matplotlib; .pgf disabled")


def _probe_duckdb():
    try:
        import duckdb
        return f"duckdb {duckdb.__version__}", _LEVEL_OK, ""
    except Exception:
        return "missing", _LEVEL_WARN, ""


@lru_cache(maxsize=1)
def backends() -> dict:
    out = {}
    for name, fn in (("stemmer", _probe_stemmer), ("json", _probe_json),
                     ("tokenizers", _probe_tokenizers), ("spaCy", _probe_spacy),
                     ("duckdb", _probe_duckdb), ("TeX", _probe_tex)):
        value, level, note = fn()
        out[name] = {"value": value, "level": level, "note": note}
    return out


def print_banner(st) -> None:
    """Startup diagnostics on stdout (never during a scan)."""
    b = backends()
    width = max(len(k) for k in b)
    for name, info in b.items():
        mark = {"ok": "·", "warn": "!", "info": "·"}[info["level"]]
        line = f" {mark} {name.ljust(width)}  {info['value']}"
        if info["level"] == "warn" and info["note"]:
            line += f"   <- {info['note']}"
        print(line)
    d = st.describe()
    print(f" · budget      {d['workers']} workers · {d['db_threads']} db threads"
          f" · {d['usable_cpus']} usable cpus"
          f" · {d['duckdb_memory_gb']} GB duckdb of {d['usable_ram_gb']} GB")
