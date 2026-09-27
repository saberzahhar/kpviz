"""Imported once by the forkserver, before any scan worker is forked.

Every scan worker forks from the forkserver, so what is imported and built
here is paid once per server process instead of once per worker per scan:
spaCy itself, and the blank tokenizers of the languages the catalog
declares. French alone is ~3.8 s per worker (spaCy compiles one very large
tokenizer-exception regex at import).

Thread pools are pinned *before* NumPy/spaCy load here: OpenBLAS and Rayon
size their pools at load time, which is before `derive.init_worker` runs in
a child, and forking a process with live BLAS threads is what the pinning
avoids. This module runs in the forkserver only; the server process keeps
its own settings.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from . import derive, textproc  # noqa: F401  (the worker entry points)
    for _lang in filter(None, os.environ.get("KPVIZ_PRELOAD_LANGS", "").split(",")):
        try:
            textproc._blank(_lang)
            textproc.stem_tokens(["preload"], _lang)
        except Exception:
            pass
except Exception:                      # a failed preload only costs speed
    pass
