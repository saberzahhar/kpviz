"""Runtime configuration and the CPU/RAM budget.

The budget matters more than any single micro-optimisation: worker processes,
DuckDB's own thread pool, and the nested thread pools inside spaCy /
tokenizers / BLAS will each happily claim the whole machine. Left alone on a
32-thread box that is ~1000 runnable threads for 32 cores. Everything is
allocated from one budget here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .hostinfo import usable_cpus, usable_ram_bytes


@dataclass
class Settings:
    # Root of the data tree (documents/, architectures/, models/, insights/,
    # inferences/).
    data_root: Path
    # Private state directory (DuckDB store, tokenizer cache, spill, stats).
    state_dir: Path
    workers: int = 0            # 0 -> derived from the budget
    io_workers: int = 0         # 0 -> derived (hashing / stat fan-out)
    db_threads: int = 0         # 0 -> derived (DuckDB during a scan)
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    # Scope of the expensive derivations. "eval" = testing/validation splits
    # only (runs are evaluated on those), "all" = every split. Distribution
    # charts always cover every split through gold_agg regardless.
    pos_scope: str = "eval"       # kept for CLI compatibility
    token_scope: str = "eval"     # per-tokenizer doc token counts + kp positions
    gold_scope: str = "eval"      # gold *instance* rows (+ always: flagged docs)
    # "auto": content-hash a file only when size/mtime moved (nothing to
    # compare against on a first scan, so hashing it then is pure I/O cost).
    # "always": hash every file every scan.
    hash_mode: str = "auto"
    # Upper bound for one worker task; the effective size also fans small
    # files across the pool (see scanner._eff_chunk).
    chunk_bytes: int = 32 * 1024 * 1024

    def __post_init__(self):
        cpus = usable_cpus()
        if not self.workers:
            self.workers = max(1, cpus)
        self.workers = max(1, min(self.workers, 4096))
        if not self.io_workers:
            # hashing releases the GIL; I/O concurrency can exceed core count
            self.io_workers = max(4, min(64, self.workers * 2))
        if not self.db_threads:
            # leave the pool the cores; DuckDB gets the remainder, min 2
            self.db_threads = max(2, cpus - self.workers) if cpus > self.workers \
                else max(2, cpus // 4)

    # -- derived paths ------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.state_dir / "kpviz.duckdb"

    @property
    def tmp_dir(self) -> Path:
        return self.state_dir / "tmp"

    @property
    def tokenizer_cache(self) -> Path:
        return self.state_dir / "tokenizers"

    @property
    def stats_dir(self) -> Path:
        return self.state_dir / "scan_stats"

    # -- RAM budget ---------------------------------------------------------
    @property
    def duckdb_memory_bytes(self) -> int:
        """What DuckDB may use *while the pool is running*.

        Workers hold parsed rows, phrase caches and (for context-window
        analysis) tokenizer state, so they are budgeted too — they are not
        idle while ingestion happens."""
        total = usable_ram_bytes()
        per_worker = 384 * 2**20          # measured peak is well under this
        worker_budget = min(total * 0.45, self.workers * per_worker)
        return int(max(1 * 2**30, total * 0.75 - worker_budget))

    @property
    def phrase_cache_max(self) -> int:
        """Worker-local phrase cache bound, scaled to the RAM each worker may
        use (a cached phrase entry is ~700 B of Python objects)."""
        per_worker = usable_ram_bytes() * 0.45 / max(1, self.workers)
        return int(max(50_000, min(4_000_000, per_worker * 0.35 / 700)))

    def ensure_dirs(self) -> None:
        for p in (self.state_dir, self.tmp_dir, self.tokenizer_cache,
                  self.stats_dir):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        return {"workers": self.workers, "io_workers": self.io_workers,
                "db_threads": self.db_threads,
                "usable_cpus": usable_cpus(),
                "usable_ram_gb": round(usable_ram_bytes() / 2**30, 1),
                "duckdb_memory_gb": round(self.duckdb_memory_bytes / 2**30, 1),
                "phrase_cache_max": self.phrase_cache_max,
                "chunk_bytes": self.chunk_bytes,
                "hash_mode": self.hash_mode,
                "token_scope": self.token_scope, "gold_scope": self.gold_scope}


_SETTINGS: Settings | None = None


def init_settings(data_root: str | Path, **kw) -> Settings:
    global _SETTINGS
    root = Path(data_root).resolve()
    state = Path(kw.pop("state_dir", root.parent / ".kpviz")).resolve()
    _SETTINGS = Settings(data_root=root, state_dir=state, **kw)
    _SETTINGS.ensure_dirs()
    return _SETTINGS


def settings() -> Settings:
    if _SETTINGS is None:
        raise RuntimeError("init_settings() must be called first")
    return _SETTINGS
