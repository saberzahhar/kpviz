"""Shared fixtures: one self-contained sample tree and one scanned store per
test session (about 20 s). Everything is generated — no downloads, no
external data — so `pytest` runs from a fresh clone.

Scans run in subprocesses (the process pool needs an importable __main__
and settings are process-global); read-side tests open the store in-process
afterwards.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
os.environ.setdefault("KPVIZ_OFFLINE", "1")      # tests never touch the network


def run(*args, **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO))
    res = subprocess.run([PY, *map(str, args)], cwd=kw.pop("cwd", REPO),
                         env=env, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise AssertionError(f"{args} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return res


def scan(data: Path, state: Path, workers: int = 4, full: bool = False) -> str:
    args = [REPO / "tools" / "scan_once.py", "--data", data, "--state", state,
            "--workers", workers]
    if full:
        args.append("--full")
    return run(*args).stdout


@pytest.fixture(scope="session")
def sample_tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tree")
    run(REPO / "tools" / "make_sample_data.py", "--src", root / "sample_src",
        "--out", root / "sample_data")
    return root / "sample_data"


@pytest.fixture(scope="session")
def store(sample_tree, tmp_path_factory) -> Path:
    state = tmp_path_factory.mktemp("state")
    scan(sample_tree, state, workers=4)
    return state


@pytest.fixture(scope="session")
def app_ctx(sample_tree, store):
    """The scanned store opened in this process (read side)."""
    sys.path.insert(0, str(REPO))
    from kpviz.config import init_settings
    init_settings(sample_tree, state_dir=store)
    from kpviz import db
    db.connect()
    return db
