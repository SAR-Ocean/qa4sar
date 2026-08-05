"""Shared pytest fixtures across the whole test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _remove_empty_new_entries(data_dir: Path, before: set) -> None:
    """Remove every top-level entry of *data_dir* that is (a) not in
    *before* (i.e. appeared after the snapshot) and (b) a directory
    containing no files anywhere within it, recursively. Never touches an
    entry present in *before*, a non-directory entry, or a directory that
    ended up with real files in it."""
    if not data_dir.is_dir():
        return
    for entry in data_dir.iterdir():
        if entry in before or not entry.is_dir():
            continue
        if not any(p.is_file() for p in entry.rglob("*")):
            shutil.rmtree(entry)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_stray_empty_data_dirs():
    """Some tests construct a real ``DataOrchestrator(..., dry_run=False)``
    with a hardcoded recipe (rather than routing every path through
    ``tmp_path``) before redirecting ``orchestrator.base_dir`` elsewhere --
    ``DataOrchestrator.__init__``'s ``_setup_base_dir()`` already created a
    real, empty output directory under ``data/`` as a side effect by then
    (``base.mkdir(parents=True, exist_ok=True)``), and nothing else ever
    removes it. Left unchecked this clutters ``data/`` with more stray
    empty folders every time the suite runs.

    Snapshots ``data/``'s existing top-level entries before the session,
    then delegates to :func:`_remove_empty_new_entries` once the session
    ends."""
    data_dir = Path("data")
    before = set(data_dir.iterdir()) if data_dir.is_dir() else set()

    yield

    _remove_empty_new_entries(data_dir, before)
