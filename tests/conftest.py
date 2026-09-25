"""Shared pytest fixtures across the whole test suite."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import pytest


def _is_empty_dir(entry: Path) -> bool:
    """True if *entry* is a directory containing no files anywhere within
    it, recursively."""
    return entry.is_dir() and not any(p.is_file() for p in entry.rglob("*"))


def _remove_empty_new_entries(data_dir: Path, before: set) -> None:
    """Remove every top-level entry of *data_dir* that is (a) not in
    *before* (i.e. appeared after the snapshot) and (b) a directory
    containing no files anywhere within it, recursively. Never touches an
    entry present in *before*, a non-directory entry, or a directory that
    ended up with real files in it."""
    if not data_dir.is_dir():
        return
    for entry in data_dir.iterdir():
        if entry in before:
            continue
        if _is_empty_dir(entry):
            shutil.rmtree(entry)


def _remove_stray_empty_entries(data_dir: Path) -> None:
    """Remove every top-level entry of *data_dir* that is CURRENTLY a
    directory containing no files anywhere within it, recursively --
    regardless of when it was created. Unlike :func:`_remove_empty_new_entries`,
    this has no "before" concept, so it also catches leftovers from a
    previous pytest session that never reached its own teardown (e.g. a
    killed/interrupted run): such an entry looks pre-existing to every
    later session's "before" snapshot, so only an unconditional sweep can
    remove it. Same safety property as :func:`_remove_empty_new_entries`:
    never touches a non-directory entry or a directory holding real
    files."""
    if not data_dir.is_dir():
        return
    for entry in data_dir.iterdir():
        if _is_empty_dir(entry):
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

    Two-phase sweep: (1) at session start, before snapshotting, calls
    :func:`_remove_stray_empty_entries` to self-heal any already-empty
    entry left over from a previous session that was interrupted before
    reaching its own teardown -- a purely "new since this session" check
    can never catch those. (2) at session end, delegates to
    :func:`_remove_empty_new_entries` for entries that appeared (and are
    still empty) during this session, as before. Both phases share the
    same safety property: an entry is only ever removed if it is
    currently empty -- any directory holding real downloaded data is
    never touched, whether pre-existing or created during this session."""
    data_dir = Path("data")
    _remove_stray_empty_entries(data_dir)

    before = set(data_dir.iterdir()) if data_dir.is_dir() else set()

    yield

    _remove_empty_new_entries(data_dir, before)


_recorded_exit_status: "Optional[int]" = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Record pytest's own exit status for pytest_unconfigure to use --
    see that hook's docstring for why."""
    global _recorded_exit_status
    _recorded_exit_status = int(exitstatus)


def pytest_unconfigure(config: pytest.Config) -> None:
    """
    In CI (Ubuntu, both Python 3.12 and 3.13), a fully green ``pytest -q``
    run reliably segfaults during process exit -- "N passed" is printed,
    then the shell reports SIGSEGV with no Python traceback (faulthandler
    never fires), the signature of a native library's own atexit/static
    destructor crashing during interpreter shutdown rather than a
    Python-level bug. This toolbox loads rasterio, pyproj, and cartopy in
    the same process, each bundling its own compiled GDAL/PROJ/GEOS; their
    exact resolved versions were confirmed identical between a clean local
    run and a crashing CI run, so this is a native cleanup ordering issue,
    not a dependency version to pin around.

    Only applies under CI (``CI`` is set by GitHub Actions and most other
    CI providers) -- a local pytest run keeps normal interpreter shutdown,
    so an actual crash during local development is never masked.
    ``os._exit()`` skips atexit callbacks and static destructors entirely,
    which is exactly the code path that segfaults; pytest's own exit
    status (captured by pytest_sessionfinish above) is preserved, so CI
    still fails on a genuine test failure. Remove this once a plain
    ``pytest -q`` run in CI no longer segfaults after printing its summary.
    """
    if os.environ.get("CI"):
        os._exit(_recorded_exit_status if _recorded_exit_status is not None else 0)
