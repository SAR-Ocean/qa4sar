"""Tests for tests/conftest.py's own helper functions."""

from __future__ import annotations


class TestRemoveEmptyNewEntries:
    def test_removes_new_empty_directory(self, tmp_path):
        from tests.conftest import _remove_empty_new_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        before = set(data_dir.iterdir())
        new_empty = data_dir / "2026-01-01-000000-2026-01-02-000000_-10.00_20.00_40.00_55.00"
        new_empty.mkdir()

        _remove_empty_new_entries(data_dir, before)

        assert not new_empty.exists()

    def test_keeps_pre_existing_empty_directory(self, tmp_path):
        from tests.conftest import _remove_empty_new_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pre_existing = data_dir / "pre_existing_run"
        pre_existing.mkdir()
        before = set(data_dir.iterdir())

        _remove_empty_new_entries(data_dir, before)

        assert pre_existing.exists()

    def test_keeps_new_directory_containing_a_file(self, tmp_path):
        from tests.conftest import _remove_empty_new_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        before = set(data_dir.iterdir())
        new_populated = data_dir / "real_run"
        (new_populated / "sub").mkdir(parents=True)
        (new_populated / "sub" / "file.nc").write_bytes(b"data")

        _remove_empty_new_entries(data_dir, before)

        assert new_populated.exists()

    def test_keeps_new_non_directory_entry(self, tmp_path):
        from tests.conftest import _remove_empty_new_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        before = set(data_dir.iterdir())
        stray_file = data_dir / "stray.txt"
        stray_file.write_text("not a directory")

        _remove_empty_new_entries(data_dir, before)

        assert stray_file.exists()

    def test_noop_when_data_dir_does_not_exist(self, tmp_path):
        from tests.conftest import _remove_empty_new_entries

        missing = tmp_path / "does_not_exist"
        _remove_empty_new_entries(missing, set())  # must not raise


class TestRemoveStrayEmptyEntries:
    """_remove_stray_empty_entries self-heals leftovers from a previous
    pytest session that never reached its own teardown (e.g. a
    killed/interrupted run) -- unlike _remove_empty_new_entries, it has
    no "before" concept: it removes ANY currently-empty top-level
    directory, regardless of when it was created."""

    def test_removes_pre_existing_empty_directory(self, tmp_path):
        from tests.conftest import _remove_stray_empty_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        stray = data_dir / "2026-01-01-000000-2026-01-02-000000_-10.00_20.00_40.00_55.00"
        stray.mkdir()

        _remove_stray_empty_entries(data_dir)

        assert not stray.exists()

    def test_keeps_pre_existing_directory_containing_a_file(self, tmp_path):
        from tests.conftest import _remove_stray_empty_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        populated = data_dir / "real_run"
        (populated / "sub").mkdir(parents=True)
        (populated / "sub" / "file.nc").write_bytes(b"data")

        _remove_stray_empty_entries(data_dir)

        assert populated.exists()

    def test_keeps_non_directory_entry(self, tmp_path):
        from tests.conftest import _remove_stray_empty_entries

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        stray_file = data_dir / "stray.txt"
        stray_file.write_text("not a directory")

        _remove_stray_empty_entries(data_dir)

        assert stray_file.exists()

    def test_noop_when_data_dir_does_not_exist(self, tmp_path):
        from tests.conftest import _remove_stray_empty_entries

        missing = tmp_path / "does_not_exist"
        _remove_stray_empty_entries(missing)  # must not raise
