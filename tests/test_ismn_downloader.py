"""Tests for ISMNDownloader (local ISMN archive selector)."""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sar_validation.downloaders.ismn_downloader import ISMNDownloader


@pytest.fixture(autouse=True)
def mock_ismn_shared_cache(monkeypatch, tmp_path):
    """Mock the shared ISMN archive cache and station-index cache to
    prevent tests from accidentally using the real cache directory.
    Tests that explicitly want to test either cache's real behavior
    override this fixture's monkeypatch locally with their own tmp_path."""
    import sar_validation.downloaders.ismn_downloader as ismn_mod

    # Set both caches to non-existent temp paths
    monkeypatch.setattr(ismn_mod, "_SHARED_ARCHIVE_CACHE_DIR", tmp_path / "mock_shared_cache")
    monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", tmp_path / "mock_station_index")


class TestISMNDownloaderNoArchive:
    def test_missing_archive_path_prints_instructions_and_returns_empty(self, capsys, tmp_path):
        dl = ISMNDownloader(output_dir=tmp_path)
        result = dl.download(
            min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
            start="2026-01-01", end="2026-01-02",
            min_depth=0.0, max_depth=0.05,
            archive_path=None,
        )
        assert result == []
        captured = capsys.readouterr()
        assert "ismn.earth" in captured.out
        assert "download_kwargs" in captured.out
        assert "CEOP" in captured.out
        assert "Good" in captured.out
        assert "Gap filling" in captured.out


def _real_shaped_sensor_meta(station, lon, lat, depth_from, depth_to):
    """
    Build a per-sensor metadata ``pandas.Series`` shaped exactly like the
    real ``ismn.meta.MetaData.to_pd()`` output that ``ISMN_Interface.read()``
    returns as its second value: a (variable, key)-style MultiIndex Series,
    NOT a flat dict.
    """
    index = pd.MultiIndex.from_tuples(
        [
            ("network", "val"),
            ("station", "val"),
            ("longitude", "val"),
            ("latitude", "val"),
            ("elevation", "val"),
            ("variable", "val"),
            ("variable", "depth_from"),
            ("variable", "depth_to"),
            ("instrument", "val"),
            ("instrument", "depth_from"),
            ("instrument", "depth_to"),
        ],
        names=["variable", "key"],
    )
    data = [
        "TESTNET", station, lon, lat, 100.0,
        "soil_moisture", depth_from, depth_to,
        "probe1", depth_from, depth_to,
    ]
    return pd.Series(data, index=index, name="data")


class TestISMNDownloaderFiltering:
    def _fake_reader(self):
        # ``reader.metadata`` is a real per-SENSOR row DataFrame (indexed by
        # the same integer sensor id used by get_dataset_ids/read), with
        # ("<name>", "val")-style MultiIndex columns. It is a plain
        # DataFrame -- there is no ``.to_pd()`` method on it.
        meta_df = pd.DataFrame(
            {
                ("longitude", "val"): [10.0, 100.0],
                ("latitude", "val"):  [45.0, -30.0],
            },
            index=[0, 1],
        )
        reader = MagicMock()
        reader.metadata = meta_df
        # get_dataset_ids returns sensor ids matching variable/depth,
        # independent of bbox -- both sensors match here.
        reader.get_dataset_ids.return_value = [0, 1]
        ts = pd.DataFrame(
            {"soil_moisture": [0.20, 0.21]},
            index=pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T12:00:00"]),
        )
        meta = _real_shaped_sensor_meta("station_in_bbox", 10.0, 45.0, 0.0, 0.05)
        reader.read.return_value = (ts, meta)
        return reader

    def test_writes_one_csv_per_sensor_in_bbox(self, tmp_path):
        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w"):
            pass  # valid, empty zip -- index_df is empty, so download() falls back to original archive path
        out_dir = tmp_path / "out"
        dl = ISMNDownloader(output_dir=out_dir)

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            written = dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                min_depth=0.0, max_depth=0.05,
                archive_path=str(archive),
            )

        # Only sensor id 0 is inside the bbox; sensor id 1 (lon=100, lat=-30)
        # is filtered out even though get_dataset_ids() "matched" it, since
        # bbox filtering intersects with the variable/depth match.
        assert len(written) == 1
        df = pd.read_csv(written[0])
        assert set(df["variable"]) == {"SOIL_MOISTURE"}
        assert set(df["platform_type"]) == {"ismn"}
        assert len(df) == 2
        assert set(df["lon"]) == {10.0}
        assert set(df["lat"]) == {45.0}
        assert set(df["depth"]) == {0.0}
        assert set(df["platform_id"]) == {"station_in_bbox"}

    def test_station_outside_bbox_is_excluded(self, tmp_path):
        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w"):
            pass  # valid, empty zip -- index_df is empty, so download() falls back to original archive path
        dl = ISMNDownloader(output_dir=tmp_path / "out")

        reader = self._fake_reader()
        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = reader

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            written = dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                archive_path=str(archive),
            )

        # sensor id 1 (station_outside_bbox, lon=100, lat=-30) must never be
        # read, since it's outside the bbox even though get_dataset_ids()
        # returned it as matching variable/depth.
        read_ids = [call.args[0] for call in reader.read.call_args_list]
        assert read_ids == [0]
        assert len(written) == 1

    def test_dry_run_writes_nothing(self, tmp_path):
        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w"):
            pass  # valid, empty zip -- index_df is empty, so download() falls back to original archive path
        out_dir = tmp_path / "out"
        dl = ISMNDownloader(output_dir=out_dir, dry_run=True)

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            written = dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                archive_path=str(archive),
            )

        assert written == []
        assert not out_dir.exists()
        # dry_run must bail out before ever constructing ISMN_Interface --
        # that call triggers ismn's own full sensor-level metadata scan
        # (slow, and floods the terminal with a per-station tqdm progress
        # bar on first run for a given archive+bbox), which defeats the
        # entire point of --dry-run being quick.
        fake_interface.ISMN_Interface.assert_not_called()


class TestArchiveResolutionPrecedence:
    """A user who downloads the ISMN zip from the portal shouldn't have to
    edit the recipe's download_kwargs at all -- dropping it directly into
    this run's own ISMN output folder should be picked up automatically.
    Precedence order when archive_path=None: explicit path > most-recently-
    modified local zip in output_dir > shared complete-archive cache."""

    def _fake_reader(self):
        meta_df = pd.DataFrame(
            {("longitude", "val"): [10.0], ("latitude", "val"): [45.0]},
            index=[0],
        )
        reader = MagicMock()
        reader.metadata = meta_df
        reader.get_dataset_ids.return_value = [0]
        ts = pd.DataFrame(
            {"soil_moisture": [0.20]},
            index=pd.to_datetime(["2026-01-01T00:00:00"]),
        )
        meta = _real_shaped_sensor_meta("station_in_bbox", 10.0, 45.0, 0.0, 0.05)
        reader.read.return_value = (ts, meta)
        return reader

    def _auto_detect(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        zip_path = out_dir / "Data_separate_files_20240703.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        return out_dir, None, zip_path

    def _most_recent_of_multiple(self, tmp_path, monkeypatch):
        import os
        import time

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        older = out_dir / "older.zip"
        newer = out_dir / "newer.zip"
        with zipfile.ZipFile(older, "w"):
            pass
        time.sleep(0.01)
        with zipfile.ZipFile(newer, "w"):
            pass
        now = time.time()
        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now, now))
        return out_dir, None, newer

    def _explicit_over_auto_detected(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        decoy = out_dir / "decoy.zip"
        with zipfile.ZipFile(decoy, "w"):
            pass
        explicit = tmp_path / "explicit_archive.zip"
        with zipfile.ZipFile(explicit, "w"):
            pass
        return out_dir, str(explicit), explicit

    def _shared_cache_fallback(self, tmp_path, monkeypatch):
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        shared_cache = tmp_path / "shared_cache"
        shared_cache.mkdir()
        shared_zip = shared_cache / "ISMN_archive_20260724.zip"
        with zipfile.ZipFile(shared_zip, "w"):
            pass
        monkeypatch.setattr(ismn_mod, "_SHARED_ARCHIVE_CACHE_DIR", shared_cache)
        out_dir = tmp_path / "out"  # empty -- no recipe-local zip
        return out_dir, None, shared_zip

    def _local_over_shared_cache(self, tmp_path, monkeypatch):
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        shared_cache = tmp_path / "shared_cache"
        shared_cache.mkdir()
        with zipfile.ZipFile(shared_cache / "ISMN_archive_20260724.zip", "w"):
            pass
        monkeypatch.setattr(ismn_mod, "_SHARED_ARCHIVE_CACHE_DIR", shared_cache)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        local_zip = out_dir / "Data_separate_files_20260101.zip"
        with zipfile.ZipFile(local_zip, "w"):
            pass
        return out_dir, None, local_zip

    @pytest.mark.parametrize(
        "setup_fn,check_written",
        [
            (_auto_detect, True),
            (_most_recent_of_multiple, False),
            (_explicit_over_auto_detected, False),
            (_shared_cache_fallback, True),
            (_local_over_shared_cache, False),
        ],
        ids=[
            "auto_detects_zip_dropped_in_output_dir",
            "prefers_most_recently_modified_zip_when_multiple_present",
            "explicit_archive_path_takes_priority_over_auto_detected_zip",
            "falls_back_to_shared_cache_when_no_local_zip",
            "local_zip_takes_priority_over_shared_cache",
        ],
    )
    def test_resolves_expected_archive(self, setup_fn, check_written, tmp_path, monkeypatch):
        out_dir, archive_path, expected_path = setup_fn(self, tmp_path, monkeypatch)
        dl = ISMNDownloader(output_dir=out_dir)

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        # check_written=True rows also cover the min_depth/max_depth kwargs and
        # the written-file count, matching the original standalone tests these
        # rows were folded from.
        download_kwargs = dict(
            min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
            start="2026-01-01", end="2026-01-02",
            archive_path=archive_path,
        )
        if check_written:
            download_kwargs.update(min_depth=0.0, max_depth=0.05)

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            written = dl.download(**download_kwargs)

        assert fake_interface.ISMN_Interface.call_args.args[0] == expected_path
        if check_written:
            assert len(written) == 1


class TestISMNDownloaderSingleReadingArchive:
    """A real-world archive downloaded for a single calendar day has every
    station .stm file containing exactly one reading. The ismn package's
    own file reader needs at least two lines per file to bootstrap
    metadata (it reads first/second/last line), so every file fails to
    parse and ISMN_Interface's construction raises a bare
    'ValueError: No objects to concatenate' with no indication of why.
    Confirmed against a real ISMN portal export (single-day request)."""

    def test_all_files_unparseable_raises_actionable_error(self, tmp_path):
        archive = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive, "w"):
            pass  # valid, empty zip -- index_df is empty, so download() falls back to original archive path
        dl = ISMNDownloader(output_dir=tmp_path / "out")

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.side_effect = ValueError("No objects to concatenate")

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            try:
                dl.download(
                    min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                    start="2026-01-01", end="2026-01-02",
                    archive_path=str(archive),
                )
                raise AssertionError("expected ValueError to propagate")
            except ValueError as exc:
                assert "single reading" in str(exc) or "date range" in str(exc)
                assert "dataviewer" in str(exc)


class TestISMNDownloaderSharedCacheFallback:
    """Age-warning and date-parsing tests for the shared-archive cache."""

    def _fake_reader(self):
        meta_df = pd.DataFrame(
            {("longitude", "val"): [10.0], ("latitude", "val"): [45.0]},
            index=[0],
        )
        reader = MagicMock()
        reader.metadata = meta_df
        reader.get_dataset_ids.return_value = [0]
        ts = pd.DataFrame(
            {"soil_moisture": [0.20]},
            index=pd.to_datetime(["2026-01-01T00:00:00"]),
        )
        meta = _real_shaped_sensor_meta("station_in_bbox", 10.0, 45.0, 0.0, 0.05)
        reader.read.return_value = (ts, meta)
        return reader

    @pytest.mark.parametrize(
        "age_days,expect_warning",
        [
            (95, True),
            (0, False),
        ],
        ids=["older_than_90_days_warns", "recent_archive_no_warning"],
    )
    def test_age_warning(self, tmp_path, monkeypatch, caplog, age_days, expect_warning):
        import logging
        import os
        import time

        import sar_validation.downloaders.ismn_downloader as ismn_mod

        shared_cache = tmp_path / "shared_cache"
        shared_cache.mkdir()
        zip_name = "ISMN_archive_old.zip" if expect_warning else "ISMN_archive_recent.zip"
        archive = shared_cache / zip_name
        with zipfile.ZipFile(archive, "w"):
            pass
        if age_days:
            old_time = time.time() - (age_days * 86400)
            os.utime(archive, (old_time, old_time))
        monkeypatch.setattr(ismn_mod, "_SHARED_ARCHIVE_CACHE_DIR", shared_cache)

        dl = ISMNDownloader(output_dir=tmp_path / "out")

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            with caplog.at_level(logging.WARNING):
                dl.download(
                    min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                    start="2026-01-01", end="2026-01-02",
                    archive_path=None,
                )

        has_warning = any("days old" in r.message for r in caplog.records)
        assert has_warning is expect_warning

    def test_prints_age_from_filename_date_not_mtime(self, tmp_path, monkeypatch, capsys):
        """A shared-cache archive named ISMN_archive_YYYYMMDD.zip must
        report its age from that embedded date, not the file's own mtime
        -- mtime resets on any copy/checkout/sync of the shared cache
        directory (observed in practice: a real cache file's mtime read
        "0 days old" even though its filename said it was over a week
        old), so it's an unreliable proxy whenever a name-embedded date
        is available."""
        import datetime
        import os
        import re
        import time

        import sar_validation.downloaders.ismn_downloader as ismn_mod

        shared_cache = tmp_path / "shared_cache"
        shared_cache.mkdir()
        archive_date = datetime.date.today() - datetime.timedelta(days=7)
        dated_zip = shared_cache / f"ISMN_archive_{archive_date:%Y%m%d}.zip"
        with zipfile.ZipFile(dated_zip, "w"):
            pass  # valid, empty zip -- index_df is empty, so download() falls back to original archive path
        # mtime deliberately set to "just now" -- if the code used mtime
        # instead of the filename date, it would report ~0 days old.
        now = time.time()
        os.utime(dated_zip, (now, now))
        monkeypatch.setattr(ismn_mod, "_SHARED_ARCHIVE_CACHE_DIR", shared_cache)

        dl = ISMNDownloader(output_dir=tmp_path / "out")

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                archive_path=None,
            )

        captured = capsys.readouterr().out
        # Not "0 days old" (mtime-based, wrong) -- 7 or 8, depending on
        # exactly where "now" falls relative to the archive date's own
        # midnight (day-boundary rounding), which proves the filename
        # date was used, not the mtime set to "just now" above.
        match = re.search(r"\((\d+) days old\)", captured)
        assert match is not None
        assert 7 <= int(match.group(1)) <= 8

    def test_falls_back_to_mtime_when_filename_has_no_date(self, tmp_path, monkeypatch, capsys):
        """A filename with no _YYYYMMDD suffix keeps today's mtime-based
        behavior unchanged (e.g. ISMN_archive_recent.zip, ISMN_archive_old.zip
        as used by the tests above)."""
        import os
        import time

        import sar_validation.downloaders.ismn_downloader as ismn_mod

        shared_cache = tmp_path / "shared_cache"
        shared_cache.mkdir()
        undated_zip = shared_cache / "ISMN_archive_recent.zip"
        with zipfile.ZipFile(undated_zip, "w"):
            pass
        old_time = time.time() - (3 * 86400)
        os.utime(undated_zip, (old_time, old_time))
        monkeypatch.setattr(ismn_mod, "_SHARED_ARCHIVE_CACHE_DIR", shared_cache)

        dl = ISMNDownloader(output_dir=tmp_path / "out")

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                archive_path=None,
            )

        captured = capsys.readouterr().out
        assert "(3 days old)" in captured


def _write_synthetic_ismn_archive(archive_path, stations):
    """stations: list of (network, station, lat_or_None, lon_or_None,
    extra_stm_count). lat/lon=None writes a deliberately malformed
    first line instead of a real CEOP header, to test the fail-safe
    path. extra_stm_count adds N additional .stm files under the same
    station directory (different sensor/depth), all sharing the same
    header line, to prove one row per STATION not per FILE."""
    with zipfile.ZipFile(archive_path, "w") as zf:
        for network, station, lat, lon, extra_stm_count in stations:
            if lat is None or lon is None:
                header = "this is not a valid CEOP header line\n"
            else:
                header = f"{network} {network} {station} {lat} {lon} 100.0 0.0000 0.0500 'Test-Sensor'\n"
            body = header + "2020/01/01 00:00 0.20 G M\n2020/01/02 00:00 0.21 G M\n"
            base = f"{network}/{station}/{network}_{network}_{station}"
            zf.writestr(f"{base}_sm_0.000000_0.050000_Test-Sensor_1_1_19500101_20260101.stm", body)
            for i in range(extra_stm_count):
                zf.writestr(f"{base}_ts_{i}_0.000000_0.050000_Test-Sensor_1_1_19500101_20260101.stm", body)


class TestBuildStationIndex:
    def test_one_row_per_station_directory_not_per_file(self, tmp_path):
        from sar_validation.downloaders.ismn_downloader import _build_station_index

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "Station1", 45.0, 10.0, 3)])

        index_df = _build_station_index(archive)

        assert len(index_df) == 1
        row = index_df.iloc[0]
        assert row["network"] == "NETA"
        assert row["station"] == "Station1"
        assert row["lat"] == pytest.approx(45.0)
        assert row["lon"] == pytest.approx(10.0)
        assert row["dir_prefix"] == "NETA/Station1/"

    def test_malformed_first_line_kept_with_nan_coords(self, tmp_path):
        from sar_validation.downloaders.ismn_downloader import _build_station_index

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("BADNET", "BadStation", None, None, 0)])

        index_df = _build_station_index(archive)

        assert len(index_df) == 1
        assert index_df.iloc[0]["dir_prefix"] == "BADNET/BadStation/"
        assert pd.isna(index_df.iloc[0]["lat"])
        assert pd.isna(index_df.iloc[0]["lon"])


class TestLoadOrBuildStationIndex:
    def test_builds_and_caches_on_first_call(self, tmp_path, monkeypatch):
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        cache_dir = tmp_path / "index_cache"
        monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", cache_dir)

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "Station1", 45.0, 10.0, 0)])

        index_df = ismn_mod._load_or_build_station_index(archive)

        assert len(index_df) == 1
        # Cache filename includes the archive's size/mtime fingerprint (not
        # just its stem) so a replaced archive with the same name gets a
        # fresh cache entry instead of silently reusing a stale one.
        expected_path = ismn_mod._station_index_path(archive)
        assert expected_path.exists()
        assert expected_path.name != f"station_index_{archive.stem}.csv"

    def test_second_call_loads_from_cache_without_reopening_zip(self, tmp_path, monkeypatch):
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        cache_dir = tmp_path / "index_cache"
        monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", cache_dir)

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "Station1", 45.0, 10.0, 0)])
        ismn_mod._load_or_build_station_index(archive)  # first call builds + caches

        with patch("zipfile.ZipFile") as mock_zip_cls:
            index_df = ismn_mod._load_or_build_station_index(archive)

        mock_zip_cls.assert_not_called()
        assert len(index_df) == 1
        assert index_df.iloc[0]["dir_prefix"] == "NETA/Station1/"


class TestExtractMatchingStations:
    def test_extracts_only_matching_station_files(self, tmp_path):
        from sar_validation.downloaders.ismn_downloader import _extract_matching_stations

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [
            ("NETA", "Station1", 45.0, 10.0, 0),
            ("NETB", "Station2", -30.0, 150.0, 0),
        ])
        out_dir = tmp_path / "subset"

        _extract_matching_stations(archive, {"NETA/Station1/"}, out_dir)

        assert (out_dir / "NETA" / "Station1").exists()
        assert list((out_dir / "NETA" / "Station1").glob("*.stm"))
        assert not (out_dir / "NETB").exists()

    def test_global_reference_files_always_copied_if_present(self, tmp_path):
        from sar_validation.downloaders.ismn_downloader import _extract_matching_stations

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "Station1", 45.0, 10.0, 0)])
        with zipfile.ZipFile(archive, "a") as zf:
            zf.writestr("ISMN_sensor_list.csv", "sensor_name;measured_variable\n5TE;soil moisture\n")

        out_dir = tmp_path / "subset"
        _extract_matching_stations(archive, {"NETA/Station1/"}, out_dir)

        assert (out_dir / "ISMN_sensor_list.csv").exists()


class TestISMNDownloaderBboxPreFilterWiring:
    def test_download_passes_extracted_subset_not_raw_archive_to_ISMN_Interface(
        self, tmp_path, monkeypatch,
    ):
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        cache_dir = tmp_path / "index_cache"
        monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", cache_dir)

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [
            ("NETA", "InBbox", 45.0, 10.0, 0),
            ("NETB", "OutBbox", -30.0, 150.0, 0),
        ])

        out_dir = tmp_path / "out"
        dl = ISMNDownloader(output_dir=out_dir)

        captured = {}
        fake_reader = MagicMock()
        fake_reader.metadata = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples([("longitude", "val"), ("latitude", "val")]),
        )
        fake_reader.get_dataset_ids.return_value = []

        def capture_and_return_reader(path, **kwargs):
            # Called while subset_dir still exists (before download() returns
            # and cleans it up) -- this is the only point at which the
            # extracted subset's contents can be observed from a test.
            captured["path"] = Path(path)
            captured["neta_exists"] = (Path(path) / "NETA" / "InBbox").exists()
            captured["netb_exists"] = (Path(path) / "NETB").exists()
            return fake_reader

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.side_effect = capture_and_return_reader

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            dl.download(
                min_lon=5.0, max_lon=15.0, min_lat=40.0, max_lat=50.0,
                start="2020-01-01", end="2020-01-02",
                archive_path=str(archive),
            )

        assert captured["path"] != archive
        assert captured["neta_exists"] is True
        assert captured["netb_exists"] is False
        # The extracted subset is a PERSISTENT cache now, not a scratch
        # temp dir -- it must still exist after download() returns, so a
        # second call (e.g. a --dry-run immediately followed by the real
        # run) can reuse it instead of re-extracting from scratch. The
        # completion marker specifically (not just the directory) is
        # what a real second call checks for before reusing it.
        assert captured["path"].exists()
        assert (captured["path"] / ismn_mod._EXTRACTION_COMPLETE_MARKER).exists()

    def test_second_call_with_same_archive_and_bbox_reuses_extracted_subset(
        self, tmp_path, monkeypatch,
    ):
        """Regression test: ismn's own internal metadata cache lives
        inside the extracted subset directory -- if that directory were
        recreated fresh on every call (e.g. a random tempfile.mkdtemp()
        target, the original design), ismn would rebuild its metadata
        from scratch every time, even for two calls with the identical
        archive and bbox (e.g. a --dry-run immediately followed by the
        real run). Proves the second call reuses the same directory
        rather than extracting again."""
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        cache_dir = tmp_path / "index_cache"
        monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", cache_dir)

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "InBbox", 45.0, 10.0, 0)])

        dl = ISMNDownloader(output_dir=tmp_path / "out")

        fake_reader = MagicMock()
        fake_reader.metadata = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples([("longitude", "val"), ("latitude", "val")]),
        )
        fake_reader.get_dataset_ids.return_value = []
        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = fake_reader

        kwargs = dict(
            min_lon=5.0, max_lon=15.0, min_lat=40.0, max_lat=50.0,
            start="2020-01-01", end="2020-01-02", archive_path=str(archive),
        )

        def fake_extract(archive_path, dir_prefixes, out_dir):
            # Real _extract_matching_stations creates out_dir (via
            # zf.extractall) and then writes the completion marker --
            # replicate both so download()'s reuse check (which looks
            # for the marker, not just directory existence, to guard
            # against reusing an interrupted extraction) behaves like
            # it would for a real extraction.
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / ismn_mod._EXTRACTION_COMPLETE_MARKER).touch()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}), \
             patch(
                 "sar_validation.downloaders.ismn_downloader._extract_matching_stations",
                 side_effect=fake_extract,
             ) as mock_extract:
            dl.download(**kwargs)
            first_call_path = fake_interface.ISMN_Interface.call_args.args[0]
            assert mock_extract.call_count == 1

            dl.download(**kwargs)
            second_call_path = fake_interface.ISMN_Interface.call_args.args[0]
            # Must NOT extract again -- that's what lets ismn's own
            # internal metadata cache (written inside this directory)
            # survive between the two calls.
            assert mock_extract.call_count == 1

        assert first_call_path == second_call_path

    def test_interrupted_extraction_without_completion_marker_is_redone(
        self, tmp_path, monkeypatch,
    ):
        """Regression test: an extraction directory that exists but was
        never completed (e.g. the process was killed mid-extractall,
        plausible given the real archive is multi-GB) must be
        re-extracted, not silently reused as if it were a valid,
        complete station subset."""
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        cache_dir = tmp_path / "index_cache"
        monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", cache_dir)

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "InBbox", 45.0, 10.0, 0)])

        dl = ISMNDownloader(output_dir=tmp_path / "out")

        # Pre-create the deterministic subset directory WITHOUT the
        # completion marker -- simulates a prior extraction that was
        # interrupted partway through.
        subset_dir = ismn_mod._extracted_subset_dir(archive, 5.0, 15.0, 40.0, 50.0)
        subset_dir.mkdir(parents=True)
        (subset_dir / "partial_leftover.txt").write_text("incomplete")
        assert not (subset_dir / ismn_mod._EXTRACTION_COMPLETE_MARKER).exists()

        fake_reader = MagicMock()
        fake_reader.metadata = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples([("longitude", "val"), ("latitude", "val")]),
        )
        fake_reader.get_dataset_ids.return_value = []
        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = fake_reader

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            dl.download(
                min_lon=5.0, max_lon=15.0, min_lat=40.0, max_lat=50.0,
                start="2020-01-01", end="2020-01-02",
                archive_path=str(archive),
            )

        # Real extraction ran (not skipped) -- the in-bbox station's
        # real files are now present, proving _extract_matching_stations
        # was actually invoked rather than trusting the stale directory.
        assert (subset_dir / "NETA" / "InBbox").exists()
        assert (subset_dir / ismn_mod._EXTRACTION_COMPLETE_MARKER).exists()

    def test_no_stations_in_bbox_returns_empty_without_calling_ISMN_Interface(
        self, tmp_path, monkeypatch,
    ):
        import sar_validation.downloaders.ismn_downloader as ismn_mod

        cache_dir = tmp_path / "index_cache"
        monkeypatch.setattr(ismn_mod, "_STATION_INDEX_CACHE_DIR", cache_dir)

        archive = tmp_path / "synthetic.zip"
        _write_synthetic_ismn_archive(archive, [("NETA", "FarAway", -80.0, 179.0, 0)])

        dl = ISMNDownloader(output_dir=tmp_path / "out")

        fake_interface = MagicMock()
        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            result = dl.download(
                min_lon=5.0, max_lon=15.0, min_lat=40.0, max_lat=50.0,
                start="2020-01-01", end="2020-01-02",
                archive_path=str(archive),
            )

        assert result == []
        fake_interface.ISMN_Interface.assert_not_called()


class TestStationDateRangesDry:
    def test_station_date_ranges_dry_reads_first_and_last_timestamp(self, tmp_path):
        # Build a minimal fake archive: a real .zip (matching the real
        # archive format -- ISMN has no download API, so the toolbox
        # always reads a manually-downloaded zip export, never an
        # extracted directory tree) containing one network/station's
        # .stm file with a header line + two data rows.
        # Header line follows the real CEOP layout confirmed against a
        # live ISMN archive (see _build_station_index's own docstring):
        # NETWORK SUBNETWORK STATION LAT LON ELEV DEPTH_FROM DEPTH_TO
        # 'SENSOR' -- lat/lon are tokens[3]/tokens[4].
        archive_path = tmp_path / "fake_ismn.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr(
                "FakeNetwork/FakeStation/FakeNetwork_FakeStation_sm_0.000_0.050_sensor_20260101_20260105.stm",
                "FakeNetwork FakeNetwork FakeStation 45.0 10.0 100 0.000 0.050 'sensor'\n"
                "2026/01/01 00:00 0.25 U\n"
                "2026/01/05 23:00 0.30 U\n",
            )

        dl = ISMNDownloader(output_dir=tmp_path / "out")
        ranges = dl.station_date_ranges_dry(
            min_lon=0.0, max_lon=20.0, min_lat=40.0, max_lat=50.0, archive_path=str(archive_path),
        )

        assert ranges is not None
        # Exact dir_prefix key format follows _build_station_index's own
        # convention (network/station/, matching the zip's directory
        # structure) -- the (lat, lon, earliest, latest) tuple shape is
        # the load-bearing part. lat=45.0, lon=10.0 come from the .stm
        # header line's own embedded coordinates (matching
        # _build_station_index's existing lat/lon extraction, reused
        # here -- not re-parsed from the header a second time).
        (only_entry,) = ranges.values()
        assert only_entry == (45.0, 10.0, datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 5, 23, 0))

    def test_station_date_ranges_dry_returns_none_when_no_archive_present(self, tmp_path, monkeypatch):
        import sar_validation.downloaders.ismn_downloader as ismn_module

        monkeypatch.setattr(ismn_module, "_SHARED_ARCHIVE_CACHE_DIR", tmp_path / "nonexistent")

        dl = ISMNDownloader(output_dir=tmp_path / "out")
        ranges = dl.station_date_ranges_dry(min_lon=0.0, max_lon=20.0, min_lat=40.0, max_lat=50.0)

        assert ranges is None
