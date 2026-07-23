"""Tests for ISMNDownloader (local ISMN archive selector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from sar_validation.downloaders.ismn_downloader import ISMNDownloader


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

    def test_nonexistent_archive_path_prints_instructions_and_returns_empty(self, capsys, tmp_path):
        dl = ISMNDownloader(output_dir=tmp_path)
        result = dl.download(
            min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
            start="2026-01-01", end="2026-01-02",
            archive_path=str(tmp_path / "does_not_exist.zip"),
        )
        assert result == []
        assert "ismn.earth" in capsys.readouterr().out


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
        archive.write_bytes(b"")
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
        archive.write_bytes(b"")
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
        archive.write_bytes(b"")
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


class TestISMNDownloaderAutoDetectArchive:
    """A user who downloads the ISMN zip from the portal shouldn't have to
    edit the recipe's download_kwargs at all -- dropping it directly into
    this run's own ISMN output folder should be picked up automatically."""

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

    def test_auto_detects_zip_dropped_in_output_dir(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        zip_path = out_dir / "Data_separate_files_20240703.zip"
        zip_path.write_bytes(b"")
        dl = ISMNDownloader(output_dir=out_dir)

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            written = dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                min_depth=0.0, max_depth=0.05,
                archive_path=None,
            )

        assert len(written) == 1
        assert fake_interface.ISMN_Interface.call_args.args[0] == zip_path

    def test_no_zip_in_output_dir_still_prints_instructions(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        dl = ISMNDownloader(output_dir=out_dir)

        result = dl.download(
            min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
            start="2026-01-01", end="2026-01-02",
            archive_path=None,
        )

        assert result == []
        assert "ismn.earth" in capsys.readouterr().out

    def test_prefers_most_recently_modified_zip_when_multiple_present(self, tmp_path):
        import os
        import time

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        older = out_dir / "older.zip"
        newer = out_dir / "newer.zip"
        older.write_bytes(b"")
        time.sleep(0.01)
        newer.write_bytes(b"")
        now = time.time()
        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now, now))
        dl = ISMNDownloader(output_dir=out_dir)

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                archive_path=None,
            )

        assert fake_interface.ISMN_Interface.call_args.args[0] == newer

    def test_explicit_archive_path_takes_priority_over_auto_detected_zip(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        decoy = out_dir / "decoy.zip"
        decoy.write_bytes(b"")
        explicit = tmp_path / "explicit_archive.zip"
        explicit.write_bytes(b"")
        dl = ISMNDownloader(output_dir=out_dir)

        fake_interface = MagicMock()
        fake_interface.ISMN_Interface.return_value = self._fake_reader()

        with patch.dict("sys.modules", {"ismn": MagicMock(), "ismn.interface": fake_interface}):
            dl.download(
                min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
                start="2026-01-01", end="2026-01-02",
                archive_path=str(explicit),
            )

        assert fake_interface.ISMN_Interface.call_args.args[0] == explicit

    def test_instructions_mention_output_dir_path(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        dl = ISMNDownloader(output_dir=out_dir)

        dl.download(
            min_lon=-10.0, max_lon=20.0, min_lat=40.0, max_lat=55.0,
            start="2026-01-01", end="2026-01-02",
            archive_path=None,
        )

        assert str(out_dir) in capsys.readouterr().out


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
        archive.write_bytes(b"")
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
