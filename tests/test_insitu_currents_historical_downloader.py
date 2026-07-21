"""Tests for InSituCurrentsHistoricalDownloader (ADCP/Argo/drifter/glider, delayed-mode)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.downloaders.insitu_currents_historical_downloader import (
    _DATASET_IDS,
    InSituCurrentsHistoricalDownloader,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -20.0, 0.0, 35.0, 60.0


class TestConstruction:
    def test_unknown_instrument_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown instrument"):
            InSituCurrentsHistoricalDownloader(instrument="mooring", output_dir=tmp_path)

    @pytest.mark.parametrize("instrument,dataset_id", sorted(_DATASET_IDS.items()))
    def test_each_instrument_maps_to_its_dataset_id(self, instrument, dataset_id, tmp_path):
        dl = InSituCurrentsHistoricalDownloader(instrument=instrument, output_dir=tmp_path)
        assert _DATASET_IDS[dl.instrument] == dataset_id


class TestRecencyGuard:
    def test_recent_end_date_returns_empty_without_touching_network(self, tmp_path, caplog):
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=100)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=101)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="adcp", output_dir=tmp_path)
        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.WARNING):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        fake_module.subset.assert_not_called()
        assert any("less than" in r.message and "days old" in r.message for r in caplog.records)

    def test_old_enough_end_date_proceeds(self, tmp_path):
        from pathlib import Path

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="argo", output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            # Simulate copernicusmarine writing the requested file — the
            # downloader checks dest_path.exists() after the call.
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        fake_module.subset.assert_called_once()


class TestSubsetCall:
    def test_variables_fixed_to_ewct_nsct(self, tmp_path):
        from pathlib import Path

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="glider", output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert fake_module.subset.call_args.kwargs["variables"] == ["EWCT", "NSCT"]
        assert fake_module.subset.call_args.kwargs["dataset_id"] == _DATASET_IDS["glider"]

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        from pathlib import Path

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="drifter", output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert "force_download" not in fake_module.subset.call_args.kwargs
        assert fake_module.subset.call_args.kwargs["skip_existing"] is True


class TestForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="adcp", output_dir=tmp_path)
        dataset_id = _DATASET_IDS["adcp"]
        # start_d != end_d here (201 days ago vs 200 days ago are different
        # calendar dates), so the downloader's date_str joins both — must
        # match _download_window's `date_str = start_d if start_d == end_d
        # else f"{start_d}-{end_d}"` exactly, or this pre-created file won't
        # be found and the skip-existing path won't trigger.
        dest_path = tmp_path / f"{dataset_id}_{start}-{end}.csv"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("time,EWCT,NSCT\n")

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        fake_module.subset.assert_not_called()
        assert out == [dest_path]


class TestDryRun:
    def test_dry_run_prints_and_calls_no_subset(self, tmp_path, capsys):
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="adcp", output_dir=tmp_path, dry_run=True)
        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        fake_module.subset.assert_not_called()
        assert "DRY RUN" in capsys.readouterr().out
