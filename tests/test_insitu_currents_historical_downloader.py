"""Tests for InSituCurrentsHistoricalDownloader (ADCP/Argo/drifter/glider, delayed-mode)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sar_validation.downloaders.insitu_currents_historical_downloader import (
    _DATASET_IDS,
    _VARIABLES,
    InSituCurrentsHistoricalDownloader,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -20.0, 0.0, 35.0, 60.0


class _FakeCopernicusMarineModule:
    """Minimal stand-in for the copernicusmarine module, exposing just
    the read_dataframe() call check_availability_dry uses -- mirrors
    InSituDownloader's own test fixture of the same name
    (test_insitu_downloader.py)."""

    def __init__(self, has_data: bool):
        self._has_data = has_data

    def read_dataframe(self, **kwargs):
        if not self._has_data:
            return pd.DataFrame()
        return pd.DataFrame({"EWCT": [0.1], "NSCT": [0.2]})


class TestConstruction:
    def test_unknown_instrument_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown instrument"):
            InSituCurrentsHistoricalDownloader(instrument="mooring", output_dir=tmp_path)


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
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_text(
                "time,EWCT,NSCT\n2024-01-01T00:00:00,0.1,0.2\n"
            )

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
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_text(
                "time,EWCT,NSCT\n2024-01-01T00:00:00,0.1,0.2\n"
            )

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert fake_module.subset.call_args.kwargs["variables"] == ["EWCT", "NSCT"]
        assert fake_module.subset.call_args.kwargs["dataset_id"] == _DATASET_IDS["glider"]

    def test_force_download_kwarg_never_passed_and_result_path_returned(self, tmp_path):
        from pathlib import Path

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="drifter", output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_text(
                "time,EWCT,NSCT\n2024-01-01T00:00:00,0.1,0.2\n"
            )

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert "force_download" not in fake_module.subset.call_args.kwargs
        assert fake_module.subset.call_args.kwargs["skip_existing"] is True
        assert len(out) == 1
        assert out[0].exists()

    def test_empty_subset_result_returns_empty_and_removes_file(self, tmp_path, caplog):
        from pathlib import Path

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="drifter", output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            # copernicusmarine "succeeds" but the requested window has no
            # matching rows — only the CSV header is written.
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_text(
                "time,EWCT,NSCT\n"
            )

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.DEBUG):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        assert list(tmp_path.glob("*.csv")) == []
        # This message is intentionally logged at DEBUG, not WARNING: the
        # orchestrator now owns the user-facing "no data" message for the
        # four delayed-mode currents instruments combined (one message
        # instead of four) -- see _report_combined_currents_status.
        assert any("drifter" in r.message and "No" in r.message for r in caplog.records)

    def test_no_file_written_at_all_is_treated_as_no_data_not_a_failure(self, tmp_path, caplog):
        """Real-world Copernicus Marine behaviour (confirmed 2026-07-23 by a
        live DeltaEbro run): for a genuinely empty result, subset() sometimes
        writes no output file at all -- not even a header-only CSV -- rather
        than always producing the empty-CSV case covered by
        test_empty_subset_result_returns_empty_and_removes_file above. This
        used to raise FileNotFoundError, which orchestrator.py treats as a
        hard failure: it skips _cleanup_if_empty (only called on the success
        path), leaving an empty output directory behind, and it reports
        "failed" instead of the intended combined "no data" message. Must
        return None like the empty-CSV case, not raise."""
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=201)).strftime("%Y-%m-%d")

        dl = InSituCurrentsHistoricalDownloader(instrument="adcp", output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            pass  # "succeeds" but writes nothing, per the real observed behaviour

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.DEBUG):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        assert list(tmp_path.glob("*.csv")) == []
        assert any("adcp" in r.message and "No" in r.message for r in caplog.records)


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


class TestCheckAvailabilityDry:
    def test_returns_true_when_data_exists(self, tmp_path):
        fake_module = _FakeCopernicusMarineModule(has_data=True)

        dl = InSituCurrentsHistoricalDownloader(instrument="adcp", output_dir=tmp_path)
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            result = dl.check_availability_dry(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2024-01-01T00:00:00", "2024-01-02T00:00:00",
            )

        assert result is True

    def test_returns_false_when_no_data(self, tmp_path):
        fake_module = _FakeCopernicusMarineModule(has_data=False)

        dl = InSituCurrentsHistoricalDownloader(instrument="argo", output_dir=tmp_path)
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            result = dl.check_availability_dry(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2024-01-01T00:00:00", "2024-01-02T00:00:00",
            )

        assert result is False

    def test_queries_correct_dataset_id_and_variables(self, tmp_path):
        fake_module = MagicMock()
        fake_module.read_dataframe.return_value = pd.DataFrame({"EWCT": [0.1]})

        dl = InSituCurrentsHistoricalDownloader(instrument="glider", output_dir=tmp_path)
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.check_availability_dry(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2024-01-01T00:00:00", "2024-01-02T00:00:00",
            )

        kwargs = fake_module.read_dataframe.call_args.kwargs
        assert kwargs["dataset_id"] == _DATASET_IDS["glider"]
        assert kwargs["variables"] == _VARIABLES

    def test_no_file_written_to_disk(self, tmp_path):
        """check_availability_dry must not create anything under
        output_dir -- it's a pure in-memory existence check."""
        fake_module = _FakeCopernicusMarineModule(has_data=True)

        dl = InSituCurrentsHistoricalDownloader(instrument="drifter", output_dir=tmp_path)
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.check_availability_dry(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2024-01-01T00:00:00", "2024-01-02T00:00:00",
            )

        assert list(tmp_path.iterdir()) == []


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
