"""Tests for InSituDownloader (Copernicus Marine in-situ platforms)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd

from sar_validation.downloaders.insitu_downloader import InSituDownloader

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = 165.0, 180.0, 60.0, 68.0


class _FakeCopernicusMarineModule:
    """Minimal stand-in for the copernicusmarine module, exposing just the
    read_dataframe() call check_availability_dry uses. platform_type
    defaults to "MO" (mooring) -- callers that need to test source_types
    filtering pass a different code."""

    def __init__(self, has_data: bool, platform_type: str = "MO"):
        self._has_data = has_data
        self._platform_type = platform_type

    def read_dataframe(self, **kwargs):
        if not self._has_data:
            return pd.DataFrame()
        return pd.DataFrame({
            "variable": ["VHM0"],
            "platform_type": [self._platform_type],
            "value": [1.0],
        })


class TestNoDataOutcome:
    def test_no_file_written_at_all_is_treated_as_no_data_not_a_failure(self, tmp_path, caplog):
        """Real-world Copernicus Marine behaviour: subset() reports success (status='000',
        file_status='DOWNLOADED') and raises no exception, yet writes no
        output file at all for a genuinely empty result. This used to fall
        through to a FileNotFoundError, which orchestrator.py's
        _download_insitu logs as an ERROR-level "In-situ download failed"
        -- alarming and misleading for what is actually just "no in-situ
        platforms in this region/period". Must return None (like the
        sibling insitu_currents_historical_downloader.py already does for
        the identical copernicusmarine quirk), not raise."""
        dl = InSituDownloader(output_dir=tmp_path)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            pass  # "succeeds" but writes nothing, per the real observed behaviour

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.DEBUG):
            out = dl.download(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2026-06-01", "2026-06-05",
                source_types=["mooring", "buoy"],
            )

        assert out == []
        assert list(tmp_path.glob("*.csv")) == []
        assert any("No in-situ observations" in r.message for r in caplog.records)

    def test_real_result_still_saved_normally(self, tmp_path, monkeypatch):
        """Sanity check that the no-data handling above doesn't also
        swallow a genuine successful download. copernicusmarine.subset()
        (as called here, with no output_directory kwarg) writes into the
        process CWD -- chdir into tmp_path so that write lands somewhere
        disposable instead of the real repo root."""
        monkeypatch.chdir(tmp_path)
        dl = InSituDownloader(output_dir=tmp_path / "out")
        fake_module = MagicMock()

        expected_name = (
            "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr_"
            "WSPD-WDIR-VAVH-VGHS-VHM0-HCDT-HCSP-EWCT-NSCT_"
            "165.00E-180.00E_60.00N-68.00N_20.00-20.00m_2026-06-01-2026-06-05.csv"
        )

        def fake_subset(**kwargs):
            from pathlib import Path
            Path(expected_name).write_text("platform_type,value\nMO,1.0\n")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(
                _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT,
                "2026-06-01", "2026-06-05",
            )

        assert len(out) == 1
        assert out[0].exists()


class TestCheckAvailabilityDry:
    def test_check_availability_dry_returns_true_when_data_exists(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True)
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert result is True

    def test_check_availability_dry_returns_false_when_no_data(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=False)
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
        )

        assert result is False

    def test_check_availability_dry_filters_by_source_types(self, monkeypatch, tmp_path):
        """Data exists, but only from a mooring ("MO") platform -- a
        caller asking for "buoy" ("DB") specifically must not see it as
        available, even though the raw fetch was non-empty."""
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, platform_type="MO")
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["buoy"],
        )

        assert result is False

    def test_check_availability_dry_matching_source_type_returns_true(self, monkeypatch, tmp_path):
        fake_copernicusmarine = _FakeCopernicusMarineModule(has_data=True, platform_type="MO")
        monkeypatch.setattr(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader._get_copernicusmarine",
            lambda self: fake_copernicusmarine,
        )

        dl = InSituDownloader(output_dir=tmp_path)
        result = dl.check_availability_dry(
            min_lon=-10.0, max_lon=10.0, min_lat=35.0, max_lat=55.0,
            start="2026-08-01T00:00:00", end="2026-08-01T01:00:00",
            source_types=["mooring"],
        )

        assert result is True
