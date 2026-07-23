"""Tests for downloader utilities: datetime parsing and dataset_part selection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.downloaders.base import (
    authenticate_osi_saf_ftp,
    copernicus_marine_download_kwargs,
    is_date_recent,
    normalize_datetime,
    split_antimeridian_bbox,
)
from sar_validation.downloaders.insitu_downloader import (
    PLATFORM_CODE_TO_SOURCE_TYPE,
    SOURCE_TYPE_TO_PLATFORM,
    _resolve_platform_codes,
)

# ---------------------------------------------------------------------------
# Tests for normalize_datetime()
# ---------------------------------------------------------------------------

class TestNormalizeDatetime:
    """Test datetime normalization with various input formats."""

    def test_date_only(self):
        """Date-only input should be normalized with T00:00:00."""
        result = normalize_datetime("2026-01-01")
        assert result == "2026-01-01T00:00:00"

    def test_date_with_iso_time(self):
        """ISO format with T separator should pass through (roughly)."""
        result = normalize_datetime("2026-01-01T12:34:56")
        assert result == "2026-01-01T12:34:56"

    def test_date_with_space_separator(self):
        """Space-separated datetime should convert space to T."""
        result = normalize_datetime("2026-01-01 12:34:56")
        assert result == "2026-01-01T12:34:56"

    def test_hhmmss_format_no_colons(self):
        """6-digit time without colons should be converted to HH:MM:SS."""
        result = normalize_datetime("2026-06-24 000000")
        assert result == "2026-06-24T00:00:00"

    def test_hhmmss_format_midnight(self):
        """HHMMSS midnight should convert correctly."""
        result = normalize_datetime("2026-06-24 000000")
        assert result == "2026-06-24T00:00:00"

    def test_hhmmss_format_midday(self):
        """HHMMSS midday should convert correctly."""
        result = normalize_datetime("2026-06-24 120000")
        assert result == "2026-06-24T12:00:00"

    def test_hhmmss_format_near_end_of_day(self):
        """HHMMSS near end of day should convert correctly."""
        result = normalize_datetime("2026-06-24 235959")
        assert result == "2026-06-24T23:59:59"

    def test_hhmmss_with_three_hour_offset(self):
        """HHMMSS from the original error case should work."""
        result = normalize_datetime("2026-06-24 030000")
        assert result == "2026-06-24T03:00:00"

    def test_trailing_z_removed(self):
        """Trailing Z should be stripped."""
        result = normalize_datetime("2026-01-01T12:34:56Z")
        assert result == "2026-01-01T12:34:56"

    def test_milliseconds_removed(self):
        """Milliseconds should be removed."""
        result = normalize_datetime("2026-01-01T12:34:56.123Z")
        assert result == "2026-01-01T12:34:56"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        result = normalize_datetime("  2026-01-01  ")
        assert result == "2026-01-01T00:00:00"

    def test_space_and_hhmmss(self):
        """Space-separated HHMMSS should convert both."""
        result = normalize_datetime("2026-01-01 120000")
        assert result == "2026-01-01T12:00:00"

    def test_iso_format_already_correct(self):
        """Already-correct ISO format should pass through."""
        result = normalize_datetime("2026-01-01T12:34:56")
        assert result == "2026-01-01T12:34:56"

    def test_iso_hhmmss_with_z(self):
        """ISO format HHMMSS with Z should be handled."""
        result = normalize_datetime("2026-01-01T120000Z")
        assert result == "2026-01-01T12:00:00"


# ---------------------------------------------------------------------------
# Tests for is_date_recent()
# ---------------------------------------------------------------------------

class TestIsDateRecent:
    """Test detection of recent dates."""

    @patch("sar_validation.downloaders.base.datetime")
    def test_today_is_recent(self, mock_datetime):
        """Today should be considered recent (0 days ago)."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-07-02", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_yesterday_is_recent(self, mock_datetime):
        """Yesterday should be considered recent (1 day ago)."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-07-01", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_30_days_ago_is_recent(self, mock_datetime):
        """30 days ago should be at the boundary of recent."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-06-02", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_31_days_ago_is_not_recent(self, mock_datetime):
        """31 days ago should exceed the 30-day threshold."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-06-01", threshold_days=30)
        assert result is False

    @patch("sar_validation.downloaders.base.datetime")
    def test_old_date_is_not_recent(self, mock_datetime):
        """Date from several months ago should not be recent."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-03-15", threshold_days=30)
        assert result is False

    def test_with_hhmmss_format_parses_correctly(self):
        """HHMMSS format should parse correctly through normalize_datetime."""
        # This test verifies that normalize_datetime handles HHMMSS correctly,
        # not the date recency comparison (which requires mocking datetime.now)
        normalized = normalize_datetime("2026-07-02 120000")
        assert normalized == "2026-07-02T12:00:00"

    @patch("sar_validation.downloaders.base.datetime")
    def test_custom_threshold(self, mock_datetime):
        """Custom threshold should be respected."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        # Should NOT be recent with 30-day threshold
        result = is_date_recent("2026-05-03", threshold_days=30)
        assert result is False
        
        # But SHOULD be recent with 60-day threshold
        result = is_date_recent("2026-05-03", threshold_days=60)
        assert result is True

    def test_invalid_datetime_returns_false(self):
        """Invalid datetime should return False gracefully."""
        result = is_date_recent("not-a-date", threshold_days=30)
        assert result is False


# ---------------------------------------------------------------------------
# Integration tests for datetime parsing scenarios
# ---------------------------------------------------------------------------

class TestDatetimeIntegration:
    """Integration tests for realistic datetime parsing scenarios."""

    def test_original_error_case(self):
        """The original error case from the bug report should work."""
        # User input from CLI
        start = "2026-06-24 000000"
        end = "2026-06-24 030000"
        
        start_norm = normalize_datetime(start)
        end_norm = normalize_datetime(end)
        
        # Should produce valid ISO format
        assert start_norm == "2026-06-24T00:00:00"
        assert end_norm == "2026-06-24T03:00:00"
        
        # Should be suitable for API URL construction
        url_start = start_norm + ".000Z"
        url_end = end_norm + ".000Z"
        
        assert url_start == "2026-06-24T00:00:00.000Z"
        assert url_end == "2026-06-24T03:00:00.000Z"

    def test_mixed_datetime_formats_in_recipe(self):
        """A recipe might have different datetime formats for start/end."""
        # Date-only format
        start = normalize_datetime("2026-01-01")
        # HHMMSS format
        end = normalize_datetime("2026-01-02 235959")
        
        assert start == "2026-01-01T00:00:00"
        assert end == "2026-01-02T23:59:59"

    def test_api_url_construction_flow(self):
        """Simulate the full flow from user input to API URL."""
        # User provides input with HHMMSS format
        user_start = "2026-06-24 000000"
        user_end = "2026-06-24 030000"

        # SAR downloader normalizes and appends .000Z
        norm_start = normalize_datetime(user_start)
        norm_end = normalize_datetime(user_end)

        api_start = norm_start + ".000Z"
        api_end = norm_end + ".000Z"

        # These should be valid for Copernicus API
        assert api_start == "2026-06-24T00:00:00.000Z"
        assert api_end == "2026-06-24T03:00:00.000Z"

        # Verify they parse as valid ISO datetime
        datetime.fromisoformat(api_start.rstrip("Z"))
        datetime.fromisoformat(api_end.rstrip("Z"))


# ---------------------------------------------------------------------------
# Tests for split_antimeridian_bbox()
# ---------------------------------------------------------------------------

class TestSplitAntimeridianBbox:
    def test_non_crossing_bbox_returned_unchanged(self):
        assert split_antimeridian_bbox(-20.0, 0.0) == [(-20.0, 0.0)]

    def test_equal_bounds_treated_as_non_crossing(self):
        assert split_antimeridian_bbox(10.0, 10.0) == [(10.0, 10.0)]

    def test_crossing_bbox_splits_into_two_windows(self):
        assert split_antimeridian_bbox(135.0, -120.0) == [(135.0, 180.0), (-180.0, -120.0)]

    def test_crossing_bbox_windows_are_each_non_crossing(self):
        windows = split_antimeridian_bbox(170.0, -170.0)
        for lo, hi in windows:
            assert lo <= hi


# ---------------------------------------------------------------------------
# Tests for copernicus_marine_download_kwargs()
# ---------------------------------------------------------------------------

class TestCopernicusMarineDownloadKwargs:
    def test_default_skips_existing_files(self):
        assert copernicus_marine_download_kwargs(force_download=False) == {
            "skip_existing": True, "overwrite": False,
        }

    def test_force_download_overwrites(self):
        assert copernicus_marine_download_kwargs(force_download=True) == {
            "skip_existing": False, "overwrite": True,
        }


# ---------------------------------------------------------------------------
# Tests for authenticate_osi_saf_ftp()
# ---------------------------------------------------------------------------

class TestAuthenticateOsiSafFtp:
    def test_explicit_args_win_over_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSI_SAF_FTP_USERNAME", "env_user")
        monkeypatch.setenv("OSI_SAF_FTP_PASSWORD", "env_pass")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        username, password = authenticate_osi_saf_ftp("explicit_user", "explicit_pass")
        assert (username, password) == ("explicit_user", "explicit_pass")

    def test_env_vars_used_when_args_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSI_SAF_FTP_USERNAME", "env_user")
        monkeypatch.setenv("OSI_SAF_FTP_PASSWORD", "env_pass")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        username, password = authenticate_osi_saf_ftp()
        assert (username, password) == ("env_user", "env_pass")

    def test_falls_back_to_credentials_file(self, monkeypatch, tmp_path):
        import json

        monkeypatch.delenv("OSI_SAF_FTP_USERNAME", raising=False)
        monkeypatch.delenv("OSI_SAF_FTP_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cred_file = tmp_path / ".eumetsat_osi_saf_wind_credentials"
        cred_file.write_text(json.dumps({"username": "file_user", "password": "file_pass"}))

        username, password = authenticate_osi_saf_ftp()
        assert (username, password) == ("file_user", "file_pass")

    def test_raises_runtime_error_when_nothing_resolves(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OSI_SAF_FTP_USERNAME", raising=False)
        monkeypatch.delenv("OSI_SAF_FTP_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(RuntimeError, match="OSI-SAF FTP credentials not found"):
            authenticate_osi_saf_ftp()


# ---------------------------------------------------------------------------
# SARDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestSARDownloaderAntimeridian:
    def _record(self, id_):
        return {
            "Id": id_, "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_query_splits_crossing_bbox_into_two_windows(self, tmp_path):
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_products.side_effect = [
            [self._record("a")], [self._record("b")],
        ]
        dl._client = fake_client

        df = dl.query(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )

        assert fake_client.query_products.call_count == 2
        first_kwargs = fake_client.query_products.call_args_list[0].kwargs
        second_kwargs = fake_client.query_products.call_args_list[1].kwargs
        assert (first_kwargs["min_lon"], first_kwargs["max_lon"]) == (135.0, 180.0)
        assert (second_kwargs["min_lon"], second_kwargs["max_lon"]) == (-180.0, -120.0)
        assert sorted(df["Id"]) == ["a", "b"]

    def test_query_dedupes_product_returned_by_both_windows(self, tmp_path):
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        dup = self._record("dup")
        fake_client.query_products.side_effect = [[dup], [dup]]
        dl._client = fake_client

        df = dl.query(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert len(df) == 1

    def test_query_non_crossing_bbox_calls_once(self, tmp_path):
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_products.return_value = []
        dl._client = fake_client

        dl.query(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-01-01", end="2026-01-02",
        )
        assert fake_client.query_products.call_count == 1
        kwargs = fake_client.query_products.call_args.kwargs
        assert (kwargs["min_lon"], kwargs["max_lon"]) == (-20.0, 0.0)


# ---------------------------------------------------------------------------
# SARDownloader — per-product existence check
# ---------------------------------------------------------------------------

class TestSARDownloaderForceDownload:
    def _fake_record(self):
        return {
            "Id": "abc", "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_skips_product_whose_directory_already_exists(self, tmp_path, capsys):
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=False)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._fake_record()]
        dl._client = fake_client
        (tmp_path / "S1A_IW_OCN__2SDV_20260702T000000").mkdir()

        dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        fake_client.download_product.assert_not_called()
        assert "Already downloaded" in capsys.readouterr().out

    def test_force_download_redownloads_existing_product(self, tmp_path):
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._fake_record()]
        dl._client = fake_client
        fake_client.download_product.return_value = (
            tmp_path / "S1A_IW_OCN__2SDV_20260702T000000.SAFE"
        )
        (tmp_path / "S1A_IW_OCN__2SDV_20260702T000000").mkdir()

        dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        fake_client.download_product.assert_called_once()


# ---------------------------------------------------------------------------
# AltimeterDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestAltimeterDownloaderAntimeridian:
    def _patch_subset(self):
        from pathlib import Path

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_crossing_bbox_splits_into_two_windows_with_distinct_filenames(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
                frequencies=["1hz"], satellites=["al"],
            )

        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert (first_kwargs["minimum_longitude"], first_kwargs["maximum_longitude"]) == (135.0, 180.0)
        assert (second_kwargs["minimum_longitude"], second_kwargs["maximum_longitude"]) == (-180.0, -120.0)
        assert first_kwargs["output_filename"] != second_kwargs["output_filename"]
        assert len(paths) == 2

    def test_non_crossing_bbox_keeps_single_call_and_original_filename(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        assert fake_module.subset.call_count == 1
        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["output_filename"] == "cmems_obs-wave_glo_phy-swh_nrt_al-l3_PT1S_2026-06-01_2026-06-02.nc"
        assert len(paths) == 1


class TestAltimeterDownloaderForceDownload:
    def _patch_subset(self):
        from pathlib import Path

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_default_skips_existing(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is True
        assert kwargs["overwrite"] is False

    def test_force_download_overwrites(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is False
        assert kwargs["overwrite"] is True

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        """Regression: copernicusmarine.subset() has no force_download
        parameter in the installed version (verified via
        inspect.signature) — passing it raises TypeError in real
        (non-mocked) usage."""
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        assert "force_download" not in fake_module.subset.call_args.kwargs


# ---------------------------------------------------------------------------
# InSituDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestInSituDownloaderAntimeridian:
    def test_download_splits_crossing_bbox_into_two_windows(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        # force_download=True: Task 8's dest_path.exists() pre-check would
        # otherwise skip subset() entirely for the pre-created files below
        # (they exist only to satisfy _download_window's post-call "already
        # at dest_path" branch, since the fake subset() doesn't write real
        # files). This test is about window splitting, not the pre-check.
        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None  # real subset writes to CWD; not needed here

        start_dt, end_dt = "2026-07-02T00:00:00", "2026-07-03T00:00:00"
        for lo, hi in [(135.0, 180.0), (-180.0, -120.0)]:
            fname = _build_csv_filename(lo, hi, -15.0, 30.0, start_dt, end_dt, -20.0, 20.0)
            # Pre-create the destination file so _download_window's
            # "already at dest_path" branch is taken instead of the
            # CWD-relative move (which the fake subset() doesn't produce).
            (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert (first_kwargs["minimum_longitude"], first_kwargs["maximum_longitude"]) == (135.0, 180.0)
        assert (second_kwargs["minimum_longitude"], second_kwargs["maximum_longitude"]) == (-180.0, -120.0)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert paths[0].name != paths[1].name

    def test_non_crossing_bbox_calls_once_and_returns_single_path(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        # force_download=True: see comment in the sibling test above — the
        # pre-created file is a mock-download workaround, not the subject
        # under test here.
        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        assert fake_module.subset.call_count == 1
        assert len(paths) == 1
        assert paths[0].name == fname


class TestInSituDownloaderForceDownload:
    def test_skips_download_when_file_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        fake_module.subset.assert_not_called()
        assert len(paths) == 1
        assert paths[0].name == fname

    def test_force_download_redownloads_existing_file(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        fake_module.subset.assert_called_once()

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        """Regression: copernicusmarine.subset() has no force_download
        parameter in the installed version."""
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        # No dest file pre-created: default force_download=False must still
        # call subset() (the pre-check only short-circuits when a file
        # already exists). The fake subset() writes straight to dest_path
        # since it doesn't produce a real CWD-relative output file.
        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        start_dt, end_dt = "2026-01-05T00:00:00", "2026-01-06T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        dest_path = tmp_path / fname

        def fake_subset(**kwargs):
            dest_path.write_text("platform_type\n")

        fake_module.subset.side_effect = fake_subset

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-05", end="2026-01-06",
            )

        assert "force_download" not in fake_module.subset.call_args.kwargs


# ---------------------------------------------------------------------------
# ScatterometerDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestScatterometerDownloaderAntimeridian:
    def test_dry_run_prints_both_windows(self, tmp_path, capsys):
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert out == []
        captured = capsys.readouterr().out.replace(" ", "")
        assert "[135.0,180.0]" in captured
        assert "[-180.0,-120.0]" in captured

    def test_search_runs_once_per_window_and_dedupes_products(self, tmp_path, capsys):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        # "dup" is returned by both window searches and must be counted once.
        # None of these IDs contain "metopb"/"metopc", so the per-product
        # download loop skips them immediately — this test only exercises
        # the search+dedup logic, not the download loop.
        fake_collection.search.side_effect = [["dup", "east_only"], ["dup", "west_only"]]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            result = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert result == []
        assert fake_collection.search.call_count == 2
        first_kwargs = fake_collection.search.call_args_list[0].kwargs
        second_kwargs = fake_collection.search.call_args_list[1].kwargs
        assert first_kwargs["bbox"] == "135.0,-15.0,180.0,30.0"
        assert second_kwargs["bbox"] == "-180.0,-15.0,-120.0,30.0"
        assert "Found 3 ASCAT products." in capsys.readouterr().out

    def test_non_crossing_bbox_searches_once(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = []
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        assert fake_collection.search.call_count == 1
        assert fake_collection.search.call_args.kwargs["bbox"] == "-20.0,35.0,0.0,60.0"


# ---------------------------------------------------------------------------
# Scatterometer downloader — per-product existence check
# ---------------------------------------------------------------------------

class TestScatterometerDownloaderForceDownload:
    def test_skips_product_whose_output_file_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"
        (tmp_path / "OASWC12_20260705_183300_71590_metopb.nc").write_bytes(b"")

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = ["71590_metopb"]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_not_called()

    def test_force_download_redownloads_existing_product(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        dl._token = "fake-token"
        (tmp_path / "OASWC12_20260705_183300_71590_metopb.nc").write_bytes(b"")

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = ["71590_metopb"]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        fake_file = MagicMock()
        fake_file.name = "OASWC12_20260705_183300_71590_metopb.nc"
        fake_file.read.side_effect = [b"data", b""]
        fake_product = MagicMock()
        fake_product.open.return_value.__enter__.return_value = fake_file
        fake_product.open.return_value.__exit__.return_value = False
        fake_datastore.get_product.return_value = fake_product

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for in-situ source-type <-> Copernicus platform-code mapping
# ---------------------------------------------------------------------------

class TestInsituPlatformCodeMapping:
    def test_drifter_covers_both_db_and_ad(self):
        assert SOURCE_TYPE_TO_PLATFORM["drifter"] == ["DB", "AD"]

    def test_buoy_is_db_only(self):
        assert SOURCE_TYPE_TO_PLATFORM["buoy"] == ["DB"]

    def test_resolve_platform_codes_dedupes_shared_db(self):
        codes = _resolve_platform_codes(["buoy", "drifter"])
        assert codes == ["DB", "AD"]

    def test_resolve_platform_codes_unknown_source_type_raises(self):
        with pytest.raises(ValueError):
            _resolve_platform_codes(["not_a_real_type"])

    def test_db_labels_as_buoy(self):
        assert PLATFORM_CODE_TO_SOURCE_TYPE["DB"] == "buoy"

    def test_ad_labels_as_drifter(self):
        assert PLATFORM_CODE_TO_SOURCE_TYPE["AD"] == "drifter"

    def test_mo_labels_as_mooring(self):
        assert PLATFORM_CODE_TO_SOURCE_TYPE["MO"] == "mooring"


# ---------------------------------------------------------------------------
# RadiometerDownloader (RSS radiometer over HTTPS)
# ---------------------------------------------------------------------------

from sar_validation.downloaders.radiometer_downloader import (
    SENSORS,
    SUPPORTED_SENSORS,
    RadiometerDownloader,
)


class TestRadiometerDownloader:
    def test_amsr2_is_a_supported_netcdf_sensor(self):
        assert "amsr2" in SUPPORTED_SENSORS
        assert SENSORS["amsr2"]["format"] == "netcdf"

    def test_bytemap_sensors_supported(self):
        # GMI/SSMIS/WindSat are RSS binary bytemaps and now downloadable.
        for s in ("gmi", "ssmis_f16", "ssmis_f17", "ssmis_f18", "windsat"):
            assert s in SENSORS
            assert SENSORS[s]["format"] == "bytemap"
            assert s in SUPPORTED_SENSORS
            assert SENSORS[s]["url_path"]        # has a configured download URL

    def test_supported_sensor_set(self):
        assert set(SUPPORTED_SENSORS) == {
            "amsr2", "gmi", "ssmis_f16", "ssmis_f17", "ssmis_f18", "windsat"
        }

    def test_only_windsat_has_direction(self):
        assert SENSORS["windsat"]["has_direction"] is True
        for s in ("amsr2", "gmi", "ssmis_f16"):
            assert SENSORS[s]["has_direction"] is False

    def test_dry_run_lists_urls_without_network(self, tmp_path, capsys):
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        paths = dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                            start="2024-06-01", end="2024-06-02")
        out = capsys.readouterr().out
        assert paths == []
        assert "DRY RUN" in out
        # Both days for the default (amsr2) sensor, with the correct URL shape.
        assert "RSS_AMSR2_ocean_L3_daily_2024-06-01_v08.2.nc" in out
        assert "RSS_AMSR2_ocean_L3_daily_2024-06-02_v08.2.nc" in out
        assert "data.remss.com/amsr2/ocean/L3" in out

    def test_dry_run_lists_bytemap_urls(self, tmp_path, capsys):
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                    start="2024-06-01", end="2024-06-01")
        out = capsys.readouterr().out
        # Monthly-subfolder .gz URLs with a YYYYMMDD stamp, per sensor.
        assert "gmi/bmaps_v08.2/y2024/m06/f35_20240601v8.2.gz" in out
        assert "ssmi/f16/bmaps_v07/y2024/m06/f16_20240601v7.gz" in out
        assert "windsat/bmaps_v07.0.1/y2024/m06/wsat_20240601v7.0.1.gz" in out

    def test_availability_window_skips_early_dates(self, tmp_path, capsys):
        # AMSR2 data starts 2012 — a 2010 request should be skipped, no download.
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                    start="2010-01-01", end="2010-01-02", sensors=["amsr2"])
        out = capsys.readouterr().out
        assert "Skipping amsr2" in out
        assert "availability" in out

    def test_unknown_sensor_warns(self, tmp_path, capsys):
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                    start="2024-06-01", end="2024-06-01", sensors=["not_a_sensor"])
        out = capsys.readouterr().out
        assert "unknown radiometer sensor" in out.lower()


# ---------------------------------------------------------------------------
# RSS binary bytemap reader (_rss_bytemap.read_rss_bytemap)
# ---------------------------------------------------------------------------

import gzip

import numpy as np

from sar_validation.downloaders._rss_bytemap import (
    BYTEMAP_LAYOUT,
    NLAT,
    NLON,
    NPASS,
    read_rss_bytemap,
)


def _write_bytemap(tmp_path, sensor, filename, cells):
    """Write a full-size RSS bytemap .gz (all-missing 255 except `cells`).

    cells: list of (pass, var_idx, lat_idx, lon_idx, byte_value).
    """
    nvar = len(BYTEMAP_LAYOUT[sensor]["vars"])
    arr = np.full((NPASS, nvar, NLAT, NLON), 255, np.uint8)
    for (p, v, la, lo, val) in cells:
        arr[p, v, la, lo] = val
    path = tmp_path / filename
    with gzip.open(path, "wb") as fh:
        fh.write(arr.tobytes())
    return path


class TestReadRssBytemap:
    def test_gmi_scale_offset_and_grid(self, tmp_path):
        # GMI var indices: 0=time (×0.1), 2=windLF (×0.2).
        p = _write_bytemap(tmp_path, "gmi", "f35_20240601v8.2.gz",
                           [(0, 2, 400, 600, 50), (0, 0, 400, 600, 100)])
        decoded, lon, lat = read_rss_bytemap(p, "gmi")
        assert decoded["windLF"].shape == (NPASS, NLAT, NLON)
        assert decoded["windLF"][0, 400, 600] == pytest.approx(10.0)   # 50×0.2
        assert decoded["time"][0, 400, 600] == pytest.approx(10.0)     # 100×0.1
        assert np.isnan(decoded["windLF"][1, 0, 0])                    # 255 → NaN
        assert lon[600] == pytest.approx(600 * 0.25 + 0.125)
        assert lat[400] == pytest.approx(400 * 0.25 - 89.875)

    def test_missing_code_threshold(self, tmp_path):
        # Byte 250 is valid; 251 is the first special/missing code.
        p = _write_bytemap(tmp_path, "gmi", "f35_20240101v8.2.gz",
                           [(0, 2, 0, 0, 250), (0, 2, 0, 1, 251)])
        decoded, _, _ = read_rss_bytemap(p, "gmi")
        assert decoded["windLF"][0, 0, 0] == pytest.approx(50.0)       # 250×0.2
        assert np.isnan(decoded["windLF"][0, 0, 1])                    # 251 masked

    def test_windsat_has_nine_vars_incl_wdir(self, tmp_path):
        p = _write_bytemap(tmp_path, "windsat", "wsat_20150601v7.0.1.gz",
                           [(0, 8, 300, 500, 40)])
        decoded, _, _ = read_rss_bytemap(p, "windsat")
        assert "wdir" in decoded and "w-lf" in decoded
        assert decoded["wdir"][0, 300, 500] == pytest.approx(60.0)     # 40×1.5

    def test_size_mismatch_raises(self, tmp_path):
        path = tmp_path / "f35_bad.gz"
        with gzip.open(path, "wb") as fh:
            fh.write(b"\x00" * 100)
        with pytest.raises(ValueError):
            read_rss_bytemap(path, "gmi")

    def test_unknown_sensor_raises(self, tmp_path):
        with pytest.raises(KeyError):
            read_rss_bytemap(tmp_path / "x.gz", "not_a_sensor")


from sar_validation.downloaders.noaa_hfradar_downloader import (
    ERDDAP_BASE,
    build_erddap_subset_url,
    clamp_to_region_bbox,
    select_backend,
    select_erddap_dataset,
)


class TestSelectErddapDataset:
    def test_us_west_6km_default(self):
        assert select_erddap_dataset(-125, -119, 33, 38, 6) == "ucsdHfrW6"

    def test_us_west_2km(self):
        assert select_erddap_dataset(-125, -119, 33, 38, 2) == "ucsdHfrW2"

    def test_us_east_gulf_6km(self):
        assert select_erddap_dataset(-80, -70, 35, 42, 6) == "ucsdHfrE6"

    def test_unsupported_region_raises(self):
        with pytest.raises(ValueError, match="No ERDDAP HF-radar dataset"):
            select_erddap_dataset(2.0, 8.0, 53.0, 55.0, 6)  # German Bight → Phase 3c

    def test_unsupported_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            select_erddap_dataset(-80, -70, 35, 42, 2)  # US-East has no 2 km


class TestClampToRegionBbox:
    def test_bbox_fully_inside_region_is_unchanged(self):
        assert clamp_to_region_bbox(-80, -70, 35, 42) == (-80.0, -70.0, 35.0, 42.0)

    def test_bbox_extending_south_of_region_is_clamped(self):
        # Reproduces the reported bug: a recipe bbox reaching down to 20.0N
        # (to also cover Puerto Rico) extends past US-East/Gulf's actual
        # southern grid edge (22.0N in _REGIONS), which ERDDAP rejects with
        # HTTP 404 rather than clipping server-side.
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(-80, -60, 20.0, 40.0)
        assert min_lat == 22.0
        assert (min_lon, max_lon, max_lat) == (-80.0, -60.0, 40.0)

    def test_clamped_bbox_stays_within_erddap_axis_bounds(self):
        # 22.0N (the _REGIONS config bound) must be >= the real ERDDAP grid's
        # minimum latitude (21.73596N per the dataset's .das), so clamping to
        # it never re-triggers the same out-of-bounds 404.
        _, _, min_lat, _ = clamp_to_region_bbox(-80, -60, 20.0, 40.0)
        assert min_lat >= 21.73596

    def test_west_coast_bbox_at_old_config_edge_is_clamped(self):
        # Reproduces the reported bug: recipes/currents_uswestcoast2.yaml's
        # bbox reaches min_lat=30.0, which used to equal the (too loose)
        # _REGIONS US_WEST config bound, so nothing got clamped and the
        # unclamped 30.0 reached ERDDAP below its real axis minimum
        # (30.25N per ucsdHfrW6's .das), triggering the same HTTP 404.
        # The recipe's max_lon=-115.0 is also past the real axis maximum
        # (-115.8056 per the .das), so it gets clamped too.
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(
            -126.0, -115.0, 30.0, 48.0
        )
        assert min_lat == 30.25
        assert (min_lon, max_lon, max_lat) == (-126.0, -115.8056, 48.0)
        assert min_lat >= 30.25  # real ERDDAP grid's minimum latitude
        assert max_lon <= -115.8056  # real ERDDAP grid's maximum longitude


class TestBuildErddapSubsetUrl:
    def test_url_has_vars_bbox_and_time_selectors(self):
        url = build_erddap_subset_url(
            "ucsdHfrW6", -125, -119, 33, 38, "2024-05-01", "2024-05-01T06:00:00"
        )
        assert url.startswith(f"{ERDDAP_BASE}/ucsdHfrW6.nc?")
        assert "water_u[(2024-05-01T00:00:00Z):(2024-05-01T06:00:00Z)]" in url
        assert "water_v[(2024-05-01T00:00:00Z):(2024-05-01T06:00:00Z)]" in url
        assert "[(33.0):(38.0)]" in url   # latitude ascending
        assert "[(-125.0):(-119.0)]" in url  # longitude ascending


class TestSelectBackend:
    def test_recent_date_uses_erddap(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        assert select_backend(recent) == "erddap"

    def test_old_date_not_yet_supported(self):
        with pytest.raises(NotImplementedError, match="Phase 3b"):
            select_backend("2015-01-01")


from unittest.mock import patch

from sar_validation.downloaders.noaa_hfradar_downloader import NOAAHFRadarDownloader

# NOTE: the task brief's verbatim test used a hardcoded "2024-05-01" date for
# `end`. select_backend() rejects any `end` older than the rolling ~90-day
# ERDDAP window relative to wall-clock "now", so a hardcoded past date goes
# stale and starts raising NotImplementedError once the suite is run more
# than ~90 days after the brief was written. Using a date a few days before
# "now" keeps these tests deterministically inside the window (mirroring the
# existing TestSelectBackend.test_recent_date_uses_erddar pattern above)
# without changing any of the brief's assertions.
_RECENT_START = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
_RECENT_END = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT06:00:00")


class TestNOAAHFRadarDownload:
    def test_dry_run_returns_empty_list_and_no_fetch(self, tmp_path, capsys):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert out == []
        m.assert_not_called()
        assert "ucsdHfrW6.nc?" in capsys.readouterr().out

    def test_download_fetches_url_to_expected_path(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert len(out) == 1
        assert out[0].parent == tmp_path
        assert out[0].suffix == ".nc"
        m.assert_called_once()
        called_url, called_path = m.call_args[0][0], m.call_args[0][1]
        assert "ucsdHfrW6.nc?" in called_url
        assert str(out[0]) == str(called_path)

    def test_download_clamps_bbox_extending_past_region_edge(self, tmp_path):
        """A bbox reaching past a region's real grid edge must be clamped in
        the built URL, not passed straight through (root cause of the
        reported HTTP 404 'axis minimum' error)."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            dl.download(-80, -60, 20.0, 40.0, _RECENT_START, _RECENT_END)
        called_url = m.call_args[0][0]
        assert "[(22.0):(40.0)]" in called_url
        assert "[(20.0)" not in called_url

    def test_download_clamps_west_coast_recipe_bbox(self, tmp_path):
        """recipes/currents_uswestcoast2.yaml's exact bbox (min_lat=30.0)
        must be clamped to the real ERDDAP axis minimum (30.25N), not passed
        straight through and 404 at the server."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            dl.download(-126.0, -115.0, 30.0, 48.0, _RECENT_START, _RECENT_END)
        called_url = m.call_args[0][0]
        assert "[(30.25):(48.0)]" in called_url
        assert "[(30.0)" not in called_url
        assert "[(-126.0):(-115.8056)]" in called_url


class TestNOAAHFRadarDownloaderAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # 135E..120W doesn't overlap US_WEST or US_EAST_GULF on either side
        # of the split (NOAA's _match_region uses each window's *center*
        # point, and neither window's center falls inside either region).
        # Note: the unsplit pre-fix code also raises a ValueError matching
        # this message for a min_lon > max_lon input (its own center-point
        # math just lands on a different, still-uncovered point), so this
        # test alone doesn't distinguish pre-fix from post-fix — it guards
        # that the "truly nothing covers this" case keeps failing loudly
        # after the fix too. The next test is the one that actually fails
        # pre-fix.
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            with pytest.raises(ValueError, match="No ERDDAP HF-radar dataset"):
                dl.download(135.0, -120.0, -15.0, 30.0, _RECENT_START, _RECENT_END)
        m.assert_not_called()

    def test_crossing_bbox_downloads_the_side_whose_window_center_resolves(self, tmp_path):
        # NOAA's region match is center-point-based (not overlap-area, unlike
        # the Copernicus HFR regions), so only a window whose *own* center
        # (after splitting) lands inside a supported region resolves. Here
        # min_lon=179, max_lon=-66 splits into [179, 180] (center 179.5,
        # 36.5 — matches nothing) and [-180, -66] (center -123.0, 36.5 —
        # inside US_WEST's bbox). The raw (unsplit) request's own center,
        # (56.5, 36.5), matches nothing — that's what makes this fail today.
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(179.0, -66.0, 35.0, 38.0, _RECENT_START, _RECENT_END)
        assert len(out) == 1
        m.assert_called_once()
        called_url = m.call_args[0][0]
        assert "ucsdHfrW6.nc?" in called_url


class TestNOAAHFRadarDownloaderForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader,
            select_erddap_dataset,
        )

        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        dataset_id = select_erddap_dataset(-125, -119, 33, 38, 6)
        out_path = tmp_path / f"{dataset_id}_6km_{_RECENT_START}.nc"
        out_path.write_bytes(b"")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        m.assert_not_called()
        assert out == [out_path]

    def test_force_download_refetches_existing_output(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader,
            select_erddap_dataset,
        )

        dl = NOAAHFRadarDownloader(
            output_dir=tmp_path, dry_run=False, resolution_km=6, force_download=True,
        )
        dataset_id = select_erddap_dataset(-125, -119, 33, 38, 6)
        out_path = tmp_path / f"{dataset_id}_6km_{_RECENT_START}.nc"
        out_path.write_bytes(b"stale")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        m.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for DataOrchestrator "hf_radar_noaa" wiring (Task 7)
# ---------------------------------------------------------------------------
# DataOrchestrator can be built cheaply from a stub Recipe (no network, no
# real base-dir creation under dry_run), so a behavioural test is preferred
# over source-inspection alone.

class TestOrchestratorHFRadarNOAAWiring:
    def test_dispatch_source_registers_hf_radar_noaa_handler(self):
        import inspect

        from sar_validation.core.orchestrator import DataOrchestrator

        src = inspect.getsource(DataOrchestrator._dispatch_source)
        assert '"hf_radar_noaa"' in src
        assert "_download_noaa_hfradar" in src
        assert hasattr(DataOrchestrator, "_download_noaa_hfradar")

    def test_download_noaa_hfradar_dry_run_sets_metadata_and_makes_no_network_call(
        self, tmp_path
    ):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-noaa",
            variable="currents",
            output_dir=str(tmp_path),
            # US_WEST bbox: the only region select_erddap_dataset() accepts
            # for the default 6 km resolution.
            geographic_bounds=GeographicBounds(-125.0, -119.0, 33.0, 38.0),
            temporal_bounds=TemporalBounds(_RECENT_START, _RECENT_END),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar_noaa")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            result = orchestrator._download_noaa_hfradar(source)

        assert result is True
        assert orchestrator.metadata["downloads"]["hf_radar_noaa"]["status"] == "dry_run"
        m.assert_not_called()

    def test_download_noaa_hfradar_honours_resolution_km_override(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-noaa-res",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(
            source_type="hf_radar_noaa",
            download_kwargs={"resolution_km": 1},
        )

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.NOAAHFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_noaa_hfradar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["resolution_km"] == 1


# ---------------------------------------------------------------------------
# Tests for DataOrchestrator depth resolution (optional min_depth/max_depth)
# ---------------------------------------------------------------------------

class TestOrchestratorDepthResolution:
    def test_hf_radar_dispatch_uses_default_depth_when_unspecified(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-depth-default",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar")

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_hf_radar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0

    def test_hf_radar_dispatch_honours_explicit_depth(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-depth-explicit",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_hf_radar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -2.0
        assert kwargs["max_depth"] == 2.0

    def test_insitu_batch_uses_default_depth_when_all_unspecified(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-insitu-depth-default",
            variable="wind",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="buoy"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls:
            mock_cls.return_value.download.return_value = None
            mock_sar_cls.return_value.download.return_value = []
            orchestrator.download_all()

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0

    def test_insitu_batch_widens_window_around_explicit_override(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-insitu-depth-mixed",
            variable="wind",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="mooring", min_depth=-5.0, max_depth=5.0),
                ValidationDataSource(source_type="buoy"),  # unspecified -> -20/20
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls:
            mock_cls.return_value.download.return_value = None
            mock_sar_cls.return_value.download.return_value = []
            orchestrator.download_all()

        _, kwargs = mock_cls.call_args
        # most permissive window across resolved depths: min(-5,-20)=-20, max(5,20)=20
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0


# ---------------------------------------------------------------------------
# Tests for _hf_radar_regions (shared Copernicus HF-radar region lookup)
# ---------------------------------------------------------------------------

from sar_validation.downloaders._hf_radar_regions import HFR_REGIONS, resolve_hfr_region


class TestHfRadarRegions:
    def test_us_east_gulf_bbox_resolves(self):
        assert resolve_hfr_region(-90.0, -60.0, 30.0, 40.0) == "US-EastGulfCoast"

    def test_us_west_coast_bbox_resolves(self):
        assert resolve_hfr_region(-125.0, -119.0, 33.0, 38.0) == "US-WestCoast"

    def test_no_overlap_raises_with_region_list(self):
        with pytest.raises(ValueError, match="US-EastGulfCoast"):
            resolve_hfr_region(100.0, 105.0, -10.0, -5.0)  # nowhere near any region

    def test_picks_largest_overlap_when_bbox_spans_two_regions(self):
        # DeltaEbro and ICATMAR genuinely overlap in the western
        # Mediterranean. A query bbox weighted toward each region's side of
        # the overlap should resolve to that region, exercising the
        # largest-overlap-area tie-break rather than "first match wins".
        assert resolve_hfr_region(0.0, 1.5, 39.6, 41.2) == "DeltaEbro"
        assert resolve_hfr_region(0.5, 4.0, 40.6, 42.9) == "ICATMAR"

    def test_all_regions_have_bbox_and_flag(self):
        assert len(HFR_REGIONS) == 25
        for name, cfg in HFR_REGIONS.items():
            assert len(cfg["bbox"]) == 4
            assert isinstance(cfg["has_latest"], bool)

    def test_regions_without_latest_feed(self):
        no_latest = {n for n, c in HFR_REGIONS.items() if not c["has_latest"]}
        assert no_latest == {
            "ARPAS", "COSYNA", "Finnmark", "US-Alaska",
            "US-EastGulfCoast", "US-Hawaii", "WHub",
        }


# ---------------------------------------------------------------------------
# Tests for HFRadarDownloader querying the gridded radar-total dataset_parts
# ---------------------------------------------------------------------------

class TestHFRadarDownloaderGrid:
    def test_dry_run_prints_resolved_region_and_part(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-06-05", "2026-06-06")
        assert out == []
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "radar-total--US-EastGulfCoast" in captured

    def test_download_calls_subset_with_resolved_region_part(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            # Simulate copernicusmarine writing the requested file.
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert len(out) == 1
        assert out[0].exists()
        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "monthly-radar-total--US-EastGulfCoast"
        assert kwargs["minimum_longitude"] == -90.0
        assert kwargs["maximum_longitude"] == -60.0

    def test_recent_date_uses_latest_part_when_region_has_one(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        recent_end = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        # US-WestCoast has a `latest` feed (unlike US-EastGulfCoast).
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-125.0, -119.0, 33.0, 38.0, recent_start, recent_end)

        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"

    def test_constructor_accepts_unused_depth_kwargs(self, tmp_path):
        # The orchestrator always passes min_depth/max_depth; the gridded
        # product has no depth axis, but the kwargs must still be accepted.
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        HFRadarDownloader(output_dir=tmp_path, dry_run=True, min_depth=-2.0, max_depth=2.0)

    def test_retries_with_monthly_part_when_latest_out_of_bounds(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        recent_end = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        call_count = {"n": 0}

        def fake_subset(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError(
                    "The requested time range appears to exceed the dataset "
                    "coordinates for this dataset_part."
                )
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        # US-WestCoast has a `latest` feed, so the first attempt uses it and
        # is expected to fail, triggering a retry with the `monthly` part.
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-125.0, -119.0, 33.0, 38.0, recent_start, recent_end)

        assert len(out) == 1
        assert out[0].exists()
        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert first_kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"
        assert second_kwargs["dataset_part"] == "monthly-radar-total--US-WestCoast"

    def test_raises_file_not_found_when_subset_writes_no_file(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()
        # subset() "succeeds" (no exception) but never writes the destination
        # file, simulating an empty/no-op response from copernicusmarine.
        fake_module.subset.side_effect = lambda **kwargs: None

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with pytest.raises(FileNotFoundError):
                dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")


class TestHFRadarDownloaderGridAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # lat 0-5 doesn't overlap any HFR_REGIONS entry on either side of
        # the split — the southernmost real region (US-Hawaii) starts at
        # 14.5N, so no window can resolve a region. Note: the *unsplit*
        # pre-fix code also raises a ValueError matching this same message
        # for a min_lon > max_lon input (its overlap-area formula degrades
        # to a spurious negative number for every region), so this test
        # alone doesn't distinguish pre-fix from post-fix — it guards that
        # the "truly nothing covers this" case keeps failing loudly after
        # the fix too. The next test is the one that actually fails pre-fix.
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError, match="No Copernicus HF-radar region overlaps"):
            dl.download(135.0, -120.0, 0.0, 5.0, "2026-07-02", "2026-07-03")

    def test_crossing_bbox_downloads_the_side_that_resolves_to_a_region(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        # US-Alaska's bbox (-174.10..-128.66) overlaps the [-180, -120]
        # window but not the [135, 180] window, so only one window should
        # produce a download.
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(135.0, -120.0, 65.0, 75.0, "2026-01-01", "2026-01-02")

        assert len(out) == 1
        assert fake_module.subset.call_count == 1
        _, kwargs = fake_module.subset.call_args
        assert kwargs["minimum_longitude"] == -180.0
        assert kwargs["maximum_longitude"] == -120.0


class TestHFRadarDownloaderGridForceDownload:
    def _patch_subset(self):
        from pathlib import Path

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_default_skips_existing(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is True
        assert kwargs["overwrite"] is False

    def test_force_download_overwrites(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["skip_existing"] is False
        assert kwargs["overwrite"] is True

    def test_force_download_kwarg_never_passed_to_subset(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert "force_download" not in fake_module.subset.call_args.kwargs


# ---------------------------------------------------------------------------
# Tests for HFRadarHistoricalDownloader
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class FileGetResult:
    files: List[Any] = field(default_factory=list)


class TestHFRadarHistoricalDownloader:
    def test_dry_run_prints_resolved_region_and_filename(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2021-06-05", "2021-06-06")
        assert out == []
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc" in captured

    def test_unavailable_region_returns_empty(self, tmp_path, caplog):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        # GoS (Italy) has an NRT feed but no delayed-mode archive. This must
        # not raise — the orchestrator's NRT-fallback logic (see
        # orchestrator.py's _HISTORICAL_FIRST_PAIRS) depends on an empty
        # list, not an exception, to know it should try hf_radar instead.
        with caplog.at_level(logging.WARNING):
            out = dl.download(13.5, 15.5, 40.0, 41.0, "2021-01-01", "2021-01-02")

        assert out == []
        assert any("no delayed-mode HF-radar archive" in r.message for r in caplog.records)

    def test_multi_year_request_not_yet_supported(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(NotImplementedError, match="single calendar year"):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2020-12-30", "2021-01-02")

    def test_year_outside_split_archive_range_returns_empty(self, tmp_path, caplog):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        # US-EastGulfCoast's historical archive is split into one file per
        # year, only for 2019-2024; a request for 2018 falls outside that
        # range. Must return [] (not raise) for the same reason as the
        # unavailable-region case above.
        with caplog.at_level(logging.WARNING):
            out = dl.download(-90.0, -60.0, 30.0, 40.0, "2018-01-01", "2018-01-02")

        assert out == []
        assert any(
            "No US-EastGulfCoast historical archive for year 2018" in r.message
            for r in caplog.records
        )

    def test_download_gets_file_then_subsets_locally(self, tmp_path):
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "GDOP": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            # FileGetResult is defined at module scope in this test file (see
            # the brief's note: it's a mock stand-in only, not a symbol the
            # implementation defines or imports).
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01T04:00:00")

        assert len(out) == 1
        assert out[0].exists()
        result = xr.open_dataset(out[0])
        assert "time" in result.dims and "latitude" in result.dims and "longitude" in result.dims
        assert "DEPTH" not in result.dims
        assert result.sizes["time"] == 5

    def test_archive_with_no_data_in_requested_window_returns_empty(self, tmp_path, caplog):
        """Reproduces the ARPAS report: the archive file exists and opens
        fine, but its real TIME coverage doesn't reach the requested window
        (e.g. the region's delayed-mode processing lags further behind than
        the fixed _MIN_AGE_DAYS guard assumes). Must return [] so the
        orchestrator can fall back to the NRT downloader, not raise."""
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        # Archive only covers early 2019; the request below (2021) falls
        # entirely outside this range, mirroring ARPAS's real archive
        # (covers 2022-11-11..2025-07-03) vs. a request past 2025-07-03.
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.WARNING):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2021-06-05", "2021-06-06")

        assert out == []
        # self.output_dir.mkdir() already ran earlier in _download_region_window,
        # so the directory exists — it just must contain no .nc output.
        assert list((tmp_path / "out").glob("*.nc")) == []
        assert any("US-WestCoast" in r.message and "2021-06-05" in r.message for r in caplog.records)

    def test_out_of_order_timestamps_elsewhere_in_archive_do_not_break_slicing(self, tmp_path):
        """Reproduces the DeltaEbro report: a handful of out-of-order TIME
        values anywhere in the multi-year archive (e.g. from delayed QC
        reprocessing swapping two adjacent hourly readings) make pandas
        refuse *any* label-based time slice on that file -- even for a
        request nowhere near the bad points -- unless TIME is sorted
        first."""
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = list(pd.date_range("2019-01-01", periods=10, freq="1h"))
        # Swap two adjacent timestamps near the end out of order, far from
        # the requested window below -- mirrors DeltaEbro's real archive,
        # which has out-of-order timestamps in 2025 that broke a 2019
        # request.
        times[8], times[9] = times[9], times[8]
        times = pd.DatetimeIndex(times)
        shape = (10, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01T00:00:00", "2019-01-01T03:00:00")

        assert len(out) == 1
        result = xr.open_dataset(out[0])
        assert result.sizes["time"] == 4


class TestHFRadarHistoricalDownloaderAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # lat 0-5 doesn't overlap any HFR_REGIONS entry on either side of
        # the split (the southernmost real region, US-Hawaii, starts at
        # 14.5N). Note: the unsplit pre-fix code also raises a ValueError
        # matching this message for a min_lon > max_lon input, so this test
        # alone doesn't distinguish pre-fix from post-fix — it guards that
        # the "truly nothing covers this" case keeps failing loudly after
        # the fix too. The next test is the one that actually fails pre-fix.
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError, match="No Copernicus HF-radar region overlaps"):
            dl.download(135.0, -120.0, 0.0, 5.0, "2021-07-02", "2021-07-03")

    def test_crossing_bbox_dry_run_resolves_the_side_that_has_a_region(self, tmp_path, capsys):
        # US-Alaska's bbox (-174.10..-128.66, 68.01..74.03) overlaps the
        # [-180, -120] window but not the [135, 180] window, so only that
        # window should resolve a region.
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(135.0, -120.0, 69.0, 73.0, "2021-07-02", "2021-07-03")
        assert out == []
        assert "US-Alaska" in capsys.readouterr().out


class TestHFRadarHistoricalDownloaderForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            DATASET_ID,
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False)
        dest_path = tmp_path / f"{DATASET_ID}_US-WestCoast_2019-01-01.nc"
        dest_path.write_bytes(b"")

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01")

        fake_module.get.assert_not_called()
        assert out == [dest_path]

    def test_force_download_refetches_existing_output(self, tmp_path):
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            DATASET_ID,
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        dest_path = tmp_path / f"{DATASET_ID}_US-WestCoast_2019-01-01.nc"
        dest_path.write_bytes(b"stale")

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01T04:00:00")

        fake_module.get.assert_called_once()

    def test_raw_archive_fetched_into_shared_cache_dir_not_per_run_output_dir(
        self, tmp_path, monkeypatch
    ):
        """The raw multi-year archive (100s of MB) must be fetched into a
        fixed, run-independent cache directory, not tmp_path/output_dir/
        _raw_archive — otherwise every dated run folder re-downloads the
        same file. monkeypatch the module's cache-dir constant so the test
        doesn't touch the real repo-relative data/_archive_cache/ path."""
        from unittest.mock import patch

        import sar_validation.downloaders.hf_radar_historical_downloader as hf_hist_mod

        shared_cache = tmp_path / "shared_cache"
        monkeypatch.setattr(hf_hist_mod, "_ARCHIVE_CACHE_DIR", shared_cache)

        per_run_output = tmp_path / "run1" / "hf_radar_historical"
        dl = hf_hist_mod.HFRadarHistoricalDownloader(output_dir=per_run_output, dry_run=False)

        fake_module = MagicMock()

        def fake_get(**kwargs):
            raise FileNotFoundError("no archive file matched (test stub, not exercised further)")

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with pytest.raises(FileNotFoundError):
                dl.download(-90.0, -60.0, 30.0, 40.0, "2021-06-05", "2021-06-06")

        get_kwargs = fake_module.get.call_args.kwargs
        assert get_kwargs["output_directory"] == str(shared_cache)
        assert shared_cache.exists()
        assert not (per_run_output / "_raw_archive").exists()


class TestOrchestratorHFRadarHistoricalWiring:
    def test_dispatch_source_registers_hf_radar_historical_handler(self):
        from unittest.mock import patch

        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="currents"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar_historical")

        with patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        mock_cls.assert_called_once()


class TestOrchestratorScatterometerFTPWiring:
    @pytest.mark.parametrize("source_type,satellite", [
        ("scatterometer_hy2b", "hy2b"),
        ("scatterometer_hy2c", "hy2c"),
        ("scatterometer_oceansat3", "oceansat3"),
    ])
    def test_dispatch_source_registers_handler_with_right_satellite(self, source_type, satellite):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="wind"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type=source_type)

        with patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.ScatterometerFTPDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        assert mock_cls.call_args.kwargs["satellite"] == satellite


class TestOrchestratorCurrentsHistoricalWiring:
    @pytest.mark.parametrize("source_type,instrument", [
        ("adcp_historical", "adcp"),
        ("argo_historical", "argo"),
        ("drifter_historical", "drifter"),
        ("glider_historical", "glider"),
    ])
    def test_dispatch_source_registers_handler_with_right_instrument(self, source_type, instrument):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="currents"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type=source_type)

        with patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        assert mock_cls.call_args.kwargs["instrument"] == instrument


class TestOrchestratorHistoricalFirstDedup:
    def test_hf_radar_skipped_when_historical_covers_the_window(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-hfradar-skip",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="hf_radar"),
                ValidationDataSource(source_type="hf_radar_historical"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_nrt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_hist_cls.return_value.download.return_value = [tmp_path / "one.nc"]
            ok = orchestrator.download_all()

        assert ok is True
        mock_hist_cls.return_value.download.assert_called_once()
        mock_nrt_cls.return_value.download.assert_not_called()
        assert orchestrator.metadata["downloads"]["hf_radar"]["status"] == "skipped"
        assert orchestrator.metadata["downloads"]["hf_radar_historical"]["file_count"] == 1

    def test_hf_radar_dispatched_when_historical_returns_empty(self, tmp_path):
        """Also covers the ARPAS report: historical resolving a region but
        finding no data for this window (recency guard, archive-coverage
        gap, or unmapped region) must still let NRT fill in."""
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-hfradar-fallback",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="hf_radar"),
                ValidationDataSource(source_type="hf_radar_historical"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_nrt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_hist_cls.return_value.download.return_value = []
            mock_nrt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        mock_nrt_cls.return_value.download.assert_called_once()
        assert orchestrator.metadata["downloads"]["hf_radar"]["status"] == "dry_run"
        assert orchestrator.metadata["downloads"]["hf_radar_historical"]["file_count"] == 0

    def test_hf_radar_alone_unaffected_by_dedup_logic(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-no-pair",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[ValidationDataSource(source_type="hf_radar")],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_nrt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_nrt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        mock_nrt_cls.return_value.download.assert_called_once()
        assert orchestrator.metadata["downloads"]["hf_radar"]["status"] == "dry_run"

    def test_drifter_excluded_from_nrt_batch_when_historical_covers_it(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-drifter-skip",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="drifter_historical"),
                ValidationDataSource(source_type="mooring"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_hist_cls.return_value.download.return_value = [tmp_path / "drifter.csv"]
            mock_insitu_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        assert orchestrator.metadata["downloads"]["insitu"]["source_types"] == ["mooring"]

    def test_drifter_kept_in_nrt_batch_when_historical_returns_empty(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-drifter-fallback",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="drifter_historical"),
                ValidationDataSource(source_type="mooring"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_hist_cls.return_value.download.return_value = []
            mock_insitu_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        source_types = orchestrator.metadata["downloads"]["insitu"]["source_types"]
        assert sorted(source_types) == ["drifter", "mooring"]

    def test_insitu_batch_fully_skipped_when_only_drifter_and_historical_covers_it(
        self, tmp_path
    ):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-insitu-full-skip",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="drifter_historical"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_hist_cls.return_value.download.return_value = [tmp_path / "drifter.csv"]
            ok = orchestrator.download_all()

        assert ok is True
        mock_insitu_cls.assert_not_called()
        assert "insitu" not in orchestrator.metadata["downloads"]

    def test_insitu_batch_depth_window_ignores_excluded_drifter(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-depth-window",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter", min_depth=-500.0, max_depth=500.0),
                ValidationDataSource(source_type="drifter_historical"),
                ValidationDataSource(source_type="mooring"),  # default -20/20
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_hist_cls.return_value.download.return_value = [tmp_path / "drifter.csv"]
            mock_insitu_cls.return_value.download.return_value = []
            orchestrator.download_all()

        # drifter's -500/500 depth override must not widen the NRT batch's
        # depth window, since drifter itself was excluded from that batch.
        _, insitu_ctor_kwargs = mock_insitu_cls.call_args
        assert insitu_ctor_kwargs["min_depth"] == -20.0
        assert insitu_ctor_kwargs["max_depth"] == 20.0


# ---------------------------------------------------------------------------
# End-to-end: orchestrator wiring for a Pacific-crossing recipe
# ---------------------------------------------------------------------------

class TestOrchestratorAntimeridianDryRun:
    def test_pacific_crossing_recipe_wires_through_without_error(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            SARDataSpec,
            TemporalBounds,
            ValidationDataSource,
        )

        cfg = RecipeConfig(
            name="pacific_dry_run_test",
            variable="waves",
            geographic_bounds=GeographicBounds(min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0),
            temporal_bounds=TemporalBounds(start="2026-07-02", end="2026-07-03"),
            sar_data=SARDataSpec(swath_mode=["WV", "SM"]),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="tidal_gauge"),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="altimeter"),
            ],
            output_dir=str(tmp_path),
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls, patch(
            "sar_validation.downloaders.altimeter_downloader.AltimeterDownloader"
        ) as mock_alt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_insitu_cls.return_value.download.return_value = []
            mock_alt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        _, sar_kwargs = mock_sar_cls.return_value.download.call_args
        assert (sar_kwargs["min_lon"], sar_kwargs["max_lon"]) == (135.0, -120.0)
        _, insitu_kwargs = mock_insitu_cls.return_value.download.call_args
        assert (insitu_kwargs["min_lon"], insitu_kwargs["max_lon"]) == (135.0, -120.0)
        _, alt_kwargs = mock_alt_cls.return_value.download.call_args
        assert (alt_kwargs["min_lon"], alt_kwargs["max_lon"]) == (135.0, -120.0)

    def test_waves_pacific_recipe_loads_with_crossing_convention(self):
        from sar_validation.core.recipe import Recipe

        recipe = Recipe.from_yaml("recipes/waves_pacific.yaml")
        bounds = recipe.config.geographic_bounds
        assert bounds.min_lon == 135.0
        assert bounds.max_lon == -120.0
        assert bounds.min_lon > bounds.max_lon  # crossing convention


# ---------------------------------------------------------------------------
# DataOrchestrator force_download wiring
# ---------------------------------------------------------------------------

class TestOrchestratorForceDownloadWiring:
    def _make_orchestrator(self, tmp_path, force_download):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe = Recipe(RecipeConfig(
            name="test-force-download",
            variable="wind",
            output_dir=str(tmp_path),
        ))
        return DataOrchestrator(recipe, dry_run=True, force_download=force_download)

    def test_sar_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.sar_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_sar()
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_insitu_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.insitu_downloader.InSituDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_insitu(["mooring"], -20.0, 20.0)
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_scatterometer_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.scatterometer_downloader.ScatterometerDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_scatterometer(ValidationDataSource(source_type="scatterometer"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_hf_radar_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_hf_radar(ValidationDataSource(source_type="hf_radar"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_noaa_hfradar_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.NOAAHFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_noaa_hfradar(ValidationDataSource(source_type="hf_radar_noaa"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_hf_radar_historical_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_hf_radar_historical(
                ValidationDataSource(source_type="hf_radar_historical")
            )
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_scatterometer_ftp_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.ScatterometerFTPDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_scatterometer_hy2b(
                ValidationDataSource(source_type="scatterometer_hy2b")
            )
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_currents_historical_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_adcp_historical(
                ValidationDataSource(source_type="adcp_historical")
            )
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_altimeter_receives_force_download(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import ValidationDataSource

        orchestrator = self._make_orchestrator(tmp_path, force_download=True)
        with patch("sar_validation.downloaders.altimeter_downloader.AltimeterDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_altimeter(ValidationDataSource(source_type="altimeter"))
        assert mock_cls.call_args.kwargs["force_download"] is True

    def test_default_force_download_is_false(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=False)
        with patch("sar_validation.downloaders.sar_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_sar()
        assert mock_cls.call_args.kwargs["force_download"] is False
