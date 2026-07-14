"""Tests for downloader utilities: datetime parsing and dataset_part selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from sar_validation.downloaders.base import normalize_datetime, is_date_recent
from sar_validation.downloaders.insitu_downloader import (
    SOURCE_TYPE_TO_PLATFORM,
    PLATFORM_CODE_TO_SOURCE_TYPE,
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
        yesterday = datetime(2026, 7, 1)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-07-01", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_30_days_ago_is_recent(self, mock_datetime):
        """30 days ago should be at the boundary of recent."""
        today = datetime(2026, 7, 2)
        thirty_days_ago = datetime(2026, 6, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-06-02", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_31_days_ago_is_not_recent(self, mock_datetime):
        """31 days ago should exceed the 30-day threshold."""
        today = datetime(2026, 7, 2)
        thirty_one_days_ago = datetime(2026, 6, 1)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-06-01", threshold_days=30)
        assert result is False

    @patch("sar_validation.downloaders.base.datetime")
    def test_old_date_is_not_recent(self, mock_datetime):
        """Date from several months ago should not be recent."""
        today = datetime(2026, 7, 2)
        old_date = datetime(2026, 3, 15)
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
        sixty_days_ago = datetime(2026, 5, 3)
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
    RadiometerDownloader, SENSORS, SUPPORTED_SENSORS,
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
    read_rss_bytemap, BYTEMAP_LAYOUT, NPASS, NLAT, NLON,
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
    select_erddap_dataset,
    build_erddap_subset_url,
    select_backend,
    ERDDAP_BASE,
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
    def test_dry_run_returns_none_and_no_fetch(self, tmp_path, capsys):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert out is None
        m.assert_not_called()
        assert "ucsdHfrW6.nc?" in capsys.readouterr().out

    def test_download_fetches_url_to_expected_path(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert out is not None
        assert out.parent == tmp_path
        assert out.suffix == ".nc"
        m.assert_called_once()
        called_url, called_path = m.call_args[0][0], m.call_args[0][1]
        assert "ucsdHfrW6.nc?" in called_url
        assert str(out) == str(called_path)
