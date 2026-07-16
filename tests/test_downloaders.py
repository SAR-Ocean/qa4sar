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
    clamp_to_region_bbox,
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


# ---------------------------------------------------------------------------
# Tests for DataOrchestrator "hf_radar_noaa" wiring (Task 7)
# ---------------------------------------------------------------------------
# DataOrchestrator can be built cheaply from a stub Recipe (no network, no
# real base-dir creation under dry_run), so a behavioural test is preferred
# over source-inspection alone.

class TestOrchestratorHFRadarNOAAWiring:
    def test_dispatch_source_registers_hf_radar_noaa_handler(self):
        from sar_validation.core.orchestrator import DataOrchestrator
        import inspect

        src = inspect.getsource(DataOrchestrator._dispatch_source)
        assert '"hf_radar_noaa"' in src
        assert "_download_noaa_hfradar" in src
        assert hasattr(DataOrchestrator, "_download_noaa_hfradar")

    def test_download_noaa_hfradar_dry_run_sets_metadata_and_makes_no_network_call(
        self, tmp_path
    ):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            Recipe, RecipeConfig, GeographicBounds, TemporalBounds,
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
        assert out is None
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "radar-total--US-EastGulfCoast" in captured

    def test_download_calls_subset_with_resolved_region_part(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            # Simulate copernicusmarine writing the requested file.
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert out is not None
        assert out.exists()
        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "monthly-radar-total--US-EastGulfCoast"
        assert kwargs["minimum_longitude"] == -90.0
        assert kwargs["maximum_longitude"] == -60.0

    def test_recent_date_uses_latest_part_when_region_has_one(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from datetime import datetime, timedelta, timezone
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
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from datetime import datetime, timedelta, timezone
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

        assert out is not None
        assert out.exists()
        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert first_kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"
        assert second_kwargs["dataset_part"] == "monthly-radar-total--US-WestCoast"

    def test_raises_file_not_found_when_subset_writes_no_file(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()
        # subset() "succeeds" (no exception) but never writes the destination
        # file, simulating an empty/no-op response from copernicusmarine.
        fake_module.subset.side_effect = lambda **kwargs: None

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with pytest.raises(FileNotFoundError):
                dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")


# ---------------------------------------------------------------------------
# Tests for HFRadarHistoricalDownloader
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Any


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
        assert out is None
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc" in captured

    def test_unavailable_region_raises_clear_error(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        # GoS (Italy) has an NRT feed but no delayed-mode archive.
        with pytest.raises(ValueError, match="no delayed-mode HF-radar archive"):
            dl.download(13.5, 15.5, 40.0, 41.0, "2021-01-01", "2021-01-02")

    def test_multi_year_request_not_yet_supported(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(NotImplementedError, match="single calendar year"):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2020-12-30", "2021-01-02")

    def test_download_gets_file_then_subsets_locally(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )
        import xarray as xr
        import numpy as np
        import pandas as pd

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

        assert out is not None
        assert out.exists()
        result = xr.open_dataset(out)
        assert "time" in result.dims and "latitude" in result.dims and "longitude" in result.dims
        assert "DEPTH" not in result.dims
        assert result.sizes["time"] == 5
