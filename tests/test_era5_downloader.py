"""Tests for ERA5Downloader."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestHoursNeededForDay:
    def test_narrow_window_within_one_day(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 12),
            datetime(2026, 7, 12, 18, 0, 0),
            datetime(2026, 7, 12, 23, 0, 0),
        )
        # [18-2, 23+2] = [16, 25] clipped to [0, 23]
        assert hours == list(range(16, 24))

    def test_window_crossing_midnight_first_day(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 12),
            datetime(2026, 7, 12, 22, 0, 0),
            datetime(2026, 7, 13, 2, 0, 0),
        )
        # [22-2, 26] clipped to day-1's own hours: [20, 23]
        assert hours == [20, 21, 22, 23]

    def test_window_crossing_midnight_second_day(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 13),
            datetime(2026, 7, 12, 22, 0, 0),
            datetime(2026, 7, 13, 2, 0, 0),
        )
        # [20, 4] clipped to day-2's own hours: [0, 4] -> [0, 0..4] but day only has 0-23,
        # buffered end 02:00+2h=04:00
        assert hours == [0, 1, 2, 3, 4]

    def test_wide_window_interior_day_gets_all_hours(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 13),
            datetime(2026, 7, 10, 0, 0, 0),
            datetime(2026, 7, 20, 0, 0, 0),
        )
        assert hours == list(range(24))

    def test_day_entirely_outside_window_returns_empty(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 20),
            datetime(2026, 7, 12, 0, 0, 0),
            datetime(2026, 7, 12, 23, 0, 0),
        )
        assert hours == []


class TestERA5DownloaderNcPath:
    def test_nc_path_naming_wind(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        assert dl._nc_path_for_day(date(2026, 7, 12)).name == "era5_wind_20260712.nc"

    def test_nc_path_naming_soil_moisture(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="soil_moisture", output_dir=tmp_path)
        assert dl._nc_path_for_day(date(2026, 7, 12)).name == "era5_soil_moisture_20260712.nc"


class TestERA5DownloaderBuildArea:
    def test_area_padded_by_native_grid_cell_wind(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        area = dl._build_area(min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0)
        # CDS area order: [north, west, south, east]
        assert area == [55.25, -10.25, 39.75, 10.25]

    def test_area_padded_by_native_grid_cell_soil_moisture(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="soil_moisture", output_dir=tmp_path)
        area = dl._build_area(min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0)
        assert area == [55.1, -10.1, 39.9, 10.1]


class TestERA5DownloaderBuildRequest:
    def test_build_request_wind(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        req = dl._build_request(date(2026, 7, 12), [18, 19, 20], -10.0, 10.0, 40.0, 55.0)
        # land_sea_mask is requested alongside u10/v10 (same CDS call, no
        # extra download round-trip) so ModelLayerCollocation can skip ERA5
        # grid cells whose own center is land -- see
        # sar_validation.core.model_collocation._collocate_cell_averaging_grid.
        assert req["variable"] == [
            "10m_u_component_of_wind", "10m_v_component_of_wind", "land_sea_mask",
        ]
        assert req["year"] == ["2026"]
        assert req["month"] == ["07"]
        assert req["day"] == ["12"]
        assert req["time"] == ["18:00", "19:00", "20:00"]
        assert req["data_format"] == "netcdf"
        # Explicit, not left to the CDS backend's per-dataset default --
        # see the comment in _build_request for why (live 2026-08-07:
        # omitting this made reanalysis-era5-land silently return a ZIP
        # archive saved with a misleading ".nc" extension).
        assert req["download_format"] == "unarchived"
        assert req["product_type"] == ["reanalysis"]

    def test_build_request_waves(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="waves", output_dir=tmp_path)
        req = dl._build_request(date(2026, 7, 12), [0], -10.0, 10.0, 40.0, 55.0)
        # No land_sea_mask here: a real downloaded era5_waves_*.nc already
        # has swh natively NaN'd over land grid points (ECMWF's ocean wave
        # model), confirmed live 2026-08-10 -- no land-mask fix needed.
        assert req["variable"] == ["significant_height_of_combined_wind_waves_and_swell"]
        assert "land_sea_mask" not in req["variable"]

    def test_build_request_soil_moisture_has_no_product_type_facet(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="soil_moisture", output_dir=tmp_path)
        req = dl._build_request(date(2026, 7, 12), [0], -10.0, 10.0, 40.0, 55.0)
        # "layer_1", not "level_1" -- the live CDS variable enum uses
        # "volumetric_soil_water_layer_1"; the "level_1" spelling passes
        # cdsapi's own client-side validation (no enum-checking there) but
        # the CDS backend then finds no matching data (MultiAdaptorNoDataError).
        assert req["variable"] == ["volumetric_soil_water_layer_1"]
        assert req["download_format"] == "unarchived"
        assert "product_type" not in req
        # reanalysis-era5-land is land-only by definition -- a land-sea
        # mask is nonsensical here.
        assert "land_sea_mask" not in req["variable"]


class TestERA5DownloaderInvalidVariable:
    def test_rejects_unknown_variable(self, tmp_path):
        import pytest

        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        with pytest.raises(ValueError, match="variable must be one of"):
            ERA5Downloader(variable="currents", output_dir=tmp_path)


class TestERA5DownloaderDryRun:
    def test_dry_run_returns_empty_without_network_call(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T18:00:00", end="2026-07-12T23:00:00",
        )
        assert paths == []
        assert not list(tmp_path.glob("*.nc"))

    def test_dry_run_returns_already_cached_files(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        existing = tmp_path / "era5_wind_20260712.nc"
        existing.write_text("fake")

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T18:00:00", end="2026-07-12T23:00:00",
        )
        assert len(paths) == 1
        assert paths[0].name == "era5_wind_20260712.nc"


class TestERA5DownloaderDayLoop:
    def test_skips_days_with_no_hours_needed(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        requested_days: list[date] = []

        def fake_download_day(self, day, hours, min_lon, max_lon, min_lat, max_lat, window_idx=None):
            requested_days.append(day)
            return None

        monkeypatch.setattr(ERA5Downloader, "_download_day", fake_download_day)

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T18:00:00", end="2026-07-12T23:00:00",
        )
        # With 2h buffer: [16:00, 01:00+1day], so both days are visited and have hours needed
        assert requested_days == [date(2026, 7, 12), date(2026, 7, 13)]

    def test_multi_day_window_requests_every_day(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        requested_days: list[date] = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, *a: requested_days.append(day) or None,
        )

        dl = ERA5Downloader(variable="soil_moisture", output_dir=tmp_path)
        dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T00:00:00", end="2026-07-14T00:00:00",
        )
        # With 2h buffer: [22:00-1day, 02:00+1day], so 2026-07-11 and 2026-07-14 also visited
        assert requested_days == [date(2026, 7, 11), date(2026, 7, 12), date(2026, 7, 13), date(2026, 7, 14)]

    def test_previous_day_requested_when_tolerance_buffer_needs_it(self, tmp_path, monkeypatch):
        """Regression: a SAR scene at 00:00 UTC on the recipe's literal
        start date with a wide tolerance (e.g. soil moisture's 720 min /
        ±12h) needs ERA5 hours from the *previous* calendar day -- but the
        day-loop's bounds came from the literal, unpadded start/end
        dates, so that day was never visited at all, silently dropping
        every hour it would have contributed. Reproduces
        recipes/soil_moisture_era5.yaml's first-day-has-no-overlap bug."""
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        requested_days: list[date] = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, *a: requested_days.append(day) or None,
        )

        dl = ERA5Downloader(
            variable="soil_moisture", output_dir=tmp_path, time_tolerance_minutes=720,
        )
        dl.download(
            min_lon=-5.0, max_lon=5.0, min_lat=45.0, max_lat=52.0,
            start="2026-07-10T00:00:00", end="2026-07-10T01:00:00",
        )

        assert date(2026, 7, 9) in requested_days, (
            "the day before the literal start date must be requested when "
            "the ±12h tolerance buffer needs hours from it"
        )
        assert requested_days == [date(2026, 7, 9), date(2026, 7, 10)]

    def test_next_day_requested_when_tolerance_buffer_needs_it(self, tmp_path, monkeypatch):
        """Symmetric case on the end side."""
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        requested_days: list[date] = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, *a: requested_days.append(day) or None,
        )

        dl = ERA5Downloader(
            variable="soil_moisture", output_dir=tmp_path, time_tolerance_minutes=720,
        )
        dl.download(
            min_lon=-5.0, max_lon=5.0, min_lat=45.0, max_lat=52.0,
            start="2026-07-12T23:00:00", end="2026-07-12T23:59:00",
        )

        assert date(2026, 7, 13) in requested_days
        assert requested_days == [date(2026, 7, 12), date(2026, 7, 13)]

    def test_existing_day_file_is_skipped_and_returned(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        called = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, *a: called.append(day) or None,
        )

        existing = tmp_path / "era5_wind_20260712.nc"
        existing.write_text("fake")

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T18:00:00", end="2026-07-12T23:00:00",
        )
        # 2026-07-12's file exists so it's not downloaded, but 2026-07-13 still needs downloading
        assert called == [date(2026, 7, 13)]
        assert paths == [existing]


class TestERA5DownloaderTimeToleranceMinutes:
    """The hour-buffer margin is now driven by time_tolerance_minutes
    (recipe-resolved by the orchestrator, see
    orchestrator._resolve_temporal_padding_minutes), not the fixed
    _HOUR_BUFFER constant -- mirrors HycomDownloader's identical fix."""

    def test_time_tolerance_minutes_constructor_arg_drives_the_hour_buffer(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        captured_hours: list = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, *a: captured_hours.append(hours) or None,
        )

        # 5h buffer (not the 2h default) on a window starting at 18:00 ->
        # hour 13 must now be included (18 - 5 = 13), which the 2h default
        # would exclude.
        dl = ERA5Downloader(variable="wind", output_dir=tmp_path, time_tolerance_minutes=300)
        dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T18:00:00", end="2026-07-12T18:30:00",
        )
        assert captured_hours == [list(range(13, 24))]


class TestERA5DownloaderAntimeridian:
    def test_non_crossing_bbox_downloads_single_unsuffixed_file(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        requested = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, mn, mx, mnlat, mxlat, window_idx=None: (
                requested.append((mn, mx, window_idx)) or self._nc_path_for_day(day, window_idx)
            ),
        )
        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T00:00:00", end="2026-07-12T23:00:00",
        )
        # With 2h buffer: [22:00-1day, 01:00+1day], visits 2026-07-11, 2026-07-12, 2026-07-13
        assert requested == [(-10.0, 10.0, None), (-10.0, 10.0, None), (-10.0, 10.0, None)]
        assert paths == [
            tmp_path / "era5_wind_20260711.nc",
            tmp_path / "era5_wind_20260712.nc",
            tmp_path / "era5_wind_20260713.nc",
        ]

    def test_crossing_bbox_downloads_two_suffixed_files(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        requested = []
        monkeypatch.setattr(
            ERA5Downloader, "_download_day",
            lambda self, day, hours, mn, mx, mnlat, mxlat, window_idx=None: (
                requested.append((mn, mx, window_idx)) or self._nc_path_for_day(day, window_idx)
            ),
        )
        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        paths = dl.download(
            min_lon=170.0, max_lon=-170.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-12T00:00:00", end="2026-07-12T23:00:00",
        )
        # With 2h buffer: [22:00-1day, 01:00+1day], visits 3 days × 2 antimeridian windows = 6 requests
        assert requested == [
            (170.0, 180.0, 0), (-180.0, -170.0, 1),  # 2026-07-11
            (170.0, 180.0, 0), (-180.0, -170.0, 1),  # 2026-07-12
            (170.0, 180.0, 0), (-180.0, -170.0, 1),  # 2026-07-13
        ]
        assert paths == [
            tmp_path / "era5_wind_20260711_w0.nc",
            tmp_path / "era5_wind_20260711_w1.nc",
            tmp_path / "era5_wind_20260712_w0.nc",
            tmp_path / "era5_wind_20260712_w1.nc",
            tmp_path / "era5_wind_20260713_w0.nc",
            tmp_path / "era5_wind_20260713_w1.nc",
        ]

    def test_build_area_clips_padding_at_dateline(self, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        # East window touches 180 exactly -- padding must not push past it.
        area = dl._build_area(min_lon=170.0, max_lon=180.0, min_lat=40.0, max_lat=55.0)
        assert area == [55.25, 169.75, 39.75, 180.0]
        # West window touches -180 exactly.
        area = dl._build_area(min_lon=-180.0, max_lon=-170.0, min_lat=40.0, max_lat=55.0)
        assert area == [55.25, -180.0, 39.75, -169.75]


class TestERA5DownloaderRealDownloadPrintsProgress:
    """A real (non-dry-run) download must print a progress message to the
    terminal, matching every other downloader (scatterometer_ftp_downloader,
    altimeter_downloader, smos_downloader, ...), each of which announces its
    fetch via ``print(...)`` rather than ``logger.info(...)`` alone -- the
    CLI's root logger defaults to WARNING (see cli.py's
    ``logging.basicConfig(level=logging.WARNING, ...)``), so an INFO-only
    message is invisible in a normal (non ``--verbose``) run and a user
    watching the terminal sees nothing happen during the CDS request, which
    can take a long time."""

    def test_download_day_prints_progress_message(self, tmp_path, capsys):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        fake_client = MagicMock()
        with patch.dict(sys.modules, {"cdsapi": MagicMock(Client=MagicMock(return_value=fake_client))}):
            dl._download_day(date(2026, 7, 12), [18, 19, 20], -10.0, 10.0, 40.0, 55.0)

        captured = capsys.readouterr().out
        assert "Downloading" in captured
        assert "ERA5" in captured
        assert "wind" in captured
        assert "2026-07-12" in captured


class TestERA5DownloaderCheckAvailabilityDry:
    """check_availability_dry is a fast, unauthenticated existence probe for
    dry-collocation prediction -- queries the CDS catalogue's live
    collection-metadata endpoint (``ecmwf.datastores.Client.get_collection``)
    for the variable's own dataset's real temporal extent, rather than
    submitting a real cdsapi.Client.retrieve() processing job the way
    _download_day does. Mirrors
    CDSSoilMoistureDownloader.check_availability_dry's own test suite."""

    @staticmethod
    def _patch_datastores(monkeypatch, begin=None, end=None, client_cls=None):
        """Install a fake ``ecmwf.datastores`` module in sys.modules whose
        Client(...).get_collection(...) returns a fake Collection exposing
        begin_datetime/end_datetime. Returns (fake_client_cls,
        fake_client_instance) so tests can assert on construction/calls."""
        fake_collection = MagicMock(begin_datetime=begin, end_datetime=end)
        fake_client_instance = MagicMock()
        fake_client_instance.get_collection.return_value = fake_collection
        fake_client_cls = client_cls or MagicMock(return_value=fake_client_instance)

        fake_datastores_module = MagicMock()
        fake_datastores_module.Client = fake_client_cls

        fake_ecmwf_module = MagicMock()
        fake_ecmwf_module.datastores = fake_datastores_module

        monkeypatch.setitem(sys.modules, "ecmwf", fake_ecmwf_module)
        monkeypatch.setitem(sys.modules, "ecmwf.datastores", fake_datastores_module)
        return fake_client_cls, fake_client_instance

    def test_true_when_day_falls_within_live_extent(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        self._patch_datastores(
            monkeypatch,
            begin=datetime(1940, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        assert dl.check_availability_dry(date(2026, 3, 15)) is True

    def test_false_when_day_falls_outside_live_extent(self, monkeypatch, tmp_path):
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        self._patch_datastores(
            monkeypatch,
            begin=datetime(1940, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        assert dl.check_availability_dry(date(2026, 9, 1)) is False

    def test_queries_the_dataset_matching_the_configured_variable(self, monkeypatch, tmp_path):
        """soil_moisture uses a different CDS dataset (reanalysis-era5-land)
        than wind/waves (reanalysis-era5-single-levels) -- see
        _CDS_DATASET_BY_VARIABLE. The right one must be looked up for
        whichever variable this downloader instance was built for."""
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        _fake_cls, fake_client = self._patch_datastores(
            monkeypatch,
            begin=datetime(1940, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        dl = ERA5Downloader(variable="soil_moisture", output_dir=tmp_path)
        dl.check_availability_dry(date(2026, 3, 15))

        fake_client.get_collection.assert_called_once_with("reanalysis-era5-land")
        # No blocking cdsapi.Client.retrieve() processing job is ever submitted.
        fake_client.retrieve.assert_not_called()

    def test_raises_when_catalogue_lookup_fails(self, monkeypatch, tmp_path):
        """A network/auth/API error while querying the catalogue (e.g. no
        connectivity, or the endpoint itself erroring) must propagate --
        never be swallowed into False -- so _predict_model_source's own
        exception handling is what converts it to an 'unknown' verdict,
        never a false 'none-predicted'."""
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        failing_client_cls = MagicMock(side_effect=RuntimeError("connection refused"))
        self._patch_datastores(monkeypatch, client_cls=failing_client_cls)

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="connection refused"):
            dl.check_availability_dry(date(2026, 3, 15))

    def test_raises_when_ecmwf_datastores_not_installed(self, monkeypatch, tmp_path):
        """A missing cdsapi/ecmwf-datastores-client dependency must
        propagate as an ImportError so the caller's own exception handling
        produces 'unknown', never a false 'none-predicted'."""
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        monkeypatch.setitem(sys.modules, "ecmwf.datastores", None)

        with pytest.raises(ImportError):
            dl.check_availability_dry(date(2026, 3, 15))

    def test_raises_when_catalogue_extent_is_missing(self, monkeypatch, tmp_path):
        """A malformed/incomplete catalogue response (no usable
        begin/end datetime) can't answer the "does data exist" question
        either -- this must raise, not silently return False."""
        from sar_validation.downloaders.era5_downloader import ERA5Downloader

        self._patch_datastores(monkeypatch, begin=None, end=None)

        dl = ERA5Downloader(variable="wind", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="temporal extent"):
            dl.check_availability_dry(date(2026, 3, 15))
