"""Tests for ERA5Downloader."""

from __future__ import annotations

from datetime import date, datetime


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
        assert requested_days == [date(2026, 7, 12)]

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
        assert requested_days == [date(2026, 7, 12), date(2026, 7, 13), date(2026, 7, 14)]

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
        assert called == []
        assert paths == [existing]


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
        assert requested == [(-10.0, 10.0, None)]
        assert paths == [tmp_path / "era5_wind_20260712.nc"]

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
        assert requested == [(170.0, 180.0, 0), (-180.0, -170.0, 1)]
        assert paths == [
            tmp_path / "era5_wind_20260712_w0.nc",
            tmp_path / "era5_wind_20260712_w1.nc",
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
