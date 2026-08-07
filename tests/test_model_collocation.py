"""Tests for model_collocation.py -- ERA5/SAR bilinear + temporal interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestBuildSpatialInterpolator:
    def test_interpolates_bilinearly_between_grid_points(self):
        from sar_validation.core.model_collocation import build_spatial_interpolator

        lat_ax = np.array([40.0, 41.0, 42.0])
        lon_ax = np.array([-10.0, -9.0, -8.0])
        field = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        interp = build_spatial_interpolator(lat_ax, lon_ax, field)

        # Exact grid point returns the exact value
        assert interp([[41.0, -9.0]])[0] == 5.0
        # Midpoint between two grid cells is the average
        assert interp([[40.5, -10.0]])[0] == 2.5

    def test_out_of_bounds_query_returns_nan(self):
        from sar_validation.core.model_collocation import build_spatial_interpolator

        lat_ax = np.array([40.0, 41.0])
        lon_ax = np.array([-10.0, -9.0])
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        interp = build_spatial_interpolator(lat_ax, lon_ax, field)

        assert np.isnan(interp([[50.0, -9.5]])[0])

    def test_nan_field_value_propagates_to_nearby_query(self):
        from sar_validation.core.model_collocation import build_spatial_interpolator

        lat_ax = np.array([40.0, 41.0])
        lon_ax = np.array([-10.0, -9.0])
        field = np.array([[1.0, np.nan], [3.0, 4.0]])
        interp = build_spatial_interpolator(lat_ax, lon_ax, field)

        assert np.isnan(interp([[40.5, -9.5]])[0])


class TestHyperbolicInterp:
    def test_returns_val2_at_t_prime_zero(self):
        from sar_validation.core.model_collocation import _hyperbolic_interp

        result = _hyperbolic_interp(
            np.array([1.0]), np.array([5.0]), np.array([9.0]), np.array([0.0]),
        )
        assert result[0] == 5.0

    def test_linear_series_interpolates_linearly(self):
        # For an exactly linear series (1, 5, 9 -- constant slope 4),
        # the quadratic term vanishes and this reduces to linear
        # interpolation between val2 and val3.
        from sar_validation.core.model_collocation import _hyperbolic_interp

        result = _hyperbolic_interp(
            np.array([1.0]), np.array([5.0]), np.array([9.0]), np.array([0.5]),
        )
        assert result[0] == 7.0

    def test_matches_reference_script_formula(self):
        # a = (val3 + val1 - 2*val2) / 2; b = (val3 - val1) / 2; c = val2
        # result = a*t^2 + b*t + c -- verify against a hand-computed case
        # with real curvature (val1=1, val2=2, val3=6).
        from sar_validation.core.model_collocation import _hyperbolic_interp

        val1, val2, val3, t = 1.0, 2.0, 6.0, 0.3
        a = (val3 + val1 - 2.0 * val2) / 2.0
        b = (val3 - val1) / 2.0
        c = val2
        expected = a * t**2 + b * t + c

        result = _hyperbolic_interp(
            np.array([val1]), np.array([val2]), np.array([val3]), np.array([t]),
        )
        assert result[0] == pytest.approx(expected)


def _make_era5_ds(n_lat=3, n_lon=3, hours=("2026-07-12T00:00:00", "2026-07-12T01:00:00", "2026-07-12T02:00:00")):
    import xarray as xr

    lat = np.linspace(40.0, 42.0, n_lat)
    lon = np.linspace(-10.0, -8.0, n_lon)
    time = pd.to_datetime(list(hours))
    # u10 is a simple, known ramp so interpolated values are predictable:
    # value(lat, lon, hour_index) = hour_index * 10 (spatially constant),
    # so temporal interpolation is exactly checkable regardless of query
    # location.
    u10 = np.stack([np.full((n_lat, n_lon), h * 10.0) for h in range(len(hours))])
    ds = xr.Dataset(
        {"u10": (("time", "lat", "lon"), u10)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds


class TestModelValuesAtPoints:
    # NOTE: _make_era5_ds's grid spans lon [-10, -8] and lat [40, 42] --
    # these ranges don't overlap, so query points below deliberately use
    # distinct lon/lat values within each's own range (e.g. lon=-9.0,
    # lat=41.0), never the same number for both.

    def test_nearest_hour_picks_closest_hour_value(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_ds()
        # 00:50 is nearest to hour 1 (01:00) -> expect value 10.0
        times = np.array([np.datetime64("2026-07-12T00:50:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "nearest",
        )
        assert result["u10"][0] == pytest.approx(10.0)

    def test_hyperbolic_interpolates_between_hours(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_ds()
        # Exactly halfway between hour 1 (value 10) and hour 2 (value 20):
        # linear part of the quadratic dominates for this evenly-spaced
        # ramp -- expect 15.0.
        times = np.array([np.datetime64("2026-07-12T01:30:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "hyperbolic",
        )
        assert result["u10"][0] == pytest.approx(15.0)

    def test_hyperbolic_returns_nan_when_no_bracketing_hour(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_ds()  # only hours 0,1,2 available
        # 02:30 needs hour 3, which doesn't exist.
        times = np.array([np.datetime64("2026-07-12T02:30:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "hyperbolic",
        )
        assert np.isnan(result["u10"][0])

    def test_multiple_points_sharing_one_time_are_batched(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_ds()
        times = np.array([np.datetime64("2026-07-12T01:00:00")] * 3)
        result = _model_values_at_points(
            np.array([-9.5, -9.0, -8.5]), np.array([40.5, 41.0, 41.5]),
            times, era5_ds, "nearest",
        )
        assert result["u10"].shape == (3,)
        assert np.allclose(result["u10"], 10.0)

    def test_out_of_grid_point_returns_nan(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_ds()
        times = np.array([np.datetime64("2026-07-12T01:00:00")])
        result = _model_values_at_points(
            np.array([80.0]), np.array([80.0]), times, era5_ds, "nearest",
        )
        assert np.isnan(result["u10"][0])


class TestModelLayerCollocationIndividualGrid:
    def test_produces_one_match_per_valid_sar_pixel(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds()
        sar_lon = np.array([[-9.5, -9.0], [-9.5, -9.0]])
        sar_lat = np.array([[41.0, 41.0], [40.5, 40.5]])
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.array([[[5.0, 6.0], [7.0, 8.0]]])}

        colloc = ModelLayerCollocation(method="individual", temporal_method="nearest")
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 4
        for r in results:
            assert r.val_data["u10"] == pytest.approx(10.0)
            assert r.spatial_distance_km == 0.0
            assert r.temporal_distance_minutes == 0.0
            assert r.val_source == "era5"
            assert r.collocation_type == "model_vs_layer"

    def test_nan_sar_pixel_produces_no_match(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds()
        sar_lon = np.array([[-9.5]])
        sar_lat = np.array([[41.0]])
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.array([[[np.nan]]])}

        colloc = ModelLayerCollocation(method="individual", temporal_method="nearest")
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert results == []


class TestModelLayerCollocationIndividualPoints:
    def test_wv_mode_always_interpolates_directly_regardless_of_method(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds()
        sar_point_vars = {"oswHs": np.array([2.0, 3.0])}
        sar_lons = np.array([-9.5, -9.0])
        sar_lats = np.array([41.0, 40.5])
        sar_times = np.array([
            np.datetime64("2026-07-12T01:00:00"),
            np.datetime64("2026-07-12T01:00:00"),
        ])

        # method="cell-averaging" globally, but WV points still use direct
        # interpolation -- there's no dense SAR grid to aggregate within one
        # ERA5 cell for sparse imagette points.
        colloc = ModelLayerCollocation(method="cell-averaging", temporal_method="nearest")
        results = colloc.collocate_points(
            sar_point_vars=sar_point_vars, sar_lons=sar_lons, sar_lats=sar_lats,
            sar_times=sar_times, era5_ds=era5_ds, val_source="era5", sar_scene_name="wv1",
        )
        assert len(results) == 2
        assert all(r.val_data["u10"] == pytest.approx(10.0) for r in results)
        assert all(r.collocation_type == "model_vs_layer" for r in results)
