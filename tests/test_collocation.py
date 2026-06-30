"""Tests for the collocation algorithms (step 3)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sar_validation.core.collocation import (
    CollocatedPoint,
    PointLayerCollocation,
    LayerLayerCollocation,
    TrajectoryLayerCollocation,
    _haversine_distance,
    _haversine_distance_grid,
    _to_datetime_array,
    _detect_collocation_type,
    TRAJECTORY_PLATFORM_TYPES,
    LAYER_DATA_TYPES,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestHaversineDistance:
    def test_same_point(self):
        assert _haversine_distance(0.0, 50.0, 0.0, 50.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # 1 degree latitude ≈ 111 km
        d = _haversine_distance(0.0, 50.0, 0.0, 51.0)
        assert 110.0 < d < 113.0

    def test_symmetry(self):
        d1 = _haversine_distance(-5.0, 52.0, 0.0, 55.0)
        d2 = _haversine_distance(0.0, 55.0, -5.0, 52.0)
        assert d1 == pytest.approx(d2, rel=1e-9)


class TestToDatetimeArray:
    def test_python_datetime(self):
        dt = datetime(2026, 1, 1)
        arr = _to_datetime_array([dt])
        assert arr[0] == dt

    def test_numpy_datetime64(self):
        dt64 = np.datetime64("2026-01-01T12:00:00")
        arr = _to_datetime_array([dt64])
        assert isinstance(arr[0], datetime)

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2026-01-01T12:00:00")
        arr = _to_datetime_array([ts])
        assert isinstance(arr[0], datetime)

    def test_scalar(self):
        dt = datetime(2026, 6, 1)
        arr = _to_datetime_array(dt)
        assert arr[0] == dt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sar_grid(n_lon: int = 5, n_lat: int = 4):
    """Return a small synthetic SAR grid centred near (0°E, 52°N)."""
    lons = np.linspace(-1.0, 1.0, n_lon)
    lats = np.linspace(51.0, 53.0, n_lat)
    grid_lon, grid_lat = np.meshgrid(lons, lats)  # shape (n_lat, n_lon)

    # One time step
    wind_speed = np.full((1, n_lat, n_lon), fill_value=8.0)
    wind_dir   = np.full((1, n_lat, n_lon), fill_value=225.0)

    sar_time = np.array([datetime(2026, 1, 1, 12, 0, 0)], dtype=object)

    return grid_lon, grid_lat, sar_time, {"wind_speed": wind_speed, "wind_direction": wind_dir}


def _make_val_dataframe(lons, lats, times, **extra_cols):
    """Build a minimal validation DataFrame."""
    data = {"lon": lons, "lat": lats, "time": times}
    data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# PointLayerCollocation
# ---------------------------------------------------------------------------

class TestPointLayerCollocation:
    def test_finds_match_within_tolerance(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Validation point right on the grid centre
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 30, 0)],
            WSPD=[7.5], WDIR=[220.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "buoy")

        assert len(results) > 0
        r = results[0]
        assert r.val_source == "buoy"
        assert r.temporal_distance_minutes <= 60.0
        assert r.spatial_distance_km <= 200.0
        assert "wind_speed" in r.sar_data
        assert "WSPD" in r.val_data

    def test_no_match_outside_spatial_tolerance(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Validation point far from the grid
        val = _make_val_dataframe(
            lons=[30.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[5.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=10)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        assert results == []

    def test_no_match_outside_time_tolerance(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Validation point at the right place but 5 hours late
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 17, 0, 0)],
            WSPD=[8.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        assert results == []

    def test_multiple_matches_per_point(self):
        """A single validation point can match multiple SAR grid cells."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[8.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=500, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        # With a 500 km tolerance, the entire small grid should match
        assert len(results) > 1

    def test_collocated_point_fields(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[7.0], WDIR=[200.0],
            platform_id=["MO_001"],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")

        assert len(results) > 0
        r = results[0]
        assert r.val_id == "MO_001"
        assert r.val_source == "mooring"
        assert isinstance(r.sar_time, datetime)
        assert isinstance(r.val_time, datetime)

    def test_nan_sar_pixels_excluded(self):
        """Grid cells with NaN SAR values should not produce collocations."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Set the entire first (closest) time/row to NaN
        sar_data_nan = {k: v.copy() for k, v in sar_data.items()}
        for v in sar_data_nan.values():
            v[:] = np.nan

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[7.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data_nan, grid_lon, grid_lat, sar_time, val, "buoy")
        assert results == []

    def test_multiple_validation_points(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[-0.5, 0.0, 0.5],
            lats=[ 51.5, 52.0, 52.5],
            times=[
                datetime(2026, 1, 1, 12,  0, 0),
                datetime(2026, 1, 1, 12, 10, 0),
                datetime(2026, 1, 1, 12, 20, 0),
            ],
            WSPD=[6.0, 8.0, 10.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        assert len(results) > 0

    def test_collocated_point_has_collocation_type(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[8.0],
        )
        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        assert len(results) > 0
        assert results[0].collocation_type == "point_vs_layer"

    def test_invalid_interpolation_method(self):
        with pytest.raises(ValueError, match="interpolation_method"):
            PointLayerCollocation(interpolation_method="bilinear")

    def test_numpy_datetime64_input(self):
        """sar_time as datetime64 array should work correctly."""
        grid_lon, grid_lat, _, sar_data = _make_sar_grid()
        sar_time_64 = np.array(["2026-01-01T12:00:00"], dtype="datetime64[ns]")

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 15, 0)],
            WSPD=[8.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time_64, val, "buoy")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# TrajectoryLayerCollocation
# ---------------------------------------------------------------------------

class TestTrajectoryLayerCollocation:
    def test_finds_match_and_labels_correctly(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        # Simulate a short ferrybox track crossing the SAR scene
        val = _make_val_dataframe(
            lons=[-0.5, 0.0, 0.5],
            lats=[51.5, 52.0, 52.5],
            times=[
                datetime(2026, 1, 1, 11, 55, 0),
                datetime(2026, 1, 1, 12,  0, 0),
                datetime(2026, 1, 1, 12,  5, 0),
            ],
            EWCT=[0.3, 0.4, 0.5],
            NSCT=[0.1, 0.2, 0.1],
        )
        colloc = TrajectoryLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "ferrybox")

        assert len(results) > 0
        assert all(r.collocation_type == "trajectory_vs_layer" for r in results)
        assert all(r.val_source == "ferrybox" for r in results)

    def test_no_match_outside_tolerance(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        val = _make_val_dataframe(
            lons=[30.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            EWCT=[0.5],
        )
        colloc = TrajectoryLayerCollocation(spatial_tolerance_km=10, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "drifter")
        assert results == []

    def test_inherits_from_point_layer(self):
        assert issubclass(TrajectoryLayerCollocation, PointLayerCollocation)


# ---------------------------------------------------------------------------
# LayerLayerCollocation
# ---------------------------------------------------------------------------

class TestLayerLayerCollocation:
    def test_finds_match_and_labels_correctly(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        # Simulate a coarse scatterometer swath overlapping the SAR grid
        scat_lons = np.linspace(-1.5, 1.5, 8)
        scat_lats = np.linspace(50.5, 53.5, 6)
        mg_lon, mg_lat = np.meshgrid(scat_lons, scat_lats)
        val = _make_val_dataframe(
            lons=mg_lon.ravel().tolist(),
            lats=mg_lat.ravel().tolist(),
            times=[datetime(2026, 1, 1, 12, 0, 0)] * mg_lon.size,
            wind_speed=[8.5] * mg_lon.size,
            wind_dir=[230.0] * mg_lon.size,
        )
        colloc = LayerLayerCollocation(spatial_tolerance_km=100, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "scatterometer")

        assert len(results) > 0
        assert all(r.collocation_type == "layer_vs_layer" for r in results)
        assert all(r.val_source == "scatterometer" for r in results)

    def test_no_match_outside_time_tolerance(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 18, 0, 0)],  # 6 h after SAR
            wind_speed=[7.0],
        )
        colloc = LayerLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "scatterometer")
        assert results == []

    def test_inherits_from_point_layer(self):
        assert issubclass(LayerLayerCollocation, PointLayerCollocation)


# ---------------------------------------------------------------------------
# _detect_collocation_type
# ---------------------------------------------------------------------------

class TestDetectCollocationTypeImport:
    def test_point_by_default(self):
        import xarray as xr
        ds = xr.Dataset()
        assert _detect_collocation_type(ds, "validation/mooring") == "point_vs_layer"

    def test_trajectory_by_platform_type(self):
        import xarray as xr
        for pt in TRAJECTORY_PLATFORM_TYPES:
            ds = xr.Dataset(attrs={"platform_type": pt})
            assert _detect_collocation_type(ds, "val/x") == "trajectory_vs_layer", pt

    def test_layer_by_data_type(self):
        import xarray as xr
        for dt in LAYER_DATA_TYPES:
            ds = xr.Dataset(attrs={"data_type": dt})
            assert _detect_collocation_type(ds, "val/x") == "layer_vs_layer", dt

    def test_layer_by_path_fragment(self):
        import xarray as xr
        ds = xr.Dataset()
        assert _detect_collocation_type(ds, "validation/osi_saf_winds/file") == "layer_vs_layer"
        assert _detect_collocation_type(ds, "validation/scatterometer/file") == "layer_vs_layer"


# ---------------------------------------------------------------------------
# Former stub tests (now verify classes are functional)
# ---------------------------------------------------------------------------

class TestAllTypesWork:
    def test_all_three_classes_return_results(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            wind_speed=[8.0],
        )
        for cls, expected_type in (
            (PointLayerCollocation,      "point_vs_layer"),
            (TrajectoryLayerCollocation, "trajectory_vs_layer"),
            (LayerLayerCollocation,      "layer_vs_layer"),
        ):
            colloc = cls(spatial_tolerance_km=200, time_tolerance_minutes=60)
            results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "test")
            assert len(results) > 0
            assert results[0].collocation_type == expected_type
