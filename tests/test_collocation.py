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
# Stub collocation types
# ---------------------------------------------------------------------------

class TestStubs:
    def test_trajectory_raises(self):
        with pytest.raises(NotImplementedError):
            TrajectoryLayerCollocation().collocate()

    def test_layer_raises(self):
        with pytest.raises(NotImplementedError):
            LayerLayerCollocation().collocate()
