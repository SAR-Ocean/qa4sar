"""Tests for the collocation algorithms (step 3)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sar_validation.core.collocation import (
    LAYER_DATA_TYPES,
    LayerLayerCollocation,
    PointLayerCollocation,
    _detect_collocation_type,
    _equal_weights,
    _gaussian_weights,
    _haversine_distance,
    _inverse_distance_weights,
    _linear_weights,
    _project_currents_to_radial,
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

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
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

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        assert results == []

    def test_multiple_matches_per_point(self):
        """Aggregation produces one match per validation point (not per SAR cell)."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[8.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=500, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        # Aggregation produces one match per validation point, not multiple for each nearby SAR cell
        assert len(results) == 1
        assert "WSPD" in results[0].val_data

    def test_collocated_point_fields(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[7.0], WDIR=[200.0],
            platform_id=["MO_001"],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
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

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
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

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "mooring")
        assert len(results) > 0

    def test_collocated_point_has_collocation_type(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[8.0],
        )
        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
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

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time_64, val, "buoy")
        assert len(results) > 0

    def test_per_row_platform_type_overrides_val_source(self):
        """A combined in-situ CSV can mix platform types — val_source should
        reflect each observation's own platform_type, not the single
        val_source argument passed to collocate()."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[-0.5, 0.5], lats=[51.5, 52.5],
            times=[datetime(2026, 1, 1, 12, 0, 0), datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[7.0, 8.0],
            platform_type=["mooring", "buoy"],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=500, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "insitu")

        sources = {r.val_source for r in results}
        assert sources == {"mooring", "buoy"}
        assert "insitu" not in sources

    def test_missing_platform_type_falls_back_to_val_source(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            WSPD=[7.0],
        )

        colloc = PointLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "insitu")
        assert len(results) > 0
        assert results[0].val_source == "insitu"


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
        colloc = LayerLayerCollocation(spatial_tolerance_km=100, time_tolerance_minutes=60,
                                        aggregation_window_km=80)
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
        colloc = LayerLayerCollocation(spatial_tolerance_km=200, time_tolerance_minutes=60,
                                        aggregation_window_km=100)
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "scatterometer")
        assert results == []

    def test_inherits_from_point_layer(self):
        assert issubclass(LayerLayerCollocation, PointLayerCollocation)

    def test_layer_collocation_defaults(self):
        """Verify LayerLayerCollocation has scatterometer-optimized defaults."""
        colloc = LayerLayerCollocation()
        assert colloc.time_tolerance_minutes == 180  # Paper spec: ±3 hours
        assert colloc.spatial_tolerance_km == 12.5   # ASCAT cell size
        assert colloc.aggregation_window_km == 12.5  # ASCAT cell size
        assert colloc.distance_weighting == "equal"  # Uniform for regular grid
        assert colloc.validation_temporal_averaging_minutes == 60  # ±1 hour window

    def test_aggregates_sar_around_each_point(self):
        """Test that SAR values are aggregated around each scatterometer point.

        ASCAT/OSI-SAF data is already delivered one observation per
        12.5 km cell, so cell-averaging no longer re-clusters points —
        each scatterometer point is its own cell and gets its own
        SAR-aggregated match.
        """
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Create 2 scatterometer point groups: one overlapping SAR, one outside
        cell1_lons = [0.0, 0.1, -0.1]
        cell1_lats = [52.0, 52.1, 52.0]
        cell2_lons = [5.0, 5.1]  # Far outside SAR grid
        cell2_lats = [52.0, 52.1]
        
        val = _make_val_dataframe(
            lons=cell1_lons + cell2_lons,
            lats=cell1_lats + cell2_lats,
            times=[datetime(2026, 1, 1, 12, 0, 0)] * (len(cell1_lons) + len(cell2_lons)),
            wind_speed=[8.0] * (len(cell1_lons) + len(cell2_lons)),
        )
        
        colloc = LayerLayerCollocation(
            spatial_tolerance_km=500, 
            time_tolerance_minutes=60,
            aggregation_window_km=50
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "scatterometer")

        # One match per cell1 point (cell2 is outside SAR range); no merging.
        assert len(results) == len(cell1_lons)
        # All matches should have collocation_type = "layer_vs_layer"
        assert all(r.collocation_type == "layer_vs_layer" for r in results)
        # SAR and validation locations should be relatively close
        for r in results:
            assert r.spatial_distance_km <= colloc.aggregation_window_km + 5  # small tolerance

    def test_each_point_matched_independently(self):
        """Test that scatterometer points are matched independently, not
        merged/averaged together even when at the same location."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Create scatterometer points at the same location, different times
        base_time = datetime(2026, 1, 1, 12, 0, 0)
        times = [base_time + timedelta(minutes=i*10) for i in range(5)]
        
        val = _make_val_dataframe(
            lons=[0.0] * 5,
            lats=[52.0] * 5,
            times=times,
            wind_speed=[7.0, 8.0, 9.0, 8.5, 7.5],  # Varying values
        )
        
        colloc = LayerLayerCollocation(
            spatial_tolerance_km=500,
            time_tolerance_minutes=60,
            aggregation_window_km=50,
            validation_temporal_averaging_minutes=30  # ±30 min window
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "scatterometer")

        # One collocation per scatterometer point — no spatial merging or
        # temporal averaging across points, even at an identical location.
        assert len(results) == 5
        val_speeds = sorted(r.val_data["wind_speed"] for r in results)
        assert val_speeds == [7.0, 7.5, 8.0, 8.5, 9.0]

    def test_individual_method_handles_nan_grid_edges(self):
        """Real S1 OCN grids have NaN lon/lat at masked/edge cells; the
        'individual' method's spatial pre-filter must skip them with
        nan-aware min/max rather than let a single NaN blank out the whole
        bounding box (and therefore every match)."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        grid_lon[0, 0] = np.nan
        grid_lat[0, 0] = np.nan

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            wind_speed=[8.5],
        )
        colloc = LayerLayerCollocation(
            spatial_tolerance_km=200, time_tolerance_minutes=60,
            aggregation_window_km=100, method="individual",
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "scatterometer")

        assert len(results) > 0
        assert all(r.collocation_type == "layer_vs_layer" for r in results)


# ---------------------------------------------------------------------------
# _detect_collocation_type
# ---------------------------------------------------------------------------

class TestDetectCollocationTypeImport:
    def test_point_by_default(self):
        import xarray as xr
        ds = xr.Dataset()
        assert _detect_collocation_type(ds, "validation/mooring") == "point_vs_layer"

    def test_ferrybox_drifter_default_to_point(self):
        import xarray as xr
        for pt in ("ferrybox", "fb", "drifter", "ad"):
            ds = xr.Dataset(attrs={"platform_type": pt})
            assert _detect_collocation_type(ds, "val/x") == "point_vs_layer", pt

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


class TestHfRadarGridDispatch:
    def test_data_type_routes_to_layer(self):
        import xarray as xr
        ds = xr.Dataset(attrs={"data_type": "hf_radar_grid"})
        assert _detect_collocation_type(ds, "validation/hfr_noaa/scene") == "layer_vs_layer"

    def test_path_fallback_routes_to_layer(self):
        import xarray as xr
        ds = xr.Dataset()  # no data_type attr
        assert _detect_collocation_type(ds, "validation/hfr_noaa/scene") == "layer_vs_layer"

    def test_default_layer_spec_present(self):
        from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS
        spec = DEFAULT_LAYER_TYPE_SPECS["hf_radar_grid"]
        assert spec["aggregation_window_km"] == 6.0
        assert spec["time_tolerance_minutes"] == 20
        assert spec["distance_weighting"] == "equal"


# ---------------------------------------------------------------------------
# Former stub tests (now verify classes are functional)
# ---------------------------------------------------------------------------

class TestAllTypesWork:
    def test_both_classes_return_results(self):
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            wind_speed=[8.0],
        )
        for cls, expected_type in (
            (PointLayerCollocation, "point_vs_layer"),
            (LayerLayerCollocation, "layer_vs_layer"),
        ):
            colloc = cls(spatial_tolerance_km=200, time_tolerance_minutes=60, aggregation_window_km=100)
            results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "test")
            assert len(results) > 0
            assert results[0].collocation_type == expected_type


# ---------------------------------------------------------------------------
# Distance weighting functions (aggregation)
# ---------------------------------------------------------------------------

class TestDistanceWeightingFunctions:
    """Test distance weighting functions for SAR aggregation."""

    def test_gaussian_weights_normalize(self):
        """Gaussian weights should sum to 1.0."""
        distances = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        weights = _gaussian_weights(distances, sigma_km=2.0)
        assert len(weights) == len(distances)
        assert np.sum(weights) == pytest.approx(1.0)
        assert np.all(weights >= 0)

    def test_gaussian_weights_favor_close(self):
        """Gaussian weights should favor closer distances."""
        distances = np.array([0.0, 5.0])
        weights = _gaussian_weights(distances, sigma_km=2.0)
        assert weights[0] > weights[1]

    def test_inverse_distance_weights_normalize(self):
        """Inverse distance weights should sum to 1.0."""
        distances = np.array([0.1, 1.0, 2.0, 3.0, 5.0])
        weights = _inverse_distance_weights(distances, power=2.0)
        assert len(weights) == len(distances)
        assert np.sum(weights) == pytest.approx(1.0)
        assert np.all(weights >= 0)

    def test_inverse_distance_weights_favor_close(self):
        """Inverse distance weights should favor closer distances."""
        distances = np.array([0.5, 5.0])
        weights = _inverse_distance_weights(distances, power=2.0)
        assert weights[0] > weights[1]

    def test_linear_weights_normalize(self):
        """Linear weights should sum to 1.0."""
        distances = np.array([0.0, 2.0, 4.0, 6.0])
        weights = _linear_weights(distances, max_distance_km=10.0)
        assert len(weights) == len(distances)
        assert np.sum(weights) == pytest.approx(1.0)
        assert np.all(weights >= 0)

    def test_linear_weights_favor_close(self):
        """Linear weights should favor closer distances."""
        distances = np.array([0.0, 8.0])
        weights = _linear_weights(distances, max_distance_km=10.0)
        assert weights[0] > weights[1]

    def test_equal_weights_uniform(self):
        """Equal weights should be uniform."""
        distances = np.array([0.1, 2.0, 5.0, 10.0])
        weights = _equal_weights(distances)
        assert len(weights) == len(distances)
        assert np.sum(weights) == pytest.approx(1.0)
        assert np.allclose(weights, 0.25)

    def test_equal_weights_empty(self):
        """Equal weights should handle empty array."""
        distances = np.array([])
        weights = _equal_weights(distances)
        assert len(weights) == 0


# ---------------------------------------------------------------------------
# Aggregation-based PointLayerCollocation
# ---------------------------------------------------------------------------

class TestPointLayerCollocationAggregation:
    """Test aggregation-based collocation functionality."""

    def test_init_with_aggregation_params(self):
        """PointLayerCollocation should accept aggregation parameters."""
        colloc = PointLayerCollocation(
            aggregation_window_km=5.0,
            validation_temporal_averaging_minutes=30,
            distance_weighting="gaussian",
            gaussian_sigma_km=2.0,
        )
        assert colloc.aggregation_window_km == 5.0
        assert colloc.validation_temporal_averaging_minutes == 30
        assert colloc.distance_weighting == "gaussian"
        assert colloc.gaussian_sigma_km == 2.0

    def test_invalid_distance_weighting_raises_error(self):
        """Invalid weighting method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown distance_weighting"):
            PointLayerCollocation(distance_weighting="invalid_method")

    def test_aggregation_single_match(self):
        """Aggregation should produce single match per validation point."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Validation point in the middle of the grid
        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            wind_speed=[8.5],
        )

        colloc = PointLayerCollocation(
            spatial_tolerance_km=200,
            time_tolerance_minutes=60,
            aggregation_window_km=50.0,  # Large enough to include nearby cells
            distance_weighting="gaussian",
            gaussian_sigma_km=2.0,
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "test")

        # Should produce exactly one match (not multiple for each nearby cell)
        assert len(results) == 1
        assert results[0].collocation_type == "point_vs_layer"
        assert "wind_speed" in results[0].sar_data
        assert "wind_speed" in results[0].val_data

    def test_no_temporal_averaging_uses_raw_value(self):
        """Each validation observation should keep its own raw value (no averaging)."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        # Multiple validation observations at same location, different times
        val = _make_val_dataframe(
            lons=[0.0, 0.0, 0.0],
            lats=[52.0, 52.0, 52.0],
            times=[
                datetime(2026, 1, 1, 11, 45, 0),  # 15 min before SAR time
                datetime(2026, 1, 1, 12, 0, 0),   # SAR time
                datetime(2026, 1, 1, 12, 15, 0),  # 15 min after SAR time
            ],
            wind_speed=[7.0, 8.0, 9.0],
        )

        colloc = PointLayerCollocation(
            spatial_tolerance_km=200,
            time_tolerance_minutes=60,
            aggregation_window_km=50.0,  # Large enough to include nearby cells
            distance_weighting="gaussian",
            gaussian_sigma_km=2.0,
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "test")

        # Each validation observation produces a match with its own raw value,
        # not averaged with neighboring observations.
        assert len(results) == 3
        val_speeds = sorted(result.val_data["wind_speed"] for result in results)
        assert val_speeds == pytest.approx([7.0, 8.0, 9.0])

    def test_aggregation_different_weighting_methods(self):
        """Aggregation should work with all weighting methods."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0], lats=[52.0],
            times=[datetime(2026, 1, 1, 12, 0, 0)],
            wind_speed=[8.5],
        )

        for method in ["gaussian", "inverse_distance", "linear", "equal"]:
            colloc = PointLayerCollocation(
                spatial_tolerance_km=200,
                time_tolerance_minutes=60,
                aggregation_window_km=50.0,  # Large enough to include nearby cells
                distance_weighting=method,
                gaussian_sigma_km=2.0,
            )
            results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "test")
            assert len(results) == 1, f"Failed for weighting method '{method}'"
            assert "wind_speed" in results[0].sar_data


class TestWvRvlProjection:
    def test_projection_added_from_ewct_nsct(self):
        from sar_validation.core.collocation import _collocate_wv_points

        # One WV RVL point with a known heading; one in-situ current obs on top.
        sar_lons = np.array([-19.5])
        sar_lats = np.array([50.5])
        sar_times = np.array([np.datetime64("2026-06-20T19:15:00", "ns")])
        sar_point_vars = {
            "rvlRadVel": np.array([1.0]),
            "rvlHeading": np.array([90.0]),  # heading_rad = radians(90-90)=0
        }
        val = pd.DataFrame({
            "lon": [-19.5],
            "lat": [50.5],
            "time": [pd.Timestamp("2026-06-20T19:20:00")],
            "EWCT": [0.4],
            "NSCT": [0.3],
        })
        matches = _collocate_wv_points(
            sar_lons=sar_lons, sar_lats=sar_lats, sar_times=sar_times,
            sar_point_vars=sar_point_vars, val_data=val, val_source="mooring",
            footprint_radius_km=14.0, time_tolerance_minutes=30,
            distance_weighting="equal", gaussian_sigma_km=5.0,
            collocation_type="point_vs_point",
        )
        assert len(matches) == 1
        proj = matches[0].val_data["rvlRadVel_projection"]
        # heading 90 -> heading_rad 0 -> projection = EWCT*cos0 + NSCT*sin0 = EWCT
        assert proj == pytest.approx(0.4, abs=1e-6)

    def test_radvel_std_propagates(self):
        from sar_validation.core.collocation import _collocate_wv_points

        sar_lons = np.array([-19.5])
        sar_lats = np.array([50.5])
        sar_times = np.array([np.datetime64("2026-06-20T19:15:00", "ns")])
        sar_point_vars = {
            "rvlRadVel": np.array([1.0]),
            "rvlRadVelStd": np.array([0.15]),
            "rvlHeading": np.array([90.0]),
        }
        val = pd.DataFrame({
            "lon": [-19.5],
            "lat": [50.5],
            "time": [pd.Timestamp("2026-06-20T19:20:00")],
            "EWCT": [0.4],
            "NSCT": [0.3],
        })
        matches = _collocate_wv_points(
            sar_lons=sar_lons, sar_lats=sar_lats, sar_times=sar_times,
            sar_point_vars=sar_point_vars, val_data=val, val_source="mooring",
            footprint_radius_km=14.0, time_tolerance_minutes=30,
            distance_weighting="equal", gaussian_sigma_km=5.0,
            collocation_type="point_vs_point",
        )
        assert len(matches) == 1
        assert matches[0].sar_data["rvlRadVelStd"] == pytest.approx(0.15, abs=1e-6)


class TestRunCollocationCurrentsFromDatatree:
    def _currents_recipe(self):
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        return Recipe(RecipeConfig(
            name="currents_it",
            variable="currents",
            geographic_bounds=GeographicBounds(-21.0, -18.0, 49.0, 52.0),
            temporal_bounds=TemporalBounds("2026-06-20T18:00:00", "2026-06-20T23:00:00"),
        ))

    def test_no_load_rvl_symbol(self):
        import sar_validation.core.collocation as coll
        assert not hasattr(coll, "_load_rvl_for_collocation")

    def test_grid_rvl_projects_against_insitu(self, tmp_path):
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        # SAR RVL grid node (y, x) with a constant heading of 90 deg.
        # Grid spacing chosen so a node coincides exactly with the in-situ
        # point below (within the default 5 km point_vs_layer aggregation
        # window; a 4x5 grid over this bbox has ~18 km spacing, which is too
        # coarse to guarantee a match).
        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        sar = xr.Dataset(
            {
                "rvlRadVel": (("y", "x"), np.full((ny, nx), 0.5, dtype="float32")),
                "rvlHeading": (("y", "x"), np.full((ny, nx), 90.0, dtype="float32")),
                "rvlIncidenceAngle": (("y", "x"), np.full((ny, nx), 30.0, dtype="float32")),
            },
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T19:15:00", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn", "swath_mode": "IW/EW/SM",
                   "measurement_type": "rvl"},
        )
        # In-situ mooring node with EWCT/NSCT at a SAR cell location + time.
        val = xr.Dataset(
            {
                "EWCT": (("point",), np.array([0.4], dtype="float32")),
                "NSCT": (("point",), np.array([0.3], dtype="float32")),
            },
            coords={
                "lon": (("point",), np.array([-19.5])),
                "lat": (("point",), np.array([50.5])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T19:20:00", "ns")])),
                "platform_type": (("point",), np.array(["mooring"])),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "mooring"},
        )
        tree = xr.DataTree.from_dict({"/sar/scene1": sar, "/validation/mooring1": val})

        result = run_collocation(self._currents_recipe(), tree, tmp_path)
        assert result is not None
        assert "sar_rvlRadVel" in result
        assert "val_rvlRadVel_projection" in result
        # heading 90 -> projection == EWCT == 0.4
        assert float(result["val_rvlRadVel_projection"].values[0]) == pytest.approx(0.4, abs=1e-5)

    def test_grid_rvl_radvel_std_propagates(self, tmp_path):
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        sar = xr.Dataset(
            {
                "rvlRadVel": (("y", "x"), np.full((ny, nx), 0.5, dtype="float32")),
                "rvlRadVelStd": (("y", "x"), np.full((ny, nx), 0.12, dtype="float32")),
                "rvlHeading": (("y", "x"), np.full((ny, nx), 90.0, dtype="float32")),
                "rvlIncidenceAngle": (("y", "x"), np.full((ny, nx), 30.0, dtype="float32")),
            },
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T19:15:00", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn", "swath_mode": "IW/EW/SM",
                   "measurement_type": "rvl"},
        )
        val = xr.Dataset(
            {
                "EWCT": (("point",), np.array([0.4], dtype="float32")),
                "NSCT": (("point",), np.array([0.3], dtype="float32")),
            },
            coords={
                "lon": (("point",), np.array([-19.5])),
                "lat": (("point",), np.array([50.5])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T19:20:00", "ns")])),
                "platform_type": (("point",), np.array(["mooring"])),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "mooring"},
        )
        tree = xr.DataTree.from_dict({"/sar/scene1": sar, "/validation/mooring1": val})

        result = run_collocation(self._currents_recipe(), tree, tmp_path)
        assert result is not None
        assert "sar_rvlRadVelStd" in result
        assert float(result["sar_rvlRadVelStd"].values[0]) == pytest.approx(0.12, abs=1e-5)


class TestProjectCurrentsToRadial:
    def test_due_east_current_zero_heading(self):
        # heading 0° → LOS is heading-90° = -90°; cos(-90)=0, sin(-90)=-1.
        # A 1 m/s eastward current projects to 0; northward projects to -1.
        assert _project_currents_to_radial(1.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
        assert _project_currents_to_radial(0.0, 1.0, 0.0) == pytest.approx(-1.0, abs=1e-12)

    def test_heading_90_east_projects_fully(self):
        # heading 90° → θ=0 → cos=1, sin=0: eastward projects fully, north to 0.
        assert _project_currents_to_radial(1.0, 0.0, 90.0) == pytest.approx(1.0, abs=1e-12)
        assert _project_currents_to_radial(0.0, 1.0, 90.0) == pytest.approx(0.0, abs=1e-12)

    def test_matches_legacy_inline_formula(self):
        ewct, nsct, heading = 0.37, -0.12, 190.0
        expected = (ewct * np.cos(np.radians(heading - 90.0))
                    + nsct * np.sin(np.radians(heading - 90.0)))
        assert _project_currents_to_radial(ewct, nsct, heading) == pytest.approx(expected)
