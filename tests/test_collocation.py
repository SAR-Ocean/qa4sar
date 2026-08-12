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
    _normalize_weights,
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


def test_normalize_weights_sums_to_one():
    weights = _normalize_weights(np.array([1.0, 2.0, 3.0]))
    assert weights.sum() == pytest.approx(1.0)


def test_normalize_weights_zero_sum_returns_unchanged():
    weights = _normalize_weights(np.array([0.0, 0.0]))
    assert list(weights) == [0.0, 0.0]


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

    @pytest.mark.parametrize(
        "spatial_tolerance_km,val_kwargs,expected_len,expected_check",
        [
            pytest.param(
                200,
                dict(
                    lons=[0.0] * 5, lats=[52.0] * 5,
                    times=[datetime(2026, 1, 1, h, 0, 0) for h in (9, 10, 11, 14, 15)],
                    WSPD=[float(h) for h in (9, 10, 11, 14, 15)],
                    platform_id=["StationA"] * 5,
                ),
                1,
                lambda results: (
                    results[0].val_data["WSPD"] == 11.0
                    and results[0].temporal_distance_minutes == 60.0
                ),
                id="keeps_only_nearest_reading_per_station",
            ),
            pytest.param(
                200,
                dict(
                    lons=[0.0, 0.0], lats=[52.0, 52.0],
                    times=[datetime(2026, 1, 1, 12, 0, 0)] * 2,
                    WSPD=[7.0, 9.0],
                    platform_id=["SensorA", "SensorB"],
                ),
                2,
                lambda results: {r.val_id for r in results} == {"SensorA", "SensorB"},
                id="keeps_distinct_stations_separate",
            ),
            pytest.param(
                500,
                dict(
                    lons=[0.0, 0.0, 0.5], lats=[52.0, 52.0, 52.5],
                    times=[
                        datetime(2026, 1, 1, 10, 0, 0),
                        datetime(2026, 1, 1, 12, 0, 0),
                        datetime(2026, 1, 1, 12, 0, 0),
                    ],
                    WSPD=[6.0, 8.0, 10.0],
                ),
                2,
                lambda results: {r.val_data["WSPD"] for r in results} == {8.0, 10.0},
                id="dedup_falls_back_to_lonlat_without_platform_id",
            ),
        ],
    )
    def test_wide_tolerance_dedup(
        self, spatial_tolerance_km, val_kwargs, expected_len, expected_check,
    ):
        """A station reporting hourly, matched against one SAR time with a
        wide time tolerance (e.g. ISMN's 720 min), must contribute only its
        single closest-in-time reading — not one collocation per hourly
        reading in the window. Regression coverage for the real-data bug
        where one ISMN station produced ~25 collocations against a single
        SAR overpass instead of 1, plus the two related guards: distinct
        stations must not be collapsed together, and dedup falls back to
        (lon, lat) grouping when no platform_id column is present."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(**val_kwargs)

        colloc = PointLayerCollocation(
            spatial_tolerance_km=spatial_tolerance_km, time_tolerance_minutes=720,
            aggregation_window_km=100, dedup_nearest_in_time=True,
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "ismn")

        assert len(results) == expected_len
        assert expected_check(results)


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

    def test_dedup_nearest_in_time_keeps_only_nearest_reading_per_cell(self):
        """A gridded HF-radar cell reporting hourly, with two candidate
        readings inside a wide time-tolerance window, must contribute only
        its single closest-in-time reading -- not one collocation per
        candidate hour. Regression guard for the Finnmark fix, where
        widening hf_radar_grid's time_tolerance_minutes to guarantee hourly
        coverage would otherwise start producing duplicate matches per
        cell."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0, 0.0], lats=[52.0, 52.0],
            times=[datetime(2026, 1, 1, 11, 45, 0), datetime(2026, 1, 1, 12, 20, 0)],
            EWCT=[1.0, 2.0], NSCT=[0.5, 0.6],
        )

        colloc = LayerLayerCollocation(
            spatial_tolerance_km=200, time_tolerance_minutes=30,
            aggregation_window_km=100, dedup_nearest_in_time=True,
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "hf_radar_grid")

        assert len(results) == 1
        assert results[0].val_data["EWCT"] == 1.0
        assert results[0].temporal_distance_minutes == 15.0

    def test_dedup_nearest_in_time_false_keeps_every_reading(self):
        """Default behaviour (dedup off) must be unchanged: both candidate
        readings produce their own collocation."""
        grid_lon, grid_lat, sar_time, sar_data = _make_sar_grid()

        val = _make_val_dataframe(
            lons=[0.0, 0.0], lats=[52.0, 52.0],
            times=[datetime(2026, 1, 1, 11, 45, 0), datetime(2026, 1, 1, 12, 20, 0)],
            EWCT=[1.0, 2.0], NSCT=[0.5, 0.6],
        )

        colloc = LayerLayerCollocation(
            spatial_tolerance_km=200, time_tolerance_minutes=30,
            aggregation_window_km=100,
        )
        results = colloc.collocate(sar_data, grid_lon, grid_lat, sar_time, val, "hf_radar_grid")

        assert len(results) == 2

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


class TestResolveLayerTypeScatterometerVariants:
    @pytest.mark.parametrize(
        "path,expected",
        [
            pytest.param(
                "validation/scatterometer_hy2b/some_file", "scatterometer_hy2b",
                id="hy2b_path_resolves_to_its_own_spec_key",
            ),
            pytest.param(
                "validation/scatterometer_oceansat3/some_file", "scatterometer_oceansat3",
                id="oceansat3_path_resolves_to_its_own_spec_key",
            ),
            pytest.param(
                "validation/scatterometer/some_file", "scatterometer",
                id="plain_ascat_path_still_resolves_to_bare_scatterometer",
            ),
        ],
    )
    def test_scatterometer_variant_path_resolution(self, path, expected):
        """Regression guard: the refinement must not over-match ASCAT nodes."""
        import xarray as xr

        from sar_validation.core.collocation import _resolve_layer_type
        from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

        ds = xr.Dataset(attrs={"data_type": "scatterometer"})
        layer_type = _resolve_layer_type(ds, path, DEFAULT_LAYER_TYPE_SPECS)
        assert layer_type == expected

    def test_altimeter_frequency_refinement_still_works(self):
        """Regression guard: extracting the helper must preserve existing behavior."""
        import xarray as xr

        from sar_validation.core.collocation import _resolve_layer_type
        from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

        ds = xr.Dataset(attrs={"data_type": "altimeter", "frequency": "5hz"})
        layer_type = _resolve_layer_type(
            ds, "validation/altimeter/Cryosat-2", DEFAULT_LAYER_TYPE_SPECS
        )
        assert layer_type == "altimeter_5hz"

    def test_resolve_layer_type_refines_radiometer_ssm_by_sensor(self):
        import xarray as xr

        from sar_validation.core.collocation import _resolve_layer_type
        from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

        ds_amsr = xr.Dataset(attrs={"data_type": "radiometer_ssm", "sensor": "amsr"})
        ds_smap = xr.Dataset(attrs={"data_type": "radiometer_ssm", "sensor": "smap"})

        assert _resolve_layer_type(ds_amsr, "validation/amsr_ssm/f1", DEFAULT_LAYER_TYPE_SPECS) == "amsr_ssm"
        assert _resolve_layer_type(ds_smap, "validation/smap_ssm/f1", DEFAULT_LAYER_TYPE_SPECS) == "smap_ssm"

    def test_resolve_layer_type_refines_radiometer_ssm_for_smos(self):
        import xarray as xr

        from sar_validation.core.collocation import _resolve_layer_type
        from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

        ds_smos = xr.Dataset(attrs={"data_type": "radiometer_ssm", "sensor": "smos"})
        assert _resolve_layer_type(ds_smos, "validation/smos_ssm/f1", DEFAULT_LAYER_TYPE_SPECS) == "smos_ssm"


class TestApplyHfRadarResolutionOverride:
    @pytest.mark.parametrize(
        "layer_type,val_ds_attrs,merged_kwargs,recipe_layer_type_specs,expected_aggregation_window_km",
        [
            pytest.param(
                "hf_radar_grid",
                {"hfr_resolution_km": 1.2},
                {"aggregation_window_km": 6.0, "time_tolerance_minutes": 30},
                {},
                1.2,
                id="overrides_when_no_recipe_override_and_attr_present",
            ),
            pytest.param(
                "hf_radar_grid",
                {"hfr_resolution_km": 1.2},
                {"aggregation_window_km": 6.0},
                {"hf_radar_grid": {"aggregation_window_km": 10.0}},
                6.0,  # unchanged -- recipe pinned it
                id="recipe_explicit_override_wins",
            ),
            pytest.param(
                "hf_radar_grid",
                {"hfr_resolution_km": 2.5},
                {"aggregation_window_km": 6.0},
                {"hf_radar_grid": {"time_tolerance_minutes": 45}},  # no aggregation_window_km
                2.5,
                id="recipe_partial_override_without_aggregation_window_km_still_derives",
            ),
            pytest.param(
                "hf_radar_grid",
                {},
                {"aggregation_window_km": 6.0},
                {},
                6.0,
                id="missing_attr_leaves_merged_kwargs_untouched",
            ),
            pytest.param(
                "scatterometer",
                {"hfr_resolution_km": 1.2},
                {"aggregation_window_km": 12.5},
                {},
                12.5,
                id="non_hf_radar_grid_layer_type_untouched",
            ),
        ],
    )
    def test_apply_hf_radar_resolution_override(
        self, layer_type, val_ds_attrs, merged_kwargs, recipe_layer_type_specs,
        expected_aggregation_window_km,
    ):
        import xarray as xr

        from sar_validation.core.collocation import _apply_hf_radar_resolution_override

        val_ds = xr.Dataset(attrs=val_ds_attrs)
        _apply_hf_radar_resolution_override(
            layer_type, val_ds, merged_kwargs, recipe_layer_type_specs,
        )
        assert merged_kwargs["aggregation_window_km"] == expected_aggregation_window_km


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

    @pytest.mark.parametrize(
        "weight_fn,distances,kwargs",
        [
            pytest.param(
                _gaussian_weights, np.array([0.0, 1.0, 2.0, 3.0, 5.0]), {"sigma_km": 2.0},
                id="gaussian",
            ),
            pytest.param(
                _inverse_distance_weights, np.array([0.1, 1.0, 2.0, 3.0, 5.0]), {"power": 2.0},
                id="inverse_distance",
            ),
            pytest.param(
                _linear_weights, np.array([0.0, 2.0, 4.0, 6.0]), {"max_distance_km": 10.0},
                id="linear",
            ),
        ],
    )
    def test_weights_normalize(self, weight_fn, distances, kwargs):
        """Weights should sum to 1.0 and be non-negative."""
        weights = weight_fn(distances, **kwargs)
        assert len(weights) == len(distances)
        assert np.sum(weights) == pytest.approx(1.0)
        assert np.all(weights >= 0)

    @pytest.mark.parametrize(
        "weight_fn,distances,kwargs",
        [
            pytest.param(_gaussian_weights, np.array([0.0, 5.0]), {"sigma_km": 2.0}, id="gaussian"),
            pytest.param(
                _inverse_distance_weights, np.array([0.5, 5.0]), {"power": 2.0}, id="inverse_distance",
            ),
            pytest.param(
                _linear_weights, np.array([0.0, 8.0]), {"max_distance_km": 10.0}, id="linear",
            ),
        ],
    )
    def test_weights_favor_close(self, weight_fn, distances, kwargs):
        """Weights should favor closer distances."""
        weights = weight_fn(distances, **kwargs)
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
    def test_projection_and_radvel_std_from_ewct_nsct(self):
        from sar_validation.core.collocation import _collocate_wv_points

        sar_lons = np.array([-19.5])
        sar_lats = np.array([50.5])
        sar_times = np.array([np.datetime64("2026-06-20T19:15:00", "ns")])
        sar_point_vars = {
            "rvlRadVel": np.array([1.0]),
            "rvlRadVelStd": np.array([0.15]),
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

    def test_grid_rvl_projects_against_insitu_and_radvel_std_propagates(self, tmp_path):
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
        assert "sar_rvlRadVelStd" in result
        assert float(result["sar_rvlRadVelStd"].values[0]) == pytest.approx(0.12, abs=1e-5)

    def test_wv_mode_multi_point_time_array_does_not_crash(self, tmp_path):
        """Regression test: a WV-mode SAR node's ``time`` coordinate is
        ``("point",)``-dimensioned -- one timestamp per vignette, not a
        single scalar like grid-mode (IW/EW/SM) scenes. `run_collocation`
        builds ``sar_scene_times`` from every SAR node up front (for the
        ISMN pre-averaging step), and used to call
        ``pd.Timestamp(ds["time"].values)`` directly on that array, which
        raises ``TypeError: Cannot convert input [...] of type
        numpy.ndarray to Timestamp``. This crashed on ANY recipe with a
        WV-mode scene, including this one, which has no ISMN/soil-moisture
        source at all -- confirming the crash was in the shared
        SAR-time-extraction code, not the ISMN branch itself."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        # Two WV vignettes at different acquisition times -- this is what
        # made the old scalar pd.Timestamp(...) call raise.
        sar = xr.Dataset(
            {
                "rvlRadVel": (("point",), np.array([1.0, -0.5], dtype="float32")),
                "rvlHeading": (("point",), np.array([90.0, 90.0], dtype="float32")),
            },
            coords={
                "lon": (("point",), np.array([-19.5, -18.7])),
                "lat": (("point",), np.array([50.5, 51.2])),
                "time": (("point",), np.array([
                    np.datetime64("2026-06-20T19:15:00", "ns"),
                    np.datetime64("2026-06-20T19:16:40", "ns"),
                ])),
            },
            attrs={"data_type": "sar_l2_ocn", "swath_mode": "WV",
                   "measurement_type": "rvl"},
        )
        # A non-ISMN (mooring) in-situ source sitting on top of the first
        # vignette only.
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
        assert result.sizes.get("collocation", 0) == 1
        assert "val_rvlRadVel_projection" in result
        assert float(result["val_rvlRadVel_projection"].values[0]) == pytest.approx(0.4, abs=1e-5)

    def test_sar_node_without_time_coord_does_not_crash(self, tmp_path):
        """A SAR node with no `time` coordinate at all must be skipped
        gracefully (as the per-scene loop below already does), not crash
        the earlier sar_scene_times construction that runs before that
        loop ever gets a chance to skip it."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        sar_no_time = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((3, 3), 1.0, dtype="float32"))},
            coords={
                "lon": (("y", "x"), np.linspace(-20.0, -19.0, 9).reshape(3, 3)),
                "lat": (("y", "x"), np.linspace(50.0, 51.0, 9).reshape(3, 3)),
            },
            attrs={"data_type": "sar_l2_ocn"},
        )
        mooring = xr.Dataset(
            {"EWCT": (("point",), np.array([0.4], dtype="float32")),
             "NSCT": (("point",), np.array([0.3], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([-19.5])),
                "lat": (("point",), np.array([50.5])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T12:00:00", "ns")])),
                "platform_type": (("point",), np.array(["mooring"])),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "mooring"},
        )
        tree = xr.DataTree.from_dict({
            "/sar/scene_no_time": sar_no_time,
            "/validation/mooring1": mooring,
        })

        result = run_collocation(self._currents_recipe(), tree, tmp_path)

        assert result is None

    def test_sar_node_with_nat_time_does_not_crash(self, tmp_path):
        """A SAR node whose acquisition time is NaT (a real, documented
        fallback elsewhere in datatree_converter.py) must not crash
        sar_scene_times's construction or the merge_asof call inside
        _average_within_sar_tolerance.

        Note: reproducing the actual observed crash requires a validation
        source with ``platform_type == "ismn"`` (the brief's originally
        proposed test used a plain "mooring" source, which never reaches
        ``_average_within_sar_tolerance`` -- only the point_vs_layer/"ismn"
        and layer_vs_layer/SSM-satellite branches call it -- so that
        version passed vacuously even against the unfixed code). Against
        unfixed code this raises ``ValueError: Merge keys contain null
        values on right side`` from inside ``pd.merge_asof``, exactly as
        the brief predicted, once the source is relabelled "ismn" so the
        averaging branch is actually exercised."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        sar_nat = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((3, 3), 1.0, dtype="float32"))},
            coords={
                "lon": (("y", "x"), np.linspace(-20.0, -19.0, 9).reshape(3, 3)),
                "lat": (("y", "x"), np.linspace(50.0, 51.0, 9).reshape(3, 3)),
                "time": np.datetime64("NaT", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn"},
        )
        sar_valid = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((3, 3), 1.0, dtype="float32"))},
            coords={
                "lon": (("y", "x"), np.linspace(-20.0, -19.0, 9).reshape(3, 3)),
                "lat": (("y", "x"), np.linspace(50.0, 51.0, 9).reshape(3, 3)),
                "time": np.datetime64("2026-06-20T12:00:00", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn"},
        )
        ismn_station = xr.Dataset(
            {"EWCT": (("point",), np.array([0.4], dtype="float32")),
             "NSCT": (("point",), np.array([0.3], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([-19.5])),
                "lat": (("point",), np.array([50.5])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T12:00:00", "ns")])),
                "platform_type": (("point",), np.array(["ismn"])),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "ismn"},
        )
        tree = xr.DataTree.from_dict({
            "/sar/scene_nat": sar_nat,
            "/sar/scene_valid": sar_valid,
            "/validation/station1": ismn_station,
        })

        result = run_collocation(self._currents_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 1


class TestGridTreeReuseAcrossValidationSources:
    """Regression test for a real performance bug: `run_collocation`'s
    grid-based (IW/EW) path used to rebuild `PointLayerCollocation`'s KD-tree
    once per validation *file*, not once per SAR scene. That's fine for a
    handful of large, pre-batched validation sources, but pathological for
    many small per-source files (e.g. ISMN's one-CSV-per-station output)
    matched against a large grid (e.g. CLMS Surface Soil Moisture's
    ~28M-cell 1 km Europe grid): a real recipe run with 118 ISMN stations
    took minutes rebuilding the same tree 118 times per SAR scene before
    this fix. See collocation.py's `grid_tree` parameter/build-once comment.
    """

    def _currents_recipe(self):
        from sar_validation.core.recipe import (
            CollocationType,
            GeographicBounds,
            PointVsLayerCollocation,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        return Recipe(RecipeConfig(
            name="grid_tree_reuse_it",
            variable="currents",
            geographic_bounds=GeographicBounds(-21.0, -18.0, 49.0, 52.0),
            temporal_bounds=TemporalBounds("2026-06-20T18:00:00", "2026-06-20T23:00:00"),
            # Wide enough to catch all three mooring points below, which
            # are deliberately offset from the nearest grid node by ~11 km
            # (more than the 5 km default) so each exercises a distinct
            # KD-tree neighbourhood query rather than all landing on the
            # same grid cell.
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(aggregation_window_km=20.0),
            ),
        ))

    def test_build_grid_tree_called_once_per_sar_scene_not_per_validation_source(self, tmp_path):
        from unittest.mock import patch

        import xarray as xr

        from sar_validation.core.collocation import PointLayerCollocation, run_collocation

        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        sar = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((ny, nx), 0.5, dtype="float32"))},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T19:15:00", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn", "swath_mode": "IW/EW/SM",
                   "measurement_type": "rvl"},
        )

        def _mooring(lon, lat):
            return xr.Dataset(
                {
                    "EWCT": (("point",), np.array([0.4], dtype="float32")),
                    "NSCT": (("point",), np.array([0.3], dtype="float32")),
                },
                coords={
                    "lon": (("point",), np.array([lon])),
                    "lat": (("point",), np.array([lat])),
                    "time": (("point",), np.array([np.datetime64("2026-06-20T19:20:00", "ns")])),
                    "platform_type": (("point",), np.array(["mooring"])),
                },
                attrs={"data_type": "insitu_observations", "platform_type": "mooring"},
            )

        # Three separate validation nodes (as ISMNDownloader's one-CSV-per-
        # station output produces) matched against the same single SAR scene.
        tree = xr.DataTree.from_dict({
            "/sar/scene1": sar,
            "/validation/mooring1": _mooring(-19.5, 50.5),
            "/validation/mooring2": _mooring(-19.6, 50.4),
            "/validation/mooring3": _mooring(-19.4, 50.6),
        })

        with patch.object(
            PointLayerCollocation, "_build_grid_tree",
            wraps=PointLayerCollocation._build_grid_tree,
        ) as mock_build:
            result = run_collocation(self._currents_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 3
        assert mock_build.call_count == 1


class TestRunCollocationIsmnTemporalAveraging:
    """ISMN reports hourly -- far more densely than one daily SAR
    overpass -- so every hourly reading within the wide
    time_tolerance_minutes (720 min, needed to tolerate ISMN's own
    reporting gaps) must collapse into a single station-day value.
    Previously this collapsed via "keep only the nearest reading", which
    is a real bias: since the SAR scene is always stamped at midnight, the
    nearest reading is always a nighttime one. run_collocation must
    instead average every in-tolerance reading -- only for platform_type
    == "ismn" sources; other point_vs_layer sources (moorings here) keep
    every in-tolerance reading, per their own pre-existing tested
    behaviour."""

    def _soil_moisture_recipe(self, time_tolerance_minutes: int = 720):
        from sar_validation.core.recipe import (
            CollocationType,
            GeographicBounds,
            PointVsLayerCollocation,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        return Recipe(RecipeConfig(
            name="soil_moisture_it",
            variable="soil_moisture",
            geographic_bounds=GeographicBounds(-21.0, -18.0, 49.0, 52.0),
            temporal_bounds=TemporalBounds("2026-06-20T00:00:00", "2026-06-20T23:59:00"),
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(
                    time_tolerance_minutes=time_tolerance_minutes,
                    aggregation_window_km=20.0,
                ),
            ),
        ))

    def _sar_grid(self):
        import xarray as xr

        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        return xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((ny, nx), 30.0, dtype="float32"))},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T12:00:00", "ns"),
            },
            attrs={"data_type": "sar_l3_ssm"},
        )

    def _ismn_station(self, lon, lat, n_hourly_readings=10):
        import xarray as xr

        times = np.array(
            [np.datetime64("2026-06-20T00:00:00", "ns") + np.timedelta64(h, "h")
             for h in range(n_hourly_readings)]
        )
        return xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.linspace(0.1, 0.2, n_hourly_readings, dtype="float32"))},
            coords={
                "lon": (("point",), np.full(n_hourly_readings, lon)),
                "lat": (("point",), np.full(n_hourly_readings, lat)),
                "time": (("point",), times),
                "platform_type": (("point",), np.array(["ismn"] * n_hourly_readings)),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "ismn"},
        )

    def test_ismn_station_collapses_to_one_averaged_match_per_scene(self, tmp_path):
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/ismn1": self._ismn_station(-19.5, 50.5, n_hourly_readings=24),
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 1
        # linspace(0.1, 0.2, 24) mean -- confirms the surviving value is
        # the true average of all 24 hourly readings, not just the single
        # nearest-to-scene-time (midnight) reading picked out of them.
        expected_mean = float(np.linspace(0.1, 0.2, 24, dtype="float32").mean())
        assert float(result["val_SOIL_MOISTURE"].values[0]) == pytest.approx(
            expected_mean, abs=1e-4
        )

    def test_mooring_source_unaffected_keeps_every_reading(self, tmp_path):
        """Same wide tolerance, same repeated-readings-at-one-location
        shape, but platform_type='mooring' instead of 'ismn' — must NOT be
        averaged/deduped (matches TestPointLayerCollocation's existing
        test_no_temporal_averaging_uses_raw_value expectation)."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        mooring = xr.Dataset(
            {"EWCT": (("point",), np.linspace(0.1, 0.2, 24, dtype="float32"))},
            coords={
                "lon": (("point",), np.full(24, -19.5)),
                "lat": (("point",), np.full(24, 50.5)),
                "time": (("point",), np.array(
                    [np.datetime64("2026-06-20T00:00:00", "ns") + np.timedelta64(h, "h")
                     for h in range(24)]
                )),
                "platform_type": (("point",), np.array(["mooring"] * 24)),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "mooring"},
        )
        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/mooring1": mooring,
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 24


class TestRunCollocationSatelliteSsmTemporalAveraging:
    """ASCAT/AMSR2/SMAP/SMOS soil-moisture sources can have both an
    ascending and descending overpass of the same grid cell within the
    ±12h tolerance around a midnight-stamped S1 SSM scene. run_collocation
    must blend them into a single day value per cell -- both to remove
    any day/night imbalance (same treatment as ISMN, see
    TestRunCollocationIsmnTemporalAveraging) and to shrink the point count
    for these dense gridded sources. See docs/design-choices.md §8.7."""

    def _sar_grid(self):
        import xarray as xr

        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(9.0, 11.0, nx), np.linspace(44.0, 46.0, ny)
        )
        return xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((ny, nx), 30.0, dtype="float32"))},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T00:00:00", "ns"),
            },
            attrs={"data_type": "sar_l3_ssm"},
        )

    def _amsr_ssm_two_passes(self, lon=10.0, lat=45.0):
        import xarray as xr

        # Descending (early morning) and ascending (evening) passes of
        # "the same" grid cell on the same day, both within +/-12h of the
        # midnight SAR scene: 06:00 (-6h from previous midnight -> +18h
        # is out of range, so use -6h/+6h around the scene itself).
        times = np.array([
            np.datetime64("2026-06-19T18:00:00", "ns"),  # 6h before scene
            np.datetime64("2026-06-20T06:00:00", "ns"),  # 6h after scene
        ])
        return xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.array([0.15, 0.25], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([lon, lon])),
                "lat": (("point",), np.array([lat, lat])),
                "time": (("point",), times),
            },
            attrs={"data_type": "radiometer_ssm", "sensor": "amsr", "platform_type": "amsr_ssm"},
        )

    def _soil_moisture_recipe(self):
        from sar_validation.core.recipe import (
            CollocationType,
            GeographicBounds,
            LayerVsLayerCollocation,
            PointVsLayerCollocation,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        return Recipe(RecipeConfig(
            name="soil_moisture_satellite_it",
            variable="soil_moisture",
            geographic_bounds=GeographicBounds(8.0, 12.0, 43.0, 47.0),
            temporal_bounds=TemporalBounds("2026-06-19T00:00:00", "2026-06-21T00:00:00"),
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(aggregation_window_km=20.0),
                layer_vs_layer=LayerVsLayerCollocation(),
            ),
        ))

    def test_am_pm_passes_blend_into_one_day_value(self, tmp_path):
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/amsr_ssm": self._amsr_ssm_two_passes(),
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 1
        assert float(result["val_SOIL_MOISTURE"].values[0]) == pytest.approx(0.20, abs=1e-4)

    def _amsr_ssm_single_pass(self, time_str, value, lon=10.0, lat=45.0):
        import xarray as xr

        return xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.array([value], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([lon])),
                "lat": (("point",), np.array([lat])),
                "time": (("point",), np.array([np.datetime64(time_str, "ns")])),
            },
            attrs={"data_type": "radiometer_ssm", "sensor": "amsr", "platform_type": "amsr_ssm"},
        )

    def test_am_pm_passes_in_separate_files_still_blend(self, tmp_path):
        """The real file layout: AMSR2 delivers each overpass as its own
        file, which becomes its own datatree node
        (validation/amsr_ssm/<stem>). Two such sibling nodes -- one per
        pass -- must still blend into a single day value, the same as
        when both passes happen to be in one file (the synthetic shape
        test_am_pm_passes_blend_into_one_day_value already covers)."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/amsr_ssm/granule_am": self._amsr_ssm_single_pass(
                "2026-06-19T18:00:00", 0.15,
            ),
            "/validation/amsr_ssm/granule_pm": self._amsr_ssm_single_pass(
                "2026-06-20T06:00:00", 0.25,
            ),
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 1
        assert float(result["val_SOIL_MOISTURE"].values[0]) == pytest.approx(0.20, abs=1e-4)

    def test_distinct_satellite_sources_never_mixed_together(self, tmp_path):
        """The averaging step must run strictly per validation source: an
        ASCAT reading and a SMAP reading at the same location/time must
        never be blended into one averaged value, only ever averaged with
        other readings of their own source. This is a structural property
        of the per-val_name loop in run_collocation (each source's
        DataFrame is built and averaged independently), not something the
        averaging helper itself enforces -- this test guards against a
        future refactor accidentally pooling sources before averaging."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        ascat = xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.array([40.0], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([10.0])),
                "lat": (("point",), np.array([45.0])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T03:00:00", "ns")])),
            },
            attrs={"data_type": "scatterometer_ssm", "platform_type": "ascat_ssm"},
        )
        smap = xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.array([0.30], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([10.0])),
                "lat": (("point",), np.array([45.0])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T03:00:00", "ns")])),
            },
            attrs={"data_type": "radiometer_ssm", "sensor": "smap", "platform_type": "smap_ssm"},
        )
        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/ascat_ssm": ascat,
            "/validation/smap_ssm": smap,
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        # Two separate collocations (one per source), not one blended
        # 20.15 (mean of 40.0 % and 0.30 m3/m3, which would also be
        # physically meaningless -- different units entirely).
        assert result.sizes.get("collocation", 0) == 2
        val_sources = set(result["val_source"].values.tolist())
        assert val_sources == {"ascat_ssm", "smap_ssm"}
        values_by_source = dict(zip(
            result["val_source"].values.tolist(),
            result["val_SOIL_MOISTURE"].values.tolist(),
        ))
        assert values_by_source["ascat_ssm"] == pytest.approx(40.0)
        assert values_by_source["smap_ssm"] == pytest.approx(0.30)

    def test_gportal_snap_uses_native_degree_grid_not_km_conversion(self, tmp_path):
        """G-Portal AMSR2's real grid-centre formula (see
        ``_from_amsr_ssm_gportal_l3_grid``, ``-180 + (i+0.5)*(360/nx)``
        with nx=3600) places every centre at an exact X.X5 offset -- e.g.
        9.05, 9.15, 9.25, 9.35, .... Two such REAL adjacent centres, 9.15
        and 9.25, are used here (9.00/9.10 never actually occur on this
        grid and don't exercise the bug).

        Both 9.15 and 9.25 sit exactly on a rounding tie under the old
        ``np.round(v / 0.1) * 0.1`` snap (91.500...06 and 92.5 in step
        units), which NumPy's round-half-to-even resolves to the SAME
        bucket (9.2) -- silently merging two genuinely adjacent native
        cells. Grouping on the raw (exact) lon/lat instead -- since a
        fixed grid reports IDENTICAL coordinates for repeated readings of
        the same cell, per docs/design-choices.md §8.7 -- must keep them
        as two separate, unblended collocations."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        distinct_pair = xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.array([0.10, 0.30], dtype="float32"))},
            coords={
                "lon": (("point",), np.array([9.15, 9.25])),
                "lat": (("point",), np.array([45.00, 45.00])),
                "time": (("point",), np.array([
                    np.datetime64("2026-06-19T18:00:00", "ns"),
                    np.datetime64("2026-06-20T06:00:00", "ns"),
                ])),
            },
            attrs={
                "data_type": "radiometer_ssm", "sensor": "amsr", "platform_type": "amsr_ssm",
                "native_grid_deg": 0.1,
            },
        )
        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/amsr_ssm": distinct_pair,
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        # Must stay as two separate, unblended collocations -- not merged
        # into one averaged 0.20 the way the old round-half-to-even tie
        # would.
        assert result.sizes.get("collocation", 0) == 2
        values = sorted(float(v) for v in result["val_SOIL_MOISTURE"].values)
        assert values == pytest.approx([0.10, 0.30], abs=1e-4)

    def test_gportal_grid_ties_do_not_merge_adjacent_cells(self, tmp_path):
        """Five consecutive real G-Portal native grid centres (9.05,
        9.15, 9.25, 9.35, 9.45) must ALL stay as distinct collocations.

        Under the old ``np.round(v / 0.1) * 0.1`` snap, 9.15 and 9.25
        both resolve to bucket 9.2 (a round-half-to-even tie), so only 4
        of these 5 cells would have survived as distinct groups. With
        Fix 1 (group on raw lon/lat for native_grid_deg sources, no
        snap), all 5 must remain distinct."""
        import xarray as xr

        from sar_validation.core.collocation import run_collocation

        lons = [9.05, 9.15, 9.25, 9.35, 9.45]
        values = [0.10, 0.20, 0.30, 0.40, 0.50]
        ds = xr.Dataset(
            {"SOIL_MOISTURE": (("point",), np.array(values, dtype="float32"))},
            coords={
                "lon": (("point",), np.array(lons)),
                "lat": (("point",), np.full(len(lons), 45.0)),
                "time": (("point",), np.array(
                    [np.datetime64("2026-06-20T03:00:00", "ns")] * len(lons)
                )),
            },
            attrs={
                "data_type": "radiometer_ssm", "sensor": "amsr", "platform_type": "amsr_ssm",
                "native_grid_deg": 0.1,
            },
        )
        tree = xr.DataTree.from_dict({
            "/sar/scene1": self._sar_grid(),
            "/validation/amsr_ssm": ds,
        })

        result = run_collocation(self._soil_moisture_recipe(), tree, tmp_path)

        assert result is not None
        assert result.sizes.get("collocation", 0) == 5


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


class TestAverageWithinSarTolerance:
    """Unit tests for the pre-collocation temporal-averaging helper used
    by run_collocation for soil-moisture sources (ISMN stations and
    ASCAT/AMSR2/SMAP/SMOS grid cells). See
    docs/superpowers/specs/2026-07-29-soil-moisture-temporal-averaging-design.md."""

    def test_averages_all_readings_within_tolerance_for_one_station(self):
        from sar_validation.core.collocation import _average_within_sar_tolerance

        # 5 hourly readings from one station, all within +/-12h of a
        # midnight SAR scene.
        hours = [0, 4, 6, 8, 12]
        val = _make_val_dataframe(
            lons=[0.0] * 5, lats=[52.0] * 5,
            times=[datetime(2026, 1, 1, h, 0, 0) for h in hours],
            SOIL_MOISTURE=[0.10, 0.12, 0.20, 0.28, 0.30],
            platform_id=["StationA"] * 5,
        )
        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["platform_id"], time_tolerance_minutes=720,
        )

        assert len(result) == 1
        assert result.iloc[0]["SOIL_MOISTURE"] == pytest.approx(0.20)
        assert result.iloc[0]["platform_id"] == "StationA"
        assert result.iloc[0]["time"] == datetime(2026, 1, 1, 0, 0, 0)

    def test_keeps_distinct_stations_separate(self):
        from sar_validation.core.collocation import _average_within_sar_tolerance

        val = _make_val_dataframe(
            lons=[0.0, 1.0], lats=[52.0, 53.0],
            times=[datetime(2026, 1, 1, 3, 0, 0), datetime(2026, 1, 1, 3, 0, 0)],
            SOIL_MOISTURE=[0.10, 0.40],
            platform_id=["StationA", "StationB"],
        )
        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["platform_id"], time_tolerance_minutes=720,
        )

        assert len(result) == 2
        assert set(result["platform_id"]) == {"StationA", "StationB"}

    def test_falls_back_to_lonlat_grouping_without_platform_id(self):
        from sar_validation.core.collocation import _average_within_sar_tolerance

        val = _make_val_dataframe(
            lons=[0.0, 0.0], lats=[52.0, 52.0],
            times=[datetime(2026, 1, 1, 3, 0, 0), datetime(2026, 1, 1, 9, 0, 0)],
            SOIL_MOISTURE=[0.10, 0.30],
        )
        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["lon", "lat"], time_tolerance_minutes=720,
        )

        assert len(result) == 1
        assert result.iloc[0]["SOIL_MOISTURE"] == pytest.approx(0.20)

    def test_drops_readings_outside_tolerance_of_every_sar_time(self):
        from sar_validation.core.collocation import _average_within_sar_tolerance

        val = _make_val_dataframe(
            lons=[0.0, 0.0], lats=[52.0, 52.0],
            times=[datetime(2026, 1, 1, 3, 0, 0), datetime(2026, 1, 2, 15, 0, 0)],
            SOIL_MOISTURE=[0.10, 0.90],
            platform_id=["StationA", "StationA"],
        )
        # 15:00 on day 2 is 39h after the single SAR scene -- outside
        # +/-12h tolerance, must be dropped rather than pulled in.
        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["platform_id"], time_tolerance_minutes=720,
        )

        assert len(result) == 1
        assert result.iloc[0]["SOIL_MOISTURE"] == pytest.approx(0.10)

    def test_assigns_readings_to_nearest_of_multiple_sar_scenes(self):
        from sar_validation.core.collocation import _average_within_sar_tolerance

        val = _make_val_dataframe(
            lons=[0.0, 0.0, 0.0, 0.0], lats=[52.0, 52.0, 52.0, 52.0],
            times=[
                datetime(2026, 1, 1, 3, 0, 0), datetime(2026, 1, 1, 9, 0, 0),
                datetime(2026, 1, 2, 3, 0, 0), datetime(2026, 1, 2, 9, 0, 0),
            ],
            SOIL_MOISTURE=[0.10, 0.20, 0.50, 0.70],
            platform_id=["StationA"] * 4,
        )
        sar_times = [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 2, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["platform_id"], time_tolerance_minutes=720,
        )

        assert len(result) == 2
        by_time = {row["time"]: row["SOIL_MOISTURE"] for _, row in result.iterrows()}
        assert by_time[datetime(2026, 1, 1, 0, 0, 0)] == pytest.approx(0.15)
        assert by_time[datetime(2026, 1, 2, 0, 0, 0)] == pytest.approx(0.60)

    def test_spatial_snap_groups_nearby_swath_pixels_together(self):
        from sar_validation.core.collocation import _average_within_sar_tolerance, _snap_to_grid

        # Two AMSR2 AU_Land-style swath pixels a few hundred metres apart
        # (not bit-identical, unlike a fixed grid) representing an
        # ascending and a descending pass of "the same" cell.
        val = _make_val_dataframe(
            lons=[10.001, 10.003], lats=[45.002, 45.004],
            times=[datetime(2026, 1, 1, 4, 0, 0), datetime(2026, 1, 1, 10, 0, 0)],
            SOIL_MOISTURE=[0.15, 0.25],
        )
        deg_step = 25.0 / 111.0  # AMSR default aggregation_window_km
        val = val.copy()
        val["_snap_lon"] = _snap_to_grid(val["lon"].values, deg_step)
        val["_snap_lat"] = _snap_to_grid(val["lat"].values, deg_step)

        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]
        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["_snap_lon", "_snap_lat"], time_tolerance_minutes=720,
        )

        assert len(result) == 1
        assert result.iloc[0]["SOIL_MOISTURE"] == pytest.approx(0.20)

    def test_nat_in_validation_time_column_is_dropped_not_crashed(self):
        """A NaT entry in the validation DataFrame's own ``time`` column
        (the LEFT side of the internal pd.merge_asof) must not crash --
        this is reachable on real data via from_amsr_ssm's NSIDC-0451
        branch, which sets time=NaT when time_coverage_start is missing
        from the file (see docs/design-choices.md and
        _subset_point_ds's docstring: NaT-timed points are deliberately
        kept, not dropped, upstream of this averaging step). The NaT
        reading itself must be excluded from the group's average, while
        the group's other valid readings still average correctly."""
        from sar_validation.core.collocation import _average_within_sar_tolerance

        val = _make_val_dataframe(
            lons=[0.0, 0.0, 0.0], lats=[52.0, 52.0, 52.0],
            times=[
                datetime(2026, 1, 1, 3, 0, 0),
                pd.NaT,
                datetime(2026, 1, 1, 9, 0, 0),
            ],
            SOIL_MOISTURE=[0.10, 999.0, 0.30],
            platform_id=["StationA", "StationA", "StationA"],
        )
        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["platform_id"], time_tolerance_minutes=720,
        )

        assert len(result) == 1
        # The NaT row's 999.0 reading must not pollute the average.
        assert result.iloc[0]["SOIL_MOISTURE"] == pytest.approx(0.20)

    def test_all_nat_readings_for_a_group_produce_no_output_rows(self):
        """If every reading for a group has a NaT time, that group must
        simply produce zero output rows, not a crash."""
        from sar_validation.core.collocation import _average_within_sar_tolerance

        val = _make_val_dataframe(
            lons=[0.0, 0.0], lats=[52.0, 52.0],
            times=[pd.NaT, pd.NaT],
            SOIL_MOISTURE=[0.10, 0.30],
            platform_id=["StationA", "StationA"],
        )
        sar_times = [datetime(2026, 1, 1, 0, 0, 0)]

        result = _average_within_sar_tolerance(
            val, sar_times, group_cols=["platform_id"], time_tolerance_minutes=720,
        )

        assert len(result) == 0


class TestRunCollocationEra5Wiring:
    def _build_datatree_and_recipe(self, tmp_path):
        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.core.recipe import (
            CollocationType,
            GeographicBounds,
            LayerVsLayerCollocation,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        # SAR grid-mode scene: 4x4 pixels, single acquisition time.
        sar_lat = np.linspace(40.5, 41.5, 4)
        sar_lon = np.linspace(-9.5, -8.5, 4)
        sar_lon2d, sar_lat2d = np.meshgrid(sar_lon, sar_lat)
        sar_time = np.datetime64("2026-07-12T01:00:00")
        sar_ds = xr.Dataset(
            {"owiWindSpeed": (("y", "x"), np.full((4, 4), 7.5))},
            coords={
                "lon": (("y", "x"), sar_lon2d), "lat": (("y", "x"), sar_lat2d),
                "time": sar_time,
            },
        )

        # ERA5 validation node: 2x2 native cells, hours 0/1/2, constant
        # ramp value = hour_index * 10 (spatially uniform, as in
        # test_model_collocation.py's _make_era5_ds) so the expected
        # interpolated value is predictable.
        era5_lat = np.linspace(40.0, 42.0, 2)
        era5_lon = np.linspace(-10.0, -8.0, 2)
        era5_time = pd.to_datetime(
            ["2026-07-12T00:00:00", "2026-07-12T01:00:00", "2026-07-12T02:00:00"]
        )
        u10 = np.stack([np.full((2, 2), h * 10.0) for h in range(3)])
        era5_ds = xr.Dataset(
            {"u10": (("time", "lat", "lon"), u10)},
            coords={"time": era5_time, "lat": era5_lat, "lon": era5_lon},
        )
        era5_ds.attrs["data_type"] = "era5_wind"
        era5_ds.attrs["platform_type"] = "era5_wind"

        tree = xr.DataTree.from_dict({
            "/sar/scene1": sar_ds,
            "/validation/era5/era5": era5_ds,
        })

        cfg = RecipeConfig(
            name="t", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 40.0, 42.0),
            temporal_bounds=TemporalBounds("2026-07-12T00:00:00", "2026-07-12T02:00:00"),
            validation_sources=[ValidationDataSource(source_type="era5")],
            collocation=CollocationType(
                layer_vs_layer=LayerVsLayerCollocation(layer_type_specs={
                    "era5_wind": {
                        "method": "cell-averaging", "temporal_method": "nearest",
                        # Deliberately generous: the ERA5 grid here spans a
                        # full 2deg x 2deg box while the SAR grid only
                        # covers the inner 1deg x 1deg, so the nearest SAR
                        # pixel to an ERA5 corner cell is ~70km away. This
                        # test only verifies era5 nodes get wired into
                        # run_collocation at all -- aggregation-window
                        # tuning itself is covered by Task 9's dedicated
                        # unit tests, so a wide window here avoids the test
                        # being fragile to the exact synthetic geometry.
                        "aggregation_window_km": 300.0, "distance_weighting": "equal",
                    },
                }),
            ),
        )
        return tree, Recipe(cfg)

    def test_era5_node_produces_model_vs_layer_matches(self, tmp_path):
        from sar_validation.core.collocation import run_collocation

        tree, recipe = self._build_datatree_and_recipe(tmp_path)
        result_ds = run_collocation(recipe, tree, tmp_path)

        assert result_ds is not None
        assert "val_source" in result_ds
        sources = set(result_ds["val_source"].values.tolist())
        assert "era5_wind" in sources

        era5_mask = result_ds["val_source"].values == "era5_wind"
        assert int(era5_mask.sum()) > 0


class TestModelSourceType:
    def test_era5_prefixed_data_types_map_to_era5(self):
        from sar_validation.core.collocation import _model_source_type

        assert _model_source_type("era5_wind") == "era5"
        assert _model_source_type("era5_waves") == "era5"
        assert _model_source_type("era5_soil_moisture") == "era5"

    def test_hycom_maps_to_hycom(self):
        from sar_validation.core.collocation import _model_source_type

        assert _model_source_type("hycom") == "hycom"

    def test_unrelated_data_type_returns_none(self):
        from sar_validation.core.collocation import _model_source_type

        assert _model_source_type("scatterometer") is None
        assert _model_source_type("") is None


class TestRunCollocationHycomModelSourceDispatch:
    def _currents_recipe_with_hycom_override(self, method: str):
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )
        return Recipe(RecipeConfig(
            name="hycom_it", variable="currents",
            geographic_bounds=GeographicBounds(-20.0, -19.0, 50.0, 51.0),
            temporal_bounds=TemporalBounds("2026-06-20T18:00:00", "2026-06-20T23:00:00"),
            validation_sources=[ValidationDataSource(
                source_type="hycom",
                collocation_kwargs={"method": method, "temporal_method": "nearest"},
            )],
        ))

    def _tree(self):
        import xarray as xr

        # Same SAR grid shape as the existing HF-radar/in-situ currents
        # integration test above (TestRunCollocationCurrentsFromDatatree),
        # so results are directly comparable: constant heading=90 makes
        # the projection hand-checkable (projection == EWCT).
        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        sar = xr.Dataset(
            {
                "rvlRadVel": (("y", "x"), np.full((ny, nx), 0.5, dtype="float32")),
                "rvlHeading": (("y", "x"), np.full((ny, nx), 90.0, dtype="float32")),
            },
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T19:15:00", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn", "swath_mode": "IW/EW/SM", "measurement_type": "rvl"},
        )
        # Gridded HyCOM node covering the whole SAR bbox, single hour-aligned
        # timestamp (temporal_method="nearest" in the recipe override above,
        # so no bracketing-hour data is needed).
        hlon, hlat = np.meshgrid(np.linspace(-20.0, -19.0, 3), np.linspace(50.0, 51.0, 3))
        hycom = xr.Dataset(
            {
                "EWCT": (("time", "lat", "lon"), np.full((1, 3, 3), 0.4, dtype="float32")),
                "NSCT": (("time", "lat", "lon"), np.full((1, 3, 3), 0.3, dtype="float32")),
            },
            coords={
                "time": [np.datetime64("2026-06-20T19:00:00", "ns")],
                "lat": np.linspace(50.0, 51.0, 3),
                "lon": np.linspace(-20.0, -19.0, 3),
            },
            attrs={"data_type": "hycom", "platform_type": "hycom"},
        )
        return xr.DataTree.from_dict({"/sar/scene1": sar, "/validation/hycom/hycom": hycom})

    def test_hycom_individual_method_override_is_actually_applied(self, tmp_path):
        from sar_validation.core.collocation import run_collocation

        # Regression test for the era5-hardcoded model-source detection
        # bug: before the fix, hycom's own collocation_kwargs override
        # (method="individual") was silently ignored in favour of
        # source_type_overrides.get("era5", {}) -- always falling back to
        # DEFAULT_LAYER_TYPE_SPECS["hycom"]["method"] == "cell-averaging"
        # regardless of what the recipe asked for. "individual" produces
        # one match per valid SAR pixel (25, for this 5x5 grid);
        # "cell-averaging" with the small default aggregation_window_km
        # (4.6 km) against this coarse 3x3 HyCOM grid produces far fewer
        # (likely 0) -- a stark, unambiguous signal either way.
        recipe = self._currents_recipe_with_hycom_override("individual")
        result = run_collocation(recipe, self._tree(), tmp_path)
        assert result is not None
        assert len(result["val_rvlRadVel_projection"]) > 5
        # heading 90 -> projection == EWCT == 0.4 (hand-checkable, same as
        # the existing mooring test above)
        assert float(result["val_rvlRadVel_projection"].values[0]) == pytest.approx(0.4, abs=1e-5)
