"""Tests for model_collocation.py -- ERA5/SAR bilinear + temporal interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


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


class TestDeriveWindWspdWdir:
    """Direct unit tests for _derive_wind_wspd_wdir -- the C1 fix's core
    helper, which derives WSPD/WDIR from FINAL (already interpolated)
    u10/v10 components instead of from_era5 deriving WDIR (a circular
    quantity) BEFORE interpolation."""

    def test_noop_without_both_components(self):
        from sar_validation.core.model_collocation import _derive_wind_wspd_wdir

        values = {"u10": np.array([1.0]), "swh": np.array([2.0])}
        result = _derive_wind_wspd_wdir(values)
        assert "WSPD" not in result and "WDIR" not in result
        assert result["u10"][0] == pytest.approx(1.0)
        assert result["swh"][0] == pytest.approx(2.0)

    def test_noop_for_non_wind_dict(self):
        from sar_validation.core.model_collocation import _derive_wind_wspd_wdir

        values = {"VHM0": np.array([1.5])}
        result = _derive_wind_wspd_wdir(values)
        assert result == values

    def test_hand_checkable_northerly_wind(self):
        """u10=0, v10=-1 is wind blowing FROM the north -- meteorological
        convention gives WDIR = 0/360 degrees exactly."""
        from sar_validation.core.model_collocation import _derive_wind_wspd_wdir

        values = {"u10": np.array([0.0]), "v10": np.array([-1.0])}
        result = _derive_wind_wspd_wdir(values)
        assert "u10" not in result and "v10" not in result
        assert result["WSPD"][0] == pytest.approx(1.0)
        assert result["WDIR"][0] == pytest.approx(0.0) or result["WDIR"][0] == pytest.approx(360.0)

    @pytest.mark.parametrize("angle_deg", [10.0, 90.0, 180.0, 270.0, 355.0, 0.5, 359.5])
    def test_wdir_round_trips_and_stays_in_valid_range(self, angle_deg):
        """Property-style check: for a range of target angles (including
        ones right next to the 0/360 seam), a unit vector built from that
        angle round-trips back through _derive_wind_wspd_wdir to the same
        angle, always within [0, 360)."""
        from sar_validation.core.model_collocation import _derive_wind_wspd_wdir

        rad = np.radians(270.0 - angle_deg)
        values = {"u10": np.array([np.cos(rad)]), "v10": np.array([np.sin(rad)])}
        result = _derive_wind_wspd_wdir(values)
        assert 0.0 <= result["WDIR"][0] < 360.0
        assert result["WDIR"][0] == pytest.approx(angle_deg % 360.0, abs=1e-3)
        assert result["WSPD"][0] == pytest.approx(1.0, abs=1e-6)


def _make_era5_wind_ds(
    angles_deg, hours=("2026-07-12T00:00:00", "2026-07-12T01:00:00", "2026-07-12T02:00:00"),
    n_lat=3, n_lon=3, speed=10.0,
):
    """Synthetic gridded ERA5 wind Dataset (u10/v10, spatially constant per
    hour) where hour ``i`` corresponds to the compass wind direction
    ``angles_deg[i]`` (meteorological "from" convention) -- built by
    inverting the same WDIR formula _derive_wind_wspd_wdir uses, so tests
    can specify a target direction per hour directly rather than raw
    components."""
    lat = np.linspace(40.0, 42.0, n_lat)
    lon = np.linspace(-10.0, -8.0, n_lon)
    time = pd.to_datetime(list(hours))
    u_list, v_list = [], []
    for a in angles_deg:
        rad = np.radians(270.0 - a)
        u_list.append(np.full((n_lat, n_lon), speed * np.cos(rad)))
        v_list.append(np.full((n_lat, n_lon), speed * np.sin(rad)))
    ds = xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), np.stack(u_list)),
            "v10": (("time", "lat", "lon"), np.stack(v_list)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds


class TestModelValuesAtPointsWindDirectionSeam:
    """Regression tests for C1: WDIR must never be interpolated as an
    ordinary linear/hyperbolic scalar across the 0/360 wrap. Exercises the
    full _model_values_at_points path (spatial + hyperbolic temporal
    interpolation of u10/v10, THEN WSPD/WDIR derivation) end to end."""

    def test_output_has_wspd_wdir_not_u10_v10_when_both_present(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_wind_ds([10.0, 20.0, 30.0])
        times = np.array([np.datetime64("2026-07-12T01:00:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "nearest",
        )
        assert "WSPD" in result and "WDIR" in result
        assert "u10" not in result and "v10" not in result

    def test_wdir_interpolated_across_seam_is_near_zero_not_180(self):
        """hour0=355deg (away from the seam), hour1=359deg, hour2=1deg --
        the true wind direction crosses the 0/360 seam between hour1 and
        hour2. Querying halfway between them (t_prime=0.5) must produce a
        WDIR close to 0/360 (the "short way around"), never close to 180
        (the "long way around" a naive linear interpolation of the angle
        itself would wrongly produce)."""
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_wind_ds([355.0, 359.0, 1.0])
        times = np.array([np.datetime64("2026-07-12T01:30:00")])  # t_prime=0.5
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "hyperbolic",
        )
        wdir = result["WDIR"][0]
        assert 0.0 <= wdir < 360.0
        wrap_distance = min(wdir, 360.0 - wdir)
        assert wrap_distance < 30.0, f"expected WDIR near the 0/360 seam, got {wdir}"
        assert abs(wdir - 180.0) > 100.0, f"WDIR landed near 180 (the wrong, long way around): {wdir}"

    @pytest.mark.parametrize("a0,a1,a2", [
        (355.0, 359.0, 1.0),
        (358.0, 0.5, 3.0),
        (350.0, 355.0, 5.0),
        (10.0, 20.0, 30.0),   # away from the seam -- no regression
        (170.0, 180.0, 190.0),  # crosses 180 -- NOT a seam, ordinary case
    ])
    def test_wdir_always_in_valid_range(self, a0, a1, a2):
        from sar_validation.core.model_collocation import _model_values_at_points

        era5_ds = _make_era5_wind_ds([a0, a1, a2])
        times = np.array([np.datetime64("2026-07-12T01:30:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "hyperbolic",
        )
        assert 0.0 <= result["WDIR"][0] < 360.0


def _make_era5_ds(
    n_lat=3, n_lon=3, hours=("2026-07-12T00:00:00", "2026-07-12T01:00:00", "2026-07-12T02:00:00"),
    lsm=None,
):
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
    if lsm is not None:
        # Matches DataTreeConverter.from_era5's representation: a
        # (lat, lon)-only non-dimension coordinate (not a data_var), so it
        # never leaks into model_vars/val_data iteration.
        ds = ds.assign_coords(lsm=(("lat", "lon"), np.broadcast_to(lsm, (n_lat, n_lon)).astype(float)))
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

    def test_lsm_never_returned_as_its_own_key(self):
        """lsm is a masking input, not a model variable -- it must never
        appear as its own key in the returned dict, whether or not it's
        present on era5_ds."""
        from sar_validation.core.model_collocation import _model_values_at_points

        lsm = np.zeros((3, 3))
        era5_ds = _make_era5_ds(lsm=lsm)
        times = np.array([np.datetime64("2026-07-12T01:00:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "nearest",
        )
        assert "lsm" not in result

    def test_land_influenced_point_masks_all_model_vars_to_nan(self):
        """Interpolation-contamination guard for the 'individual'
        collocation path (bilinear ERA5 interpolation at an exact SAR
        pixel location): a query point close enough to a land grid cell
        that the bilinearly-interpolated lsm value itself exceeds 0.5 is
        treated as too land-influenced to be a valid ocean wind match --
        mirrors the cell-averaging land-skip's rationale, applied via
        interpolation instead of grid-cell-center-is-land."""
        from sar_validation.core.model_collocation import _model_values_at_points

        # 3x3 grid, corner cell (lat=40, lon=-10) is fully land (lsm=1.0),
        # everything else fully sea (lsm=0.0).
        lsm = np.zeros((3, 3))
        lsm[0, 0] = 1.0
        era5_ds = _make_era5_ds(lsm=lsm)
        times = np.array([np.datetime64("2026-07-12T01:00:00")])
        # Query point right at the land cell's own center -> interpolated
        # lsm = 1.0 > 0.5 -> masked to NaN.
        result = _model_values_at_points(
            np.array([-10.0]), np.array([40.0]), times, era5_ds, "nearest",
        )
        assert np.isnan(result["u10"][0])

    def test_sea_influenced_point_keeps_model_vars_with_lsm_present(self):
        """No regression: a query point far from any land cell (bilinearly
        interpolated lsm <= 0.5) still returns its model value normally
        when lsm is present on era5_ds."""
        from sar_validation.core.model_collocation import _model_values_at_points

        lsm = np.zeros((3, 3))
        lsm[0, 0] = 1.0
        era5_ds = _make_era5_ds(lsm=lsm)
        times = np.array([np.datetime64("2026-07-12T01:00:00")])
        # Query point at the grid's opposite corner (lat=42, lon=-8) --
        # far from the single land cell at (lat=40, lon=-10).
        result = _model_values_at_points(
            np.array([-8.0]), np.array([42.0]), times, era5_ds, "nearest",
        )
        assert result["u10"][0] == pytest.approx(10.0)


class TestModelValuesAtPointsThreeHourlySpacing:
    """Regression tests for the hyperbolic-interpolation bracket-spacing
    bug: the pre-fix code hardcoded a 1-hour gap between bracket points in
    ``t_prime = (t - t2) / 1h``, which is correct for ERA5's genuinely
    hourly granules but silently WRONG for HyCOM's real 3-hourly cadence
    (00:00, 03:00, 06:00, ...). These use synthetic data spaced 3 hours
    apart -- modeling HyCOM's real granule spacing -- and hand-computed
    expected values, so they fail against the pre-fix hardcoded-1h code
    (which computes t_prime = 1.0 instead of the correct 1/3) and pass
    after the fix."""

    def test_hyperbolic_uses_actual_bracket_spacing_not_hardcoded_1h(self):
        from sar_validation.core.model_collocation import _model_values_at_points

        # 4 granules, 3 hours apart: 00:00, 03:00, 06:00, 09:00. u10 is a
        # linear ramp (value = hour_index * 10), so the correct
        # (spacing-aware) hyperbolic result reduces to ordinary linear
        # interpolation -- hand-checkable independent of the quadratic
        # formula's curvature term.
        era5_ds = _make_era5_ds(
            hours=(
                "2026-07-12T00:00:00", "2026-07-12T03:00:00",
                "2026-07-12T06:00:00", "2026-07-12T09:00:00",
            ),
        )
        # Bracket center t2 = 03:00 (value 10.0), forward neighbor 06:00
        # (value 20.0). Query 1 hour past t2: true fractional position
        # within the 3-hour gap is 1/3, NOT 1.0 (what hardcoded-1h would
        # compute). Hand-computed expected value:
        #   t_prime = 1/3
        #   a = (20 + 0 - 2*10) / 2 = 0.0
        #   b = (20 - 0) / 2 = 10.0
        #   c = 10.0
        #   expected = a*(1/3)**2 + b*(1/3) + c = 10/3 + 10 = 13.333...
        times = np.array([np.datetime64("2026-07-12T04:00:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "hyperbolic",
        )
        expected = 0.0 * (1 / 3) ** 2 + 10.0 * (1 / 3) + 10.0
        assert expected == pytest.approx(13.333333, abs=1e-4)
        assert result["u10"][0] == pytest.approx(expected, abs=1e-6)
        # The pre-fix hardcoded-1h bug would instead compute t_prime=1.0,
        # landing exactly on val3 (20.0) -- explicitly guard against that
        # wrong answer resurfacing.
        assert result["u10"][0] != pytest.approx(20.0, abs=1e-6)

    def test_irregular_bracket_spacing_is_nan_not_a_fabricated_value(self):
        """When the backward and forward gaps around the bracket center
        differ (a genuinely irregular cadence -- e.g. around a real HyCOM
        data gap), the quadratic formula's equal-spacing assumption no
        longer holds for a single shared unit. Fabricating an answer using
        either gap would be silently wrong, so this must come back NaN
        (with a debug-level log, not a crash) rather than guessing."""
        from sar_validation.core.model_collocation import _model_values_at_points

        # Backward gap (03:00 -> 00:00) = 3h, forward gap (07:00 -> 03:00)
        # = 4h -- irregular.
        era5_ds = _make_era5_ds(
            hours=(
                "2026-07-12T00:00:00", "2026-07-12T03:00:00",
                "2026-07-12T07:00:00", "2026-07-12T10:00:00",
            ),
        )
        times = np.array([np.datetime64("2026-07-12T04:00:00")])
        result = _model_values_at_points(
            np.array([-9.0]), np.array([41.0]), times, era5_ds, "hyperbolic",
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

    def test_wind_dataset_produces_wspd_wdir_not_u10_v10(self):
        """C1 regression: when era5_ds has both u10/v10 (a wind Dataset),
        the final collocated val_data must carry WSPD/WDIR -- derived from
        the interpolated components -- not raw u10/v10."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_wind_ds([10.0, 20.0, 30.0])
        sar_lon = np.array([[-9.0]])
        sar_lat = np.array([[41.0]])
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.array([[[7.0]]])}

        colloc = ModelLayerCollocation(method="individual", temporal_method="nearest")
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 1
        assert "WSPD" in results[0].val_data and "WDIR" in results[0].val_data
        assert "u10" not in results[0].val_data and "v10" not in results[0].val_data
        assert 0.0 <= results[0].val_data["WDIR"] < 360.0


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
        # ERA5 cell for sparse vignette points.
        colloc = ModelLayerCollocation(method="cell-averaging", temporal_method="nearest")
        results = colloc.collocate_points(
            sar_point_vars=sar_point_vars, sar_lons=sar_lons, sar_lats=sar_lats,
            sar_times=sar_times, era5_ds=era5_ds, val_source="era5", sar_scene_name="wv1",
        )
        assert len(results) == 2
        assert all(r.val_data["u10"] == pytest.approx(10.0) for r in results)
        assert all(r.collocation_type == "model_vs_layer" for r in results)

    def test_wind_dataset_produces_wspd_wdir_not_u10_v10(self):
        """C1 regression: WV-mode (collocate_points) also shares
        _model_values_at_points -- must produce WSPD/WDIR, not u10/v10."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_wind_ds([10.0, 20.0, 30.0])
        sar_point_vars = {"oswHs": np.array([2.0])}
        sar_lons = np.array([-9.0])
        sar_lats = np.array([41.0])
        sar_times = np.array([np.datetime64("2026-07-12T01:00:00")])

        colloc = ModelLayerCollocation(method="individual", temporal_method="nearest")
        results = colloc.collocate_points(
            sar_point_vars=sar_point_vars, sar_lons=sar_lons, sar_lats=sar_lats,
            sar_times=sar_times, era5_ds=era5_ds, val_source="era5", sar_scene_name="wv1",
        )
        assert len(results) == 1
        assert "WSPD" in results[0].val_data and "WDIR" in results[0].val_data
        assert "u10" not in results[0].val_data and "v10" not in results[0].val_data


class TestModelLayerCollocationCellAveraging:
    def test_produces_one_match_per_era5_cell_with_nearby_sar(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds(n_lat=2, n_lon=2)  # 4 native cells
        # Dense SAR grid covering the same small area, 10x10 pixels.
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 4  # one per ERA5 native cell
        for r in results:
            assert r.val_data["u10"] == pytest.approx(10.0)
            assert r.sar_data["owiWindSpeed"] == pytest.approx(7.5)
            assert r.collocation_type == "model_vs_layer"
            assert r.temporal_distance_minutes == 0.0

    def test_cell_with_no_nearby_sar_produces_no_match(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds(n_lat=2, n_lon=2)
        # SAR grid far away from the ERA5 cells -- no overlap within the
        # aggregation window.
        sar_lon = np.array([[50.0]])
        sar_lat = np.array([[50.0]])
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.array([[[7.5]]])}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest", aggregation_window_km=5.0,
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert results == []

    def test_land_cell_produces_no_match_even_with_nearby_ocean_sar(self):
        """A coastal ERA5 grid cell whose own center is over land (lsm >
        0.5) must be skipped entirely -- even though there are valid
        ocean SAR pixels within its aggregation window. Root cause: ERA5's
        wind field uses different surface-roughness/friction physics over
        land vs. sea, so a land grid point's wind isn't comparable to SAR
        ocean wind retrieval regardless of nearby ocean pixels."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        lsm = np.array([[0.9, 0.0], [0.0, 0.0]])  # cell (0,0) is land
        era5_ds = _make_era5_ds(n_lat=2, n_lon=2, lsm=lsm)
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        # 4 native cells total, minus the 1 land cell -> 3 matches.
        assert len(results) == 3
        land_lat, land_lon = era5_ds["lat"].values[0], era5_ds["lon"].values[0]
        assert not any(
            r.val_lat == pytest.approx(land_lat) and r.val_lon == pytest.approx(land_lon)
            for r in results
        )

    def test_sea_cell_still_matches_with_lsm_present(self):
        """No regression for the sea case: when lsm is present but every
        cell is sea (lsm <= 0.5), all cells still produce a match exactly
        as before this fix."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        lsm = np.zeros((2, 2))  # all sea
        era5_ds = _make_era5_ds(n_lat=2, n_lon=2, lsm=lsm)
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 4

    def test_no_lsm_in_dataset_is_backward_compatible(self):
        """A waves/soil_moisture-shaped era5_ds (or an old wind fixture
        from before this fix) has no lsm coord at all -- must not crash,
        and no land-skip is applied since there's no data to skip on."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds(n_lat=2, n_lon=2)  # no lsm
        assert "lsm" not in era5_ds.variables
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 4

    def test_lsm_never_appears_as_val_data_key(self):
        """Even when lsm is present, it must never leak into a result's
        val_data as a spurious val_lsm statistics column."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        lsm = np.zeros((2, 2))
        era5_ds = _make_era5_ds(n_lat=2, n_lon=2, lsm=lsm)
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 4
        for r in results:
            assert "lsm" not in r.val_data

    def test_hyperbolic_no_bracketing_hour_returns_no_matches(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds(n_lat=2, n_lon=2)  # hours 0,1,2 only
        lat_pix = np.linspace(39.8, 42.2, 5)
        lon_pix = np.linspace(-10.2, -7.8, 5)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        # 02:30 needs hour 3, which doesn't exist.
        sar_time = np.array([np.datetime64("2026-07-12T02:30:00")])
        sar_data = {"owiWindSpeed": np.full((1, 5, 5), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="hyperbolic", aggregation_window_km=60.0,
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert results == []

    def test_wind_dataset_produces_wspd_wdir_not_u10_v10(self):
        """C1 regression: cell-averaging's per-cell val_point must carry
        WSPD/WDIR (derived from interpolated u10/v10), not raw
        u10/v10."""
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_wind_ds([10.0, 20.0, 30.0], n_lat=2, n_lon=2)
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="nearest",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 4
        for r in results:
            assert "WSPD" in r.val_data and "WDIR" in r.val_data
            assert "u10" not in r.val_data and "v10" not in r.val_data
            assert 0.0 <= r.val_data["WDIR"] < 360.0


class TestModelLayerCollocationCellAveragingThreeHourlySpacing:
    """Cell-averaging-path counterpart to
    TestModelValuesAtPointsThreeHourlySpacing -- same bracket-spacing bug,
    same fix, exercised through ``_collocate_cell_averaging_grid`` instead
    of ``_model_values_at_points`` (its ``t_prime`` is computed
    independently, at a different call site)."""

    def test_hyperbolic_uses_actual_bracket_spacing_not_hardcoded_1h(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds(
            n_lat=2, n_lon=2,
            hours=(
                "2026-07-12T00:00:00", "2026-07-12T03:00:00",
                "2026-07-12T06:00:00", "2026-07-12T09:00:00",
            ),
        )
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        # Same 1-hour-past-t2 query as the _model_values_at_points version.
        sar_time = np.array([np.datetime64("2026-07-12T04:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="hyperbolic",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="hycom", sar_scene_name="scene1",
        )
        assert len(results) == 4
        expected = 0.0 * (1 / 3) ** 2 + 10.0 * (1 / 3) + 10.0
        for r in results:
            assert r.val_data["u10"] == pytest.approx(expected, abs=1e-6)
            assert r.val_data["u10"] != pytest.approx(20.0, abs=1e-6)

    def test_irregular_bracket_spacing_skips_pass_not_fabricated_value(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        era5_ds = _make_era5_ds(
            n_lat=2, n_lon=2,
            hours=(
                "2026-07-12T00:00:00", "2026-07-12T03:00:00",
                "2026-07-12T07:00:00", "2026-07-12T10:00:00",
            ),
        )
        lat_pix = np.linspace(39.8, 42.2, 10)
        lon_pix = np.linspace(-10.2, -7.8, 10)
        sar_lon, sar_lat = np.meshgrid(lon_pix, lat_pix)
        sar_time = np.array([np.datetime64("2026-07-12T04:00:00")])
        sar_data = {"owiWindSpeed": np.full((1, 10, 10), 7.5)}

        colloc = ModelLayerCollocation(
            method="cell-averaging", temporal_method="hyperbolic",
            aggregation_window_km=60.0, distance_weighting="equal",
        )
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="hycom", sar_scene_name="scene1",
        )
        assert results == []


class TestAntimeridianLonHelpers:
    def test_normalize_query_lon_no_op_when_grid_not_stitched(self):
        from sar_validation.core.model_collocation import _normalize_query_lon

        lon_ax = np.array([170.0, 175.0, 180.0])
        query = np.array([-179.0, 172.0])
        result = _normalize_query_lon(query, lon_ax)
        assert np.array_equal(result, query)

    def test_normalize_query_lon_shifts_negative_values_for_stitched_grid(self):
        from sar_validation.core.model_collocation import _normalize_query_lon

        lon_ax = np.array([175.0, 180.0, 182.5, 185.0])  # stitched, max > 180
        query = np.array([-178.0, 177.0])
        result = _normalize_query_lon(query, lon_ax)
        assert result[0] == pytest.approx(182.0)  # -178 + 360
        assert result[1] == pytest.approx(177.0)  # already positive, unchanged

    def test_wrap_lon_to_pm180(self):
        from sar_validation.core.model_collocation import _wrap_lon_to_pm180

        assert _wrap_lon_to_pm180(185.0) == pytest.approx(-175.0)
        assert _wrap_lon_to_pm180(170.0) == pytest.approx(170.0)


class TestModelLayerCollocationAntimeridian:
    def test_individual_grid_interpolates_correctly_across_stitched_axis(self):
        from sar_validation.core.model_collocation import ModelLayerCollocation

        # Stitched ERA5 grid: lon axis [175, 180, 182.5, 185] (originally
        # [175,180] east + [-180,-177.5] shifted to [180,182.5] west --
        # simplified to a 4-point axis for a clean, hand-checkable test.
        lat = np.array([40.0, 41.0])
        lon = np.array([175.0, 180.0, 182.5, 185.0])
        time = pd.to_datetime(["2026-07-12T01:00:00"])
        u10 = np.full((1, 2, 4), 20.0)
        era5_ds = xr.Dataset(
            {"u10": (("time", "lat", "lon"), u10)},
            coords={"time": time, "lat": lat, "lon": lon},
        )

        # SAR pixel at lon=-178 (standard convention) sits at the
        # equivalent stitched-axis position 182 -- squarely inside the
        # stitched grid's coverage, between 180 and 182.5.
        sar_lon = np.array([[-178.0]])
        sar_lat = np.array([[40.5]])
        sar_time = np.array([np.datetime64("2026-07-12T01:00:00")])
        sar_data = {"owiWindSpeed": np.array([[[7.0]]])}

        colloc = ModelLayerCollocation(method="individual", temporal_method="nearest")
        results = colloc.collocate(
            sar_data=sar_data, sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
            era5_ds=era5_ds, val_source="era5", sar_scene_name="scene1",
        )
        assert len(results) == 1
        # Field is spatially uniform (20.0 everywhere) so the interpolated
        # value is exact regardless of exact position within bounds.
        assert results[0].val_data["u10"] == pytest.approx(20.0)
        # Reported lon must stay in the SAR pixel's own standard
        # convention, not the grid's internal shifted axis.
        assert results[0].sar_lon == pytest.approx(-178.0)


class TestDeriveCurrentsRadialProjection:
    def test_noop_without_both_components(self):
        from sar_validation.core.model_collocation import _derive_currents_radial_projection

        values = {"EWCT": np.array([1.0])}
        out = _derive_currents_radial_projection(values, np.array([90.0]))
        assert out is values
        assert "rvlRadVel_projection" not in out

    def test_noop_without_heading(self):
        from sar_validation.core.model_collocation import _derive_currents_radial_projection

        values = {"EWCT": np.array([1.0]), "NSCT": np.array([0.0])}
        out = _derive_currents_radial_projection(values, None)
        assert out is values
        assert "rvlRadVel_projection" not in out

    def test_projection_matches_existing_collocation_py_formula(self):
        from sar_validation.core.collocation import _project_currents_to_radial
        from sar_validation.core.model_collocation import _derive_currents_radial_projection

        ewct = np.array([1.5, -0.5])
        nsct = np.array([0.3, 0.8])
        heading = np.array([45.0, 200.0])

        out = _derive_currents_radial_projection({"EWCT": ewct, "NSCT": nsct}, heading)

        expected = _project_currents_to_radial(ewct, nsct, heading)
        np.testing.assert_allclose(out["rvlRadVel_projection"], expected)

    def test_ewct_nsct_are_kept_not_dropped(self):
        from sar_validation.core.model_collocation import _derive_currents_radial_projection

        ewct = np.array([1.0])
        nsct = np.array([2.0])
        out = _derive_currents_radial_projection({"EWCT": ewct, "NSCT": nsct}, np.array([0.0]))
        assert "EWCT" in out and "NSCT" in out

    def test_scalar_heading_and_values_also_work(self):
        from sar_validation.core.model_collocation import _derive_currents_radial_projection

        out = _derive_currents_radial_projection({"EWCT": 1.0, "NSCT": 0.5}, 30.0)
        assert "rvlRadVel_projection" in out
        assert isinstance(out["rvlRadVel_projection"], float)
