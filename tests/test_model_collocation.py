"""Tests for model_collocation.py -- ERA5/SAR bilinear + temporal interpolation."""

from __future__ import annotations

import numpy as np
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
