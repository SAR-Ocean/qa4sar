"""Tests for model_collocation.py -- ERA5/SAR bilinear + temporal interpolation."""

from __future__ import annotations

import numpy as np


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
