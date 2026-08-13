"""Tests for orbit_coverage.py's satellite registry and spherical-geometry math."""

from __future__ import annotations

import math

import pytest

from sar_validation.core.orbit_coverage import (
    SATELLITE_ORBIT_SPECS,
    SatelliteOrbitSpec,
    _bearing_deg,
    _destination_point,
)

_ONE_DEGREE_KM = math.radians(1.0) * 6371.0  # exact distance for a 1-degree great-circle step


class TestSatelliteOrbitSpecs:
    def test_metop_b_and_c_registered_with_real_norad_ids(self):
        assert SATELLITE_ORBIT_SPECS["metop-b"] == SatelliteOrbitSpec(norad_id=38771, swath_half_width_km=600.0)
        assert SATELLITE_ORBIT_SPECS["metop-c"] == SatelliteOrbitSpec(norad_id=43689, swath_half_width_km=600.0)

    def test_metop_a_not_registered(self):
        assert "metop-a" not in SATELLITE_ORBIT_SPECS


class TestBearingDeg:
    def test_due_north_is_zero(self):
        assert _bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_due_east_is_ninety(self):
        assert _bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=1e-6)

    def test_due_south_is_180(self):
        assert _bearing_deg(10.0, 0.0, 0.0, 0.0) == pytest.approx(180.0, abs=1e-6)

    def test_due_west_is_270(self):
        assert _bearing_deg(0.0, 10.0, 0.0, 0.0) == pytest.approx(270.0, abs=1e-6)


class TestDestinationPoint:
    def test_zero_distance_returns_start_point(self):
        lat, lon = _destination_point(45.0, -10.0, 90.0, 0.0)
        assert lat == pytest.approx(45.0, abs=1e-9)
        assert lon == pytest.approx(-10.0, abs=1e-9)

    def test_due_north_increases_latitude_by_exactly_one_degree(self):
        lat, lon = _destination_point(0.0, 0.0, 0.0, _ONE_DEGREE_KM)
        assert lat == pytest.approx(1.0, abs=1e-9)
        assert lon == pytest.approx(0.0, abs=1e-9)

    def test_due_east_at_equator_increases_longitude_by_exactly_one_degree(self):
        lat, lon = _destination_point(0.0, 0.0, 90.0, _ONE_DEGREE_KM)
        assert lat == pytest.approx(0.0, abs=1e-9)
        assert lon == pytest.approx(1.0, abs=1e-9)

    def test_round_trip_bearing_matches_destination(self):
        """Travelling from a point along bearing theta for a distance,
        then computing the bearing from the start to the resulting
        point, must recover theta -- self-consistency between the two
        formulas, at a real mid-latitude/short-range value."""
        lat0, lon0, bearing, dist = 40.0, -30.0, 37.0, 500.0
        lat1, lon1 = _destination_point(lat0, lon0, bearing, dist)
        recovered_bearing = _bearing_deg(lat0, lon0, lat1, lon1)
        assert recovered_bearing == pytest.approx(bearing, abs=1e-6)
