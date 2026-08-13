"""Tests for orbit_coverage.py's satellite registry and spherical-geometry math."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

import sar_validation.core.orbit_coverage as orbit_coverage
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


class TestGetTle:
    SATELLITE = "metop-b"

    def _patch_auth_and_session(self, monkeypatch):
        monkeypatch.setattr(
            "sar_validation.downloaders.base.authenticate_space_track",
            lambda *a, **k: ("user", "pass"),
        )
        monkeypatch.setattr(
            orbit_coverage, "_space_track_session", lambda username, password: object(),
        )

    def test_selects_nearer_of_before_and_after_candidates(self, monkeypatch, tmp_path):
        target = datetime(2026, 6, 8, 12, 0, 0)
        before_candidate = {
            "EPOCH": "2026-06-07T00:00:00.000000", "TLE_LINE1": "1 BEFORE", "TLE_LINE2": "2 BEFORE",
        }
        after_candidate = {
            "EPOCH": "2026-06-08T18:00:00.000000", "TLE_LINE1": "1 AFTER", "TLE_LINE2": "2 AFTER",
        }
        self._patch_auth_and_session(monkeypatch)

        def _fake_query(session, norad_id, tt, before):
            return before_candidate if before else after_candidate

        monkeypatch.setattr(orbit_coverage, "_query_nearest_candidate", _fake_query)

        line1, line2 = orbit_coverage.get_tle(self.SATELLITE, target, cache_dir=tmp_path)

        # "after" (18:00, ~6h gap) is closer to target (12:00) than
        # "before" (00:00 the previous day, 36h gap).
        assert (line1, line2) == ("1 AFTER", "2 AFTER")

    def test_missing_credentials_raises_tle_fetch_error(self, monkeypatch, tmp_path):
        def _raise(*a, **k):
            raise RuntimeError("Space-Track credentials not found")
        monkeypatch.setattr("sar_validation.downloaders.base.authenticate_space_track", _raise)

        with pytest.raises(orbit_coverage.TleFetchError):
            orbit_coverage.get_tle(self.SATELLITE, datetime(2026, 6, 8), cache_dir=tmp_path)

    def test_epoch_gap_beyond_threshold_raises(self, monkeypatch, tmp_path):
        target = datetime(2026, 6, 8, 12, 0, 0)
        far = {"EPOCH": "2026-05-01T00:00:00.000000", "TLE_LINE1": "1 FAR", "TLE_LINE2": "2 FAR"}
        self._patch_auth_and_session(monkeypatch)
        monkeypatch.setattr(
            orbit_coverage, "_query_nearest_candidate",
            lambda session, norad_id, tt, before: (far if before else None),
        )

        with pytest.raises(orbit_coverage.TleFetchError, match="beyond"):
            orbit_coverage.get_tle(self.SATELLITE, target, cache_dir=tmp_path)

    def test_second_call_for_same_satellite_and_date_hits_cache(self, monkeypatch, tmp_path):
        target = datetime(2026, 6, 8, 12, 0, 0)
        candidate = {"EPOCH": "2026-06-08T11:00:00.000000", "TLE_LINE1": "1 X", "TLE_LINE2": "2 X"}
        self._patch_auth_and_session(monkeypatch)
        call_count = {"n": 0}

        def _fake_query(session, norad_id, tt, before):
            call_count["n"] += 1
            return candidate if before else None

        monkeypatch.setattr(orbit_coverage, "_query_nearest_candidate", _fake_query)

        first = orbit_coverage.get_tle(self.SATELLITE, target, cache_dir=tmp_path)
        second = orbit_coverage.get_tle(self.SATELLITE, target, cache_dir=tmp_path)

        assert first == second == ("1 X", "2 X")
        assert call_count["n"] == 2  # (before + after) queries fired only on the FIRST call

    def test_unregistered_satellite_raises_before_any_network_call(self, tmp_path):
        with pytest.raises(orbit_coverage.TleFetchError, match="Unknown satellite"):
            orbit_coverage.get_tle("metop-a", datetime(2026, 6, 8), cache_dir=tmp_path)
