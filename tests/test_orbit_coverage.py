"""Tests for orbit_coverage.py's satellite registry and spherical-geometry math."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import sar_validation.core.orbit_coverage as orbit_coverage
from sar_validation.core.orbit_coverage import (
    SATELLITE_ORBIT_SPECS,
    SatelliteOrbitSpec,
    _bearing_deg,
    _destination_point,
)

_ONE_DEGREE_KM = math.radians(1.0) * 6371.0  # exact distance for a 1-degree great-circle step


@pytest.fixture(autouse=True)
def _reset_space_track_module_state():
    """orbit_coverage caches an authenticated requests.Session and a
    circuit-breaker flag at MODULE level (see get_tle) so they persist
    across calls within the same process -- exactly the point of Fix 5.
    But that means they'd otherwise also leak across unrelated tests
    (and even unrelated test files) within the same pytest session.
    Reset both before and after every test in this module."""
    orbit_coverage._cached_space_track_session = None
    orbit_coverage._space_track_unavailable = False
    yield
    orbit_coverage._cached_space_track_session = None
    orbit_coverage._space_track_unavailable = False


class TestSatelliteOrbitSpecs:
    def test_metop_b_and_c_registered_with_real_norad_ids(self):
        assert SATELLITE_ORBIT_SPECS["metop-b"] == SatelliteOrbitSpec(norad_id=38771, swath_half_width_km=600.0)
        assert SATELLITE_ORBIT_SPECS["metop-c"] == SatelliteOrbitSpec(norad_id=43689, swath_half_width_km=600.0)

    def test_metop_a_not_registered(self):
        assert "metop-a" not in SATELLITE_ORBIT_SPECS

    def test_hy2_and_oceansat3_registered(self):
        assert SATELLITE_ORBIT_SPECS["hy2b"] == SatelliteOrbitSpec(norad_id=43655, swath_half_width_km=900.0)
        assert SATELLITE_ORBIT_SPECS["hy2c"] == SatelliteOrbitSpec(norad_id=46469, swath_half_width_km=900.0)
        assert SATELLITE_ORBIT_SPECS["oceansat3"] == SatelliteOrbitSpec(norad_id=54361, swath_half_width_km=720.0)

    def test_gcom_w1_and_smos_registered(self):
        assert SATELLITE_ORBIT_SPECS["gcom-w1"] == SatelliteOrbitSpec(norad_id=38337, swath_half_width_km=800.0)
        assert SATELLITE_ORBIT_SPECS["smos"] == SatelliteOrbitSpec(norad_id=36036, swath_half_width_km=525.0)

    def test_sentinel1_a_b_c_registered(self):
        assert SATELLITE_ORBIT_SPECS["sentinel-1a"] == SatelliteOrbitSpec(norad_id=39634, swath_half_width_km=250.0)
        assert SATELLITE_ORBIT_SPECS["sentinel-1b"] == SatelliteOrbitSpec(norad_id=41456, swath_half_width_km=250.0)
        assert SATELLITE_ORBIT_SPECS["sentinel-1c"] == SatelliteOrbitSpec(norad_id=62261, swath_half_width_km=250.0)


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

    def test_two_cache_misses_reuse_one_session_not_two(self, monkeypatch, tmp_path):
        """H-SAF's motivating scenario can involve hundreds of files (and
        so hundreds of cache misses) per run -- _space_track_session (a
        real login POST) must only be called once per process, not once
        per cache miss."""
        self._patch_auth_and_session(monkeypatch)
        session_mock = MagicMock(wraps=lambda username, password: object())
        monkeypatch.setattr(orbit_coverage, "_space_track_session", session_mock)
        candidate = {"EPOCH": "2026-06-08T11:00:00.000000", "TLE_LINE1": "1 X", "TLE_LINE2": "2 X"}
        monkeypatch.setattr(
            orbit_coverage, "_query_nearest_candidate",
            lambda session, norad_id, tt, before: (candidate if before else None),
        )

        orbit_coverage.get_tle(self.SATELLITE, datetime(2026, 6, 8, 12), cache_dir=tmp_path)
        orbit_coverage.get_tle(self.SATELLITE, datetime(2026, 6, 9, 12), cache_dir=tmp_path)

        assert session_mock.call_count == 1

    def test_session_establishment_failure_trips_circuit_breaker(self, monkeypatch, tmp_path):
        """After the first get_tle call fails while establishing a
        session/connection (not a per-request "no TLE for this date"
        outcome), every subsequent call in this process must fail
        immediately (TleFetchError, fail open for callers) WITHOUT
        attempting any further network I/O -- not re-authenticate, not
        re-query -- so a Space-Track outage doesn't block for the full
        request timeout on every remaining file in the run."""
        monkeypatch.setattr(
            "sar_validation.downloaders.base.authenticate_space_track",
            lambda *a, **k: ("user", "pass"),
        )
        session_mock = MagicMock(side_effect=RuntimeError("connection refused"))
        monkeypatch.setattr(orbit_coverage, "_space_track_session", session_mock)
        query_mock = MagicMock()
        monkeypatch.setattr(orbit_coverage, "_query_nearest_candidate", query_mock)

        with pytest.raises(orbit_coverage.TleFetchError):
            orbit_coverage.get_tle(self.SATELLITE, datetime(2026, 6, 8, 12), cache_dir=tmp_path)
        assert session_mock.call_count == 1
        assert query_mock.call_count == 0

        # A fresh cache miss (different date) -- would normally attempt a
        # new session/query, but the circuit breaker must short-circuit
        # it before any network I/O is attempted.
        with pytest.raises(orbit_coverage.TleFetchError):
            orbit_coverage.get_tle(self.SATELLITE, datetime(2026, 6, 9, 12), cache_dir=tmp_path)
        assert session_mock.call_count == 1
        assert query_mock.call_count == 0


_METOP_B_TLE = (
    "1 38771U 12049A   26224.57256041  .00000038  00000+0  37173-4 0  9993",
    "2 38771  98.6465 274.9355 0002624 156.2537 203.8762 14.21452520721277",
)
_METOP_B_TLE_EPOCH = datetime(2026, 8, 12, 13, 44, 29, 219424)


class TestOrbitOverlapsBbox:
    def _patch_tle(self, monkeypatch):
        monkeypatch.setattr(
            orbit_coverage, "get_tle", lambda satellite, target_time, cache_dir=None: _METOP_B_TLE,
        )

    def test_bbox_on_real_ground_track_overlaps(self, monkeypatch):
        """A bbox placed directly around a real, live-computed ground-track
        point (lat=5.297, lon=106.611 at TLE epoch + 90s) must overlap."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", start, end,
            min_lon=106.0, max_lon=107.0, min_lat=5.0, max_lat=5.6,
        ) is True

    def test_bbox_on_opposite_side_of_globe_does_not_overlap(self, monkeypatch):
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        # Antipode of (lat~5.3, lon~106.6) is roughly (lat~-5.3, lon~-73.4).
        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", start, end,
            min_lon=-75.0, max_lon=-72.0, min_lat=-7.0, max_lat=-4.0,
        ) is False

    def test_off_track_swath_point_overlaps_but_naive_same_lon_offset_would_have_missed_it(
        self, monkeypatch,
    ):
        """A real point 400km off the ground track (well inside
        swath_half_width_km=600) at the real computed heading must
        overlap -- proving the heading-aware sweep, not just the
        ground-track center line, drives the result. The bbox is placed
        where the REAL swath edge is (lon~110.14), not where a naive
        same-longitude offset would incorrectly place it (lon~106.61,
        ~3.5 degrees / ~390km away) -- a naive implementation would
        report no overlap for this bbox."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", start, end,
            min_lon=109.8, max_lon=110.5, min_lat=5.8, max_lat=6.3,
            margin_km=50.0,
        ) is True

    def test_far_beyond_swath_and_margin_does_not_overlap(self, monkeypatch):
        """A bbox 2000km off-track (well beyond swath_half_width_km=600 +
        even a generous margin) must not overlap."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", start, end,
            min_lon=130.0, max_lon=133.0, min_lat=5.0, max_lat=6.0,
            margin_km=100.0,
        ) is False

    def test_unregistered_satellite_fails_open(self):
        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-a", _METOP_B_TLE_EPOCH, _METOP_B_TLE_EPOCH + timedelta(minutes=3),
            min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        ) is True

    def test_tle_fetch_error_fails_open(self, monkeypatch):
        def _raise(*a, **k):
            raise orbit_coverage.TleFetchError("no TLE available")
        monkeypatch.setattr(orbit_coverage, "get_tle", _raise)

        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", _METOP_B_TLE_EPOCH, _METOP_B_TLE_EPOCH + timedelta(minutes=3),
            min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        ) is True

    def test_degenerate_zero_duration_window_fails_open(self, monkeypatch):
        """sensing_start == sensing_end is a degenerate window with only
        one ground-track sample -- no adjacent sample exists to derive a
        heading from, so the cross-track sweep can't be predicted. This
        module's entire contract is "never risk a false negative": the
        inability to predict a heading must fail OPEN (return True), not
        silently drop the file by falling through to `return False`. Uses
        a bbox nowhere near the satellite's real ground track to prove
        this is fail-open behavior, not a coincidental real overlap."""
        self._patch_tle(monkeypatch)

        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", _METOP_B_TLE_EPOCH, _METOP_B_TLE_EPOCH,
            min_lon=-170.0, max_lon=-160.0, min_lat=-60.0, max_lat=-50.0,
        ) is True

    def test_unexpected_propagation_error_fails_open(self, monkeypatch):
        self._patch_tle(monkeypatch)
        monkeypatch.setattr(
            "pyorbital.orbital.Orbital.get_lonlatalt",
            lambda self, t: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        assert orbit_coverage.orbit_overlaps_bbox(
            "metop-b", _METOP_B_TLE_EPOCH, _METOP_B_TLE_EPOCH + timedelta(minutes=3),
            min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        ) is True
