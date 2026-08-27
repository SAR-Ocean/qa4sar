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
    _haversine_distance_km,
    _point_to_great_circle_segment_distance_km,
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


class TestHaversineDistanceKm:
    def test_same_point_is_zero(self):
        assert _haversine_distance_km(12.0, -34.0, 12.0, -34.0) == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_of_latitude_matches_known_great_circle_distance(self):
        assert _haversine_distance_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(_ONE_DEGREE_KM, abs=1e-6)

    def test_agrees_with_destination_point_round_trip(self):
        """A point reached by travelling a known distance from a start
        point must report that same distance back via haversine --
        cross-checks the two independent formulas against each other."""
        lat0, lon0, bearing, dist = 40.0, -30.0, 37.0, 500.0
        lat1, lon1 = _destination_point(lat0, lon0, bearing, dist)
        assert _haversine_distance_km(lat0, lon0, lat1, lon1) == pytest.approx(dist, abs=1e-6)


class TestPointToGreatCircleSegmentDistanceKm:
    """A segment from (0, 0) to (0, 10) along the equator -- known
    landmarks along and around it exercise each of the three cases:
    closest point within the segment, beyond its end, and beyond its
    start."""

    _SEG_START = (0.0, 0.0)
    _SEG_END = (0.0, 10.0)

    def test_point_on_the_segment_is_zero(self):
        d = _point_to_great_circle_segment_distance_km(0.0, 5.0, *self._SEG_START, *self._SEG_END)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_point_perpendicular_to_midpoint_matches_known_distance(self):
        """A point 1 degree of latitude directly off the segment's own
        midpoint is much closer to the segment (perpendicular distance)
        than to either endpoint -- the segment-distance calculation must
        report that shorter perpendicular distance, not the endpoint
        distance."""
        d = _point_to_great_circle_segment_distance_km(1.0, 5.0, *self._SEG_START, *self._SEG_END)
        assert d == pytest.approx(_ONE_DEGREE_KM, abs=1.0)
        d_to_start = _haversine_distance_km(1.0, 5.0, *self._SEG_START)
        assert d < d_to_start

    def test_point_beyond_the_end_uses_distance_to_end_not_infinite_line(self):
        """A point further east than the segment's own end must report
        distance to that end -- not the ~zero cross-track distance the
        infinite great circle through both endpoints would give (since
        this point still sits on the same equatorial great circle)."""
        beyond_end = (0.0, 15.0)
        d = _point_to_great_circle_segment_distance_km(*beyond_end, *self._SEG_START, *self._SEG_END)
        expected = _haversine_distance_km(*beyond_end, *self._SEG_END)
        assert d == pytest.approx(expected, abs=1e-6)
        assert d > 100.0  # not the near-zero on-great-circle distance

    def test_point_beyond_the_start_uses_distance_to_start(self):
        beyond_start = (0.0, -5.0)
        d = _point_to_great_circle_segment_distance_km(*beyond_start, *self._SEG_START, *self._SEG_END)
        expected = _haversine_distance_km(*beyond_start, *self._SEG_START)
        assert d == pytest.approx(expected, abs=1e-6)

    def test_degenerate_zero_length_segment_falls_back_to_point_distance(self):
        d = _point_to_great_circle_segment_distance_km(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        expected = _haversine_distance_km(1.0, 1.0, 0.0, 0.0)
        assert d == pytest.approx(expected, abs=1e-6)


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


class TestPointInPolygon:
    # A 10x10 degree square: lat 0-10, lon 0-10.
    _SQUARE = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]

    def test_point_inside_simple_square(self):
        assert orbit_coverage._point_in_polygon(5.0, 5.0, self._SQUARE) is True

    def test_point_outside_simple_square(self):
        assert orbit_coverage._point_in_polygon(15.0, 15.0, self._SQUARE) is False
        assert orbit_coverage._point_in_polygon(5.0, 15.0, self._SQUARE) is False
        assert orbit_coverage._point_in_polygon(-5.0, 5.0, self._SQUARE) is False

    def test_degenerate_polygon_fails_open(self):
        """Fewer than 3 vertices isn't a real polygon -- this module's
        fail-toward-inclusion convention means _point_in_polygon must
        return True (not raise, not silently misbehave) rather than fail
        closed, since Plan 3 imports this function directly with no
        surrounding try/except of its own."""
        assert orbit_coverage._point_in_polygon(5.0, 5.0, []) is True
        assert orbit_coverage._point_in_polygon(5.0, 5.0, [(0.0, 0.0), (1.0, 1.0)]) is True

    def test_point_near_but_outside_a_corner(self):
        """The specific case this function exists for: a point that
        would pass a bbox-corner check but is outside the true polygon
        shape -- a triangle occupying only half of its own bounding
        box's square."""
        triangle = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0)]  # right triangle
        # (8, 8) is inside the triangle's bbox (0-10, 0-10) but outside
        # the triangle itself (above the hypotenuse lat+lon=10).
        assert orbit_coverage._point_in_polygon(8.0, 8.0, triangle) is False
        # (2, 2) is inside both the bbox and the triangle.
        assert orbit_coverage._point_in_polygon(2.0, 2.0, triangle) is True

    def test_antimeridian_crossing_polygon(self):
        """A footprint spanning the dateline: lon 170 to -170 (a 20-degree
        span through 180), lat 0-10. Points on both sides of the seam
        must be recognized as inside; a point nowhere near the seam must
        not."""
        polygon = [(0.0, 170.0), (0.0, -170.0), (10.0, -170.0), (10.0, 170.0)]
        assert orbit_coverage._point_in_polygon(5.0, 175.0, polygon) is True   # east side
        assert orbit_coverage._point_in_polygon(5.0, -175.0, polygon) is True  # west side
        assert orbit_coverage._point_in_polygon(5.0, 0.0, polygon) is False    # far away
        assert orbit_coverage._point_in_polygon(5.0, 90.0, polygon) is False   # far away


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


class TestOrbitOverlapWindows:
    def _patch_tle(self, monkeypatch):
        monkeypatch.setattr(
            orbit_coverage, "get_tle", lambda satellite, target_time, cache_dir=None: _METOP_B_TLE,
        )

    def test_narrow_window_on_real_ground_track_returns_one_window_matching_input(self, monkeypatch):
        """A window already narrow enough that the whole thing overlaps
        (mirrors TestOrbitOverlapsBbox.test_bbox_on_real_ground_track_overlaps)
        must return a single window approximately equal to the input.

        Only the sample at EPOCH+90s (lat~5.297, lon~106.611) actually
        falls in the bbox; with the default sample_interval_s=15.0, the
        returned window is that single matching sample's timestamp
        padded by 15s on each side -- EPOCH+75s to EPOCH+105s -- not the
        raw, un-padded sample timestamp (which would make this a
        zero-duration window). Pinned via a real run of this exact TLE/
        bbox/window combination, not a hand derivation."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end,
            min_lon=106.0, max_lon=107.0, min_lat=5.0, max_lat=5.6,
        )

        assert windows == [
            (_METOP_B_TLE_EPOCH + timedelta(seconds=75), _METOP_B_TLE_EPOCH + timedelta(seconds=105)),
        ]

    def test_no_overlap_anywhere_returns_empty_list(self, monkeypatch):
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end,
            min_lon=-75.0, max_lon=-72.0, min_lat=-7.0, max_lat=-4.0,
        )

        assert windows == []

    def test_unregistered_satellite_fails_open_with_whole_window(self):
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(minutes=3)

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-a", start, end, min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        )

        assert windows == [(start, end)]

    def test_tle_fetch_error_fails_open_with_whole_window(self, monkeypatch):
        def _raise(*a, **k):
            raise orbit_coverage.TleFetchError("no TLE available")
        monkeypatch.setattr(orbit_coverage, "get_tle", _raise)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(minutes=3)

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end, min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        )

        assert windows == [(start, end)]

    def test_degenerate_zero_duration_window_fails_open_with_whole_window(self, monkeypatch):
        self._patch_tle(monkeypatch)
        t = _METOP_B_TLE_EPOCH

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", t, t, min_lon=-170.0, max_lon=-160.0, min_lat=-60.0, max_lat=-50.0,
        )

        assert windows == [(t, t)]

    def test_whole_day_window_returns_multiple_disjoint_sub_windows(self, monkeypatch):
        """A synthetic satellite whose ground track sweeps past the same
        small bbox twice during a longer window must return two disjoint
        sub-windows, not one covering the whole span -- this is
        specifically what orbit_overlap_windows exists to expose over
        orbit_overlaps_bbox's plain boolean.

        Samples land at t=0,30,...,180s. Samples at t=0,30s match (inside
        the bbox); t=60,90s don't (far away); t=120,150,180s match again.
        Each padded by sample_interval_s=30s on each side and clamped to
        [start, end]: first window (0-30s matches) pads to [0-30, 30+30]
        clamped to [0, 60]; second window (120-180s matches) pads to
        [120-30, 180+30] clamped to [90, 180]. Pinned via a real run of
        this exact synthetic scenario, not a hand derivation."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=180)

        # Real ground track at start+60s..+90s is near (lat~5.3, lon~106.6)
        # (see TestOrbitOverlapsBbox). Build a synthetic longitude track
        # via monkeypatch that revisits that neighborhood twice: once
        # early, once late, with real METOP-B geometry ignored in favor
        # of a fully controlled synthetic path.
        visits = [
            (5.3, 106.6), (5.3, 106.6),   # samples 0-1: inside
            (20.0, 106.6), (20.0, 106.6), # samples 2-3: far away (outside)
            (5.3, 106.6), (5.3, 106.6),   # samples 4-5: inside again
        ]

        def _fake_lonlatalt(self, t):
            idx = min(
                int((t - start).total_seconds() // 30), len(visits) - 1,
            )
            lat, lon = visits[idx]
            return lon, lat, 800.0

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end,
            min_lon=106.0, max_lon=107.0, min_lat=5.0, max_lat=5.6,
            sample_interval_s=30.0,
        )

        assert windows == [
            (start, start + timedelta(seconds=60)),
            (start + timedelta(seconds=90), end),
        ]

    def test_polygon_excludes_a_point_the_bbox_alone_would_accept(self, monkeypatch):
        """Precision test: a synthetic ground-track sample lands inside
        the target bbox but outside a smaller polygon within it -- with
        polygon=None the bbox alone must match; with polygon supplied,
        that same sample must be correctly rejected."""

        def _fake_lonlatalt(self, t):
            # Constant sub-satellite point at (lat=5.0, lon=5.0) for
            # every sample -- deliberately inside the bbox below but
            # outside the small polygon below.
            return 5.0, 5.0, 800.0

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)
        monkeypatch.setattr(
            orbit_coverage, "get_tle", lambda satellite, target_time, cache_dir=None: _METOP_B_TLE,
        )
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=15)
        bbox_kwargs = dict(min_lon=0.0, max_lon=10.0, min_lat=0.0, max_lat=10.0)
        small_polygon = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]  # excludes (5, 5)

        without_polygon = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end, **bbox_kwargs,
        )
        with_polygon = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end, **bbox_kwargs, polygon=small_polygon,
        )

        assert without_polygon == [(start, end)]
        assert with_polygon == []

    def test_wrap_convention_bbox_with_antimeridian_polygon_matches_correctly(self, monkeypatch):
        """Pins the CORRECT usage documented in _region_contains's and
        orbit_overlap_windows's docstrings: when polygon crosses the
        antimeridian, min_lon/max_lon must use the wrap convention
        (min_lon > max_lon), NOT a naive min(lons)/max(lons) taken over
        the polygon's own vertices. Reuses the same wrapping polygon as
        TestPointInPolygon.test_antimeridian_crossing_polygon (lon 170 to
        -170, lat 0-10) paired with a bbox given in the wrap convention
        (min_lon=170.0, max_lon=-170.0). A sample point at lon=175
        (inside both the wrap-convention bbox and the polygon) must be
        matched, not silently excluded."""

        def _fake_lonlatalt(self, t):
            # Constant sub-satellite point at (lat=5.0, lon=175.0) --
            # east of the seam, inside both the wrap-convention bbox and
            # the antimeridian-crossing polygon below.
            return 175.0, 5.0, 800.0

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)
        monkeypatch.setattr(
            orbit_coverage, "get_tle", lambda satellite, target_time, cache_dir=None: _METOP_B_TLE,
        )
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=15)
        polygon = [(0.0, 170.0), (0.0, -170.0), (10.0, -170.0), (10.0, 170.0)]

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end,
            min_lon=170.0, max_lon=-170.0, min_lat=0.0, max_lat=10.0,
            polygon=polygon,
        )

        assert windows == [(start, end)]


class TestSampleGroundTrack:
    def _patch_tle(self, monkeypatch):
        monkeypatch.setattr(
            orbit_coverage, "get_tle", lambda satellite, target_time, cache_dir=None: _METOP_B_TLE,
        )

    def test_returns_time_lat_lon_triples_at_the_requested_cadence(self, monkeypatch):
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=60)

        samples = orbit_coverage.sample_ground_track("metop-b", start, end, sample_interval_s=15.0)

        assert [t for t, _lat, _lon in samples] == [
            start,
            start + timedelta(seconds=15),
            start + timedelta(seconds=30),
            start + timedelta(seconds=45),
            start + timedelta(seconds=60),
        ]
        for _t, lat, lon in samples:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0

    def test_trailing_sample_added_when_window_not_evenly_divisible(self, monkeypatch):
        """Mirrors orbit_overlap_windows' own sampling: a window whose
        length isn't an exact multiple of sample_interval_s still gets a
        sample exactly at sensing_end, not just truncated early."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=50)  # not a multiple of 15

        samples = orbit_coverage.sample_ground_track("metop-b", start, end, sample_interval_s=15.0)

        assert samples[-1][0] == end
        assert [t for t, _lat, _lon in samples[:-1]] == [
            start, start + timedelta(seconds=15), start + timedelta(seconds=30), start + timedelta(seconds=45),
        ]

    def test_tle_fetch_error_propagates_not_fail_open(self, monkeypatch):
        """Unlike orbit_overlap_windows, sample_ground_track has no target
        region to fail open *about* -- it's the caller's job (e.g.
        orbit_overlap_windows itself) to decide what failing open means."""
        def _raise(*a, **k):
            raise orbit_coverage.TleFetchError("no TLE available")
        monkeypatch.setattr(orbit_coverage, "get_tle", _raise)

        with pytest.raises(orbit_coverage.TleFetchError):
            orbit_coverage.sample_ground_track(
                "metop-b", _METOP_B_TLE_EPOCH, _METOP_B_TLE_EPOCH + timedelta(seconds=60),
            )


class TestMatchGroundTrack:
    """match_ground_track is orbit_overlap_windows' matching half, split
    out to work against an already-propagated samples array -- see
    sample_ground_track's docstring for why. The core contract these
    tests pin: calling match_ground_track against a samples array that
    covers a WIDER span than [sensing_start, sensing_end] must return
    exactly what orbit_overlap_windows would have returned for that
    narrower window alone."""

    def _patch_tle(self, monkeypatch):
        monkeypatch.setattr(
            orbit_coverage, "get_tle", lambda satellite, target_time, cache_dir=None: _METOP_B_TLE,
        )

    def test_matches_orbit_overlap_windows_for_an_equivalent_single_window(self, monkeypatch):
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)
        bbox_kwargs = dict(min_lon=106.0, max_lon=107.0, min_lat=5.0, max_lat=5.6)

        expected = orbit_coverage.orbit_overlap_windows("metop-b", start, end, **bbox_kwargs)

        samples = orbit_coverage.sample_ground_track("metop-b", start, end)
        actual = orbit_coverage.match_ground_track(samples, "metop-b", start, end, **bbox_kwargs)

        assert actual == expected
        assert actual == [
            (_METOP_B_TLE_EPOCH + timedelta(seconds=75), _METOP_B_TLE_EPOCH + timedelta(seconds=105)),
        ]

    def test_shared_wider_samples_array_gives_identical_results_to_independent_calls(self, monkeypatch):
        """The actual point of this split: propagate ONCE over a window
        wide enough to cover two independent, non-overlapping target
        windows, then match_ground_track each of them against that same
        shared array -- results must be identical to what two separate
        orbit_overlap_windows calls (each with its own independent
        propagation) would have produced."""
        self._patch_tle(monkeypatch)
        window_a_start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        window_a_end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)
        window_b_start = _METOP_B_TLE_EPOCH + timedelta(seconds=600)
        window_b_end = _METOP_B_TLE_EPOCH + timedelta(seconds=660)
        bbox_kwargs = dict(min_lon=106.0, max_lon=107.0, min_lat=5.0, max_lat=5.6)

        expected_a = orbit_coverage.orbit_overlap_windows("metop-b", window_a_start, window_a_end, **bbox_kwargs)
        expected_b = orbit_coverage.orbit_overlap_windows("metop-b", window_b_start, window_b_end, **bbox_kwargs)

        shared_samples = orbit_coverage.sample_ground_track("metop-b", window_a_start, window_b_end)
        actual_a = orbit_coverage.match_ground_track(
            shared_samples, "metop-b", window_a_start, window_a_end, **bbox_kwargs,
        )
        actual_b = orbit_coverage.match_ground_track(
            shared_samples, "metop-b", window_b_start, window_b_end, **bbox_kwargs,
        )

        assert actual_a == expected_a
        assert actual_b == expected_b

    def test_context_samples_outside_own_window_never_start_or_end_a_reported_window(self, monkeypatch):
        """A shared samples array reaching well past [sensing_start,
        sensing_end] must never let an out-of-range sample leak into a
        reported window's own start/end, even if that out-of-range
        sample happens to match the target region too."""
        def _fake_lonlatalt(self, t):
            return 5.0, 5.0, 800.0  # every sample matches the bbox below

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)
        self._patch_tle(monkeypatch)
        shared_start = _METOP_B_TLE_EPOCH
        shared_end = _METOP_B_TLE_EPOCH + timedelta(seconds=300)
        sensing_start = _METOP_B_TLE_EPOCH + timedelta(seconds=120)
        sensing_end = _METOP_B_TLE_EPOCH + timedelta(seconds=180)
        bbox_kwargs = dict(min_lon=0.0, max_lon=10.0, min_lat=0.0, max_lat=10.0)

        shared_samples = orbit_coverage.sample_ground_track("metop-b", shared_start, shared_end)
        windows = orbit_coverage.match_ground_track(
            shared_samples, "metop-b", sensing_start, sensing_end, **bbox_kwargs,
        )

        assert len(windows) == 1
        w_start, w_end = windows[0]
        assert sensing_start <= w_start <= w_end <= sensing_end

    def test_unregistered_satellite_fails_open_with_whole_window(self, monkeypatch):
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        samples = orbit_coverage.sample_ground_track("metop-b", start, end)

        windows = orbit_coverage.match_ground_track(
            samples, "metop-a", start, end, min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        )

        assert windows == [(start, end)]

    def test_too_few_in_range_samples_fails_open_with_whole_window(self, monkeypatch):
        """samples covers a window not overlapping [sensing_start,
        sensing_end] at all -- fewer than 2 usable (in-range + context)
        samples, matching orbit_overlap_windows' own degenerate-window
        fail-open behavior."""
        self._patch_tle(monkeypatch)
        samples = orbit_coverage.sample_ground_track(
            "metop-b", _METOP_B_TLE_EPOCH, _METOP_B_TLE_EPOCH + timedelta(seconds=30),
        )
        sensing_start = _METOP_B_TLE_EPOCH + timedelta(hours=1)
        sensing_end = _METOP_B_TLE_EPOCH + timedelta(hours=1, seconds=30)

        windows = orbit_coverage.match_ground_track(
            samples, "metop-b", sensing_start, sensing_end,
            min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
        )

        assert windows == [(sensing_start, sensing_end)]

    def test_target_point_matches_within_swath_plus_margin_ignoring_bbox(self, monkeypatch):
        """A genuine point target (e.g. one WV vignette) has zero area,
        so the bbox/polygon containment sweep can only ever match it on
        an exact floating-point coordinate equality -- effectively never.
        target_point must use a real distance check instead, and must
        ignore min_lon/max_lon/min_lat/max_lat/polygon entirely -- proven
        here by passing a bbox nowhere near the fixed sample position."""
        def _fake_lonlatalt(self, t):
            return 5.0, 5.0, 800.0

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        samples = orbit_coverage.sample_ground_track("metop-b", start, end)

        windows = orbit_coverage.match_ground_track(
            samples, "metop-b", start, end,
            min_lon=50.0, max_lon=60.0, min_lat=50.0, max_lat=60.0,  # nowhere near (5.0, 5.0)
            margin_km=0.0, target_point=(5.0, 5.0),  # exact same point as every sample
        )

        assert windows == [(start, end)]

    def test_target_point_no_match_beyond_swath_plus_margin(self, monkeypatch):
        def _fake_lonlatalt(self, t):
            return 5.0, 5.0, 800.0

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        samples = orbit_coverage.sample_ground_track("metop-b", start, end)

        # metop-b's swath_half_width_km is 600.0; ~7 degrees latitude is
        # ~778km away -- well beyond 600 + 0 margin.
        windows = orbit_coverage.match_ground_track(
            samples, "metop-b", start, end,
            min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
            margin_km=0.0, target_point=(12.0, 5.0),
        )

        assert windows == []

    def test_target_point_matches_a_crossing_between_two_samples(self, monkeypatch):
        """A real crossing can fall entirely between two consecutive
        propagated samples -- close enough to the segment joining them
        to be within margin, yet farther from BOTH sample endpoints
        individually. Checking only distance-to-sample would report no
        match here; the segment check must still catch it.

        Track moves 1 degree of longitude along the equator per 15s
        sample step. The target sits 1 degree of latitude (~111km) north
        of the first segment's own midpoint: ~111km from the segment,
        but ~124km from either endpoint -- a margin between those two
        values matches only via the segment check."""
        monkeypatch.setitem(
            orbit_coverage.SATELLITE_ORBIT_SPECS, "metop-b",
            SatelliteOrbitSpec(norad_id=38771, swath_half_width_km=0.0),
        )

        def _fake_lonlatalt(self, t):
            elapsed_s = (t - _METOP_B_TLE_EPOCH).total_seconds()
            return elapsed_s / 15.0, 0.0, 800.0  # 1 degree longitude per 15s sample

        monkeypatch.setattr("pyorbital.orbital.Orbital.get_lonlatalt", _fake_lonlatalt)
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=15)
        samples = orbit_coverage.sample_ground_track("metop-b", start, end)

        target = (1.0, 0.5)
        d_to_start = _haversine_distance_km(*target, 0.0, 0.0)
        d_to_end = _haversine_distance_km(*target, 0.0, 1.0)
        assert d_to_start > 115.0 and d_to_end > 115.0  # neither endpoint alone is close enough

        windows = orbit_coverage.match_ground_track(
            samples, "metop-b", start, end,
            min_lon=0.0, max_lon=0.0, min_lat=0.0, max_lat=0.0,
            margin_km=115.0, target_point=target,
        )

        assert windows == [(start, end)]
