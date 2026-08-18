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


class TestPointInPolygon:
    # A 10x10 degree square: lat 0-10, lon 0-10.
    _SQUARE = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]

    def test_point_inside_simple_square(self):
        assert orbit_coverage._point_in_polygon(5.0, 5.0, self._SQUARE) is True

    def test_point_outside_simple_square(self):
        assert orbit_coverage._point_in_polygon(15.0, 15.0, self._SQUARE) is False
        assert orbit_coverage._point_in_polygon(5.0, 15.0, self._SQUARE) is False
        assert orbit_coverage._point_in_polygon(-5.0, 5.0, self._SQUARE) is False

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
        must return a single window approximately equal to the input."""
        self._patch_tle(monkeypatch)
        start = _METOP_B_TLE_EPOCH + timedelta(seconds=60)
        end = _METOP_B_TLE_EPOCH + timedelta(seconds=120)

        windows = orbit_coverage.orbit_overlap_windows(
            "metop-b", start, end,
            min_lon=106.0, max_lon=107.0, min_lat=5.0, max_lat=5.6,
        )

        assert len(windows) == 1
        w_start, w_end = windows[0]
        assert start <= w_start <= end
        assert start <= w_end <= end

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
        orbit_overlaps_bbox's plain boolean."""
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

        assert len(windows) == 2

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
