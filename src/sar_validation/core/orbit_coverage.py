"""
General-purpose, satellite-agnostic orbit-based geographic pre-filter.

Given a satellite name and a sensing time window, predicts whether
that satellite's ground track could have passed near a given
bounding box, so a downloader with no server-side bbox filter can
skip clearly irrelevant files before downloading them.

Also provides a windowed, polygon-aware variant, orbit_overlap_windows,
which returns every matching sub-window (instead of a single bool) and
can test against a true footprint polygon instead of just its bounding
box -- needed for wide-window sources (e.g. AMSR2's whole-day window)
where "yes, sometime today" alone isn't useful.

Fails open on any prediction failure (unregistered satellite,
unavailable TLE, propagation error), since this is only a pre-filter
to reduce unnecessary downloads, not a substitute for the real
geographic filtering applied downstream, and must never risk a
false negative.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = [
    "SatelliteOrbitSpec",
    "SATELLITE_ORBIT_SPECS",
    "TleFetchError",
    "get_tle",
    "orbit_overlaps_bbox",
    "orbit_overlap_windows",
]

_EARTH_RADIUS_KM = 6371.0

#: A cached TLE is keyed to a fixed historical (satellite, date) pair, so
#: once fetched it is immutable -- no max-age eviction needed (unlike a
#: "current TLE" cache would need). See get_tle().
_MAX_ACCEPTABLE_EPOCH_GAP_DAYS = 10.0

_SPACE_TRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
_SPACE_TRACK_QUERY_BASE = "https://www.space-track.org/basicspacedata/query/class/gp_history"

_DEFAULT_TLE_CACHE_DIR = Path("data") / "_archive_cache" / "tle"

#: Authenticated Space-Track session, cached at module level and reused
#: across every get_tle() call.
_cached_space_track_session: Optional[requests.Session] = None

#: Circuit breaker scoped to this process's lifetime (module-level,
#: no expiry/reset/persistence). Set the first time get_tle hits an
#: exception while authenticating or establishing/using a
#: Space-Track session -- not for a normal "no TLE found for this
#: date" outcome, which is handled separately via the
#: empty-``candidates`` check and must not trip this. Once set, every
#: subsequent get_tle call fails open immediately (TleFetchError)
#: without further network I/O, so a Space-Track outage does not
#: block for the full request timeout on every remaining file in the
#: run.
_space_track_unavailable: bool = False


@dataclass(frozen=True)
class SatelliteOrbitSpec:
    """
    One satellite this module can predict ground-track/swath overlap for.

    ``norad_id`` feeds the Space-Track TLE lookup (see :func:`get_tle`). 
    ``swath_half_width_km`` is the outer edge of the satellite's real 
    swath from its ground track, in kilometers. For a satellite with 
    two side-looking swaths, the inner gap between them is not modeled; 
    treating both sides as one continuous corridor out to the outer edge 
    is an over-inclusion.
    """

    norad_id: int
    swath_half_width_km: float

#: NORAD catalog IDs (CelesTrak numbering; Space-Track uses the same
#: scheme). Add entries here (not in any per-source downloader file)
#: to extend orbit pre-filtering to a new satellite.
SATELLITE_ORBIT_SPECS: Dict[str, SatelliteOrbitSpec] = {
    "metop-b": SatelliteOrbitSpec(norad_id=38771, swath_half_width_km=600.0),
    "metop-c": SatelliteOrbitSpec(norad_id=43689, swath_half_width_km=600.0),
    # HY-2B/HY-2C HSCAT (Ku-band rotating pencil-beam scatterometer).
    "hy2b": SatelliteOrbitSpec(norad_id=43655, swath_half_width_km=900.0),
    "hy2c": SatelliteOrbitSpec(norad_id=46469, swath_half_width_km=900.0),
    # Oceansat-3/EOS-06 OSCAT-3 (Ku-band conical-scan scatterometer).
    "oceansat3": SatelliteOrbitSpec(norad_id=54361, swath_half_width_km=720.0),
    # GCOM-W1 "Shizuku" -- AMSR2's host satellite (conical scanner).
    "gcom-w1": SatelliteOrbitSpec(norad_id=38337, swath_half_width_km=800.0),
    # SMOS (MIRAS interferometric radiometer, hexagonal FOV).
    "smos": SatelliteOrbitSpec(norad_id=36036, swath_half_width_km=525.0),
    # Sentinel-1A/B/C (C-SAR, IW mode). NORAD IDs (SATCAT numbers)
    # live-confirmed via Wikipedia's Sentinel-1A/1B/1C pages, 2026-08-18:
    # 1A=39634 (launched 2014-04-03), 1B=41456 (launched 2016-04-25,
    # deactivated 2022-08-03 -- kept registered anyway, since a recipe can
    # validate SAR data from before that date), 1C=62261 (launched
    # 2024-12-05). Swath width: IW mode's documented 250km (ESA Sentinel
    # Online user guide, sentinel.esa.int, 2026-08-18) -- Sentinel-1's SAR
    # instrument images one side of the ground track only (unlike
    # ASCAT/HY-2's genuinely two-sided swaths), so treating 250km as the
    # per-side half-width (checking both sides via this module's existing
    # bearing +/- 90 sweep) is a deliberate, conservative over-inclusion,
    # consistent with every other entry in this dict.
    "sentinel-1a": SatelliteOrbitSpec(norad_id=39634, swath_half_width_km=250.0),
    "sentinel-1b": SatelliteOrbitSpec(norad_id=41456, swath_half_width_km=250.0),
    "sentinel-1c": SatelliteOrbitSpec(norad_id=62261, swath_half_width_km=250.0),
}


class TleFetchError(Exception):
    """
    Raised when no usable historical TLE can be obtained for a given
    satellite/time -- callers (orbit_overlaps_bbox and
    orbit_overlap_windows) must treat this as "cannot predict, fail
    open", never propagate it as a hard error.
    """


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Initial great-circle bearing (degrees, 0-360, 0=north, clockwise)
    from (lat1, lon1) to (lat2, lon2). Standard spherical-Earth forward-
    azimuth formula (Movable Type Scripts' "Bearing" reference).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360.0) % 360.0


def _destination_point(
    lat: float, lon: float, bearing_deg: float, distance_km: float,
) -> Tuple[float, float]:
    """
    (lat, lon) reached by travelling distance_km along bearing_deg
    (great-circle) from (lat, lon), spherical Earth. Standard "direct
    geodesic" formula (Movable Type Scripts' "Destination point given
    distance and bearing from start point"):

        delta = distance_km / _EARTH_RADIUS_KM
        phi2 = asin(sin(phi1)*cos(delta) + cos(phi1)*sin(delta)*cos(theta))
        lambda2 = lambda1 + atan2(sin(theta)*sin(delta)*cos(phi1),
                                   cos(delta) - sin(phi1)*sin(phi2))
    """
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    theta = math.radians(bearing_deg)
    delta = distance_km / _EARTH_RADIUS_KM

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    lon2 = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0  # normalize to [-180, 180)
    return math.degrees(phi2), lon2


def _cache_path(cache_dir: Path, satellite: str, day) -> Path:
    return cache_dir / satellite / f"{day.isoformat()}.json"


def _space_track_session(username: str, password: str) -> requests.Session:
    session = requests.Session()
    resp = session.post(
        _SPACE_TRACK_LOGIN_URL, data={"identity": username, "password": password}, timeout=30,
    )
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("Login") == "Failed":
        raise TleFetchError("Space-Track authentication failed -- check username/password.")
    return session


def _query_nearest_candidate(
    session: requests.Session, norad_id: int, target_time: datetime, before: bool,
) -> Optional[Dict[str, str]]:
    """
    One gp_history record: the single nearest-EPOCH candidate strictly
    before (before=True) or strictly after (before=False) target_time,
    or None if the historical archive has nothing on that side.

    Uses a plain "<"/">" rather than ">=": Space-Track's query parser
    rejects the "=" character in a predicate value with an HTTP 400
    ("The URI you submitted has disallowed characters"). This loses
    only the negligible edge case of a TLE epoch landing on the exact
    microsecond of target_time.
    """
    target_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
    operator = "<" if before else ">"
    direction = "desc" if before else "asc"
    url = (
        f"{_SPACE_TRACK_QUERY_BASE}/NORAD_CAT_ID/{norad_id}"
        f"/EPOCH/{operator}{target_str}/orderby/EPOCH {direction}/limit/1/format/json"
    )
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        return None
    return records[0]


def _epoch_gap_seconds(candidate: Dict[str, str], target_time: datetime) -> float:
    epoch = pd.Timestamp(candidate["EPOCH"]).to_pydatetime()
    if epoch.tzinfo is not None:
        epoch = epoch.astimezone(timezone.utc).replace(tzinfo=None)
    return abs((epoch - target_time).total_seconds())


def get_tle(
    satellite: str, target_time: datetime, cache_dir: Optional[Path] = None,
) -> "tuple[str, str]":
    """
    (line1, line2) for *satellite*, nearest in epoch to *target_time*.

    Cached on disk keyed by (satellite, target_time.date()): since the
    result is "the historical TLE nearest this fixed past date", it never
    changes once fetched, so the cache entry is permanent. 

    On a cache miss, authenticates to Space-Track (credentials via
    authenticate_space_track(), see downloaders/base.py), and queries the
    gp_history class for the NORAD ID nearest target_time, both before
    and after (target_time is always in the past for this use case), 
    picking whichever candidate has the smaller |epoch - target_time|.

    Raises TleFetchError, not a bare requests/auth exception, on any
    failure -- unregistered satellite, missing credentials, network/auth
    error, empty result, or a found candidate whose epoch is more than
    _MAX_ACCEPTABLE_EPOCH_GAP_DAYS away from target_time (a
    defensive backstop for sparse tracking periods).
    """
    if satellite not in SATELLITE_ORBIT_SPECS:
        raise TleFetchError(f"Unknown satellite {satellite!r} -- not in SATELLITE_ORBIT_SPECS.")

    if target_time.tzinfo is not None:
        target_time = target_time.astimezone(timezone.utc).replace(tzinfo=None)

    cache_dir = cache_dir or _DEFAULT_TLE_CACHE_DIR
    day = target_time.date()
    path = _cache_path(cache_dir, satellite, day)
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            return cached["line1"], cached["line2"]
        except Exception:
            pass  # corrupted cache entry -- fall through and re-fetch

    global _cached_space_track_session, _space_track_unavailable

    if _space_track_unavailable:
        raise TleFetchError(
            "Space-Track was marked unavailable earlier in this process "
            "(a previous authentication/connection attempt failed) -- "
            "not retrying."
        )

    norad_id = SATELLITE_ORBIT_SPECS[satellite].norad_id
    try:
        if _cached_space_track_session is None:
            from ..downloaders.base import authenticate_space_track

            username, password = authenticate_space_track()
            _cached_space_track_session = _space_track_session(username, password)
        before = _query_nearest_candidate(
            _cached_space_track_session, norad_id, target_time, before=True,
        )
        after = _query_nearest_candidate(
            _cached_space_track_session, norad_id, target_time, before=False,
        )
    except TleFetchError:
        _space_track_unavailable = True
        raise
    except Exception as exc:
        _space_track_unavailable = True
        raise TleFetchError(f"Space-Track query failed for {satellite}: {exc}") from exc

    candidates = [c for c in (before, after) if c is not None]
    if not candidates:
        raise TleFetchError(
            f"No historical TLE found for {satellite} near {target_time.isoformat()}."
        )

    best = min(candidates, key=lambda c: _epoch_gap_seconds(c, target_time))
    gap_days = _epoch_gap_seconds(best, target_time) / 86400.0
    if gap_days > _MAX_ACCEPTABLE_EPOCH_GAP_DAYS:
        raise TleFetchError(
            f"Nearest historical TLE for {satellite} is {gap_days:.1f} days from "
            f"{target_time.isoformat()}, beyond the {_MAX_ACCEPTABLE_EPOCH_GAP_DAYS}-day "
            f"acceptable propagation gap."
        )

    line1, line2 = best["TLE_LINE1"], best["TLE_LINE2"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"line1": line1, "line2": line2}))
    return line1, line2


def _point_in_bbox(
    lat: float, lon: float, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
) -> bool:
    from ..downloaders.base import split_antimeridian_bbox

    if not (min_lat <= lat <= max_lat):
        return False
    return any(lo <= lon <= hi for lo, hi in split_antimeridian_bbox(min_lon, max_lon))


def _point_in_polygon(lat: float, lon: float, polygon: "list[tuple[float, float]]") -> bool:
    """Standard even-odd ray-casting point-in-polygon test. *polygon* is a
    list of (lat, lon) vertices; it does not need to be explicitly closed
    (the last vertex need not repeat the first).

    Antimeridian-crossing polygons (a real case in this codebase -- see
    the antimeridian-crossing support already in downloaders/base.py's
    split_antimeridian_bbox) are handled by shifting every negative
    longitude -- both the polygon's own vertices and the test point -- by
    +360 degrees first, whenever the polygon's own longitude span exceeds
    180 degrees (the signal that it wraps through the seam rather than
    genuinely spanning most of the globe, since no real SAR/validation
    footprint is that wide). This runs the ray-casting math in one
    continuous, unwrapped frame instead of jumping across +/-180.

    Fails open (returns True) on degenerate input -- fewer than 3
    vertices isn't a real polygon, and this module's convention
    throughout is to fail OPEN (assume overlap) rather than closed
    whenever something can't be genuinely evaluated. This matters
    because callers outside this module may call _point_in_polygon
    directly with no surrounding try/except of their own.
    """
    if len(polygon) < 3:
        return True

    lons = [v[1] for v in polygon]
    if (max(lons) - min(lons)) > 180.0:
        polygon = [(plat, plon + 360.0 if plon < 0.0 else plon) for plat, plon in polygon]
        if lon < 0.0:
            lon = lon + 360.0

    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        if (lat_i > lat) != (lat_j > lat):
            lon_intercept = (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i) + lon_i
            if lon < lon_intercept:
                inside = not inside
        j = i
    return inside


def orbit_overlaps_bbox(
    satellite: str,
    sensing_start: datetime,
    sensing_end: datetime,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    margin_km: float = 100.0,
    sample_interval_s: float = 15.0,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    True if *satellite*'s predicted ground track/swath during
    [sensing_start, sensing_end] comes within margin_km of the given
    bbox. Fails open (returns True) if *satellite* is unregistered, the
    TLE cannot be fetched, or propagation raises for any reason -- this
    is a download-time optimization, not a substitute for the real
    domain-cropping applied downstream, and must never risk a false
    negative.

    Algorithm: fetch the TLE nearest sensing_start, sample the
    sub-satellite point every sample_interval_s across the window, and
    at each sample sweep both sides of the instantaneous heading out to
    swath_half_width_km + margin_km. Returns True on the first sample or
    swept point found inside the bbox (antimeridian-aware).
    """
    spec = SATELLITE_ORBIT_SPECS.get(satellite)
    if spec is None:
        return True

    try:
        from pyorbital.orbital import Orbital

        line1, line2 = get_tle(satellite, sensing_start, cache_dir=cache_dir)
        orb = Orbital(satellite.upper(), line1=line1, line2=line2)

        samples = []
        t = sensing_start
        step = timedelta(seconds=sample_interval_s)
        while t <= sensing_end:
            lon, lat, _alt = orb.get_lonlatalt(t)
            samples.append((t, lat, lon))
            t = t + step
        if not samples or samples[-1][0] < sensing_end:
            lon, lat, _alt = orb.get_lonlatalt(sensing_end)
            samples.append((sensing_end, lat, lon))

        max_offset_km = spec.swath_half_width_km + margin_km
        # Sweeps every ~50km out to max_offset_km, not just a couple of fixed
        # rings, so a bbox falling between two widely-spaced rings is never
        # silently missed. Cheap to compute (pure arithmetic, no propagation),
        # so there is no reason to sample coarser.
        _CROSS_TRACK_STEP_KM = 50.0
        n_steps = max(1, math.ceil(max_offset_km / _CROSS_TRACK_STEP_KM))
        sweep_distances_km = [max_offset_km * (i / n_steps) for i in range(1, n_steps + 1)]

        for i, (_t, lat, lon) in enumerate(samples):
            if _point_in_bbox(lat, lon, min_lon, max_lon, min_lat, max_lat):
                return True
            if i + 1 < len(samples):
                _next_t, next_lat, next_lon = samples[i + 1]
                heading = _bearing_deg(lat, lon, next_lat, next_lon)
            elif i > 0:
                _prev_t, prev_lat, prev_lon = samples[i - 1]
                heading = _bearing_deg(prev_lat, prev_lon, lat, lon)
            else:
                # A degenerate sensing_start == sensing_end window yields only one
                # ground-track sample, with no adjacent sample to derive a heading
                # from, so the cross-track sweep cannot be predicted. Per this
                # module's "never risk a false negative" contract, failing to
                # predict must fail open here, not silently fall through to
                # `return False` below.
                return True
            for side_bearing in (heading + 90.0, heading - 90.0):
                for dist in sweep_distances_km:
                    swept_lat, swept_lon = _destination_point(lat, lon, side_bearing, dist)
                    if _point_in_bbox(swept_lat, swept_lon, min_lon, max_lon, min_lat, max_lat):
                        return True
        return False
    except TleFetchError:
        return True
    except Exception:
        logger.debug(
            "orbit_overlaps_bbox: propagation failed for %s, failing open.", satellite, exc_info=True,
        )
        return True


def _region_contains(
    lat: float, lon: float,
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    polygon: "Optional[list[tuple[float, float]]]",
) -> bool:
    """bbox is always checked first (a cheap O(1) reject). Only when the
    bbox check passes is polygon (if supplied) checked too.

    IMPORTANT: the bbox is a superset of polygon (and so never causes a
    false negative) ONLY when min_lon/max_lon use the same wrap
    convention _point_in_bbox/split_antimeridian_bbox require -- i.e.
    min_lon > max_lon signals a region that crosses the antimeridian.
    _point_in_polygon detects wrapping completely differently (whether
    the polygon's OWN vertices span more than 180 degrees of longitude).
    A bbox derived the "obvious" way -- min(lons), max(lons) taken
    directly over a wrapping polygon's vertices -- does NOT use the wrap
    convention and produces the WRONG region: points genuinely inside
    the polygon can test as outside the AND'd (bbox and polygon) region,
    a real false negative. Callers passing a polygon that may cross the
    antimeridian must derive min_lon/max_lon using the wrap convention,
    not a naive min/max over the polygon's vertices."""
    if not _point_in_bbox(lat, lon, min_lon, max_lon, min_lat, max_lat):
        return False
    if polygon is not None:
        return _point_in_polygon(lat, lon, polygon)
    return True


def orbit_overlap_windows(
    satellite: str,
    sensing_start: datetime,
    sensing_end: datetime,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    polygon: "Optional[list[tuple[float, float]]]" = None,
    margin_km: float = 100.0,
    sample_interval_s: float = 15.0,
    cache_dir: Optional[Path] = None,
) -> "list[tuple[datetime, datetime]]":
    """Like orbit_overlaps_bbox, but returns every matching sub-window
    (grouping consecutive overlapping samples) instead of a single bool,
    and optionally tests against a true footprint polygon instead of just
    its bounding box.

    Needed because orbit_overlaps_bbox only answers "did the ground track
    cross the bbox at *some* point in this window" -- for a source whose
    window is already narrow (H-SAF's ~3-minute real sensing window,
    HY-2/Oceansat-3's padded single timestamp), that's equivalent. But
    AMSR2 (and SMOS, when it falls back to a whole-day window) uses a
    WHOLE-DAY window, where "yes, sometime today" is true almost always
    and tells a caller nothing about *when* -- which is exactly what's
    needed to compare against a SAR scene's own
    [sensing_start, sensing_end] +/- a time tolerance.

    Empty list if no sample crosses the target region (equivalent to
    orbit_overlaps_bbox returning False); the whole window as a
    single-element list on any fail-open condition (unregistered
    satellite, TleFetchError, propagation exception, or a degenerate
    single-sample window) -- same fail-open contract as
    orbit_overlaps_bbox, extended to "assume the whole window overlaps"
    since which sub-window can't be known when failing open.

    Each returned window is padded by sample_interval_s on each side
    (clamped to [sensing_start, sensing_end]) before being returned: the
    bounds of the matching SAMPLES are not the true underlying crossing,
    which can begin/end anywhere between two samples, so returning the
    raw sample timestamps would under-cover the true overlap by up to
    sample_interval_s at each edge -- the opposite of this module's
    fail-toward-inclusion principle. Without padding, a single matching
    sample would even produce a zero-duration window.

    PRECONDITION when passing a polygon that may cross the antimeridian:
    min_lon/max_lon must use the wrap convention _point_in_bbox and
    split_antimeridian_bbox require (min_lon > max_lon signals a
    wrapping region) -- NOT a bbox derived by taking min(lons), max(lons)
    over the polygon's own vertices. See _region_contains's docstring
    for why the naive derivation produces false negatives.
    """
    spec = SATELLITE_ORBIT_SPECS.get(satellite)
    if spec is None:
        return [(sensing_start, sensing_end)]

    try:
        from pyorbital.orbital import Orbital

        line1, line2 = get_tle(satellite, sensing_start, cache_dir=cache_dir)
        orb = Orbital(satellite.upper(), line1=line1, line2=line2)

        samples = []
        t = sensing_start
        step = timedelta(seconds=sample_interval_s)
        while t <= sensing_end:
            lon, lat, _alt = orb.get_lonlatalt(t)
            samples.append((t, lat, lon))
            t = t + step
        if not samples or samples[-1][0] < sensing_end:
            lon, lat, _alt = orb.get_lonlatalt(sensing_end)
            samples.append((sensing_end, lat, lon))

        if len(samples) < 2:
            # Degenerate window -- no adjacent sample to derive a heading
            # from, matching orbit_overlaps_bbox's fail-open behavior for
            # this same case.
            return [(sensing_start, sensing_end)]

        max_offset_km = spec.swath_half_width_km + margin_km
        # Mirrors orbit_overlaps_bbox's sampling/sweep; keep in sync.
        _CROSS_TRACK_STEP_KM = 50.0
        n_steps = max(1, math.ceil(max_offset_km / _CROSS_TRACK_STEP_KM))
        sweep_distances_km = [max_offset_km * (i / n_steps) for i in range(1, n_steps + 1)]

        matched = [False] * len(samples)
        for i, (_t, lat, lon) in enumerate(samples):
            if _region_contains(lat, lon, min_lon, max_lon, min_lat, max_lat, polygon):
                matched[i] = True
                continue
            if i + 1 < len(samples):
                _next_t, next_lat, next_lon = samples[i + 1]
                heading = _bearing_deg(lat, lon, next_lat, next_lon)
            else:
                _prev_t, prev_lat, prev_lon = samples[i - 1]
                heading = _bearing_deg(prev_lat, prev_lon, lat, lon)
            for side_bearing in (heading + 90.0, heading - 90.0):
                for dist in sweep_distances_km:
                    swept_lat, swept_lon = _destination_point(lat, lon, side_bearing, dist)
                    if _region_contains(swept_lat, swept_lon, min_lon, max_lon, min_lat, max_lat, polygon):
                        matched[i] = True
                        break
                if matched[i]:
                    break

        windows: "list[tuple[datetime, datetime]]" = []
        i = 0
        n = len(samples)
        while i < n:
            if not matched[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and matched[j + 1]:
                j += 1
            window_start = max(sensing_start, samples[i][0] - timedelta(seconds=sample_interval_s))
            window_end = min(sensing_end, samples[j][0] + timedelta(seconds=sample_interval_s))
            windows.append((window_start, window_end))
            i = j + 1
        return windows
    except TleFetchError:
        return [(sensing_start, sensing_end)]
    except Exception:
        logger.debug(
            "orbit_overlap_windows: propagation failed for %s, failing open.", satellite, exc_info=True,
        )
        return [(sensing_start, sensing_end)]
