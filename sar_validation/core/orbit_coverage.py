"""
General-purpose, satellite-agnostic orbit-based geographic pre-filter.

Given a satellite name and a sensing time window, predicts whether that
satellite's ground track/swath could have passed near a given bounding
box -- so a downloader with no server-side bbox filter (e.g. H-SAF's FTP
archive, see hsaf_downloader.py) can skip files whose orbit segment
never came near the requested region, before downloading them.

Fails open throughout (returns True -- "could overlap, don't filter this
file out") whenever prediction isn't possible or trustworthy: an
unregistered satellite, a TLE that can't be fetched, or any propagation
error. This is a pre-filter to reduce unnecessary downloads, never a
substitute for the real geographic filtering domain-cropping already
does downstream -- so it must never risk a false negative.

See docs/superpowers/specs/2026-08-13-sar-scene-aware-download-narrowing-and-orbit-prefilter-design.md
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
]

_EARTH_RADIUS_KM = 6371.0

#: A cached TLE is keyed to a fixed historical (satellite, date) pair, so
#: once fetched it is immutable -- no max-age eviction needed (unlike a
#: "current TLE" cache would need). See get_tle().
_MAX_ACCEPTABLE_EPOCH_GAP_DAYS = 10.0

_SPACE_TRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
_SPACE_TRACK_QUERY_BASE = "https://www.space-track.org/basicspacedata/query/class/gp_history"

#: Matches this codebase's existing repo-relative shared-cache convention
#: (see hf_radar_historical_downloader.py's _ARCHIVE_CACHE_DIR,
#: ismn_downloader.py's _SHARED_ARCHIVE_CACHE_DIR) rather than a
#: per-user home-directory cache.
_DEFAULT_TLE_CACHE_DIR = Path("data") / "_archive_cache" / "tle"


@dataclass(frozen=True)
class SatelliteOrbitSpec:
    """One satellite this module can predict ground-track/swath overlap
    for. norad_id feeds the Space-Track TLE lookup (see get_tle);
    swath_half_width_km is the outer edge of the satellite's real swath
    from its ground track, in km -- e.g. ASCAT/MetOp's ~550-600km-wide
    swath on each side of the ground track (the inner gap between the
    two swaths is deliberately not modeled; treating both sides as one
    continuous corridor out to the outer edge is a safe over-inclusion,
    see the design doc)."""

    norad_id: int
    swath_half_width_km: float


#: Real, live-confirmed NORAD catalog IDs (CelesTrak, 2026-08-13; Space-
#: Track uses the same numbering). Add entries here (not in any
#: per-source downloader file) to extend orbit pre-filtering to a new
#: satellite -- MetOp-A is deliberately absent (not in CelesTrak's active
#: weather-satellite list; every real H-SAF sample file this codebase
#: has seen only ever shows METOPB/METOPC). A satellite absent here
#: simply isn't pre-filtered -- see orbit_overlaps_bbox's fail-open
#: behavior.
SATELLITE_ORBIT_SPECS: Dict[str, SatelliteOrbitSpec] = {
    "metop-b": SatelliteOrbitSpec(norad_id=38771, swath_half_width_km=600.0),
    "metop-c": SatelliteOrbitSpec(norad_id=43689, swath_half_width_km=600.0),
}


class TleFetchError(Exception):
    """Raised when no usable historical TLE can be obtained for a given
    satellite/time -- callers (orbit_overlaps_bbox) must treat this as
    "cannot predict, fail open", never propagate it as a hard error."""


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing (degrees, 0-360, 0=north, clockwise)
    from (lat1, lon1) to (lat2, lon2). Standard spherical-Earth forward-
    azimuth formula (Movable Type Scripts' "Bearing" reference)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360.0) % 360.0


def _destination_point(
    lat: float, lon: float, bearing_deg: float, distance_km: float,
) -> Tuple[float, float]:
    """(lat, lon) reached by travelling distance_km along bearing_deg
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
    """One gp_history record: the single nearest-EPOCH candidate strictly
    before (before=True) or strictly after (before=False) target_time, or
    None if the historical archive has nothing on that side.

    Uses a plain "<"/">" (not ">=") -- Space-Track's query parser rejects
    the "=" character in a predicate value with an HTTP 400 ("The URI you
    submitted has disallowed characters"), confirmed live against the
    real API. This loses only the negligible edge case of a TLE epoch
    landing on the exact microsecond of target_time.
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
    """(line1, line2) for *satellite*, nearest in epoch to *target_time*.

    Cached on disk keyed by (satellite, target_time.date()) -- since the
    result is "the historical TLE nearest this fixed past date", it never
    changes once fetched, so the cache entry is permanent (many files
    from the same day share one cache read, and one pair of Space-Track
    queries on a cache miss, which matters given H-SAF's ~3-minute file
    cadence and Space-Track's API rate limits).

    On a cache miss: authenticates to Space-Track (credentials via
    authenticate_space_track(), see downloaders/base.py), queries the
    gp_history class for the NORAD ID nearest target_time (both before
    and after, since target_time is always in the past for this design's
    use case), and picks whichever candidate has the smaller
    |epoch - target_time|.

    Raises TleFetchError (not a bare requests/auth exception) on any
    failure -- unregistered satellite, missing credentials, network/auth
    error, empty result, or a found candidate whose epoch is still more
    than _MAX_ACCEPTABLE_EPOCH_GAP_DAYS away from target_time (a
    defensive backstop for sparse tracking periods) -- so callers have
    one exception type to catch and always fail open.
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

    norad_id = SATELLITE_ORBIT_SPECS[satellite].norad_id
    try:
        from ..downloaders.base import authenticate_space_track

        username, password = authenticate_space_track()
        session = _space_track_session(username, password)
        before = _query_nearest_candidate(session, norad_id, target_time, before=True)
        after = _query_nearest_candidate(session, norad_id, target_time, before=False)
    except TleFetchError:
        raise
    except Exception as exc:
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
    """True if *satellite*'s predicted ground track/swath during
    [sensing_start, sensing_end] comes within margin_km of the given
    bbox. Fails open (returns True -- never filters when uncertain) if
    *satellite* isn't in SATELLITE_ORBIT_SPECS, the TLE can't be fetched
    (TleFetchError), or orbit propagation raises for any reason -- this
    is a download-time optimization, never a substitute for the real
    domain-cropping that happens downstream, so it must never risk a
    false negative.

    Algorithm: fetch the TLE nearest sensing_start via get_tle(satellite,
    sensing_start, cache_dir). Sample the sub-satellite point every
    sample_interval_s across [sensing_start, sensing_end] via
    Orbital.get_lonlatalt. At each sample, derive the instantaneous
    heading via _bearing_deg against the adjacent sample (next sample if
    available, else the previous one). At each sample, sweep both
    perpendicular sides of that heading (bearing +/- 90) at increasing
    distances up to swath_half_width_km + margin_km via
    _destination_point, and check whether the sub-satellite point OR any
    swept point falls within the bbox (antimeridian-aware). Returns True
    on the first match found (short-circuits).
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
        # Sweep every ~50km out to max_offset_km, not just a couple of
        # fixed rings -- a bbox exactly between two widely-spaced sample
        # rings would otherwise be silently missed even though it's well
        # inside the real corridor. Recipe bboxes in this codebase are
        # routinely thousands of km wide (see e.g. recipes/soil_moisture_
        # hsaf_ascat.yaml's -10..30 lon span), so a 50km cross-track grid
        # is dense relative to any bbox this filter is actually run
        # against -- cheap to compute (pure arithmetic, no propagation),
        # so there's no reason to sample coarser.
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
                continue
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
