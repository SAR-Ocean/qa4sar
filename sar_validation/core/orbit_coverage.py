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

import math
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["SatelliteOrbitSpec", "SATELLITE_ORBIT_SPECS", "TleFetchError"]

_EARTH_RADIUS_KM = 6371.0


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
