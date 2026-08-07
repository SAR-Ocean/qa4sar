"""
Download RADARSAT-2 SAR-derived ocean surface wind scenes from the NOAA
NCEI THREDDS archive (2014-05-02 - present, global coverage concentrated
over Alaska/the North Pacific).

Catalog layout: thredds-ocean/catalog/sar-winds/radarsat2/{YYYY}/{MM}/
One file per SAR scene (not merged into a shared grid, unlike
noaa_hfradar_thredds_downloader.py -- each granule is an independent SAR
scene, kept as its own file). Filename format changed sometime between
2023 and 2024 (bisected live against NOAA's real catalog):
  - Before: RSAT2_{PROVIDER}_{YYYY}_{MM}_{DD}_{HH}_{MM}_{SS}_{seq}_
    {lon}{E|W}_{lat}{N|S}_{POL}_C5_{MODEL}_wind_level2_norcs.nc
    e.g. RSAT2_GSS_2019_06_01_02_01_52_0612669712_131.54W_71.53N_HH_C5_GFS05CDF_wind_level2_norcs.nc
  - From ~2024: SAR-Wind-{POL}-{lat}{N|S}-{lon}{E|W}_v{maj}r{min}_rsat2_
    s{start}_e{end}_c{created}.nc
    e.g. SAR-Wind-HH-64N-174E_v3r0_rsat2_s202606040552510_e202606040554070_c202606041745293.nc

Both eras embed a scene-center lon/lat in the filename, used as a coarse
pre-filter before downloading (THREDDS' catalog.xml has no spatial
search API). A surviving candidate is then precisely checked against
its own lightweight NCML metadata (THREDDS' /ncml/ service -- a small
XML document with zero data values) before any full ~38MB scene is
downloaded; see RADARSAT2WindDownloader._download_window.

Library usage::

    from sar_validation.downloaders.radarsat2_wind_downloader import (
        RADARSAT2WindDownloader,
    )
    dl = RADARSAT2WindDownloader(output_dir=Path("data/run1/RADARSAT2_WIND"))
    dl.download(min_lon=165, max_lon=180, min_lat=60, max_lat=68,
                start="2026-06-01", end="2026-06-30")
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import months_touched, normalize_datetime, prefer_ipv4_dns, split_antimeridian_bbox

__all__ = ["RADARSAT2WindDownloader"]

logger = logging.getLogger(__name__)

THREDDS_BASE = "https://www.ncei.noaa.gov/thredds-ocean"
_CATALOG_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"
_NCML_NS = "{http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2}"

# This pad only needs to be generous enough to never exclude a real
# candidate before the precise, NCML-based check (RADARSAT2WindDownloader
# ._passes_ncml_check, Task 3) gets a chance to look at it -- unlike an
# earlier draft of this design, it is no longer the thing responsible for
# correctness, so erring wide costs a few extra ~30KB NCML requests, not
# extra full downloads. A scene is ~0.5km/px over roughly 1005-1055px per
# side (~500-525km, so ~350km center-to-corner half-diagonal); 5 degrees
# comfortably covers that with margin at every latitude this archive's
# scenes occur at.
_BBOX_PAD_DEG = 5.0

_OLD_PATTERN = re.compile(
    r"^RSAT2_[A-Za-z0-9]+_(?P<y>\d{4})_(?P<mo>\d{2})_(?P<d>\d{2})_"
    r"(?P<h>\d{2})_(?P<mi>\d{2})_(?P<s>\d{2})_\d+_"
    r"(?P<lon>\d+(?:\.\d+)?)(?P<lonhem>[EW])_"
    r"(?P<lat>\d+(?:\.\d+)?)(?P<lathem>[NS])_.*_wind_level2_norcs\.nc$"
)
_NEW_PATTERN = re.compile(
    r"^SAR-Wind-[A-Z]+-(?P<lat>\d+)(?P<lathem>[NS])-(?P<lon>\d+)(?P<lonhem>[EW])_"
    r"v\d+r\d+_rsat2_s(?P<ts>\d{15})_e\d{15}_c\d{15}\.nc$"
)


def _signed(value: float, hem: str, positive: str) -> float:
    """*value* is unsigned; *hem* is the axis letter parsed from the
    filename. Returns the signed value -- negative when *hem* is the
    "negative" letter for that axis (S for latitude, W for longitude)."""
    return value if hem == positive else -value


def _parse_granule_name(name: str) -> Optional[Tuple[datetime, float, float]]:
    """Return (timestamp, center_lon [-180..180], center_lat) for a
    RADARSAT-2 THREDDS granule filename in either the old (pre-2024) or
    new (2024-onward) naming era, or None if *name* matches neither."""
    m = _OLD_PATTERN.match(name)
    if m:
        ts = datetime(
            int(m["y"]), int(m["mo"]), int(m["d"]),
            int(m["h"]), int(m["mi"]), int(m["s"]),
        )
        lon = _signed(float(m["lon"]), m["lonhem"], "E")
        lat = _signed(float(m["lat"]), m["lathem"], "N")
    else:
        m = _NEW_PATTERN.match(name)
        if not m:
            return None
        ts = datetime.strptime(m["ts"][:14], "%Y%m%d%H%M%S")
        lon = _signed(float(m["lon"]), m["lonhem"], "E")
        lat = _signed(float(m["lat"]), m["lathem"], "N")
    lon = ((lon + 180) % 360) - 180
    return ts, lon, lat


def _lon_within_padded_bbox(lon: float, min_lon: float, max_lon: float) -> bool:
    """True if *lon* (a filename-parsed center longitude, already
    normalized to [-180, 180]) falls within [min_lon, max_lon] padded by
    _BBOX_PAD_DEG on each side.

    The naive `min_lon - PAD <= lon <= max_lon + PAD` check silently
    fails near the antimeridian: padding max_lon=180 by 5 degrees gives
    an upper bound of 185, a value no normalized longitude can ever
    reach, even though a real candidate at e.g. -178 is geographically
    only 2 degrees past the 180 edge. When the *padded* window itself
    would exceed +-180, this also accepts the wrapped-around slice on
    the other side of the antimeridian (e.g. lon <= -175 when max_lon +
    PAD == 185). This is independent of whether the original,
    unpadded [min_lon, max_lon] request already crosses the
    antimeridian (callers -- see RADARSAT2WindDownloader.download --
    pre-split such requests via split_antimeridian_bbox so min_lon <=
    max_lon always holds here); it only concerns the pad's own overflow
    past +-180."""
    padded_min = min_lon - _BBOX_PAD_DEG
    padded_max = max_lon + _BBOX_PAD_DEG
    if padded_min <= lon <= padded_max:
        return True
    if padded_max > 180 and lon <= padded_max - 360:
        return True
    if padded_min < -180 and lon >= padded_min + 360:
        return True
    return False


def _list_radarsat2_granules(
    catalog_xml_text: str, start: datetime, end: datetime,
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    end_exclusive: bool = False,
) -> List[Tuple[datetime, str, float, float]]:
    """Parse one month's THREDDS catalog.xml and return (timestamp,
    urlPath, center_lon, center_lat) for every granule whose timestamp
    falls in [start, end] (or [start, end) if *end_exclusive*) AND whose
    filename-embedded center point falls within the requested bbox padded
    by _BBOX_PAD_DEG on every side. Filenames matching neither known
    naming era are skipped."""
    root = ET.fromstring(catalog_xml_text)
    results: List[Tuple[datetime, str, float, float]] = []
    for ds_elem in root.iter(f"{_CATALOG_NS}dataset"):
        name = ds_elem.get("name")
        url_path = ds_elem.get("urlPath")
        if not name or not url_path:
            continue
        parsed = _parse_granule_name(name)
        if parsed is None:
            continue
        ts, lon, lat = parsed
        end_ok = ts < end if end_exclusive else ts <= end
        if not (start <= ts and end_ok):
            continue
        if not _lon_within_padded_bbox(lon, min_lon, max_lon):
            continue
        if not (min_lat - _BBOX_PAD_DEG <= lat <= max_lat + _BBOX_PAD_DEG):
            continue
        results.append((ts, url_path, lon, lat))
    return sorted(results)


def _parse_ncml_bbox(ncml_xml_text: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse a THREDDS NCML metadata document (the `/ncml/{urlPath}`
    service) and return (lon_min, lon_max, lat_min, lat_max), or None if
    the four geospatial_*_min/max attributes aren't found anywhere in it.

    New-era granules carry these as root-level global attributes (the
    file's own stated values). Old-era granules' raw NetCDF files carry
    no such attributes at all, but THREDDS' NCML service still reports
    them -- auto-computed server-side from the actual coordinate data --
    nested inside a <group name="CFMetadata"> element instead (confirmed
    live against a real 2019 granule). root.iter() finds either location
    regardless of nesting; taking the *first* match of each attribute
    name in document order picks up the root-level value when present
    (it always appears before any nested group in a real NCML document)
    and falls back to the CFMetadata-group value otherwise -- no
    era-conditional branching needed here.
    """
    root = ET.fromstring(ncml_xml_text)
    values: Dict[str, float] = {}
    wanted = ("geospatial_lon_min", "geospatial_lon_max", "geospatial_lat_min", "geospatial_lat_max")
    for attr in root.iter(f"{_NCML_NS}attribute"):
        name = attr.get("name")
        value_str = attr.get("value")
        if name in wanted and name not in values and value_str is not None:
            try:
                values[name] = float(value_str)
            except ValueError:
                continue
    if not all(k in values for k in wanted):
        return None
    return (
        values["geospatial_lon_min"], values["geospatial_lon_max"],
        values["geospatial_lat_min"], values["geospatial_lat_max"],
    )


class RADARSAT2WindDownloader:
    """Download RADARSAT-2 SAR wind scenes from the NOAA NCEI THREDDS
    archive for a bbox/time window. One `.nc` file per scene -- scenes
    are independent SAR granules, never merged.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded granules.
    dry_run : bool
        If True, query each touched month's (lightweight) catalog.xml and
        report the real candidate scene count/filenames, but never touch
        the per-candidate NCML check or a full fileServer download; always
        returns [].
    force_download : bool
        Re-download a granule even if a same-named file already exists.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False, force_download: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download

    def download(self, min_lon, max_lon, min_lat, max_lat, start: str, end: str) -> list[Path]:
        windows = split_antimeridian_bbox(min_lon, max_lon)
        downloaded: list[Path] = []
        for win_min_lon, win_max_lon in windows:
            downloaded.extend(
                self._download_window(win_min_lon, win_max_lon, min_lat, max_lat, start, end)
            )
        return downloaded

    def _download_window(self, min_lon, max_lon, min_lat, max_lat, start: str, end: str) -> list[Path]:
        start_dt = datetime.fromisoformat(normalize_datetime(start))
        # A bare date (e.g. "2026-06-05") normalizes to that date's literal
        # midnight -- the same instant normalize_datetime and every other
        # exact-instant comparison in this pipeline (e.g. the in-situ
        # download window) already treat it as. A granule timestamped
        # later that same day is therefore genuinely outside the requested
        # window and must not be matched; end_dt is used as-is, with no
        # "widen to the whole day" special-casing.
        end_dt = datetime.fromisoformat(normalize_datetime(end))

        granules: List[Tuple[datetime, str, float, float]] = []
        for year, month in months_touched(start_dt, end_dt):
            catalog_url = f"{THREDDS_BASE}/catalog/sar-winds/radarsat2/{year}/{month:02d}/catalog.xml"
            try:
                with prefer_ipv4_dns(), urllib.request.urlopen(catalog_url, timeout=15) as resp:
                    text = resp.read().decode()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            granules.extend(
                _list_radarsat2_granules(
                    text, start_dt, end_dt,
                    min_lon, max_lon, min_lat, max_lat,
                )
            )

        logger.info("radarsat2_wind: %d candidate granule(s) matched", len(granules))

        if self.dry_run:
            # Still queries each touched month's (lightweight) catalog.xml
            # above -- so this reports real candidate availability -- but
            # never touches the per-candidate NCML check or a full
            # ~38MB fileServer download.
            print(
                f"[dry-run] RADARSAT-2 wind: {len(granules)} candidate scene(s) "
                f"in [{start_dt}, {end_dt}] within "
                f"[{min_lon - _BBOX_PAD_DEG}, {max_lon + _BBOX_PAD_DEG}] x "
                f"[{min_lat - _BBOX_PAD_DEG}, {max_lat + _BBOX_PAD_DEG}]"
            )
            for _ts, url_path, _lon, _lat in granules:
                print(f"  {Path(url_path).name}")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for _ts, url_path, _lon, _lat in granules:
            dest = self.output_dir / Path(url_path).name
            if not self.force_download and dest.exists():
                print(f"  Already downloaded: {dest}")
                downloaded.append(dest)
                continue

            if not self._passes_ncml_check(url_path, min_lon, max_lon, min_lat, max_lat):
                logger.info(
                    "radarsat2_wind: skipping %s -- NCML metadata bbox does "
                    "not overlap the requested bbox", Path(url_path).name,
                )
                continue

            with prefer_ipv4_dns(), urllib.request.urlopen(
                f"{THREDDS_BASE}/fileServer/{url_path}", timeout=15
            ) as resp:
                dest.write_bytes(resp.read())
            print(f"  Downloaded: {dest}")
            downloaded.append(dest)

        return downloaded

    @staticmethod
    def _passes_ncml_check(url_path: str, min_lon, max_lon, min_lat, max_lat) -> bool:
        """True if the granule's real footprint (from its lightweight
        NCML metadata) overlaps the requested bbox, or if the NCML
        metadata is unavailable/unparseable -- fails OPEN in that case
        (returns True) so a transient metadata-service hiccup doesn't
        silently drop a real granule. A false positive here just costs
        one unnecessary ~38MB download, not a science error; nothing
        downstream re-checks the footprint once downloaded."""
        try:
            with prefer_ipv4_dns(), urllib.request.urlopen(
                f"{THREDDS_BASE}/ncml/{url_path}", timeout=15
            ) as resp:
                text = resp.read().decode()
        except urllib.error.URLError:
            # Covers HTTPError too (it subclasses URLError).
            return True
        bbox = _parse_ncml_bbox(text)
        if bbox is None:
            return True
        lon_min, lon_max, lat_min, lat_max = bbox
        return not (lon_max < min_lon or lon_min > max_lon or lat_max < min_lat or lat_min > max_lat)
