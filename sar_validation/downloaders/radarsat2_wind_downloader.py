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

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# __all__ is declared in Task 3, once RADARSAT2WindDownloader (the only
# public symbol) actually exists -- declaring it here would make ruff's
# F822 flag an undefined name in __all__ (the class doesn't exist until
# Task 3 appends it to this same file).

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
        if not (min_lon - _BBOX_PAD_DEG <= lon <= max_lon + _BBOX_PAD_DEG):
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
