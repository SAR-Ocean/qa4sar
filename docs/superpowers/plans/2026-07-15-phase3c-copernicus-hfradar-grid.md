# Phase 3c — Copernicus HF-Radar Gridded Currents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken Copernicus `hf_radar` source (which queries a
sparse in-situ dataset_part that carries no HF-radar platforms and always
returns empty) with a working downloader that fetches the *actual*
HF-radar current grids Copernicus Marine publishes, and add the delayed-mode
historical archive as a second source (`hf_radar_historical`).

**Architecture:** Both new/rewritten downloaders fetch a regular
`(time, lat, lon)` grid with canonical `EWCT`/`NSCT` variables and normalize
it to the same on-disk NetCDF shape the existing NOAA `hf_radar_grid`
converter already expects — reusing that converter (generalized to accept
configurable source variable names) rather than adding a parallel
CSV/point-based path, per the design's Approach A principle ("the data
decides the geometry, not the source"). A new shared module resolves a
recipe's bounding box to one of Copernicus's ~25 named HF-radar regions.

**Tech Stack:** `copernicusmarine` 2.4.1, `xarray`, `pandas`. Same libraries
and CLI/orchestrator/recipe conventions already used throughout
`sar_validation/downloaders/` and `sar_validation/core/`.

## Global Constraints

- All dataset ids, `dataset_part` names, region bounding boxes, file-naming
  patterns, and variable/dimension names below were verified live against
  `copernicusmarine` 2.4.1 on 2026-07-15 (`copernicusmarine.describe()`,
  `copernicusmarine.subset()`, `copernicusmarine.get(dry_run=True)`, and one
  real downloaded file per dataset) — they are not guesses.
- `ValidationDataSource.min_depth`/`max_depth` must remain accepted
  constructor kwargs on `HFRadarDownloader` even though the gridded product
  has no depth axis to filter — two existing orchestrator tests
  (`tests/test_downloaders.py::TestOrchestratorDepthResolution::test_hf_radar_dispatch_*`)
  assert the orchestrator forwards `resolved_min_depth`/`resolved_max_depth`
  into the constructor, and changing that contract is out of scope here.
- Follow the existing `HFRadarDownloader`/`NOAAHFRadarDownloader`
  dataset-part/backend retry pattern (try the preferred part, catch the
  specific "out of bounds" error, retry with the fallback part).
- Reuse `normalize_datetime`/`is_date_recent`/`build_output_dir` from
  `sar_validation/downloaders/base.py` — do not reimplement datetime
  handling.
- NOAA THREDDS/OPeNDAP archive backend (design §3.1 second bullet, "Phase
  3b") is explicitly **out of scope** for this plan — `hfrnet-tds.ucsd.edu`
  was unreachable during design verification (connection refused / timeout
  from both direct network access and the fetch service), so its catalog
  structure could not be verified. Track separately.

---

## Verified facts this plan is built on

**NRT dataset** — `cmems_obs-ins_glo_phybgcwav_mynrt_na_irr`
(`INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030`). HF-radar current grids live
in per-region `dataset_part`s named `"<latest|monthly>-radar-total--<Region>"`
(NOT the plain `"latest"`/`"monthly"` sparse parts the current code queries —
those carry zero HF-radar rows). `monthly-radar-total--<Region>` covers all
25 regions; `latest-radar-total--<Region>` only exists for 18 of them (the
other 7 — `ARPAS`, `COSYNA`, `Finnmark`, `US-Alaska`, `US-EastGulfCoast`,
`US-Hawaii`, `WHub` — have no near-real-time feed, `monthly` only).
`copernicusmarine.subset(dataset_id=..., dataset_part=<part>, minimum_longitude=..., ...)`
returns a NetCDF with dims `(time, latitude, longitude)` and variables
`EWCT, NSCT, CCOV, EWCS, NSCS, GDOP, GDOP_QC, CSPD_QC, DDNS_QC, POSITION_QC,
QCflag, UACC, VACC, VART_QC` — canonical current-component names already,
no rename needed. Confirmed region bounding boxes (from
`monthly-radar-total--*` service metadata, `(min_lon, max_lon, min_lat,
max_lat)`):

```
ARPAS:                      (8.164931297302246, 8.878068923950195, 40.756874084472656, 41.243125915527344)
CALYPSO:                    (13.69489860534668, 15.365100860595703, 35.742252349853516, 36.87774658203125)
COSYNA:                     (5.916669845581055, 8.970867156982422, 53.41814422607422, 55.20000076293945)
DeltaEbro:                  (0.06353014707565308, 2.078089952468872, 39.59859848022461, 41.219600677490234)
EUSKOOS:                    (-3.1965925693511963, -1.2034072875976562, 43.31999969482422, 44.58000183105469)
Finnmark:                   (18.042926788330078, 35.45707321166992, 69.00939178466797, 73.49060821533203)
Galicia:                    (-11.327714920043945, -7.9691948890686035, 40.35469055175781, 44.67578887939453)
Gibraltar:                  (-5.8481950759887695, -4.994204998016357, 35.805084228515625, 36.1926155090332)
GoS:                        (13.072955131530762, 15.999811172485352, 39.672447204589844, 41.38346862792969)
Granitola:                  (12.176433563232422, 13.496066093444824, 37.00240707397461, 37.59709167480469)
ICATMAR:                    (1.0088672637939453, 4.291132926940918, 40.50752258300781, 42.99247741699219)
Ibiza:                      (0.5038551688194275, 1.4006848335266113, 38.3229866027832, 39.1067008972168)
Lisboa:                     (-10.595585823059082, -8.704414367675781, 37.90182113647461, 38.8981819152832)
NAdr:                       (13.375, 13.780560493469238, 45.526851654052734, 45.783329010009766)
PLOCAN:                     (-15.420296669006348, -14.89370346069336, 27.85053062438965, 28.34457015991211)
Skagerrak:                  (7.502049922943115, 11.997950553894043, 57.011016845703125, 59.488983154296875)
South:                      (-9.593195915222168, -6.206803798675537, 36.005245208740234, 37.19475555419922)
TirLig:                     (7.50698709487915, 10.493012428283691, 43.25399398803711, 44.49600601196289)
US-Alaska:                  (-174.09815979003906, -128.659912109375, 68.0091781616211, 74.0321044921875)
US-EastGulfCoast:           (-97.88055419921875, -57.2345085144043, 21.75611114501953, 46.47426986694336)
US-Hawaii:                  (-163.11717224121094, -152.01162719726562, 16.224061965942383, 24.895137786865234)
US-PuertoRicoVirginIslands: (-70.49944305419922, -61.024757385253906, 14.508465766906738, 21.989192962646484)
US-WestCoast:                (-130.33265686035156, -115.83291625976562, 30.25959587097168, 49.982444763183594)
Vestlandet:                  (1.0030561685562134, 8.496943473815918, 56.50277328491211, 63.99722671508789)
WHub:                        (-6.13332986831665, -5.0837883949279785, 50.21573257446289, 51.01667022705078)
```

Regions **without** a `latest-radar-total--<Region>` part (monthly only):
`ARPAS`, `COSYNA`, `Finnmark`, `US-Alaska`, `US-EastGulfCoast`, `US-Hawaii`,
`WHub`.

**Delayed-mode dataset** — `cmems_obs-ins_glo_phy-cur_my_radar-total_irr`
(`INSITU_GLO_PHY_UV_DISCRETE_MY_013_044`). Not subsettable server-side — it's
an "original-files" bulk-download service: one NetCDF per region under
`history/HF/GL_TV_HF_HFR-<Region>_Total.nc`, except `US-EastGulfCoast` which
is split by year (`GL_TV_HF_HFR-US-EastGulfCoast_Total_<YYYY>.nc`, years
2019–2024). File sizes are per-region, not per-request (e.g. `US-WestCoast`
full archive is 1.22 GB, `US-Alaska` 35.5 MB, one `US-EastGulfCoast` year
~130–275 MB) — manageable for a single recipe run (one region), not for
downloading the whole 82 GB dataset. Dims are `(TIME, DEPTH, LATITUDE,
LONGITUDE)` (uppercase, OceanSITES convention) with a singleton `DEPTH`;
variables include the same `EWCT`/`NSCT` names plus `GDOP`, `EWCS`, `NSCS`,
`QCflag`, `CSPD_QC`, `DDNS_QC`, `GDOP_QC`, `VART_QC`, plus station-metadata
variables (`NARX`, `SLTR`, `SDN_*`, ...) that are simply not copied over.
Available regions (verified from the live file listing): `ARPAS`, `CALYPSO`,
`COSYNA`, `DeltaEbro`, `EUSKOOS`, `Finnmark`, `Galicia`, `Gibraltar`, `GoM`,
`ICATMAR`, `Ibiza`, `Lisboa`, `MATROOS`, `NAdr`, `PLOCAN`, `Skagerrak`,
`South`, `TirLig`, `US-Alaska`, `US-EastGulfCoast`, `US-Hawaii`,
`US-PuertoRicoVirginIslands`, `US-WestCoast`, `Vestlandet`, `Vigo`. Three of
those (`GoM`, `MATROOS`, `Vigo`) have no counterpart in the NRT region table
above (so a request bbox can never resolve to them — out of scope, no bbox
is known for them) and three NRT regions (`GoS`, `Granitola`, `WHub`) have no
historical archive at all.

**No existing converter path reads a `hf_radar/` folder today** —
`convert_downloaded_data` has an `hfr_noaa/` block but never had one for
`hf_radar/`; the old CSV downloader's output was dead-ended (never wired
into conversion). This plan adds the missing folder-discovery blocks.

---

## Task 1: Shared HF-radar region-resolution module

**Files:**
- Create: `sar_validation/downloaders/_hf_radar_regions.py`
- Test: `tests/test_downloaders.py` (new `TestHfRadarRegions` class)

**Interfaces:**
- Produces: `HFR_REGIONS: Dict[str, Dict[str, object]]` (keys: region name;
  values: `{"bbox": (min_lon, max_lon, min_lat, max_lat), "has_latest": bool}`),
  `resolve_hfr_region(min_lon, max_lon, min_lat, max_lat) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# Added near the other downloader-utility tests in tests/test_downloaders.py
from sar_validation.downloaders._hf_radar_regions import HFR_REGIONS, resolve_hfr_region


class TestHfRadarRegions:
    def test_us_east_gulf_bbox_resolves(self):
        assert resolve_hfr_region(-90.0, -60.0, 30.0, 40.0) == "US-EastGulfCoast"

    def test_us_west_coast_bbox_resolves(self):
        assert resolve_hfr_region(-125.0, -119.0, 33.0, 38.0) == "US-WestCoast"

    def test_no_overlap_raises_with_region_list(self):
        with pytest.raises(ValueError, match="US-EastGulfCoast"):
            resolve_hfr_region(100.0, 105.0, -10.0, -5.0)  # nowhere near any region

    def test_picks_largest_overlap_when_bbox_spans_two_regions(self):
        # A bbox mostly inside US-WestCoast but touching PLOCAN-scale noise
        # should not happen in practice; this instead checks the tie-break
        # is deterministic for a bbox fully inside exactly one region.
        assert resolve_hfr_region(-124.0, -122.0, 36.0, 37.0) == "US-WestCoast"

    def test_all_regions_have_bbox_and_flag(self):
        assert len(HFR_REGIONS) == 25
        for name, cfg in HFR_REGIONS.items():
            assert len(cfg["bbox"]) == 4
            assert isinstance(cfg["has_latest"], bool)

    def test_regions_without_latest_feed(self):
        no_latest = {n for n, c in HFR_REGIONS.items() if not c["has_latest"]}
        assert no_latest == {
            "ARPAS", "COSYNA", "Finnmark", "US-Alaska",
            "US-EastGulfCoast", "US-Hawaii", "WHub",
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_downloaders.py::TestHfRadarRegions -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sar_validation.downloaders._hf_radar_regions'`

- [ ] **Step 3: Write the implementation**

```python
# sar_validation/downloaders/_hf_radar_regions.py
"""
Shared Copernicus Marine HF-radar-total region table.

Bounding boxes and the "has_latest" flag were read from
``copernicusmarine.describe(dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr")``
on 2026-07-15: each ``<latest|monthly>-radar-total--<Region>`` dataset_part's
declared variable bbox is the region's real grid extent. ``monthly-radar-total``
covers all 25 regions; ``latest-radar-total`` only exists for the 18 with a
near-real-time feed.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["HFR_REGIONS", "resolve_hfr_region"]

_NO_LATEST = {
    "ARPAS", "COSYNA", "Finnmark", "US-Alaska",
    "US-EastGulfCoast", "US-Hawaii", "WHub",
}

_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "ARPAS": (8.164931297302246, 8.878068923950195, 40.756874084472656, 41.243125915527344),
    "CALYPSO": (13.69489860534668, 15.365100860595703, 35.742252349853516, 36.87774658203125),
    "COSYNA": (5.916669845581055, 8.970867156982422, 53.41814422607422, 55.20000076293945),
    "DeltaEbro": (0.06353014707565308, 2.078089952468872, 39.59859848022461, 41.219600677490234),
    "EUSKOOS": (-3.1965925693511963, -1.2034072875976562, 43.31999969482422, 44.58000183105469),
    "Finnmark": (18.042926788330078, 35.45707321166992, 69.00939178466797, 73.49060821533203),
    "Galicia": (-11.327714920043945, -7.9691948890686035, 40.35469055175781, 44.67578887939453),
    "Gibraltar": (-5.8481950759887695, -4.994204998016357, 35.805084228515625, 36.1926155090332),
    "GoS": (13.072955131530762, 15.999811172485352, 39.672447204589844, 41.38346862792969),
    "Granitola": (12.176433563232422, 13.496066093444824, 37.00240707397461, 37.59709167480469),
    "ICATMAR": (1.0088672637939453, 4.291132926940918, 40.50752258300781, 42.99247741699219),
    "Ibiza": (0.5038551688194275, 1.4006848335266113, 38.3229866027832, 39.1067008972168),
    "Lisboa": (-10.595585823059082, -8.704414367675781, 37.90182113647461, 38.8981819152832),
    "NAdr": (13.375, 13.780560493469238, 45.526851654052734, 45.783329010009766),
    "PLOCAN": (-15.420296669006348, -14.89370346069336, 27.85053062438965, 28.34457015991211),
    "Skagerrak": (7.502049922943115, 11.997950553894043, 57.011016845703125, 59.488983154296875),
    "South": (-9.593195915222168, -6.206803798675537, 36.005245208740234, 37.19475555419922),
    "TirLig": (7.50698709487915, 10.493012428283691, 43.25399398803711, 44.49600601196289),
    "US-Alaska": (-174.09815979003906, -128.659912109375, 68.0091781616211, 74.0321044921875),
    "US-EastGulfCoast": (-97.88055419921875, -57.2345085144043, 21.75611114501953, 46.47426986694336),
    "US-Hawaii": (-163.11717224121094, -152.01162719726562, 16.224061965942383, 24.895137786865234),
    "US-PuertoRicoVirginIslands": (-70.49944305419922, -61.024757385253906, 14.508465766906738, 21.989192962646484),
    "US-WestCoast": (-130.33265686035156, -115.83291625976562, 30.25959587097168, 49.982444763183594),
    "Vestlandet": (1.0030561685562134, 8.496943473815918, 56.50277328491211, 63.99722671508789),
    "WHub": (-6.13332986831665, -5.0837883949279785, 50.21573257446289, 51.01667022705078),
}

HFR_REGIONS: Dict[str, Dict[str, object]] = {
    name: {"bbox": bbox, "has_latest": name not in _NO_LATEST}
    for name, bbox in _BBOXES.items()
}


def resolve_hfr_region(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> str:
    """Return the HFR_REGIONS name whose bbox overlaps the request the most.

    Raises ``ValueError`` if no known region overlaps the request bbox at all.
    """
    best_name = None
    best_area = 0.0
    for name, cfg in HFR_REGIONS.items():
        r_min_lon, r_max_lon, r_min_lat, r_max_lat = cfg["bbox"]
        overlap_lon = min(max_lon, r_max_lon) - max(min_lon, r_min_lon)
        overlap_lat = min(max_lat, r_max_lat) - max(min_lat, r_min_lat)
        if overlap_lon > 0 and overlap_lat > 0:
            area = overlap_lon * overlap_lat
            if area > best_area:
                best_area = area
                best_name = name
    if best_name is None:
        raise ValueError(
            f"No Copernicus HF-radar region overlaps bbox lon[{min_lon},{max_lon}] "
            f"lat[{min_lat},{max_lat}]. Known regions: {sorted(HFR_REGIONS)}"
        )
    return best_name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_downloaders.py::TestHfRadarRegions -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/_hf_radar_regions.py tests/test_downloaders.py
git commit -m "feat: add shared Copernicus HF-radar region-resolution table"
```

---

## Task 2: Rewrite `HFRadarDownloader` to query the gridded radar-total parts

**Files:**
- Modify: `sar_validation/downloaders/hf_radar_downloader.py` (full rewrite of
  `download()`/`_download_with_part()`; keep the class name, constructor
  signature, and module docstring conventions)
- Test: `tests/test_downloaders.py` (new `TestHFRadarDownloaderGrid` class)

**Interfaces:**
- Consumes: `resolve_hfr_region`, `HFR_REGIONS` (Task 1);
  `normalize_datetime`, `is_date_recent` (`sar_validation/downloaders/base.py`,
  already exist).
- Produces: `HFRadarDownloader.download(min_lon, max_lon, min_lat, max_lat,
  start, end) -> Optional[Path]` — same signature as before, orchestrator
  call site (`sar_validation/core/orchestrator.py:256-266`) is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# Added to tests/test_downloaders.py
class TestHFRadarDownloaderGrid:
    def test_dry_run_prints_resolved_region_and_part(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-06-05", "2026-06-06")
        assert out is None
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "radar-total--US-EastGulfCoast" in captured

    def test_download_calls_subset_with_resolved_region_part(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            # Simulate copernicusmarine writing the requested file.
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert out is not None
        assert out.exists()
        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "monthly-radar-total--US-EastGulfCoast"
        assert kwargs["minimum_longitude"] == -90.0
        assert kwargs["maximum_longitude"] == -60.0

    def test_recent_date_uses_latest_part_when_region_has_one(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from datetime import datetime, timedelta, timezone
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        recent_end = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        # US-WestCoast has a `latest` feed (unlike US-EastGulfCoast).
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-125.0, -119.0, 33.0, 38.0, recent_start, recent_end)

        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"

    def test_constructor_accepts_unused_depth_kwargs(self, tmp_path):
        # The orchestrator always passes min_depth/max_depth; the gridded
        # product has no depth axis, but the kwargs must still be accepted.
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        HFRadarDownloader(output_dir=tmp_path, dry_run=True, min_depth=-2.0, max_depth=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_downloaders.py::TestHFRadarDownloaderGrid -v`
Expected: FAIL (dry-run output doesn't mention "US-EastGulfCoast"; `subset()`
called with the old `dataset_part="monthly"`/CSV-shaped kwargs, not
`dataset_part="monthly-radar-total--US-EastGulfCoast"`).

- [ ] **Step 3: Write the implementation**

Replace the whole body of `sar_validation/downloaders/hf_radar_downloader.py`
from the `DATASET_ID`/`HF_RADAR_VARS` constants down through
`_download_with_part` with:

```python
"""
Download HF-radar surface-current *grids* from Copernicus Marine.

Data source: INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030
    Dataset ID: cmems_obs-ins_glo_phybgcwav_mynrt_na_irr
    dataset_part: "<latest|monthly>-radar-total--<Region>"

The dataset's plain "latest"/"monthly" parts are a *sparse per-platform*
in-situ feed (moorings/buoys/drifters/ferrybox/tide gauges) that carries no
HF-radar rows at all — verified empty for any bbox/time/variable combination
tried on 2026-07-15. HF-radar current data is delivered separately, as a
regular (time, lat, lon) grid per named coastal region, via
"<latest|monthly>-radar-total--<Region>" dataset_parts. This downloader
resolves the request bbox to one of those named regions and subsets that
grid directly.

Library usage::

    from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader
    dl = HFRadarDownloader(output_dir=Path("data/run1/hf_radar"))
    dl.download(min_lon=-90, max_lon=-60, min_lat=30, max_lat=40,
                start="2026-06-05", end="2026-06-06")

CLI usage::

    python -m sar_validation.downloaders.hf_radar_downloader \\
        --min-lon -90 --max-lon -60 --min-lat 30 --max-lat 40 \\
        --start 2026-06-05 --end 2026-06-06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .base import normalize_datetime, is_date_recent, build_output_dir
from ._hf_radar_regions import HFR_REGIONS, resolve_hfr_region

__all__ = ["HFRadarDownloader"]

DATASET_ID = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"


def _build_filename(region: str, start_dt: str, end_dt: str) -> str:
    start_d = start_dt.split("T")[0]
    end_d = end_dt.split("T")[0]
    date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
    return f"{DATASET_ID}_radar-total_{region}_{date_str}.nc"


class HFRadarDownloader:
    """
    Download a Copernicus Marine HF-radar current grid for the region that
    overlaps the request bbox.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded NetCDF.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    min_depth, max_depth : float
        Accepted for interface compatibility with the orchestrator's
        recipe-level depth-resolution machinery. The HF-radar-total grid has
        no depth axis (it's a fixed near-surface radar measurement), so
        these are unused.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        min_depth: float = -2.0,
        max_depth: float = 2.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> Optional[Path]:
        region = resolve_hfr_region(min_lon, max_lon, min_lat, max_lat)
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        use_latest = HFR_REGIONS[region]["has_latest"] and is_date_recent(end_dt)
        dataset_part = f"{'latest' if use_latest else 'monthly'}-radar-total--{region}"
        filename = _build_filename(region, start_dt, end_dt)
        dest_path = self.output_dir / filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download Copernicus HF-radar grid for region "
                f"'{region}' (dataset_part='{dataset_part}') to:\n  {dest_path}"
            )
            return None

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for HF radar downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading Copernicus HF-radar surface-current grid …")
        print(f"  Region: {region}")
        print(f"  BBox:   lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Dataset part: {dataset_part}")

        try:
            self._subset_with_part(
                copernicusmarine, dataset_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt, dest_path,
            )
        except Exception as e:
            error_msg = str(e)
            if use_latest and "exceed the dataset coordinates" in error_msg:
                dataset_part = f"monthly-radar-total--{region}"
                print(f"  Retrying with dataset_part='{dataset_part}' due to: {error_msg[:120]}…")
                self._subset_with_part(
                    copernicusmarine, dataset_part,
                    min_lon, max_lon, min_lat, max_lat,
                    start_dt, end_dt, dest_path,
                )
            else:
                raise

        if not dest_path.exists():
            raise FileNotFoundError(
                f"Copernicus HF-radar grid download completed but produced no "
                f"file for region '{region}' in [{start_dt}, {end_dt}] "
                f"(dataset_part='{dataset_part}')."
            )

        print(f"  Saved to {dest_path}")
        return dest_path

    def _subset_with_part(
        self, copernicusmarine, dataset_part,
        min_lon, max_lon, min_lat, max_lat,
        start_dt, end_dt, dest_path,
    ) -> None:
        # No `variables=` filter: omitting it makes copernicusmarine return
        # every variable in the dataset_part (verified live — 14 vars for
        # *-radar-total--<Region>, including EWCS/NSCS standard deviations
        # and all *_QC/QCflag fields), so the converter (Task 3) always has
        # the full ancillary set to pick from on disk.
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_part=dataset_part,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            output_directory=str(dest_path.parent),
            output_filename=dest_path.name,
            force_download=True,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a Copernicus Marine HF-radar current grid.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon = params["minimum_longitude"]
        max_lon = params["maximum_longitude"]
        min_lat = params["minimum_latitude"]
        max_lat = params["maximum_latitude"]
        start = params["start_datetime"]
        end = params["end_datetime"]
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "hf_radar"
    )

    dl = HFRadarDownloader(output_dir=output_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_downloaders.py::TestHFRadarDownloaderGrid -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `python -m pytest tests/test_downloaders.py -v`
Expected: All PASS, including
`TestOrchestratorDepthResolution::test_hf_radar_dispatch_uses_default_depth_when_unspecified`
and `test_hf_radar_dispatch_honours_explicit_depth` (they only assert on the
constructor kwargs, which are unchanged).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/hf_radar_downloader.py tests/test_downloaders.py
git commit -m "fix: query Copernicus HF-radar gridded radar-total dataset_parts instead of the empty sparse feed"
```

---

## Task 3: Generalize `from_hf_radar_grid` for Copernicus variable/ancillary names

**Files:**
- Modify: `sar_validation/core/datatree_converter.py:797-912`
  (`from_hf_radar_grid`)
- Test: `tests/test_datatree_converter.py` (extend `TestFromHfRadarGrid`,
  reusing the `_make_hfr_grid_nc` helper pattern at line 984)

**Interfaces:**
- Produces: `DataTreeConverter.from_hf_radar_grid(nc_path, u_var="water_u",
  v_var="water_v", source_label="NOAA HFRnet RTV") -> Optional[xr.Dataset]`
  — defaults are unchanged so all existing NOAA call sites and tests keep
  working with zero edits.

- [ ] **Step 1: Write the failing test**

```python
# Added to tests/test_datatree_converter.py, near _make_hfr_grid_nc (line 984)
def _make_copernicus_hfr_grid_nc(tmp_path, n_time=2, n_lat=3, n_lon=4):
    """Write a minimal Copernicus radar-total-shaped gridded NetCDF."""
    rng = np.random.default_rng(11)
    times = pd.date_range("2026-06-05T00:00:00", periods=n_time, freq="1h").values
    lats = np.linspace(30.0, 40.0, n_lat)
    lons = np.linspace(-90.0, -60.0, n_lon)
    shape = (n_time, n_lat, n_lon)
    ds = xr.Dataset(
        {
            "EWCT": (("time", "latitude", "longitude"), rng.uniform(-0.6, 0.6, shape),
                     {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"}),
            "NSCT": (("time", "latitude", "longitude"), rng.uniform(-0.6, 0.6, shape),
                     {"standard_name": "northward_sea_water_velocity", "units": "m s-1"}),
            "GDOP": (("time", "latitude", "longitude"), rng.uniform(0, 2, shape)),
            "EWCS": (("time", "latitude", "longitude"), rng.uniform(0, 0.1, shape)),
            "NSCS": (("time", "latitude", "longitude"), rng.uniform(0, 0.1, shape)),
            "QCflag": (("time", "latitude", "longitude"), rng.integers(0, 2, shape).astype(float)),
            "CSPD_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "DDNS_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "GDOP_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "VART_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "POSITION_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={"title": "Copernicus HFR radar-total", "institution": "HFR-EU"},
    )
    path = tmp_path / "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr_radar-total_US-EastGulfCoast_2026-06-05.nc"
    ds.to_netcdf(path)
    return path


class TestFromHfRadarGridCopernicus:
    def test_reads_ewct_nsct_directly(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(
            _make_copernicus_hfr_grid_nc(tmp_path), u_var="EWCT", v_var="NSCT",
            source_label="Copernicus Marine HFR radar-total",
        )
        assert ds is not None
        assert "EWCT" in ds and "NSCT" in ds

    def test_data_type_tag_is_grid(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(
            _make_copernicus_hfr_grid_nc(tmp_path), u_var="EWCT", v_var="NSCT",
            source_label="Copernicus Marine HFR radar-total",
        )
        assert ds.attrs["data_type"] == "hf_radar_grid"
        assert ds.attrs["source"] == "Copernicus Marine HFR radar-total"

    def test_retains_copernicus_ancillary_fields(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(
            _make_copernicus_hfr_grid_nc(tmp_path), u_var="EWCT", v_var="NSCT",
            source_label="Copernicus Marine HFR radar-total",
        )
        assert "hfr_gdop" in ds
        assert "hfr_ewcs" in ds and "hfr_nscs" in ds
        assert "hfr_qc" in ds  # overall QCflag

    def test_retains_per_parameter_qc_flags(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(
            _make_copernicus_hfr_grid_nc(tmp_path), u_var="EWCT", v_var="NSCT",
            source_label="Copernicus Marine HFR radar-total",
        )
        for name in ("hfr_qc_cspd", "hfr_qc_ddns", "hfr_qc_gdop",
                     "hfr_qc_vart", "hfr_qc_position"):
            assert name in ds, f"{name} missing from converted dataset"

    def test_noaa_default_args_unaffected(self, tmp_path):
        # Default u_var/v_var must still resolve NOAA's water_u/water_v.
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert ds is not None
        assert "EWCT" in ds and ds.attrs["source"] == "NOAA HFRnet RTV"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datatree_converter.py::TestFromHfRadarGridCopernicus -v`
Expected: FAIL with `TypeError: from_hf_radar_grid() got an unexpected keyword argument 'u_var'`

- [ ] **Step 3: Write the implementation**

In `sar_validation/core/datatree_converter.py`, replace the
`from_hf_radar_grid` signature and body (lines 797–912) with:

```python
    @staticmethod
    def from_hf_radar_grid(
        nc_path: Union[str, Path],
        u_var: str = "water_u",
        v_var: str = "water_v",
        source_label: str = "NOAA HFRnet RTV",
    ) -> Optional[xr.Dataset]:
        """
        Open a gridded HF-radar-current NetCDF (dims ``time, lat, lon``) and
        return a standardised point-frame Dataset tagged
        ``data_type="hf_radar_grid"``.

        ``u_var``/``v_var`` name the eastward/northward current variables on
        the wire — NOAA HFRnet RTV ships ``water_u``/``water_v`` (the
        defaults); Copernicus Marine HFR radar-total products ship
        ``EWCT``/``NSCT`` directly.

        The regular grid is flattened to a ``point`` dimension (one point per
        cell per time) so it collocates through the ``layer_vs_layer`` path,
        exactly like the scatterometer converter. Ancillary uncertainty/QC
        fields are *retained but not used* (design §3.7): NOAA's
        ``DOPx``/``DOPy`` are combined into ``hfr_gdop``, its radial/site
        counts are kept as ``hfr_n_radials``/``hfr_n_sites``; Copernicus's
        ``GDOP`` is copied to ``hfr_gdop`` directly, ``EWCS``/``NSCS`` (the
        per-cell eastward/northward current standard deviations) to
        ``hfr_ewcs``/``hfr_nscs``, the overall ``QCflag`` to ``hfr_qc``, and
        each per-parameter QC flag (``CSPD_QC``, ``DDNS_QC``, ``GDOP_QC``,
        ``VART_QC``, ``POSITION_QC``) to its own ``hfr_qc_<param>`` field —
        so the deferred correction/QC phase can filter on them later.

        Returns
        -------
        xr.Dataset or None
            Dataset with ``data_type="hf_radar_grid"``, or None on failure.
        """
        nc_path = Path(nc_path)
        if not nc_path.exists():
            logger.warning("NetCDF not found: %s", nc_path)
            return None
        try:
            raw = xr.open_dataset(nc_path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", nc_path, exc)
            return None

        # Resolve coordinate names (tolerate lat/lon, latitude/longitude, LAT/LON).
        lat_name = next((n for n in ("lat", "latitude", "LAT") if n in raw.coords or n in raw), None)
        lon_name = next((n for n in ("lon", "longitude", "LON") if n in raw.coords or n in raw), None)
        time_name = next((n for n in ("time", "Time", "TIME") if n in raw.coords or n in raw), None)
        if not (lat_name and lon_name and u_var in raw and v_var in raw):
            logger.warning(
                "from_hf_radar_grid: %s missing lat/lon or %s/%s (have %s)",
                nc_path.name, u_var, v_var, list(raw.coords) + list(raw.data_vars),
            )
            raw.close()
            return None

        lats = np.asarray(raw[lat_name].values, dtype=float)
        lons = np.asarray(raw[lon_name].values, dtype=float)
        if time_name is not None:
            times = pd.to_datetime(raw[time_name].values)
        else:
            times = pd.to_datetime([np.datetime64("NaT")])

        n_t, n_la, n_lo = len(times), len(lats), len(lons)

        # Broadcast (time, lat, lon) → flat point vectors.
        tt, la, lo = np.meshgrid(np.arange(n_t), lats, lons, indexing="ij")
        time_flat = np.repeat(times.values, n_la * n_lo)
        lat_flat = la.ravel()
        lon_flat = ((lo.ravel() + 180.0) % 360.0) - 180.0  # normalise to −180..180

        def _flat(varname):
            return np.asarray(raw[varname].values, dtype=float).reshape(n_t, n_la, n_lo).ravel()

        ewct = _flat(u_var)
        nsct = _flat(v_var)

        data_vars: Dict[str, tuple] = {
            "EWCT": ("point", ewct),
            "NSCT": ("point", nsct),
        }
        var_attrs: Dict[str, Dict] = {
            "EWCT": dict(raw[u_var].attrs),
            "NSCT": dict(raw[v_var].attrs),
        }

        # --- Retained ancillary fields (not used yet; design §3.7) ---
        if "DOPx" in raw and "DOPy" in raw:
            dopx, dopy = _flat("DOPx"), _flat("DOPy")
            data_vars["hfr_gdop"] = ("point", np.sqrt(dopx ** 2 + dopy ** 2))
            var_attrs["hfr_gdop"] = {
                "long_name": "geometric dilution of precision (sqrt(DOPx^2+DOPy^2))",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
        for src, dst in (("number_of_radials", "hfr_n_radials"),
                         ("number_of_sites", "hfr_n_sites")):
            if src in raw:
                data_vars[dst] = ("point", _flat(src))
                var_attrs[dst] = {"long_name": src.replace("_", " "),
                                  "comment": "Retained for a future HF-radar QC filter."}
        if "GDOP" in raw and "hfr_gdop" not in data_vars:
            data_vars["hfr_gdop"] = ("point", _flat("GDOP"))
            var_attrs["hfr_gdop"] = {
                "long_name": "geometric dilution of precision",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
        if "EWCS" in raw and "NSCS" in raw:
            data_vars["hfr_ewcs"] = ("point", _flat("EWCS"))
            data_vars["hfr_nscs"] = ("point", _flat("NSCS"))
            var_attrs["hfr_ewcs"] = {
                "long_name": "eastward current component std error", "units": "m s-1",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
            var_attrs["hfr_nscs"] = {
                "long_name": "northward current component std error", "units": "m s-1",
                "comment": "Retained for a future HF-radar QC/uncertainty filter.",
            }
        if "QCflag" in raw:
            data_vars["hfr_qc"] = ("point", _flat("QCflag"))
            var_attrs["hfr_qc"] = {
                "long_name": "HF-radar overall QC flag",
                "comment": "Retained for a future HF-radar QC filter (design §3.7).",
            }
        # Per-parameter QC flags (Copernicus radar-total product): each one
        # is retained under its own field rather than folded into hfr_qc, so
        # a future QC phase can filter per-parameter instead of only on the
        # overall flag.
        for src, param in (
            ("CSPD_QC", "cspd"), ("DDNS_QC", "ddns"), ("GDOP_QC", "gdop"),
            ("VART_QC", "vart"), ("POSITION_QC", "position"),
        ):
            if src in raw:
                dst = f"hfr_qc_{param}"
                data_vars[dst] = ("point", _flat(src))
                var_attrs[dst] = {
                    "long_name": f"HF-radar {param} QC flag",
                    "comment": "Retained for a future HF-radar QC filter (design §3.7).",
                }

        # Drop points where both current components are NaN (masked land/gaps).
        valid = np.isfinite(ewct) | np.isfinite(nsct)
        if not np.any(valid):
            logger.warning("from_hf_radar_grid: all cells NaN in %s.", nc_path.name)
            raw.close()
            return None

        ds = xr.Dataset(
            {k: ("point", v[valid]) for k, (_, v) in data_vars.items()},
            coords={
                "lon": ("point", lon_flat[valid]),
                "lat": ("point", lat_flat[valid]),
                "time": ("point", time_flat[valid]),
            },
        )
        for gattr in ("title", "institution"):
            if raw.attrs.get(gattr):
                ds.attrs[gattr] = str(raw.attrs[gattr])
        apply_cf_metadata(ds, "hf_radar", var_attrs)

        ds.attrs["data_type"]     = "hf_radar_grid"
        ds.attrs["platform_type"] = "radar"
        ds.attrs["source"]        = source_label
        ds.attrs["filename"]      = nc_path.name

        raw.close()
        return ds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_datatree_converter.py::TestFromHfRadarGridCopernicus tests/test_datatree_converter.py::TestFromHfRadarGrid -v`
Expected: All PASS (existing NOAA tests untouched, new Copernicus tests pass).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "feat: generalize from_hf_radar_grid to accept Copernicus EWCT/NSCT + ancillary fields"
```

---

## Task 4: Wire the Copernicus `hf_radar/` grid folder into `convert_downloaded_data`

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`convert_downloaded_data`,
  next to the existing `hfr_noaa` block at lines 2110–2121)
- Test: `tests/test_datatree_converter.py` (extend `TestBuildDatatreeHfrNoaa`
  or add a sibling class)

**Interfaces:**
- Consumes: `DataTreeConverter.from_hf_radar_grid(nc_path, u_var="EWCT",
  v_var="NSCT", source_label="Copernicus Marine HFR radar-total")` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
class TestBuildDatatreeHfRadarCopernicus:
    def test_hf_radar_folder_becomes_validation_node(self, tmp_path):
        base = tmp_path / "run"
        (base / "hf_radar").mkdir(parents=True)
        _make_copernicus_hfr_grid_nc(base / "hf_radar")
        tree = DataTreeConverter.convert_downloaded_data(base, product_type="currents")
        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any("hf_radar" in p and "hf_radar_noaa" not in p for p in node_paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datatree_converter.py::TestBuildDatatreeHfRadarCopernicus -v`
Expected: FAIL — no node path contains `"hf_radar"` (folder is never scanned).

- [ ] **Step 3: Write the implementation**

In `sar_validation/core/datatree_converter.py`, immediately after the
existing `hfr_noaa` block (ends at line 2121), add:

```python
        # Copernicus Marine HF-radar current grid (per-region radar-total
        # product). Same layer-node shape as hfr_noaa; EWCT/NSCT are already
        # the wire variable names so no rename is needed.
        subdir = base_dir / "hf_radar"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(
                    DataTreeConverter.from_hf_radar_grid(
                        nc_path, u_var="EWCT", v_var="NSCT",
                        source_label="Copernicus Marine HFR radar-total",
                    ),
                    nc_path.name,
                )
                if ds is not None:
                    datasets[f"validation/hf_radar/{nc_path.stem}"] = ds
                    logger.info("Converted hf_radar (Copernicus HF-radar grid): %s", nc_path.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_datatree_converter.py::TestBuildDatatreeHfRadarCopernicus -v`
Expected: PASS

- [ ] **Step 5: Run the full converter test file to check for regressions**

Run: `python -m pytest tests/test_datatree_converter.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "feat: discover hf_radar/ Copernicus grid NetCDFs in convert_downloaded_data"
```

---

## Task 5: `HFRadarHistoricalDownloader` for the delayed-mode archive (013_044)

**Files:**
- Create: `sar_validation/downloaders/hf_radar_historical_downloader.py`
- Test: `tests/test_downloaders.py` (new `TestHFRadarHistoricalDownloader`
  class)

**Interfaces:**
- Consumes: `resolve_hfr_region` (Task 1); `normalize_datetime` (`base.py`).
- Produces: `HFRadarHistoricalDownloader.download(min_lon, max_lon, min_lat,
  max_lat, start, end) -> Optional[Path]`, writing a NetCDF with the same
  on-disk shape Task 3's converter expects (dims `time, latitude, longitude`;
  vars `EWCT, NSCT, GDOP, EWCS, NSCS, QCflag`, ...).

- [ ] **Step 1: Write the failing test**

```python
# Added to tests/test_downloaders.py
class TestHFRadarHistoricalDownloader:
    def test_dry_run_prints_resolved_region_and_filename(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2021-06-05", "2021-06-06")
        assert out is None
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc" in captured

    def test_unavailable_region_raises_clear_error(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        # GoS (Italy) has an NRT feed but no delayed-mode archive.
        with pytest.raises(ValueError, match="no delayed-mode HF-radar archive"):
            dl.download(13.5, 15.5, 40.0, 41.0, "2021-01-01", "2021-01-02")

    def test_multi_year_request_not_yet_supported(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(NotImplementedError, match="single calendar year"):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2020-12-30", "2021-01-02")

    def test_download_gets_file_then_subsets_locally(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )
        import xarray as xr
        import numpy as np
        import pandas as pd

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "GDOP": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            from sar_validation.downloaders.hf_radar_historical_downloader import FileGetResult
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01T04:00:00")

        assert out is not None
        assert out.exists()
        result = xr.open_dataset(out)
        assert "time" in result.dims and "latitude" in result.dims and "longitude" in result.dims
        assert "DEPTH" not in result.dims
        assert result.sizes["time"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_downloaders.py::TestHFRadarHistoricalDownloader -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sar_validation.downloaders.hf_radar_historical_downloader'`

- [ ] **Step 3: Write the implementation**

```python
# sar_validation/downloaders/hf_radar_historical_downloader.py
"""
Download delayed-mode (historical) HF-radar surface-current grids from
Copernicus Marine.

Data source: INSITU_GLO_PHY_UV_DISCRETE_MY_013_044
    Dataset ID: cmems_obs-ins_glo_phy-cur_my_radar-total_irr

Unlike the NRT product (see ``hf_radar_downloader.py``), this dataset is not
subsettable server-side: it exposes one bulk NetCDF file per named region
under its "original-files" service (``history/HF/GL_TV_HF_HFR-<Region>_Total[_<YYYY>].nc``),
1979-2026-ish depending on region. This downloader fetches the one matching
region file (cached via ``skip_existing``, since a region's file covers many
years and multiple runs will reuse it), then subsets it locally in xarray to
the requested time window and bbox, normalizing the on-disk shape (uppercase
OceanSITES dims + a singleton DEPTH axis) to match the NRT grid downloader's
output so both share the same converter path.

Library usage::

    from sar_validation.downloaders.hf_radar_historical_downloader import (
        HFRadarHistoricalDownloader,
    )
    dl = HFRadarHistoricalDownloader(output_dir=Path("data/run1/hf_radar_historical"))
    dl.download(min_lon=-90, max_lon=-60, min_lat=30, max_lat=40,
                start="2021-06-05", end="2021-06-06")

CLI usage::

    python -m sar_validation.downloaders.hf_radar_historical_downloader \\
        --min-lon -90 --max-lon -60 --min-lat 30 --max-lat 40 \\
        --start 2021-06-05 --end 2021-06-06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .base import normalize_datetime, build_output_dir
from ._hf_radar_regions import resolve_hfr_region

__all__ = ["HFRadarHistoricalDownloader"]

DATASET_ID = "cmems_obs-ins_glo_phy-cur_my_radar-total_irr"

# Regions present in the delayed-mode archive, verified via
# copernicusmarine.get(dataset_id=DATASET_ID, dry_run=True) on 2026-07-15.
# Regions in HFR_REGIONS but absent here (GoS, Granitola, WHub) have no
# historical archive. US-EastGulfCoast is split into one file per year
# (2019-2024); every other region is one file covering its whole record.
_SPLIT_BY_YEAR_REGION = "US-EastGulfCoast"
_SPLIT_BY_YEAR_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)

_REGION_FILENAMES = {
    "ARPAS": "GL_TV_HF_HFR-ARPAS_Total.nc",
    "CALYPSO": "GL_TV_HF_HFR-CALYPSO_Total.nc",
    "COSYNA": "GL_TV_HF_HFR-COSYNA_Total.nc",
    "DeltaEbro": "GL_TV_HF_HFR-DeltaEbro_Total.nc",
    "EUSKOOS": "GL_TV_HF_HFR-EUSKOOS_Total.nc",
    "Finnmark": "GL_TV_HF_HFR-Finnmark_Total.nc",
    "Galicia": "GL_TV_HF_HFR-Galicia_Total.nc",
    "Gibraltar": "GL_TV_HF_HFR-Gibraltar_Total.nc",
    "ICATMAR": "GL_TV_HF_HFR-ICATMAR_Total.nc",
    "Ibiza": "GL_TV_HF_HFR-Ibiza_Total.nc",
    "Lisboa": "GL_TV_HF_HFR-Lisboa_Total.nc",
    "NAdr": "GL_TV_HF_HFR-NAdr_Total.nc",
    "PLOCAN": "GL_TV_HF_HFR-PLOCAN_Total.nc",
    "Skagerrak": "GL_TV_HF_HFR-Skagerrak_Total.nc",
    "South": "GL_TV_HF_HFR-South_Total.nc",
    "TirLig": "GL_TV_HF_HFR-TirLig_Total.nc",
    "US-Alaska": "GL_TV_HF_HFR-US-Alaska_Total.nc",
    "US-Hawaii": "GL_TV_HF_HFR-US-Hawaii_Total.nc",
    "US-PuertoRicoVirginIslands": "GL_TV_HF_HFR-US-PuertoRicoVirginIslands_Total.nc",
    "US-WestCoast": "GL_TV_HF_HFR-US-WestCoast_Total.nc",
    "Vestlandet": "GL_TV_HF_HFR-Vestlandet_Total.nc",
}


def _region_filename(region: str, start_dt: str, end_dt: str) -> str:
    if region == _SPLIT_BY_YEAR_REGION:
        start_year = int(start_dt[:4])
        end_year = int(end_dt[:4])
        if start_year != end_year:
            raise NotImplementedError(
                f"{_SPLIT_BY_YEAR_REGION}'s historical archive is split into one "
                "file per year; requests spanning more than a single calendar "
                f"year (got {start_dt} .. {end_dt}) are not yet supported."
            )
        if start_year not in _SPLIT_BY_YEAR_YEARS:
            raise ValueError(
                f"No {_SPLIT_BY_YEAR_REGION} historical archive for year {start_year}; "
                f"available years: {_SPLIT_BY_YEAR_YEARS}"
            )
        return f"GL_TV_HF_HFR-US-EastGulfCoast_Total_{start_year}.nc"
    if region not in _REGION_FILENAMES:
        raise ValueError(
            f"No delayed-mode HF-radar archive for region '{region}'. "
            f"Available: {sorted(_REGION_FILENAMES) + [_SPLIT_BY_YEAR_REGION]}"
        )
    return _REGION_FILENAMES[region]


class HFRadarHistoricalDownloader:
    """
    Download and locally subset a Copernicus Marine delayed-mode HF-radar
    current grid (dataset 013_044) for the region overlapping the request bbox.

    Parameters
    ----------
    output_dir : Path
        Directory to save the subsetted, normalized NetCDF.
    dry_run : bool
        If True, print what would be downloaded without fetching anything.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> Optional[Path]:
        region = resolve_hfr_region(min_lon, max_lon, min_lat, max_lat)
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        remote_filename = _region_filename(region, start_dt, end_dt)

        start_d = start_dt.split("T")[0]
        end_d = end_dt.split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
        dest_path = self.output_dir / f"{DATASET_ID}_{region}_{date_str}.nc"

        if self.dry_run:
            print(
                f"[DRY RUN] Would fetch Copernicus HF-radar historical archive "
                f"'{remote_filename}' for region '{region}' and subset to:\n  {dest_path}"
            )
            return None

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for HF radar downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc
        import xarray as xr

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_cache_dir = self.output_dir / "_raw_archive"
        raw_cache_dir.mkdir(parents=True, exist_ok=True)

        print("Fetching Copernicus HF-radar delayed-mode archive …")
        print(f"  Region: {region}")
        print(f"  Archive file: {remote_filename}")
        resp = copernicusmarine.get(
            dataset_id=DATASET_ID,
            filter=f"*{remote_filename}",
            output_directory=str(raw_cache_dir),
            no_directories=True,
            skip_existing=True,
            disable_progress_bar=True,
        )
        if not resp.files:
            raise FileNotFoundError(
                f"No archive file matched '{remote_filename}' for region '{region}'."
            )
        raw_path = Path(resp.files[0].file_path)

        raw = xr.open_dataset(raw_path)
        try:
            # Keep EWCT/NSCT plus every ancillary uncertainty/QC field the
            # converter (Task 3) knows how to retain — standard deviations
            # (EWCS/NSCS), the geometric-dilution field (GDOP), the overall
            # QCflag, and each per-parameter QC flag — whichever of these
            # this archive file actually has.
            _ancillary_vars = (
                "GDOP", "EWCS", "NSCS", "QCflag",
                "CSPD_QC", "DDNS_QC", "GDOP_QC", "VART_QC", "POSITION_QC",
            )
            normalized = (
                raw[["EWCT", "NSCT"] + [v for v in _ancillary_vars if v in raw]]
                .squeeze("DEPTH", drop=True)
                .rename({"TIME": "time", "LATITUDE": "latitude", "LONGITUDE": "longitude"})
                .sortby(["latitude", "longitude"])
                .sel(
                    time=slice(start_dt, end_dt),
                    latitude=slice(min_lat, max_lat),
                    longitude=slice(min_lon, max_lon),
                )
            )
            if normalized.sizes.get("time", 0) == 0:
                raise FileNotFoundError(
                    f"Copernicus HF-radar historical archive for region '{region}' has "
                    f"no data in [{start_dt}, {end_dt}]."
                )
            normalized.to_netcdf(dest_path)
        finally:
            raw.close()

        print(f"  Saved to {dest_path}")
        return dest_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a Copernicus Marine delayed-mode HF-radar current grid.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon = params["minimum_longitude"]
        max_lon = params["maximum_longitude"]
        min_lat = params["minimum_latitude"]
        max_lat = params["maximum_latitude"]
        start = params["start_datetime"]
        end = params["end_datetime"]
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "hf_radar_historical"
    )

    dl = HFRadarHistoricalDownloader(output_dir=output_dir, dry_run=args.dry_run)
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
    )


if __name__ == "__main__":
    main()
```

Note on the test's `FileGetResult` helper: it's a minimal stand-in for
`copernicusmarine.get()`'s real response object (which the test mocks out
entirely), not a symbol the implementation needs to define or import — add a
tiny local dataclass in the test file itself:

```python
# In tests/test_downloaders.py, near TestHFRadarHistoricalDownloader
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class FileGetResult:
    files: List[Any] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_downloaders.py::TestHFRadarHistoricalDownloader -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/hf_radar_historical_downloader.py tests/test_downloaders.py
git commit -m "feat: add HFRadarHistoricalDownloader for the Copernicus delayed-mode HF-radar archive (013_044)"
```

---

## Task 6: Orchestrator wiring + converter folder discovery for `hf_radar_historical`

**Files:**
- Modify: `sar_validation/core/orchestrator.py` (add `_download_hf_radar_historical`,
  register in `_dispatch_source`'s `handlers` dict at line 204-210)
- Modify: `sar_validation/core/datatree_converter.py` (add a folder-discovery
  block for `hf_radar_historical/`, mirroring Task 4)
- Test: `tests/test_downloaders.py` (extend `TestOrchestratorDepthResolution`
  or add a sibling dispatch test), `tests/test_datatree_converter.py`

**Interfaces:**
- Consumes: `HFRadarHistoricalDownloader` (Task 5); `from_hf_radar_grid`
  (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# Added to tests/test_downloaders.py
class TestOrchestratorHFRadarHistoricalWiring:
    def test_dispatch_source_registers_hf_radar_historical_handler(self):
        from unittest.mock import patch
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="currents"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar_historical")

        with patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        mock_cls.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_downloaders.py::TestOrchestratorHFRadarHistoricalWiring -v`
Expected: FAIL — `_dispatch_source` logs `"No downloader for source_type
'hf_radar_historical'"` and returns `False`.

- [ ] **Step 3: Write the implementation**

In `sar_validation/core/orchestrator.py`, add `"hf_radar_historical"` to the
`handlers` dict in `_dispatch_source` (line 204-210):

```python
    def _dispatch_source(self, source) -> bool:
        handlers = {
            "scatterometer": self._download_scatterometer,
            "hf_radar":      self._download_hf_radar,
            "hf_radar_noaa": self._download_noaa_hfradar,
            "hf_radar_historical": self._download_hf_radar_historical,
            "altimeter":     self._download_altimeter,
            "radiometer":    self._download_radiometer,
        }
```

Then add the handler method right after `_download_noaa_hfradar` (after line
312):

```python
    def _download_hf_radar_historical(self, source) -> bool:
        from ..downloaders.hf_radar_historical_downloader import HFRadarHistoricalDownloader

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "hf_radar_historical"

        try:
            dl = HFRadarHistoricalDownloader(output_dir=out_dir, dry_run=self.dry_run)
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
            self.metadata["downloads"]["hf_radar_historical"] = {
                "status": "dry_run" if self.dry_run else "success",
            }
            return True
        except Exception as exc:
            msg = f"HF radar historical download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_historical"] = {"status": "failed", "error": msg}
            return False
```

In `sar_validation/core/datatree_converter.py`, immediately after the
`hf_radar` block added in Task 4, add:

```python
        # Copernicus Marine delayed-mode HF-radar current grid (013_044),
        # already normalized to the same shape as hf_radar/ by the downloader.
        subdir = base_dir / "hf_radar_historical"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(
                    DataTreeConverter.from_hf_radar_grid(
                        nc_path, u_var="EWCT", v_var="NSCT",
                        source_label="Copernicus Marine HFR radar-total (delayed-mode)",
                    ),
                    nc_path.name,
                )
                if ds is not None:
                    datasets[f"validation/hf_radar_historical/{nc_path.stem}"] = ds
                    logger.info("Converted hf_radar_historical (Copernicus HF-radar grid): %s", nc_path.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_downloaders.py::TestOrchestratorHFRadarHistoricalWiring -v`
Expected: PASS

- [ ] **Step 5: Run the full downloader + converter test files to check for regressions**

Run: `python -m pytest tests/test_downloaders.py tests/test_datatree_converter.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/orchestrator.py sar_validation/core/datatree_converter.py tests/test_downloaders.py
git commit -m "feat: wire hf_radar_historical source through orchestrator and datatree conversion"
```

---

## Task 7: Re-enable `hf_radar` and add `hf_radar_historical` in the CLI currents template

**Files:**
- Modify: `sar_validation/cli.py` (`_build_currents_config`, lines 262-292 —
  currently has the "hf_radar omitted, see comment" block added when the
  source was disabled)
- Modify: `tests/test_recipe.py` (`TestCurrentsTemplate.test_preserves_existing_currents_content`)

**Interfaces:**
- Consumes: nothing new — `ValidationDataSource(source_type=...)` (existing).

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_recipe.py, replace test_preserves_existing_currents_content
    def test_preserves_existing_currents_content(self):
        """The extraction into a builder must not drop any existing sources/specs."""
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=7)
        source_types = [s.source_type for s in recipe.validation_sources]
        assert source_types == [
            "hf_radar", "hf_radar_historical", "hf_radar_noaa",
            "drifter", "ferrybox", "mooring",
        ]

        assert recipe.sar_data.max_downloads == 7

    def test_hf_radar_source_has_no_leftover_depth_kwargs(self):
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        hf_radar_src = next(
            s for s in recipe.validation_sources if s.source_type == "hf_radar"
        )
        # The gridded product has no depth axis; the recipe shouldn't imply one.
        assert hf_radar_src.min_depth is None
        assert hf_radar_src.max_depth is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_recipe.py::TestCurrentsTemplate -v`
Expected: FAIL — `source_types` is currently `["hf_radar_noaa", "drifter", "ferrybox", "mooring"]`.

- [ ] **Step 3: Write the implementation**

In `sar_validation/cli.py`, replace the `validation_sources` block in
`_build_currents_config` (the one currently starting with the "omitted:
hf_radar" comment) with:

```python
        validation_sources=[
            ValidationDataSource(source_type="hf_radar"),
            ValidationDataSource(source_type="hf_radar_historical"),
            ValidationDataSource(
                source_type="hf_radar_noaa",
                min_depth=-2.0, max_depth=2.0,
                download_kwargs={"resolution_km": 6},
            ),
            ValidationDataSource(source_type="drifter"),
            ValidationDataSource(source_type="ferrybox"),
            ValidationDataSource(source_type="mooring"),
        ],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_recipe.py -v`
Expected: All PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/cli.py tests/test_recipe.py
git commit -m "feat: re-enable hf_radar and add hf_radar_historical to the currents recipe template"
```

---

## Task 8: End-to-end validation on real data

**Files:** none (manual verification task, no code changes)

**Interfaces:** none — this task exercises Tasks 1-7 together against the
live Copernicus Marine service.

- [ ] **Step 1: Create a small real recipe**

```bash
sar-validate --create-recipe currents \
  --min-lon -90 --max-lon -60 --min-lat 30 --max-lat 40 \
  --start 2026-04-05 --end 2026-04-06 \
  --recipe-name hfr_copernicus_e2e --limit 2
```

- [ ] **Step 2: Run the download step and confirm `hf_radar` succeeds**

Run: `sar-validate --download recipes/hfr_copernicus_e2e.yaml`
Expected: `download_metadata.json`'s `downloads.hf_radar.status == "success"`
(no longer `"failed"` with "output CSV not found") and a `.nc` file appears
under `<run_dir>/hf_radar/`.

- [ ] **Step 3: Run the full pipeline**

Run: `sar-validate --convert --collocate --stats --plot recipes/hfr_copernicus_e2e.yaml`
Expected: Completes without error; `validation_report.pdf` includes a
currents scatter/geographic panel populated from `hf_radar` collocated pairs
(check the stats JSON for `hf_radar` under `layer_vs_layer` sources with a
non-zero pair count).

- [ ] **Step 4: Spot-check a delayed-mode (historical) date**

Repeat Steps 1–3 with `--start 2021-06-05 --end 2021-06-06` (outside the NRT
window, forcing `hf_radar` to `monthly-radar-total--US-EastGulfCoast` and
`hf_radar_historical` to fetch `GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc`).
Expected: both sources succeed; `hf_radar_historical`'s first run downloads
the ~250 MB region-year archive into `<run_dir>/hf_radar_historical/_raw_archive/`
(cached for subsequent runs via `skip_existing`).

- [ ] **Step 5: Record findings**

Write a short results note to `docs/runs/2026-07-15-phase3c-integration.md`
(pair counts, any warnings, comparison to the Phase 3a NOAA end-to-end
result in `docs/runs/2026-07-14-phase3a-integration.md`), following the
existing convention referenced from `phase3a_implementation_status` memory.

- [ ] **Step 6: Commit**

```bash
git add docs/runs/2026-07-15-phase3c-integration.md
git commit -m "docs: record Phase 3c Copernicus HF-radar end-to-end validation results"
```
