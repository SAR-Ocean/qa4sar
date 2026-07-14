# Phase 3a — NOAA ERDDAP HF-Radar Current Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end validation of NOAA HFRnet gridded surface currents against Sentinel-1 L2 OCN radial velocity, delivering a working US-West current-validation result.

**Architecture:** A new `NOAAHFRadarDownloader` fetches gridded `water_u`/`water_v` via an ERDDAP griddap NetCDF-subset URL. A new `from_hf_radar_grid` converter flattens that grid to the toolbox's canonical point-frame Dataset (`EWCT`/`NSCT` + retained ancillary uncertainty/QC fields), tagged `data_type="hf_radar_grid"`. Collocation routes that tag to the existing `layer_vs_layer` path, and a single shared `_project_currents_to_radial` helper projects `EWCT`/`NSCT` onto the SAR line-of-sight (`rvlHeading − 90°`) in every collocation path. Recipe/CLI/orchestrator wire a new `hf_radar_noaa` source.

**Tech Stack:** Python ≥3.10 (`pyproject` floor; the `sar_validation` conda env runs 3.14.5 — test against that), xarray, pandas, numpy, scipy; ERDDAP griddap over HTTPS; pytest.

## Global Constraints

- **Comparison quantity:** scalar radial projection `rvlRadVel_projection` compared against `rvlRadVel`. No current-direction (HCDT) comparison in this phase.
- **Projection convention:** `radial = EWCT·cos(θ) + NSCT·sin(θ)` where `θ = radians(rvlHeading − 90°)`. Identical in every path (mirrors the existing inline sites).
- **Canonical current codes:** eastward → `EWCT`, northward → `NSCT` (already in `INSITU_VARIABLE_ATTRS` and `_variable_map.VARIABLE_PAIRS`). Do not rename these downstream.
- **No corrections in this phase.** Raw `rvlRadVel` is a radial *velocity* (includes wind-wave artefact bias up to ~2 m/s), not a current; the WASV/instrument correction chain is deferred (see design §3.7). Weak raw agreement is expected and must be documented, not "fixed."
- **Retain, don't use, ancillary fields.** Converters carry uncertainty/QC fields per design §3.7 as an explicit allow-list; Phase 3a never filters or corrects on them.
- **Longitude convention:** normalise to −180…+180 (as every existing converter does).
- **Reference:** Martin, Gommenginger, Jacob & Staneva (2022), *RSE* 268:112758, doi:10.1016/j.rse.2021.112758.
- **Default resolution:** 6 km (robust coverage), overridable to 2/1 km.

---

## File Structure

**Modify:**
- `sar_validation/core/collocation.py` — add `_project_currents_to_radial`; refactor the two inline projection sites; add projection to `_collocate_individual`; register `hf_radar_grid` in `LAYER_DATA_TYPES` and the two layer-type-resolution blocks.
- `sar_validation/core/recipe.py` — add `hf_radar_grid` entry to `DEFAULT_LAYER_TYPE_SPECS`.
- `sar_validation/core/datatree_converter.py` — add `from_hf_radar_grid`; discover a new `hfr_noaa/` download folder in `convert_downloaded_data`.
- `sar_validation/core/_cf_metadata.py` — add `PRODUCT_REFERENCES["hf_radar"]`.
- `sar_validation/core/orchestrator.py` — add `_download_noaa_hfradar` and register `hf_radar_noaa` in `_dispatch_source`.
- `sar_validation/cli.py` — add the `hf_radar_noaa` source + `hf_radar_grid` layer spec to the `currents` recipe template.

**Create:**
- `sar_validation/downloaders/noaa_hfradar_downloader.py` — `NOAAHFRadarDownloader` (ERDDAP griddap backend) + pure helpers.

**Test:**
- `tests/test_collocation.py` — projection helper + `hf_radar_grid` dispatch.
- `tests/test_downloaders.py` — region/resolution→dataset map, ERDDAP URL builder, backend selection.
- `tests/test_datatree_converter.py` — `from_hf_radar_grid` rename/tag/ancillary-retention.

---

## Task 1: Shared radial-projection helper

**Files:**
- Modify: `sar_validation/core/collocation.py` (add module-level helper near the other geometry helpers ~line 96; edit sites at ~599-614, ~1009-1013; add block in `_collocate_individual` ~line 1730)
- Test: `tests/test_collocation.py`

**Interfaces:**
- Produces: `_project_currents_to_radial(ewct: float, nsct: float, heading_deg: float) -> float` (module-level function in `collocation.py`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_collocation.py`:

```python
import numpy as np
from sar_validation.core.collocation import _project_currents_to_radial


class TestProjectCurrentsToRadial:
    def test_due_east_current_zero_heading(self):
        # heading 0° → LOS is heading-90° = -90°; cos(-90)=0, sin(-90)=-1.
        # A 1 m/s eastward current projects to 0; northward projects to -1.
        assert _project_currents_to_radial(1.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
        assert _project_currents_to_radial(0.0, 1.0, 0.0) == pytest.approx(-1.0, abs=1e-12)

    def test_heading_90_east_projects_fully(self):
        # heading 90° → θ=0 → cos=1, sin=0: eastward projects fully, north to 0.
        assert _project_currents_to_radial(1.0, 0.0, 90.0) == pytest.approx(1.0, abs=1e-12)
        assert _project_currents_to_radial(0.0, 1.0, 90.0) == pytest.approx(0.0, abs=1e-12)

    def test_matches_legacy_inline_formula(self):
        ewct, nsct, heading = 0.37, -0.12, 190.0
        expected = (ewct * np.cos(np.radians(heading - 90.0))
                    + nsct * np.sin(np.radians(heading - 90.0)))
        assert _project_currents_to_radial(ewct, nsct, heading) == pytest.approx(expected)
```

Ensure `import pytest` is present at the top of the test file (it is used elsewhere; add if missing).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collocation.py::TestProjectCurrentsToRadial -v`
Expected: FAIL with `ImportError: cannot import name '_project_currents_to_radial'`

- [ ] **Step 3: Add the helper**

In `sar_validation/core/collocation.py`, after the `_haversine_distance` helper (~line 118), add:

```python
def _project_currents_to_radial(ewct: float, nsct: float, heading_deg: float) -> float:
    """Project an eastward/northward current onto the SAR radial (line-of-sight).

    The SAR line-of-sight is the range direction, perpendicular to the platform
    heading ``rvlHeading`` (azimuth/along-track), hence the ``- 90``. The result
    is the quantity compared against the L2 OCN ``rvlRadVel`` product
    (``rvlRadVel_projection``).

    Reference: Martin, Gommenginger, Jacob & Staneva (2022), RSE 268:112758.
    """
    heading_rad = np.radians(heading_deg - 90.0)
    return ewct * np.cos(heading_rad) + nsct * np.sin(heading_rad)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collocation.py::TestProjectCurrentsToRadial -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Refactor the two existing inline sites to call the helper**

In `PointLayerCollocation.collocate` (~lines 606-612), replace the inner computation:

```python
                        if len(valid_headings) > 0:
                            heading_deg = np.nanmean(valid_headings)
                            heading_rad = np.radians(float(heading_deg) - 90.0)
                            ewct = float(val_aggregated["EWCT"])
                            nsct = float(val_aggregated["NSCT"])
                            radial_vel = ewct * np.cos(heading_rad) + nsct * np.sin(heading_rad)
                            val_aggregated["rvlRadVel_projection"] = radial_vel
```

with:

```python
                        if len(valid_headings) > 0:
                            val_aggregated["rvlRadVel_projection"] = _project_currents_to_radial(
                                float(val_aggregated["EWCT"]),
                                float(val_aggregated["NSCT"]),
                                float(np.nanmean(valid_headings)),
                            )
```

In `_collocate_wv_points` (~lines 1009-1013), replace:

```python
            heading_rad = np.radians(float(sar_aggregated["rvlHeading"]) - 90.0)
            val_aggregated["rvlRadVel_projection"] = (
                float(val_aggregated["EWCT"]) * np.cos(heading_rad)
                + float(val_aggregated["NSCT"]) * np.sin(heading_rad)
            )
```

with:

```python
            val_aggregated["rvlRadVel_projection"] = _project_currents_to_radial(
                float(val_aggregated["EWCT"]),
                float(val_aggregated["NSCT"]),
                float(sar_aggregated["rvlHeading"]),
            )
```

- [ ] **Step 6: Add projection to the `individual` layer path**

In `LayerLayerCollocation._collocate_individual`, immediately after the `if not val_aggregated:` block that `continue`s (~line 1739, before `sar_cell_lon = ...`), add:

```python
                # Project currents onto the SAR line-of-sight so gridded
                # HF-radar (EWCT/NSCT) is comparable to rvlRadVel — the
                # cell-averaging path does this via PointLayerCollocation, but
                # the SAR-anchor 'individual' path must do it explicitly.
                if (
                    "rvlRadVel" in sar_aggregated
                    and "rvlHeading" in sar_aggregated
                    and "EWCT" in val_aggregated
                    and "NSCT" in val_aggregated
                ):
                    val_aggregated["rvlRadVel_projection"] = _project_currents_to_radial(
                        float(val_aggregated["EWCT"]),
                        float(val_aggregated["NSCT"]),
                        float(sar_aggregated["rvlHeading"]),
                    )
```

- [ ] **Step 7: Run the full collocation suite to confirm no regression**

Run: `pytest tests/test_collocation.py -v`
Expected: PASS (all existing tests + the 3 new ones)

- [ ] **Step 8: Commit**

```bash
git add sar_validation/core/collocation.py tests/test_collocation.py
git commit -m "refactor: shared _project_currents_to_radial helper across all collocation paths"
```

---

## Task 2: Register `hf_radar_grid` as a layer type

**Files:**
- Modify: `sar_validation/core/collocation.py` (`LAYER_DATA_TYPES` ~line 273; layer-type resolution blocks ~1268-1280 and ~1352-1365)
- Modify: `sar_validation/core/recipe.py` (`DEFAULT_LAYER_TYPE_SPECS` ~line 120)
- Test: `tests/test_collocation.py`

**Interfaces:**
- Consumes: `_detect_collocation_type(val_ds, source_path)` from Task-0 code (existing).
- Produces: a Dataset attr value `data_type="hf_radar_grid"` that resolves to `"layer_vs_layer"` and picks up `DEFAULT_LAYER_TYPE_SPECS["hf_radar_grid"]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_collocation.py`:

```python
import xarray as xr
from sar_validation.core.collocation import _detect_collocation_type
from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS


class TestHfRadarGridDispatch:
    def test_data_type_routes_to_layer(self):
        ds = xr.Dataset(attrs={"data_type": "hf_radar_grid"})
        assert _detect_collocation_type(ds, "validation/hfr_noaa/scene") == "layer_vs_layer"

    def test_path_fallback_routes_to_layer(self):
        ds = xr.Dataset()  # no data_type attr
        assert _detect_collocation_type(ds, "validation/hfr_noaa/scene") == "layer_vs_layer"

    def test_default_layer_spec_present(self):
        spec = DEFAULT_LAYER_TYPE_SPECS["hf_radar_grid"]
        assert spec["aggregation_window_km"] == 6.0
        assert spec["time_tolerance_minutes"] == 20
        assert spec["distance_weighting"] == "equal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collocation.py::TestHfRadarGridDispatch -v`
Expected: FAIL (`hf_radar_grid` not in `LAYER_DATA_TYPES`; `hfr_noaa` not a source path; KeyError on the spec)

- [ ] **Step 3: Add the layer type + spec + path fragment**

In `sar_validation/core/collocation.py`, extend the two constants (~lines 273-275):

```python
LAYER_DATA_TYPES = {"scatterometer", "altimeter", "hf_radar", "hf_radar_grid", "radiometer"}
LAYER_SOURCE_PATHS = {"osi_saf_winds", "scatterometer", "altimeter", "hf_radar", "hf_radar_grid", "hfr_noaa", "radiometer"}
```

In `sar_validation/core/recipe.py`, add to `DEFAULT_LAYER_TYPE_SPECS` (~after line 124):

```python
    "hf_radar_grid":  {"time_tolerance_minutes": 20,  "aggregation_window_km": 6.0,  "distance_weighting": "equal"},
```

In `collocation.py`, in **both** layer-type-resolution fallback blocks (the WV block ~line 1279 and the grid block ~line 1364), add an `hfr_noaa`/`hf_radar_grid` branch alongside the existing `hf_radar` one. In each block, after:

```python
                            elif "hf_radar" in path_parts:
                                layer_type = "hf_radar"
```

add:

```python
                            elif "hfr_noaa" in path_parts or "hf_radar_grid" in path_parts:
                                layer_type = "hf_radar_grid"
```

(When `data_type="hf_radar_grid"` is set on the node, `layer_type` is taken from the attr directly and already matches the `DEFAULT_LAYER_TYPE_SPECS` key; the path branch is only the attribute-absent fallback.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collocation.py::TestHfRadarGridDispatch -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/collocation.py sar_validation/core/recipe.py tests/test_collocation.py
git commit -m "feat: route hf_radar_grid to layer_vs_layer with 6km/20min spec"
```

---

## Task 3: NOAA downloader — pure helpers (dataset map, URL builder, backend selection)

**Files:**
- Create: `sar_validation/downloaders/noaa_hfradar_downloader.py`
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Produces:
  - `select_erddap_dataset(min_lon, max_lon, min_lat, max_lat, resolution_km: int) -> str` (raises `ValueError` for unsupported regions/resolutions).
  - `build_erddap_subset_url(dataset_id: str, min_lon, max_lon, min_lat, max_lat, start: str, end: str) -> str`.
  - `select_backend(end: str) -> str` (returns `"erddap"`; raises `NotImplementedError` for dates older than the ERDDAP window — THREDDS is Phase 3b).
  - Module constants: `ERDDAP_BASE`, `ERDDAP_WINDOW_DAYS = 90`, `DEFAULT_RESOLUTION_KM = 6`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_downloaders.py`:

```python
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from sar_validation.downloaders.noaa_hfradar_downloader import (
    select_erddap_dataset,
    build_erddap_subset_url,
    select_backend,
    ERDDAP_BASE,
)


class TestSelectErddapDataset:
    def test_us_west_6km_default(self):
        assert select_erddap_dataset(-125, -119, 33, 38, 6) == "ucsdHfrW6"

    def test_us_west_2km(self):
        assert select_erddap_dataset(-125, -119, 33, 38, 2) == "ucsdHfrW2"

    def test_us_east_gulf_6km(self):
        assert select_erddap_dataset(-80, -70, 35, 42, 6) == "ucsdHfrE6"

    def test_unsupported_region_raises(self):
        with pytest.raises(ValueError, match="No ERDDAP HF-radar dataset"):
            select_erddap_dataset(2.0, 8.0, 53.0, 55.0, 6)  # German Bight → Phase 3c

    def test_unsupported_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            select_erddap_dataset(-80, -70, 35, 42, 2)  # US-East has no 2 km


class TestBuildErddapSubsetUrl:
    def test_url_has_vars_bbox_and_time_selectors(self):
        url = build_erddap_subset_url(
            "ucsdHfrW6", -125, -119, 33, 38, "2024-05-01", "2024-05-01T06:00:00"
        )
        assert url.startswith(f"{ERDDAP_BASE}/ucsdHfrW6.nc?")
        assert "water_u[(2024-05-01T00:00:00Z):(2024-05-01T06:00:00Z)]" in url
        assert "water_v[(2024-05-01T00:00:00Z):(2024-05-01T06:00:00Z)]" in url
        assert "[(33.0):(38.0)]" in url   # latitude ascending
        assert "[(-125.0):(-119.0)]" in url  # longitude ascending


class TestSelectBackend:
    def test_recent_date_uses_erddap(self):
        recent = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
        assert select_backend(recent) == "erddap"

    def test_old_date_not_yet_supported(self):
        with pytest.raises(NotImplementedError, match="Phase 3b"):
            select_backend("2015-01-01")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_downloaders.py -k "Erddap or Backend" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sar_validation.downloaders.noaa_hfradar_downloader'`

- [ ] **Step 3: Create the module with the pure helpers**

Create `sar_validation/downloaders/noaa_hfradar_downloader.py`:

```python
"""
Download NOAA HFRnet gridded surface currents (Real-Time Velocities, RTV).

Backend for Phase 3a: ERDDAP griddap NetCDF subset (recent ~3-month window).
The THREDDS/OPeNDAP archive backend (2012–present) is Phase 3b.

Data source: NOAA/UCSD HFRnet Regional/National RTV, distributed via ERDDAP
griddap on coastwatch.pfeg.noaa.gov. Variables ``water_u``/``water_v`` carry CF
standard names ``surface_eastward/northward_sea_water_velocity``.

Reference for the SAR-vs-HF-radar comparison this feeds:
Martin, Gommenginger, Jacob & Staneva (2022), RSE 268:112758.

CLI usage::

    python -m sar_validation.downloaders.noaa_hfradar_downloader \\
        --min-lon -125 --max-lon -119 --min-lat 33 --max-lat 38 \\
        --start 2024-05-01 --end 2024-05-02 --resolution 6 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .base import normalize_datetime

__all__ = [
    "NOAAHFRadarDownloader",
    "select_erddap_dataset",
    "build_erddap_subset_url",
    "select_backend",
]

# ERDDAP griddap host serving the UCSD HFRnet RTV datasets.
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# ERDDAP keeps a rolling ~3-month window; older dates need the THREDDS archive
# (Phase 3b).
ERDDAP_WINDOW_DAYS = 90
DEFAULT_RESOLUTION_KM = 6

# Region bounding boxes (lon_min, lon_max, lat_min, lat_max) and their
# resolution → ERDDAP dataset-id maps. Non-CONUS regions (Hawaii, Alaska,
# PR/USVI, Great Lakes) are deferred (design §6) and raise a clear error.
_REGIONS = {
    "US_WEST": {
        "bbox": (-130.0, -116.0, 30.0, 50.0),
        "datasets": {1: "ucsdHfrW1", 2: "ucsdHfrW2", 6: "ucsdHfrW6"},
    },
    "US_EAST_GULF": {
        "bbox": (-98.0, -60.0, 22.0, 46.0),
        "datasets": {1: "ucsdHfrE1", 6: "ucsdHfrE6"},
    },
}


def _bbox_center(min_lon, max_lon, min_lat, max_lat):
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def select_erddap_dataset(min_lon, max_lon, min_lat, max_lat, resolution_km: int) -> str:
    """Choose the ERDDAP RTV dataset id from the request bbox and resolution.

    Raises ``ValueError`` if the region is outside the supported CONUS coasts
    or the requested resolution is unavailable for that region.
    """
    clon, clat = _bbox_center(min_lon, max_lon, min_lat, max_lat)
    for name, cfg in _REGIONS.items():
        lo, hi, la, ha = cfg["bbox"]
        if lo <= clon <= hi and la <= clat <= ha:
            datasets = cfg["datasets"]
            if resolution_km not in datasets:
                raise ValueError(
                    f"resolution {resolution_km} km not available for region "
                    f"{name}; available: {sorted(datasets)} km"
                )
            return datasets[resolution_km]
    raise ValueError(
        "No ERDDAP HF-radar dataset for bbox center "
        f"({clon:.2f}, {clat:.2f}). Phase 3a supports US West and US "
        "East/Gulf coasts; other regions arrive in later phases."
    )


def build_erddap_subset_url(
    dataset_id: str, min_lon, max_lon, min_lat, max_lat, start: str, end: str
) -> str:
    """Build an ERDDAP griddap NetCDF-subset URL for water_u/water_v.

    Dimension order is ``[time][latitude][longitude]``; ERDDAP accepts value
    selectors ``[(min):(max)]`` and returns the enclosing grid subset
    server-side. This URL is also the intended seed for a future
    granule-search / dry-collocation feature.
    """
    t0 = normalize_datetime(start)
    t1 = normalize_datetime(end)
    # ERDDAP wants explicit Z-suffixed ISO timestamps.
    t0 = t0 if t0.endswith("Z") else t0 + "Z"
    t1 = t1 if t1.endswith("Z") else t1 + "Z"
    sel = (
        f"[({t0}):({t1})]"
        f"[({float(min_lat)}):({float(max_lat)})]"
        f"[({float(min_lon)}):({float(max_lon)})]"
    )
    query = f"water_u{sel},water_v{sel}"
    return f"{ERDDAP_BASE}/{dataset_id}.nc?{query}"


def select_backend(end: str) -> str:
    """Pick the download backend from the requested end date.

    Phase 3a implements only the ERDDAP griddap backend (recent window). Dates
    older than ``ERDDAP_WINDOW_DAYS`` require the THREDDS archive backend, which
    is delivered in Phase 3b.
    """
    end_norm = normalize_datetime(end).rstrip("Z")
    end_dt = datetime.fromisoformat(end_norm)
    age_days = (datetime.utcnow() - end_dt).days
    if age_days > ERDDAP_WINDOW_DAYS:
        raise NotImplementedError(
            f"end date {end} is older than the ERDDAP ~{ERDDAP_WINDOW_DAYS}-day "
            "window; the THREDDS/OPeNDAP archive backend is Phase 3b."
        )
    return "erddap"
```

(The `NOAAHFRadarDownloader` class and CLI `main` come in Task 4; this step only needs the helpers to exist so the module imports.)

Add a minimal class stub so `__all__` resolves and the module imports cleanly — it will be fully implemented in Task 4:

```python
class NOAAHFRadarDownloader:
    """See Task 4 for the full implementation."""

    def __init__(self, output_dir: Path, dry_run: bool = False,
                 resolution_km: int = DEFAULT_RESOLUTION_KM) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_downloaders.py -k "Erddap or Backend" -v`
Expected: PASS (all helper tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/noaa_hfradar_downloader.py tests/test_downloaders.py
git commit -m "feat: NOAA HF-radar ERDDAP dataset map, subset-URL builder, backend selection"
```

---

## Task 4: NOAA downloader — `download()` + CLI

**Files:**
- Modify: `sar_validation/downloaders/noaa_hfradar_downloader.py` (flesh out `NOAAHFRadarDownloader`, add `_parse_args`/`main`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `select_backend`, `select_erddap_dataset`, `build_erddap_subset_url` (Task 3).
- Produces: `NOAAHFRadarDownloader(output_dir, dry_run=False, resolution_km=6).download(min_lon, max_lon, min_lat, max_lat, start, end) -> Optional[Path]` — returns the written `.nc` path, or `None` in dry-run.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_downloaders.py`:

```python
from pathlib import Path
from unittest.mock import patch
from sar_validation.downloaders.noaa_hfradar_downloader import NOAAHFRadarDownloader


class TestNOAAHFRadarDownload:
    def test_dry_run_returns_none_and_no_fetch(self, tmp_path, capsys):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, "2024-05-01", "2024-05-01T06:00:00")
        assert out is None
        m.assert_not_called()
        assert "ucsdHfrW6.nc?" in capsys.readouterr().out

    def test_download_fetches_url_to_expected_path(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, "2024-05-01", "2024-05-01T06:00:00")
        assert out is not None
        assert out.parent == tmp_path
        assert out.suffix == ".nc"
        m.assert_called_once()
        called_url, called_path = m.call_args[0][0], m.call_args[0][1]
        assert "ucsdHfrW6.nc?" in called_url
        assert str(out) == str(called_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_downloaders.py::TestNOAAHFRadarDownload -v`
Expected: FAIL (`download` not implemented — `AttributeError` or `TypeError`)

- [ ] **Step 3: Implement `download()` and the CLI**

Replace the `NOAAHFRadarDownloader` stub in `noaa_hfradar_downloader.py` with:

```python
class NOAAHFRadarDownloader:
    """Download NOAA HFRnet gridded RTV currents via ERDDAP griddap.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded NetCDF.
    dry_run : bool
        If True, print the subset URL and return None without fetching.
    resolution_km : int
        Grid resolution (1/2/6 km); default 6 km for robust coverage.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False,
                 resolution_km: int = DEFAULT_RESOLUTION_KM) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resolution_km = resolution_km

    def download(self, min_lon, max_lon, min_lat, max_lat,
                 start: str, end: str) -> Optional[Path]:
        backend = select_backend(end)  # raises if archive (Phase 3b) needed
        dataset_id = select_erddap_dataset(
            min_lon, max_lon, min_lat, max_lat, self.resolution_km
        )
        url = build_erddap_subset_url(
            dataset_id, min_lon, max_lon, min_lat, max_lat, start, end
        )

        if self.dry_run:
            print(f"[dry-run] NOAA HF-radar ({backend}) would download:\n  {url}")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        start_d = normalize_datetime(start).split("T")[0]
        end_d = normalize_datetime(end).split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}_{end_d}"
        out_path = self.output_dir / f"{dataset_id}_{self.resolution_km}km_{date_str}.nc"
        urllib.request.urlretrieve(url, str(out_path))
        return out_path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download NOAA HFRnet RTV currents (ERDDAP).")
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION_KM,
                   choices=[1, 2, 6])
    p.add_argument("--output-dir", default="data/hfr_noaa")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    dl = NOAAHFRadarDownloader(
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        resolution_km=args.resolution,
    )
    out = dl.download(
        args.min_lon, args.max_lon, args.min_lat, args.max_lat,
        args.start, args.end,
    )
    if out is not None:
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_downloaders.py::TestNOAAHFRadarDownload -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/noaa_hfradar_downloader.py tests/test_downloaders.py
git commit -m "feat: NOAAHFRadarDownloader.download + CLI (ERDDAP griddap fetch, dry-run)"
```

---

## Task 5: `from_hf_radar_grid` converter

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (add `from_hf_radar_grid` static method after `from_scatterometer_nc` ~line 794)
- Modify: `sar_validation/core/_cf_metadata.py` (`PRODUCT_REFERENCES` ~line 43)
- Test: `tests/test_datatree_converter.py`

**Interfaces:**
- Produces: `DataTreeConverter.from_hf_radar_grid(nc_path) -> Optional[xr.Dataset]` — a `point`-dim Dataset with `EWCT`/`NSCT`, retained ancillary fields (`hfr_gdop`, `hfr_n_radials`, `hfr_n_sites`), coords `lon`/`lat`/`time`, and attrs `data_type="hf_radar_grid"`, `platform_type="radar"`, `source="NOAA HFRnet RTV"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_datatree_converter.py`:

```python
def _make_hfr_grid_nc(tmp_path, n_time=2, n_lat=3, n_lon=4):
    """Write a minimal NOAA HFRnet-shaped gridded RTV NetCDF (time, lat, lon)."""
    rng = np.random.default_rng(7)
    times = pd.date_range("2024-05-01T00:00:00", periods=n_time, freq="1h").values
    lats = np.linspace(33.0, 38.0, n_lat)
    lons = np.linspace(-125.0, -119.0, n_lon)
    shape = (n_time, n_lat, n_lon)
    ds = xr.Dataset(
        {
            "water_u": (("time", "lat", "lon"), rng.uniform(-0.6, 0.6, shape),
                        {"standard_name": "surface_eastward_sea_water_velocity",
                         "units": "m s-1"}),
            "water_v": (("time", "lat", "lon"), rng.uniform(-0.6, 0.6, shape),
                        {"standard_name": "surface_northward_sea_water_velocity",
                         "units": "m s-1"}),
            "DOPx": (("time", "lat", "lon"), rng.uniform(0, 2, shape)),
            "DOPy": (("time", "lat", "lon"), rng.uniform(0, 2, shape)),
            "number_of_radials": (("time", "lat", "lon"),
                                  rng.integers(1, 8, shape).astype(float)),
            "number_of_sites": (("time", "lat", "lon"),
                                rng.integers(1, 4, shape).astype(float)),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"title": "NOAA HFRnet RTV", "institution": "UCSD/NOAA"},
    )
    path = tmp_path / "ucsdHfrW6_6km_2024-05-01.nc"
    ds.to_netcdf(path)
    return path


class TestFromHfRadarGrid:
    def test_renames_uv_to_ewct_nsct(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert ds is not None
        assert "EWCT" in ds and "NSCT" in ds
        assert "water_u" not in ds and "water_v" not in ds

    def test_point_dimension_flattened(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path, 2, 3, 4))
        assert "point" in ds.dims
        assert ds.sizes["point"] == 2 * 3 * 4
        for c in ("lon", "lat", "time"):
            assert c in ds.coords

    def test_data_type_and_platform_tags(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert ds.attrs["data_type"] == "hf_radar_grid"
        assert ds.attrs["platform_type"] == "radar"

    def test_retains_ancillary_uncertainty_fields(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert "hfr_gdop" in ds        # derived from DOPx/DOPy
        assert "hfr_n_radials" in ds
        assert "hfr_n_sites" in ds

    def test_returns_none_for_missing_file(self, tmp_path):
        assert DataTreeConverter.from_hf_radar_grid(tmp_path / "nope.nc") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datatree_converter.py::TestFromHfRadarGrid -v`
Expected: FAIL (`AttributeError: ... has no attribute 'from_hf_radar_grid'`)

- [ ] **Step 3: Add `PRODUCT_REFERENCES["hf_radar"]`**

In `sar_validation/core/_cf_metadata.py`, add to `PRODUCT_REFERENCES` (after the `radiometer` entry ~line 43):

```python
    "hf_radar": "https://hfradar.ioos.us/",
```

- [ ] **Step 4: Implement `from_hf_radar_grid`**

In `datatree_converter.py`, after `from_scatterometer_nc` (~line 794), add:

```python
    @staticmethod
    def from_hf_radar_grid(
        nc_path: Union[str, Path],
    ) -> Optional[xr.Dataset]:
        """
        Open a NOAA HFRnet gridded RTV NetCDF (dims ``time, lat, lon``; vars
        ``water_u``/``water_v``) and return a standardised point-frame Dataset
        tagged ``data_type="hf_radar_grid"``.

        The regular grid is flattened to a ``point`` dimension (one point per
        cell per time) so it collocates through the ``layer_vs_layer`` path,
        exactly like the scatterometer converter. ``water_u``/``water_v`` are
        renamed to canonical ``EWCT``/``NSCT``. Ancillary uncertainty/QC fields
        are *retained but not used* (design §3.7): ``DOPx``/``DOPy`` are
        combined into ``hfr_gdop`` (geometric dilution of precision) and the
        radial/site counts are kept as ``hfr_n_radials``/``hfr_n_sites`` so the
        deferred correction/QC phase can filter on them.

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

        # Resolve coordinate names (HFRnet uses lat/lon; tolerate latitude/longitude).
        lat_name = next((n for n in ("lat", "latitude", "LAT") if n in raw.coords or n in raw), None)
        lon_name = next((n for n in ("lon", "longitude", "LON") if n in raw.coords or n in raw), None)
        time_name = next((n for n in ("time", "Time", "TIME") if n in raw.coords or n in raw), None)
        if not (lat_name and lon_name and "water_u" in raw and "water_v" in raw):
            logger.warning(
                "from_hf_radar_grid: %s missing lat/lon or water_u/water_v (have %s)",
                nc_path.name, list(raw.coords) + list(raw.data_vars),
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

        ewct = _flat("water_u")
        nsct = _flat("water_v")

        data_vars: Dict[str, tuple] = {
            "EWCT": ("point", ewct),
            "NSCT": ("point", nsct),
        }
        var_attrs: Dict[str, Dict] = {
            "EWCT": dict(raw["water_u"].attrs),
            "NSCT": dict(raw["water_v"].attrs),
        }

        # --- Retained ancillary fields (not used in Phase 3a; design §3.7) ---
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
        ds.attrs["source"]        = "NOAA HFRnet RTV"
        ds.attrs["filename"]      = nc_path.name

        raw.close()
        return ds
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_datatree_converter.py::TestFromHfRadarGrid -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/datatree_converter.py sar_validation/core/_cf_metadata.py tests/test_datatree_converter.py
git commit -m "feat: from_hf_radar_grid converter with retained ancillary QC fields"
```

---

## Task 6: Discover the `hfr_noaa/` download folder in `convert_downloaded_data`

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`convert_downloaded_data` discovery block ~line 2100-2110)
- Test: `tests/test_datatree_converter.py`

**Interfaces:**
- Consumes: `DataTreeConverter.from_hf_radar_grid` (Task 5).
- Produces: a DataTree node keyed `validation/hfr_noaa/<stem>` for each `.nc` in `<base_dir>/hfr_noaa/`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_datatree_converter.py` (reuse `_make_hfr_grid_nc` from Task 5):

```python
class TestBuildDatatreeHfrNoaa:
    def test_hfr_noaa_folder_becomes_validation_node(self, tmp_path):
        base = tmp_path / "run"
        (base / "hfr_noaa").mkdir(parents=True)
        _make_hfr_grid_nc(base / "hfr_noaa")  # writes ucsdHfrW6_6km_2024-05-01.nc
        tree = DataTreeConverter.convert_downloaded_data(base, product_type="currents")
        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any("hfr_noaa" in p for p in node_paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datatree_converter.py::TestBuildDatatreeHfrNoaa -v`
Expected: FAIL (no `validation/hfr_noaa/` node — folder not discovered)

- [ ] **Step 3: Add the discovery block**

In `convert_downloaded_data`, after the scatterometer discovery loop (~line 1989, before the altimeter block), add:

```python
        # NOAA HFRnet gridded RTV currents (flattened to points, tagged
        # hf_radar_grid). Domain-filtered like the scatterometer path.
        subdir = base_dir / "hfr_noaa"
        if subdir.exists():
            for nc_path in sorted(subdir.glob("*.nc")):
                ds = _filtered(
                    DataTreeConverter.from_hf_radar_grid(nc_path),
                    nc_path.name,
                )
                if ds is not None:
                    datasets[f"validation/hfr_noaa/{nc_path.stem}"] = ds
                    logger.info("Converted hfr_noaa (HF-radar grid): %s", nc_path.name)
```

Also update the `Discovery rules` docstring list (~line 1913) by adding:

```python
        - ``hfr_noaa/*.nc``            → ``validation/hfr_noaa/<stem>`` nodes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datatree_converter.py::TestBuildDatatreeHfrNoaa -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "feat: discover hfr_noaa/ folder in build_datatree"
```

---

## Task 7: Orchestrator wiring — `hf_radar_noaa` source

**Files:**
- Modify: `sar_validation/core/orchestrator.py` (`_dispatch_source` handlers ~line 204-209; add `_download_noaa_hfradar` after `_download_hf_radar` ~line 275)
- Test: `tests/test_downloaders.py` (or a light orchestrator test if a suitable harness exists — otherwise assert the handler mapping directly)

**Interfaces:**
- Consumes: `NOAAHFRadarDownloader` (Task 4); `source.source_type == "hf_radar_noaa"`.
- Produces: `Orchestrator._download_noaa_hfradar(source) -> bool`, writing to `<base_dir>/hfr_noaa/`; registered in `_dispatch_source`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_downloaders.py`:

```python
def test_orchestrator_registers_hf_radar_noaa_handler():
    # The dispatch table must map the new source_type to a bound handler.
    from sar_validation.core.orchestrator import Orchestrator
    import inspect
    src = inspect.getsource(Orchestrator._dispatch_source)
    assert '"hf_radar_noaa"' in src
    assert "_download_noaa_hfradar" in src
    assert hasattr(Orchestrator, "_download_noaa_hfradar")
```

(If a full `Orchestrator` can be instantiated cheaply with a stub recipe in the existing tests, prefer a behavioural test that calls `_download_noaa_hfradar` with `dry_run=True` and asserts `metadata["downloads"]["hf_radar_noaa"]["status"] == "dry_run"`. Use the source-inspection test only if no such harness exists.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_downloaders.py::test_orchestrator_registers_hf_radar_noaa_handler -v`
Expected: FAIL (`hf_radar_noaa` not in dispatch; method missing)

- [ ] **Step 3: Register handler + implement download method**

In `_dispatch_source`, add to the `handlers` dict:

```python
            "hf_radar_noaa": self._download_noaa_hfradar,
```

After `_download_hf_radar` (~line 275), add:

```python
    def _download_noaa_hfradar(self, source) -> bool:
        from ..downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader,
            DEFAULT_RESOLUTION_KM,
        )

        cfg    = self.recipe.config
        bounds = cfg.geographic_bounds
        temp   = cfg.temporal_bounds
        out_dir = self.base_dir / "hfr_noaa"
        # Resolution is an optional per-source override, forwarded via the
        # established ValidationDataSource.download_kwargs channel.
        resolution_km = int(source.download_kwargs.get("resolution_km", DEFAULT_RESOLUTION_KM))

        try:
            dl = NOAAHFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                resolution_km=resolution_km,
            )
            dl.download(
                min_lon=bounds.min_lon, max_lon=bounds.max_lon,
                min_lat=bounds.min_lat, max_lat=bounds.max_lat,
                start=temp.start, end=temp.end,
            )
            self.metadata["downloads"]["hf_radar_noaa"] = {
                "status": "dry_run" if self.dry_run else "success",
            }
            return True
        except Exception as exc:
            msg = f"NOAA HF-radar download failed: {exc}"
            logger.error(msg)
            self.metadata["errors"].append(msg)
            self.metadata["downloads"]["hf_radar_noaa"] = {"status": "failed", "error": msg}
            return False
```

(`resolution_km` is read from `ValidationDataSource.download_kwargs` — the existing per-source override channel — so no dataclass change is needed. Absent the key, it defaults to 6 km.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_downloaders.py::test_orchestrator_registers_hf_radar_noaa_handler -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/orchestrator.py tests/test_downloaders.py
git commit -m "feat: orchestrator routes hf_radar_noaa to NOAA ERDDAP downloader"
```

---

## Task 8: CLI `currents` recipe template

**Files:**
- Modify: `sar_validation/cli.py` (`currents` template ~line 342-366)
- Test: `tests/test_recipe.py` (or `tests/test_downloaders.py` if CLI recipe creation is tested there)

**Interfaces:**
- Consumes: nothing new; emits a recipe whose validation sources include `hf_radar_noaa` and whose `layer_vs_layer.layer_type_specs` includes `hf_radar_grid`.

- [ ] **Step 1: Write the failing test**

Find how `_create_recipe` is exercised in the tests (grep `test_recipe.py` for `create_recipe`/`currents`). Add a test that builds the `currents` template and asserts the new wiring. If `_create_recipe` writes a YAML file, load it; if it returns a `RecipeConfig`, assert on the object. Template (adjust to the actual return/IO shape):

```python
def test_currents_template_includes_noaa_hfradar(tmp_path, monkeypatch):
    from sar_validation import cli
    # Drive _create_recipe for the 'currents' template into tmp_path, then
    # load the written recipe and assert the new source + layer spec.
    recipe = cli._build_currents_config(limit=None)  # use whatever builder exists
    source_types = {s.source_type for s in recipe.validation_sources}
    assert "hf_radar_noaa" in source_types
    specs = recipe.collocation.layer_vs_layer.layer_type_specs
    assert "hf_radar_grid" in specs
    assert specs["hf_radar_grid"]["aggregation_window_km"] == 6.0
    assert specs["hf_radar_grid"]["time_tolerance_minutes"] == 20
```

If there is no separate builder function, refactor the `currents` `RecipeConfig(...)` literal into a small helper `_build_currents_config(limit)` in `cli.py` and call it from the `templates` dict — this makes the template unit-testable and is a reasonable, in-scope improvement.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recipe.py -k currents -v`
Expected: FAIL (`hf_radar_noaa` / `hf_radar_grid` absent)

- [ ] **Step 3: Add the source + layer spec to the template**

In `sar_validation/cli.py`, in the `currents` `RecipeConfig`, add the source to `validation_sources` (after the existing `hf_radar` source ~line 350):

```python
                ValidationDataSource(
                    source_type="hf_radar_noaa",
                    min_depth=-2.0, max_depth=2.0,
                    download_kwargs={"resolution_km": 6},
                ),
```

and add the layer spec to `layer_type_specs` (alongside the existing `hf_radar` entry ~line 359-364):

```python
                        "hf_radar_grid": {
                            "time_tolerance_minutes": 20,
                            "aggregation_window_km": 6.0,
                            "distance_weighting": "equal",
                        },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_recipe.py -k currents -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sar_validation/cli.py tests/test_recipe.py
git commit -m "feat: add hf_radar_noaa source and hf_radar_grid spec to currents recipe template"
```

---

## Task 9: End-to-end validation on a real US-West subset

This task is a **manual/integration validation** (needs network), not a unit test. It confirms the spec's success criteria (design §4).

**Files:** none (uses the installed CLI + real data).

- [ ] **Step 1: Confirm the ERDDAP subset URL builds and is reachable (dry-run)**

```bash
python -m sar_validation.downloaders.noaa_hfradar_downloader \
    --min-lon -125 --max-lon -119 --min-lat 33 --max-lat 38 \
    --start 2024-05-01 --end 2024-05-02 --resolution 6 --dry-run
```
Expected: prints a `https://coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW6.nc?water_u[...]...` URL. Paste it into a browser or `curl -I` it and confirm HTTP 200.

- [ ] **Step 2: Download a small real subset**

```bash
python -m sar_validation.downloaders.noaa_hfradar_downloader \
    --min-lon -125 --max-lon -119 --min-lat 33 --max-lat 38 \
    --start 2024-05-01 --end 2024-05-02 --resolution 6 \
    --output-dir /tmp/hfr_run/hfr_noaa
```
Expected: writes `ucsdHfrW6_6km_2024-05-01_2024-05-02.nc`. Open it and confirm `water_u`/`water_v` with dims `(time, lat, lon)` and non-empty ocean cells.

- [ ] **Step 3: Convert and inspect the datatree node**

Place (or symlink) a real Sentinel-1 IW/WV **currents** SAFE with overlapping coverage under `/tmp/hfr_run/S1_L2_OCN/`, then:

```bash
python -c "from sar_validation.core.datatree_converter import DataTreeConverter as D; \
t=D.convert_downloaded_data('/tmp/hfr_run', product_type='currents'); \
print([n.path for n in t.subtree if 'hfr_noaa' in n.path]); \
print(t['validation/hfr_noaa'])"
```
Expected: a `validation/hfr_noaa/...` node with `EWCT`/`NSCT` + `hfr_gdop`/`hfr_n_radials`/`hfr_n_sites`, `data_type="hf_radar_grid"`.

- [ ] **Step 4: Run the full pipeline and check the currents comparison**

Create a `currents` recipe over the US-West bbox/date and run collocation → statistics → plots (use the project's standard run entrypoint, e.g. `sar-validate --recipe <file>` or the run skill). Expected observations:
- A **non-empty** set of collocated pairs over the HF-radar/SAR overlap.
- `rvlRadVel_projection` present in the collocation output for `hfr_noaa` pairs.
- A scatter plot and geographic map are produced for the currents comparison.
- **Document** (in the run notes/PR description) that raw `rvlRadVel` vs `rvlRadVel_projection` agreement is expected to be weak (r < 0.5 per Martin et al. 2022) because the WASV/instrument correction chain is deferred — this is the expected Phase 3a outcome, not a bug.

- [ ] **Step 5: Run the whole unit suite once more**

Run: `pytest -q`
Expected: all green.

---

## Self-Review

**Spec coverage (design §3):**
- §3.1 NOAA downloader (ERDDAP backend, region/resolution map, dry-run URL) → Tasks 3–4. (THREDDS backend = §3.2/3b, out of scope.)
- §3.3 `from_hf_radar_grid` + ancillary retention (§3.7) → Task 5; folder discovery → Task 6.
- §3.4 shared projection helper across all paths + `_collocate_individual` gap → Task 1; dispatch → Task 2.
- §3.5 recipe/CLI/orchestrator + `hf_radar_grid` layer spec → Tasks 2, 7, 8.
- §3.6 variable-map/CF reuse → no change (EWCT/NSCT already registered); added only `PRODUCT_REFERENCES["hf_radar"]` in Task 5.
- §4 tests + integration success criteria → per-task unit tests + Task 9.
- §3.7 ancillary fields retained (`hfr_gdop`, `hfr_n_radials`, `hfr_n_sites`) → Task 5. `EWCS`/`NSCS`/`GDOP`/`*_QC`/`CSPD`/`CDIR` are Copernicus-side fields → handled when the Copernicus converter path lands in Phase 3c; noted here, not implemented, matching the NOAA-only scope of 3a.

**Deferred (not in this plan, by design):** THREDDS archive backend (3b); Copernicus NRT/MY sources and their point routing (3c); the WASV/instrument correction chain (later phase); non-CONUS ERDDAP dataset ids (design §6 — `select_erddap_dataset` raises a clear error for them).

**Placeholder scan:** no TBD/TODO; every code step shows complete code; `select_backend`/`select_erddap_dataset` raise defined, tested errors rather than stubbing.

**Type consistency:** `_project_currents_to_radial(float, float, float) -> float` used identically in Tasks 1; `from_hf_radar_grid` node key `validation/hfr_noaa/<stem>` matches the `hfr_noaa` path fragment added to `LAYER_SOURCE_PATHS` and the resolution blocks; `data_type="hf_radar_grid"` matches the `DEFAULT_LAYER_TYPE_SPECS` key and dispatch branch; `NOAAHFRadarDownloader.download(...)` signature matches its orchestrator call.

**Open verification item for the implementer:** Task 8 assumes a `_build_currents_config(limit)` builder extracted from the inline `currents` `RecipeConfig` literal — confirm/create that refactor so the template is unit-testable, or instead drive `_create_recipe("currents", …)`, load the written recipe file, and assert on it. (Resolved during planning: resolution flows through `ValidationDataSource.download_kwargs`; DataTree nodes are iterated via `.subtree`/`.path`.)
