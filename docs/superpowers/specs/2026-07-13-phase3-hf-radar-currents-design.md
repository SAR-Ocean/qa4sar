# Phase 3 — HF-Radar Surface-Current Validation

**Status:** Approved design; ready for implementation planning
**Date:** 2026-07-13
**Supersedes:** the "Phase 3: Coastal HF-Radar Current Vectors" outline in
[`docs/future-phases.md`](../../future-phases.md), which assumed MET-Norway
radial reconstruction via `makeTotalVector`. That reconstruction is **not
needed**: the sources chosen here deliver *total* (u, v) vectors already.

---

## 1. Goal & scope

Add HF-radar surface-current validation to the toolbox across three data
sources. All three are compared against SAR L2 OCN radial velocity
(`rvlRadVel`) by projecting the validation current vector (eastward `EWCT`,
northward `NSCT`) onto the SAR line-of-sight using the scene's `rvlHeading`,
producing the comparable quantity `rvlRadVel_projection` (this projection and
variable pair already exist in the codebase).

| Source | Access | Delivered as | Coverage | Availability | Status |
|--------|--------|--------------|----------|--------------|--------|
| **NOAA IOOS RTV** | ERDDAP griddap + THREDDS/OPeNDAP | **gridded** `water_u`/`water_v` (CF), hourly, 1/2/6 km | US coasts (W, E+Gulf, HI, AK, PR/USVI, Great Lakes) | ERDDAP: rolling ~3-month window; THREDDS: 2012–present | **new (primary)** |
| **Copernicus in-situ NRT** (`INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030`) | `copernicusmarine` | **point/station** EWCT/NSCT | Global (incl. Europe) | NRT | exists — verify & finish |
| **Copernicus in-situ delayed-mode** (`INSITU_GLO_PHY_UV_DISCRETE_MY_013_044`) | `copernicusmarine` | point u/v (HF radar + drifter/ADCP/etc.) | Global | Historical only, 1979 → Jan 2026 | **new** |

Out of scope for this phase (explicitly deferred):
- Current **direction** comparison (HCDT) with circular statistics — the
  comparison quantity remains the scalar radial projection.
- A full "dry collocation" (granule search without download). The NOAA ERDDAP
  backend is built so its subset URLs can later seed that feature, but it is not
  implemented here.

---

## 2. Chosen architecture (Approach A)

**Two data-types sharing one projection.** Gridded and point HF-radar are
different collocation geometries, so they get different routing, but they share
the canonical current variables (`EWCT`/`NSCT` → `rvlRadVel_projection`) and the
same projection math.

- Gridded HF-radar → `data_type="hf_radar_grid"` → `layer_vs_layer` collocation.
- Point/station HF-radar → `data_type="hf_radar"` → point/in-situ collocation.

**The gridded-vs-point tag is assigned by the converter from the delivered data
structure, not from the provider.** A node with a regular lat/lon grid is tagged
`hf_radar_grid`; scattered station rows are tagged `hf_radar`. If a Copernicus
current product is ever delivered as a grid, it is tagged `hf_radar_grid` and
routes as a layer automatically — the data decides the geometry, not the source.

Rejected alternatives:
- **B — one `hf_radar` type, branch internally on gridded-vs-point.** Overloads
  a single type with two geometries; brittle dispatch, can't tune per family.
- **C — flatten the NOAA grid to points and reuse the point path.** A full RTV
  grid is ~10^5–10^6 cells; misrepresents a dense regular grid as scattered
  points and only works with tight server-side subsetting.

---

## 3. Components

### 3.1 NOAA downloader — `sar_validation/downloaders/noaa_hfradar_downloader.py` (new)

Two backends, auto-selected by the requested end date (with fallback between
them, mirroring the existing `HFRadarDownloader` dataset-part retry pattern):

- **ERDDAP griddap** (recent, ~last 3 months). Build a NetCDF subset URL:
  ```
  https://coastwatch.pfeg.noaa.gov/erddap/griddap/<dataset>.nc?
      water_u[(t0):(t1)][(lat0):(lat1)][(lon0):(lon1)],
      water_v[(t0):(t1)][(lat0):(lat1)][(lon0):(lon1)]
  ```
  No credentials; bbox + time subsetting happen server-side. `--dry-run` prints
  the URL (this URL is the intended seed for a future granule-search / "dry
  collocation" feature).
- **THREDDS/OPeNDAP** (archive, 2012–present). Open the UCSD `hfrnet-tds`
  aggregation through xarray's OPeNDAP engine, `.sel()` the bbox + time window,
  and write NetCDF.

**Region + resolution → dataset map.** Choose the regional dataset from the
request bounding box:

| Region | ERDDAP dataset ids (by resolution) |
|--------|------------------------------------|
| US West | `ucsdHfrW1` (1 km), `ucsdHfrW2` (2 km), `ucsdHfrW6` (6 km) |
| US East + Gulf | `ucsdHfrE1` (1 km), `ucsdHfrE6` (6 km) |
| Hawaii, Alaska, PR/USVI, Great Lakes | region-specific ids (resolve from ERDDAP catalog at build time) |

Resolution is configurable via `download_kwargs`; default **6 km** for robust
coverage, overridable to 2/1 km. Variables on the wire are `water_u`/`water_v`
with CF standard names `surface_eastward_sea_water_velocity` /
`surface_northward_sea_water_velocity`.

### 3.2 Copernicus historical downloader — `013_044`

Add support for `INSITU_GLO_PHY_UV_DISCRETE_MY_013_044` (delayed-mode). Reuse
the `copernicusmarine.subset` pattern of the existing `HFRadarDownloader`;
filter to the HF-radar platform for current-vs-SAR consistency (other platform
types in this product — drifter/ADCP/glider — are out of scope here). Source
selection between NRT `013_030` and MY `013_044` is by requested date: recent →
NRT, historical (pre-~2026 / outside the NRT window) → MY, with fallback.

### 3.3 Converter — `sar_validation/core/datatree_converter.py`

- **`from_hf_radar_grid(ds)` (new).** Read gridded NetCDF (dims `time, lat,
  lon`; vars `water_u`, `water_v`), rename to canonical `EWCT` / `NSCT`, attach
  CF metadata, and tag the node `data_type="hf_radar_grid"`,
  `platform_type="radar"`, plus `sensor`/`source` provenance. Emit it in the
  same **layer-node shape** the scatterometer converter produces, so the
  collocation layer path flattens it to a `lon/lat/time/EWCT/NSCT` frame.
- **Copernicus point path.** Reuse the existing in-situ CSV conversion for both
  `013_030` and `013_044`. Assign `data_type` by inspecting delivered structure
  (regular grid → `hf_radar_grid`; station rows → `hf_radar`).

### 3.4 Collocation — `sar_validation/core/collocation.py` (core correctness item)

- **Extract the projection** currently inlined at
  [`collocation.py:591`](../../../sar_validation/core/collocation.py#L591) into a
  shared helper `_project_currents_to_radial(ewct, nsct, heading_deg)` and call
  it from **both** the point-vs-layer aggregation **and** the layer-vs-layer
  aggregation (`_collocate_cell_averaging`). Today the projection runs only in
  the point path, so gridded currents would never be projected — this is the
  key fix that makes NOAA gridded currents comparable at all.
- **Dispatch.** Route `data_type="hf_radar_grid"` to `layer_vs_layer` and
  `hf_radar` (point) to the point/in-situ path. Extend the existing layer-type
  detection blocks (`data_type` attr, with path-part fallback) to recognise
  `hf_radar_grid`.

### 3.5 Recipe / CLI / orchestrator

- **Recipe `source_type` values:** keep `hf_radar` (Copernicus NRT `013_030`);
  add `hf_radar_noaa` (gridded RTV) and `hf_radar_historical` (Copernicus
  `013_044`).
- **Orchestrator:** add `_download_noaa_hfradar` and a historical-Copernicus
  branch alongside the existing `_download_hf_radar`.
- **CLI:** `--create-recipe currents` template lists all three sources with
  sensible defaults and comments.
- **`DEFAULT_LAYER_TYPE_SPECS`:** add an `hf_radar_grid` entry —
  `aggregation_window_km` ≈ chosen grid resolution (default 6 km),
  `time_tolerance_minutes` ≈ 20 (matching the existing fast-decorrelation choice
  for the point `hf_radar` spec), `distance_weighting="equal"`.

### 3.6 Variable mapping / CF — no change required

The currents pair `("rvlRadVel", "rvlRadVel_projection")` in `_variable_map.py`
and the `EWCT`/`NSCT` CF metadata in `_cf_metadata.py` already exist and are
reused as-is.

---

## 4. Tests & validation

**Unit tests**
- ERDDAP subset-URL construction (variables, bbox, time selectors).
- Backend selection by requested end date (ERDDAP vs THREDDS) and fallback.
- Region + resolution → dataset id mapping.
- `from_hf_radar_grid`: `water_u`/`water_v` → `EWCT`/`NSCT` rename, correct
  `data_type="hf_radar_grid"` tag, layer-node shape.
- `_project_currents_to_radial`: projection math (e.g. a due-east current with a
  known heading yields the expected radial component; NaN handling).
- Dispatch routing: gridded → `layer_vs_layer`, point → point path.

**Integration / success criteria**
- Download a small US-West RTV subset (ERDDAP, 6 km) for a date with a real
  Sentinel-1 currents scene over the overlap region.
- Run the full pipeline (convert → collocate → statistics → plots).
- Confirm a non-empty set of collocated pairs over the known overlap, and that
  projected radial magnitudes are physically plausible.
- Produce a scatter plot and geographic map for the currents comparison.

---

## 5. Execution order (one spec, phased)

- **3a — NOAA ERDDAP end-to-end.** Shared-projection refactor
  (`_project_currents_to_radial`) + NOAA ERDDAP downloader +
  `from_hf_radar_grid` converter + dispatch routing + recipe/CLI wiring +
  `hf_radar_grid` layer spec. Validated on a US-West region. *Delivers a working
  US current-validation result first.*
- **3b — NOAA THREDDS/OPeNDAP archive backend.** Adds historical depth
  (2012–present) behind the same downloader interface and date-based selection.
- **3c — Copernicus.** Verify/finish the existing NRT `013_030` path end-to-end
  for currents, then add the delayed-mode `013_044` historical source and its
  date-based selection.

Rationale: 3a is self-contained and matches "first tests on US coastal areas";
3b and 3c extend coverage (archive depth, European/global + historical) without
disturbing the 3a pipeline.

---

## 6. Open items to resolve during implementation

- **Copernicus geometry.** Confirm whether `013_030`/`013_044` HF-radar are ever
  delivered gridded rather than as station points; the converter's
  structure-based tagging already handles either outcome, but the tests should
  cover whichever is actually returned.
- **Exact non-West ERDDAP dataset ids** (Hawaii/Alaska/PR-USVI/Great Lakes) to
  be resolved from the live ERDDAP catalog when 3a/3b are built.
- **ERDDAP rolling-window boundary.** Confirm the exact cutover date where
  ERDDAP stops and THREDDS must take over, and make the boundary a single tunable
  constant rather than a hard-coded date.
