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
- **Retain ancillary uncertainty/QC fields (do not use them yet).** Both
  converters carry an explicit allow-list of ancillary variables alongside
  `EWCT`/`NSCT`, at native resolution, so the deferred correction/QC phase (see
  §3.7) is a converter no-op to enable. Phase 3 does not filter or correct on
  them — it just refuses to drop them.

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

### 3.7 HF-radar ancillary parameters to retain (for a future correction/QC phase)

Martin, Gommenginger, Jacob & Staneva (2022), *"First multi-year assessment of
Sentinel-1 radial velocity products using HF radar currents in a coastal
environment"*, Remote Sensing of Environment 268:112758
([doi:10.1016/j.rse.2021.112758](https://doi.org/10.1016/j.rse.2021.112758),
open-access PDF at nora.nerc.ac.uk/id/eprint/532190) is the methodological
reference for this comparison. It confirms the toolbox's core approach —
projecting the HF-radar Cartesian `(EWCT, NSCT)` field onto the S1 line-of-sight
(the range direction, ⊥ to `rvlHeading`) and comparing the scalar radial
component — and its collocation choices (±20 min temporal window; box/median
averaging over ~21×27 km improves std to ~0.24 m/s and r up to 0.93).

**Important caveat for interpreting Phase 3 results:** the paper shows that raw
operational L2 OCN `rvlRadVel` is a radial *velocity* (current + Stokes drift +
wind-wave artefact bias, the latter up to ~2 m/s), **not** a radial *current*,
and agrees poorly with HF-radar currents (r < 0.5, std > 0.65 m/s) until a
correction chain is applied — outlier flagging (3σ-MAD in range then azimuth),
de-scalloping, antenna-mispointing and land-bias corrections, and above all the
**WASV** (Wind-wave Artefact Surface Velocity) correction via the Y19C-Dop
(Yurovsky 2019) model driven by ERA5 sea state. That correction chain is
**explicitly deferred** to a later phase; Phase 3 produces the raw comparison and
should document the expected weak agreement rather than treat it as a failure.

To make that later phase cheap to add, the Phase 3 converters (§3.3) **retain but
do not use** the following ancillary fields, renamed to a canonical scheme and
kept at native resolution:

| Source | Wire variable(s) | Retain as | Enables later |
|--------|------------------|-----------|---------------|
| NOAA RTV (grid) | `water_u`, `water_v` | `EWCT`, `NSCT` | projection (already core) |
| NOAA RTV (grid) | `DOPx`, `DOPy` | `hfr_gdop` (= √(DOPx²+DOPy²)), plus components | LOS-accuracy QC filter; project error onto LOS |
| NOAA RTV (grid) | `number_of_radials`, `number_of_sites` | `hfr_n_radials`, `hfr_n_sites` | coverage/quality gating |
| Copernicus HFR total | `EWCS`, `NSCS` | per-cell std error of u/v | LOS-accuracy QC filter; uncertainty projection |
| Copernicus HFR total | `GDOP` | `hfr_gdop` | geometry-based QC |
| Copernicus HFR total | `*_QC` / `QCflag` | `hfr_qc` | keep only good cells (paper's QC step) |
| Copernicus HFR total | `CSPD`, `CDIR` (if present) | speed/direction | deferred HCDT **direction** comparison |

The QC/uncertainty threshold the paper uses (retain HF-radar cells with LOS
accuracy < 0.09 m/s; median accuracy 0.04 m/s) is the intended first consumer of
these fields.

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
