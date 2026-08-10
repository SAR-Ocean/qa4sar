# Toolbox Overview and Design Choices

This document describes the SAR L2 validation toolbox at a high level and —
more importantly — explains the *choices* baked into the code: unit and
convention conversions, quality filtering, collocation geometry, aggregation
windows, tolerances, and the statistical treatment of circular variables.
Wherever a choice comes from the literature or a product specification, the
source is named. Each section ends with a pointer to the code that implements
it, so every statement here can be verified.

For the *mechanics* of each step see the companion docs:
[creating-recipes.md](creating-recipes.md), [collocation.md](collocation.md),
[cli-statistics-and-plots.md](cli-statistics-and-plots.md).

---

## 1. What the toolbox is

A standalone Python package that validates L2 SAR data, such as Sentinel-1 **L2_OCN** (Level-2
Ocean) products against independent observations. One YAML **recipe**
(variable + region + time window + validation sources) drives the entire
pipeline:

| Step | What happens | Output |
|---|---|---|
| 1 — Download | SAR scenes + all validation sources for the recipe domain | `data/<run>/...` |
| 2 — Convert | Every source is standardised into one hierarchical `xarray.DataTree` | `datatree.nc` |
| 3 — Collocate | SAR cells are matched to validation observations within spatial/temporal tolerances | `collocation_results.nc` |
| 4 — Statistics | Bias, RMSE, correlation, scatter index per platform type | `validation_statistics_*.nc/.csv` |
| 5 — Report | Scatter/geographic/residual plots + PDF report | `plots/`, `validation_report.pdf` |

Four validated quantities are supported — **wind** (speed + direction,
OWI grids), **currents** (RVL radial velocity), **waves** (significant wave
height, OSW/WV vignettes), **soil moisture** (Sentinel-1 CLMS SSM 1 km
Europe rasters) — against in-situ platforms (moorings, buoys, drifters,
ferryboxes, tidal gauges, ISMN soil moisture stations), scatterometers, altimeters,
radiometers, and HF radars.

> Code: `sar_validation/cli.py` (pipeline driver), `core/recipe.py`
> (recipe schema), `core/orchestrator.py` (step 1).

---

## 2. Canonical variable naming

**Choice: every source is renamed at conversion time to the Copernicus Marine
in-situ parameter codes.** The validation statistics and report are organised
per *(SAR variable, validation variable)* pair; if two sources used different
names for the same physical quantity they would end up in different report
sections, each showing the other source as "no data". Renaming once, at
ingestion, means no downstream component ever needs alias logic.

| Source | Raw name | Canonical code | Quantity |
|---|---|---|---|
| OSI-SAF scatterometer | `wind_speed` | `WSPD` | 10-m wind speed |
| OSI-SAF scatterometer | `wind_dir` | `WDIR` | 10-m wind direction |
| CMEMS altimeter L3 (1 Hz) | `WIND_SPEED` | `WSPD` | altimeter-derived wind speed |
| RSS radiometer (AMSR2) | `wind_speed_LF` | `WSPD` | 10-m radiometer wind speed |
| Copernicus in-situ CSV | already `WSPD`, `WDIR`, `VHM0`, `EWCT`, `NSCT`, … | unchanged | — |

The mapping from recipe variable ("wind" / "currents" / "waves") to the
compared pairs — e.g. `owiWindSpeed` vs `WSPD` — is the single source of
truth for both statistics and plots.

**Exception: ERA5 wind.** `from_era5` deliberately does NOT rename/derive
`u10`/`v10` to `WSPD`/`WDIR` at conversion time — they stay as raw vector
components through conversion and collocation-time spatial/temporal
interpolation, and `WSPD`/`WDIR` are derived only afterwards, in
`model_collocation.py`. This is the one source where the "renamed at
conversion time" rule above doesn't hold. See §5.7 for why.

> Code: `core/_variable_map.py` (`VARIABLE_PAIRS`),
> `core/datatree_converter.py` (`from_scatterometer_nc`, `from_altimeter`
> rename maps).

---

## 3. Data-ingestion choices (step 2)

### 3.1 Scatterometer wind direction is rotated 180°

ASCAT/OSI-SAF reports wind direction in the **oceanographic** convention
(the direction the wind blows *towards*; CF `wind_to_direction`). Sentinel-1
OWI `owiWindDirection` and the in-situ `WDIR` code both use the
**meteorological** convention (the direction the wind blows *from*).

**Choice: rotate the scatterometer direction by 180° once, at conversion**,
so collocation, statistics, and every plot see a single consistent
convention. Without this, scatterometer-vs-SAR direction comparisons would
show a spurious ~180° bias. The stored variable's CF metadata is corrected
accordingly (`standard_name: wind_from_direction`) and a `comment` attribute
documents the rotation.

### 3.2 ASCAT quality-flag rejection

Wind-vector cells whose `wvc_quality_flag` carries any of these bits are
dropped at conversion:

- `some_portion_of_wvc_is_over_land`
- `some_portion_of_wvc_is_over_ice`
- `wind_inversion_not_successful`
- `not_enough_good_sigma0_for_wind_retrieval`
- `distance_to_gmf_too_large`

**Why:** these flags mark the retrieval itself as invalid or contaminated.
If kept, such cells appear as collocated "observations" with bogus or missing
values — often sitting over land — and pollute the statistics. The flag bits
are read from the file's own `flag_masks`/`flag_meanings` attributes rather
than hardcoded bit positions, so the filter survives product format updates.

### 3.3 Altimeter 1 Hz and 5 Hz are separate layer types

CMEMS along-track altimetry comes at 1 Hz (~7 km along-track spacing, carries
`WIND_SPEED`) and 5 Hz (~1.4 km spacing, waves only). **Choice: treat them as
distinct layer types** (`altimeter_1hz` / `altimeter_5hz`) because the
collocation aggregation window must match the sensor's footprint spacing —
a single window would be wrong for at least one of them (§5.2). The frequency
is detected from the file content (5 Hz files do not carry `WIND_SPEED`). 
Wind recipes download only 1 Hz (as 5 Hz has no wind); wave recipes download both.

### 3.4 Radiometer products are pre-gridded to 0.25° bins

Remote Sensing Systems (RSS) distributes each microwave radiometer mission as
a **daily gridded product already resampled to a common 0.25° (~25 km) global
grid** — the toolbox never touches the native swath. One file per sensor per
day covers the whole globe, with **two passes** (ascending / descending) and a
**per-cell measurement time**; `from_radiometer_nc` flattens every
`(pass, lat, lon)` cell to a point that carries its own timestamp, so each cell
collocates against the SAR scene nearest in time rather than sharing one file
time.

Design choices:

- **Wind speed is taken from the Low-Frequency (LF, 10.7 GHz) channel**
  (`wind_speed_LF` → `WSPD`). LF is RSS's standard all-purpose 10-m wind and is
  set to fill (NaN) under rain, so keeping LF also **drops rain-contaminated
  cells for free**; the Medium-Frequency and All-Weather winds are only
  fallbacks if a product lacks LF. Cells with a NaN wind (land, ice, rain) are
  dropped at ingestion, which is why the global grid shrinks to ocean points.
- **Two formats, one pipeline.** On `data.remss.com` only AMSR2 publishes
  **NetCDF**; GMI, SSMIS (F16/F17/F18) and WindSat are distributed as RSS
  **binary bytemaps** (gzipped `uint8`), decoded by a compact in-house reader
  (`downloaders/_rss_bytemap.py`) rather than RSS's Python-2 routines. Both
  formats funnel through the shared `_finalize_radiometer_points` tail so every
  sensor produces an identical node. The downloader's `SENSORS` table carries a
  `format` field and per-sensor URL templates; download is over **public
  HTTPS**, no account required. The bytemap format (confirmed empirically): a
  `(pass=2, var, lat=720, lon=1440)` grid, `physical = byte·scale + offset`,
  and **byte ≥ 251 = special/missing → NaN** (land/ice/coast/rain/no-obs). The
  per-cell time-of-day is fractional **hours** for most sensors but **minutes**
  for WindSat, so the reader tracks the unit per sensor.
- **WindSat adds wind direction.** WindSat is the one radiometer with a
  direction retrieval; its `wdir` is in the **oceanographic** convention, so it
  is rotated 180° to meteorological at ingestion — the same rotation applied to
  the ASCAT scatterometer (§3.1) — so WindSat `WDIR` compares directly against
  `owiWindDirection` and the in-situ `WDIR` code.
- **Per-sensor collocation specs.** Because every RSS product shares the 0.25°
  grid, the aggregation window is the same for all sensors, but each sensor
  gets its own `radiometer_<sensor>` spec key (e.g. `radiometer_amsr2`),
  mirroring the altimeter split, so it stays individually tunable (§5.2). Each
  node is tagged with its `sensor` attribute and collocation refines the bare
  `radiometer` layer type to `radiometer_<sensor>`.

### 3.5 Other ingestion conventions

- **Longitude** is normalised to −180…180° everywhere (source products mix
  0–360 and ±180 conventions).
- **In-situ CSVs** arrive in long format (one row per variable reading) and
  are pivoted to wide format. The platform type (`mooring`/`buoy`/`drifter`)
  is kept **per point**, not per file — a single Copernicus in-situ extract
  mixes platform types, and statistics are grouped by platform type (§7).
- **Depth window** for in-situ sources defaults to −20…+20 m; when several
  in-situ sources are requested, the most permissive window across them is
  used for the (single, batched) download.

> Code: `core/datatree_converter.py` (`_ASCAT_REJECT_FLAGS`,
> `from_scatterometer_nc`, `from_altimeter`, `from_radiometer_nc`,
> `from_insitu_csv`), `downloaders/radiometer_downloader.py` (`SENSORS`),
> `core/orchestrator.py` (`_ALTIMETER_FREQUENCIES_BY_VARIABLE`,
> depth-window merge in `download_all`).

### 3.6 HF-radar: NOAA primary for US regions, Copernicus fallback, QCflag filtering

Copernicus Marine's US-region HF-radar-total product (`US-WestCoast`,
`US-EastGulfCoast`) is sourced from the same U.S. IOOS/HFRNet national
network NOAA distributes directly via ERDDAP — but a live comparison for an
identical bbox/date/resolution found Copernicus's re-ingestion has ~5.8x-17x
fewer valid grid cells than NOAA's own distribution (173,116 vs 26,227 raw,
or vs 10,001 restricted to Copernicus cells flagged "good"). NOAA has no
per-cell quality flag at all — it filters upstream before publishing, so
whatever it serves has already passed QC; Copernicus ships an explicit
`QCflag` (1=good, 4=bad) but includes flagged-bad cells anyway.

**Why:** US recipes therefore use a `hf_radar_us` source that tries three
backends in order, first to return at least one file wins:

1. **ERDDAP griddap** (`noaa_hfradar_downloader.py`) — NOAA's rolling
   ~90-day real-time window, the densest and freshest coverage.
2. **THREDDS archive** (`noaa_hfradar_thredds_downloader.py`) — NOAA's
   NCEI-hosted archive, 2006-present but published a few weeks behind
   real-time, so it picks up the gap ERDDAP's rolling window leaves behind:
   older, historical dates. It cannot cover very recent/near-real-time
   dates -- those aren't published to THREDDS yet and remain ERDDAP's
   domain. THREDDS serves whole-region grids with no server-side bbox
   subsetting (unlike ERDDAP), so the downloader trims the merged output to
   the requested bbox client-side after download.
3. **Copernicus Marine** (`hf_radar_downloader.py` /
   `hf_radar_historical_downloader.py`, historical-first to avoid
   double-counting the same stations twice) — only ever reached when
   neither NOAA backend produced anything for the exact window: an even
   older date than THREDDS covers, or a non-US region.

`HFRadarUSDownloader.download()` (`hf_radar_us_downloader.py`) implements
this waterfall directly — there is no separate date-threshold picker
function; each backend's own `download()` is simply tried in turn inside
one method, and the first non-empty result short-circuits the rest. This
supersedes an earlier, same-day decision to retire NOAA everywhere in favor
of Copernicus alone (the network-identity argument was correct, but ignored
Copernicus's incomplete re-ingestion of it), and a later, separate decision
that stopped at ERDDAP-only NOAA coverage (superseded once the THREDDS
archive backend was added) — see `docs/superpowers/specs/` for that
reasoning's full history. Separately, and independent of which backend is
used, all Copernicus HF-radar data (US or not) now drops cells where the
overall `QCflag == 4` ("bad") — Copernicus ships them unfiltered, and this
toolbox previously retained them uncritically; per-parameter flags
(`CSPD_QC` etc.) remain retained but unused.

> Code: `downloaders/hf_radar_us_downloader.py` (`HFRadarUSDownloader.download()`),
> `downloaders/noaa_hfradar_thredds_downloader.py`
> (`NOAATHREDDSHFRadarDownloader`), `core/orchestrator.py`
> (`_download_hf_radar_us`), `core/datatree_converter.py`
> (`from_hf_radar_grid`'s `QCflag` filter).

---

## 4. datatree.nc content choices

### 4.1 Only in-domain points are stored

Full-orbit scatterometer files are global; for a typical regional recipe
fewer than 5 % of their points can ever collocate. **Choice: at conversion,
every validation node is filtered to the recipe's bounding box and time
window, expanded by the largest spatial and temporal collocation tolerance
that any source in the recipe could use.** Because the filter envelope is the
*maximum* tolerance (over the recipe's point-vs-layer settings, all
layer-type specs, per-source overrides, and the WV footprint radius), it can
never discard a point that some collocation pass would have matched. Points
with an unknown timestamp are kept — they cannot be proven out-of-window.

Measured effect: a one-evening North-East-Atlantic wind run shrank from
64 MB to 6.1 MB with bit-identical collocation results.

**SAR grids are never cropped**: scenes were already selected by bounding
box, and cropping a grid could remove cells that an in-bounds observation
near the box edge legitimately aggregates over.

### 4.2 Compression

All numeric variables are written with zlib (level 4). SAR OWI grids are
float64 with large NaN land masks and deflate extremely well; this is pure
size win at negligible read cost.

### 4.3 CF metadata

The raw products (OSI-SAF, CMEMS, Sentinel-1 SAFE) are already CF-annotated,
so **variable attributes are copied from the source files** rather than
maintained by hand — with two corrections:

- packing/range attributes (`scale_factor`, `valid_min/max`, `_FillValue`)
  are dropped, because they describe the raw integer packing and are wrong
  for the unpacked float values the toolbox stores;
- the scatterometer `WDIR` `standard_name` is rewritten to
  `wind_from_direction` after the 180° rotation (§3.1).

The in-situ CSVs carry no attributes, so their parameter codes get CF
attributes from a built-in table. Every node receives `Conventions: CF-1.8`,
a `history` entry, and a `references` attribute pointing at the product
documentation:

| Source | References |
|---|---|
| Scatterometer | https://osi-saf.eumetsat.int/products/osi-104-b, https://osi-saf.eumetsat.int/products/osi-104-c |
| Altimeter | https://data.marine.copernicus.eu/product/WAVE_GLO_PHY_SWH_L3_NRT_014_001/services |
| SAR | https://s1.pages.eopf.copernicus.eu/s1-l12-rp/main/pfs/index.html |
| In-situ | https://data.marine.copernicus.eu/product/INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/description |

The same attributes are propagated onto the `sar_*`/`val_*` columns of
`collocation_results.nc`. One deliberate omission: `temporal_distance_minutes`
has **no** `units` attribute (the unit is stated in `long_name`) — a
timedelta-like units string would make xarray silently decode the float
column as `timedelta64` on every re-open.

> Code: `core/_cf_metadata.py`, `core/datatree_converter.py`
> (`_subset_point_ds`, `_build_subset_kwargs`, `convert_downloaded_data`).

---

## 5. Collocation choices (step 3)

### 5.1 Who anchors the match

Different validation geometries call for different anchoring:

| Geometry | Anchor | One match per… | Used for |
|---|---|---|---|
| `point_vs_layer` | validation observation | observation (× SAR time) | moorings, buoys, drifters, … vs IW/EW grid |
| `layer_vs_layer`, `cell-averaging` (default) | validation cell | scatterometer/altimeter point | pre-gridded swaths vs IW/EW grid |
| `layer_vs_layer`, `individual` | SAR pixel | SAR pixel | sensitivity studies; many matches, validation points reused |
| WV mode (`point_vs_point` / `point_vs_layer`) | SAR vignette | vignette | sparse OSW point measurements |

**Why observation-anchored by default:** the observation is the scarce,
independent quantity; anchoring on it yields one statistically independent
pair per observation. The `individual` method exists for comparison (each SAR
pixel matched to its nearest scatterometer point, points reused), and the CLI
flag `--layer-vs-layer-collocation-method both` runs both and writes
distinctly-suffixed outputs.

ASCAT/OSI-SAF products are delivered **pre-gridded** — one observation per
12.5 km wind-vector cell — so cell-averaging needs no spatial re-clustering:
each scatterometer point already *is* its own cell, and the SAR pixels inside
that cell are averaged against it.

### 5.2 SAR aggregation instead of nearest-pixel sampling

For each anchor, all SAR pixels within a circular `aggregation_window_km`
are combined into one distance-weighted average, rather than sampling the
single nearest pixel. **Why:** a point observation is compared against the
SAR *footprint-scale* signal, suppressing single-pixel speckle/noise; and a
pre-gridded swath cell physically corresponds to an *area* of SAR pixels.
The window size is matched to the validation sensor:

| Layer type | Window (km) | Rationale |
|---|---|---|
| in-situ (default) | 5.0 | small neighbourhood around a point sensor |
| scatterometer | 12.5 | ASCAT wind-vector-cell size |
| scatterometer | 25 | OceanSat3 & HY-2B/C wind-vector-cell size |
| altimeter 1 Hz | 7.0 | 1 Hz along-track spacing |
| altimeter 5 Hz | 1.4 | 5 Hz along-track spacing |
| radiometer (AMSR2, …) | 25.0 | RSS 0.25° grid-cell size |
| HF radar | 6.0 | typical radial grid spacing |

These built-in defaults apply even when a recipe declares no
`layer_vs_layer` section; a recipe's own `layer_type_specs` override them
per key.

### 5.3 Time and distance tolerances

- **In-situ vs SAR: ±30 min** — following the buoy-vs-SAR interval of
  Abderrahim et al. (2019). The operative spatial constraint is the
  aggregation window (§5.2); the separate `spatial_tolerance_km` recipe
  parameter (default 25 km) is a legacy pre-filter kept for API
  compatibility.
- **Satellite layers vs SAR: ±180 min** (scatterometer, altimeter, radiometer) —
  following the 3-hour match window used for Sentinel-1/scatterometer
  validation in hal-04202202; the 12.5 km spatial scale is the ASCAT cell
  size from the same reference.
- **HF radar: ±30 min** — surface currents decorrelate quickly; HF radar
  fields are typically 10–30 min composites. Note: Finnmark radar only has a
  temporal resolution of 60 min --> needs a longer temporal tolerance
- **soil moisture: ±12 hours** — Sentinel 1 Surface Soil Moisture product
  is a daily file (time stamp at midnight).

### 5.4 Distance weighting

Weighting kernels available: `gaussian`, `inverse_distance`, `linear`,
`equal`, always renormalised over the non-NaN pixels.

- **In-situ default: Gaussian, σ = 2 km** — pixels near the point sensor
  should dominate the aggregate; the Gaussian gives a smooth, tunable
  falloff inside the 5 km window.
- **Satellite layers default: `equal`** — the anchor is itself an area
  average over a regular cell, so every SAR pixel inside that cell
  contributes the same.

### 5.5 The area around WV/OSW vignette points

Sentinel-1 WV mode produces isolated **~20 × 20 km vignettes ~200 km
apart**, each stored as a single point (e.g. `oswHs`). Requiring validation
data within a few km of the vignette *centre* (as the grid matchers would if
the point were faked into a 1×1 grid) would discard almost everything.

**Choice: anchor on the vignette and gather every validation observation
within `sar_footprint_radius_km` = 14 km** — approximately the
centre-to-corner half-diagonal of the 20 × 20 km footprint
(½·√(20² + 20²) ≈ 14.1 km), i.e. the radius that fully covers the footprint.
Observations inside the footprint are aggregated into one match per
vignette:

- **in-situ** observations are plain-averaged (`equal` weights) — labelled
  `point_vs_point`;
- **layer sources** (altimeter, scatterometer) use their own layer-type time
  tolerance and weighting — labelled `point_vs_layer`.

This radius only affects WV/point-mode SAR; IW/EW grid collocation is
untouched.

#### WV wave height: `oswTotalHs`, not an `oswHs` partition

The OSW component reports significant wave height two ways: `oswHs`, which
carries **one Hs per wave partition** (`oswPartitions` axis, `-1` fill for
unused slots), and `oswTotalHs`, the **integrated total significant wave
height** of all wave systems combined. Validation compares against the in-situ
`VHM0` (total significant wave height), so **we extract `oswTotalHs`** — not
`oswHs`.

Taking a single `oswHs` partition (previously partition 0) picks one wave
system, which is not the total sea state; in a real vignette the partitions were
`[0.65, 0.78, 0.54, …]` m while `oswTotalHs` was 2.85 m. Note `oswTotalHs` is
**not** the root-sum-square of the partitions either — it is integrated from the
full spectrum, so it cannot be reconstructed from `oswHs`.

When a (legacy) product lacks `oswTotalHs`, we fall back to the **mean of the
valid `oswHs` partitions** (dropping the `-1`/NaN fill codes) rather than a
single partition, giving a representative total. This is why the WV waves pair
in `_variable_map` is `("oswTotalHs", "VHM0")` with `("oswHs", "VHM0")` kept only
as a legacy fallback.

### 5.6 Smaller collocation choices

- **Missing-reading forward-fill:** if an in-situ observation has a NaN for
  one variable (a sensor gap), the value is taken from that platform's *next*
  reading, provided it still lies within the time tolerance — otherwise the
  variable is simply absent from the match.
- **RVL projection:** in-situ current components (`EWCT`, `NSCT`) cannot be
  compared to the SAR radial velocity directly; they are projected onto the
  SAR look direction using the scene's `rvlHeading`, producing
  `rvlRadVel_projection` as the comparable validation quantity.
- **Exact spherical search:** neighbourhood queries use a KD-tree over
  unit-sphere Cartesian coordinates. Chord length is a strictly monotone
  function of great-circle distance, so a chord-radius ball query returns
  exactly the great-circle neighbourhood — this is a fast *exact* search,
  not an approximation (verified bit-identical to the brute-force Haversine
  implementation it replaced).
- **Plot de-duplication:** grid collocation matches *every* SAR pixel within
  tolerance to an observation, so one buoy reading can produce N rows. For
  scatter/residual plots these are collapsed to one row per observation
  (SAR side averaged), so the annotated N counts observations, not pixels.
  The statistics tables operate on the full pair set.

> Code: `core/collocation.py` (`PointLayerCollocation`,
> `LayerLayerCollocation`, `_collocate_wv_points`, `run_collocation`),
> `core/recipe.py` (`DEFAULT_LAYER_TYPE_SPECS`, `CollocationType`),
> `core/visualization.py` (`_deduplicate_obs`).

### 5.7 ERA5 model validation

ERA5 is collocated as a distinct third collocation type, `model_vs_layer`,
implemented by `ModelLayerCollocation` in `model_collocation.py` — bilinear
spatial + nearest-hour/hyperbolic temporal interpolation, ported from
`relevant_code_for_toolbox/s1_ocn_nwp_coloc/collocate_nwp_to_sat.py`. This is
a different match strategy from `point_vs_layer`/`layer_vs_layer`
(§5.1–§5.6) because ERA5, unlike every other validation source, is a
complete background field defined everywhere, not a real observation with
coverage gaps.

**Why this method does NOT replace the existing `layer_vs_layer` sources**

Raised explicitly by the user during brainstorming ("check whether this
method could also be suitable to take over some of the existing
collocation methods") and investigated after the design was otherwise
settled. Conclusion: no — and this reasoning is written up here since it
documents a real design boundary someone could otherwise reasonably
question again later:

- **Scatterometer/altimeter are geometrically incompatible**: they're
  swath (2-D curved) or along-track (1-D) products, not a regular lat/lon
  grid. `RegularGridInterpolator` requires strictly monotonic 1-D lat/lon
  axes — meaningless for a single track, and not applicable to a curved
  swath without a lossy re-gridding preprocessing step of its own.
- **Radiometer / `hf_radar_grid` / CDS satellite SSM are geometrically
  compatible (regular grids) but still shouldn't switch**: unlike ERA5,
  these are real observations with genuine coverage gaps (swath limits,
  land/RFI/rain flags, radar range, QC drops), not a field defined
  everywhere by construction. Bilinear interpolation across real gaps
  either (a) returns NaN whenever any of the 4 surrounding cells is
  missing — silently losing matches the current nearest-real-observation
  approach would have found — or (b) if gap-handling isn't done
  carefully, risks fabricating a value across a genuine data gap, which
  is a real correctness regression for a tool whose entire job is
  comparing SAR against real independent observations. The current
  point-matching + aggregation-window/distance-weighting approach is
  well-suited to sparse/gapped real data by construction: it only
  produces a match where a real observation exists within tolerance, and
  never invents one.
- **Speed is not a differentiator**: building one `RegularGridInterpolator`
  per time slice and querying it vectorized is roughly comparable in cost
  to building the current unit-sphere KD-tree once and querying it — both
  are effectively O(n log n) build / O(m log n) query at these data
  sizes. Gap-handling logic needed to make bilinear interpolation safe for
  real gridded observations would likely erase whatever small edge it
  has.
- The property that makes ERA5's method both feasible and correct is
  specifically that a model field is *complete* — defined everywhere, no
  missing-data concept. That's also why this same method is the right
  tool again for any *future model* layer (e.g. the ORAS5 currents model
  mentioned as a later addition), but not a general upgrade path for the
  existing observational layer types.

**Cell-averaging needs no spatial interpolation.** In `cell-averaging` mode
the match point *is* the ERA5 grid's own native cell center, so the
"bilinear-interpolated" ERA5 value at that point is just the value already
sitting on the grid node — only the temporal interpolation (nearest-hour or
hyperbolic) does real work. Actual spatial bilinear interpolation across
grid cells is only exercised by `individual` mode (arbitrary SAR pixel/point
locations that don't coincide with a grid node) and by WV-mode's
`collocate_points` (sparse, non-grid vignette locations).

**Antimeridian handling.** The reference script's longitude wrap-padding is
deliberately not ported into `build_spatial_interpolator` (Task 5), since
it's only correct for a global grid and ERA5 downloads here are regional.
Instead, a crossing recipe bbox is split into two non-crossing download
windows (`split_antimeridian_bbox`), stitched into one contiguous grid by
the converter (shifting one window's longitude axis by +360°), with SAR
query longitudes remapped to match at collocation time
(`_normalize_query_lon`/`_wrap_lon_to_pm180` in `model_collocation.py`) —
see Task 14.

**Land-pixel filtering, on both sides of the SAR/ERA5 comparison.**

- *SAR side (`owiMask`, all Sentinel-1 IW/EW wind recipes, not ERA5-
  specific):* `_extract_owi_grid_data` NaNs out `owiWindSpeed`/
  `owiWindDirection` wherever OWI's own `owiMask` bitmask carries the land
  bit (bit 0; a CF flag_values bitmask, so e.g. mask value 5 = land +
  no_data simultaneously). This applies to *every* Sentinel-1 OWI wind
  conversion this toolbox does, regardless of which validation source the
  recipe uses — not something added specifically for ERA5. Live-verified
  to be a low-risk no-op against real Sentinel-1 products: ESA's own OCN
  processor already NaNs `owiWindSpeed`/`owiWindDirection` over land
  before this toolbox ever sees the file. It's kept anyway as a
  defensive/correctness measure — relying on an upstream processor's
  behavior with no independent check would be fragile if that ever
  changes.
- *ERA5 side (`land_sea_mask`/`lsm`, wind only):* ERA5's own `lsm` field
  (requested only for the `wind` variable — see
  `era5_downloader.py`'s `_CDS_VARIABLE_NAMES_BY_VARIABLE`) is used to
  exclude ERA5 grid cells/query points that are themselves over land,
  using the standard oceanographic/ECMWF `lsm > 0.5` threshold. In
  `cell-averaging` mode a native ERA5 cell whose own center is land-
  flagged is skipped entirely, even if valid ocean SAR pixels exist
  nearby within the aggregation window (`_collocate_cell_averaging_grid`).
  In `individual`/WV-mode (`_model_values_at_points`), `lsm` is itself
  bilinearly interpolated to each query point, and any point whose
  interpolated `lsm` exceeds 0.5 has every model variable masked to NaN
  — a point close enough to a land grid cell that its bilinearly-
  interpolated wind value is itself meaningfully blended with land-
  physics wind is treated as too land-contaminated to be a valid ocean
  wind match. The rationale is physical, not just cosmetic: ERA5's land
  and sea near-surface wind fields use different surface-roughness/
  friction physics, so a land grid point's "wind" isn't a comparable
  quantity to SAR ocean wind retrieval regardless of proximity to the
  coast. This side of the filtering is scoped to `wind` only — `waves`
  already gets native NaN-over-land behavior from ERA5's own
  ocean-wave-model output (no separate `lsm` request needed), and
  `soil_moisture` uses the land-only `reanalysis-era5-land` dataset,
  where an ocean/land mask would be nonsensical (the whole point of that
  request is land).

**`u10`/`v10` stay raw through conversion; `WSPD`/`WDIR` are derived only
after interpolation.** Every other validation source is renamed to the
canonical `WSPD`/`WDIR` codes at conversion time (§2). ERA5 wind is the
one deliberate exception: `from_era5` keeps `u10`/`v10` as raw vector
components, and `model_collocation.py`'s `_derive_wind_wspd_wdir` derives
`WSPD`/`WDIR` from them only AFTER bilinear-spatial/hyperbolic-temporal
interpolation has completed (called from `_model_values_at_points` and
`_collocate_cell_averaging_grid`). This is necessary because `WDIR` is a
circular quantity (§6) — 0° and 360° are the same direction — and cannot
be correctly linearly or hyperbolically blended as an ordinary scalar the
way `u10`/`v10` can. Deriving a direction first and then interpolating it
as a plain number produces wrong results whenever the true direction
crosses the 0°/360° seam (e.g. blending 359° and 1° naively yields ~180°,
not ~0°). This was a real bug: an earlier version of `from_era5` derived
`WSPD`/`WDIR` at conversion time, and the interpolation in
`model_collocation.py` treated `WDIR` as an ordinary linear scalar,
producing `val_WDIR` values as far out as -14.77°/376.77° (outside
`[0, 360)`) against real CDS data. Fixed by moving the derivation
downstream, past the point where interpolation happens (commit
`0b196ee`). Because `WSPD`/`WDIR` never exist as a datatree-node variable
for this source, `annotate_collocation_ds` (`core/_cf_metadata.py`)
carries a small fixed CF-attrs fallback keyed on `("era5_wind", "WSPD"
| "WDIR")` so era5-only wind recipes still get correct `units`/
`standard_name` on the final `val_WSPD`/`val_WDIR` columns, matching what
`from_era5` used to stamp directly before the derivation moved.

> Code: `core/model_collocation.py` (`ModelLayerCollocation`,
> `build_spatial_interpolator`, `collocate_points`,
> `_derive_wind_wspd_wdir`), `core/datatree_converter.py` (`from_era5`),
> `core/_cf_metadata.py` (`annotate_collocation_ds`,
> `_DERIVED_VAL_VAR_ATTRS`), `core/collocation.py` (`run_collocation`
> dispatch table).

---

## 6. Circular variables (wind direction)

Plain arithmetic is wrong for angles: a SAR direction of 359° against a buoy
reading of 1° is a 2° error, but linear subtraction reports 358°. Any
validation variable listed in `CIRCULAR_VAL_VARS` (currently `WDIR`) is
therefore treated with directional statistics everywhere a difference or
correlation is computed:

| Metric | Linear variables | Circular variables |
|---|---|---|
| difference | `sar − val` | wrapped to (−180°, 180°] |
| bias | mean difference | circular mean of wrapped differences |
| std | sample std of differences | circular std, √(−2·ln R̄) from the mean resultant length |
| RMSE | RMS of differences | RMS of *wrapped* differences |
| correlation | Pearson r | Jammalamadaka–Sarma circular-circular correlation |

The same wrapped difference is used for the annotations on scatter plots and
for the residual plots, so figures and tables always agree.

> Code: `core/_variable_map.py` (`CIRCULAR_VAL_VARS`, `circular_diff_deg`),
> `core/statistics.py` (`compute_statistics`, `_circular_mean_deg`,
> `_circular_corrcoef_deg`), `core/visualization.py` (scatter/residual
> branches).

---

## 7. Statistics definitions

Per *(SAR variable, validation variable)* pair, grouped by **platform type**
(`val_source`: mooring, buoy, drifter, scatterometer, altimeter, radiometer, …) so every
platform type gets its own row rather than drowning in a single pooled
number:

- **N** — number of valid (non-NaN) collocated pairs
- **bias** — mean(SAR − validation); positive = SAR overestimates
- **std** — standard deviation of the differences (ddof = 1)
- **rmse** — root-mean-square difference
- **correlation** — Pearson r (circular variant for `WDIR`, §6)
- **scatter_index** — RMSE / |mean(validation)|, dimensionless

Grouping by platform type (rather than per-station `val_id`) keeps the
categories coarse enough to be statistically meaningful while still
separating fundamentally different measurement systems.

> Code: `core/statistics.py` (`compute_statistics`, `run_statistics`).

---

## 8. Soil moisture (Sentinel-1 CLMS SSM + ISMN)

### 8.1 Why pytesmo CDF-matching + ubRMSD, not the shared bias/RMSE path

Sentinel-1 CLMS SSM and ISMN in-situ measurements live in different
physical domains (a percent-saturation-like SAR index vs. ISMN's
volumetric m³/m³), so a raw `sar - val` difference is not meaningful. The
soil-moisture recipe rescales the **SAR** series onto the **ISMN**
series' domain via `pytesmo.scaling.scale(method="cdf_match")` before
computing bias/RMSE/correlation — the satellite product is adjusted to
match the in-situ reference, not the reverse, so ISMN's
physically-interpretable volumetric units stay the common comparison
domain (matching standard practice, e.g. ESA CCI SM is CDF-matched to its
reference). A new `ubrmsd` metric (unbiased RMS difference, from
`pytesmo.metrics.ubrmsd`) is reported alongside the existing metrics,
since it is the standard soil-moisture literature metric.

This is implemented as a new parallel function,
`compute_statistics_soil_moisture`, rather than a branch inside the shared
`compute_statistics` — there is no existing plug-in point for external
metrics libraries, and `pytesmo`/`ismn` are optional dependencies (the
`soil_moisture` extra) that the rest of the toolbox must not require.

### 8.2 Depth sign convention: positive below ground, not negative below sea surface

Every other validation source's `min_depth`/`max_depth` is negative
(metres below sea surface, per `DEFAULT_MIN_DEPTH`/`DEFAULT_MAX_DEPTH` in
`core/recipe.py`). ISMN depths are the opposite: **positive metres below
ground**. No dataclass change was needed — `ValidationDataSource.min_depth`/
`max_depth` are already plain floats — but the soil-moisture recipe
template sets them explicitly (`min_depth=0.0, max_depth=0.05`, matching
C-band Sentinel-1's ~5 cm sensing depth) rather than relying on the
ocean-oriented global defaults. A future NISAR (L-band) recipe would use a
deeper window (~0-0.25 m) the same way.

### 8.3 Collocation tolerances are pixel-scale, not buoy-footprint-scale

The soil-moisture template's `point_vs_layer` collocation
(`spatial_tolerance_km=2.0`, `aggregation_window_km=1.0`,
`distance_weighting="equal"`, `time_tolerance_minutes=720`) is
deliberately much tighter than the ocean default
(`aggregation_window_km=5.0`, Gaussian weighting, 30-minute tolerance): a
1 km SAR pixel and a point ISMN station don't need a 25 km buoy-scale
footprint, and the product is a daily composite, so only a same-calendar-day
match is meaningful (±12 h tolerance).

### 8.4 Why ISMN has no automated downloader

ISMN (https://ismn.earth) has no public download API — only a
registration-gated web portal. `ISMNDownloader` is therefore a
**local-archive selector**, not a network client: it expects a
manually-downloaded zip/folder, and on first run without one it prints the
exact bbox/date/depth/variable filter values to paste into the portal
form, along with the run's own ISMN output folder to drop the resulting
zip into — `download()` auto-detects the most-recently-modified `*.zip`
sitting there, so no recipe edits are needed for the common one-archive-
per-run case. Since the portal itself supports the same filters, one
manually-downloaded archive can also be scoped tightly and reused across
multiple recipes whose windows fall inside it, by setting its path
explicitly via the `ismn` validation source's
`download_kwargs: {ismn_archive_path: ...}` (which still takes priority
over auto-detection when set).

The printed instructions also recommend three portal download options:
**CEOP-formatted** archives (variables in separate files, zipped) — the
`ismn` Python package auto-detects this format from the file structure
(confirmed against the installed package's `filehandlers.py`, which reads
`ceop_sep` and `header_values` both natively; only the legacy single-file
CEOP format was dropped upstream), and CEOP is the more actively-maintained
of the two;
**"Good" quality flags only** — flagged-bad observations would otherwise
corrupt validation statistics, and `ISMNDownloader` itself does no
quality-flag filtering, so this is enforced at download time instead;
**gap filling disabled** — this toolbox's `point_vs_layer` collocation
matches on actual observation timestamps within a tolerance window, not a
fixed calendar grid, so NaN-filled placeholder rows (to guarantee 24
points/day) add nothing but archive size and get dropped downstream anyway.

### 8.5 Report plots compare the CDF-matched pair, not the raw one

§8.1's CDF-matching happens inside `compute_statistics_soil_moisture` and,
originally, was discarded once the metrics were computed — so
`validation_report`'s scatter/geographic/residual plots were still reading
the **raw** `sar_sarSSM`/`val_SOIL_MOISTURE` columns from
`collocation_results.nc` directly. Since those live in different physical
domains (confirmed against a real run: `sar_sarSSM` 2.75-100 "%",
`val_SOIL_MOISTURE` 0.047-0.78 "1"), every plot was comparing
non-comparable values — points didn't cluster anywhere near the 1:1 line,
and there was no principled way to read the result.

`statistics.add_rescaled_sar_column(collocation_ds, sar_var, val_var)`
reuses the same per-group CDF-matching (`_cdf_match_sar_series`, factored
out of `_rescale_and_compute_soil_moisture_stats` so both call sites share
one implementation) to return a copy of the collocation Dataset with
`sar_<var>`'s values replaced by their rescaled equivalent — units/long_name
attrs copied from the validation column, since the values now live in that
domain. `validation_report` calls this once per pair, only when
`recipe.config.variable == "soil_moisture"`, before `plot_scatter`,
`plot_residuals`, and `plot_temporal_offset` run, so those point-based plots
compare like with like.

`plot_geographic` is the deliberate exception: it keeps the **pre-rescale**
collocation Dataset (`geo_pair_ds` in `validation_report`'s loop), not the
rescaled one. Its background SAR field comes from the full `(y, x)` grid in
`datatree.nc`, not from `collocation_ds`, and can't be point-rescaled the
same way — CDF-matching needs a paired validation value, which only exists
at collocated points, not at every background pixel. Instead,
`statistics.fit_sar_to_val_transform(collocation_ds, sar_var, val_var)` fits
a `pytesmo.cdf_matching.CDFMatching` transform on the collocated
`(sar_<var>, val_<var>)` pairs and returns a callable that `plot_geographic`
applies to *every* pixel of the SAR field, converting the whole background
layer into the validation's domain so one shared, meaningful colorbar covers
both. This transform must be fit from the **raw** SAR/validation pairs, not
the already-rescaled ones — fitting on rescaled input and then applying the
result to the real, raw SAR field (still 0-100) extrapolates wildly, since
the fit's training domain no longer matches what it's applied to (confirmed
against real data: predicted values above 300 for a variable that should
span roughly 0-1). This is why `validation_report` threads two separate
dataset variables through its per-pair loop — the rescaled one for
scatter/residuals/temporal-offset, the original raw one for
`plot_geographic`.

When there isn't enough collocated data to fit a transform (fewer than two
valid pairs, or the underlying CDF-matching degenerates), `plot_geographic`
falls back to giving the SAR field and the validation points independent
percentile ranges and colorbars instead — better two honestly-scaled
colorbars than one silently wrong or crashing one. Recipes where both sides
already share units (wind, currents, waves) skip all of this — pooling one
colorbar directly is still correct there.

Axis/colorbar labels also now append each variable's CF `units` attribute
generically (`_labeled_var` in `visualization.py`) — a side benefit for
every existing variable pair (e.g. `"WSPD (m s-1)"`), not just soil moisture.

### 8.6 ISMN stations averaged over ±12h per SAR scene, not deduped to nearest

ISMN's recommended recipe tolerance (`time_tolerance_minutes: 720`, ±12h)
is deliberately wide, to tolerate ISMN's own reporting gaps. But ISMN
reports hourly — far more densely than one daily SAR overpass — so without
any further change, every hourly reading within that ±12h window became
its own separate collocation. Confirmed against a real 118-station recipe
run: one station alone produced ~25 matches against a single SAR scene
(all sharing the same `sar_y_idx`/`sar_x_idx`, differing only in which
hourly ISMN reading they carried), inflating a real run from a sane
station-count to 1517 total collocations.

The original fix — "only keep the closest-in-time validation reading" —
worked for the inflation problem, but introduced a real bias: since S1 SSM
scenes are always stamped at exactly midnight UTC, the "closest" reading
is *always* a nighttime one. Soil moisture has a strong diurnal cycle, so
comparing S1 SSM (representing the whole day) against only ISMN's
nighttime value systematically misrepresents the comparison.

**Current behaviour: average, don't pick.** `run_collocation` now
pre-aggregates each ISMN station's raw hourly readings — grouped by
`platform_id` (falling back to `(lon, lat)` when no `platform_id` column
exists) and matched to its nearest SAR scene time via
`pd.merge_asof(direction="nearest", tolerance=...)` — into a single
station-day mean *before* `PointLayerCollocation.collocate()` ever runs
(`_average_within_sar_tolerance` in `core/collocation.py`). Every reading
within `time_tolerance_minutes` of a SAR scene contributes to that scene's
average; readings outside every scene's tolerance window are dropped, same
as before. This applies only to sources whose `platform_type` attr is
`"ismn"` — every other point_vs_layer source (moorings, buoys) keeps every
in-tolerance reading as its own independent collocation, unchanged
(`test_no_temporal_averaging_uses_raw_value`,
`test_each_point_matched_independently`, `test_mooring_source_unaffected_keeps_every_reading`
document this).

`dedup_nearest_in_time` (the mechanism this superseded for ISMN) remains
available on `PointLayerCollocation`/`LayerLayerCollocation` and is still
used by `hf_radar_grid` — it was never ISMN-specific machinery, just no
longer ISMN's chosen behaviour.

### 8.7 Satellite soil-moisture sources (ASCAT/AMSR-E-2/SMAP)

These three sources are gridded satellite products (like scatterometer/
radiometer wind), so they plug into `layer_vs_layer` collocation, not
`point_vs_layer`. They reuse the ISMN's own `SOIL_MOISTURE` canonical code
(§2) rather than getting their own — same reasoning as `WSPD` unifying
wind sources: one shared code means one report section, not four separate
ones each showing the others as "no data".

**Time tolerance: 720 minutes (±12h), not the 180 minutes inherited from
wind/wave scatterometer defaults.** Soil moisture is a slowly-evolving
quantity relative to a satellite's one-or-so daily overpass — the same
reasoning already applied to ISMN's own ±12h tolerance (§8.3) — whereas
180 minutes reflects wind/waves' much faster decorrelation time. Applying
the wind-derived 180-minute default here would under-match a genuinely
slow-changing quantity for no benefit.

**Overpasses within that ±12h window are averaged together, including
ascending and descending passes.** Like ISMN (§8.6), these sources are now
pre-aggregated before collocation runs, but the grouping key depends on
the source's format:

- **Km-based sources** — ASCAT (WARP5 grid), SMAP (EASE grid), SMOS (EASE
  grid), AMSR2's NSIDC-0451 (25 km EASE grid), and AMSR2's AU_Land
  half-orbit swath (lon/lat vary continuously, not a fixed grid) — are
  grouped by `(lon, lat)` spatially snapped via `_snap_to_grid`, using
  each source's own `aggregation_window_km` converted to a degree step
  with a cos(latitude) correction on the longitude step (a degree of
  longitude covers less physical distance than a degree of latitude away
  from the equator). This correctly merges nearby-but-not-identical
  pixels for AU_Land, and is effectively a no-op for the other, already-
  gridded km-based sources.
- **AMSR2's G-Portal format specifically** is stamped with a
  `native_grid_deg` attribute (0.1°) at conversion time
  (`_from_amsr_ssm_gportal_l3_grid`) and is grouped by its raw, exact
  `(lon, lat)` instead of being snapped at all — it's already a fixed
  equirectangular lattice, so repeated readings of the same cell report
  IDENTICAL coordinates and snapping buys nothing. (An earlier version of
  this code snapped G-Portal too, rounding its cell centres — which sit
  at exact half-step offsets like 9.05°/9.15°/9.25°/... — to a 0.1°
  step; every centre landed exactly on a rounding tie, and NumPy's
  round-half-to-even resolved those ties inconsistently, silently merging
  roughly 1 in 5 pairs of genuinely adjacent native cells. Grouping on
  the raw coordinates avoids the rounding step entirely.)

Either way, groups are then matched to the nearest SAR scene time and
averaged, exactly as ISMN readings are. **This means an ascending (e.g.
early-morning) and a descending (e.g. evening) overpass of the same cell
on the same day are blended into a single value**
if both fall within the ±12h window — a deliberate methodological choice,
consistent with the ISMN treatment, not an incidental side effect. It also
shrinks the point count for these sources, which without any collapsing
could otherwise approach ~100,000 collocations for a large domain/date
range, since multiple overpasses per cell no longer each produce their own
row before the per-row collocation loop runs.

**A real, not just cosmetic, limitation of pytesmo's CDF-matching (§8.1)
for these sources:** `pytesmo.scaling.scale(method="cdf_match")` rescales
by rank/percentile, so it is blind to *why* two series differ. For SAR vs.
ISMN today, that gap is "SAR's saturation-like index vs. ISMN's volumetric
m³/m³" — one cause, cleanly corrected. For SAR vs. these four new sources,
the CDF-match will *also* silently absorb at least three more, physically
distinct gaps at once:

1. **Retrieval units** — ASCAT reports % saturation (same domain as SAR);
   AMSR-E/2, SMAP, SMOS, and SAR CLMS all report volumetric m³/m³, but S1 SAR's
   own SSM product is itself a % saturation index, so every one of these
   comparisons still crosses the same unit boundary § 8.1 already crosses
   for ISMN.
2. **Sensing depth / band** — C-band SAR and ASCAT (~0-5 cm), X/Ka-band
   AMSR-E/2 (~0-1 cm skin depth), and L-band SMAP/SMOS (~0-5 cm, but a
   different retrieval physics) do not sense the same soil layer. No
   depth-adjustment model is implemented (would require external soil
   dielectric/texture data, out of scope) — depths are only *documented*,
   per source, via each converter's `long_name` and surfaced in the
   validation report cover page (§ new `plot_...` cover-page text, listing
   each present source's representative depth/band) so a reader isn't misled
   about what's being compared.
3. **Native footprint/resolution** — 9-50 km depending on source, vs. SAR's
   1 km CLMS grid.

Because CDF-matching corrects only the *marginal distribution*, a good
post-rescale statistical fit (low RMSE/high correlation in the rescaled
domain) does **not** certify that two sources are physically equivalent,
and comparing fit quality *across* validation sources (e.g. "SAR agrees
better with SMAP than with ASCAT") is not a like-for-like measure of
retrieval accuracy — each pair is scored in its own bespoke rescaled
domain, shaped by whichever mix of the three gaps above that pair happens
to have. This is an inherited limitation of the existing ISMN pipeline
(§8.1), not a new one introduced by these sources — but it compounds
across more, and more varied, gaps than the single ISMN case did, which is
why it's worth stating explicitly rather than leaving implicit.

**Mitigation: a second, unit-native comparison, for pairs where it's
actually possible.** Since gap 1 (units) only exists for *some* pairs — SAR's
`%` matches ASCAT's `%` exactly, and a future L-band SAR source (e.g. NISAR)
would match ISMN/AMSR/SMAP/SMOS's `m³/m³` exactly — those same-unit pairs
don't need CDF-matching's rank-based workaround at all, and can be compared
directly in physical units instead. **Choice: whenever `sar_var`'s and
`val_var`'s `units` attrs fall in the same unit family** (`%` vs `%`, or any
of `m3 m-3`/`cm3 cm-3`/`1` — all volumetric — vs each other), **compute a
second statistics table using the plain, generic `compute_statistics` path**
(the same bias/RMSE/correlation/SI/ubrmsd machinery every other recipe
variable already uses, with no rescaling) **alongside** the existing
CDF-matched one — not instead of it, since CDF-matching still covers the
pairs that don't share units. This second table only ever contains the
subset of platform types whose units genuinely match SAR's; e.g. for
Sentinel-1 (`%`), that's `ascat_ssm` only, while ISMN/AMSR/SMAP/SMOS stay
CDF-matched-only. A future NISAR (`m3 m-3`) recipe would instead see
ISMN/AMSR/SMAP/SMOS in the native-units table and ASCAT CDF-matched-only —
the same mechanism handles both without a per-source-type allow-list, since
it's driven by the `units` attrs already set at conversion (§ new converters'
step 3), not a hardcoded list of "which sources are volumetric."

**Mechanism detail: units are resolved per `val_source`, not read off the
pooled collocation column.** `collocation_results.nc`'s `val_SOIL_MOISTURE`
column pools every validation source's raw values into one column,
distinguished only by the `val_source` group label (e.g. `"ismn"`,
`"ascat_ssm"`) — a single `units` attribute on that column cannot represent
per-row units once sources with genuinely different units (ASCAT's `%`
alongside ISMN's `m3 m-3`) are pooled together. So the native-units gate
uses a small explicit lookup, keyed by `val_source` (the same platform-type
label already used for per-group statistics), not by reading `units` off
the pooled `val_<var>` column:
```python
_VAL_SOURCE_UNITS_FAMILY = {
    "ismn": "volumetric", "ascat_ssm": "percent_saturation",
    "amsr_ssm": "volumetric", "smap_ssm": "volumetric", "smos_ssm": "volumetric",
}
```
The **SAR** side has no such ambiguity — one recipe run has exactly one SAR
product, so its `sar_<var>` column's `units` attr is read directly and
normalized (`"%"` → `"percent_saturation"`; `"m3 m-3"`/`"cm3 cm-3"`/`"1"` →
`"volumetric"`) to get the family to match against.

Output: a second file per soil-moisture run,
`validation_statistics_<sar_var>_vs_<val_var>_native_units.nc` (parallel to
the existing `..._individual` suffix convention for the alternate
collocation method), written only when at least one same-unit platform type
is present. `validation_report` gains a parallel scatter/residual/geographic
plot section for these same-unit pairs, titled distinctly (e.g. "— native
units") so it's never confused with the CDF-matched plots (§8.5) covering
the full source set.

> Code: `core/datatree_converter.py` (`from_ascat_ssm`, `from_amsr_ssm`,
> `from_smap_ssm`, `from_smos_ssm`),
> `core/statistics.py` (`run_statistics_native_units`, `add_rescaled_sar_column`,
> `fit_sar_to_val_transform`, `_VAL_SOURCE_UNITS_FAMILY`),
> `downloaders/ascat_soil_moisture_downloader.py`,
> `downloaders/earthdata_soil_moisture_downloader.py`,
> `downloaders/smos_downloader.py` (`SMOSDownloader`, `authenticate_smos_ftp`).

### 8.8 SMOS: OADS's real HTML/data formats, confirmed against live runs

`SMOSDownloader` and `from_smos_ssm` were originally written and shipped
without ever running against ESA's live OADS portal or a real downloaded
product (no credentials were available at the time) — both the login flow
and the NetCDF converter were later found to make several wrong,
unconfirmed assumptions once a real user ran them, all fixed and now
CONFIRMED live. Recorded here since each one is easy to silently
reintroduce by "cleaning up" the code without a real account to test
against.

**FTPS was replaced with the OADS HTTP portal, not merely retried.**
`smos-diss.eo.esa.int`'s FTP/FTPS control channel completes its TCP
handshake but never sends the welcome banner — confirmed hung
indefinitely from this toolbox's environment, and not a general
FTP-blocking issue (other public FTP servers responded normally from the
same environment). ESA's OADS web portal serves the same NRT product over
plain HTTPS and responds normally, so `SMOSDownloader` browses/logs in/
downloads via OADS instead of FTP entirely.

**OADS's SAML2/WSO2 SSO login HTML mixes single- and double-quoted
attributes unpredictably — never assume one quote style.** Confirmed
across three separate parsing bugs, each only caught by a real login
attempt: the login form's `sessionDataKey` hidden input is single-quoted
(`value='...'`) while the surrounding form uses double quotes; the login
form's own `action` is a *relative* URL (`action="../samlsso"`), resolved
via `urllib.parse.urljoin(resp.url, action_url)` against the actual
post-redirect page URL, not posted as-is; and the IdP's ACS auto-submit
response form uses single quotes for **both** `name=` and `value=` on its
`SAMLResponse`/`RelayState` inputs, a different style again from the
login form. `SMOSDownloader._login`'s regexes now match every quoted
attribute (`name=`, `value=`, `action=`) as `["\']` rather than assuming
`"`.

**OADS's product-listing link format also depends on session state, and
carries an unexpected extra attribute.** An authenticated session's
"Download Product" link is a direct download URL
(`/oads/data/NRT_Open/<filename>`), not the login-gated redirect
(`/oads/access/login?r=...&d=...`) an unauthenticated fetch shows — and
the `<a>` tag carries an extra `target="_blank"` attribute between `href`
and the closing `>`, which the original regex's exact
`href="...">Download Product</a>` assumption didn't allow for and
silently matched zero products against. `_list_products_for_day`'s regex
now allows arbitrary attributes (`[^>]*`) after `href`'s closing quote,
and both href shapes are covered by tests.

**`from_smos_ssm`'s real NetCDF field names/time convention, confirmed
against real downloaded products:** fields are lowercase
(`soil_moisture`/`longitude`/`latitude`), not the capitalised names
originally guessed; there is no single `time` variable — per-point
acquisition time is split across `days_since_01-01-2000` (int, days since
the SMOS epoch 2000-01-01T00:00:00) and `seconds_since_midnight` (int,
seconds within that day), combined as
`pd.Timestamp("2000-01-01") + pd.to_timedelta(days, unit="D") +
pd.to_timedelta(seconds, unit="s")` — verified against a real file's own
filename-encoded acquisition window. Real `soil_moisture` data showed no
`-999.0`-style fill sentinel (a full granule's min/max were both clean
physical values, no NaN), so validity now relies primarily on
`~np.isnan` (whatever `xr.open_dataset`'s default `mask_and_scale=True`
already decoded), keeping the `-999.0` check only as cheap, harmless
insurance rather than the load-bearing check it was before.

> Code: `downloaders/smos_downloader.py` (`SMOSDownloader._login`,
> `_list_products_for_day`, `download`),
> `core/datatree_converter.py` (`from_smos_ssm`).

### 8.9 CDF-matched section: ASCAT converted into ISMN's volumetric domain

ASCAT reports `%` saturation — the same domain as Sentinel-1 SSM's own raw
retrieval (§8.7) — while ISMN/SMAP/SMOS report `m3 m-3` volumetric
fraction. Pooling both domains onto one shared scatter axis or geographic
colorbar (the original CDF-matched report section's behavior) squashed the
volumetric sources near the origin/one end of the color scale and made the
pooled N/bias/RMSE/r annotation physically meaningless.

**Fix: convert ASCAT into ISMN's volumetric domain, reusing the transform
already fit for the ISMN group.** Since SAR's own raw retrieval shares
ASCAT's `%` domain, the CDF-match transform fit from SAR-vs-ISMN
collocated pairs (`SAR raw % → ISMN raw m3 m-3`) is equally valid applied
to ASCAT's own raw values — this avoids needing to collocate ASCAT against
ISMN directly (they are never paired with each other, only each with SAR),
and reuses the same out-of-sample-application pattern
`fit_sar_to_val_transform` already used for painting a full SAR scene.
`_harmonize_percent_domain_sources` (`core/statistics.py`) implements this,
called by `add_rescaled_sar_column`, `compute_statistics_soil_moisture`,
and `fit_sar_to_val_transform` — so the CDF-matched scatterplot,
geographic plot, and statistics table all agree, and no downstream
plotting code needs its own domain-splitting logic.

**Mechanism: detection via existing `_VAL_SOURCE_UNITS_FAMILY` lookup, not
per-row inspection.** Which sources need converting is determined by checking
whether each source's units family (via the existing `_VAL_SOURCE_UNITS_FAMILY`
dictionary, the same lookup already used by `run_statistics_native_units` in
§8.7) matches SAR's own units family — correct for today's Sentinel-1 (`%`)
case. This function is a true no-op for every non-soil-moisture recipe and
every soil-moisture recipe whose val_source labels aren't in that dictionary,
so the overhead is negligible.

**Fit details: RAW input paired with HARMONIZED target.** Two separate
CDF-matching fits are involved, and the pre-fix bug came from conflating
them. `_harmonize_percent_domain_sources` fits its own internal SAR→ISMN
transform (from raw SAR-vs-ISMN pairs; this fit itself was never buggy) and
uses it to convert ASCAT's raw `sar_col` *and* raw `val_col` into ISMN's
volumetric domain. `fit_sar_to_val_transform` then runs its own, separate
pooled fit on top of that harmonized dataset, producing the callable applied
later to an entire raw SAR scene raster (`plot_geographic`'s background
layer) — not to ASCAT's values specifically. Pre-fix, that second fit's `df`
took *both* its x (`sar_col`) and y (`val_col`) columns from the harmonized
dataset: for ASCAT rows `sar_col` had already been converted to volumetric,
while for ISMN rows — never touched by the harmonize step, since ISMN is the
reference — `sar_col` stayed raw percent. The pooled x-input was therefore
domain-inconsistent (ISMN rows in percent, ASCAT rows in volumetric), which
skewed the percentile binning used by CDF-matching; it was not a case of "no
harmonization." The fix keeps `sar_col` raw for every row (read straight
from `collocation_ds`, which the harmonize step never touches) and pairs it
with `val_col` from the harmonized dataset — volumetric for ASCAT via the
conversion above, and volumetric for ISMN because it was already volumetric.
The pooled fit's x-input is now consistently raw-percent across every row,
and its y-target is consistently volumetric across every row.

**Fallback:** if ISMN has too few (or zero) collocated points to fit that
transform for a given run, ASCAT is dropped from the CDF-matched section
for that run (logged) — it still appears in the native-units section
(§8.7), which compares ASCAT against SAR directly in `%` and doesn't
depend on ISMN at all. The native-units section remains the *only* place
`%`-domain values appear in the report; the CDF-matched section is always
volumetric once any conversion happens.

**Missing colorbar (unrelated bug, fixed alongside).** Soil moisture's
geographic plot uses a dedicated two-column-by-scene layout
(`_build_scene_pair_figure`, added 2026-07-24) that never called
`fig.colorbar()` at all — structurally absent, not an intermittent
failure from the domain-mixing bug above. Fixed by adding the same
shared/two-colorbar logic `_build_figure` (every other variable's
geographic layout) already had.

**Scope limitation.** This hardcodes ISMN as the volumetric reference and
detects "needs converting" as "shares SAR's own raw units family" — correct
for today's Sentinel-1 (`%`) case. A future non-percent SAR product (e.g. a
hypothetical NISAR, `m3 m-3`) would need this reference choice revisited,
since ASCAT would then be the odd one out with no analogous "percent
reference source" to convert onto.

> Code: `core/statistics.py` (`_harmonize_percent_domain_sources`,
> `_soil_moisture_metrics`, `add_rescaled_sar_column`,
> `compute_statistics_soil_moisture`, `fit_sar_to_val_transform`),
> `core/visualization.py` (`_build_scene_pair_figure`).

### 8.10 CDF-matched scatter forces a per-source split once harmonization ran

`plot_scatter` already splits into one small-multiples subplot per
`val_source` when a single source holds >70% of the points
(`split_when_imbalanced`, §9.2) — but a harmonized soil-moisture pair (§8.9)
can stay visually busy even when no source is that dominant: real data with
ASCAT (~7400), SMAP (~7400), ISMN (~30) and SMOS (~1600) points all sharing
one now-common volumetric domain and one shared axes produced an unreadable
overlapping mess at ASCAT's ~45% share, well under the 70% trigger.

**Fix: `plot_scatter(..., force_split=True)`.** `validation_report` sets
this whenever `_harmonize_percent_domain_sources` actually converted a
source for the current pair (non-empty `converted_sources`) — i.e. whenever
the CDF-matched section's sources no longer share their original native
domain — regardless of `dominant_share`. Applies to both the main
CDF-matched scatter and its "colored by temporal offset" twin (same
`force_split` value passed to both calls). A true no-op for every
non-soil-moisture recipe, since harmonization itself never runs for them.

> Code: `core/visualization.py` (`plot_scatter`'s `force_split` param,
> `validation_report`'s `harmonized_sources` check).

### 8.11 NISAR SME2 (beta): a second SAR source, selected per recipe

`soil_moisture` is the first recipe category to offer more than one real
SAR-side product: Sentinel-1 CLMS Surface Soil Moisture (`%`, C-band,
once-daily merged raster, `source: sentinel1_clms_ssm`) or NISAR SME2
(`m3 m-3`, L-band, twice-daily per-overpass granules,
`source: nisar_sme2`). `sar_validation/core/sar_sources.py`'s `SAR_SOURCES`
registry is the single place mapping this key to its downloader, output
subdirectory, converter, and per-source recipe-template defaults (depth,
collocation tolerances) — the same registry also carries `wind`/`waves`/
`currents`' existing `sentinel1_l2_ocn` entry, so the same `sar_data.source`
mechanism is available to every category, even though only `soil_moisture`
has a second real choice today.

**Units require no `statistics.py` changes.** `_harmonize_percent_domain_sources`
and `run_statistics_native_units` already key off the SAR variable's actual
`units` attribute at runtime, not a hardcoded "Sentinel-1 = %" assumption —
confirmed by tracing both functions directly. A NISAR (`m3 m-3`) recipe
automatically classifies ISMN/AMSR/SMAP/SMOS as the "native units" group
and ASCAT as the CDF-matched-only outlier, the reverse of today's
Sentinel-1 case, with zero code changes.

**Time-averaging requires no `collocation.py` changes either.**
`_average_within_sar_tolerance` (§8.6/§8.7) matches validation readings to
whichever real `sar_scene_times` it's given — Sentinel-1 CLMS SSM's scenes
happen to always sit at midnight UTC, but NISAR SME2's twice-daily granules
each carry their own real overpass timestamp, so the NISAR recipe template
simply uses a tighter `time_tolerance_minutes=360` (±6h) instead of 720
(±12h) — a template default, not a code change.

**Depth window (0–0.05 m) and spatial tolerances (`aggregation_window_km=0.2`,
`spatial_tolerance_km=2.0`) are documented assumptions**, not confirmed
NISAR SME2 product-documentation values — 0–0.05 m matches the same window
already used for Sentinel-1 CLMS SSM and (implicitly) SMAP's own documented
L-band near-surface retrieval depth in this codebase; 200 m resolution
(hence `aggregation_window_km=0.2`) was recorded from an earlier deferred
spec, not independently re-verified. `spatial_tolerance_km` is kept equal
to Sentinel-1's, by explicit choice, rather than deriving a separate
resolution-scaled value. All are trivially overridable per-recipe.

**CMR identifiers and HDF5 layout both confirmed 2026-07-31 against real
data — Task 9 complete.** `sar_sources.py`'s `NISAR_SME2_SHORT_NAME`/
`NISAR_SME2_VERSION` were originally a guess (`"NISAR_L3_PR_SME2_BETA"`/
`None`) and returned zero results against the real CMR — the actual
collection is `NISAR_L3_SME2_BETA_V1` (version `"1"`, concept id
`C2850265000-ASF`, "NISAR Beta Soil Moisture (Version 1)"), confirmed
directly against NASA's live CMR catalog (13,881 granules total as of the
check date; granule ids do start with `NISAR_L3_PR_SME2_` and are `.h5`,
matching this source's `file_glob`). A separate, more granule-dense
`NISAR_L3_SME2_PROVISIONAL_V1` collection (`C2854344945-ASF`) also exists —
not used here, since BETA matches the maturity level this source's
docs/attrs were written for; switching would be a deliberate
product-choice decision.

`datatree_converter.py`'s `from_nisar_sme2` was then verified against a
real downloaded granule (`NISAR_L3_PR_SME2_003_005_A_014_..._001.h5`,
inspected directly with `h5py`) and every placeholder assumption turned
out wrong: `soilMoisture`/`longitude`/`latitude` live directly under
`science/LSAR/SME2/grids`, not a `frequencyA` subgroup (a
`grids/radarData/frequencyA` subgroup does exist, but holds only
backscatter/sigma0 fields, no soil moisture); `longitude`/`latitude` are
1-D EASE-grid axis arrays, not a 2-D meshgrid (now meshed via
`np.meshgrid`, the same way `from_sar_l3_ssm_geotiff` meshes its GeoTIFF
axes); the fill value is `soilMoisture`'s own `_FillValue` dataset
attribute, not a group-level attribute; the acquisition time is a scalar
string dataset at `science/LSAR/identification/zeroDopplerStartTime`, not
a root file attribute. `retrievalQualityFlag` (sibling to `soilMoisture`)
was checked and flags exactly the same cells `soilMoisture`'s own fill
value already does, so no separate quality-flag masking was added. Fixed
and re-verified end-to-end: `--sar-source nisar_sme2` now correctly finds,
downloads, converts (325,109 valid soil-moisture cells confirmed on one
real granule), and collocates real NISAR SME2 data (5,480 collocated
pairs against a real recipe run). The synthetic test fixture in
`TestFromNisarSme2` was updated in lockstep to match this real layout.

**Known gap: `_harmonize_percent_domain_sources` cannot harmonize ASCAT
for a NISAR recipe.** `_harmonize_percent_domain_sources` (in
`core/statistics.py`) converts a validation source sharing SAR's own raw
units family (e.g. ASCAT's `%`) into the reference source's domain, by
reusing the already-fitted SAR-vs-reference CDF transform. This only works
when SAR's raw domain matches the source needing conversion — true for
Sentinel-1 CLMS SSM, whose raw domain is `%`, the same as ASCAT's. For
NISAR SME2, whose raw domain is volumetric (`m3 m-3`), the same as the
ISMN reference already, the function's `to_convert` set is always empty by
construction: a source can't simultaneously equal `sar_family` and differ
from `reference_family` when `sar_family == reference_family`. So for a
NISAR recipe, ASCAT's raw `%` values are never harmonized into the shared
volumetric domain, unlike the Sentinel-1 case. Concretely, this means
`fit_sar_to_val_transform` (used for `plot_geographic`'s background-raster
color scale) and the CDF-matched-scatter `force_split` decision in
`visualization.py` may pool ASCAT's un-harmonized raw percent values with
volumetric values from ISMN/AMSR/SMAP/SMOS for a NISAR recipe — the exact
"pooling raw percent and raw volumetric pairs with no harmonization"
scenario that `fit_sar_to_val_transform`'s own docstring calls
"nonsensical." This is a known, currently-unaddressed gap for
`nisar_sme2` recipes with ASCAT enabled. It is left unresolved here
deliberately (fixing it is a design task, not a documentation one) and is
deferred to be resolved as part of the real-data verification pass
(Task 9), once NISAR SME2 data can actually be run end-to-end and this
path can be exercised and tested for real.

**NISAR SME2's underlying CMR collection changed mid-mission, with a real
~5-month gap in between — `EarthdataSoilMoistureDownloader` now queries
multiple candidates and merges results.** Confirmed 2026-07-31 directly
against NASA's live CMR catalog, cross-checked against a real
user-reported coverage gap that was itself independently cross-checked
against ASF Vertex: `NISAR_L3_SME2_BETA_V1`'s real granules run
2025-10-01 through 2026-01-20, then nothing (not even in Vertex) until
`NISAR_L3_SME2_PROVISIONAL_V1`'s real granules pick up on 2026-06-17 — a
hard product-maturity transition (beta → provisional) with zero temporal
overlap between the two collections. Hardcoding a single short_name (as
originally shipped) or a date-based cutoff to pick between the two (the
pattern already used for AMSR2's NSIDC-0451/AU_Land switch in
`orchestrator.py`) would both be guesses about exactly when NASA's
processing pipeline transitioned — and this codebase already has one
example of that kind of guess turning out wrong (`AU_Land_NRT_R02`, see
below). Instead, `EarthdataSoilMoistureDownloader.__init__`'s `dataset`
parameter now accepts either a single short_name (backward compatible,
used unchanged by AMSR2/SMAP) or a list of `(short_name, version)`
candidates; `download()` queries every candidate and merges the results,
letting CMR itself be the source of truth for which collection actually
has data in the requested window. `sar_sources.py`'s
`NISAR_SME2_CANDIDATES` wires both `NISAR_L3_SME2_BETA_V1` and
`NISAR_L3_SME2_PROVISIONAL_V1` in for `nisar_sme2`.

**Diagnostics-plot per-scene splitting and auto-zoom are now
registry-driven / data-driven instead of hardcoded for soil_moisture in
general.** `plot_collocation_diagnostics` used to split into one PNG per
SAR scene for *any* soil_moisture recipe with multiple scenes — correct
for Sentinel-1 CLMS SSM (daily, mutually-overlapping, continent-wide
mosaics, where overlaying more than one makes individual days
indistinguishable) but wrong for NISAR SME2 (small, non-overlapping
per-orbit granules that coexist fine on one map, where splitting just
produces many near-empty PNGs instead of one useful overview). Fixed via
a new `SARSourceSpec.diagnostics_split_by_scene` flag (`True` only for
`sentinel1_clms_ssm`). Separately, the plot always set its extent to the
recipe's full requested bbox, which is fine when SAR coverage roughly
fills that bbox (Sentinel-1) but leaves a tiny, hard-to-read data cluster
in an otherwise-empty map when it doesn't (NISAR granules against a
continent-scale bbox). Fixed by computing the actual combined extent of
every SAR scene/footprint/coverage-pixel and validation point the plot
already draws, padding it, and clamping the result to never exceed the
recipe's own bounds (so it only ever zooms *in*, never out) — a no-op for
Sentinel-1-scale coverage, a clear improvement for NISAR-scale coverage.
Dateline-crossing recipes keep the original always-full-bounds behavior
(combining that longitude-shifting logic with a data-driven zoom safely
was out of scope for this fix).

> Code: `core/sar_sources.py` (`SARSourceSpec`, `SAR_SOURCES`,
> `NISAR_SME2_CANDIDATES`), `core/recipe.py` (`SARDataSpec.source`,
> `_build_sar_data_spec`), `core/orchestrator.py` (`_download_sar`),
> `core/datatree_converter.py` (`convert_downloaded_data`'s SAR-scanning
> block, `from_nisar_sme2`), `cli.py` (`--sar-source`,
> `_build_soil_moisture_config`), `downloaders/base.py`
> (`authenticate_earthdata`), `downloaders/earthdata_soil_moisture_downloader.py`
> (multi-candidate `EarthdataSoilMoistureDownloader`), `core/statistics.py`
> (`_harmonize_percent_domain_sources`, `fit_sar_to_val_transform`),
> `core/visualization.py` (`plot_collocation_diagnostics`,
> `_diagnostics_zoom_extent`).

### 8.12 AU_Land AMSR2: real format confirmed, NPD chosen over SCA

`_from_amsr_ssm_au_land_points` (formerly `_from_amsr_ssm_au_land_swath`)
originally guessed its field layout from NSIDC's user guide, with no real
granule available to check against (§ its own docstring said so
explicitly). Once NASA Earthdata credentials were available, a live
download (`AMSR_U2_L2_Land_B02_202312312326_D.he5`) showed the guess was
wrong in three independent ways, all of which caused `from_amsr_ssm` to
silently drop every AU_Land file with a "Missing vsm/longitude/latitude
field(s)" warning:

1. **Group type**: the real file uses HDF-EOS5's `POINTS` structure
   (`HDFEOS/POINTS/AMSR-2 Level 2 Land Data/...`), not `SWATHS` — despite
   the product's own name describing it as a "half-orbit swath". The
   detection check (`"HDFEOS/SWATHS" in f`) never matched, so the format
   silently fell through to the unrelated NSIDC-0451 branch.
2. **Field storage**: fields are not separate named datasets
   (`Data Fields/Soil_Moisture` etc.) — they're columns of one compound
   (structured) dataset, `Data/Combined NPD and SCA Output Fields`.
3. **Time epoch**: the `Time` field is seconds since **1993-01-01**
   (TAI93 — common across NASA/JAXA AMSR products), not the Unix epoch.
   Confirmed numerically: a real granule's `Time` values only land on its
   own filename-embedded acquisition timestamp under the 1993 epoch: e.g.
   `978221822` -> `2024-01-01T00:17:02`, matching a granule named
   `..._202312312326_...` (swath started 2023-12-31 23:26, continuing a
   few minutes past midnight).

**NPD vs. SCA**: the real product carries two independent, co-equal
soil-moisture retrievals per point — `SoilMoistureNPD` (Normalized
Polarization Difference) and `SoilMoistureSCA` (Single Channel
Algorithm) — with no "primary" one stated anywhere (not in the field
metadata, not in NSIDC's own collection abstract: "estimated ... using
two different approaches"). NPD was chosen because it matches the
algorithm NSIDC-0451 — this product's direct predecessor, whose
coverage AU_Land extends — used exclusively; SCA is never used as a
fallback (a fill-value NPD row is dropped even when SCA has a real
value for that same point, rather than silently mixing two differently-
biased retrieval algorithms within one dataset).

Fill value (`-9999.0`) and its NSIDC-0451-matching drop convention are
unchanged. Live-verified end to end: 19,004 of 31,781 points in the
sample granule survived filtering, with soil-moisture values in the
expected 0-0.5 m³ m⁻³ range and timestamps landing within the granule's
own acquisition window.

> Code: `core/datatree_converter.py` (`from_amsr_ssm`,
> `_from_amsr_ssm_au_land_points`), `tests/test_datatree_converter.py`
> (`TestFromAmsrSsmAuLandFormat`).

## 9. Visualization / report choices

### 9.1 Adaptive geographic marker sizing

The geographic plot overlays validation points on the SAR background field
at a fixed marker size (`s=`); too large and dense validation coverage
tiles edge-to-edge and hides the SAR field underneath it entirely, too
small and sparse in-situ points (e.g. a handful of ISMN stations) become
hard to see at all. One fixed size per variable can't serve both cases.

**Wind and soil moisture: density-based, per `collocation_type`.** For each
`collocation_type` present (`point_vs_layer`, `layer_vs_layer`), `avg =
(collocation_type == ctype).sum() / len(matched_scenes)`; `>~300` uses a
smaller marker (`5`) for that type, otherwise `15`. `plot_geographic`'s
`point_size` accordingly accepts either a plain `int` (uniform) or a
`dict[collocation_type, int]` (per-type). Originally wind-only, pooled
across the whole pair rather than per-type (adaptive scatterometer
sizing); soil moisture reused the check rather than its own flat
`point_size=10`, but pooling one average across both types let ASCAT/
SMAP/SMOS's thousands of `layer_vs_layer` points dominate the average and
made ISMN's sparse `point_vs_layer` points too small too — split per-type
once a mixed-density soil-moisture fixture caught it.

**Currents: always `15`.** HF radar (currents' only layer-type source)
forms a near-continuous coverage grid that tiles edge-to-edge at the
default size regardless of density, so it doesn't need — and doesn't
benefit from — the adaptive check.

**Everything else: default `40`.**

`pair_ds.sizes["collocation"]`, not `len(pair_ds)`: an `xr.Dataset`'s
`len()` counts its *data variables* (`sar_<var>`, `val_<var>`,
`val_source`, …), not its collocation-row count — using it here silently
undercounted every real dataset (typically 5-10 variables vs. hundreds to
thousands of rows), so the `>300` branch could never actually fire; fixed
alongside soil moisture's adoption of this check once a realistic-scale
test caught it.

> Code: `core/visualization.py` (`validation_report`'s `geo_point_size`
> block, `plot_geographic`'s `_resolve_point_size`).

### 9.2 Scatter plots split into per-source small multiples when imbalanced

A single shared scatter axes with one source holding the vast majority of
points (e.g. ASCAT's thousands vs. SMOS's dozens) visually buries every
other source under the dominant one's overplotting.

**`plot_scatter` renders one subplot per `val_source` instead** whenever a
single source holds `>70%` of the deduplicated points *and* at least 2
sources are present (`split_when_imbalanced`, default on) — matching
`plot_residuals`' existing by-source small-multiples layout — or whenever
the caller explicitly requests it regardless of share via `force_split`
(soil moisture's harmonized-domain case, §8.10). Each subplot keeps its own
1:1 reference line, scaled to that source's own data range, and — when
`color_by="temporal_offset"` — its own per-point coloring, sharing one
color scale/colorbar across every subplot.

> Code: `core/visualization.py` (`plot_scatter`, `_plot_scatter_small_multiples`).

### 9.3 Report page order: geographic before scatter

Every pair's plots used to lead with the scatter comparison, then the
geographic overview. Reordered so geographic comes first — a reader sees
*where* the matches are and how dense they are before the more abstract
point-cloud comparison, which is easier to interpret once that spatial
context is established. Applies to every recipe type (the loop generating
these sections is shared across wind/currents/waves/soil_moisture), not
just soil moisture — and to soil moisture's separate native-units section
(§8.7) too, which follows the same geographic-then-scatter-then-residuals
order as the CDF-matched section above it.

> Code: `core/visualization.py` (`validation_report`'s per-pair loop and
> native-units block).

## 10. RADARSAT-2 (NOAA NCEI): a second SAR source, selected per recipe

A second SAR-side source for `wind` recipes, alongside Sentinel-1
L2_OCN, following the exact registry pattern §8.11 established for
NISAR SME2/soil_moisture: `sar_sources.SAR_SOURCES["radarsat2"]`,
`variables=frozenset({"wind"})`.

**Two filename eras in NOAA's THREDDS archive, no hardcoded cutoff
date.** Live-confirmed: pre-2024 catalogs use
`RSAT2_{PROVIDER}_{YYYY}_{MM}_{DD}_{HH}_{MM}_{SS}_{seq}_{lon}{E|W}_{lat}{N|S}_{POL}_C5_{MODEL}_wind_level2_norcs.nc`;
2024-onward catalogs use
`SAR-Wind-{POL}-{lat}{N|S}-{lon}{E|W}_v{maj}r{min}_rsat2_s{start}_e{end}_c{created}.nc`.
`radarsat2_wind_downloader._parse_granule_name` matches whichever regex a
filename fits, rather than picking by date — this codebase already has
one example of a hardcoded-transition-date guess turning out wrong
(NISAR SME2's `AU_Land_NRT_R02`/CMR-collection cutover, §8.11).

**Catalog XML has no spatial search, but THREDDS' NCML metadata service
gives an exact per-granule bbox for free — a two-stage filter, coarse
then precise, with no full-scene download wasted on a non-overlapping
candidate.** THREDDS' `catalog.xml` exposes only `name`/`urlPath`/
`dataSize` per granule, no bbox. Both filename eras embed a scene-center
lon/lat (decimal precision pre-2024, integer degrees from 2024);
`_list_radarsat2_granules` uses it as a coarse, purely-local pre-filter,
keeping any candidate whose center falls within the requested bbox
padded by 5°. Every THREDDS granule also has a lightweight (~25-30KB,
confirmed live, zero data values) `/ncml/{urlPath}` metadata endpoint.
`RADARSAT2WindDownloader._passes_ncml_check` fetches it for each
surviving candidate and parses its real `geospatial_lat/lon_min/max`
(`_parse_ncml_bbox` — new era: root-level global attributes, the file's
own stated values; old era: nested under `<group name="CFMetadata">`,
auto-computed server-side by THREDDS since the raw old-era file has no
such attributes at all) before deciding whether to issue the actual
~38MB `fileServer` download. A candidate whose real footprint doesn't
overlap the requested bbox is never downloaded. If the NCML fetch or
parse fails for any reason, the check fails *open* (treats the
candidate as passing) rather than silently dropping a possibly-real
granule — the cost of a false positive is one extra download, not a
missed scene.

**Land/ice/quality masking rule, empirically derived from a real
downloaded granule** (`SAR-Wind-HH-64N-174E_v3r0_rsat2_...`,
2026-06-04) — and corrected once, mid-design, after re-verifying against
that same file. Cross-tabulating the file's `pixel_level_quality_flags`
against its `mask` (`-1`=water/`0`=shore/`1`=land) and `icemask`
(`1`=water/`2`=land/`3`=sea_ice/`4`=snow, `0`=no-data):

| flag | meaning | n pixels | mostly |
|---|---|---|---|
| 5 | valid wind, valid water | 63,810 | 100% water (both mask and icemask) |
| 4 | valid wind, buffer region | 977,221 (92% of scene) | 90% **land** |
| 1 | invalid wind, buffer region | 19,836 | 85% land |
| 0 | invalid wind, valid water | 1,518 | 100% water |

Every flag-`5` pixel is `mask == -1 AND icemask == 1` — but **the
reverse is not true**. Filtering by `mask == -1 AND icemask == 1` alone
(an earlier design draft's assumption, before this correction) actually
keeps 115,267 pixels — 51,457 more than flag `5`. The extra pixels are
genuinely water per `mask`/`icemask`, but the retrieval algorithm itself
flagged them unreliable for other reasons: flag `0`'s "water" pixels all
have `sar_wind == 0.0` exactly (fill-like, not real calm wind), and flag
`4`'s "buffer region" pixels — despite being labeled "valid" — are
lower-confidence retrievals near a coast/ice edge, and 90% of *all*
flag-`4` pixels (not just the water/water subset) are land. So
`mask`/`icemask` are not a substitute for the quality flag when it's
available.

`pixel_level_quality_flags` does not exist in the pre-2024 filename era
(confirmed live against a 2019 granule). `DataTreeConverter.from_radarsat2_wind`
therefore uses `pixel_level_quality_flags == 5` directly when present,
and falls back to `mask == -1 AND icemask == 1` only for the old era —
a documented, era-specific approximation known to be slightly more
permissive than the new era's flag-based precision, not claimed
equivalent to it. This still mirrors the existing ASCAT precedent
(`_ASCAT_REJECT_FLAGS`) of explicit, documented land/ice/quality
rejection rather than trusting a product's raw fill values alone.

**Wind speed only — no `owiWindDirection`.** The product's `input_dir`
field is "interpolated directions used for wind inversion" — the NWP
model direction fed into the CMOD retrieval to resolve its 180°
ambiguity, not an independently SAR-measured direction. Producing it as
`owiWindDirection` would silently compare SAR-derived speed against a
model's own direction, masquerading as a SAR quantity.
`_variable_map.filter_variable_pairs` already filters
`VARIABLE_PAIRS["wind"]` down to pairs where both the `sar_<var>` and
`val_<var>` columns exist in the collocation dataset — simply omitting
`owiWindDirection` from the converter's output is sufficient by itself;
no other code changes anywhere in the pipeline.

> Code: `downloaders/radarsat2_wind_downloader.py`
> (`RADARSAT2WindDownloader`, `_parse_granule_name`,
> `_list_radarsat2_granules`, `_parse_ncml_bbox`,
> `RADARSAT2WindDownloader._passes_ncml_check`), `downloaders/base.py`
> (`months_touched`, shared with the NOAA HF-radar THREDDS downloader),
> `core/datatree_converter.py` (`from_radarsat2_wind`),
> `core/sar_sources.py` (`SAR_SOURCES["radarsat2"]`), `cli.py`
> (`_build_wind_config`'s `radarsat2` description branch).
