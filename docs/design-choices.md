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

A standalone Python package that validates Sentinel-1 **L2_OCN** (Level-2
Ocean) products against independent observations. One YAML **recipe**
(variable + region + time window + validation sources) drives the entire
pipeline:

| Step | What happens | Output |
|---|---|---|
| 1 — Download | SAR L2_OCN scenes + all validation sources for the recipe domain | `data/<run>/...` |
| 2 — Convert | Every source is standardised into one hierarchical `xarray.DataTree` | `datatree.nc` |
| 3 — Collocate | SAR cells are matched to validation observations within spatial/temporal tolerances | `collocation_results.nc` |
| 4 — Statistics | Bias, RMSE, correlation, scatter index per platform type | `validation_statistics_*.nc/.csv` |
| 5 — Report | Scatter/geographic/residual plots + PDF report | `plots/`, `validation_report.pdf` |

Three validated quantities are supported — **wind** (speed + direction,
OWI grids), **currents** (RVL radial velocity), **waves** (significant wave
height, OSW/WV imagettes) — against in-situ platforms (moorings, buoys,
drifters, ferryboxes, tidal gauges), ASCAT scatterometer, satellite
altimeters, RSS radiometers (AMSR2), and HF radar.

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
is detected from the file content (5 Hz files carry `VAVH_UNCERTAINTY`
instead of `WIND_SPEED`). Wind recipes download only 1 Hz (5 Hz has no wind
and 5× the point density for no benefit); wave recipes download both.

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
| WV mode (`point_vs_point` / `point_vs_layer`) | SAR imagette | imagette | sparse OSW point measurements |

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
| altimeter 1 Hz | 7.0 | 1 Hz along-track spacing |
| altimeter 5 Hz | 1.4 | 5 Hz along-track spacing |
| radiometer (AMSR2, …) | 25.0 | RSS 0.25° grid-cell size |
| HF radar | 5.0 | typical radial grid spacing |

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
- **HF radar: ±20 min** — surface currents decorrelate quickly; HF radar
  fields are typically 10–30 min composites.

### 5.4 Distance weighting

Weighting kernels available: `gaussian`, `inverse_distance`, `linear`,
`equal`, always renormalised over the non-NaN pixels.

- **In-situ default: Gaussian, σ = 2 km** — pixels near the point sensor
  should dominate the aggregate; the Gaussian gives a smooth, tunable
  falloff inside the 5 km window.
- **Satellite layers default: `equal`** — the anchor is itself an area
  average over a regular cell, so every SAR pixel inside that cell
  contributes the same.

### 5.5 The area around WV/OSW imagette points

Sentinel-1 WV mode produces isolated **~20 × 20 km imagettes ~200 km
apart**, each stored as a single point (e.g. `oswHs`). Requiring validation
data within a few km of the imagette *centre* (as the grid matchers would if
the point were faked into a 1×1 grid) would discard almost everything.

**Choice: anchor on the imagette and gather every validation observation
within `sar_footprint_radius_km` = 14 km** — approximately the
centre-to-corner half-diagonal of the 20 × 20 km footprint
(½·√(20² + 20²) ≈ 14.1 km), i.e. the radius that fully covers the footprint.
Observations inside the footprint are aggregated into one match per
imagette:

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
system, which is not the total sea state; in a real imagette the partitions were
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
