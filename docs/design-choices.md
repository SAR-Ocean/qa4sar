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

**Why:** US recipes therefore use a `hf_radar_us` source that automatically
prefers NOAA (denser real coverage) whenever the request's region and date
fall inside NOAA's ~90-day rolling ERDDAP window, falling back to
Copernicus (NRT + delayed-mode, historical-first to avoid double-counting
the same stations twice) only when NOAA can't serve the request — an older
date, or a non-US region. NOAA's own THREDDS/OPeNDAP archive, which would
extend its coverage to older dates directly, isn't implemented yet. This
supersedes an earlier, same-day decision to retire NOAA everywhere in favor
of Copernicus alone (the network-identity argument was correct, but ignored
Copernicus's incomplete re-ingestion of it) — see `docs/superpowers/specs/`
for that reasoning's full history. Separately, and independent of which
backend is used, all Copernicus HF-radar data (US or not) now drops cells
where the overall `QCflag == 4` ("bad") — Copernicus ships them unfiltered,
and this toolbox previously retained them uncritically; per-parameter flags
(`CSPD_QC` etc.) remain retained but unused.

> Code: `downloaders/hf_radar_us_downloader.py` (`resolve_hf_radar_us_backend`,
> `HFRadarUSDownloader`), `core/orchestrator.py` (`_download_hf_radar_us`),
> `core/datatree_converter.py` (`from_hf_radar_grid`'s `QCflag` filter).

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
