# Visualization & Statistics Improvements — 2026-06-30

## Background

After the first end-to-end run of the validation pipeline the plots produced by
steps 4b and 5 revealed three problems:

1. **Geographic plot — observation "blob"** — scatter points formed an
   indistinct cloud rather than distinct observation locations.
2. **Scatter plot — horizontal stripes** — each in-situ wind speed appeared as
   a horizontal band paired with many SAR values, making the 1:1 comparison
   unreadable.
3. **Statistics bar chart — single bar per metric** — only one statistical
   summary value was produced for all in-situ data combined.

All three problems share the same root cause: the `collocate()` algorithm
(step 3) finds **every SAR pixel within the spatial tolerance** of each
in-situ observation.  One buoy or mooring reading therefore produces *N* rows
in `collocation_results.nc` — one per matched SAR pixel — all sharing the
same validation-side values but carrying different SAR pixel positions and
values.

---

## Changes Applied

### `sar_validation/core/visualization.py`

#### New helper — `_deduplicate_obs()`

Collapses the many-SAR-pixel-per-observation rows to one row per observation
before plotting.  Groups by `(val_source, val_id, val_time, val_lat, val_lon)`
— whichever columns are present — and aggregates:

| Column | Aggregation |
|--------|-------------|
| `sar_<var>` | **mean** across all matched pixels |
| `val_<var>` | **first** — identical for all rows of the same observation |

#### `plot_scatter()` — deduplication applied

`_deduplicate_obs()` is called before building the scatter plot.  The figure
title now reports `(N=X obs, avg Y px/obs)` to make the aggregation
transparent.  The result is a clean 1:1 scatter instead of horizontal stripes.

#### `plot_geographic()` — three improvements

| Before | After |
|--------|-------|
| Points at SAR-pixel position (`sar_lat/sar_lon`) | Points at **observation position** (`val_lat/val_lon`) |
| All sources in one figure | **Separate figures per `collocation_type`** |
| *N* dots per observation | **One dot per observation** (deduplicated) |

New parameter `split_by: str = "collocation_type"` controls grouping.
The default splits collocations into:
- `point_vs_layer` — in-situ (mooring / buoy) observations
- `layer_vs_layer` — scatterometer (ASCAT) grid cells

The function now returns `dict[str, Figure]` when `split_by` is set, so
notebook loops must iterate the returned dict.  Pass `split_by=None` to revert
to a single combined figure.

### `sar_validation/core/statistics.py`

#### `run_statistics()` — per-station grouping

Previously always grouped by `val_source` (the dataset/file name).  For
in-situ data this yields a single aggregate statistic for the whole CSV file.

Now: if `val_id` (platform / station identifier stored in `collocation_results.nc`)
has more than one unique non-`"unknown"` value, statistics are grouped by
`val_id` instead.  This produces **one bias / RMSE / correlation per buoy or
mooring station**, which is the scientifically meaningful unit of comparison.

---

## Known Limitations

- **Deduplication uses the mean SAR value** across matched pixels.  Near sharp
  gradients (fronts, coastal zones) the mean may not represent the most
  relevant co-located value.
- **`val_id` grouping** requires `platform_id` to be present in the in-situ
  CSV and correctly propagated to `collocation_results.nc`.  When missing the
  fallback to `val_source` still yields only one group.
- **Scatterometer statistics** are grouped by swath file (`val_source`), giving
  one stat per ASCAT overpass.  Grouping by wind-speed bin or incidence-angle
  class would be more informative.
- The `_deduplicate_obs` proxy trick used in `plot_geographic` (inserting a
  `__sar_proxy__` column to drive the groupby) is a workaround; a dedicated
  observation-position deduplication path would be cleaner.

---

## Suggested Future Upgrades

### Nearest-pixel collocation mode

Add `reduce_to_nearest: bool = False` to `CollocationType` and `collocate()`.
When enabled store only the single SAR pixel with the smallest Haversine
distance, eliminating the many-to-one problem at source and making
`_deduplicate_obs` unnecessary.

### Per-station quality screening

Before computing statistics, filter collocated pairs where:

- `|temporal_distance_minutes| > threshold` (already in dataset)
- SAR wind speed is quality-flagged (OWI flag variables present in the L2 OCN product)
- In-situ observation fails a range or gross-error check

### Binned / conditional statistics

Stratify statistics by:

- SAR wind speed bins (0–5, 5–10, 10–15 m/s, …)
- Incidence angle class (relevant for scatterometer)
- Season or month

This is a standard step in satellite wind-product validation reports (e.g. the
ESA Sentinel-1 Marine User Handbook).

### Bootstrap confidence intervals

Wrap `compute_statistics()` with a bootstrap resampling loop to attach
uncertainty bounds to bias, RMSE, and correlation estimates.

### Taylor diagram

A Taylor diagram compactly encodes standard deviation, correlation, and
centred RMSE in one polar-coordinate plot and is the canonical summary figure
in multi-source wind validation.

### Improved scatterometer variable mapping

OSI-SAF ASCAT files use raw NetCDF names (`wind_speed`, `wind_dir`) that are
not mapped to the `VARIABLE_PAIRS` dict (`WSPD`, `WDIR`).  A normalisation
step in `from_scatterometer_nc()` would allow SAR vs. ASCAT statistics
out-of-the-box without manual recipe overrides.

### Interactive statistics dashboard

Replace static bar charts with a `plotly` or `panel` dashboard showing
per-station time series, scatter plots, and the statistics table side-by-side
— useful for operational validation workflows.

### Automated PDF validation report

Extend `validation_report()` to write a multi-page PDF
(`matplotlib.backends.backend_pdf.PdfPages`) containing all figures, a summary
statistics table, and full dataset provenance metadata (satellite pass,
temporal bounds, spatial bounds, number of collocated pairs).
