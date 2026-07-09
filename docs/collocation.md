# Collocation

Collocation is step 3 of the validation pipeline.  It matches SAR L2 OCN
grid cells to validation observations (in-situ measurements, moving-platform
tracks, or a second satellite swath) and writes the paired values to
`collocation_results.nc`.

---

## Collocation types

Two types are supported, each targeting a different kind of validation
source:

| Type | When used | Typical sources |
|---|---|---|
| `point_vs_layer` | Fixed or moving observation vs SAR grid | Mooring, buoy, tidal gauge, HF radar, ferrybox, drifting buoy |
| `layer_vs_layer` | Gridded satellite swath vs SAR grid | ASCAT scatterometer (OSI-SAF), altimeter |

The type is **auto-detected per source** from the `platform_type` and
`data_type` attributes stored in the DataTree.  You do not need to set it
explicitly — the converter writes the right attributes when it builds
`datatree.nc`.

---

## Algorithm

Both types use the same matching algorithm.  For each validation
observation (whether a single point or one cell of a swath grid) and each
SAR scene:

1. **Spatial filter** — compute the great-circle distance (Haversine) from the
   observation's `(lon, lat)` to every SAR grid cell.  Keep cells within
   `spatial_tolerance_km`.
2. **Temporal filter** — compute the absolute time difference between the
   observation's timestamp and the SAR acquisition time.  Keep pairs within
   `time_tolerance_minutes`.
3. **Value extraction** — for each accepted SAR cell, read all `(y, x)`
   variables at that pixel.  Pixels where **all** SAR variables are NaN are
   skipped.
4. **Record** — store a `CollocatedPoint` containing the SAR location, time,
   and values; the validation location, time, and values; the spatial and
   temporal distances; and a `collocation_type` label.

For `layer_vs_layer`, the gridded validation product (e.g. an OSI-SAF
scatterometer swath) is first flattened to individual `(lon, lat, time, …)`
observations, then the same per-observation loop is applied.

> **Interpolation** — only `nearest` (closest grid cell) is currently
> implemented.  `linear` and `cubic` are planned.

---

## Running collocation

```bash
# Download + build DataTree + run collocation in one step
sar-validate --recipe recipes/test.yaml --collocate

# Run only collocation on an already-downloaded DataTree
sar-validate --recipe recipes/test.yaml --collocate --no-download
```

---

## Recipe configuration

### Paper-standard settings (hal-04202202)

The recommended collocation settings from **Abderrahim et al. (2019)**
[hal-04202202](https://hal.science/hal-04202202/document) are:

- **Point vs. Layer** (moorings, buoys, tidal gauges):
  - Temporal offset: **±30 minutes**
  - Spatial offset: **12.5 km**

- **Layer vs. Layer** (scatterometer, OSI-SAF):
  - Temporal offset: **±3 hours (180 minutes)**
  - Spatial offset: **12.5 km**

The default recipe template now uses these values. For an example, see
[examples/paper-standard-wind-validation.yaml](../examples/paper-standard-wind-validation.yaml).

### Configuration

```yaml
collocation:
  type: point_vs_layer          # kept for backward compat; auto-detected per source
  time_tolerance_minutes: 30    # ← paper standard for point-vs-layer
  spatial_tolerance_km: 12.5    # ← paper standard
  interpolation_method: nearest # only "nearest" is implemented

validation_sources:
  - source_type: mooring        # → point_vs_layer  (auto-detected)
    collocation_kwargs: {}      # use global settings

  - source_type: ferrybox       # → point_vs_layer  (auto-detected)
    collocation_kwargs:
      time_tolerance_minutes: 30  # example: same as point settings

  - source_type: scatterometer  # → layer_vs_layer  (auto-detected)
    collocation_kwargs:
      time_tolerance_minutes: 180  # ← paper standard for layer-vs-layer
      spatial_tolerance_km: 12.5   # ← paper standard
```

**Per-source `collocation_kwargs` override the global tolerances for that source.**
This allows different tolerances for different validation platforms (e.g., tighter
tolerances for scatterometer vs. in-situ buoys)

---

## Inspecting `collocation_results.nc`

`collocation_results.nc` is a **flat Dataset** (one record per matched pair),
**not** a hierarchical DataTree.  Open it with `xr.open_dataset()`:

```python
import xarray as xr
import pandas as pd

ds = xr.open_dataset("data/.../collocation_results.nc")

# Overview
print(ds)
# Dimensions: collocation: N
# Variables: sar_lon, sar_lat, val_lon, val_lat,
#            spatial_distance_km, temporal_distance_minutes,
#            collocation_type, val_source,
#            sar_<variable>, val_<variable>, …
# Coordinates: time (SAR acq. time), val_time, val_id

# List available variables
print(list(ds.data_vars))

# Convert to a pandas DataFrame for easy analysis
df = ds.to_dataframe()
print(df.head())
```

### Filter by collocation type

```python
# Only point-vs-layer matches (moorings, buoys)
point_df = df[df["collocation_type"] == "point_vs_layer"]

# Only layer-vs-layer matches (scatterometer)
layer_df = df[df["collocation_type"] == "layer_vs_layer"]
```

### Filter by validation source

```python
# Only mooring matches
mooring_df = df[df["val_source"] == "mooring"]
```

### Scatter plot (wind speed example)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, ctype, label in zip(
    axes,
    ["point_vs_layer", "layer_vs_layer"],
    ["Point (mooring/buoy/ferrybox/drifter)", "Layer (scatterometer)"],
):
    sub = df[df["collocation_type"] == ctype].dropna(
        subset=["sar_owiWindSpeed", "val_WSPD"]
    )
    if sub.empty:
        ax.set_title(f"{label}\n(no data)")
        continue
    ax.scatter(sub["val_WSPD"], sub["sar_owiWindSpeed"], alpha=0.5, s=10)
    lims = [0, max(sub[["val_WSPD", "sar_owiWindSpeed"]].max())]
    ax.plot(lims, lims, "r--", lw=1, label="1:1")
    ax.set_xlabel("In-situ wind speed (m/s)")
    ax.set_ylabel("SAR wind speed (m/s)")
    ax.set_title(label)
    ax.legend()

plt.tight_layout()
plt.show()
```

### Quality metrics

```python
print("Median spatial distance (km):", df["spatial_distance_km"].median())
print("Median temporal distance (min):", df["temporal_distance_minutes"].median())
print("Matches per type:\n", df["collocation_type"].value_counts())
print("Matches per source:\n", df["val_source"].value_counts())
```

---

## Output variables

Every row of `collocation_results.nc` contains:

| Variable | Description |
|---|---|
| `time` *(coord)* | SAR acquisition time |
| `val_time` *(coord)* | Validation observation time |
| `val_id` *(coord)* | Platform identifier (if available) |
| `sar_lon`, `sar_lat` | SAR grid cell centre coordinates |
| `val_lon`, `val_lat` | Validation observation coordinates |
| `spatial_distance_km` | Great-circle distance between the pair |
| `temporal_distance_minutes` | Absolute time difference |
| `collocation_type` | `point_vs_layer` or `layer_vs_layer` |
| `val_source` | Source label (e.g. `mooring`, `ferrybox`, `scatterometer`) |
| `sar_<variable>` | SAR value at the matched pixel (e.g. `sar_owiWindSpeed`) |
| `val_<variable>` | Validation value (e.g. `val_WSPD`, `val_wind_speed`) |
