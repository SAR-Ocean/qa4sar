# Steps 4 & 5 — Statistics and Visualization

Steps 4 and 5 extend the validation pipeline beyond collocation (step 3) to cover
per-source statistics and a full suite of plots.

---

## Pipeline overview

```
Step 1  Download          → S1_L2_OCN/ + copernicus_insitu/ + osi_saf_winds/
Step 2  Convert           → datatree.nc
Step 3  Collocation       → collocation_results.nc
Step 4  Statistics        → validation_statistics_*.nc/.csv  (new)
Step 5  Visualization     → plots/*.png  + interactive objects  (new)
```

---

## Step 4 — Validation Statistics

### What it does

Computes per-source bias, RMSE, Pearson correlation, standard deviation, and
scatter index for each (SAR variable, validation variable) pair inferred from the
recipe.

### Output files

```
<data_dir>/validation_statistics_<sar_var>_vs_<val_var>.nc
<data_dir>/validation_statistics_<sar_var>_vs_<val_var>.csv
```

Example for a wind recipe:

```
validation_statistics_owiWindSpeed_vs_WSPD.nc/.csv
validation_statistics_owiWindDirection_vs_WDIR.nc/.csv
```

### Metrics

| Metric | Definition |
|--------|-----------|
| `N` | Number of valid (non-NaN) pairs in the group |
| `bias` | mean(SAR − val) |
| `std` | std(SAR − val, ddof=1) |
| `rmse` | √ mean((SAR − val)²) |
| `correlation` | Pearson r |
| `scatter_index` | RMSE / \|mean(val)\| |

### Variable pairs

Pairs are looked up from `sar_validation/core/_variable_map.py`:

| Recipe variable | SAR var | Validation var |
|----------------|---------|----------------|
| `wind` | `owiWindSpeed` | `WSPD` |
| `wind` | `owiWindDirection` | `WDIR` |
| `currents` | `owiEastwardCurrent` | `EWCT` |
| `currents` | `owiNorthwardCurrent` | `NSCT` |
| `waves` | `owiSignificantWaveHeight` | `VHM0` |

To add a new validated quantity, edit `VARIABLE_PAIRS` in `_variable_map.py`.

### API

```python
from sar_validation.core.statistics import (
    compute_statistics,   # -> xr.Dataset
    save_statistics,      # writes .nc + .csv
    run_statistics,       # top-level: infers pairs from recipe, saves all
)
from sar_validation.core._variable_map import infer_variable_pairs
```

### CLI

```bash
sar-validate --recipe recipes/wind_validation.yaml --collocate --stats
```

`--stats` implies `--collocate` which implies `--convert`.

---

## Step 5 — Visualization

### Functions

All functions live in `sar_validation/core/visualization.py` and accept an
`interactive=False` keyword argument.

#### `plot_scatter(collocation_ds, sar_var, val_var, *, by_source, interactive, ax)`

Scatter plot of SAR vs. validation variable with:
- Points coloured by `val_source`
- 1:1 reference line
- Annotation box: N, bias, RMSE, r
- Static: `matplotlib.figure.Figure`
- Interactive: `plotly.graph_objects.Figure` (requires `plotly`)

#### `plot_geographic(datatree, collocation_ds, sar_var, *, ncols, cmap, interactive)`

**One subplot per SAR scene.**  For each scene:
- SAR field (`sar_var`) rendered as `pcolormesh` with a cartopy `PlateCarree`
  projection and coastlines
- Collocated points from `collocation_results.nc` filtered to that scene and
  overlaid as a scatter, coloured by `val_source`
- Layout: `nrows = ceil(n_scenes / ncols)`, `ncols` configurable (default 2)
- Static: `matplotlib.figure.Figure`
- Interactive: `folium.Map` with one `FeatureGroup` per scene (requires `folium`)

#### `plot_statistics(stats_ds, metrics, *, interactive)`

Grouped bar chart of the requested metrics per `val_source`:
- Static: `matplotlib.figure.Figure`
- Interactive: `plotly.graph_objects.Figure`

#### `plot_residuals(collocation_ds, sar_var, val_var, *, by_source, interactive, ax)`

Histogram / density of (SAR − val) residuals:
- Static: `matplotlib.figure.Figure`
- Interactive: `plotly.graph_objects.Figure`

#### `validation_report(collocation_ds, datatree, recipe, stats_ds_map, out_dir)`

Convenience wrapper — runs all four functions for every variable pair inferred
from `recipe.config.variable` and saves PNG files to `<out_dir>/plots/`.

### Output files

```
<data_dir>/plots/
  owiWindSpeed_vs_WSPD_scatter.png
  owiWindSpeed_vs_WSPD_geographic.png
  owiWindSpeed_vs_WSPD_statistics.png
  owiWindSpeed_vs_WSPD_residuals.png
  owiWindDirection_vs_WDIR_scatter.png
  …
```

### Optional dependencies

Interactive backends are optional.  If not installed, calling with
`interactive=True` raises an `ImportError` with an installation hint.

| Backend | Used by | Install |
|---------|---------|---------|
| `plotly` | `plot_scatter`, `plot_statistics`, `plot_residuals` | `pip install plotly` |
| `folium` | `plot_geographic` | `pip install folium` |

Install all at once:

```bash
pip install 'sar-l2-validation-toolbox[plot]'
# or via conda:
conda install -c conda-forge plotly folium
```

### CLI

```bash
# Scatter + geographic + statistics + residuals after collocation
sar-validate --recipe recipes/wind_validation.yaml --collocate --stats --plot
```

`--plot` implies `--stats` which implies `--collocate` which implies `--convert`.

---

## New files

| File | Purpose |
|------|---------|
| `sar_validation/core/_variable_map.py` | `VARIABLE_PAIRS` dict + `infer_variable_pairs()` |
| `sar_validation/core/statistics.py` | `compute_statistics`, `save_statistics`, `run_statistics` |
| `sar_validation/core/visualization.py` | `plot_scatter`, `plot_geographic`, `plot_statistics`, `plot_residuals`, `validation_report` |
| `examples/validation_notebook.ipynb` | End-to-end notebook demo |
| `tests/test_statistics.py` | Unit tests for step 4 |
| `tests/test_visualization.py` | Smoke tests for all plot functions |

## Modified files

| File | Change |
|------|--------|
| `sar_validation/core/collocation.py` | `CollocatedPoint` gains `sar_y_idx`, `sar_x_idx`, `sar_scene_name`; `collocate()` gains `sar_scene_name` param |
| `sar_validation/core/datatree_converter.py` | `from_collocations()` persists the three new index fields |
| `sar_validation/core/__init__.py` | Exports new modules |
| `sar_validation/cli.py` | Adds `--stats`, `--plot` flags and `_compute_stats`, `_generate_plots` helpers |
| `pyproject.toml` | `plotly`, `folium` added to `[plot]` and `[dev]` extras |
| `environment.yml` | `plotly`, `folium` added |
