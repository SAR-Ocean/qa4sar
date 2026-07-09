# Running Steps 4 & 5 — Statistics and Visualization

Steps 4 and 5 consume the collocation results produced by step 3 and generate:

- **Step 4** — `validation_statistics_*.nc/.csv`: per-source bias, RMSE, and correlation
- **Step 5** — `plots/*.png`: scatter, geographic, statistics bar chart, and residual histogram

Prerequisites: steps 1–3 must have run and produced `datatree.nc` and
`collocation_results.nc` in the data directory.

---

## CLI quick reference

| Flag | Implies | What it does |
|------|---------|-------------|
| `--collocate` | `--convert` | Steps 1–3: download → convert → collocate |
| `--stats` | `--collocate` | + Step 4: compute and save statistics |
| `--plot` | `--stats` | + Step 5: generate and save all plots |

All flags are additive — `--plot` automatically runs everything before it.

---

## Step 4 — Validation statistics

```bash
sar-validate --recipe recipes/wind_validation.yaml --collocate --stats
```

Output files (one pair per validated component):

```
data/<run>/validation_statistics_owiWindSpeed_vs_WSPD.nc
data/<run>/validation_statistics_owiWindSpeed_vs_WSPD.csv
data/<run>/validation_statistics_owiWindDirection_vs_WDIR.nc
data/<run>/validation_statistics_owiWindDirection_vs_WDIR.csv
```

The CSV is suitable for pasting into a report; the NetCDF can be reloaded with
`xr.open_dataset()` for further analysis.

---

## Step 5 — Plots

```bash
sar-validate --recipe recipes/wind_validation.yaml --plot
```

Four plot types are generated for every validated variable pair and saved as PNGs:

```
data/<run>/plots/
  owiWindSpeed_vs_WSPD_scatter.png       # SAR vs. validation scatter + 1:1 line
  owiWindSpeed_vs_WSPD_geographic.png    # SAR field + collocated points (one subplot per scene)
  owiWindSpeed_vs_WSPD_statistics.png    # bias / RMSE / correlation bar chart per source
  owiWindSpeed_vs_WSPD_residuals.png     # histogram of (SAR − validation) residuals
  owiWindDirection_vs_WDIR_scatter.png
  ...
```

---

## Full pipeline — download through plots

```bash
# 1. Create a recipe
sar-validate --create-recipe wind \
  --min-lon -10 --max-lon 5 \
  --min-lat 50 --max-lat 65 \
  --start 2026-03-01 --end 2026-03-02

# 2. Run the complete pipeline (steps 1–5)
sar-validate --recipe recipes/wind_validation.yaml --plot
```

---

## Running from Python

### Step 4 — statistics

```python
import xarray as xr
from sar_validation.core.recipe import Recipe
from sar_validation.core.statistics import run_statistics

recipe         = Recipe.from_yaml("recipes/wind_validation.yaml")
collocation_ds = xr.open_dataset("data/<run>/collocation_results.nc")

# Infers variable pairs from recipe.config.variable automatically
stats_map = run_statistics(collocation_ds, recipe, base_dir="data/<run>")

# stats_map["owiWindSpeed_vs_WSPD"] -> xr.Dataset
print(stats_map["owiWindSpeed_vs_WSPD"].to_dataframe())
```

To compute statistics for a single pair explicitly:

```python
from sar_validation.core.statistics import compute_statistics

stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
# Metrics per val_source: N, bias, std, rmse, correlation, scatter_index
print(stats_ds.to_dataframe())
```

### Step 5 — generate all plots and save PNGs

```python
import xarray as xr
from sar_validation.core.recipe import Recipe
from sar_validation.core.statistics import run_statistics
from sar_validation.core.visualization import validation_report

recipe         = Recipe.from_yaml("recipes/wind_validation.yaml")
collocation_ds = xr.open_dataset("data/<run>/collocation_results.nc")
datatree       = xr.open_datatree("data/<run>/datatree.nc", engine="netcdf4")
stats_map      = run_statistics(collocation_ds, recipe, "data/<run>")

validation_report(
    collocation_ds,
    datatree,
    recipe,
    stats_ds_map=stats_map,
    out_dir="data/<run>",          # PNGs saved to data/<run>/plots/
)
```

### Individual plot functions

```python
import matplotlib.pyplot as plt
from sar_validation.core.visualization import (
    plot_scatter,
    plot_geographic,
    plot_statistics,
    plot_residuals,
)

# Scatter: SAR vs. validation, coloured by source, annotated with N/bias/RMSE/r
fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD")
plt.show()

# Geographic: SAR wind field as pcolormesh, one subplot per SAR scene
fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", ncols=2)
plt.show()

# Statistics bar chart
stats_ds = stats_map["owiWindSpeed_vs_WSPD"]
fig = plot_statistics(stats_ds, metrics=["bias", "rmse", "correlation"])
plt.show()

# Residuals histogram / KDE per source
fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD")
plt.show()
```

### Interactive plots

Install the optional backends first:

```bash
pip install plotly folium
# or: conda install -c conda-forge plotly folium
```

```python
# Interactive scatter (plotly, opens in browser or Jupyter widget)
fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", interactive=True)
fig.show()

# Interactive geographic map (folium, renders inline in Jupyter)
m = plot_geographic(datatree, collocation_ds, "owiWindSpeed", interactive=True)
display(m)
```

All functions fall back to a friendly `ImportError` if the required backend is
not installed.

---

## Variable pairs per recipe type

| Recipe `variable` | SAR variable | Validation variable |
|-------------------|-------------|---------------------|
| `wind` | `owiWindSpeed` | `WSPD` |
| `wind` | `owiWindDirection` | `WDIR` |
| `currents` | `owiEastwardCurrent` | `EWCT` |
| `currents` | `owiNorthwardCurrent` | `NSCT` |
| `waves` | `owiSignificantWaveHeight` | `VHM0` |

To add a new pair, edit `VARIABLE_PAIRS` in
`sar_validation/core/_variable_map.py`.

---

## Interactive notebook

A full end-to-end walkthrough covering all steps is in
[`examples/validation_notebook.ipynb`](../examples/validation_notebook.ipynb).
 