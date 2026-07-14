# Running Statistics and Plots from the CLI

## Pipeline overview

`sar-validate` covers all six validation steps.  The flags are cumulative:
each flag implies all earlier steps.

```
Step 1  (always)      Download SAR + validation data
Step 2  --convert     Build datatree.nc
Step 3  --collocate   Build collocation_results.nc          (implies --convert)
Step 5a --stats       Compute bias / RMSE / correlation      (implies --collocate)
Step 5b --plot        Generate plots + PDF report            (implies --stats)
```

---

## Quick Reference

```bash
# Full pipeline from scratch (download → collocate → stats → plots)
sar-validate --recipe recipes/wind_validation.yaml --plot

# Re-run stats + plots only (data directory already exists)
sar-validate --recipe recipes/wind_validation.yaml --stats --plot

# Statistics only (no plots)
sar-validate --recipe recipes/wind_validation.yaml --stats

# Steps 1–3 only (collocation, inspect data before generating plots)
sar-validate --recipe recipes/wind_validation.yaml --collocate

# Enable verbose logging
sar-validate --recipe recipes/wind_validation.yaml --plot --verbose
```

---

## Output files

All outputs are written to the **data directory** derived from the recipe's
geographic and temporal bounds.  For example:

```
data/2026-03-01-000000-2026-03-02-000000_-10.00_5.00_50.00_65.00/
│
│  ── Step 3 ──
├── collocation_results.nc
│
│  ── Step 5a ──
├── validation_statistics_owiWindSpeed_vs_WSPD.nc
├── validation_statistics_owiWindSpeed_vs_WSPD.csv
├── validation_statistics_owiWindDirection_vs_WDIR.nc
├── validation_statistics_owiWindDirection_vs_WDIR.csv
│
│  ── Step 5b ──
├── validation_report.pdf          ← combined PDF (all plots, one file)
└── plots/
    ├── owiWindSpeed_vs_WSPD_scatter.png
    ├── owiWindSpeed_vs_WSPD_geographic_point_vs_layer.png
    ├── owiWindSpeed_vs_WSPD_geographic_layer_vs_layer.png
    ├── owiWindSpeed_vs_WSPD_statistics.png
    ├── owiWindSpeed_vs_WSPD_residuals.png
    ├── owiWindDirection_vs_WDIR_scatter.png
    └── ...
```

### Statistics files (`.nc` + `.csv`)

Each `validation_statistics_<sar_var>_vs_<val_var>.nc` contains an xarray
Dataset with dimension `source` (one entry per buoy station or scatterometer
swath) and data variables:

| Variable | Description |
|----------|-------------|
| `N` | Number of valid collocated pairs |
| `bias` | Mean(SAR − in-situ) |
| `std` | Standard deviation of residuals |
| `rmse` | Root-mean-square error |
| `correlation` | Pearson r |
| `scatter_index` | RMSE / \|mean(in-situ)\| |

The companion `.csv` is the same table in spreadsheet-friendly form.

### PDF report (`validation_report.pdf`)

A multi-page PDF saved directly in the data directory alongside the `.nc`/`.csv`
statistics files.  Each page corresponds to one plot type for one variable pair:

1. **Cover page** — recipe name, variable, generation date
2. **Scatter plot** — SAR vs. validation (one point per observation; SAR values
   are averaged over the matched pixel neighbourhood)
3. **Geographic — in-situ** (`point_vs_layer`) — SAR field as background,
   buoy/mooring positions coloured by measured wind speed
4. **Geographic — scatterometer** (`layer_vs_layer`) — same layout for ASCAT
5. **Statistics bar chart** — per-station bias, RMSE, and correlation
6. **Residuals histogram** — distribution of (SAR − validation) by source

---

## Flags Summary

| Flag | Implies | What it does |
|------|---------|--------------|
| *(none)* | — | Download SAR + validation data (step 1) |
| `--convert` | — | Convert downloaded files to DataTree (step 2) |
| `--collocate` | `--convert` | Spatiotemporal collocation (step 3) |
| `--stats` | `--collocate` | Bias, RMSE, correlation per station (step 5a) |
| `--plot` | `--stats` | Scatter, geographic, statistics, residuals + PDF (step 5b) |
| `--dry-run` | — | Preview what would be downloaded; no files written |
| `--output-dir DIR` | — | Override the output directory from the recipe |
| `--verbose` / `-v` | — | Enable DEBUG-level logging |

---

## Re-running After Configuration Changes

If you change the recipe's collocation tolerances, delete
`collocation_results.nc` and rerun with `--collocate --plot`:

```bash
rm data/<data-dir>/collocation_results.nc
sar-validate --recipe recipes/wind_validation.yaml --collocate --plot
```

To force a full re-download (e.g. extended time window), remove the entire
data directory:

```bash
rm -rf data/<data-dir>
sar-validate --recipe recipes/wind_validation.yaml --plot
```

---

## Python API

The same workflow is available directly from Python:

```python
import xarray as xr
from pathlib import Path
from sar_validation.core.recipe import Recipe
from sar_validation.core.statistics import run_statistics
from sar_validation.core.visualization import validation_report

DATA_DIR = Path(
    "data/2026-03-01-000000-2026-03-02-000000_-10.00_5.00_50.00_65.00"
)
recipe = Recipe.from_yaml("recipes/wind_validation.yaml")

collocation_ds = xr.open_dataset(DATA_DIR / "collocation_results.nc")
datatree = xr.open_datatree(str(DATA_DIR / "datatree.nc"), engine="netcdf4")

# Step 5a — statistics saved as .nc + .csv in DATA_DIR
stats_map = run_statistics(collocation_ds, recipe, DATA_DIR)

# Step 5b — PNGs saved to DATA_DIR/plots/, PDF to DATA_DIR/validation_report.pdf
validation_report(
    collocation_ds=collocation_ds,
    datatree=datatree,
    recipe=recipe,
    stats_ds_map=stats_map,
    out_dir=DATA_DIR,
)
```
