# SAR L2 Ocean Data Validation Toolbox

A standalone Python package for validating Sentinel-1 L2_OCN (Level 2 Ocean) products
against multi-source in-situ and satellite observations.

Supports validation of **wind** (speed + direction), **ocean currents**, and **significant wave height**.

---

## Architecture

The toolbox is structured around 5 sequential steps that together form a complete validation workflow:

```
Step 1 — Download data
          │  define variable (wind / currents / waves)
          │  define region (lon/lat) and time window
          │  download SAR L2_OCN + validation sources
          ▼
Step 2 — Convert to xarray.DataTree
          │  standardize all formats
          │  handle different grids / resolutions
          ▼
Step 3 — Collocation
          │  point vs. layer   (mooring / buoy / ferrybox / drifter vs. SAR)
          │  layer vs. layer   (scatterometer / HF-radar vs. SAR)
          ▼
Step 4 — Store collocated pairs
          │  keep only overlapping data
          │  store spatial/temporal offset metadata
          ▼
Step 5 — Visualisation & statistics
```

Steps 2–5 are dataset-agnostic: they work the same regardless of which validation sources were
downloaded in step 1.

---

## Package Layout

```
sar_validation/
├── cli.py                  # Command-line entry point
├── core/
│   ├── recipe.py           # Recipe dataclasses (YAML ↔ Python)
│   ├── orchestrator.py     # Orchestrates step 1 (download all sources)
│   ├── datatree_converter.py  # Step 2: convert to xarray.DataTree
│   ├── collocation.py      # Step 3: collocation algorithms
│   ├── statistics.py       # Step 4b: compute bias, RMSE, correlation, scatter index
│   ├── visualization.py    # Step 5: scatter plots, geographic maps, statistics charts, residuals
│   ├── _variable_map.py    # Variable mapping (wind/currents/waves)
│   └── __init__.py         # Package exports
└── downloaders/
    ├── base.py             # Shared credential handling & helpers
    ├── sar_downloader.py   # Sentinel-1 L2_OCN via Copernicus Dataspace
    ├── insitu_downloader.py   # Moorings/buoys/ferrybox via Copernicus Marine
    ├── hf_radar_downloader.py # HF radar via Copernicus Marine
    ├── scatterometer_downloader.py  # ASCAT (MetOp) via EUMETSAT EUMDAC
    ├── altimeter_downloader.py      # Along-track SWH/wind via Copernicus Marine
    └── radiometer_downloader.py     # RSS radiometer ocean winds (AMSR2) via public HTTPS
```

---

## Quick Start

### 1. Install

```bash
# Editable install with all extras
pip install -e ".[dev]"

# Or just core
pip install -e .
```

### 2. Create a recipe

```bash
sar-validate --create-recipe wind
sar-validate --create-recipe currents
sar-validate --create-recipe waves
```

This writes a YAML recipe to `recipes/<name>_validation.yaml`.  
Edit the file to adjust the geographic region, time window, and validation sources.

### 3. Dry-run (check what will be downloaded)

```bash
sar-validate --recipe recipes/wind_validation.yaml --dry-run
```

### 4. Execute (download all data)

```bash
sar-validate --recipe recipes/wind_validation.yaml
```

Downloads are saved under `data/<timerange>_<bounds>/`.

### 5. Convert, collocate, and analyse (Python)

```python
from sar_validation.core.datatree_converter import DataTreeConverter
from sar_validation.core.collocation import PointLayerCollocation
import xarray as xr
import pandas as pd

# --- Step 2: convert ---
converter = DataTreeConverter()
ds_sar    = converter.from_sar_l2_ocn(xr.open_dataset("data/.../S1_L2_OCN/product.nc"))
ds_insitu = converter.from_insitu_csv("data/.../copernicus_insitu_data/obs.csv", source_type="buoy")

# --- Step 3: collocate ---
colloc = PointLayerCollocation(spatial_tolerance_km=50, time_tolerance_minutes=60)
results = colloc.collocate(
    sar_data={"wind_speed": sar_ws_array},
    sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
    val_data=df_insitu,
    val_source="buoy",
)

# --- Step 4: store as DataFrame ---
df = DataTreeConverter.to_dataframe(results)
df.to_csv("data/.../collocations_wind_buoy.csv", index=False)

# --- Step 5: analyse ---
import matplotlib.pyplot as plt
plt.scatter(df["val_wind_speed"], df["sar_wind_speed"])
plt.xlabel("Buoy wind speed (m/s)")
plt.ylabel("SAR wind speed (m/s)")
plt.show()
```

---

## Supported Data Sources

| Source | Variable | Downloader | Service |
|--------|----------|------------|---------|
| Sentinel-1 L2_OCN | wind / currents / waves | `sar_downloader` | Copernicus Dataspace (CDSE) |
| Moorings / Buoys / Ferryboxes | wind / currents / waves | `insitu_downloader` | Copernicus Marine |
| HF Radar | ocean currents | `hf_radar_downloader` | Copernicus Marine |
| ASCAT (MetOp-B/C) | wind | `scatterometer_downloader` | EUMETSAT EUMDAC |
| Radiometer (AMSR2) | wind | `radiometer_downloader` | RSS `data.remss.com` (public HTTPS) |

### Collocation types

| Type | Example | Status |
|------|---------|--------|
| Point vs. Layer | mooring/buoy/ferrybox/drifter vs. SAR | ✅ implemented |
| Layer vs. Layer | scatterometer (ASCAT) / altimeter / radiometer (AMSR2) vs. SAR | ✅ implemented |

---

## Credentials

### Copernicus Dataspace (CDSE) — for SAR downloads

```bash
mkdir -p ~/.config/cdse
printf 'username=your@email.com\npassword=yourpassword' | base64 > ~/.config/cdse/credentials
chmod 600 ~/.config/cdse/credentials
```

Or use environment variables: `COPERNICUS_USERNAME` / `COPERNICUS_PASSWORD`.

Register at: https://dataspace.copernicus.eu

### Copernicus Marine — for in-situ and HF radar downloads

```bash
copernicusmarine login
```

### EUMETSAT EUMDAC — for ASCAT scatterometer downloads

```bash
mkdir -p ~/.eumdac
echo "username,password" > ~/.eumdac/credentials
chmod 600 ~/.eumdac/credentials
```

Or use environment variables: `EUMDAC_USERNAME` / `EUMDAC_PASSWORD`.

Register at: https://eoportal.eumetsat.int

---

## Running Tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov=sar_validation --cov-report=term-missing
```

---

## Documentation

- [Design choices](docs/design-choices.md) — toolbox overview and the rationale behind conventions: wind-direction rotation, collocation aggregation windows and tolerances, WV footprint handling, circular statistics, datatree filtering and CF metadata
- [Creating recipes](docs/creating-recipes.md) — recipe options and CLI flags
- [Collocation](docs/collocation.md) — matching algorithm mechanics and parameters
- [Statistics & plots](docs/cli-statistics-and-plots.md) — steps 4–5 outputs and CLI usage
