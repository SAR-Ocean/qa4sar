# SAR L2 Ocean Data Validation Toolbox

A standalone Python package for validating Sentinel-1 L2_OCN (Level 2 Ocean) products
against multi-source in-situ and satellite observations.

Supports validation of **wind** (speed + direction), **ocean currents**, **significant wave height**, and **soil moisture**.

---

## Architecture

The toolbox is structured around 6 sequential steps that together form a complete validation workflow:

```
Step 0 — Create recipe (.yaml)
          │  define validation variable (wind / currents / waves) --> selects the validation data sources
          │  define geographic bounds (min/max lon/lat)
          │  define temporal bounds (start/end dates)
          ▼
Step 1 — Download data
          │  download SAR L2_OCN + all validation sources
          │  for the recipe region and time window
          ▼
Step 2 — Convert to xarray.DataTree
          │  standardize all formats
          │  filter to recipe geographic/temporal bounds +/- tolerances
          │  handle different grids / resolutions
          ▼
Step 3 — Collocation
          │  point vs. point    (mooring / buoy / drifter / ferrybox / tidal gauge vs. SAR wave parameter)
          │  point vs. layer    (mooring / buoy / drifter / ferrybox / tidal gauge vs. SAR)
          │  layer vs. layer    (scatterometer / altimeter / radiometer / HF-radar vs. SAR)
          |  create a collocation diagnostics plot
          ▼
Step 4 — Store collocated pairs
          │  keep only overlapping data
          │  store spatial/temporal offset metadata
          ▼
Step 5a — Compute statistics
          │  bias, RMSE, correlation, scatter index
          │  per platform/source
          ▼
Step 5b — Generate visualisation & report
          │  scatter plots, geographic maps
          │  statistics charts, residuals
          │  PDF validation report
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
│   ├── statistics.py       # Step 5a: compute bias, RMSE, correlation, scatter index
│   ├── visualization.py    # Step 5b: scatter plots, geographic maps, statistics charts, residuals
│   ├── _variable_map.py    # Variable mapping (wind/currents/waves)
│   └── __init__.py         # Package exports
└── downloaders/
    ├── base.py             # Shared credential handling & helpers
    ├── sar_downloader.py   # Sentinel-1 L2_OCN via Copernicus Dataspace
    ├── insitu_downloader.py   # Moorings/buoys/ferrybox via Copernicus Marine
    ├── hf_radar_downloader.py # HF radar via Copernicus Marine
    ├── scatterometer_downloader.py  # ASCAT (MetOp) via EUMETSAT EUMDAC
    ├── altimeter_downloader.py      # Along-track SWH/wind via Copernicus Marine
    ├── radiometer_downloader.py     # RSS radiometer ocean winds (AMSR2 NetCDF + GMI/SSMIS/WindSat bytemaps)
    ├── soil_moisture_downloader.py   # Sentinel-1 CLMS Surface Soil Moisture via Copernicus Dataspace (CDSE)
    ├── ismn_downloader.py            # ISMN local-archive station selector (no download API)
    └── _rss_bytemap.py              # Decoder for RSS binary bytemap (.gz) radiometer products
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
sar-validate --create-recipe soil_moisture
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

# --- Step 5a/5b: analyse / visualize ---
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
| Radiometer — AMSR2 (NetCDF); GMI, SSMIS F16/F17/F18, WindSat (binary bytemaps) | wind (+ direction from WindSat) | `radiometer_downloader` | RSS `data.remss.com` (public HTTPS) |
| Sentinel-1 CLMS Surface Soil Moisture | soil moisture | `soil_moisture_downloader` | Copernicus Dataspace (CDSE) |
| ISMN (International Soil Moisture Network) | soil moisture | `ismn_downloader` | Manual portal download (no API) |

> **Note:** The Sentinel-1 CLMS Surface Soil Moisture downloader's CDSE query
> parameters (`DATASET_IDENTIFIER`, `PRODUCT_EXTENT` in
> `sar_validation/downloaders/soil_moisture_downloader.py`) and GeoTIFF value
> decoding (`from_sar_l3_ssm_geotiff` in `sar_validation/core/datatree_converter.py`)
> have been confirmed against a real downloaded CEURO product (embedded
> `scale_factor`/`add_offset`/`flag_values`/geospatial-extent GDAL tags) and a
> successful end-to-end recipe run. Each CDSE product is served as a zip
> containing a soil-moisture GeoTIFF plus a sibling uncertainty-layer GeoTIFF
> (and, separately, a redundant NetCDF variant) — both are downloaded/unzipped
> and filtered automatically, no manual handling needed.

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

### ISMN — for soil moisture in-situ validation

ISMN has no download API. Register at https://ismn.earth/en/dataviewer/,
filter by bounding box / date range / depth / variable ("soil moisture") on
the portal, and download the resulting zip. Running the recipe before the
archive exists prints these exact filter values so they can be
copy-pasted into the portal form, along with the recommended download
options: **CEOP-formatted** (variables stored in separate files, zipped),
**"Good" quality flags only**, and **gap filling disabled**, plus the
exact folder to drop the downloaded zip into — no recipe edits needed,
just drop it in and re-run. To reuse one archive across multiple recipes
instead, set its path explicitly as `ismn_archive_path` in the recipe's
`download_kwargs` for the `ismn` validation source.

---

## Running Tests and Checks

```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov=sar_validation --cov-report=term-missing
```

Linting and type checking are both required to pass before merging:

```bash
ruff check .
python -m mypy -p sar_validation
```

---

## Documentation

- [Design choices](docs/design-choices.md) — toolbox overview and the rationale behind conventions: wind-direction rotation, collocation aggregation windows and tolerances, WV footprint handling, circular statistics, datatree filtering and CF metadata
- [Creating recipes](docs/creating-recipes.md) — recipe options and CLI flags
- [Collocation](docs/collocation.md) — matching algorithm mechanics and parameters
- [Statistics & plots](docs/cli-statistics-and-plots.md) — steps 4–5 outputs and CLI usage
