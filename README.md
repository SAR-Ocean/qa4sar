# SAR L2 Ocean Data Validation Toolbox

A standalone Python package for validating Level 2 SAR products against multi-source in-situ and satellite observations.

Supports validation of **wind** (speed + direction), **ocean currents**, **significant wave height**, and **soil moisture** -- against in-situ, satellite, and model (ERA5, HyCOM) sources.

Supported SAR products:
- Sentinel-1 L2_OCN (Level 2 Ocean): wind/waves/currents
- Sentinel1 clms ssm (Surface Soil Moisture; Europe; daily): soil moisture
- NISAR SME2 (beta & provisional): soil_moisture 
- RADARSAT-2 (NOAA NCEI SAR-derived ocean wind, speed only): wind

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
          |  --dry-run available to check product availibility
          ▼
Step 2 — Convert to xarray.DataTree
          │  standardize all formats + attach CF-convention metadata 
          │  filter to recipe geographic/temporal bounds +/- tolerances
          │  handle different grids / resolutions
          ▼
Step 3 — Collocation + store collocated pairs
          │  point vs. point    (mooring / buoy / drifter / ferrybox / tidal gauge vs. SAR wave parameter)
          │  point vs. layer    (mooring / buoy / drifter / ferrybox / tidal gauge vs. SAR)
          │  layer vs. layer    (scatterometer / altimeter / radiometer / HF-radar vs. SAR)
          |  create a collocation diagnostics plot
          │  store collocated data + spatial/temporal offset metadata
          ▼
Step 4 — Compute statistics
          │  bias, RMSE, correlation, scatter index
          │  per platform/source
          ▼
Step 5 — Generate visualisation & report
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
│   ├── statistics.py       # Step 4: compute bias, RMSE, correlation, scatter index
│   ├── visualization.py    # Step 5: scatter plots, geographic maps, statistics charts, residuals
│   ├── _variable_map.py    # Variable mapping (wind/currents/waves)
│   └── __init__.py         # Package exports
└── downloaders/
    ├── base.py                                  # Shared credential handling & helpers
    ├── _hf_radar_regions.py                     # Copernicus Marine HF-radar region bbox/flag table (shared, not a downloader)
    ├── _noaa_hfr_regions.py                     # NOAA HF-radar region bbox/dataset-id table (shared, not a downloader)
    ├── _rss_bytemap.py                          # Decoder for RSS binary bytemap (.gz) radiometer products
    ├── altimeter_downloader.py                  # Along-track SWH/wind altimetry via Copernicus Marine
    ├── ascat_soil_moisture_downloader.py        # ASCAT (MetOp) soil moisture (SOMO12) via EUMETSAT EUMDAC (discontinued after 2025-07-15)
    ├── hsaf_downloader.py                        # ASCAT (MetOp) soil moisture NRT (H122 6.25km default, H29 12.5km opt-in) via H-SAF FTP, rolling last 60 days
    ├── earthdata_soil_moisture_downloader.py    # AMSR-E/2, SMAP, and NISAR SME2 soil moisture via NASA Earthdata
    ├── era5_downloader.py                       # ERA5 reanalysis (wind/waves/soil_moisture) via Copernicus CDS
    ├── gportal_downloader.py                    # AMSR2 soil moisture via JAXA G-Portal (SFTP)
    ├── hf_radar_downloader.py                   # Near-real-time HF-radar surface currents via Copernicus Marine
    ├── hf_radar_historical_downloader.py        # Delayed-mode/historical HF-radar currents via Copernicus Marine
    ├── hf_radar_us_downloader.py                # US HF-radar waterfall selector: NOAA ERDDAP → NOAA THREDDS → Copernicus Marine
    ├── hycom_downloader.py                      # HyCOM ocean model surface currents (water_u/water_v) via THREDDS OPeNDAP
    ├── insitu_currents_historical_downloader.py # Delayed-mode in-situ currents (ADCP/Argo/drifter/glider) via Copernicus Marine
    ├── insitu_downloader.py                      # Moorings/buoys/ferrybox via Copernicus Marine
    ├── ismn_downloader.py                        # ISMN local-archive soil-moisture station selector (no download API)
    ├── noaa_hfradar_downloader.py                # NOAA HFRnet gridded surface currents via ERDDAP (rolling ~90-day window)
    ├── noaa_hfradar_thredds_downloader.py        # NOAA HFRnet gridded surface currents via NCEI THREDDS archive (2006-present)
    ├── radarsat2_wind_downloader.py               # RADARSAT-2 SAR-derived ocean wind speed via NOAA NCEI THREDDS archive
    ├── radiometer_downloader.py                  # RSS radiometer ocean winds (AMSR2 NetCDF + GMI/SSMIS/WindSat bytemaps)
    ├── scatterometer_downloader.py               # ASCAT (MetOp) wind via EUMETSAT EUMDAC
    ├── scatterometer_ftp_downloader.py           # HY-2B/HY-2C/Oceansat-3 scatterometer wind via OSI-SAF FTP (only last 3 days)
    ├── sentinel1_l2_ocn_downloader.py            # Sentinel-1 L2_OCN (wind/currents/waves) via Copernicus Dataspace (CDSE)
    ├── sentinel1_soil_moisture_downloader.py     # Sentinel-1 CLMS Surface Soil Moisture via Copernicus Dataspace (CDSE)
    ├── cds_soil_moisture_downloader.py           # C3S CDS satellite soil moisture (ACTIVE/PASSIVE/COMBINED, 0.25°) via cdsapi
    └── smos_downloader.py                        # SMOS L2 soil moisture via ESA SMOS FTPS
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
ds_sar    = converter.from_sar_l2_ocn_safe("data/.../S1_L2_OCN/S1A_IW_OCN_...SAFE", product_type="wind")
ds_insitu = converter.from_insitu_csv("data/.../copernicus_insitu_data/obs.csv", source_type="buoy")

# --- Step 3: collocate and store as DataFrame ---
colloc = PointLayerCollocation(spatial_tolerance_km=50, time_tolerance_minutes=60)
results = colloc.collocate(
    sar_data={"wind_speed": sar_ws_array},
    sar_lon=sar_lon, sar_lat=sar_lat, sar_time=sar_time,
    val_data=df_insitu,
    val_source="buoy",
)
df = DataTreeConverter.to_dataframe(results)
df.to_csv("data/.../collocations_wind_buoy.csv", index=False)

# --- Step 4: calculate statistics ---

# --- Step 5: visualize and create PDF report ---
import matplotlib.pyplot as plt
plt.scatter(df["val_wind_speed"], df["sar_wind_speed"])
plt.xlabel("Buoy wind speed (m/s)")
plt.ylabel("SAR wind speed (m/s)")
plt.show()
```

---

## Supported Data Sources

| Source | Variable | Downloader | Service | Temporal coverage |
|--------|----------|------------|---------| ----------------- |
| Sentinel-1 L2_OCN | wind / currents / waves | `sentinel1_l2_ocn_downloader` | Copernicus Dataspace (CDSE) | 2014-10-03 - present |
| RADARSAT-2 | wind (speed only) | `radarsat2_wind_downloader` | NOAA NCEI THREDDS | 2014-05-02 - present |
| Moorings / Buoys / Ferryboxes | wind / currents / waves | `insitu_downloader` | Copernicus Marine | varies by platform; max 2020-01-01 - present |
| Delayed-mode in-situ currents (ADCP / Argo / drifter / glider) | ocean currents |   `insitu_currents_historical_downloader` | Copernicus Marine | varies by platform (6 - 24 months latency) |
| HF Radar (near-real-time) | ocean currents | `hf_radar_downloader` | Copernicus Marine | varies by radar; max 2020-01-01 - present |
| HF Radar (delayed-mode/historical) | ocean currents | `hf_radar_historical_downloader` | Copernicus Marine | varies by platform |
| HF Radar (US regions) | ocean currents | `hf_radar_us_downloader` | NOAA ERDDAP → NOAA THREDDS → Copernicus Marine (waterfall) | see HF Rader (NOAA HFRnet) |
| HF Radar (NOAA HFRnet, rolling window) | ocean currents | `noaa_hfradar_downloader` | NOAA ERDDAP | 90 days rolling window |
| HF Radar (NOAA HFRnet, full archive) | ocean currents | `noaa_hfradar_thredds_downloader` | NOAA NCEI THREDDS | 2006 - present (~1 month latency) |
| ASCAT (MetOp-B/C) | wind | `scatterometer_downloader` | EUMETSAT EUMDAC | MetOp-B/C: 2012/2019 - present |
| HY-2B / HY-2C / Oceansat-3 | wind | `scatterometer_ftp_downloader` | OSI-SAF FTP | last 3 days |
| Radiometer — AMSR2 (NetCDF); GMI, SSMIS F16/F17/F18, WindSat (binary bytemaps) | wind (+ direction from WindSat) | `radiometer_downloader` | RSS `data.remss.com` (public HTTPS) | AMSR2/GMI/SSMIS F16/F17/F18: 2012-07-02/2014-03-04/2003-10-26/2006-11-04/2009-10-18 - present |
| Sentinel-1 CLMS Surface Soil Moisture | soil moisture | `sentinel1_soil_moisture_downloader` | Copernicus Dataspace (CDSE) | 2014 - present (Europe only) |
| ASCAT Soil Moisture (SOMO12) | soil moisture | `ascat_soil_moisture_downloader` | EUMETSAT EUMDAC | 2007 - 2025-07-15 |
| ASCAT Soil Moisture NRT (H122/H29) | soil moisture | `hsaf_downloader` | H-SAF FTP | rolling last 60 days (⚠️ gap between 2025-07-15 and 60 days ago is not covered); H122 (6.25km) by default, H29 (12.5km) via `download_kwargs: {hsaf_product: h29}` |
| AMSR-E/AMSR2 (NSIDC-0451 / AU_Land) | soil moisture | `earthdata_soil_moisture_downloader` | NASA Earthdata | AMSR-E: 2002 - 2011; AMSR2: 2012 - 2025-09-01 ⚠️ frozen |
| AMSR2 (JAXA G-Portal) | soil moisture | `gportal_downloader` | JAXA G-Portal (SFTP) | 2012 - present |
| SMAP (SPL2SMP_E) | soil moisture | `earthdata_soil_moisture_downloader` | NASA Earthdata | 2015 - present |
| NISAR SME2 (beta & provisional) | soil moisture | `earthdata_soil_moisture_downloader` | NASA Earthdata | 2025-10-01 - present |
| SMOS (SM_OPER_MIR_SMUDP2) | soil moisture | `smos_downloader` | ESA SMOS FTPS | 2010 - present |
| C3S CDS Soil Moisture (ACTIVE / PASSIVE / COMBINED) | soil moisture | `cds_soil_moisture_downloader` | Copernicus CDS | 1991 - present (ICDR: ~10-day latency) |
| ISMN (International Soil Moisture Network) | soil moisture | `ismn_downloader` | Manual portal download (no API) | Varies by station |
| ERA5 model | wind / waves / soil moisture | `era5_downloader` | Copernicus CDS | 1940 (ERA5) / 1950 (ERA5-Land) - present (~5 day latency) |
| HYCOM model | currents | `hycom_downloader` | HYCOM | 2018-12-04 - present (~48 hour latency) |

NISAR SME2 (`m3 m-3`, L-band, twice-daily per-overpass granules) is a second,
beta/provisional SAR-side source for soil moisture, selectable per recipe via
`sar-validate --create-recipe soil_moisture --sar-source nisar_sme2` (or
`source: nisar_sme2` in the recipe YAML).

RADARSAT-2 (C-band, 0.5 km, NOAA NCEI) is a second SAR-side source for
wind, selectable via `sar-validate --create-recipe wind --sar-source
radarsat2` (or `source: radarsat2` in the recipe YAML). It provides
**wind speed only**. Coverage is global but concentrated over 
Alaska/the North Pacific.

### Collocation types

| Type | Example |
|------|---------|
| Point vs. Point | mooring / buoy / ferrybox / drifter / tidal gauge vs. SAR WV-mode vignettes | 
| Point vs. Layer | mooring / buoy / ferrybox / drifter / tidal gauge vs. SAR | 
| Layer vs. Layer | scatterometer / altimeter / radiometer (flattened) vs. SAR | 
| Model vs. Layer | ERA5 / HYCOM (gridded) vs. SAR |

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

Store credentials in your OS keyring (GNOME Keyring/libsecret on Linux):

```bash
sar-validate --set-credential eumdac
```

You'll be prompted for a username and password; nothing is written to disk
in plaintext. Or use environment variables: `EUMDAC_USERNAME` /
`EUMDAC_PASSWORD`.

If you have an existing `~/.eumdac/credentials` file from before this
change, it's picked up automatically and migrated into the OS keyring the
first time it's needed — a one-time console notice confirms the migration
and tells you it's then safe to delete the old file yourself (it is never
deleted automatically).

Register at: https://eoportal.eumetsat.int

### EUMETSAT OSI-SAF wind FTP — for OSI-SAF wind downloads

Store credentials in your OS keyring:

```bash
sar-validate --set-credential osi_saf
```

Or use environment variables: `OSI_SAF_FTP_USERNAME` / `OSI_SAF_FTP_PASSWORD`.

An existing `~/.eumetsat_osi_saf_wind_credentials` file is migrated into
the OS keyring automatically on first use (with a one-time console notice);
you can delete the old file afterwards.

Register at: https://osi-saf.eumetsat.int/register

### H-SAF FTP — for ASCAT soil moisture NRT downloads

Store credentials in your OS keyring:

```bash
sar-validate --set-credential hsaf
```

Or use environment variables: `HSAF_FTP_USERNAME` / `HSAF_FTP_PASSWORD`.

Register at: https://hsaf.meteoam.it/User/Register

### JAXA G-Portal — for AMSR2 soil moisture downloads (SFTP fallback)

Store credentials in your OS keyring:

```bash
sar-validate --set-credential gportal
```

Or use environment variables: `GPORTAL_USERNAME` / `GPORTAL_PASSWORD`. If
none of these resolve, the downloader falls back to an interactive
terminal prompt (deliberately not persisted anywhere).

An existing `~/.jaxa_gportal_credentials` file is migrated into the OS
keyring automatically on first use (with a one-time console notice); you
can delete the old file afterwards.

Register at: https://gportal.jaxa.jp

### Earthdata (AMSR-E/2, SMAP, NISAR) — for satellite/SAR soil moisture downloads

```bash
sar-validate --set-credential earthdata
```

This prompts for your NASA Earthdata Login username/password and stores
them in the OS keyring. Credential resolution priority is: explicit
arguments, then environment variables (`EARTHDATA_USERNAME` /
`EARTHDATA_PASSWORD`), then the OS keyring, then an existing `~/.netrc`
`urs.earthdata.nasa.gov` entry — the `~/.netrc` entry is not an ongoing
override; it is only read as a one-time migration source when the
keyring has nothing stored, and is then copied into the keyring
automatically (with a one-time console notice) so subsequent runs use
the keyring directly.

Register at: https://urs.earthdata.nasa.gov

`earthdata_soil_moisture_downloader` uses these credentials to download
AMSR-E/2 and SMAP soil moisture products, and NISAR SME2 soil moisture
(as a SAR source), from the NASA Earthdata archive.

### ESA SMOS FTPS — for SMOS soil moisture downloads

Store credentials in your OS keyring:

```bash
sar-validate --set-credential smos
```

Or use environment variables: `SMOS_FTP_USERNAME` / `SMOS_FTP_PASSWORD`.

An existing `~/.esa_smos_credentials` file is migrated into the OS keyring
automatically on first use (with a one-time console notice); you can
delete the old file afterwards.

Register at: https://eoiam-idp.eo.esa.int/ 

The `smos_downloader` uses these credentials to download SMOS L2 soil moisture products
from the ESA SMOS FTPS archive.

### Copernicus CDS — for C3S satellite soil moisture downloads and the ERA5 model

`cds_soil_moisture_downloader` downloads via the
[`cdsapi`](https://cds.climate.copernicus.eu/how-to-api) library, which reads
credentials automatically from `~/.cdsapirc`. Create that file after registering:

```ini
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

Register and generate a token at: https://cds.climate.copernicus.eu

Note: no `sar-validate --set-credential` command is needed; `cdsapi` reads
`~/.cdsapirc` natively.

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

- [Design choices](docs/design-choices.md) — toolbox overview and the rationale behind conventions such as wind-direction rotation, collocation aggregation windows and tolerances, WV footprint handling, circular statistics, datatree filtering and CF metadata
