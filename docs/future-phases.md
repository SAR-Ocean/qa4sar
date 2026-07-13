# Future Implementation Phases

This document outlines the planned expansion of the SAR validation toolbox beyond the completed radiometer phases. These phases were scoped during the Phase 2 (bytemap radiometers) implementation but are scheduled for future work.

## Phase 3: Coastal HF-Radar Current Vectors

### Objective
Add current vector validation from coastal HF-radar networks to complement existing wind and wave validation.

### Data Source
- **MET Norway THREDDS server** — two overlapping radial-current radar networks:
  - FEDJ (Fedje/Sture)
  - SLAT (Slataroyden)
- Combine radial-component overlaps into total current vectors
- Project vectors onto SAR line-of-sight for direct comparison

### Key Technical Challenges
1. **Vector Geometry** — this is the highest-risk piece:
   - Combine two radial current measurements (each sensor measures only the radial component)
   - Use `makeTotalVector` from LorenzoCorgnati/EU_HFR_NODE_pyHFR (`totals.py`) to reconstruct vectors
   - Project total vector onto SAR line-of-sight using `rvlHeading` (radial velocity from heading)
   - Requires understanding of radar geometry and SAR acquisition heading metadata

2. **Collocation Type** — new geometry:
   - Current vectors are not simple scalar values like wind speed
   - Requires vector alignment (direction matching) before comparison
   - May need custom collocation logic beyond point_vs_layer / layer_vs_layer
   - Consider temporal alignment of rotating radars vs. SAR swath

3. **Data Structure**:
   - HF radar grids are typically irregular or cell-based (not uniform lat/lon)
   - May need interpolation to common grid or point-to-point collocation
   - Per-cell velocity measurement time varies (radar dwell time)

### Implementation Outline
1. **Downloader** — `hf_radar_downloader.py` (likely already exists for in-situ currents; extend for total vectors)
   - Query MET Norway THREDDS for FEDJ + SLAT hourly radial data
   - Download daily or hourly products
   
2. **Converter** — `datatree_converter.py`
   - `from_hf_radar_totals()`: read two radial grids, call `makeTotalVector`, produce vector pairs
   - Tag with `data_type="hf_radar"`, `platform_type="radar"`, `sensor="fedj_slat"`
   
3. **Collocation** — `collocation.py`
   - New collocation type or variant for vector comparison
   - Alignment strategy: component-wise (u/v separately) or magnitude+direction
   - Line-of-sight projection using SAR heading metadata
   
4. **Variable Mapping** — `_variable_map.py`
   - Add EWCT/NSCT (eastward/northward current components) or HCDT (current direction)
   - Circular statistics for direction comparison (like WDIR)

### Success Criteria
- Download FEDJ + SLAT radial data for a test period
- Reconstruct total vectors without gaps
- Collocate with SAR-derived rvlVelocity or rvlHeading over a known overlap region
- Validate vector magnitude and direction against in-situ current meters (buoys)

---

## Phase 4: More Scatterometers

### Objective
Extend scatterometer wind coverage from ASCAT (MetOp-B/C) to newer / higher-resolution sensors.

### Candidate Sensors
1. **Oceansat-3** (OSCAT-3, Indian Space Research Organisation)
   - 25 km wind resolution (vs. ASCAT's ~25 km)
   - Availability: 2024-present
   - Potential data source: PO.DAAC or ISRO

2. **HY-2B / HY-2C** (China)
   - RA-2/SCAT combinations, 25 km winds
   - Availability: 2021-present
   - Data source: NOAA, PO.DAAC, or direct from China

### Key Challenges
1. **Data Access** — existing altimeter/scatterometer downloaders use specific portals (Copernicus, EUMDAC)
   - Oceansat-3: likely OPeNDAP (PO.DAAC) or direct FTP
   - HY-2: may require direct archive access or OPeNDAP

2. **Format Variation**:
   - NetCDF structure may differ from ASCAT (dimension names, variable names)
   - Scale factors / fill values may differ
   - Some sensors may ship as HDF5 or custom binary

3. **Duplicate / Overlapping Coverage**:
   - Multiple scatterometers observing the same region at nearly the same time
   - Collocation spec must handle disambiguation (per-sensor specs like radiometer)

### Implementation Outline
1. **Downloader** — extend `scatterometer_downloader.py`
   - Add Oceansat-3 and HY-2 entries to sensor table
   - Implement PO.DAAC OPeNDAP client if not already present (or use xarray + kerchunk for lazy loading)
   - Fallback to FTP if OPeNDAP unavailable

2. **Converter** — extend `from_scatterometer_nc()`
   - Auto-detect sensor from file metadata
   - Map sensor-specific wind variable names to canonical `WSPD`
   - Handle different grid/dimension conventions

3. **Collocation** — extend `DEFAULT_LAYER_TYPE_SPECS`
   - Add `scatterometer_oceansat3` and `scatterometer_hy2b`/`_hy2c` per-sensor specs
   - Tuning: aggregation window, time tolerance may differ from ASCAT

### Success Criteria
- Download and ingest Oceansat-3 or HY-2 wind data for a test region
- Confirm wind values match published validation datasets
- Collocate with SAR and verify spatial/temporal alignment
- Produce statistics comparing ASCAT vs. new sensor biases

---

## Phase 5: Recipe-from-File Entry Point

### Objective
Simplify recipe creation by inferring bounds and time window from an existing data file (SAR, in-situ, or gridded product).

### Use Case
User has a SAR scene or in-situ observation file and wants to automatically create a recipe to download validation data for that time/region. Current workflow requires manually editing a recipe YAML.

### Implementation Outline

#### 5a. Recipe Inference from SAR L2_OCN File
```bash
sar-validate --infer-recipe /path/to/S1A_IW_OCN__2024-01-15T10:30:00_L2.nc --output recipes/auto_wind.yaml
```

1. **Metadata Extraction**:
   - Read SAR file: scene_id, acquisition_time, bounds (from slant-range geo-correction)
   - Apply spatial buffer (e.g., ±50 km beyond SAR swath)
   - Set time window to ±24 hours around acquisition (tunable)

2. **Variable Selection**:
   - Detect available variables in SAR file (owiWindSpeed, owiWaveHeight, etc.)
   - Infer validation variable from SAR var (owiWindSpeed → wind, VAVH → waves)

3. **Recipe Generation**:
   - Auto-populate geographic_bounds, temporal_bounds, variable
   - Use sensible defaults: all sources (scatterometer, altimeter, radiometer, in-situ)
   - Write to YAML, prompt user to confirm before download

#### 5b. Recipe Inference from In-Situ File
```bash
sar-validate --infer-recipe /path/to/buoy_obs_2024-01-15.csv --output recipes/auto_insitu_wind.yaml
```

1. **Metadata Extraction**:
   - Read CSV header and data: location columns (lon/lat), time column, variable columns
   - Infer variable type from column names (WSPD → wind, VAVH → waves)
   - Determine time range from CSV rows
   - Apply spatial buffer (±100 km around buoy)

2. **Validation Source Selection**:
   - Keep in-situ source fixed (to avoid recursive validation)
   - Auto-select SAR + satellite sources for comparison

#### 5c. Recipe Inference from Gridded Product
```bash
sar-validate --infer-recipe /path/to/altimeter_global_sla.nc --output recipes/auto_altimeter_waves.yaml
```

1. Parse global or regional grid for bounds
2. Extract time range from product
3. Create recipe to validate against SAR + other wave sources

### Partial-Overlap Handling
A key requirement: if the SAR scene is from Jan 15, infer a recipe that:
- Downloads SAR for Jan 15 (exact acquisition time)
- Downloads validation sources for Jan 14–16 (24-hour window before and after)
- On collocation, matches pairs within ±180 min time tolerance
- Expect sparse matches at edges but full overlap in the middle

### Implementation Steps
1. **CLI** — add `--infer-recipe` flag to `cli.py`
   - Detect input file type (NetCDF, CSV, HDF5)
   - Call appropriate inference function

2. **Inference Module** — new file `sar_validation/core/recipe_inference.py`
   - `infer_from_sar_l2_ocn(path) -> RecipeConfig`
   - `infer_from_insitu_csv(path) -> RecipeConfig`
   - `infer_from_gridded_product(path) -> RecipeConfig`
   - Shared helper to apply buffer + time window defaults

3. **Validation** — before writing YAML:
   - Check that inferred bounds are valid (not wrapping, reasonable lat range)
   - Warn if time window is <12 hours (may be too small for collocation matches)
   - Prompt user to review and accept

### Success Criteria
- Infer recipe from a SAR L2_OCN file with correct bounds, time, variable
- Infer recipe from a buoy CSV with correct spatial buffer, time range
- Run inferred recipe in dry-run mode and confirm URLs are generated
- Download and collocate against inferred recipe; produce statistics

---

## Execution Order & Dependencies

### Priority 1 (Recommended Next)
**Phase 3: HF Radar** — highest learning value but highest risk
- Establishes vector collocation pattern for future current sensors
- Tight coupling with SAR geometry (heading, line-of-sight) opens new validation opportunities
- **Risk**: vector reconstruction and radar geometry are unfamiliar territory — likely 1-2 week exploratory phase before implementation

### Priority 2
**Phase 5: Recipe Inference** — highest user ergonomic value
- Eliminates manual YAML editing, simplifies entry point
- Low technical risk (mostly file I/O and string manipulation)
- Quick win to improve usability

### Priority 3
**Phase 4: More Scatterometers** — incremental feature, moderate effort
- Low risk (reuses existing scatterometer pattern)
- Benefit is marginal (ASCAT + new sensor = modest coverage improvement)
- Defer until HF radar and recipe inference complete

---

## Open Questions & Decisions Deferred

### HF Radar
- Should vectors be stored as (u, v) components or (magnitude, direction)?
- What is the expected collocation geometry? (grid-to-grid, point-to-grid, point-to-point?)
- How to handle temporal misalignment (radar dwell time vs. SAR pulse)?

### More Scatterometers
- Is OPeNDAP preferred over FTP for Oceansat-3? (OPeNDAP is slower but doesn't require credentials)
- Should Oceansat-3 and HY-2 be treated as separate sensors (per-spec tuning) or grouped under "scatterometer"?

### Recipe Inference
- Should --infer-recipe also auto-detect variable type from file structure, or require user to specify (--variable wind)?
- What is a sensible default spatial buffer? (±50 km for SAR scenes, ±100 km for buoys?)

---

## References

- **HF Radar**: LorenzoCorgnati/EU_HFR_NODE_pyHFR ([GitHub](https://github.com/LorenzoCorgnati/EU_HFR_NODE_pyHFR)) — `totals.py` and SAR line-of-sight projection utilities
- **Oceansat-3**: NASA PO.DAAC OPeNDAP ([link](https://podaac.jpl.nasa.gov/)) — Oceansat-3 scatterometer wind data
- **HY-2B/C**: NOAA Archive ([link](https://www.ncei.noaa.gov/)) — Chinese satellite data

---

**Last Updated**: 2026-07-13  
**Status**: Documented for future reference; implementation deferred pending Phase 1 & 2 completion and stabilization.
