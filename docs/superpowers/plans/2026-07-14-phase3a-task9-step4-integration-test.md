# Phase 3a Task 9, Step 4 — End-to-End Collocation Integration Test

**Parent Plan:** `docs/superpowers/plans/2026-07-14-phase3a-noaa-hfradar-currents.md` (Task 9, Step 4)

**Status:** Deferred after technical implementation complete (steps 1–3, 5 ✅).

**Goal:** Validate the full collocation pipeline with real NOAA HFRnet + Sentinel-1 L2 OCN currents data, confirming that:
- Collocated pairs are produced over the HF-radar/SAR geographic overlap
- `rvlRadVel_projection` is present in the collocation output
- Scatter plot and geographic map visualizations are generated
- Raw agreement is documented as expected to be weak (r < 0.5) due to deferred WASV/instrument corrections

---

## Prerequisite State

**Already complete:**
- ✅ All Tasks 1–8 implemented and committed
- ✅ Steps 1–3 of Task 9 validated:
  - ERDDAP URL builds correctly
  - Real HFRnet data downloaded to `/tmp/hfr_run/hfr_noaa/ucsdHfrW6_6km_2026-07-10.nc`
  - Datatree converter creates `/validation/hfr_noaa/ucsdHfrW6_6km_2026-07-10` node with 3,852 points
- ✅ Unit test suite passes (240 tests)
- ✅ Sentinel-1 L2 OCN sample data available at `/home/chvan0015/Documents/data/sentinel1/SAFE_files/`

---

## Steps

### Step 1: Prepare the test environment

- [ ] Ensure HFRnet and S1 data are in place:
  ```bash
  ls -la /tmp/hfr_run/hfr_noaa/ucsdHfrW6_6km_2026-07-10.nc
  ls -la /tmp/hfr_run/S1_L2_OCN/S1C_IW_OCN*
  ```
  Expected: both files/folders exist from the manual prerequisite setup.

### Step 2: Create a `currents` recipe for US-West (June–July 2026)

Build a minimal currents recipe YAML with:
- Geographic bounds: `min_lon=-125, max_lon=-119, min_lat=33, max_lat=38` (US-West coast)
- Temporal bounds: `start=2026-06-01, end=2026-07-15` (covers both S1 and HFR overlap)
- Validation sources: both `hf_radar_noaa` and existing inline/altimeter sources (if needed)
- Collocation spec: use default `layer_vs_layer` for `hf_radar_grid`

Example template:
```yaml
geographic_bounds:
  min_lon: -125
  max_lon: -119
  min_lat: 33
  max_lat: 38
temporal_bounds:
  start: "2026-06-01"
  end: "2026-07-15"
validation_sources:
  - source_type: hf_radar_noaa
    min_depth: -2.0
    max_depth: 2.0
    download_kwargs:
      resolution_km: 6
collocation:
  type: layer_vs_layer
  layer_type_specs:
    hf_radar_grid:
      time_tolerance_minutes: 20
      aggregation_window_km: 6.0
      distance_weighting: equal
```

- [ ] Create recipe file at `tests/recipes/test_currents_integration.yaml` or a temporary location.

### Step 3: Run the collocation pipeline

- [ ] Execute the full pipeline on the test data:
  ```bash
  python -m sar_validation.orchestrator --base-dir /tmp/hfr_run --recipe <path_to_recipe>
  ```
  Or use the project's standard `sar-validate` CLI entrypoint if available.

- [ ] Expected output structure:
  - `collocation/` directory with collocated datasets (NetCDF or Parquet)
  - `statistics/` directory with comparison metrics
  - `plots/` directory with scatter plots and geographic maps

### Step 4: Verify collocations and projections

- [ ] Check the collocation output for `hfr_noaa` pairs:
  ```python
  import xarray as xr
  ds = xr.open_dataset("<output>/collocation_hfr_noaa.nc")
  print("rvlRadVel_projection" in ds.data_vars)  # Should be True
  print(f"Collocated pairs: {len(ds.collocation)}")  # Should be > 0
  ```
  Expected:
  - `rvlRadVel_projection` present (the radial projection computed by `_project_currents_to_radial`)
  - Non-empty collocation set over the geographic overlap

### Step 5: Inspect visualizations

- [ ] Open the generated plots:
  - Scatter plot: `rvlRadVel` (x-axis) vs `rvlRadVel_projection` (y-axis)
  - Geographic map: collocated points overlaid on the HF-radar grid extent

- [ ] Expected observations:
  - Points cluster around but with substantial scatter (r < 0.5 expected per Martin et al. 2022)
  - Geographic coverage shows overlap over the US-West coast region

### Step 6: Document findings

- [ ] Create or update a run notes file (`docs/runs/2026-07-14-phase3a-integration.md`) with:
  - Date/time of run
  - Data sources used (S1 acquisition date, HFR date range)
  - Number of collocated pairs
  - Correlation coefficient (raw, without corrections)
  - Key observation: **raw agreement is weak (expected)** because WASV/instrument corrections are deferred to a later phase; this is the expected Phase 3a outcome, not a bug
  - Reference: Martin, Gommenginger, Jacob & Staneva (2022), RSE 268:112758

- [ ] If successful, optionally add the run notes to the PR description.

---

## Success Criteria

- [ ] Collocation produces ≥1 pair (ideally 100+) over the HF-radar/SAR geographic overlap
- [ ] `rvlRadVel_projection` is present in output
- [ ] Scatter plot and geographic map are generated without errors
- [ ] Documentation notes expected weak correlation as a Phase 3a limitation, not a failure

---

## Notes

- The HFRnet data used (June 2026) is slightly earlier than the HFR data fetched in Steps 1–3 (July 10). If temporal mismatch causes zero collocations, adjust the S1 date or HFR temporal window and retry.
- If the orchestrator/run entrypoint is unavailable, the step can also be driven manually via the collocation module's public API.
