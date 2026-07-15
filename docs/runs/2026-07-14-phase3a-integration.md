# Phase 3a — NOAA HF-Radar Currents Integration Run

**Date:** 2026-07-15
**Plan:** `docs/superpowers/plans/2026-07-14-phase3a-task9-step4-integration-test.md` (Task 9, Step 4 of the parent `docs/superpowers/plans/2026-07-14-phase3a-noaa-hfradar-currents.md`)

## Data sources

- **NOAA HFRnet ERDDAP:** `ucsdHfrE6` (US East/Gulf coast, 6 km), 2026-07-11 → 2026-07-12.
- **Sentinel-1 L2 OCN:** 8 scenes (S1C/S1D, IW and WV mode) over the same bbox/window,
  downloaded via `recipes/currents_useastcoast.yaml`
  (bbox `lon [-90, -60]`, `lat [30, 40]`, US East/Gulf coast).

The originally-suggested prerequisite S1 scenes (at
`/home/chvan0015/Documents/data/sentinel1/SAFE_files/`) turned out to be over the
Skagerrak and North Sea — outside all NOAA HFRnet coverage — so a fresh US
East/Gulf-coast download was used instead.

## Result

- Collocation: **57 collocated pairs** total; the `hf_radar_grid` layer-vs-layer
  path alone produced 19 + 28 = 47 matches across two overlapping IW scenes.
- `rvlRadVel_projection` present in the collocation output, computed via the
  shared `_project_currents_to_radial` helper (Task 1 of the parent plan).
- Statistics (`radar` source, N=47): bias 0.973, std 0.207, RMSE 0.994,
  **r = 0.141**, scatter index 60.7.
- Scatter, residuals, statistics, temporal-offset, and geographic plots all
  generated without error.

## Expected weak correlation

The weak raw correlation (r = 0.141) and large positive bias (~0.97 m/s) match
the Phase 3a global constraint: raw `rvlRadVel` includes a wind-wave artefact
bias of up to ~2 m/s that is not yet corrected (the WASV/instrument
correction chain is deferred to a later phase). The scatter plot shows
`rvlRadVel` sitting systematically ~0.85–1.0 m/s above `rvlRadVel_projection`,
consistent with Martin, Gommenginger, Jacob & Staneva (2022), RSE 268:112758.
**This is the expected Phase 3a outcome, not a defect.**

## Bug found and fixed during this run

`plot_geographic()` failed for the `currents` gridded (IW) scenes with:

```
plot_geographic failed for rvlRadVel: x and y arguments to pcolormesh cannot
have non-finite values or be of type numpy.ma.MaskedArray with masked values
```

Root cause: S1 OCN products carry NaN `lon`/`lat` at swath-edge/invalid-retrieval
cells; `matplotlib.pcolormesh` rejects non-finite coordinate grids outright.
This is a **pre-existing bug**, unrelated to the `hf_radar_grid`/NOAA converter
work — the NOAA integration test was simply the first `currents` scenario to
exercise a gridded IW scene with NaN geolocation through `--plot` end-to-end.

Fixed in `sar_validation/core/visualization.py`: added `_fill_nan_nearest()`
(nearest-neighbour fill via `scipy.ndimage.distance_transform_edt`) to repair
the coordinate grid before `pcolormesh`, and mask the data array at the
originally-invalid geolocation cells so they render as transparent rather than
being assigned a wrong position. Regression test added in
`tests/test_visualization.py::TestPlotGeographic::test_gridded_scene_with_nan_geolocation_does_not_raise`
(confirmed to fail without the fix, pass with it).

## Success criteria (from the Step 4 plan)

- [x] Collocation produces ≥1 pair (57, well above the 100+ "ideal" bar is not
      met but ≥1 and non-trivial is)
- [x] `rvlRadVel_projection` present in output
- [x] Scatter plot and geographic map generated without errors
- [x] Weak correlation documented as expected Phase 3a limitation, not a failure
