# RVL land-flag masking + rvlRadVelStd extraction — Design

**Date:** 2026-07-17
**Status:** Approved (pending spec review)
**Type:** Bugfix + small enhancement

## 1. Problem

Currents recipes extract `rvlRadVel` (and `rvlHeading`, `rvlIncidenceAngle`) from
Sentinel-1 L2 OCN RVL products via `DataTreeConverter._extract_rvl_grid_data`
(`sar_validation/core/datatree_converter.py`), covering both the 2-D grid path
(IW/EW/SM) and the flatten-to-points path (WV, reused by
`_extract_rvl_from_wv_safe`). Two gaps remain:

1. **No land masking.** The raw products carry `rvlLandFlag` (set to 1 when
   land coverage of the cell exceeds 10%; same dims as `rvlRadVel` — 2-D
   `(rvlAzSize, rvlRaSize)` for SM/WV, 3-D `(rvlAzSize, rvlRaSize, rvlSwath)`
   for EW/IW), but the extractor ignores it. Land-contaminated cells flow into
   collocation and validation like any other measurement. Spot-checking real
   fixtures confirms this is not a hypothetical: several IW scenes under
   `data/2026-06-01-.../` have tens of thousands of land-flagged cells with a
   mean `rvlRadVel` of 0.6–0.8 m/s — non-zero, i.e. exactly the kind of
   contamination that should be excluded and flagged for attention rather than
   silently averaged into the SAR-side aggregate.
2. **`rvlRadVelStd` is never extracted.** The raw products carry it (dims
   identical to `rvlRadVel`) but the extractor doesn't read it, so it never
   reaches `datatree.nc` and consequently never reaches
   `collocation_results.nc`.

## 2. Approach

Fix both gaps at the single point where RVL is extracted
(`_extract_rvl_grid_data`, both its grid branch and its
`flatten_to_points=True` branch), and add a small reporting step so land
contamination is visible in the PDF report rather than only in logs.

Collocation (`sar_validation/core/collocation.py`) needs **no code changes**:
both the grid path (`sar_data_3d`) and the WV path (`sar_point_vars`) already
build their per-variable dict generically from *every* data variable with
matching dims (`("y","x")` or `("point",)`), and
`_compute_aggregated_sar_value` already skips NaN cells per variable before
weighting. So once `rvlRadVelStd` is extracted with the right dims, and land
cells are NaN'd out in `rvlRadVel`/`rvlRadVelStd`, both effects propagate into
`collocation_results.nc` automatically as `sar_rvlRadVelStd` and NaN-free
`sar_rvlRadVel` land cells.

RVL extraction is already currents-only in this codebase (`_extract_rvl_grid_data`
is only called for `product_type="currents"`), so this change is inherently
scoped to currents recipes without needing a new flag.

## 3. Component changes

### a. `_extract_rvl_grid_data` — extract `rvlLandFlag` and `rvlRadVelStd`, mask, compute QA stats

File: `sar_validation/core/datatree_converter.py`.

Applies to both branches (grid, and flatten-to-points used by WV):

- Read `rvlLandFlag` the same way as the other RVL variables — through the
  existing `_swaths_to_grid` reshape helper in the grid branch, raveled in the
  points branch.
- Read `rvlRadVelStd` the same way, defaulting to an all-NaN array of the same
  shape if the raw file doesn't have it (mirroring the existing
  `rvlHeading`/`rvlIncidenceAngle` defaulting behavior).
- Before masking, compute:
  - `land_mask = (rvlLandFlag == 1)`
  - `land_pixel_count = land_mask.sum()`
  - `total_classified = count of cells where rvlLandFlag is not NaN` (the
    scene's classified footprint — excludes out-of-swath NaN cells)
  - `land_mean_radvel = nanmean(rvlRadVel[land_mask])` if `land_pixel_count > 0`,
    else NaN
- Mask: `rvlRadVel = where(land_mask, NaN, rvlRadVel)` and
  `rvlRadVelStd = where(land_mask, NaN, rvlRadVelStd)`. `rvlHeading` and
  `rvlIncidenceAngle` are **not** masked — they are geometry, not
  measurements, and remain valid over land.
- Add `rvlRadVelStd` as a new data variable on the output Dataset with the
  same dims as `rvlRadVel` (`("y","x")` for grid, `("point",)` for WV/points).
- Stamp QA attrs on the dataset (after `apply_cf_metadata`, so they aren't
  touched by its per-variable attribute merge):
  - `rvl_land_pixel_count` (int)
  - `rvl_land_pixel_fraction` (float, `land_pixel_count / total_classified`,
    NaN if `total_classified == 0`)
  - `rvl_land_mean_radvel` (float, NaN if `land_pixel_count == 0`)
- In the points branch (which loops over multiple imagette files per SAFE dir
  and concatenates into one `point` dataset), accumulate these counts/means
  across all files belonging to that scene and stamp them once on the final
  concatenated dataset.
- Log once per scene, only when `land_pixel_count > 0`:
  ```
  logger.warning(
      "scene %s: %d/%d RVL cells land-flagged (%.1f%%) — "
      "mean rvlRadVel over land = %.4f m/s (expected ~0)",
      safe_dir.name, land_pixel_count, total_classified,
      100 * land_pixel_fraction, land_mean_radvel,
  )
  ```
- Missing `rvlLandFlag` in the raw file (defensive case; not seen in current
  fixtures) → skip masking and QA-stat computation entirely, no crash, no log.

### b. Report page — `visualization.py`

New function `plot_rvl_land_qa(datatree)`:

- Walks `datatree["sar"].children.items()`, reads `node.to_dataset().attrs`,
  and keeps scenes where `rvl_land_pixel_count > 0`.
- Returns `None` if no scene in the run has land-flagged cells — no page is
  added in that case.
- Otherwise builds a single table figure: one row per land-affected scene,
  columns `scene`, `land pixel count`, `land %`, `mean rvlRadVel over land
  (m/s)`.

`validation_report()` calls this only when `recipe.config.variable ==
"currents"`, and — if it returns a figure — inserts it into `pdf_pages` right
after the collocation-diagnostics page.

## 4. Data flow (currents, after this change)

```
raw OCN file (rvlLandFlag, rvlRadVel, rvlRadVelStd, rvlHeading, rvlIncidenceAngle)
  → _extract_rvl_grid_data / flatten branch:
      - land_mask = rvlLandFlag == 1
      - QA stats computed from PRE-mask rvlRadVel over land_mask
      - rvlRadVel, rvlRadVelStd → NaN where land_mask
      - dataset attrs: rvl_land_pixel_count / _fraction / _mean_radvel
      - logger.warning(...) if land_pixel_count > 0
        ↓  datatree.nc  (SAR node carries rvlRadVel, rvlRadVelStd, rvlHeading,
                          rvlIncidenceAngle + QA attrs; land cells NaN in
                          rvlRadVel/rvlRadVelStd)
  → collocation (unchanged code): NaN cells skipped automatically in
    per-variable weighted aggregation → rvlRadVelStd column appears in
    collocation_results.nc alongside rvlRadVel, same as any other sar_* var
  → validation_report(): walks datatree["sar"] children, collects scenes with
    rvl_land_pixel_count > 0 → QA table page in the PDF (omitted if none)
```

## 5. Testing

**Unit (`test_datatree_converter.py`)**

- Synthetic `rvlLandFlag`/`rvlRadVel`/`rvlRadVelStd` arrays (both grid and
  points branch) verify land cells become NaN in `rvlRadVel`/`rvlRadVelStd`
  but not `rvlHeading`/`rvlIncidenceAngle`.
- `rvl_land_pixel_count` / `rvl_land_pixel_fraction` / `rvl_land_mean_radvel`
  attrs match a hand-computed expectation from the synthetic input.
- Zero-land input → `land_pixel_count == 0`, no masking applied, mean attr is
  NaN.
- Missing `rvlLandFlag` in the raw file → no crash, no masking, no QA attrs.
- `rvlRadVelStd` present in output dataset with correct dims for both grid and
  points branches.

**Integration**

- `convert_downloaded_data(product_type="currents")` over the real IW fixture
  with known land pixels
  (`data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00/S1_L2_OCN/S1C_IW_OCN__2SDV_20260601T020810_...4884.SAFE`,
  64407 land px found during design) yields a SAR node with NaN `rvlRadVel` at
  land cells and QA attrs matching the pre-mask land mean.
- The same over a land-free fixture yields `rvl_land_pixel_count == 0` and an
  unmodified `rvlRadVel`.

**Collocation (`test_collocation.py`)**

- A synthetic SAR grid/point dataset with `rvlRadVelStd` present asserts it
  appears as `sar_rvlRadVelStd` in `run_collocation`'s output — regression
  guard confirming the generic dims-based propagation picks it up without any
  variable-name special-casing.

**Report (`test_visualization.py`)**

- `plot_rvl_land_qa` returns `None` for a datatree with no land-flagged
  scenes.
- `plot_rvl_land_qa` returns a Figure with one row per land-affected scene for
  a datatree that has them.
- `validation_report()` includes/excludes the QA page accordingly for
  `variable="currents"` runs.

## 6. Scope boundaries (YAGNI)

- Wind and waves extraction paths are untouched — RVL extraction is already
  currents-only (`_extract_rvl_grid_data` is only invoked for
  `product_type="currents"`).
- No new CLI flags or recipe config — masking and QA reporting are
  unconditional, correctness-fixing behavior for currents recipes.
- `rvlRadVelStd` is extracted and propagated through to `collocation_results.nc`,
  but not added to `VARIABLE_PAIRS` / statistics / scatter-plots — it rides
  alongside `rvlRadVel` as QA data, not as a newly validated quantity.
- `rvlHeading` and `rvlIncidenceAngle` are not masked by the land flag — they
  are geometry, not measurements, and remain valid over land.
- No stale-`datatree.nc` / `collocation_results.nc` migration — existing files
  on disk are regenerated by re-running conversion, done manually by the user
  when needed.
