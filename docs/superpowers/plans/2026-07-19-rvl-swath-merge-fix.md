# RVL Sub-Swath Merge Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the RVL sub-swath merge in `_swaths_to_grid` so IW/EW currents grids are geometrically contiguous, eliminating the "SAR field smeared inland" artifact in currents geographic plots.

**Architecture:** One-function fix in `sar_validation/core/datatree_converter.py`. The 3-D RVL arrays `(rvlAzSize, rvlRaSize, rvlSwath)` are currently merged to 2-D with `arr.reshape(arr.shape[0], -1)`, which (C-order, last axis fastest) **interleaves** sub-swath columns. The fix concatenates sub-swaths side by side along the range axis instead. Everything downstream (pcolormesh plotting, distance-based collocation) consumes the same `(y, x)` grid, so no other code changes.

**Tech Stack:** Python 3.10 venv at `.venv/`, pytest, xarray, numpy. Spec: `docs/superpowers/specs/2026-07-19-rvl-swath-merge-and-ci-chores-design.md`.

## Global Constraints

- Run everything with the project venv: `.venv/bin/pytest`, `.venv/bin/python`, `.venv/bin/sar-validate`.
- `ruff check .` must pass (ruff config in `pyproject.toml`: E, F, I; line length 120).
- Baseline: 414 tests passing, 0 warnings locally. Do not break any.
- `data/` is gitignored — Task 2's regenerated artifacts are never committed.
- The datatree schema is unchanged: RVL grids stay one 2-D `(y, x)` dataset per scene; only the column ordering changes.

---

### Task 1: Fix the sub-swath merge order in `_swaths_to_grid`

**Files:**
- Modify: `sar_validation/core/datatree_converter.py:1593-1600` (the `_swaths_to_grid` helper inside `_extract_rvl_grid_data`)
- Test: `tests/test_datatree_converter.py` (class `TestExtractRvlGridData`, near line 903)

**Interfaces:**
- Consumes: `DataTreeConverter._extract_rvl_grid_data(measurement_dir, safe_dir, flatten_to_points=False)` — existing private classmethod, unchanged signature.
- Produces: same method, now returning grids whose `x` axis is swath-contiguous: columns `[0, nx)` are sub-swath 0, `[nx, 2*nx)` sub-swath 1, etc. Task 2 relies on this making per-row longitudes smooth.

**Background for the implementer:** IW/EW OCN products store RVL variables as 3-D `(rvlAzSize, rvlRaSize, rvlSwath)` — e.g. `(233, 131, 3)` for IW. Sub-swath index order (iw1→iw3, ew1→ew5) matches ground-range order, verified on real data (mean longitudes −122.6, −121.7, −120.9 for swath k = 0, 1, 2). The existing test `test_multiswath_reshaped_to_grid_keeps_all_swaths` only checks shape and cell count, both of which are identical under interleaved and contiguous ordering — that's why the bug survived. The existing land-flag tests compute expectations from full azimuth rows (order-independent within a row), so they stay green under the fix without edits.

- [ ] **Step 1: Write the failing test**

Add to class `TestExtractRvlGridData` in `tests/test_datatree_converter.py` (after `test_multiswath_reshaped_to_grid_keeps_all_swaths`). It builds its own measurement file (not `_make_ocn_safe`, whose lon values are random and can't distinguish orderings) with structured longitudes: sub-swath k spans a distinct block, ascending in range.

```python
    def test_multiswath_merge_is_swath_contiguous(self, tmp_path):
        # Sub-swath k occupies its own longitude block [k, k+0.03], ascending
        # in range within the swath — mirroring real IW/EW products, where
        # sub-swath index order matches ground-range order. The merged grid
        # must lay swaths side by side along x, not interleave their columns
        # (the interleave smears pcolormesh quads across the whole swath).
        ny, nx, ns = 2, 4, 3
        rdims = ("rvlAzSize", "rvlRaSize", "rvlSwath")
        lon = np.zeros((ny, nx, ns), dtype="float32")
        radvel = np.zeros((ny, nx, ns), dtype="float32")
        for k in range(ns):
            lon[:, :, k] = k + 0.01 * np.arange(nx, dtype="float32")
            radvel[:, :, k] = k
        lat = np.full((ny, nx, ns), 50.0, dtype="float32")

        safe = tmp_path / "S1A_EW_OCN.SAFE"
        meas = safe / "measurement"
        meas.mkdir(parents=True)
        ds_raw = xr.Dataset(
            {
                "rvlRadVel": (rdims, radvel),
                "rvlLon": (rdims, lon),
                "rvlLat": (rdims, lat),
            },
            attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"},
        )
        ds_raw.to_netcdf(
            meas / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
        )

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        merged_lon = ds["lon"].values
        merged_radvel = ds["rvlRadVel"].values
        assert merged_lon.shape == (ny, nx * ns)

        # Longitudes increase monotonically across each row — the interleaved
        # ordering zig-zags (0.00, 1.00, 2.00, 0.01, ...) and fails this.
        assert (np.diff(merged_lon, axis=1) > 0).all()

        # Values travel with their coordinates: columns [k*nx, (k+1)*nx) are
        # exactly sub-swath k.
        for k in range(ns):
            assert (merged_radvel[:, k * nx:(k + 1) * nx] == k).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_datatree_converter.py::TestExtractRvlGridData::test_multiswath_merge_is_swath_contiguous -v`
Expected: FAIL on `assert (np.diff(merged_lon, axis=1) > 0).all()` (the interleaved reshape produces negative lon steps).

- [ ] **Step 3: Fix `_swaths_to_grid`**

In `sar_validation/core/datatree_converter.py`, replace the helper (currently at lines 1598-1600):

```python
                def _swaths_to_grid(arr):
                    arr = np.asarray(arr)
                    return arr.reshape(arr.shape[0], -1) if arr.ndim == 3 else arr
```

with:

```python
                def _swaths_to_grid(arr):
                    # Lay sub-swaths side by side along the range axis, in
                    # swath-index order (== ground-range order: iw1→iw3,
                    # ew1→ew5). A plain C-order reshape would interleave
                    # sub-swath columns instead, scrambling grid adjacency
                    # and smearing pcolormesh quads across the whole swath.
                    arr = np.asarray(arr)
                    if arr.ndim != 3:
                        return arr
                    return np.concatenate(
                        [arr[:, :, k] for k in range(arr.shape[2])], axis=1
                    )
```

Also update the stale wording in the comment block directly above the helper (lines 1593-1597): it says "Merge the swath axis into the range axis" — keep the rationale about not dropping swaths, but make sure it doesn't describe the old reshape. Suggested replacement for that block:

```python
                # RVL is 3-D (rvlAzSize, rvlRaSize, rvlSwath) for multi-swath
                # modes (IW/EW). Concatenate the sub-swaths side by side along
                # the range axis so the grid keeps EVERY sub-swath — slicing
                # [:, :, 0] would silently drop all but the first swath
                # (4 of 5 for EW, 2 of 3 for IW). Single-swath products (SM)
                # are already 2-D and pass through.
```

- [ ] **Step 4: Run the new test and the whole RVL test class**

Run: `.venv/bin/pytest tests/test_datatree_converter.py::TestExtractRvlGridData -v`
Expected: all PASS (including the pre-existing land-flag and shape tests — their expectations are row-based and order-independent).

- [ ] **Step 5: Run the full suite and lint**

Run: `.venv/bin/pytest`
Expected: 415 passed (414 baseline + 1 new), 0 warnings.

Run: `.venv/bin/ruff check .`
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "fix: concatenate RVL sub-swaths along range axis instead of interleaving

The C-order reshape merged (az, ra, swath) grids with the swath axis
varying fastest, interleaving sub-swath columns. pcolormesh draws quads
between grid neighbours, so the zig-zag (measured: +0.88deg, +0.93deg,
-1.79deg adjacent-column lon steps on a real IW file) smeared the RVL
field across the swath bounding box, including over land. Side-by-side
concatenation restores contiguous geometry (0.009deg steps)."
```

---

### Task 2: Regenerate the west-coast currents run and verify

**Files:**
- No source changes. Operates on `data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00/` (gitignored) via `recipes/currents_uswestcoast.yaml`.

**Interfaces:**
- Consumes: the fixed `_extract_rvl_grid_data` from Task 1 (via `sar-validate --plot`, which cascades convert → collocate → stats → plot).
- Produces: verification evidence only — nothing for later tasks.

**Background for the implementer:** the pipeline caches every step: downloads are skipped when `download_metadata.json` has no errors, conversion is skipped when `datatree.nc` exists, collocation when `collocation_results.nc` exists. The cached `datatree.nc` was built with the interleaved merge, so it must be deleted to force reconversion; the downloaded SAFE/HF-radar/in-situ files are reused as-is (no network needed for SAR; the run may still touch the network for nothing since Step 1 is skipped). If you are working in a worktree, the data lives in the main checkout — pass `--output-dir` with the absolute path of the run directory so the worktree run reuses it (`output_dir` in the recipe/CLI *is* the run directory, not its parent).

- [ ] **Step 1: Preserve the pre-fix statistics baseline**

```bash
RUN_DIR=/home/chvan0015/git/sar-l2-validation-toolbox/data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00
cp "$RUN_DIR/validation_statistics_rvlRadVel_vs_rvlRadVel_projection.csv" /tmp/wc_stats_before.csv
```

- [ ] **Step 2: Delete stale cached products (NOT the downloads)**

```bash
rm "$RUN_DIR/datatree.nc" "$RUN_DIR/collocation_results.nc"
```

Do not touch `S1_L2_OCN/`, `hfr_noaa/`, `copernicus_insitu/`, or `download_metadata.json`.

- [ ] **Step 3: Re-run the pipeline through plotting**

```bash
.venv/bin/sar-validate --recipe recipes/currents_uswestcoast.yaml --plot --output-dir "$RUN_DIR"
```

Expected console output: "Step 1 skipped — data already present …", then Step 2 (conversion), Step 3 (collocation), stats, plots, and a regenerated `validation_report.pdf`. Takes a few minutes.

- [ ] **Step 4: Assert the rebuilt grids are contiguous**

```bash
.venv/bin/python - <<'EOF'
import numpy as np
import xarray as xr

run = "/home/chvan0015/git/sar-l2-validation-toolbox/data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00"
dt = xr.open_datatree(f"{run}/datatree.nc", engine="netcdf4")
checked = 0
for name, node in dt["sar"].children.items():
    ds = node.to_dataset()
    # Only multi-swath 2-D grids (IW/EW); WV point scenes have no y/x dims.
    if "lon" in ds.coords and ds["lon"].ndim == 2 and ds.sizes.get("x", 0) > 13:
        lon = ds["lon"].values
        step = float(np.nanmax(np.abs(np.diff(lon, axis=1))))
        print(f"{name}: max adjacent-column lon step = {step:.3f} deg")
        # Interleaved grids showed ~1.8 deg jumps; contiguous ones ~0.01
        # (sub-swath seam overlap can reach a few hundredths).
        assert step < 0.2, name
        checked += 1
print(f"OK - {checked} IW/EW grids contiguous")
assert checked > 0
EOF
```

Expected: every IW scene prints a step well under 0.2° and the script ends with `OK`.

- [ ] **Step 5: Confirm collocation statistics are unchanged**

```bash
.venv/bin/python - <<'EOF'
import numpy as np
import pandas as pd

run = "/home/chvan0015/git/sar-l2-validation-toolbox/data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00"
a = pd.read_csv("/tmp/wc_stats_before.csv")
b = pd.read_csv(f"{run}/validation_statistics_rvlRadVel_vs_rvlRadVel_projection.csv")
assert a.shape == b.shape, (a.shape, b.shape)
for c in a.columns:
    if np.issubdtype(a[c].dtype, np.number):
        # Collocation selects cells by distance, not grid order, so results
        # should match to float round-off (aggregation sums reordered).
        assert np.allclose(a[c], b[c], rtol=1e-6, atol=1e-9, equal_nan=True), c
    else:
        assert (a[c].fillna("") == b[c].fillna("")).all(), c
print("statistics unchanged")
EOF
```

Expected: `statistics unchanged`. If a numeric column differs beyond round-off, STOP — that contradicts the order-independence analysis; investigate before proceeding (do not loosen the tolerance to make it pass).

- [ ] **Step 6: Visual check of the report**

Open `$RUN_DIR/validation_report.pdf` (and/or the PNGs in `$RUN_DIR/plots/`, e.g. `rvlRadVel_vs_rvlRadVel_projection_geographic_layer_vs_layer.png`). Confirm: the IW SAR field now renders as a coherent swath offshore, its edge follows the coastline, and it no longer bleeds over land; HF-radar points sit on/inside the SAR coverage. Report the observation to the human partner with the file path — this is the acceptance evidence for the original bug report.

- [ ] **Step 7 (if east-coast cached data is present): repeat for the east coast**

The east-coast recipe is `recipes/currents_useastcoast_example.yaml`. Check for a matching run dir (bounds −77…−65, 33…45):

```bash
ls -d /home/chvan0015/git/sar-l2-validation-toolbox/data/*-77.00_-65.00_33.00_45.00 2>/dev/null
```

If it exists, repeat Steps 1-6 with `RUN_DIR` pointed at it and the east-coast recipe (baseline copy to `/tmp/ec_stats_before.csv`). If it doesn't exist, skip — do not download fresh data for this.

- [ ] **Step 8: Note the cache-invalidation caveat for the PR description**

No commit in this task (all artifacts are gitignored). Record for the PR text (used by the finishing task/PR author): *"`datatree.nc` files produced before this fix contain interleaved RVL grids — delete `datatree.nc` and `collocation_results*.nc` in existing run directories and re-run `--plot` to regenerate them; downloaded source data is reusable."*

---

## Self-review checklist (run after writing code)

- Spec coverage: merge fix (Task 1), swath-contiguity TDD test (Task 1), collocation invariance proof (Task 2 Step 5), west-coast visual regeneration (Task 2 Steps 3-6), east-coast (Task 2 Step 7), cache caveat documented (Task 2 Step 8). CI items are deliberately in the separate plan `2026-07-19-ci-cartopy-cache-mpl-guard.md`.
