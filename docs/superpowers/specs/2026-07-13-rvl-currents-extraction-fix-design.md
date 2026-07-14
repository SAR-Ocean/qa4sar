# Fix RVL / currents extraction (WV + IW/EW/SM) — Design

**Date:** 2026-07-13
**Status:** Approved (pending spec review)
**Type:** Bugfix + small enhancement

## 1. Problem

A `currents` recipe (e.g. `recipes/currents_test.yaml`) produces a
`datatree.nc` whose SAR nodes contain `owiWindSpeed` / `owiWindDirection`
(wind) instead of the RVL radial-velocity coordinates that currents
validation needs. Two independent defects cause this, plus a structural
inconsistency that would keep currents validation broken even once the
`datatree.nc` looks right.

### 1a. Multi-swath RVL extraction crashes, then silently falls back to wind

In `DataTreeConverter._extract_rvl_grid_data`
(`sar_validation/core/datatree_converter.py`), the 2-D-grid branch handles
IW/EW/SM products. For multi-swath modes `rvlRadVel` / `rvlLat` / `rvlLon`
are 3-D with dims `(rvlAzSize, rvlRaSize, rvlSwath)`. The code slices
`[:, :, 0]` to reduce each array to 2-D, but then reads the dimension **names**
from the *un-sliced* variable:

```python
dims = ds_raw["rvlLat"].dims  # ('rvlAzSize', 'rvlRaSize', 'rvlSwath') — 3 names
...
coords = {"lon": (dims, rvl_lons), ...}  # rvl_lons is 2-D → mismatch
```

Building a coordinate with a 3-tuple of dim names over 2-D data makes xarray
raise `Could not convert tuple of form (dims, data[, attrs, encoding])`. The
exception is swallowed at DEBUG level and the function returns `None`.

The caller `_from_sar_l2_ocn_iw_safe`, for `product_type="currents"`, then
**falls back to OWI wind extraction** — so the SAR node ends up holding
`owiWindSpeed` with no error surfaced to the user. This is the observed
symptom.

**Second, latent defect — the `[:, :, 0]` slice discards sub-swaths.** Even if
the dim-name mismatch were fixed by keeping the slice, taking swath index 0
keeps only the first sub-swath and drops the rest, and those are not padding.
Confirmed against real data
(`data/2026-06-20-180000-2026-06-20-230000_-20.00_0.00_35.00_60.00/`): the EW
`-ocn-` file's `rvlRadVel` is `(382, 114, 5)` — **5 sub-swaths (EW1–EW5)**, each
carrying tens of thousands of distinct finite values (swath finite counts
41,143 / 33,379 / 41,907 / 41,907 / 35,777; value ranges differ per swath).
Slicing `[:, :, 0]` therefore throws away ~194k of ~228k valid radial-velocity
measurements (4 of 5 swaths for EW; 2 of 3 for IW). The fix must retain all
sub-swaths (see §3a). The `rvl*` variables are otherwise fully present (27
`rvl*` variables including `rvlRadVel`), so this is purely a shape-handling bug,
not missing data.

### 1b. WV conversion ignores `product_type` (always extracts waves)

`from_sar_l2_ocn_safe` routes WV-mode SAFE directories to
`from_sar_l2_ocn_wv_safe`, which hardcodes `oswHs` (significant wave height)
regardless of `product_type`. A currents run over WV data therefore extracts
wave height, never radial velocity — even though each WV imagette OCN file
**does** contain `rvlRadVel` (verified: dims `(rvlAzSize, rvlRaSize)`, shape
13×13 per imagette).

### 1c. RVL lives in two disconnected paths

Currents validation reads RVL from two places that do not agree:

- **Datatree conversion** — writes `datatree.nc` (buggy, per 1a/1b).
- **Collocation** — `run_collocation` ignores the datatree's SAR node for
  currents and separately re-scans **WV-only** SAFE dirs from disk via
  `_load_rvl_for_collocation`, adding `{scene}_rvl` point nodes.

Consequences today:

- WV currents partly works *in collocation* (via the disk re-scan) but never
  appears correctly in `datatree.nc`.
- **IW/EW/SM currents get no RVL at all in collocation** (the re-scan is
  WV-only), so they would silently collocate against the wrong OWI node.

"Fix the datatree" and "make currents actually validate" are therefore the
same problem.

## 2. Approach (chosen: A — datatree is the single source of truth)

Fix conversion so `datatree.nc` carries correct RVL for **both** WV
(13×13 imagettes → points) and IW/EW/SM (2-D `y,x` grid), driven by
`product_type=currents`; then **delete** the separate WV-only disk re-scan
and let collocation consume RVL straight from the datatree like every other
node.

Rejected alternative — **B (minimal)**: only fix conversion to *display* RVL
correctly and keep the disk re-scan. Rejected because it preserves the
duplication and leaves IW/EW/SM currents broken in collocation, so the
datatree would look right while validation still silently misbehaves.

Approach A is barely more work than B, removes a whole parallel code path,
and is the only option that makes IW/EW/SM currents validate. Both leave
wind and waves behavior untouched.

## 3. Component changes

### a. `_extract_rvl_grid_data` — fix the multi-swath crash

File: `sar_validation/core/datatree_converter.py` (2-D-grid branch,
`flatten_to_points=False`).

Do **not** slice `[:, :, 0]`. For 3-D multi-swath arrays, **merge the
`rvlSwath` axis into the range axis** to form a single 2-D `(y, x)` grid that
retains every sub-swath:

```
(rvlAzSize, rvlRaSize, rvlSwath) → (rvlAzSize, rvlRaSize * rvlSwath) = (y, x)
```

Apply the identical reshape to `rvlRadVel`, `rvlLat`, `rvlLon`, `rvlHeading`,
and `rvlIncidenceAngle` so all arrays stay cell-aligned. Because the grid
collocation path matches each cell by its own 2-D `lon`/`lat` (haversine), not
by regular grid spacing, the swath overlaps and coordinate discontinuities
introduced by the merge are harmless — each cell is collocated independently.

Build the Dataset with dim names **`("y", "x")`** to mirror the OWI grid from
`_extract_owi_grid_data`, keeping the SAR grid model uniform so `is_wv_mode`
detection (`"point" in dims and "y" not in dims`) and the grid collocation path
treat RVL and OWI grids identically. `lon` / `lat` are 2-D coordinates over
`("y", "x")`. The already-2-D single-swath case (e.g. SM) passes through
unchanged (no swath axis to merge).

Rejected alternative — flatten all swaths to a `point` dataset: also preserves
the data, but a `point` node routes IW/EW into the WV footprint-anchored
collocation path (≈14 km radius, designed for sparse ~200 km-spaced imagettes),
which is wrong for dense grids. The reshape keeps IW/EW/SM on the correct grid
path.

### b. `_from_sar_l2_ocn_iw_safe` — remove the currents → OWI fallback

File: `sar_validation/core/datatree_converter.py`.

For `product_type="currents"`, if RVL extraction returns `None`, **return
`None` and emit a WARNING** (e.g. `"scene <name>: no RVL/currents data —
skipping"`) instead of falling back to `_extract_owi_grid_data`.
`convert_downloaded_data` already drops `None` SAR nodes, so the scene simply
does not appear in the datatree. A currents run must never silently produce
wind data. The `wind` and `waves` branches (and their existing fallbacks) are
left unchanged.

### c. `from_sar_l2_ocn_safe` — make the WV branch product-type-aware

File: `sar_validation/core/datatree_converter.py`.

Route the WV branch on `product_type`:

- WV + `currents` → `_extract_rvl_from_wv_safe` (existing helper: extracts
  13×13 imagette RVL grids, flattened to a `point` dataset, tagged
  `swath_mode="WV"`).
- WV + `waves` / `wind` → existing `from_sar_l2_ocn_wv_safe` (oswHs).

Non-currents WV behavior is unchanged.

### d. Collocation — single source of truth

File: `sar_validation/core/collocation.py`.

Delete `_load_rvl_for_collocation` and its call in `run_collocation`.
Collocation then reads RVL directly from the datatree SAR nodes:

- WV RVL arrives as `point` nodes → handled by the existing WV
  (SAR-footprint-anchored) collocation path.
- IW/EW/SM RVL arrives as `y,x` grid nodes → handled by the existing grid
  collocation path, including the RVL → radial-velocity projection already
  present there (projects in-situ `EWCT`/`NSCT` onto `rvlHeading` to produce
  `rvlRadVel_projection`).

This also fixes IW/EW/SM currents, which the WV-only re-scan never handled.

## 4. Data flow (currents, after fix)

```
recipe.variable = currents  →  product_type = currents
  SAFE is WV        → _extract_rvl_from_wv_safe → point node (rvlRadVel, rvlHeading, …)
  SAFE is IW/EW/SM  → _extract_rvl_grid_data   → (y, x) grid node (rvlRadVel, rvlHeading, …)
  neither yields RVL → node skipped + WARNING (never OWI)
        ↓  datatree.nc  (RVL only)
  collocation reads SAR nodes directly → projects EWCT/NSCT → rvlRadVel_projection
```

## 5. Testing

**Unit tests**

- 3-D `(rvlAzSize, rvlRaSize, rvlSwath)` RVL array → `(y, x)` grid node with
  `rvlRadVel` present (regression guard for the crash in 1a).
- **All sub-swaths retained**: a 3-D input with `S` swaths yields a grid whose
  `x` size equals `rvlRaSize * S`, and the count of finite `rvlRadVel` cells
  equals the sum across all input swaths (regression guard against the
  `[:, :, 0]` data-loss defect).
- Single-swath 2-D RVL array still produces a valid `(y, x)` grid node.
- WV SAFE + `product_type="currents"` → `point` node with `rvlRadVel`.
- Currents extraction with no RVL available → returns `None` and logs a
  warning; **no** `owi*` variables produced.

**Integration tests**

- `convert_downloaded_data(product_type="currents")` over the existing EW dir
  (`data/2026-06-20-180000-…`) yields SAR nodes containing `rvlRadVel`, not
  `owiWindSpeed`.
- The same over a WV dir yields RVL `point` nodes.
- `run_collocation` runs end-to-end for currents with
  `_load_rvl_for_collocation` removed (RVL sourced from the datatree).

Fixtures already exist on disk: an EW currents dir and a WV dir under `data/`.

## 6. Scope boundaries (YAGNI)

- Wind and waves extraction and their fallbacks are unchanged.
- No change to the HF-radar Phase 3 surface-current projection design; this
  work only makes the SAR-side RVL input correct so that effort has valid data.
- **No stale-`datatree.nc` migration or regeneration code.** Existing
  `datatree.nc` files on disk are regenerated by re-running conversion, done
  manually by the user when needed.

## 7. Open items to resolve during implementation

- Confirm the WV point-collocation path computes `rvlRadVel_projection`
  (the grid path does; the WV path is to be verified).
- Confirm which node attributes (`grid_shape`, `measurement_type`, …) any
  report / plot code reads for RVL nodes, so the fixed extractors set them
  consistently.
