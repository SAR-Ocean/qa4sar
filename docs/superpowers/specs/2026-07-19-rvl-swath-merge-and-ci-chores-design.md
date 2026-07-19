# RVL sub-swath merge fix + CI chores — Design

**Date:** 2026-07-19
**Status:** Approved (pending user review of this document)
**Packaging:** Two separate PRs.

## Problem

### 1. SAR currents field plotted "inland" (correctness bug)

Geographic plots for currents recipes (`recipes/currents_uswestcoast.yaml`,
`recipes/currents_useastcoast_example.yaml`) show the SAR RVL field smeared
over land, mismatching the coastline and the HF-radar data. Most visible on
the US west coast (see
`data/2026-06-01-000000-2026-06-01-235959_-130.00_-115.00_33.00_48.00/validation_report.pdf`).

**Root cause (verified on real data).** IW/EW RVL arrays are 3-D:
`(rvlAzSize, rvlRaSize, rvlSwath)` — e.g. `(233, 131, 3)` for IW. The helper
`_swaths_to_grid` in `sar_validation/core/datatree_converter.py` merges the
swath axis with `arr.reshape(arr.shape[0], -1)`. NumPy's C-order reshape
makes the last axis vary fastest, so the merged columns **interleave**
sub-swaths (`ra0-sw0, ra0-sw1, ra0-sw2, ra1-sw0, …`) instead of laying them
side by side. Measured on a real west-coast IW file, adjacent-column
longitude steps after the reshape are `+0.88°, +0.93°, −1.79°` (a zig-zag
across the ~250 km swath); a proper per-swath concatenation gives smooth
`0.009°` steps. `pcolormesh` draws quadrilaterals between grid neighbours,
so the zig-zag smears the field across the swath's whole bounding box —
including over land.

**Why only currents recipes.** Verified in the same IW OCN measurement
file: `owiLon`/`owiWindSpeed` are already 2-D `(owiAzSize, owiRaSize)` — the
OWI component mosaics sub-swaths into one seamless grid at product level —
and OSW variables carry no swath grid at all. Only RVL keeps a per-sub-swath
third axis, so only the RVL extraction path calls `_swaths_to_grid`. Wind
and wave plots never execute the buggy code.

### 2. Cartopy deprecation warnings on Matplotlib 3.11 (CI)

The CI runner installs Matplotlib 3.11; cartopy's tick formatter trips
~3000 deprecation warnings there (third-party code). This becomes a real
breakage when Matplotlib 3.13 releases, presumably fixed by a cartopy
update before then.

### 3. Natural Earth coastline download dominates CI runtime

The 10m Natural Earth coastline/land download in CI works but is slow —
most of the test job's runtime — and remains a flake risk.

## Design

### PR 1 — Fix RVL sub-swath merge

**Core change** (one function, `datatree_converter.py`): replace the
interleaving reshape in `_swaths_to_grid` with side-by-side concatenation
along the range axis:

```python
np.concatenate([arr[:, :, k] for k in range(arr.shape[2])], axis=1)
```

Sub-swath index order (iw1→iw3, ew1→ew5) matches ground-range order in the
product (verified: mean longitudes −122.6, −121.7, −120.9 for k = 0, 1, 2 on
a real file). The helper is applied uniformly to `rvlRadVel`,
`rvlRadVelStd`, lon, lat, heading, incidence, and `rvlLandFlag`, so the one
change keeps the whole grid internally consistent. Adjacent sub-swaths
overlap slightly at seams; `pcolormesh` tolerates that. 2-D (SM) inputs
pass through unchanged, as before.

**Tests (TDD).** Extend the existing RVL extraction tests (check first;
don't duplicate) with a unit test feeding a synthetic `(az, ra, swath)`
array whose swaths occupy distinct longitude blocks, asserting the merged
grid is swath-contiguous: within-row longitude steps stay small and
monotonic (no zig-zag).

**Verification.**
- Read `collocation.py` to confirm the layer-vs-layer / cell-averaging path
  treats grid cells as independent points (order-independent). Expectation:
  collocation numbers are identical before/after the fix, up to float
  round-off where an aggregation sums in array order (the fix reorders
  cells). Prove it by regenerating the west-coast run and diffing
  `validation_statistics_rvlRadVel_vs_rvlRadVel_projection.csv` against the
  cached copy.
- Regenerate the west-coast `validation_report.pdf` from the cached SAFE
  data and visually confirm the SAR field hugs the coastline. The cached
  `datatree.nc` holds scrambled grids, so the datatree must be rebuilt
  (delete/ignore the cached `datatree.nc`; SAFE downloads are reusable).
- Note the caveat in the PR text: existing `datatree.nc` caches produced
  before the fix contain interleaved RVL grids and must be regenerated.

### PR 2 — CI chores

**Coastline cache** (`.github/workflows/ci.yml`):
- `actions/cache` on `~/.local/share/cartopy` with a static key
  (e.g. `cartopy-natural-earth-10m-v1`).
- A pre-fetch step that downloads the 10m coastline + land features before
  pytest runs (fast no-op when the cache is warm). This removes the
  dominant chunk of test-job runtime on warm-cache runs and shrinks the
  flake window to cold-cache runs only.

**Warnings filter** (`pyproject.toml`, `[tool.pytest.ini_options]`):
- Add a narrowly-targeted `filterwarnings` ignore for the cartopy tick
  formatter deprecation. Pull the exact warning text from the CI log
  (`gh run view --log`) so the filter matches only that message/module —
  no blanket `MatplotlibDeprecationWarning` ignore.

**Matplotlib pin** (CI install step only; package metadata untouched):
- `pip install -e .[dev] "matplotlib<3.13"` with a comment stating the
  removal condition: cartopy has released a fix for the deprecated tick
  formatter call.

## Out of scope

- Per-sub-swath datatree schema (approach B) and plot-side quad masking
  (approach C) — rejected in favour of the minimal merge-order fix.
- The two deferred items from PR #8 follow-ups (NOAA urlretrieve timeout,
  collocation-log plot flag).
- RVL heading/projection interpretation questions (separate notebook
  investigation).

## Success criteria

- West-coast and east-coast currents reports show the SAR field aligned
  with the coastline and HF-radar data.
- Collocation statistics unchanged by the fix (CSV diff clean).
- All tests pass; new swath-contiguity test fails on the old reshape.
- CI test job runtime drops substantially on warm cache; CI logs free of
  the cartopy deprecation spam; CI immune to Matplotlib 3.13 until cartopy
  ships a fix.
