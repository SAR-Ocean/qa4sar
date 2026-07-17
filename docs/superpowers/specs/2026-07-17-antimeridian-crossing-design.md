# Antimeridian (180°) crossing support — design

## Problem

`recipes/waves_pacific.yaml` requests a bbox spanning the Pacific
(nominally 135°E to 120°W). There is currently no way to express this
correctly:

- `--min-lon 135 --max-lon 240` ("0–360 style") triggers repeated
  "exceeding dataset coordinates" warnings from the Copernicus downloaders,
  and `data/2026-07-02-000000-2026-07-03-000000_135.00_240.00_-15.00_30.00/validation_report.pdf`
  shows the altimeter data cut off exactly at 180°E.
- `--min-lon 135 --max-lon -120` ("wraps through 180°") fails outright,
  since every consumer of `min_lon`/`max_lon` in the codebase assumes
  `min_lon < max_lon`.

### Root cause

Investigated `copernicusmarine.subset()` (the API behind the altimeter and
in-situ downloaders): its docstring says longitude values are "transposed to
the interval [-180, 360)", but the actual datasets (altimeter L3, in-situ)
are stored on a native **-180°..180° grid**. Requesting
`minimum_longitude=135, maximum_longitude=240` doesn't wrap around the
antimeridian — it asks for a contiguous slice that runs off the edge of the
real data, so everything past 180° is silently missing and a warning is
raised for the out-of-bounds portion. Genuine antimeridian crossing
requires **splitting the query into two non-crossing windows**
(`[135, 180]` and `[-180, -120]`) and merging the results — a pattern that
differs per downloader:

| Downloader family | Mechanism | Crossing behavior today |
|---|---|---|
| Altimeter, in-situ | `copernicusmarine.subset()` | Slice runs off the -180..180 grid edge |
| SAR | CDSE OData WKT polygon query | Malformed/self-intersecting polygon for `min_lon > max_lon` |
| Scatterometer | EUMDAC bbox search (`"min_lon,min_lat,max_lon,max_lat"`) | Same — assumes `min_lon < max_lon` |
| HF radar (Copernicus), HF radar historical, NOAA HF radar (ERDDAP) | Named-region matching against a fixed `_REGIONS`/`_BBOXES` table | Region-overlap math breaks when `min_lon > max_lon` |
| Radiometer (RSS) | Downloads whole global 0.25° grid, crops later in `datatree_converter.py` | Crop mask is `lon >= min_lon & lon <= max_lon` (AND) — wrong for a wrapped range |

None of these currently validate `min_lon`/`max_lon` order — `GeographicBounds`
([`recipe.py:29`](../../../sar_validation/core/recipe.py#L29)) is a plain
dataclass with no `__post_init__`, so there's nothing to relax; every
consumer just needs to stop *assuming* `min_lon < max_lon`.

Two other spots assume a simple (non-crossing) band and need a dateline-aware
version:

- `datatree_converter.py:85`, the point-data bounds-crop mask
  (`(lon >= min_lon - buf) & (lon <= max_lon + buf) & ...`), used for
  radiometer and other point-geometry datasets after ingest.
- `visualization.py:1540`, `ax.set_extent([min_lon, max_lon, min_lat,
  max_lat], ...)` for the collocation-diagnostics geographic map — this is
  what produced the 180°-cutoff appearance in the Pacific run's PDF, since
  the underlying data was already missing past 180° (see above) rather than
  a plotting-only bug, but the map extent itself also needs to handle a
  genuinely-crossing bbox once the data is fixed.

## Convention

`GeographicBounds.min_lon > max_lon` means **"the box wraps through 180°"**
— e.g. `min_lon: 135, max_lon: -120` for the Pacific recipe (the exact form
that currently fails outright). This matches common GIS/OGC bbox convention,
keeps all stored bounds in -180°..180°, and requires no new validation logic
(no `min < max` invariant is currently enforced anywhere, so there's nothing
to loosen). Non-crossing boxes — the overwhelming majority of recipes — are
completely unaffected; this is a purely additive capability.

Scope: **all** downloaders get real antimeridian support (not just the
altimeter/in-situ/SAR sources `waves_pacific.yaml` happens to use), so any
future Pacific-crossing recipe works regardless of which sources it
combines.

## Design

### 1. Shared splitting helper

New helper in [`downloaders/base.py`](../../../sar_validation/downloaders/base.py):

```python
def split_antimeridian_bbox(min_lon: float, max_lon: float) -> list[tuple[float, float]]:
    """Split a bbox into 1 or 2 non-crossing (lon_min, lon_max) windows.

    Returns the box unchanged if min_lon <= max_lon. If min_lon > max_lon,
    treats it as wrapping through 180 deg and returns the two windows
    [min_lon, 180] and [-180, max_lon].
    """
    if min_lon <= max_lon:
        return [(min_lon, max_lon)]
    return [(min_lon, 180.0), (-180.0, max_lon)]
```

Every downloader's public entry point (`download()`, and SAR's `query()`)
is restructured as: split → run the existing single-window logic once per
window → merge results. The single-window logic itself is untouched
wherever possible (extracted into a private per-window method only where
the current method mixes "resolve the query" with "loop and merge").

### 2. Per-downloader integration

- **Altimeter, in-situ** (`copernicusmarine.subset()`-based): call
  `subset()` once per window. Altimeter currently derives one
  `output_filename` per `dataset_id` + date
  ([`altimeter_downloader.py:189`](../../../sar_validation/downloaders/altimeter_downloader.py#L189));
  when the bbox is split, the two calls would silently overwrite each
  other's file, so the filename gets a window-index suffix (e.g. `_w0`/
  `_w1`) *only* when actually split — the non-crossing case keeps today's
  filenames exactly, so existing runs/tests are unaffected.
- **SAR** (CDSE OData WKT polygon,
  [`sar_downloader.py:88`](../../../sar_validation/downloaders/sar_downloader.py#L88)):
  query once per window, concatenate the result DataFrames, de-duplicate on
  `Id` (a scene whose swath straddles 180° could otherwise be returned by
  both window queries and downloaded twice).
- **Scatterometer** (EUMDAC bbox search,
  [`scatterometer_downloader.py:91`](../../../sar_validation/downloaders/scatterometer_downloader.py#L91)):
  same pattern — search once per window, de-duplicate on `product_id`.
- **HF radar, HF radar historical, NOAA HF radar** (named-region matching):
  loop over windows, but a window that resolves to "no covering region"
  (`ValueError` from `resolve_hfr_region`/`_match_region`) is **skipped**,
  not treated as a fatal error — known HF-radar regions are all localized
  coastal areas that never straddle the dateline themselves, so for a real
  crossing recipe at most one window will ever resolve to an actual region.
  If *neither* window resolves, the existing "no region" error still
  propagates.
- **Radiometer** (global-grid download): no change at download time — it
  already fetches the whole file unconditionally and crops afterward.

### 3. Dateline-aware cropping and plotting

- `datatree_converter.py`'s bounds-crop mask becomes conditional on
  crossing:
  ```python
  if min_lon <= max_lon:
      lon_mask = (lon >= min_lon - deg_buf) & (lon <= max_lon + deg_buf)
  else:
      lon_mask = (lon >= min_lon - deg_buf) | (lon <= max_lon + deg_buf)
  ```
  (OR instead of AND — the valid region is the union of the two windows.)
- `visualization.py`'s `plot_collocation_diagnostics` map extent
  (`ax.set_extent(...)`) needs equivalent crossing-awareness. Exact cartopy
  mechanics (e.g. whether to pass a `central_longitude=180` projection for
  crossing recipes, versus another supported way to express a wrapped
  extent) will be confirmed empirically during implementation against the
  installed cartopy version, since `set_extent`'s handling of `min > max`
  isn't already exercised anywhere in this codebase.

### 4. Recipe / output-dir / CLI

- `build_output_dir` needs no code change — `f"{min_lon:.2f}_{max_lon:.2f}"`
  already produces an unambiguous (if unusual-looking) directory name, e.g.
  `135.00_-120.00`.
- `--create-recipe --min-lon/--max-lon` CLI flags need no new validation
  (floats pass through as-is); the `--create-recipe` epilog/help text gets a
  one-line note documenting the crossing convention.
- If a source's split queries both come back empty or both fail, that
  source fails exactly as it does today for zero-result queries — this spec
  does not change per-source error handling or pipeline continuation
  (tracked separately as the "continue past a failed download" bug).

## Testing

- `split_antimeridian_bbox`: unit tests for the non-crossing case
  (unchanged single window), the crossing case (two correct windows), and
  the boundary case `min_lon == max_lon` (treated as non-crossing, per the
  `<=` in the implementation).
- Per downloader family, extend `tests/test_downloaders.py` with a crossing
  case verifying: (a) the underlying query/subset call is invoked once per
  window with the split (non-crossing) bounds, never with the original
  crossing bounds, and (b) results from both windows are merged and
  de-duplicated where applicable (SAR `Id`, scatterometer `product_id`).
- `datatree_converter.py` crop mask: unit test with synthetic points on
  both sides of 180° against a crossing bbox, confirming both sides are
  kept and points strictly between `max_lon` and `min_lon` (the excluded
  middle) are dropped.
- End-to-end smoke: re-run `recipes/waves_pacific.yaml` with
  `min_lon: 135, max_lon: -120` through `--dry-run` first (confirms every
  downloader accepts the bbox and reports its per-window plan without
  erroring), then a real run if credentials are available, checking the
  resulting `validation_report.pdf` map no longer cuts off at 180°.

## Out of scope

- Continuing the pipeline past a single failed download and surfacing
  download failures in the PDF (separate bug, separate spec).
- Per-file download resume / skip-already-downloaded (separate bug,
  separate spec).
- The `collocation_results_individual.nc` filename/caching bug and the
  collocation-diagnostics transparency bug (separate spec).
- Any change to how a *non-crossing* bbox is validated or represented —
  today's `min_lon < max_lon` recipes are untouched by this design.
