# Fix RVL / currents extraction (WV + IW/EW/SM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `currents` recipes extract RVL radial velocity (not OWI wind) into `datatree.nc` for both WV and IW/EW/SM Sentinel-1 products, and validate end-to-end.

**Architecture:** Fix the SAR-side extractors in `datatree_converter.py` so `product_type="currents"` yields RVL nodes (WV → `point`, IW/EW/SM → `(y,x)` grid retaining all sub-swaths), never falling back to wind. Make the datatree the single source of truth by deleting collocation's separate WV-only disk re-scan, and add the RVL→radial projection to the WV point-collocation path so WV currents actually produce comparison pairs.

**Tech Stack:** Python, xarray, numpy, pandas, pytest. Sentinel-1 L2 OCN NetCDF (`owi*` / `rvl*` / `osw*` variable groups).

**Spec:** `docs/superpowers/specs/2026-07-13-rvl-currents-extraction-fix-design.md`

## Global Constraints

- Wind and waves extraction behavior must remain unchanged — only the `currents` path changes.
- No stale-`datatree.nc` migration/regeneration code; the user regenerates manually.
- SAR grid nodes use dim names `("y", "x")` with 2-D `lon`/`lat` coords, mirroring `_extract_owi_grid_data`, so `is_wv_mode` detection (`"point" in dims and "y" not in dims`) and the grid collocation path treat RVL and OWI grids identically.
- A `currents` run must never silently produce wind (`owi*`) data.
- Tests live in `tests/`, run with `pytest` (see `[tool.pytest.ini_options]`, `testpaths = ["tests"]`). Build synthetic NetCDF fixtures with `xr.Dataset(...).to_netcdf(path)` into `tmp_path`, following the existing helpers in `tests/test_datatree_converter.py`.

---

### Task 1: Retain all sub-swaths in `_extract_rvl_grid_data` (2-D grid branch)

Replace the `[:, :, 0]` first-swath slice with a reshape that merges the `rvlSwath` axis into the range axis, producing a single `(y, x)` grid that keeps every sub-swath. This fixes both the dims/shape crash (3 dim names over 2-D data) and the silent data loss (4/5 swaths for EW, 2/3 for IW).

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`_extract_rvl_grid_data`, `flatten_to_points=False` branch — currently the slicing at ~lines 1386-1420 and the `dims` inference at ~line 1440)
- Test: `tests/test_datatree_converter.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DataTreeConverter._extract_rvl_grid_data(measurement_dir: Path, safe_dir: str|Path, flatten_to_points: bool = False) -> Optional[xr.Dataset]`. In grid mode returns a Dataset with data_vars `rvlRadVel`, `rvlHeading`, `rvlIncidenceAngle` on dims `("y", "x")`; coords `lon`/`lat` on `("y", "x")` and scalar `time`; attrs include `data_type="sar_l2_ocn"`, `measurement_type="rvl"`. For a 3-D input `(Az, Ra, S)` the grid is `(Az, Ra*S)`.

- [ ] **Step 1: Add a shared test fixture builder for a synthetic OCN SAFE**

Add near the top of `tests/test_datatree_converter.py` (after the existing `_make_insitu_csv` helper):

```python
def _make_ocn_safe(
    tmp_path: Path,
    safe_name: str,
    *,
    rvl_swaths: int | None = None,
    wv: bool = False,
    with_owi: bool = True,
    ny: int = 5,
    nx: int = 4,
    seed: int = 0,
) -> Path:
    """
    Build a *.SAFE dir containing one '-ocn-' measurement NetCDF.

    rvl_swaths=None -> no rvl* variables written.
    rvl_swaths=S (wv=False) -> 3-D rvl (rvlAzSize, rvlRaSize, rvlSwath=S).
    wv=True -> 2-D 13x13 rvl (rvlAzSize, rvlRaSize), as in WV imagettes.
    """
    rng = np.random.default_rng(seed)
    safe = tmp_path / safe_name
    meas = safe / "measurement"
    meas.mkdir(parents=True)

    data: dict = {}
    if with_owi:
        odims = ("owiAzSize", "owiRaSize")
        data["owiWindSpeed"] = (odims, rng.uniform(2, 15, (ny, nx)).astype("float32"))
        data["owiWindDirection"] = (odims, rng.uniform(0, 360, (ny, nx)).astype("float32"))
        data["owiLon"] = (odims, rng.uniform(-20.0, -19.0, (ny, nx)).astype("float32"))
        data["owiLat"] = (odims, rng.uniform(50.0, 51.0, (ny, nx)).astype("float32"))

    if rvl_swaths is not None:
        if wv:
            shape, rdims = (13, 13), ("rvlAzSize", "rvlRaSize")
        else:
            shape, rdims = (ny, nx, rvl_swaths), ("rvlAzSize", "rvlRaSize", "rvlSwath")
        data["rvlRadVel"] = (rdims, rng.uniform(-3, 3, shape).astype("float32"))
        data["rvlLon"] = (rdims, rng.uniform(-20.0, -19.0, shape).astype("float32"))
        data["rvlLat"] = (rdims, rng.uniform(50.0, 51.0, shape).astype("float32"))
        data["rvlHeading"] = (rdims, rng.uniform(0, 360, shape).astype("float32"))
        data["rvlIncidenceAngle"] = (rdims, rng.uniform(20, 45, shape).astype("float32"))

    ds = xr.Dataset(data, attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"})
    mode = "wv1" if wv else "ew"
    fname = f"s1a-{mode}-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
    ds.to_netcdf(meas / fname)
    return safe
```

- [ ] **Step 2: Write the failing tests**

Add a new test class to `tests/test_datatree_converter.py`:

```python
class TestExtractRvlGridData:
    def test_multiswath_reshaped_to_grid_keeps_all_swaths(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=5, ny=5, nx=4)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert "rvlRadVel" in ds
        assert ds["rvlRadVel"].dims == ("y", "x")
        # All 5 sub-swaths retained: x == rvlRaSize * n_swaths, not just rvlRaSize.
        assert ds.sizes["y"] == 5
        assert ds.sizes["x"] == 4 * 5
        # No data lost: every input cell survives (fixture has no NaNs).
        assert int(np.isfinite(ds["rvlRadVel"].values).sum()) == 5 * 4 * 5

    def test_single_swath_2d_passes_through(self, tmp_path):
        # WV-style 13x13 2-D rvl, read through the grid (non-flatten) branch.
        safe = _make_ocn_safe(tmp_path, "S1A_SM_OCN.SAFE", rvl_swaths=1, wv=True)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert ds["rvlRadVel"].dims == ("y", "x")
        assert ds.sizes == {"y": 13, "x": 13}

    def test_returns_none_when_no_rvl(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_datatree_converter.py::TestExtractRvlGridData -v`
Expected: `test_multiswath_reshaped_to_grid_keeps_all_swaths` FAILS — before the fix the function raises internally (caught) and returns `None`, so `assert ds is not None` fails.

- [ ] **Step 4: Replace the slicing block with a reshape**

In `sar_validation/core/datatree_converter.py`, replace this block (the 3-D `[:, :, 0]` slicing for radvel/lats/lons/heading/incidence):

```python
                # If 3D data (with swaths), use first swath only for collocation compatibility
                if rvl_radvel_full.ndim == 3:
                    rvl_radvel = rvl_radvel_full[:, :, 0]
                    rvl_lats = rvl_lats_full[:, :, 0]
                    rvl_lons = rvl_lons_full[:, :, 0]
                else:
                    rvl_radvel = rvl_radvel_full
                    rvl_lats = rvl_lats_full
                    rvl_lons = rvl_lons_full

                rvl_heading_full = (
                    ds_raw["rvlHeading"].values
                    if "rvlHeading" in ds_raw
                    else None
                )
                if rvl_heading_full is not None:
                    rvl_heading = (
                        rvl_heading_full[:, :, 0] if rvl_heading_full.ndim == 3
                        else rvl_heading_full
                    )
                else:
                    rvl_heading = np.full_like(rvl_radvel, np.nan)

                rvl_incidence_full = (
                    ds_raw["rvlIncidenceAngle"].values
                    if "rvlIncidenceAngle" in ds_raw
                    else None
                )
                if rvl_incidence_full is not None:
                    rvl_incidence = (
                        rvl_incidence_full[:, :, 0] if rvl_incidence_full.ndim == 3
                        else rvl_incidence_full
                    )
                else:
                    rvl_incidence = np.full_like(rvl_radvel, np.nan)
```

with:

```python
                # RVL is 3-D (rvlAzSize, rvlRaSize, rvlSwath) for multi-swath
                # modes (IW/EW). Merge the swath axis into the range axis so the
                # grid keeps EVERY sub-swath — slicing [:, :, 0] would silently
                # drop all but the first swath (4 of 5 for EW, 2 of 3 for IW).
                # Single-swath products (SM) are already 2-D and pass through.
                def _swaths_to_grid(arr):
                    arr = np.asarray(arr)
                    return arr.reshape(arr.shape[0], -1) if arr.ndim == 3 else arr

                rvl_radvel = _swaths_to_grid(rvl_radvel_full)
                rvl_lats = _swaths_to_grid(rvl_lats_full)
                rvl_lons = _swaths_to_grid(rvl_lons_full)

                rvl_heading = (
                    _swaths_to_grid(ds_raw["rvlHeading"].values)
                    if "rvlHeading" in ds_raw
                    else np.full_like(rvl_radvel, np.nan)
                )
                rvl_incidence = (
                    _swaths_to_grid(ds_raw["rvlIncidenceAngle"].values)
                    if "rvlIncidenceAngle" in ds_raw
                    else np.full_like(rvl_radvel, np.nan)
                )
```

- [ ] **Step 5: Set standardized `(y, x)` dim names**

In the same function, replace:

```python
                # Infer dimension names from rvlLat shape
                dims = ds_raw["rvlLat"].dims if hasattr(ds_raw["rvlLat"], "dims") else ("rvlLat", "rvlLon")
```

with:

```python
                # Standard (y, x) naming to mirror the OWI grid, so is_wv_mode
                # detection and the grid collocation path treat RVL and OWI
                # grids identically.
                dims = ("y", "x")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_datatree_converter.py::TestExtractRvlGridData -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "fix: retain all RVL sub-swaths as (y,x) grid instead of slicing [:,:,0]"
```

---

### Task 2: Remove the currents → OWI fallback in `_from_sar_l2_ocn_iw_safe`

A currents run must never silently produce wind. When RVL is absent, return `None` and log a WARNING; `convert_downloaded_data` already drops `None` SAR nodes.

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`_from_sar_l2_ocn_iw_safe`, the `elif product_type.lower() == "currents":` branch, ~lines 1841-1856)
- Test: `tests/test_datatree_converter.py`

**Interfaces:**
- Consumes: `_extract_rvl_grid_data` (Task 1).
- Produces: `DataTreeConverter._from_sar_l2_ocn_iw_safe(safe_dir, product_type="wind")`. For `product_type="currents"`: returns the RVL grid Dataset if RVL exists, else `None` (never an OWI dataset).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_datatree_converter.py`:

```python
class TestIwSafeCurrentsNoOwiFallback:
    def test_currents_with_rvl_returns_rvl_grid(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=3, with_owi=True)
        ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="currents")
        assert ds is not None
        assert "rvlRadVel" in ds.data_vars
        assert "owiWindSpeed" not in ds.data_vars
        assert ds.attrs["swath_mode"] == "IW/EW/SM"

    def test_currents_without_rvl_returns_none_and_warns(self, tmp_path, caplog):
        # OWI present but no rvl* variables — must NOT fall back to wind.
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, with_owi=True)
        with caplog.at_level("WARNING"):
            ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="currents")
        assert ds is None
        assert any("no RVL" in r.message for r in caplog.records)

    def test_wind_still_extracts_owi(self, tmp_path):
        # Regression: wind behavior unchanged.
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, with_owi=True)
        ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="wind")
        assert ds is not None
        assert "owiWindSpeed" in ds.data_vars
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_datatree_converter.py::TestIwSafeCurrentsNoOwiFallback -v`
Expected: `test_currents_without_rvl_returns_none_and_warns` FAILS — current code falls back to OWI and returns a non-`None` dataset.

- [ ] **Step 3: Replace the currents branch**

In `sar_validation/core/datatree_converter.py`, replace:

```python
        elif product_type.lower() == "currents":
            # Try RVL extraction for currents products
            ds_rvl = DataTreeConverter._extract_rvl_grid_data(
                measurement_dir, safe_dir, flatten_to_points=False
            )
            if ds_rvl is not None:
                ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted RVL data from IW/EW/SM product %s", safe_dir.name)
                return ds_rvl
            # Fall back to OWI if RVL not available
            logger.debug("RVL data not found in %s; trying OWI fallback", safe_dir.name)
            ds = DataTreeConverter._extract_owi_grid_data(measurement_dir, safe_dir)
            if ds is not None:
                ds.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted OWI data (fallback) from IW/EW/SM product %s", safe_dir.name)
                return ds
```

with:

```python
        elif product_type.lower() == "currents":
            # RVL is the currents observable. Do NOT fall back to OWI wind — a
            # currents run must never silently produce wind data. If no RVL is
            # found, skip the scene (the caller drops None nodes) with a warning.
            ds_rvl = DataTreeConverter._extract_rvl_grid_data(
                measurement_dir, safe_dir, flatten_to_points=False
            )
            if ds_rvl is not None:
                ds_rvl.attrs["swath_mode"] = "IW/EW/SM"
                logger.info("Extracted RVL data from IW/EW/SM product %s", safe_dir.name)
                return ds_rvl
            logger.warning(
                "scene %s: no RVL/currents data — skipping (no OWI fallback for currents)",
                safe_dir.name,
            )
            return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_datatree_converter.py::TestIwSafeCurrentsNoOwiFallback -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "fix: currents extraction skips scene with warning instead of OWI fallback"
```

---

### Task 3: Route WV + currents to RVL extraction in `from_sar_l2_ocn_safe`

WV imagette OCN files carry both `oswHs` and a 13×13 `rvlRadVel` grid. Make the WV branch respect `product_type`: currents → RVL points, wind/waves → existing oswHs.

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`from_sar_l2_ocn_safe`, the WV dispatch at ~lines 1200-1204)
- Test: `tests/test_datatree_converter.py`

**Interfaces:**
- Consumes: existing `_extract_rvl_from_wv_safe(safe_dir) -> Optional[xr.Dataset]` (returns a `point` Dataset with `rvlRadVel`, `rvlHeading`, `rvlIncidenceAngle`, attr `swath_mode="WV"`), and existing `from_sar_l2_ocn_wv_safe`.
- Produces: `from_sar_l2_ocn_safe(safe_dir, product_type="wind")` — for a WV SAFE it returns an RVL `point` node when `product_type="currents"`, else the oswHs `point` node.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_datatree_converter.py`:

```python
class TestWvSafeProductTypeRouting:
    def test_wv_currents_returns_rvl_points(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_WV_OCN.SAFE", rvl_swaths=1, wv=True, with_owi=False)
        ds = DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type="currents")
        assert ds is not None
        assert "point" in ds.dims
        assert "rvlRadVel" in ds.data_vars
        assert ds.attrs.get("swath_mode") == "WV"

    def test_wv_waves_still_returns_oswhs(self, tmp_path):
        # Regression: non-currents WV behavior unchanged. Build a WV SAFE whose
        # measurement file also carries oswHs so the oswHs path has data.
        safe = tmp_path / "S1A_WV_OCN.SAFE"
        meas = safe / "measurement"
        meas.mkdir(parents=True)
        rng = np.random.default_rng(1)
        ds_raw = xr.Dataset(
            {
                "oswHs": (("oswPartitions",), rng.uniform(1, 4, 1).astype("float32")),
                "oswLon": ((), np.float32(-19.5)),
                "oswLat": ((), np.float32(50.5)),
            }
        )
        ds_raw.to_netcdf(meas / "s1a-wv1-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc")
        out = DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type="waves")
        assert out is not None
        assert "oswHs" in out.data_vars
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_datatree_converter.py::TestWvSafeProductTypeRouting -v`
Expected: `test_wv_currents_returns_rvl_points` FAILS — WV branch currently always calls `from_sar_l2_ocn_wv_safe` (oswHs), so `rvlRadVel` is absent.

- [ ] **Step 3: Make the WV branch product-type-aware**

In `sar_validation/core/datatree_converter.py`, replace:

```python
        # Detect mode from SAFE directory name
        if "WV" in safe_name:
            return DataTreeConverter.from_sar_l2_ocn_wv_safe(safe_dir)
        else:
            return DataTreeConverter._from_sar_l2_ocn_iw_safe(safe_dir, product_type=product_type)
```

with:

```python
        # Detect mode from SAFE directory name
        if "WV" in safe_name:
            # WV imagette OCN files carry oswHs AND a 13x13 rvlRadVel grid.
            # Route currents to RVL extraction; wind/waves keep oswHs.
            if product_type.lower() == "currents":
                return DataTreeConverter._extract_rvl_from_wv_safe(safe_dir)
            return DataTreeConverter.from_sar_l2_ocn_wv_safe(safe_dir)
        else:
            return DataTreeConverter._from_sar_l2_ocn_iw_safe(safe_dir, product_type=product_type)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_datatree_converter.py::TestWvSafeProductTypeRouting -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "feat: route WV+currents to RVL extraction in from_sar_l2_ocn_safe"
```

---

### Task 4: Add the RVL→radial projection to the WV point-collocation path

The grid path projects in-situ `EWCT`/`NSCT` onto `rvlHeading` to produce `rvlRadVel_projection`, but `_collocate_wv_points` does not — so WV currents would yield no comparison pairs. Add the same projection (heading is a scalar per WV point here).

**Files:**
- Modify: `sar_validation/core/collocation.py` (`_collocate_wv_points`, after `val_aggregated` is built and the empty-guard, before the `nearest = ...` block, ~line 1045)
- Test: `tests/test_collocation.py`

**Interfaces:**
- Consumes: `_collocate_wv_points(sar_lons, sar_lats, sar_times, sar_point_vars, val_data, val_source, footprint_radius_km, time_tolerance_minutes, distance_weighting, gaussian_sigma_km, collocation_type, sar_scene_name="") -> List[CollocatedPoint]` (existing signature, unchanged).
- Produces: when a WV point carries `rvlRadVel` + `rvlHeading` and the aggregated validation carries `EWCT` + `NSCT`, each resulting `CollocatedPoint.val_data` gains `rvlRadVel_projection`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_collocation.py`:

```python
class TestWvRvlProjection:
    def test_projection_added_from_ewct_nsct(self):
        from sar_validation.core.collocation import _collocate_wv_points

        # One WV RVL point with a known heading; one in-situ current obs on top.
        sar_lons = np.array([-19.5])
        sar_lats = np.array([50.5])
        sar_times = np.array([np.datetime64("2026-06-20T19:15:00", "ns")])
        sar_point_vars = {
            "rvlRadVel": np.array([1.0]),
            "rvlHeading": np.array([90.0]),  # heading_rad = radians(90-90)=0
        }
        val = pd.DataFrame({
            "lon": [-19.5],
            "lat": [50.5],
            "time": [pd.Timestamp("2026-06-20T19:20:00")],
            "EWCT": [0.4],
            "NSCT": [0.3],
        })
        matches = _collocate_wv_points(
            sar_lons=sar_lons, sar_lats=sar_lats, sar_times=sar_times,
            sar_point_vars=sar_point_vars, val_data=val, val_source="mooring",
            footprint_radius_km=14.0, time_tolerance_minutes=30,
            distance_weighting="equal", gaussian_sigma_km=5.0,
            collocation_type="point_vs_point",
        )
        assert len(matches) == 1
        proj = matches[0].val_data["rvlRadVel_projection"]
        # heading 90 -> heading_rad 0 -> projection = EWCT*cos0 + NSCT*sin0 = EWCT
        assert proj == pytest.approx(0.4, abs=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_collocation.py::TestWvRvlProjection -v`
Expected: FAIL with `KeyError: 'rvlRadVel_projection'`.

- [ ] **Step 3: Add the projection block**

In `sar_validation/core/collocation.py`, inside `_collocate_wv_points`, locate:

```python
        if not val_aggregated:
            continue

        nearest = int(np.argmin(dists))
```

and insert the projection block between them:

```python
        if not val_aggregated:
            continue

        # Project the in-situ current vector (EWCT/NSCT) onto the SAR radial
        # look direction so it can be compared against rvlRadVel — mirrors the
        # grid collocation path. Here rvlHeading is a scalar per imagette point.
        if (
            "rvlRadVel" in sar_aggregated
            and "rvlHeading" in sar_aggregated
            and "EWCT" in val_aggregated
            and "NSCT" in val_aggregated
        ):
            heading_rad = np.radians(float(sar_aggregated["rvlHeading"]) - 90.0)
            val_aggregated["rvlRadVel_projection"] = (
                float(val_aggregated["EWCT"]) * np.cos(heading_rad)
                + float(val_aggregated["NSCT"]) * np.sin(heading_rad)
            )

        nearest = int(np.argmin(dists))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_collocation.py::TestWvRvlProjection -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/collocation.py tests/test_collocation.py
git commit -m "feat: project EWCT/NSCT to rvlRadVel_projection in WV point collocation"
```

---

### Task 5: Delete the WV-only disk re-scan; read RVL from the datatree

With conversion fixed (Tasks 1-3), RVL now lives in `datatree.nc` for all modes. Remove `_load_rvl_for_collocation` and its call so the datatree is the single source of truth; the grid path (IW/EW/SM) and the WV point path (Task 4) consume RVL directly.

**Files:**
- Modify: `sar_validation/core/collocation.py` (delete `_load_rvl_for_collocation` at ~lines 834-874; delete its call in `run_collocation` at ~lines 1195-1200)
- Test: `tests/test_collocation.py`

**Interfaces:**
- Consumes: `run_collocation(recipe, datatree, base_dir, ...) -> Optional[xr.Dataset]` (existing). Reads SAR nodes from `datatree["sar"].children` and validation nodes from `datatree["validation"].children`.
- Produces: no `_load_rvl_for_collocation` symbol; for a currents datatree with an RVL `(y,x)` SAR node and an in-situ node carrying `EWCT`/`NSCT`, the returned results Dataset contains `sar_rvlRadVel` and `val_rvlRadVel_projection`.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_collocation.py`:

```python
class TestRunCollocationCurrentsFromDatatree:
    def _currents_recipe(self):
        from sar_validation.core.recipe import (
            Recipe, RecipeConfig, GeographicBounds, TemporalBounds,
        )
        return Recipe(RecipeConfig(
            name="currents_it",
            variable="currents",
            geographic_bounds=GeographicBounds(-21.0, -18.0, 49.0, 52.0),
            temporal_bounds=TemporalBounds("2026-06-20T18:00:00", "2026-06-20T23:00:00"),
        ))

    def test_no_load_rvl_symbol(self):
        import sar_validation.core.collocation as coll
        assert not hasattr(coll, "_load_rvl_for_collocation")

    def test_grid_rvl_projects_against_insitu(self, tmp_path):
        from sar_validation.core.collocation import run_collocation

        # SAR RVL grid node (y, x) with a constant heading of 90 deg.
        ny, nx = 4, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        sar = xr.Dataset(
            {
                "rvlRadVel": (("y", "x"), np.full((ny, nx), 0.5, dtype="float32")),
                "rvlHeading": (("y", "x"), np.full((ny, nx), 90.0, dtype="float32")),
                "rvlIncidenceAngle": (("y", "x"), np.full((ny, nx), 30.0, dtype="float32")),
            },
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": np.datetime64("2026-06-20T19:15:00", "ns"),
            },
            attrs={"data_type": "sar_l2_ocn", "swath_mode": "IW/EW/SM",
                   "measurement_type": "rvl"},
        )
        # In-situ mooring node with EWCT/NSCT at a SAR cell location + time.
        val = xr.Dataset(
            {
                "EWCT": (("point",), np.array([0.4], dtype="float32")),
                "NSCT": (("point",), np.array([0.3], dtype="float32")),
            },
            coords={
                "lon": (("point",), np.array([-19.5])),
                "lat": (("point",), np.array([50.5])),
                "time": (("point",), np.array([np.datetime64("2026-06-20T19:20:00", "ns")])),
                "platform_type": (("point",), np.array(["mooring"])),
            },
            attrs={"data_type": "insitu_observations", "platform_type": "mooring"},
        )
        tree = xr.DataTree.from_dict({"/sar/scene1": sar, "/validation/mooring1": val})

        result = run_collocation(self._currents_recipe(), tree, tmp_path)
        assert result is not None
        assert "sar_rvlRadVel" in result
        assert "val_rvlRadVel_projection" in result
        # heading 90 -> projection == EWCT == 0.4
        assert float(result["val_rvlRadVel_projection"].values[0]) == pytest.approx(0.4, abs=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_collocation.py::TestRunCollocationCurrentsFromDatatree -v`
Expected: `test_no_load_rvl_symbol` FAILS — the symbol still exists. (`test_grid_rvl_projects_against_insitu` may also error because `run_collocation` calls `_load_rvl_for_collocation` with a `base_dir` that has no `S1_L2_OCN/`, which is harmless, but the symbol-removal test pins the deletion.)

- [ ] **Step 3: Delete the call in `run_collocation`**

In `sar_validation/core/collocation.py`, delete this block:

```python
    # Load RVL (radial velocity) data on-demand — only for currents recipes.
    # RVL is the currents observable; loading it for a wind/waves recipe would
    # add spurious {scene}_rvl SAR nodes that get collocated against the
    # in-situ/altimeter data instead of the intended OWI/OSW measurement.
    if str(getattr(recipe.config, "variable", "")).lower() == "currents":
        _load_rvl_for_collocation(sar_scenes, base_dir)
```

(Leave the surrounding `sar_scenes` assembly and the following validation-node loop intact.)

- [ ] **Step 4: Delete the `_load_rvl_for_collocation` function**

In `sar_validation/core/collocation.py`, delete the entire `def _load_rvl_for_collocation(...)` function (from its `# ---- Helper to load RVL data on-demand ----` comment header through its `logger.debug("Could not load RVL for %s: %s", ...)` body). Removing it is safe because Step 3 removed its only caller.

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pytest tests/test_collocation.py::TestRunCollocationCurrentsFromDatatree -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/collocation.py tests/test_collocation.py
git commit -m "refactor: read RVL from datatree, drop WV-only disk re-scan"
```

---

### Task 6: Full-suite regression + end-to-end sanity on real data

Confirm nothing else regressed and that the fix works against the real fixtures on disk.

**Files:**
- No source changes (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass (including the new classes from Tasks 1-5).

- [ ] **Step 2: End-to-end check against the real EW currents fixture**

Run:

```bash
python -c "
from sar_validation.core.datatree_converter import DataTreeConverter
safe='data/2026-06-20-180000-2026-06-20-230000_-20.00_0.00_35.00_60.00/S1_L2_OCN/S1A_EW_OCN__2SDV_20260620T191525_20260620T191625_065057_083333_0711.SAFE'
ds=DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type='currents')
print('vars:', list(ds.data_vars))
print('dims:', dict(ds.sizes))
assert 'rvlRadVel' in ds.data_vars and 'owiWindSpeed' not in ds.data_vars
print('OK: EW currents now extracts RVL, not OWI')
"
```

Expected: prints `rvlRadVel` in vars, no `owiWindSpeed`, and `x` ≈ `rvlRaSize * 5` (all sub-swaths retained).

- [ ] **Step 3: End-to-end check against the real WV fixture**

Run:

```bash
python -c "
from sar_validation.core.datatree_converter import DataTreeConverter
safe='data/2026-04-01-000000-2026-04-03-000000_-20.00_0.00_35.00_60.00/S1_L2_OCN/S1A_WV_OCN__2SSV_20260401T184856_20260401T185012_063890_080920_95D6.SAFE'
ds=DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type='currents')
print('dims:', dict(ds.sizes), 'vars:', list(ds.data_vars))
assert 'point' in ds.dims and 'rvlRadVel' in ds.data_vars
print('OK: WV currents now extracts RVL points')
"
```

Expected: a `point`-dimensioned dataset containing `rvlRadVel`.

- [ ] **Step 4: Commit (if any doc/notes updated; otherwise skip)**

```bash
git commit --allow-empty -m "test: verify RVL currents extraction end-to-end on real fixtures"
```

---

## Self-Review

**Spec coverage:**
- §3a (multi-swath reshape, no data loss, `(y,x)` dims) → Task 1. ✓
- §3b (remove currents→OWI fallback, skip+warn) → Task 2. ✓
- §3c (WV product-type routing) → Task 3. ✓
- §3d (single source of truth, delete disk re-scan) → Task 5. ✓
- §5 unit tests (multiswath→grid, all-swaths retained, single-swath, WV currents points, no-RVL→None+warn) → Tasks 1-3. ✓
- §5 integration tests (EW dir → RVL not OWI, WV dir → RVL points, run_collocation end-to-end) → Tasks 5-6. ✓
- §7 open item #1 (WV path lacked projection) → resolved and implemented in Task 4. ✓
- §7 open item #2 (attrs) → resolved during planning: `grid_shape`/`measurement_type` are write-only; only `swath_mode` is read (visualization WV detection) and is already set. No task needed. ✓
- §6 scope (wind/waves unchanged, no regeneration code) → enforced by Global Constraints; Task 2 keeps wind/waves branches; regression tests in Tasks 2-3. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete before/after code and exact commands. ✓

**Type consistency:** `_extract_rvl_grid_data` / `_from_sar_l2_ocn_iw_safe` / `from_sar_l2_ocn_safe` / `_extract_rvl_from_wv_safe` / `_collocate_wv_points` / `run_collocation` signatures used consistently across tasks and match the current source. Variable/data-var names (`rvlRadVel`, `rvlHeading`, `rvlIncidenceAngle`, `rvlRadVel_projection`, `EWCT`, `NSCT`, `sar_`/`val_` result prefixes) are consistent throughout. ✓
