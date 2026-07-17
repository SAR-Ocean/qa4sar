# RVL Land-Flag Masking + rvlRadVelStd Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For currents recipes, mask `rvlRadVel`/`rvlRadVelStd` cells flagged as land by `rvlLandFlag`, report the pre-mask mean `rvlRadVel` over land as a QA signal (expected ≈0) in the PDF validation report, and extract `rvlRadVelStd` end-to-end into `datatree.nc` and `collocation_results.nc`.

**Architecture:** All real logic changes live in one function, `DataTreeConverter._extract_rvl_grid_data` in `sar_validation/core/datatree_converter.py` (both its grid branch for IW/EW/SM and its flatten-to-points branch used by WV) — mask land cells, compute QA stats, stamp them as dataset attrs, and extract `rvlRadVelStd` alongside the existing RVL variables. `sar_validation/core/collocation.py` needs zero code changes: both its grid and WV/point paths already build their per-variable dict from every data var with matching dims, and already skip NaN cells per variable, so masking and the new variable propagate automatically — this is covered by regression tests only. `sar_validation/core/visualization.py` gets one new function, `plot_rvl_land_qa`, wired into `validation_report()` to render a QA table page when any scene has land-flagged cells.

**Tech Stack:** Python, xarray, NumPy, pytest, matplotlib.

## Global Constraints

- `rvlLandFlag` semantics (from the L2 OCN product spec): set to 1 if land coverage of the cell exceeds 10%, else 0. Dims match `rvlRadVel`: `(rvlAzSize, rvlRaSize)` for SM/WV, `(rvlAzSize, rvlRaSize, rvlSwath)` for EW/IW.
- Only `rvlRadVel` and `rvlRadVelStd` are masked by the land flag. `rvlHeading` and `rvlIncidenceAngle` are geometry, not measurements, and are left untouched.
- RVL extraction is already currents-only in this codebase (`_extract_rvl_grid_data` is only invoked for `product_type="currents"`) — no new flags or recipe config are introduced.
- QA attrs stamped on the extracted RVL dataset: `rvl_land_pixel_count` (int), `rvl_land_pixel_fraction` (float, NaN if no classified cells), `rvl_land_mean_radvel` (float, NaN if no land cells).
- The QA report page must be omitted entirely (not shown empty) when no scene in the run has land-flagged cells.
- Spec: `docs/superpowers/specs/2026-07-17-rvl-land-flag-mask-and-radvel-std-design.md`

---

## Task 1: Grid branch (IW/EW/SM) — land masking + `rvlRadVelStd` extraction

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`_extract_rvl_grid_data`, non-`flatten_to_points` branch, currently around lines 1596–1652)
- Modify: `tests/test_datatree_converter.py` (extend the `_make_ocn_safe` helper, add tests to `TestExtractRvlGridData`)

**Interfaces:**
- Consumes: raw OCN `xr.Dataset` (`ds_raw`) already opened in `_extract_rvl_grid_data`; the existing `_swaths_to_grid(arr)` closure (reshapes `(az, ra, swath)` → `(az, ra*swath)`, passes `(az, ra)` through unchanged).
- Produces: the returned `xr.Dataset` gains a `"rvlRadVelStd"` data variable (dims `("y","x")`, same shape as `"rvlRadVel"`) and three new dataset attrs: `rvl_land_pixel_count` (int), `rvl_land_pixel_fraction` (float), `rvl_land_mean_radvel` (float). `"rvlRadVel"` and `"rvlRadVelStd"` have land-flagged cells set to `np.nan`.

- [ ] **Step 1: Extend the `_make_ocn_safe` test fixture to support `rvlLandFlag` and always write `rvlRadVelStd`**

In `tests/test_datatree_converter.py`, replace the `_make_ocn_safe` function (lines 42–88) with:

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
    land_rows: int = 0,
) -> Path:
    """
    Build a *.SAFE dir containing one '-ocn-' measurement NetCDF.

    rvl_swaths=None -> no rvl* variables written.
    rvl_swaths=S (wv=False) -> 3-D rvl (rvlAzSize, rvlRaSize, rvlSwath=S).
    wv=True -> 2-D 13x13 rvl (rvlAzSize, rvlRaSize), as in WV imagettes.
    land_rows=N -> the first N rows of the rvlAzSize axis are written with
        rvlLandFlag=1 (land) across every column/swath; the rest are 0.
        land_rows=0 (default) omits rvlLandFlag entirely, simulating a
        product that doesn't carry it.
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
        data["rvlRadVelStd"] = (rdims, rng.uniform(0.0, 0.5, shape).astype("float32"))
        if land_rows > 0:
            land_flag = np.zeros(shape, dtype="float32")
            land_flag[:land_rows, ...] = 1.0
            data["rvlLandFlag"] = (rdims, land_flag)

    ds = xr.Dataset(data, attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"})
    mode = "wv1" if wv else "ew"
    fname = f"s1a-{mode}-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
    ds.to_netcdf(meas / fname)
    return safe
```

This is additive (new `land_rows` keyword defaulting to 0, `rvlRadVelStd` always written alongside `rvlRadVel`) — every existing call site of `_make_ocn_safe` keeps working unchanged.

- [ ] **Step 2: Write the failing tests**

Append to the `TestExtractRvlGridData` class in `tests/test_datatree_converter.py` (after `test_returns_none_when_no_rvl`, still inside the class, so keep the 4-space indentation):

```python
    def test_land_flag_masks_radvel_and_std_grid(self, tmp_path):
        # Multi-swath EW-style grid with the first 2 of 5 azimuth rows
        # land-flagged across every range cell and every swath.
        safe = _make_ocn_safe(
            tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=3, ny=5, nx=4, land_rows=2,
        )
        raw = xr.open_dataset(
            safe / "measurement" / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
        )
        raw_radvel = raw["rvlRadVel"].values.reshape(5, -1)  # (az=5, ra*swath=12)
        expected_land_mean = float(np.nanmean(raw_radvel[:2, :]))
        raw.close()

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None

        # rvlRadVel / rvlRadVelStd are NaN in the land rows, finite elsewhere.
        assert np.isnan(ds["rvlRadVel"].values[:2, :]).all()
        assert np.isfinite(ds["rvlRadVel"].values[2:, :]).all()
        assert np.isnan(ds["rvlRadVelStd"].values[:2, :]).all()
        assert np.isfinite(ds["rvlRadVelStd"].values[2:, :]).all()

        # Geometry variables are untouched by the land mask.
        assert np.isfinite(ds["rvlHeading"].values).all()
        assert np.isfinite(ds["rvlIncidenceAngle"].values).all()

        assert ds.attrs["rvl_land_pixel_count"] == 2 * 4 * 3  # land_rows * nx * rvl_swaths
        assert ds.attrs["rvl_land_pixel_fraction"] == pytest.approx((2 * 4 * 3) / (5 * 4 * 3))
        assert ds.attrs["rvl_land_mean_radvel"] == pytest.approx(expected_land_mean, abs=1e-5)

    def test_zero_land_pixels_no_masking_grid(self, tmp_path):
        # land_rows=0 (default) -> no rvlLandFlag written at all.
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=3, ny=5, nx=4)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert np.isfinite(ds["rvlRadVel"].values).all()
        assert ds.attrs["rvl_land_pixel_count"] == 0
        assert math.isnan(ds.attrs["rvl_land_mean_radvel"])

    def test_single_swath_land_flag_grid(self, tmp_path):
        # SM/WV-style single-swath 2-D grid (13x13), 3 land rows.
        safe = _make_ocn_safe(tmp_path, "S1A_SM_OCN.SAFE", rvl_swaths=1, wv=True, land_rows=3)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert ds.sizes == {"y": 13, "x": 13}
        assert np.isnan(ds["rvlRadVel"].values[:3, :]).all()
        assert np.isfinite(ds["rvlRadVel"].values[3:, :]).all()
        assert ds.attrs["rvl_land_pixel_count"] == 3 * 13
```

Add `import math` to the top of `tests/test_datatree_converter.py` if it isn't already imported (check the existing import block at lines 3–19 first — it currently has no `math` import).

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `pytest tests/test_datatree_converter.py::TestExtractRvlGridData -v`
Expected: `test_land_flag_masks_radvel_and_std_grid`, `test_zero_land_pixels_no_masking_grid`, and `test_single_swath_land_flag_grid` FAIL (`KeyError: 'rvlRadVelStd'` or `AssertionError` — `rvlRadVelStd` isn't extracted yet and no masking happens). The two pre-existing tests in this class still PASS.

- [ ] **Step 4: Implement masking + `rvlRadVelStd` extraction in the grid branch**

In `sar_validation/core/datatree_converter.py`, inside `_extract_rvl_grid_data`, find this block (the non-`flatten_to_points` branch — currently lines ~1596–1652):

```python
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

                # Get acquisition time (scalar for grid)
```

Replace it with:

```python
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
                rvl_radvel_std = (
                    _swaths_to_grid(ds_raw["rvlRadVelStd"].values)
                    if "rvlRadVelStd" in ds_raw
                    else np.full_like(rvl_radvel, np.nan)
                )

                # Land-flag masking. rvlLandFlag is set to 1 where land
                # coverage of the cell exceeds 10%. Land-contaminated cells
                # must not feed into currents validation, so rvlRadVel and
                # rvlRadVelStd are NaN'd out there — but the pre-mask mean is
                # kept as a QA stat since it is expected to be ~0 and a
                # meaningfully non-zero value signals a data-quality issue.
                # rvlHeading/rvlIncidenceAngle are geometry, not
                # measurements, and are left unmasked.
                land_pixel_count = 0
                land_pixel_fraction = float("nan")
                land_mean_radvel = float("nan")
                if "rvlLandFlag" in ds_raw:
                    rvl_landflag = _swaths_to_grid(ds_raw["rvlLandFlag"].values).astype(float)
                    land_mask = rvl_landflag == 1
                    total_classified = int(np.sum(~np.isnan(rvl_landflag)))
                    land_pixel_count = int(np.sum(land_mask))
                    if total_classified > 0:
                        land_pixel_fraction = land_pixel_count / total_classified
                    if land_pixel_count > 0:
                        land_mean_radvel = float(np.nanmean(rvl_radvel[land_mask]))
                        rvl_radvel = np.where(land_mask, np.nan, rvl_radvel)
                        rvl_radvel_std = np.where(land_mask, np.nan, rvl_radvel_std)
                        logger.warning(
                            "scene %s: %d/%d RVL cells land-flagged (%.1f%%) — "
                            "mean rvlRadVel over land = %.4f m/s (expected ~0)",
                            safe_dir.name, land_pixel_count, total_classified,
                            100 * land_pixel_fraction, land_mean_radvel,
                        )

                # Get acquisition time (scalar for grid)
```

Then find the `data_vars` dict a little further down:

```python
                # Create Dataset with 2D grid structure
                data_vars = {
                    "rvlRadVel": (dims, rvl_radvel),
                    "rvlHeading": (dims, rvl_heading),
                    "rvlIncidenceAngle": (dims, rvl_incidence),
                }
```

Replace it with:

```python
                # Create Dataset with 2D grid structure
                data_vars = {
                    "rvlRadVel": (dims, rvl_radvel),
                    "rvlRadVelStd": (dims, rvl_radvel_std),
                    "rvlHeading": (dims, rvl_heading),
                    "rvlIncidenceAngle": (dims, rvl_incidence),
                }
```

Then find the attrs block near the end of the same branch:

```python
                ds.attrs["measurement_type"] = "rvl"
                ds.attrs["grid_shape"] = rvl_radvel.shape
```

Replace it with:

```python
                ds.attrs["measurement_type"] = "rvl"
                ds.attrs["grid_shape"] = rvl_radvel.shape
                ds.attrs["rvl_land_pixel_count"] = land_pixel_count
                ds.attrs["rvl_land_pixel_fraction"] = land_pixel_fraction
                ds.attrs["rvl_land_mean_radvel"] = land_mean_radvel
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_datatree_converter.py::TestExtractRvlGridData -v`
Expected: all 5 tests in this class PASS.

- [ ] **Step 6: Run the full datatree_converter test file to check for regressions**

Run: `pytest tests/test_datatree_converter.py -v`
Expected: all tests PASS (in particular `TestIwSafeCurrentsNoOwiFallback` and `TestWvSafeProductTypeRouting`, which call functions that route into `_extract_rvl_grid_data`).

- [ ] **Step 7: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "feat: mask land-flagged RVL cells and extract rvlRadVelStd (grid path)"
```

---

## Task 2: Points/WV branch — land masking + `rvlRadVelStd` extraction, accumulated across imagette files

**Files:**
- Modify: `sar_validation/core/datatree_converter.py` (`_extract_rvl_grid_data`, `flatten_to_points=True` branch, currently around lines 1666–1767)
- Modify: `tests/test_datatree_converter.py` (add a new fixture helper and tests to `TestExtractRvlGridData`)

**Interfaces:**
- Consumes: same as Task 1, but this branch loops over every `*-ocn-*.nc` file in `measurement_dir` (one per WV imagette) and concatenates into a single `point`-dimensioned Dataset.
- Produces: the returned Dataset gains a `"rvlRadVelStd"` data variable (dims `("point",)`) and the same three QA attrs as Task 1, accumulated across every file in the scene (not just the first).

- [ ] **Step 1: Add a multi-file WV fixture helper and write the failing tests**

Append this helper function to `tests/test_datatree_converter.py`, right after `_make_ocn_safe` (before `_make_collocations`):

```python
def _make_wv_rvl_safe(
    tmp_path: Path,
    *,
    land_rows_per_file: list[int],
    seed: int = 0,
) -> Path:
    """
    Build a WV *.SAFE dir with one 13x13-imagette RVL measurement file per
    entry in land_rows_per_file. Entry i controls how many of that file's
    13 rvlAzSize rows are land-flagged (0 = no rvlLandFlag var at all for
    that file).
    """
    rng = np.random.default_rng(seed)
    safe = tmp_path / "S1A_WV_OCN.SAFE"
    meas = safe / "measurement"
    meas.mkdir(parents=True)
    shape, rdims = (13, 13), ("rvlAzSize", "rvlRaSize")

    for i, land_rows in enumerate(land_rows_per_file):
        data = {
            "rvlRadVel": (rdims, rng.uniform(-3, 3, shape).astype("float32")),
            "rvlLon": (rdims, rng.uniform(-20.0, -19.0, shape).astype("float32")),
            "rvlLat": (rdims, rng.uniform(50.0, 51.0, shape).astype("float32")),
            "rvlHeading": (rdims, rng.uniform(0, 360, shape).astype("float32")),
            "rvlIncidenceAngle": (rdims, rng.uniform(20, 45, shape).astype("float32")),
            "rvlRadVelStd": (rdims, rng.uniform(0.0, 0.5, shape).astype("float32")),
        }
        if land_rows > 0:
            land_flag = np.zeros(shape, dtype="float32")
            land_flag[:land_rows, :] = 1.0
            data["rvlLandFlag"] = (rdims, land_flag)
        ds_raw = xr.Dataset(data, attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"})
        fname = f"s1a-wv1-ocn-vv-20260620t19152{i}-20260620t19162{i}-065057-08333{i}-001.nc"
        ds_raw.to_netcdf(meas / fname)

    return safe
```

Append these tests to the `TestExtractRvlGridData` class:

```python
    def test_land_flag_masks_points_single_file(self, tmp_path):
        safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[3])
        raw = xr.open_dataset(sorted((safe / "measurement").glob("*.nc"))[0])
        raw_radvel = raw["rvlRadVel"].values  # (13, 13)
        expected_land_mean = float(np.nanmean(raw_radvel[:3, :]))
        raw.close()

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=True
        )
        assert ds is not None
        assert ds.sizes["point"] == 13 * 13
        # Ravel order is row-major, so the first 3*13 points are the land rows.
        assert np.isnan(ds["rvlRadVel"].values[: 3 * 13]).all()
        assert np.isfinite(ds["rvlRadVel"].values[3 * 13 :]).all()
        assert np.isnan(ds["rvlRadVelStd"].values[: 3 * 13]).all()
        assert np.isfinite(ds["rvlHeading"].values).all()

        assert ds.attrs["rvl_land_pixel_count"] == 3 * 13
        assert ds.attrs["rvl_land_pixel_fraction"] == pytest.approx((3 * 13) / (13 * 13))
        assert ds.attrs["rvl_land_mean_radvel"] == pytest.approx(expected_land_mean, abs=1e-5)

    def test_land_flag_accumulates_across_files(self, tmp_path):
        safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[3, 5])
        files = sorted((safe / "measurement").glob("*.nc"))
        assert len(files) == 2
        raw0, raw1 = xr.open_dataset(files[0]), xr.open_dataset(files[1])
        land_sum = (
            float(np.nansum(raw0["rvlRadVel"].values[:3, :]))
            + float(np.nansum(raw1["rvlRadVel"].values[:5, :]))
        )
        raw0.close()
        raw1.close()
        expected_count = 3 * 13 + 5 * 13
        expected_mean = land_sum / expected_count

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=True
        )
        assert ds is not None
        assert ds.sizes["point"] == 2 * 13 * 13
        assert ds.attrs["rvl_land_pixel_count"] == expected_count
        assert ds.attrs["rvl_land_mean_radvel"] == pytest.approx(expected_mean, abs=1e-5)

    def test_zero_land_points(self, tmp_path):
        safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[0])
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=True
        )
        assert ds is not None
        assert np.isfinite(ds["rvlRadVel"].values).all()
        assert ds.attrs["rvl_land_pixel_count"] == 0
        assert math.isnan(ds.attrs["rvl_land_mean_radvel"])

    def test_rvl_radvel_std_extracted_points(self, tmp_path):
        safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[0])
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=True
        )
        assert ds is not None
        assert "rvlRadVelStd" in ds.data_vars
        assert ds["rvlRadVelStd"].dims == ("point",)
        assert ds.sizes["point"] == 13 * 13
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_datatree_converter.py::TestExtractRvlGridData -k "points" -v`
Expected: all 4 new tests FAIL (`KeyError: 'rvlRadVelStd'` or assertion errors — the points branch doesn't extract `rvlRadVelStd` or mask land yet).

- [ ] **Step 3: Implement masking + `rvlRadVelStd` extraction + cross-file accumulation in the points branch**

In `sar_validation/core/datatree_converter.py`, inside `_extract_rvl_grid_data`, find the start of the `else:` (flatten-to-points) branch:

```python
        else:
            # Flatten RVL grids to points (for WV mode backward compat)
            point_lons: list[float] = []
            point_lats: list[float] = []
            point_radvel: list[float] = []
            point_heading: list[float] = []
            point_incidence: list[float] = []
            point_times = []
            file_names = []
            rvl_attrs: Dict[str, Dict] = {}
```

Replace it with:

```python
        else:
            # Flatten RVL grids to points (for WV mode backward compat)
            point_lons: list[float] = []
            point_lats: list[float] = []
            point_radvel: list[float] = []
            point_radvel_std: list[float] = []
            point_heading: list[float] = []
            point_incidence: list[float] = []
            point_times = []
            file_names = []
            rvl_attrs: Dict[str, Dict] = {}

            # Land-flag QA accumulated across every imagette file in this
            # scene (see the grid branch above for the rationale — same
            # masking rule, same QA stats, just summed across files here
            # since one WV scene is many small imagette files).
            land_pixel_count_total = 0
            total_classified_total = 0
            land_radvel_sum = 0.0
```

Then find:

```python
                    if not rvl_attrs:
                        rvl_attrs = {
                            v: dict(ds_raw[v].attrs)
                            for v in ("rvlRadVel", "rvlHeading", "rvlIncidenceAngle")
                            if v in ds_raw
                        }

                    # Extract RVL grid arrays and flatten to 1D
                    rvl_radvel = ds_raw["rvlRadVel"].values.ravel()
                    rvl_lats = ds_raw["rvlLat"].values.ravel()
                    rvl_lons = ds_raw["rvlLon"].values.ravel()

                    rvl_heading = (
                        ds_raw["rvlHeading"].values.ravel()
                        if "rvlHeading" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )
                    rvl_incidence = (
                        ds_raw["rvlIncidenceAngle"].values.ravel()
                        if "rvlIncidenceAngle" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )

                    # Get acquisition time
```

Replace it with:

```python
                    if not rvl_attrs:
                        rvl_attrs = {
                            v: dict(ds_raw[v].attrs)
                            for v in ("rvlRadVel", "rvlRadVelStd", "rvlHeading", "rvlIncidenceAngle")
                            if v in ds_raw
                        }

                    # Extract RVL grid arrays and flatten to 1D
                    rvl_radvel = ds_raw["rvlRadVel"].values.ravel()
                    rvl_lats = ds_raw["rvlLat"].values.ravel()
                    rvl_lons = ds_raw["rvlLon"].values.ravel()

                    rvl_heading = (
                        ds_raw["rvlHeading"].values.ravel()
                        if "rvlHeading" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )
                    rvl_incidence = (
                        ds_raw["rvlIncidenceAngle"].values.ravel()
                        if "rvlIncidenceAngle" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )
                    rvl_radvel_std = (
                        ds_raw["rvlRadVelStd"].values.ravel()
                        if "rvlRadVelStd" in ds_raw
                        else np.full_like(rvl_radvel, np.nan)
                    )

                    # Land-flag masking (see the grid branch for rationale).
                    # rvlHeading/rvlIncidenceAngle are left unmasked.
                    if "rvlLandFlag" in ds_raw:
                        rvl_landflag = ds_raw["rvlLandFlag"].values.ravel().astype(float)
                        land_mask = rvl_landflag == 1
                        total_classified_total += int(np.sum(~np.isnan(rvl_landflag)))
                        file_land_count = int(np.sum(land_mask))
                        if file_land_count > 0:
                            land_pixel_count_total += file_land_count
                            land_radvel_sum += float(np.nansum(rvl_radvel[land_mask]))
                            rvl_radvel = np.where(land_mask, np.nan, rvl_radvel)
                            rvl_radvel_std = np.where(land_mask, np.nan, rvl_radvel_std)

                    # Get acquisition time
```

Then find the point-accumulation block:

```python
                    # Add all RVL points from this file
                    n_points = len(rvl_lons)
                    point_lons.extend(rvl_lons)
                    point_lats.extend(rvl_lats)
                    point_radvel.extend(rvl_radvel)
                    point_heading.extend(rvl_heading)
                    point_incidence.extend(rvl_incidence)
                    point_times.extend([acq_time_ns] * n_points)
                    file_names.extend([nc_path.name] * n_points)
```

Replace it with:

```python
                    # Add all RVL points from this file
                    n_points = len(rvl_lons)
                    point_lons.extend(rvl_lons)
                    point_lats.extend(rvl_lats)
                    point_radvel.extend(rvl_radvel)
                    point_radvel_std.extend(rvl_radvel_std)
                    point_heading.extend(rvl_heading)
                    point_incidence.extend(rvl_incidence)
                    point_times.extend([acq_time_ns] * n_points)
                    file_names.extend([nc_path.name] * n_points)
```

Then find the final Dataset construction:

```python
            # Create Dataset with point dimension (flattened RVL grids)
            data_vars = {
                "rvlRadVel": (("point",), np.asarray(point_radvel)),
                "rvlHeading": (("point",), np.asarray(point_heading)),
                "rvlIncidenceAngle": (("point",), np.asarray(point_incidence)),
            }

            coords = {
                "lon": (["point"], point_lons),
                "lat": (["point"], point_lats),
                "time": (["point"], point_times),
                "filename": (["point"], file_names),
            }

            ds = xr.Dataset(data_vars, coords=coords)
            apply_cf_metadata(ds, "sar", rvl_attrs)
            ds.attrs["data_type"] = "sar_l2_ocn"
            ds.attrs["source"] = "Sentinel-1"
            ds.attrs["safe_dir"] = safe_dir.name
            ds.attrs["measurement_type"] = "rvl"
            ds.attrs["num_points"] = len(point_radvel)
```

Replace it with:

```python
            # Create Dataset with point dimension (flattened RVL grids)
            data_vars = {
                "rvlRadVel": (("point",), np.asarray(point_radvel)),
                "rvlRadVelStd": (("point",), np.asarray(point_radvel_std)),
                "rvlHeading": (("point",), np.asarray(point_heading)),
                "rvlIncidenceAngle": (("point",), np.asarray(point_incidence)),
            }

            coords = {
                "lon": (["point"], point_lons),
                "lat": (["point"], point_lats),
                "time": (["point"], point_times),
                "filename": (["point"], file_names),
            }

            land_pixel_fraction = (
                land_pixel_count_total / total_classified_total
                if total_classified_total > 0 else float("nan")
            )
            land_mean_radvel = (
                land_radvel_sum / land_pixel_count_total
                if land_pixel_count_total > 0 else float("nan")
            )
            if land_pixel_count_total > 0:
                logger.warning(
                    "scene %s: %d/%d RVL cells land-flagged (%.1f%%) — "
                    "mean rvlRadVel over land = %.4f m/s (expected ~0)",
                    safe_dir.name, land_pixel_count_total, total_classified_total,
                    100 * land_pixel_fraction, land_mean_radvel,
                )

            ds = xr.Dataset(data_vars, coords=coords)
            apply_cf_metadata(ds, "sar", rvl_attrs)
            ds.attrs["data_type"] = "sar_l2_ocn"
            ds.attrs["source"] = "Sentinel-1"
            ds.attrs["safe_dir"] = safe_dir.name
            ds.attrs["measurement_type"] = "rvl"
            ds.attrs["num_points"] = len(point_radvel)
            ds.attrs["rvl_land_pixel_count"] = land_pixel_count_total
            ds.attrs["rvl_land_pixel_fraction"] = land_pixel_fraction
            ds.attrs["rvl_land_mean_radvel"] = land_mean_radvel
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_datatree_converter.py::TestExtractRvlGridData -v`
Expected: all 9 tests in this class PASS (5 from Task 1 + 4 from this task).

- [ ] **Step 5: Run the full datatree_converter test file to check for regressions**

Run: `pytest tests/test_datatree_converter.py -v`
Expected: all tests PASS, including `TestWvSafeProductTypeRouting::test_wv_currents_returns_rvl_points` (which routes through this same branch via `_extract_rvl_from_wv_safe`).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "feat: mask land-flagged RVL cells and extract rvlRadVelStd (WV points path)"
```

---

## Task 3: Collocation regression tests — confirm `rvlRadVelStd` propagates automatically

**Files:**
- Modify: `tests/test_collocation.py` (add tests to `TestRunCollocationCurrentsFromDatatree` and `TestWvRvlProjection`)

No production code changes in this task — `sar_validation/core/collocation.py` already builds `sar_data_3d`/`sar_point_vars` from every SAR data variable whose dims match `("y","x")`/`("point",)` (see `run_collocation`, lines ~1346–1350 and ~1251–1255), and `_compute_aggregated_sar_value` already skips NaN cells per variable. These tests are a regression guard confirming that behavior — if a future change hardcodes a variable allowlist, this task's tests will catch it.

**Interfaces:**
- Consumes: `run_collocation(recipe, datatree, tmp_path)` → `xr.Dataset | None` with `sar_<var>` / `val_<var>` columns (existing signature, unchanged). `_collocate_wv_points(...)` (existing signature, unchanged, used directly in `TestWvRvlProjection`).
- Produces: nothing new — test-only task.

- [ ] **Step 1: Write the failing tests**

Add this test to the `TestRunCollocationCurrentsFromDatatree` class in `tests/test_collocation.py`, after `test_grid_rvl_projects_against_insitu`:

```python
    def test_grid_rvl_radvel_std_propagates(self, tmp_path):
        import xarray as xr
        from sar_validation.core.collocation import run_collocation

        ny, nx = 5, 5
        lon2d, lat2d = np.meshgrid(
            np.linspace(-20.0, -19.0, nx), np.linspace(50.0, 51.0, ny)
        )
        sar = xr.Dataset(
            {
                "rvlRadVel": (("y", "x"), np.full((ny, nx), 0.5, dtype="float32")),
                "rvlRadVelStd": (("y", "x"), np.full((ny, nx), 0.12, dtype="float32")),
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
        assert "sar_rvlRadVelStd" in result
        assert float(result["sar_rvlRadVelStd"].values[0]) == pytest.approx(0.12, abs=1e-5)
```

Add this test to the `TestWvRvlProjection` class in `tests/test_collocation.py`, after `test_projection_added_from_ewct_nsct`:

```python
    def test_radvel_std_propagates(self):
        from sar_validation.core.collocation import _collocate_wv_points

        sar_lons = np.array([-19.5])
        sar_lats = np.array([50.5])
        sar_times = np.array([np.datetime64("2026-06-20T19:15:00", "ns")])
        sar_point_vars = {
            "rvlRadVel": np.array([1.0]),
            "rvlRadVelStd": np.array([0.15]),
            "rvlHeading": np.array([90.0]),
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
        assert matches[0].sar_data["rvlRadVelStd"] == pytest.approx(0.15, abs=1e-6)
```

- [ ] **Step 2: Run the new tests to verify current behavior**

Run: `pytest tests/test_collocation.py::TestRunCollocationCurrentsFromDatatree::test_grid_rvl_radvel_std_propagates tests/test_collocation.py::TestWvRvlProjection::test_radvel_std_propagates -v`
Expected: both PASS already, since no production code needs to change (confirms the spec's "no code changes needed" claim). If either fails, that means the generic dims-based propagation described in the spec is not actually happening — stop and investigate `run_collocation`/`_collocate_wv_points` in `sar_validation/core/collocation.py` before proceeding, rather than adding special-case code for `rvlRadVelStd`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_collocation.py
git commit -m "test: regression guard for rvlRadVelStd propagation through collocation"
```

---

## Task 4: PDF report — RVL land-contamination QA page

**Files:**
- Modify: `sar_validation/core/visualization.py` (new function `plot_rvl_land_qa`, wire into `validation_report`)
- Modify: `tests/test_visualization.py` (new tests)

**Interfaces:**
- Consumes: `datatree` (`xr.DataTree`) — same object already passed into `validation_report`; per-scene QA attrs `rvl_land_pixel_count` / `rvl_land_pixel_fraction` / `rvl_land_mean_radvel` produced by Tasks 1–2 and readable via `datatree["sar"].children[name].to_dataset().attrs`.
- Produces: `plot_rvl_land_qa(datatree) -> Optional[matplotlib.figure.Figure]`. Returns `None` if no `"sar"` node exists or no scene has `rvl_land_pixel_count > 0`. Otherwise returns a Figure containing one table with columns `Scene`, `Land pixels`, `Land %`, `Mean rvlRadVel over land (m/s)` — one row per land-affected scene.

- [ ] **Step 1: Write the failing tests for `plot_rvl_land_qa` directly**

Add a new test class to `tests/test_visualization.py`, after `TestValidationReportCurrentsPointSize` (i.e. after line ~1710, before `TestImagePageFigure`):

```python
class TestPlotRvlLandQa:
    def _make_sar_node(self, *, land_count=0, land_fraction=float("nan"), land_mean=float("nan")):
        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        attrs = {"measurement_type": "rvl"}
        if land_count:
            attrs["rvl_land_pixel_count"] = land_count
            attrs["rvl_land_pixel_fraction"] = land_fraction
            attrs["rvl_land_mean_radvel"] = land_mean
        return xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
            attrs=attrs,
        )

    def test_returns_none_when_no_scene_has_land(self):
        from sar_validation.core.visualization import plot_rvl_land_qa
        from sar_validation.core.datatree_converter import DataTreeConverter

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": self._make_sar_node(land_count=0),
        })
        assert plot_rvl_land_qa(datatree) is None

    def test_returns_none_when_no_sar_node(self):
        from sar_validation.core.visualization import plot_rvl_land_qa
        from sar_validation.core.datatree_converter import DataTreeConverter

        datatree = DataTreeConverter.to_datatree({})
        assert plot_rvl_land_qa(datatree) is None

    def test_returns_table_with_one_row_per_land_scene(self):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_rvl_land_qa
        from sar_validation.core.datatree_converter import DataTreeConverter

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": self._make_sar_node(land_count=0),
            "sar/sceneB": self._make_sar_node(land_count=24, land_fraction=0.4, land_mean=0.71),
        })
        fig = plot_rvl_land_qa(datatree)
        assert fig is not None
        table = fig.axes[0].tables[0]
        cells = table.get_celld()
        n_rows = len({r for (r, _c) in cells.keys()})
        assert n_rows == 2  # header + 1 data row (sceneA has no land, omitted)
        assert cells[(1, 0)].get_text().get_text() == "sceneB"
        assert cells[(1, 1)].get_text().get_text() == "24"
        plt.close(fig)
```

Add these tests to a new class in the same file, after `TestPlotRvlLandQa`:

```python
class TestValidationReportRvlLandQaPage:
    def _sar_node(self, *, with_land: bool):
        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        attrs = {}
        if with_land:
            attrs.update(
                rvl_land_pixel_count=9, rvl_land_pixel_fraction=1.0, rvl_land_mean_radvel=0.65,
            )
        return xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
            attrs=attrs,
        )

    def _coll_ds(self):
        return xr.Dataset({
            "sar_rvlRadVel":            ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection": ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":               ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":           ("collocation", ["sceneA"] * 4),
            "val_lon":                  ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                  ("collocation", [50.5, 50.6, 50.7, 50.8]),
        })

    def _count_image_pages(self, monkeypatch, datatree, recipe, tmp_path):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from sar_validation.core.visualization import validation_report

        recorded_figs = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_figs.append(fig)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)
        validation_report(self._coll_ds(), datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        def is_image_page(fig):
            return fig is not None and len(fig.axes) == 1 and len(fig.axes[0].images) > 0

        return sum(1 for f in recorded_figs if is_image_page(f))

    def test_qa_page_added_for_currents_with_land(self, tmp_path, monkeypatch):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=True)})
        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))
        # Diagnostics page (1) + QA page (1) = 2 image pages.
        assert self._count_image_pages(monkeypatch, datatree, recipe, tmp_path) == 2

    def test_qa_page_omitted_for_currents_without_land(self, tmp_path, monkeypatch):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=False)})
        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))
        assert self._count_image_pages(monkeypatch, datatree, recipe, tmp_path) == 1

    def test_qa_page_omitted_for_non_currents_variable(self, tmp_path, monkeypatch):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=True)})
        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="wind"))
        assert self._count_image_pages(monkeypatch, datatree, recipe, tmp_path) == 1
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_visualization.py::TestPlotRvlLandQa tests/test_visualization.py::TestValidationReportRvlLandQaPage -v`
Expected: `TestPlotRvlLandQa` tests FAIL with `ImportError: cannot import name 'plot_rvl_land_qa'`. `TestValidationReportRvlLandQaPage::test_qa_page_added_for_currents_with_land` FAILS (`assert 1 == 2`); the other two `TestValidationReportRvlLandQaPage` tests may already PASS (no QA page exists yet, so counts are already 1) — that's fine, they'll stay green as regression guards once the feature exists.

- [ ] **Step 3: Implement `plot_rvl_land_qa` and wire it into `validation_report`**

In `sar_validation/core/visualization.py`, find the end of `_finalize_figure_for_report` and the start of `validation_report` (currently lines 1886–1889):

```python
    buf.seek(0)
    return _image_page_figure(plt.imread(buf, format="png"), dpi=dpi)


def validation_report(
```

Replace it with:

```python
    buf.seek(0)
    return _image_page_figure(plt.imread(buf, format="png"), dpi=dpi)


def plot_rvl_land_qa(datatree) -> Optional["plt.Figure"]:
    """
    Build a table figure listing, for every SAR RVL scene with at least one
    land-flagged cell, the land pixel count/fraction and the pre-mask mean
    ``rvlRadVel`` over those land cells (expected ~0 m/s — a meaningfully
    non-zero value signals land contamination worth investigating).

    Reads the ``rvl_land_pixel_count`` / ``rvl_land_pixel_fraction`` /
    ``rvl_land_mean_radvel`` attrs stamped on each SAR node by
    ``DataTreeConverter._extract_rvl_grid_data``.

    Returns
    -------
    matplotlib.figure.Figure or None
        None if *datatree* has no "sar" node, or no scene in it has any
        land-flagged cells — callers should skip adding a page in that case.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if "sar" not in datatree.children:
        return None

    rows = []
    for name, node in datatree["sar"].children.items():
        attrs = node.to_dataset().attrs
        land_count = attrs.get("rvl_land_pixel_count", 0)
        if not land_count:
            continue
        rows.append((
            name,
            int(land_count),
            100 * attrs.get("rvl_land_pixel_fraction", float("nan")),
            attrs.get("rvl_land_mean_radvel", float("nan")),
        ))

    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(rows) + 2))
    ax.axis("off")
    ax.set_title(
        "RVL land-contamination QA — cells masked out of rvlRadVel/rvlRadVelStd",
        fontsize=12, fontweight="bold",
    )
    table = ax.table(
        cellText=[
            [scene, str(count), f"{frac:.1f}%", f"{mean:.4f}"]
            for scene, count, frac, mean in rows
        ],
        colLabels=["Scene", "Land pixels", "Land %", "Mean rvlRadVel over land (m/s)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.tight_layout()
    return fig


```

Then, in `validation_report`, find the collocation-diagnostics block:

```python
    # Collocation diagnostics plot — generated once per recipe
    if base_dir is not None:
        try:
            diag_path = plot_collocation_diagnostics(
                datatree, collocation_ds, recipe, base_dir, filename_suffix
            )
            if diag_path is not None:
                logger.info("Collocation diagnostics plot saved to %s", diag_path)
                # Embed the saved PNG as a page in the combined PDF report —
                # plot_collocation_diagnostics() closes its own figure
                # internally (it's also called standalone from cli.py), so
                # the only way to include it in pdf_pages is to reload the
                # rendered image.
                diag_img = plt.imread(str(diag_path))
                # Lead the report body with the diagnostics overview (the
                # cover page is written separately, so index 0 here becomes
                # the first page after the cover).
                pdf_pages.insert(
                    0,
                    (f"Collocation diagnostics — {recipe.config.name}", _image_page_figure(diag_img)),
                )
        except Exception as exc:
            logger.warning("plot_collocation_diagnostics failed: %s", exc)

    # Combined PDF — saved alongside the validation_statistics_*.nc files
```

Replace it with:

```python
    # Collocation diagnostics plot — generated once per recipe
    if base_dir is not None:
        try:
            diag_path = plot_collocation_diagnostics(
                datatree, collocation_ds, recipe, base_dir, filename_suffix
            )
            if diag_path is not None:
                logger.info("Collocation diagnostics plot saved to %s", diag_path)
                # Embed the saved PNG as a page in the combined PDF report —
                # plot_collocation_diagnostics() closes its own figure
                # internally (it's also called standalone from cli.py), so
                # the only way to include it in pdf_pages is to reload the
                # rendered image.
                diag_img = plt.imread(str(diag_path))
                # Lead the report body with the diagnostics overview (the
                # cover page is written separately, so index 0 here becomes
                # the first page after the cover).
                pdf_pages.insert(
                    0,
                    (f"Collocation diagnostics — {recipe.config.name}", _image_page_figure(diag_img)),
                )
        except Exception as exc:
            logger.warning("plot_collocation_diagnostics failed: %s", exc)

    # RVL land-contamination QA page — currents recipes only, and only when
    # at least one scene actually has land-flagged cells (plot_rvl_land_qa
    # returns None otherwise, so no empty page is added).
    if base_dir is not None and variable == "currents":
        try:
            fig_land_qa = plot_rvl_land_qa(datatree)
            if fig_land_qa is not None:
                pdf_pages.append(
                    ("RVL land-contamination QA", _finalize_figure_for_report(fig_land_qa, None))
                )
        except Exception as exc:
            logger.warning("plot_rvl_land_qa failed: %s", exc)

    # Combined PDF — saved alongside the validation_statistics_*.nc files
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_visualization.py::TestPlotRvlLandQa tests/test_visualization.py::TestValidationReportRvlLandQaPage -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Run the full visualization test file to check for regressions**

Run: `pytest tests/test_visualization.py -v`
Expected: all tests PASS, in particular `TestValidationReportIncludesDiagnostics` and `TestValidationReportCurrentsPointSize` (both call `validation_report` for scenes without RVL QA attrs, so `plot_rvl_land_qa` must return `None` for them and leave existing page counts/behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "feat: add RVL land-contamination QA page to the PDF validation report"
```

---

## Task 5: Full-suite regression check

**Files:** none (verification-only task).

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS, no failures or errors anywhere in the suite (not just the three files touched above — `_extract_rvl_grid_data` is also exercised indirectly via `convert_downloaded_data` in integration-style tests elsewhere in `tests/test_datatree_converter.py`).

- [ ] **Step 2: If everything passes, no commit needed — Task 5 is a checkpoint, not a code change.**
