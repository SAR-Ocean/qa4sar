# Antimeridian (180°) Crossing Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a recipe's `GeographicBounds` express a bbox that wraps through 180° longitude (e.g. `min_lon: 135, max_lon: -120` for `recipes/waves_pacific.yaml`), and make every downloader, the point-data domain crop, and the collocation-diagnostics map handle that convention correctly.

**Architecture:** `GeographicBounds.min_lon > max_lon` means "wraps through 180°". A single new helper, `split_antimeridian_bbox()`, turns that into one or two non-crossing `(lon_min, lon_max)` windows. Every downloader's public entry point loops over those windows against its existing (unchanged) single-window query logic and merges/de-duplicates the results. The point-data domain crop and the collocation-diagnostics map get an OR-based (instead of AND-based) longitude test / a `central_longitude=180` projection for the crossing case.

**Tech Stack:** Python 3.10, xarray, pandas, cartopy 0.25, matplotlib, copernicusmarine 2.4, eumdac, pytest.

## Global Constraints

- `GeographicBounds.min_lon > max_lon` is the *only* signal for "crosses the antimeridian" — there is no new boolean flag or separate field.
- Non-crossing boxes (`min_lon <= max_lon`, the overwhelming majority of existing recipes) must produce byte-for-byte identical behavior to today: same function calls, same output filenames, same return values. Every task's tests must include a non-crossing regression case proving this.
- `split_antimeridian_bbox(min_lon, max_lon)` returns `[(min_lon, max_lon)]` when `min_lon <= max_lon` (including `min_lon == max_lon`), and `[(min_lon, 180.0), (-180.0, max_lon)]` otherwise. This is the single source of truth for the split — no downloader re-implements the crossing check.
- Downloaders whose `download()` currently returns `Optional[Path]` (`HFRadarDownloader`, `HFRadarHistoricalDownloader`, `NOAAHFRadarDownloader`, `InSituDownloader`) change to `list[Path]`, matching the shape `SARDownloader`, `AltimeterDownloader`, `ScatterometerDownloader`, and `RadiometerDownloader` already use. No orchestrator code depends on the old singular shape (verified — `sar_validation/core/orchestrator.py`'s `_download_hf_radar`, `_download_hf_radar_historical`, `_download_noaa_hfradar`, `_download_insitu` never capture `dl.download(...)`'s return value).
- When a downloader must disambiguate output filenames across two windows, use a `_w{i}` suffix inserted before the file extension, and only add it when `len(windows) > 1` — the non-crossing case must keep today's exact filenames.

---

### Task 1: `split_antimeridian_bbox` helper + convention docs

**Files:**
- Modify: `sar_validation/downloaders/base.py`
- Modify: `sar_validation/core/recipe.py:28-37` (`GeographicBounds` docstring)
- Modify: `sar_validation/cli.py:94-99` (`--max-lon` help text)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Produces: `split_antimeridian_bbox(min_lon: float, max_lon: float) -> list[tuple[float, float]]`, importable as `from sar_validation.downloaders.base import split_antimeridian_bbox`. Every later task in this plan imports this function.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, immediately after `class TestDatetimeIntegration:` ends (before the `# Tests for in-situ source-type <-> Copernicus platform-code mapping` comment block that precedes `class TestInsituPlatformCodeMapping:`):

```python
# ---------------------------------------------------------------------------
# Tests for split_antimeridian_bbox()
# ---------------------------------------------------------------------------

class TestSplitAntimeridianBbox:
    def test_non_crossing_bbox_returned_unchanged(self):
        assert split_antimeridian_bbox(-20.0, 0.0) == [(-20.0, 0.0)]

    def test_equal_bounds_treated_as_non_crossing(self):
        assert split_antimeridian_bbox(10.0, 10.0) == [(10.0, 10.0)]

    def test_crossing_bbox_splits_into_two_windows(self):
        assert split_antimeridian_bbox(135.0, -120.0) == [(135.0, 180.0), (-180.0, -120.0)]

    def test_crossing_bbox_windows_are_each_non_crossing(self):
        windows = split_antimeridian_bbox(170.0, -170.0)
        for lo, hi in windows:
            assert lo <= hi
```

Also update the existing import line at the top of `tests/test_downloaders.py`:

```python
from sar_validation.downloaders.base import normalize_datetime, is_date_recent, split_antimeridian_bbox
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestSplitAntimeridianBbox -v`
Expected: FAIL with `ImportError: cannot import name 'split_antimeridian_bbox'`

- [ ] **Step 3: Implement `split_antimeridian_bbox`**

In `sar_validation/downloaders/base.py`, add `"split_antimeridian_bbox"` to the `__all__` list (currently at lines 24-31), then add the function after `build_output_dir` (end of file):

```python
def split_antimeridian_bbox(min_lon: float, max_lon: float) -> list[tuple[float, float]]:
    """
    Split a longitude range into 1 or 2 non-crossing (lon_min, lon_max) windows.

    A recipe's ``GeographicBounds.min_lon > max_lon`` means the bbox wraps
    through the antimeridian (180 degrees) rather than being invalid, e.g.
    ``min_lon=135, max_lon=-120`` covers the Pacific from 135E to 120W.
    Returns the box unchanged (as a single window) when ``min_lon <=
    max_lon``; otherwise returns the two windows ``[min_lon, 180]`` and
    ``[-180, max_lon]`` that together cover the same region without either
    window itself crossing the antimeridian.
    """
    if min_lon <= max_lon:
        return [(min_lon, max_lon)]
    return [(min_lon, 180.0), (-180.0, max_lon)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestSplitAntimeridianBbox -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Update convention docs (no test — doc-only edit)**

In `sar_validation/core/recipe.py`, replace the `GeographicBounds` class (lines 28-37):

```python
@dataclass
class GeographicBounds:
    """Bounding box in decimal degrees (WGS-84).

    ``min_lon > max_lon`` means the box wraps through the antimeridian
    (180 degrees) rather than being invalid — e.g. ``min_lon=135,
    max_lon=-120`` covers the Pacific from 135E to 120W. Downloaders split
    such a bbox into two non-crossing windows internally via
    ``sar_validation.downloaders.base.split_antimeridian_bbox``.
    """
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
```

In `sar_validation/cli.py`, replace the `--max-lon` argument definition (lines 94-99):

```python
    parser.add_argument(
        "--max-lon",
        type=float,
        metavar="DEG",
        help=(
            "Eastern bound in decimal degrees (used with --create-recipe). "
            "If less than --min-lon, the bbox wraps through 180 deg "
            "(e.g. --min-lon 135 --max-lon -120 covers the Pacific)."
        ),
    )
```

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/base.py sar_validation/core/recipe.py sar_validation/cli.py tests/test_downloaders.py
git commit -m "feat: add split_antimeridian_bbox helper and document the crossing convention"
```

---

### Task 2: SAR downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/sar_downloader.py:29-34` (imports), `:88-129` (`query()`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox(min_lon, max_lon) -> list[tuple[float, float]]` (Task 1).
- Produces: `SARDownloader.query(...)` still returns a `pd.DataFrame` (unchanged shape), now de-duplicated on `Id` and covering both windows when crossing. `SARDownloader.download(...)`'s return type (`list[Path]`) is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, as a new section right before `# ---------------------------------------------------------------------------\n# Tests for in-situ source-type <-> Copernicus platform-code mapping` (i.e. directly after Task 1's `TestSplitAntimeridianBbox` class):

```python
# ---------------------------------------------------------------------------
# SARDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestSARDownloaderAntimeridian:
    def _record(self, id_):
        return {
            "Id": id_, "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_query_splits_crossing_bbox_into_two_windows(self, tmp_path):
        from unittest.mock import MagicMock
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_products.side_effect = [
            [self._record("a")], [self._record("b")],
        ]
        dl._client = fake_client

        df = dl.query(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )

        assert fake_client.query_products.call_count == 2
        first_kwargs = fake_client.query_products.call_args_list[0].kwargs
        second_kwargs = fake_client.query_products.call_args_list[1].kwargs
        assert (first_kwargs["min_lon"], first_kwargs["max_lon"]) == (135.0, 180.0)
        assert (second_kwargs["min_lon"], second_kwargs["max_lon"]) == (-180.0, -120.0)
        assert sorted(df["Id"]) == ["a", "b"]

    def test_query_dedupes_product_returned_by_both_windows(self, tmp_path):
        from unittest.mock import MagicMock
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        dup = self._record("dup")
        fake_client.query_products.side_effect = [[dup], [dup]]
        dl._client = fake_client

        df = dl.query(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert len(df) == 1

    def test_query_non_crossing_bbox_calls_once(self, tmp_path):
        from unittest.mock import MagicMock
        from sar_validation.downloaders.sar_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_products.return_value = []
        dl._client = fake_client

        dl.query(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-01-01", end="2026-01-02",
        )
        assert fake_client.query_products.call_count == 1
        kwargs = fake_client.query_products.call_args.kwargs
        assert (kwargs["min_lon"], kwargs["max_lon"]) == (-20.0, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestSARDownloaderAntimeridian -v`
Expected: FAIL — `query()` calls `client.query_products` once with the raw (crossing) bounds, so `call_count == 2` assertions fail.

- [ ] **Step 3: Implement the split in `query()`**

In `sar_validation/downloaders/sar_downloader.py`, update the import block (lines 29-34):

```python
from .base import (
    CopernicusODataClient,
    authenticate_cdse,
    normalize_datetime,
    build_output_dir,
    split_antimeridian_bbox,
)
```

Replace the `query()` method body (lines 88-129):

```python
    def query(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        modes: Optional[list[str]] = None,
        top: int = 100,
    ) -> pd.DataFrame:
        """
        Query available Sentinel-1 L2_OCN products.

        Returns a DataFrame with columns:
            Id, Name, ContentDate_Start, ContentDate_End, ContentLength_GB, Online
        """
        start_norm = normalize_datetime(start) + ".000Z"
        end_norm   = normalize_datetime(end)   + ".000Z"

        client = self._get_client()
        frames = []
        for lo, hi in split_antimeridian_bbox(min_lon, max_lon):
            records = client.query_products(
                collection="SENTINEL-1",
                product_type="OCN",
                start_date=start_norm,
                end_date=end_norm,
                min_lon=lo,
                max_lon=hi,
                min_lat=min_lat,
                max_lat=max_lat,
                top=top,
            )
            frames.append(pd.DataFrame(records))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df.drop_duplicates(subset="Id", keep="first").reset_index(drop=True)

        # Filter by mode if specified
        if modes:
            pattern = "^S1[ABCD]_(" + "|".join(modes) + ")_"
            df = df[df["Name"].str.match(pattern)].reset_index(drop=True)

        return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestSARDownloaderAntimeridian -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS (no pre-existing SAR tests exist, so no regressions possible there; other classes unaffected)

- [ ] **Step 6: Commit**

```bash
git add sar_validation/downloaders/sar_downloader.py tests/test_downloaders.py
git commit -m "feat(sar): split antimeridian-crossing bbox queries into two windows"
```

---

### Task 3: Altimeter downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/altimeter_downloader.py:34` (imports), `:186-244` (download loop)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox` (Task 1).
- Produces: `AltimeterDownloader.download(...)` return type (`list[Path]`) unchanged; output filenames unchanged for non-crossing bboxes, get a `_w{i}` suffix when crossing.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `TestSARDownloaderAntimeridian`:

```python
# ---------------------------------------------------------------------------
# AltimeterDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestAltimeterDownloaderAntimeridian:
    def _patch_subset(self):
        from pathlib import Path
        from unittest.mock import MagicMock

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_crossing_bbox_splits_into_two_windows_with_distinct_filenames(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
                frequencies=["1hz"], satellites=["al"],
            )

        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert (first_kwargs["minimum_longitude"], first_kwargs["maximum_longitude"]) == (135.0, 180.0)
        assert (second_kwargs["minimum_longitude"], second_kwargs["maximum_longitude"]) == (-180.0, -120.0)
        assert first_kwargs["output_filename"] != second_kwargs["output_filename"]
        assert len(paths) == 2

    def test_non_crossing_bbox_keeps_single_call_and_original_filename(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        assert fake_module.subset.call_count == 1
        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["output_filename"] == "cmems_obs-wave_glo_phy-swh_nrt_al-l3_PT1S_2026-06-01_2026-06-02.nc"
        assert len(paths) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestAltimeterDownloaderAntimeridian -v`
Expected: FAIL — `subset.call_count == 1` for the crossing case (no split yet).

- [ ] **Step 3: Implement the split in `download()`**

In `sar_validation/downloaders/altimeter_downloader.py`, update the import (line 34):

```python
from .base import normalize_datetime, build_output_dir, split_antimeridian_bbox
```

Replace the inner `for sat_code in sat_codes:` loop body (lines 186-244):

```python
            for sat_code in sat_codes:
                dataset_id = template.format(sat=sat_code)
                start_d = eff_start_dt.split("T")[0]
                end_d = end_dt.split("T")[0]

                windows = split_antimeridian_bbox(min_lon, max_lon)
                for i, (win_min_lon, win_max_lon) in enumerate(windows):
                    suffix = f"_w{i}" if len(windows) > 1 else ""
                    filename = f"{dataset_id}_{start_d}_{end_d}{suffix}.nc"
                    dest_path = self.output_dir / filename

                    if self.dry_run:
                        print(
                            f"[DRY RUN] Would download {freq.upper()} altimeter data "
                            f"({sat_map[sat_code]}, dataset_id={dataset_id}) to:\n"
                            f"  {dest_path}"
                        )
                        continue

                    print(f"Downloading {freq.upper()} altimeter data ({sat_map[sat_code]}) …")
                    print(f"  Dataset: {dataset_id}")
                    print(f"  Region:  lon [{win_min_lon}, {win_max_lon}] lat [{min_lat}, {max_lat}]")
                    print(f"  Time:    {eff_start_dt} → {end_dt}")

                    try:
                        copernicusmarine.subset(
                            dataset_id=dataset_id,
                            variables=variables,
                            minimum_longitude=win_min_lon,
                            maximum_longitude=win_max_lon,
                            minimum_latitude=min_lat,
                            maximum_latitude=max_lat,
                            start_datetime=eff_start_dt,
                            end_datetime=end_dt,
                            minimum_depth=0,
                            maximum_depth=0,
                            output_directory=self.output_dir,
                            output_filename=filename,
                            force_download=False,
                        )
                        if dest_path.is_dir():
                            # copernicusmarine can't always merge the request
                            # into a single file (e.g. multiple platforms in
                            # one dataset_id) — it then writes a directory
                            # named after output_filename containing one .nc
                            # per platform instead.
                            new_files = sorted(dest_path.rglob("*.nc"))
                            downloaded.extend(new_files)
                            for f in new_files:
                                print(f"  Saved to {f}")
                            if not new_files:
                                print("  No data in this region/time window — skipped.")
                        elif dest_path.exists():
                            downloaded.append(dest_path)
                            print(f"  Saved to {dest_path}")
                        else:
                            # copernicusmarine.subset() writes nothing (and
                            # raises no error) when the satellite's ground
                            # track doesn't cross this region/time window —
                            # expected for most satellites on a small bbox.
                            print("  No data in this region/time window — skipped.")
                    except Exception as exc:
                        print(f"  Skipping {dataset_id}: {exc}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestAltimeterDownloaderAntimeridian -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/altimeter_downloader.py tests/test_downloaders.py
git commit -m "feat(altimeter): split antimeridian-crossing bbox downloads into two windows"
```

---

### Task 4: In-situ downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/insitu_downloader.py:39` (imports), `:146-295` (`download()` / `_download_with_part()`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox` (Task 1).
- Produces: `InSituDownloader.download(...)` return type changes from `Optional[Path]` to `list[Path]`. New private method `InSituDownloader._download_window(...) -> Optional[Path]` (not part of the public API, but later tasks don't depend on it).

**Note:** `_build_csv_filename()` already embeds `min_lon`/`max_lon` in the filename it derives (`f"{abs(min_lon):.2f}{lon_sfx}-{abs(max_lon):.2f}{lon_sfx}_..."`), so each split window — which by construction has different `min_lon`/`max_lon` than the other — already gets a distinct filename with no extra suffix needed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `TestAltimeterDownloaderAntimeridian`:

```python
# ---------------------------------------------------------------------------
# InSituDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestInSituDownloaderAntimeridian:
    def test_download_splits_crossing_bbox_into_two_windows(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader, _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None  # real subset writes to CWD; not needed here

        start_dt, end_dt = "2026-07-02T00:00:00", "2026-07-03T00:00:00"
        for lo, hi in [(135.0, 180.0), (-180.0, -120.0)]:
            fname = _build_csv_filename(lo, hi, -15.0, 30.0, start_dt, end_dt, -20.0, 20.0)
            # Pre-create the destination file so _download_window's
            # "already at dest_path" branch is taken instead of the
            # CWD-relative move (which the fake subset() doesn't produce).
            (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert (first_kwargs["minimum_longitude"], first_kwargs["maximum_longitude"]) == (135.0, 180.0)
        assert (second_kwargs["minimum_longitude"], second_kwargs["maximum_longitude"]) == (-180.0, -120.0)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert paths[0].name != paths[1].name

    def test_non_crossing_bbox_calls_once_and_returns_single_path(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader, _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        assert fake_module.subset.call_count == 1
        assert len(paths) == 1
        assert paths[0].name == fname
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestInSituDownloaderAntimeridian -v`
Expected: FAIL — `download()` currently calls `subset` once and returns a single `Path`/`None`, so both the call-count and `len(paths)` assertions fail (and `dl.download(...)` returning a `Path` object breaks `len(paths)`/`paths[0]` indexing).

- [ ] **Step 3: Implement the split**

In `sar_validation/downloaders/insitu_downloader.py`, update the import (line 39):

```python
from .base import normalize_datetime, is_date_recent, build_output_dir, split_antimeridian_bbox
```

Replace the `download()` method (lines 146-268) with a thin splitting wrapper plus an extracted `_download_window()`:

```python
    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        source_types: Optional[list[str]] = None,
        dataset_part: Optional[str] = None,
    ) -> list[Path]:
        """
        Download in-situ observations and save one CSV per non-crossing
        longitude window (see ``split_antimeridian_bbox``).

        Parameters
        ----------
        source_types : list[str], optional
            Filter by platform type(s): mooring, buoy, ferrybox, drifter, tidal_gauge.
            None or empty list means keep all platform types.
        dataset_part : str, optional
            Which dataset part to use: "monthly" (historical) or "latest" (recent).
            If None, auto-detects based on whether end_date is within 30 days.

        Returns
        -------
        list[Path]
            Paths to the downloaded CSVs (one per window that produced data).
        """
        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for in-situ downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)

        downloaded: list[Path] = []
        for win_min_lon, win_max_lon in split_antimeridian_bbox(min_lon, max_lon):
            path = self._download_window(
                copernicusmarine, win_min_lon, win_max_lon, min_lat, max_lat,
                start_dt, end_dt, source_types, dataset_part,
            )
            if path is not None:
                downloaded.append(path)
        return downloaded

    def _download_window(
        self,
        copernicusmarine,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        source_types: Optional[list[str]],
        dataset_part: Optional[str],
    ) -> Optional[Path]:
        """Download and save one CSV for a single (non-crossing) window."""
        expected_filename = _build_csv_filename(
            min_lon, max_lon, min_lat, max_lat,
            start_dt, end_dt, self.min_depth, self.max_depth,
        )
        dest_path = self.output_dir / expected_filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download in-situ data to:\n  {dest_path}"
            )
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect dataset_part if not provided
        resolved_part = dataset_part
        if resolved_part is None:
            resolved_part = "latest" if is_date_recent(end_dt) else "monthly"

        # Run the copernicusmarine subset call (downloads to CWD by default)
        print(f"Downloading in-situ data …")
        print(f"  Region: lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Depth:  {self.min_depth} to {self.max_depth} m")
        print(f"  Dataset: {resolved_part}")

        # Try initial dataset_part, with fallback if data not available
        try:
            self._download_with_part(
                copernicusmarine,
                resolved_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt,
            )
        except Exception as e:
            error_msg = str(e)
            # Check if error is about data exceeding coordinates (date outside available range)
            if "exceed the dataset coordinates" in error_msg or "out of bounds" in error_msg.lower():
                # Try the opposite dataset_part
                alt_dataset_part = "monthly" if resolved_part == "latest" else "latest"
                print(f"  Retrying with dataset_part='{alt_dataset_part}' due to: {error_msg[:100]}…")
                try:
                    self._download_with_part(
                        copernicusmarine,
                        alt_dataset_part,
                        min_lon, max_lon, min_lat, max_lat,
                        start_dt, end_dt,
                    )
                    resolved_part = alt_dataset_part
                except Exception as e2:
                    # Both failed, raise the second error
                    raise e2
            else:
                # Not a data availability error, re-raise original
                raise

        # Move the file (copernicusmarine writes it to CWD) to our output_dir
        if Path(expected_filename).exists():
            shutil.move(str(expected_filename), str(dest_path))
            print(f"  Saved to {dest_path}")
        elif dest_path.exists():
            print(f"  Already at {dest_path}")
        else:
            # Try to find a recently-created CSV in CWD
            candidates = sorted(Path(".").glob(f"{DATASET_ID}*.csv"), key=os.path.getmtime, reverse=True)
            if candidates:
                shutil.move(str(candidates[0]), str(dest_path))
                print(f"  Saved to {dest_path}")
            else:
                raise FileNotFoundError(
                    f"copernicusmarine download completed but output CSV not found.\n"
                    f"Expected: {expected_filename}"
                )

        # Apply platform-type filter
        if source_types:
            platform_codes = _resolve_platform_codes(source_types)
            if platform_codes:
                df = pd.read_csv(dest_path)
                if "platform_type" in df.columns:
                    df = df[df["platform_type"].isin(platform_codes)]
                    df.to_csv(dest_path, index=False)
                    print(f"  Filtered to {len(df)} rows ({', '.join(source_types)})")

        return dest_path
```

(`_download_with_part` is unchanged — leave it exactly as-is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestInSituDownloaderAntimeridian -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/insitu_downloader.py tests/test_downloaders.py
git commit -m "feat(insitu): split antimeridian-crossing bbox downloads into two windows"
```

---

### Task 5: Scatterometer downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/scatterometer_downloader.py:51` (imports), `:91-174` (`download()`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox` (Task 1).
- Produces: `ScatterometerDownloader.download(...)` return type (`list[Path]`) unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_downloaders.py`, directly after `TestInSituDownloaderAntimeridian`:

```python
# ---------------------------------------------------------------------------
# ScatterometerDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestScatterometerDownloaderAntimeridian:
    def test_dry_run_prints_both_windows(self, tmp_path, capsys):
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert out == []
        captured = capsys.readouterr().out.replace(" ", "")
        assert "[135.0,180.0]" in captured
        assert "[-180.0,-120.0]" in captured

    def test_search_runs_once_per_window_and_dedupes_products(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        # "dup" is returned by both window searches and must be counted once.
        # None of these IDs contain "metopb"/"metopc", so the per-product
        # download loop skips them immediately — this test only exercises
        # the search+dedup logic, not the download loop.
        fake_collection.search.side_effect = [["dup", "east_only"], ["dup", "west_only"]]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            result = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert result == []
        assert fake_collection.search.call_count == 2
        first_kwargs = fake_collection.search.call_args_list[0].kwargs
        second_kwargs = fake_collection.search.call_args_list[1].kwargs
        assert first_kwargs["bbox"] == "135.0,-15.0,180.0,30.0"
        assert second_kwargs["bbox"] == "-180.0,-15.0,-120.0,30.0"
        assert "Found 3 ASCAT products." in capsys.readouterr().out

    def test_non_crossing_bbox_searches_once(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = []
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        assert fake_collection.search.call_count == 1
        assert fake_collection.search.call_args.kwargs["bbox"] == "-20.0,35.0,0.0,60.0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestScatterometerDownloaderAntimeridian -v`
Expected: FAIL — dry-run prints one bbox (the raw crossing one), and `search` is called once instead of twice.

- [ ] **Step 3: Implement the split**

In `sar_validation/downloaders/scatterometer_downloader.py`, update the import (line 51):

```python
from .base import authenticate_eumdac, normalize_datetime, build_output_dir, split_antimeridian_bbox
```

Replace the `download()` method (lines 91-174):

```python
    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        """
        Download ASCAT products that intersect the given region and time window.

        Returns
        -------
        list[Path]
            Paths to the downloaded (and unzipped) files.
        """
        try:
            import eumdac
        except ImportError as exc:
            raise ImportError(
                "eumdac is required for scatterometer downloads.\n"
                "Install it with:  pip install eumdac"
            ) from exc

        start_dt = normalize_datetime(start)
        end_dt   = normalize_datetime(end)
        windows = split_antimeridian_bbox(min_lon, max_lon)

        if self.dry_run:
            for lo, hi in windows:
                print(
                    f"[DRY RUN] Would download ASCAT data\n"
                    f"  Region: lon [{lo},{hi}] lat [{min_lat},{max_lat}]\n"
                    f"  Time:   {start_dt} → {end_dt}\n"
                    f"  Output: {self.output_dir}"
                )
            return []

        token = self._get_token()
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(COLLECTION_ID)

        seen_ids: set[str] = set()
        products = []
        for lo, hi in windows:
            bbox = f"{lo},{min_lat},{hi},{max_lat}"
            for product_id in collection.search(bbox=bbox, dtstart=start_dt, dtend=end_dt):
                key = str(product_id)
                if key not in seen_ids:
                    seen_ids.add(key)
                    products.append(product_id)

        print(f"Found {len(products)} ASCAT products.")
        if not products:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for product_id in products:
            product_str = str(product_id)
            if not any(sat in product_str.lower() for sat in SATELLITES):
                continue

            try:
                product = datastore.get_product(
                    # eumdac's search yields product objects that get_product
                    # accepts here; its stub types product_id as str.
                    product_id=product_id,  # type: ignore[arg-type]
                    collection_id=COLLECTION_ID,
                )
                with product.open() as fsrc:
                    out_path = self.output_dir / fsrc.name
                    print(f"  Downloading {fsrc.name} …")
                    with open(out_path, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)

                if out_path.suffix == ".zip":
                    with zipfile.ZipFile(out_path, "r") as zf:
                        zf.extractall(self.output_dir)
                    out_path.unlink()
                    print(f"  Unzipped to {self.output_dir}")
                else:
                    downloaded.append(out_path)
                    print(f"  Saved to {out_path}")

            except Exception as exc:
                print(f"  ERROR downloading {product_id}: {exc}")

        print(f"Downloaded {len(downloaded)} ASCAT file(s).")
        return downloaded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestScatterometerDownloaderAntimeridian -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sar_validation/downloaders/scatterometer_downloader.py tests/test_downloaders.py
git commit -m "feat(scatterometer): split antimeridian-crossing bbox searches into two windows"
```

---

### Task 6: HF radar (Copernicus grid) downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/hf_radar_downloader.py:39` (imports), `:82-153` (`download()` → split into `download()` + `_download_region_window()`)
- Modify (existing test fixes): `tests/test_downloaders.py` (`TestHFRadarDownloaderGrid`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox` (Task 1).
- Produces: `HFRadarDownloader.download(...)` return type changes from `Optional[Path]` to `list[Path]`. New private method `_download_region_window(region, min_lon, max_lon, min_lat, max_lat, start_dt, end_dt, filename_suffix) -> Optional[Path]`.

**Note:** every entry in `HFR_REGIONS` (`sar_validation/downloaders/_hf_radar_regions.py`) is a small coastal box far from 180°, so at most one split window will ever resolve to a real region for a genuinely crossing recipe. A window that resolves to no region must be *skipped*, not treated as fatal — only re-raise if *no* window resolved.

- [ ] **Step 1: Update the existing tests for the new return type**

In `tests/test_downloaders.py`, inside `class TestHFRadarDownloaderGrid:`:

Change `test_dry_run_prints_resolved_region_and_part`:
```python
    def test_dry_run_prints_resolved_region_and_part(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-06-05", "2026-06-06")
        assert out == []
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "radar-total--US-EastGulfCoast" in captured
```

Change `test_download_calls_subset_with_resolved_region_part`'s assertions:
```python
        assert len(out) == 1
        assert out[0].exists()
        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "monthly-radar-total--US-EastGulfCoast"
        assert kwargs["minimum_longitude"] == -90.0
        assert kwargs["maximum_longitude"] == -60.0
```

Change `test_retries_with_monthly_part_when_latest_out_of_bounds`'s assertions:
```python
        assert len(out) == 1
        assert out[0].exists()
        assert fake_module.subset.call_count == 2
```

(the rest of that test is unchanged)

- [ ] **Step 2: Write the new failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestHFRadarDownloaderGrid:` ends (before the `# Tests for HFRadarHistoricalDownloader` comment block):

```python
class TestHFRadarDownloaderGridAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # lat 0-5 doesn't overlap any HFR_REGIONS entry on either side of
        # the split — the southernmost real region (US-Hawaii) starts at
        # 14.5N, so no window can resolve a region. Note: the *unsplit*
        # pre-fix code also raises a ValueError matching this same message
        # for a min_lon > max_lon input (its overlap-area formula degrades
        # to a spurious negative number for every region), so this test
        # alone doesn't distinguish pre-fix from post-fix — it guards that
        # the "truly nothing covers this" case keeps failing loudly after
        # the fix too. The next test is the one that actually fails pre-fix.
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError, match="No Copernicus HF-radar region overlaps"):
            dl.download(135.0, -120.0, 0.0, 5.0, "2026-07-02", "2026-07-03")

    def test_crossing_bbox_downloads_the_side_that_resolves_to_a_region(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        # US-Alaska's bbox (-174.10..-128.66) overlaps the [-180, -120]
        # window but not the [135, 180] window, so only one window should
        # produce a download.
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(135.0, -120.0, 65.0, 75.0, "2026-01-01", "2026-01-02")

        assert len(out) == 1
        assert fake_module.subset.call_count == 1
        _, kwargs = fake_module.subset.call_args
        assert kwargs["minimum_longitude"] == -180.0
        assert kwargs["maximum_longitude"] == -120.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestHFRadarDownloaderGrid tests/test_downloaders.py::TestHFRadarDownloaderGridAntimeridian -v`
Expected: `TestHFRadarDownloaderGrid` tests FAIL (return-type mismatch). In `TestHFRadarDownloaderGridAntimeridian`, `test_crossing_bbox_with_no_covering_region_on_either_side_raises` already PASSes (both pre- and post-fix code raise a matching `ValueError` for this bbox, just for different underlying reasons); `test_crossing_bbox_downloads_the_side_that_resolves_to_a_region` FAILs, because today's unsplit call to `resolve_hfr_region(135.0, -120.0, 65.0, 75.0)` raises `ValueError` instead of resolving `US-Alaska` from the split `[-180, -120]` window.

- [ ] **Step 4: Implement the split**

In `sar_validation/downloaders/hf_radar_downloader.py`, update the import (line 39):

```python
from .base import normalize_datetime, is_date_recent, build_output_dir, split_antimeridian_bbox
```

Replace `download()` (lines 82-153) with a splitting wrapper plus an extracted `_download_region_window()`:

```python
    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        windows = split_antimeridian_bbox(min_lon, max_lon)

        downloaded: list[Path] = []
        last_error: Optional[ValueError] = None
        resolved_any = False
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            try:
                region = resolve_hfr_region(win_min_lon, win_max_lon, min_lat, max_lat)
            except ValueError as exc:
                if len(windows) == 1:
                    raise
                last_error = exc
                continue
            resolved_any = True
            path = self._download_region_window(
                region, win_min_lon, win_max_lon, min_lat, max_lat,
                start_dt, end_dt, suffix,
            )
            if path is not None:
                downloaded.append(path)

        if not resolved_any and last_error is not None:
            raise last_error
        return downloaded

    def _download_region_window(
        self,
        region: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        filename_suffix: str,
    ) -> Optional[Path]:
        use_latest = HFR_REGIONS[region]["has_latest"] and is_date_recent(end_dt)
        dataset_part = f"{'latest' if use_latest else 'monthly'}-radar-total--{region}"
        filename = _build_filename(region, start_dt, end_dt)
        if filename_suffix:
            filename = filename.replace(".nc", f"{filename_suffix}.nc")
        dest_path = self.output_dir / filename

        if self.dry_run:
            print(
                f"[DRY RUN] Would download Copernicus HF-radar grid for region "
                f"'{region}' (dataset_part='{dataset_part}') to:\n  {dest_path}"
            )
            return None

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for HF radar downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading Copernicus HF-radar surface-current grid …")
        print(f"  Region: {region}")
        print(f"  BBox:   lon [{min_lon}, {max_lon}] lat [{min_lat}, {max_lat}]")
        print(f"  Time:   {start_dt} → {end_dt}")
        print(f"  Dataset part: {dataset_part}")

        try:
            self._subset_with_part(
                copernicusmarine, dataset_part,
                min_lon, max_lon, min_lat, max_lat,
                start_dt, end_dt, dest_path,
            )
        except Exception as e:
            error_msg = str(e)
            if use_latest and (
                "exceed the dataset coordinates" in error_msg
                or "out of bounds" in error_msg.lower()
            ):
                dataset_part = f"monthly-radar-total--{region}"
                print(f"  Retrying with dataset_part='{dataset_part}' due to: {error_msg[:120]}…")
                self._subset_with_part(
                    copernicusmarine, dataset_part,
                    min_lon, max_lon, min_lat, max_lat,
                    start_dt, end_dt, dest_path,
                )
            else:
                raise

        if not dest_path.exists():
            raise FileNotFoundError(
                f"Copernicus HF-radar grid download completed but produced no "
                f"file for region '{region}' in [{start_dt}, {end_dt}] "
                f"(dataset_part='{dataset_part}')."
            )

        print(f"  Saved to {dest_path}")
        return dest_path
```

(`_subset_with_part` and `_build_filename` are unchanged — leave them exactly as-is.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestHFRadarDownloaderGrid tests/test_downloaders.py::TestHFRadarDownloaderGridAntimeridian -v`
Expected: PASS (7 tests total across both classes)

- [ ] **Step 6: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add sar_validation/downloaders/hf_radar_downloader.py tests/test_downloaders.py
git commit -m "feat(hf_radar): split antimeridian-crossing bbox downloads into two windows"
```

---

### Task 7: HF radar historical downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/hf_radar_historical_downloader.py:42` (imports), `:123-211` (`download()` → split into `download()` + `_download_region_window()`)
- Modify (existing test fix): `tests/test_downloaders.py` (`TestHFRadarHistoricalDownloader`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox` (Task 1).
- Produces: `HFRadarHistoricalDownloader.download(...)` return type changes from `Optional[Path]` to `list[Path]`.

- [ ] **Step 1: Update the existing tests for the new return type**

In `tests/test_downloaders.py`, inside `class TestHFRadarHistoricalDownloader:`:

Change `test_dry_run_prints_resolved_region_and_filename`:
```python
    def test_dry_run_prints_resolved_region_and_filename(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2021-06-05", "2021-06-06")
        assert out == []
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc" in captured
```

Change `test_download_gets_file_then_subsets_locally`'s assertions (keep the rest of the test unchanged):
```python
        assert len(out) == 1
        assert out[0].exists()
        result = xr.open_dataset(out[0])
        assert "time" in result.dims and "latitude" in result.dims and "longitude" in result.dims
        assert "DEPTH" not in result.dims
        assert result.sizes["time"] == 5
```

(`test_unavailable_region_raises_clear_error`, `test_multi_year_request_not_yet_supported`, and `test_year_outside_split_archive_range_raises_clear_error` are unaffected — each uses a single, non-crossing bbox, and the error they check for is raised by `_region_filename` before any list/scalar-return distinction matters.)

- [ ] **Step 2: Write the new failing test**

Add to `tests/test_downloaders.py`, directly after `class TestHFRadarHistoricalDownloader:` ends (before `class TestOrchestratorHFRadarHistoricalWiring:`):

```python
class TestHFRadarHistoricalDownloaderAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # lat 0-5 doesn't overlap any HFR_REGIONS entry on either side of
        # the split (the southernmost real region, US-Hawaii, starts at
        # 14.5N). Note: the unsplit pre-fix code also raises a ValueError
        # matching this message for a min_lon > max_lon input, so this test
        # alone doesn't distinguish pre-fix from post-fix — it guards that
        # the "truly nothing covers this" case keeps failing loudly after
        # the fix too. The next test is the one that actually fails pre-fix.
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError, match="No Copernicus HF-radar region overlaps"):
            dl.download(135.0, -120.0, 0.0, 5.0, "2021-07-02", "2021-07-03")

    def test_crossing_bbox_dry_run_resolves_the_side_that_has_a_region(self, tmp_path, capsys):
        # US-Alaska's bbox (-174.10..-128.66, 68.01..74.03) overlaps the
        # [-180, -120] window but not the [135, 180] window, so only that
        # window should resolve a region.
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(135.0, -120.0, 69.0, 73.0, "2021-07-02", "2021-07-03")
        assert out == []
        assert "US-Alaska" in capsys.readouterr().out
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestHFRadarHistoricalDownloader tests/test_downloaders.py::TestHFRadarHistoricalDownloaderAntimeridian -v`
Expected: `TestHFRadarHistoricalDownloader` FAILs on the two updated assertions. In `TestHFRadarHistoricalDownloaderAntimeridian`, `test_crossing_bbox_with_no_covering_region_on_either_side_raises` already PASSes; `test_crossing_bbox_dry_run_resolves_the_side_that_has_a_region` FAILs, because today's unsplit call to `resolve_hfr_region(135.0, -120.0, 69.0, 73.0)` raises `ValueError` instead of resolving `US-Alaska` from the split `[-180, -120]` window.

- [ ] **Step 4: Implement the split**

In `sar_validation/downloaders/hf_radar_historical_downloader.py`, update the import (line 42):

```python
from .base import normalize_datetime, build_output_dir, split_antimeridian_bbox
```

Replace `download()` (lines 123-211) with a splitting wrapper plus an extracted `_download_region_window()`:

```python
    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
    ) -> list[Path]:
        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)
        windows = split_antimeridian_bbox(min_lon, max_lon)

        downloaded: list[Path] = []
        last_error: Optional[ValueError] = None
        resolved_any = False
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            try:
                region = resolve_hfr_region(win_min_lon, win_max_lon, min_lat, max_lat)
            except ValueError as exc:
                if len(windows) == 1:
                    raise
                last_error = exc
                continue
            resolved_any = True
            path = self._download_region_window(
                region, win_min_lon, win_max_lon, min_lat, max_lat,
                start_dt, end_dt, suffix,
            )
            if path is not None:
                downloaded.append(path)

        if not resolved_any and last_error is not None:
            raise last_error
        return downloaded

    def _download_region_window(
        self,
        region: str,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start_dt: str,
        end_dt: str,
        filename_suffix: str,
    ) -> Optional[Path]:
        remote_filename = _region_filename(region, start_dt, end_dt)

        start_d = start_dt.split("T")[0]
        end_d = end_dt.split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}-{end_d}"
        dest_path = self.output_dir / f"{DATASET_ID}_{region}_{date_str}{filename_suffix}.nc"

        if self.dry_run:
            print(
                f"[DRY RUN] Would fetch Copernicus HF-radar historical archive "
                f"'{remote_filename}' for region '{region}' and subset to:\n  {dest_path}"
            )
            return None

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for HF radar downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc
        import xarray as xr

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_cache_dir = self.output_dir / "_raw_archive"
        raw_cache_dir.mkdir(parents=True, exist_ok=True)

        print("Fetching Copernicus HF-radar delayed-mode archive …")
        print(f"  Region: {region}")
        print(f"  Archive file: {remote_filename}")
        resp = copernicusmarine.get(
            dataset_id=DATASET_ID,
            filter=f"*{remote_filename}",
            output_directory=str(raw_cache_dir),
            no_directories=True,
            skip_existing=True,
            disable_progress_bar=True,
        )
        if not resp.files:
            raise FileNotFoundError(
                f"No archive file matched '{remote_filename}' for region '{region}'."
            )
        raw_path = Path(resp.files[0].file_path)

        raw = xr.open_dataset(raw_path)
        try:
            # Keep EWCT/NSCT plus every ancillary uncertainty/QC field the
            # converter (Task 3) knows how to retain — standard deviations
            # (EWCS/NSCS), the geometric-dilution field (GDOP), the overall
            # QCflag, and each per-parameter QC flag — whichever of these
            # this archive file actually has.
            _ancillary_vars = (
                "GDOP", "EWCS", "NSCS", "QCflag",
                "CSPD_QC", "DDNS_QC", "GDOP_QC", "VART_QC", "POSITION_QC",
            )
            normalized = (
                raw[["EWCT", "NSCT"] + [v for v in _ancillary_vars if v in raw]]
                .squeeze("DEPTH", drop=True)
                .rename({"TIME": "time", "LATITUDE": "latitude", "LONGITUDE": "longitude"})
                .sortby(["latitude", "longitude"])
                .sel(
                    time=slice(start_dt, end_dt),
                    latitude=slice(min_lat, max_lat),
                    longitude=slice(min_lon, max_lon),
                )
            )
            if normalized.sizes.get("time", 0) == 0:
                raise FileNotFoundError(
                    f"Copernicus HF-radar historical archive for region '{region}' has "
                    f"no data in [{start_dt}, {end_dt}]."
                )
            normalized.to_netcdf(dest_path)
        finally:
            raw.close()

        print(f"  Saved to {dest_path}")
        return dest_path
```

(`_region_filename` is unchanged — leave it exactly as-is.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestHFRadarHistoricalDownloader tests/test_downloaders.py::TestHFRadarHistoricalDownloaderAntimeridian -v`
Expected: PASS

- [ ] **Step 6: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add sar_validation/downloaders/hf_radar_historical_downloader.py tests/test_downloaders.py
git commit -m "feat(hf_radar_historical): split antimeridian-crossing bbox downloads into two windows"
```

---

### Task 8: NOAA HF radar (ERDDAP) downloader — antimeridian splitting

**Files:**
- Modify: `sar_validation/downloaders/noaa_hfradar_downloader.py:30` (imports), `:177-200` (`download()` → split into `download()` + `_download_window()`)
- Modify (existing test fixes): `tests/test_downloaders.py` (`TestNOAAHFRadarDownload`)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `split_antimeridian_bbox` (Task 1).
- Produces: `NOAAHFRadarDownloader.download(...)` return type changes from `Optional[Path]` to `list[Path]`.

- [ ] **Step 1: Update the existing tests for the new return type**

In `tests/test_downloaders.py`, inside `class TestNOAAHFRadarDownload:`:

Change `test_dry_run_returns_none_and_no_fetch`:
```python
    def test_dry_run_returns_empty_list_and_no_fetch(self, tmp_path, capsys):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert out == []
        m.assert_not_called()
        assert "ucsdHfrW6.nc?" in capsys.readouterr().out
```

Change `test_download_fetches_url_to_expected_path`'s assertions (keep the rest unchanged):
```python
        assert len(out) == 1
        assert out[0].parent == tmp_path
        assert out[0].suffix == ".nc"
        m.assert_called_once()
        called_url, called_path = m.call_args[0][0], m.call_args[0][1]
        assert "ucsdHfrW6.nc?" in called_url
        assert str(out[0]) == str(called_path)
```

(`test_download_clamps_bbox_extending_past_region_edge` and `test_download_clamps_west_coast_recipe_bbox` don't inspect the return value — unaffected.)

- [ ] **Step 2: Write the new failing tests**

Add to `tests/test_downloaders.py`, directly after `class TestNOAAHFRadarDownload:` ends (before the `# Tests for DataOrchestrator "hf_radar_noaa" wiring` comment block):

```python
class TestNOAAHFRadarDownloaderAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # 135E..120W doesn't overlap US_WEST or US_EAST_GULF on either side
        # of the split (NOAA's _match_region uses each window's *center*
        # point, and neither window's center falls inside either region).
        # Note: the unsplit pre-fix code also raises a ValueError matching
        # this message for a min_lon > max_lon input (its own center-point
        # math just lands on a different, still-uncovered point), so this
        # test alone doesn't distinguish pre-fix from post-fix — it guards
        # that the "truly nothing covers this" case keeps failing loudly
        # after the fix too. The next test is the one that actually fails
        # pre-fix.
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            with pytest.raises(ValueError, match="No ERDDAP HF-radar dataset"):
                dl.download(135.0, -120.0, -15.0, 30.0, _RECENT_START, _RECENT_END)
        m.assert_not_called()

    def test_crossing_bbox_downloads_the_side_whose_window_center_resolves(self, tmp_path):
        # NOAA's region match is center-point-based (not overlap-area, unlike
        # the Copernicus HFR regions), so only a window whose *own* center
        # (after splitting) lands inside a supported region resolves. Here
        # min_lon=179, max_lon=-66 splits into [179, 180] (center 179.5,
        # 36.5 — matches nothing) and [-180, -66] (center -123.0, 36.5 —
        # inside US_WEST's bbox). The raw (unsplit) request's own center,
        # (56.5, 36.5), matches nothing — that's what makes this fail today.
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlretrieve"
        ) as m:
            out = dl.download(179.0, -66.0, 35.0, 38.0, _RECENT_START, _RECENT_END)
        assert len(out) == 1
        m.assert_called_once()
        called_url = m.call_args[0][0]
        assert "ucsdHfrW6.nc?" in called_url
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestNOAAHFRadarDownload tests/test_downloaders.py::TestNOAAHFRadarDownloaderAntimeridian -v`
Expected: `TestNOAAHFRadarDownload` FAILs on the two updated assertions. In `TestNOAAHFRadarDownloaderAntimeridian`, `test_crossing_bbox_with_no_covering_region_on_either_side_raises` already PASSes; `test_crossing_bbox_downloads_the_side_whose_window_center_resolves` FAILs, because today's unsplit call computes the whole request's center (56.5, 36.5), which matches no region, instead of splitting and finding that the `[-180, -66]` window's own center matches `US_WEST`.

- [ ] **Step 4: Implement the split**

In `sar_validation/downloaders/noaa_hfradar_downloader.py`, update the import (line 30):

```python
from .base import normalize_datetime, split_antimeridian_bbox
```

Replace the `download()` method (lines 177-200):

```python
    def download(self, min_lon, max_lon, min_lat, max_lat,
                 start: str, end: str) -> list[Path]:
        windows = split_antimeridian_bbox(min_lon, max_lon)
        downloaded: list[Path] = []
        last_error: Optional[ValueError] = None
        resolved_any = False
        for i, (win_min_lon, win_max_lon) in enumerate(windows):
            suffix = f"_w{i}" if len(windows) > 1 else ""
            try:
                path = self._download_window(
                    win_min_lon, win_max_lon, min_lat, max_lat, start, end, suffix,
                )
            except ValueError as exc:
                if len(windows) == 1:
                    raise
                last_error = exc
                continue
            resolved_any = True
            if path is not None:
                downloaded.append(path)

        if not resolved_any and last_error is not None:
            raise last_error
        return downloaded

    def _download_window(
        self, min_lon, max_lon, min_lat, max_lat, start: str, end: str, filename_suffix: str,
    ) -> Optional[Path]:
        backend = select_backend(end)  # raises if archive (Phase 3b) needed
        dataset_id = select_erddap_dataset(
            min_lon, max_lon, min_lat, max_lat, self.resolution_km
        )
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(
            min_lon, max_lon, min_lat, max_lat
        )
        url = build_erddap_subset_url(
            dataset_id, min_lon, max_lon, min_lat, max_lat, start, end
        )

        if self.dry_run:
            print(f"[dry-run] NOAA HF-radar ({backend}) would download:\n  {url}")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        start_d = normalize_datetime(start).split("T")[0]
        end_d = normalize_datetime(end).split("T")[0]
        date_str = start_d if start_d == end_d else f"{start_d}_{end_d}"
        out_path = self.output_dir / f"{dataset_id}_{self.resolution_km}km_{date_str}{filename_suffix}.nc"
        urllib.request.urlretrieve(url, str(out_path))
        return out_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_downloaders.py::TestNOAAHFRadarDownload tests/test_downloaders.py::TestNOAAHFRadarDownloaderAntimeridian -v`
Expected: PASS

- [ ] **Step 6: Run the full downloader test file to check for regressions**

Run: `pytest tests/test_downloaders.py -v`
Expected: PASS (all tests, including `TestOrchestratorHFRadarNOAAWiring`)

- [ ] **Step 7: Commit**

```bash
git add sar_validation/downloaders/noaa_hfradar_downloader.py tests/test_downloaders.py
git commit -m "feat(noaa_hfradar): split antimeridian-crossing bbox downloads into two windows"
```

---

### Task 9: Dateline-aware point-data domain crop

**Files:**
- Modify: `sar_validation/core/datatree_converter.py:81-87`
- Test: `tests/test_datatree_converter.py`

**Interfaces:**
- Consumes: nothing new (pure logic change inside `_subset_point_ds`, already imported by its test file).
- Produces: `_subset_point_ds(...)` behavior unchanged for `min_lon <= max_lon`; for `min_lon > max_lon` it now keeps points on *either* side of the antimeridian (union) instead of discarding everything (empty intersection).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_datatree_converter.py`, directly after the `class TestSubsetPointDs:` block ends (before `def _make_scatterometer_nc_at(...)`):

```python
_CROSSING_SUBSET_KW = dict(
    min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
    t_start="2026-07-02", t_end="2026-07-03",
    buffer_km=25.0, time_tolerance_minutes=180,
)


class TestSubsetPointDsAntimeridian:
    def test_keeps_points_on_both_sides_of_the_dateline(self):
        ds = _make_point_ds(
            lons=[170.0, -170.0], lats=[0.0, 0.0],
            times=["2026-07-02T12:00"] * 2,
        )
        out = _subset_point_ds(ds, **_CROSSING_SUBSET_KW)
        assert out.sizes["point"] == 2

    def test_drops_points_in_the_excluded_middle(self):
        ds = _make_point_ds(
            lons=[0.0, 45.0], lats=[0.0, 0.0],
            times=["2026-07-02T12:00"] * 2,
        )
        assert _subset_point_ds(ds, **_CROSSING_SUBSET_KW) is None

    def test_keeps_latitude_filtering_alongside_the_lon_union(self):
        ds = _make_point_ds(
            lons=[170.0, 170.0], lats=[0.0, 80.0],
            times=["2026-07-02T12:00"] * 2,
        )
        out = _subset_point_ds(ds, **_CROSSING_SUBSET_KW)
        assert out.sizes["point"] == 1
        assert float(out["lat"].values[0]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datatree_converter.py::TestSubsetPointDsAntimeridian -v`
Expected: FAIL — the current AND-based mask treats `min_lon=135.0 > max_lon=-120.0` as an impossible range, so every point is dropped (`test_keeps_points_on_both_sides_of_the_dateline` and `test_keeps_latitude_filtering_alongside_the_lon_union` get `None`/0 points instead of the expected counts).

- [ ] **Step 3: Implement the dateline-aware mask**

In `sar_validation/core/datatree_converter.py`, replace lines 81-87:

```python
    deg_buf = buffer_km / 55.0
    lon = ds["lon"].values
    lat = ds["lat"].values
    if min_lon <= max_lon:
        lon_mask = (lon >= min_lon - deg_buf) & (lon <= max_lon + deg_buf)
    else:
        # Antimeridian-crossing bbox (GeographicBounds.min_lon > max_lon):
        # valid longitudes are the union of the two wrap-around windows,
        # not their (empty) intersection.
        lon_mask = (lon >= min_lon - deg_buf) | (lon <= max_lon + deg_buf)
    mask = (
        lon_mask
        & (lat >= min_lat - deg_buf) & (lat <= max_lat + deg_buf)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datatree_converter.py::TestSubsetPointDsAntimeridian tests/test_datatree_converter.py::TestSubsetPointDs -v`
Expected: PASS (all tests in both classes — confirms the non-crossing path is untouched)

- [ ] **Step 5: Run the full datatree_converter test file to check for regressions**

Run: `pytest tests/test_datatree_converter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/datatree_converter.py tests/test_datatree_converter.py
git commit -m "fix(datatree_converter): union-based longitude crop for antimeridian-crossing bounds"
```

---

### Task 10: Dateline-aware collocation-diagnostics map

**Files:**
- Modify: `sar_validation/core/visualization.py:176-239` (`_set_lonlat_ticks`), `:1528-1542` (map/extent setup inside `plot_collocation_diagnostics`)
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `recipe.config.geographic_bounds` (already available inside `plot_collocation_diagnostics` as `bounds`, line ~1508).
- Produces: no signature changes. `plot_collocation_diagnostics(...)` still returns `Path | None`.

**Verified empirically** (see plan investigation, not re-derived here): for a crossing bbox, using `ccrs.PlateCarree(central_longitude=180)` as the axes projection and shifting each true longitude `L` via `(L % 360) - 180` before calling `ax.set_extent(..., crs=proj)` produces the correct (non-wrapped) extent. `_set_lonlat_ticks`'s existing `ax.set_xticks(xticks, crs=ccrs.PlateCarree())` hardcodes the *unshifted* CRS, which is wrong once the axes projection itself is shifted — replacing that hardcoded `ccrs.PlateCarree()` with `ax.projection` fixes tick labels for both the crossing and non-crossing cases (confirmed: with `crs=ax.projection`, tick values `-30, 0, 30, 60` in the shifted axes frame correctly label as `150°E, 180°, 150°W, 120°W`; the same code path with a non-shifted `ax.projection` behaves exactly as it does today, since `ax.projection == ccrs.PlateCarree()` in that case).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_visualization.py`. First, add two new fixtures directly after the `diagnostics_recipe` fixture (which ends around line 840, right before `@pytest.fixture\ndef diagnostics_recipe_waves():`):

```python
@pytest.fixture
def geo_datatree_and_collocation_dateline():
    """Synthetic DataTree + collocation_ds whose SAR scene and validation
    points straddle the antimeridian (170E..170W), used to test
    plot_collocation_diagnostics' dateline-crossing map extent."""
    from sar_validation.core.datatree_converter import DataTreeConverter

    lon2d = np.array([
        [170.0, 175.0, 180.0, -175.0, -170.0],
        [170.0, 175.0, 180.0, -175.0, -170.0],
    ])
    lat2d = np.array([
        [-2.0, -2.0, -2.0, -2.0, -2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ])
    wind = np.linspace(5.0, 12.0, lon2d.size).reshape(lon2d.shape)
    sar_ds = xr.Dataset(
        {"owiWindSpeed": (("y", "x"), wind)},
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
            "time": pd.Timestamp("2026-07-02T12:00:00"),
        },
    )

    n = 4
    mooring_ds = xr.Dataset(
        {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
        coords={
            "lon": ("point", np.array([172.0, 178.0, -178.0, -172.0])),
            "lat": ("point", np.array([-1.0, -0.5, 0.5, 1.0])),
            "time": ("point", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
        },
        attrs={"platform_type": "mooring"},
    )

    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": sar_ds,
        "validation/mooring": mooring_ds,
    })

    collocation_ds = xr.Dataset({
        "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.9, 8.2, 9.3])),
        "val_WSPD":                    ("collocation", np.array([6.0, 7.0, 8.0, 9.5])),
        "val_source":                  ("collocation", ["mooring"] * n),
        "sar_scene_name":              ("collocation", ["sceneA"] * n),
        "val_lon":                     ("collocation", np.array([172.0, 178.0, -178.0, -172.0])),
        "val_lat":                     ("collocation", np.array([-1.0, -0.5, 0.5, 1.0])),
        "val_id":                      ("collocation", ["mo0", "mo1", "mo2", "mo3"]),
        "temporal_distance_minutes":   ("collocation", np.array([10.0, 20.0, 30.0, 40.0])),
    })
    collocation_ds = collocation_ds.assign_coords(
        val_time=("collocation", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
    )
    return datatree, collocation_ds


@pytest.fixture
def diagnostics_recipe_dateline():
    from sar_validation.core.recipe import (
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
    )
    config = RecipeConfig(
        name="test_recipe_dateline",
        variable="wind",
        geographic_bounds=GeographicBounds(min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0),
        validation_sources=[ValidationDataSource(source_type="mooring")],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)
```

Then add a new test class at the end of the file:

```python
class TestPlotCollocationDiagnosticsAntimeridian:
    def test_uses_central_longitude_180_projection_when_crossing(
        self, geo_datatree_and_collocation_dateline, diagnostics_recipe_dateline, tmp_path, monkeypatch
    ):
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation_dateline
        seen_ax = []
        original = viz._set_lonlat_ticks

        def spy(ax, gl):
            seen_ax.append(ax)
            return original(ax, gl)

        monkeypatch.setattr(viz, "_set_lonlat_ticks", spy)
        out_path = viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe_dateline, tmp_path,
        )

        assert out_path is not None
        assert len(seen_ax) == 1
        assert seen_ax[0].projection.proj4_params.get("lon_0") == 180

    def test_non_crossing_recipe_keeps_default_projection(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation
        seen_ax = []
        original = viz._set_lonlat_ticks

        def spy(ax, gl):
            seen_ax.append(ax)
            return original(ax, gl)

        monkeypatch.setattr(viz, "_set_lonlat_ticks", spy)
        viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )

        assert len(seen_ax) == 1
        assert seen_ax[0].projection.proj4_params.get("lon_0", 0) == 0

    def test_produces_a_valid_png_for_crossing_bbox(
        self, geo_datatree_and_collocation_dateline, diagnostics_recipe_dateline, tmp_path
    ):
        import matplotlib.image as mpimg
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation_dateline
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe_dateline, tmp_path,
        )
        img = mpimg.imread(str(out_path))
        assert img.shape[0] > 100 and img.shape[1] > 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsAntimeridian -v`
Expected: FAIL on `test_uses_central_longitude_180_projection_when_crossing` (`proj4_params.get("lon_0") == 0.0`, not `180`, since the axes projection is unconditionally `ccrs.PlateCarree()` today).

- [ ] **Step 3: Implement the dateline-aware projection, extent, and tick fix**

In `sar_validation/core/visualization.py`, replace the two `ax.set_xticks`/`ax.set_yticks` lines inside `_set_lonlat_ticks` (lines 238-239):

```python
    ax.set_xticks(xticks, crs=ax.projection)
    ax.set_yticks(yticks, crs=ax.projection)
```

Then replace the map-setup block inside `plot_collocation_diagnostics` (lines 1528-1542):

```python
    # ── Create geographic plot ──────────────────────────────────────────
    # A crossing bbox (min_lon > max_lon, see GeographicBounds' antimeridian
    # convention) is centered on 180 deg instead of Greenwich, so the map
    # itself doesn't get cut at the dateline.
    crosses_dateline = bounds.min_lon > bounds.max_lon
    proj = ccrs.PlateCarree(central_longitude=180) if crosses_dateline else ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    transform = ccrs.PlateCarree()

    # Add coastlines and features
    land, coastline = _land_coastline_features()
    ax.add_feature(land, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(coastline, linewidth=0.5, zorder=0)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    # ── Set plot extent to the recipe's geographic bounds ────────────────
    if crosses_dateline:
        # In the central_longitude=180 axes frame, true longitude L maps to
        # (L % 360) - 180, which turns the wrapped [min_lon, 180] +
        # [-180, max_lon] range into one contiguous span with no wraparound.
        def _shift(lon: float) -> float:
            return (lon % 360) - 180
        ax.set_extent(
            [_shift(bounds.min_lon), _shift(bounds.max_lon), bounds.min_lat, bounds.max_lat],
            crs=proj,
        )
    else:
        ax.set_extent([bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat],
                      crs=transform)
    _set_lonlat_ticks(ax, gl)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_visualization.py::TestPlotCollocationDiagnosticsAntimeridian -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full visualization test file to check for regressions**

Run: `pytest tests/test_visualization.py -v`
Expected: PASS — in particular `TestPlotCollocationDiagnosticsTicks` and every other `TestPlotCollocationDiagnostics*` class, since `crosses_dateline` is `False` for all of their (non-crossing) fixtures and `ax.projection` for those cases is identical to the previous hardcoded `ccrs.PlateCarree()`.

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/visualization.py tests/test_visualization.py
git commit -m "fix(visualization): central-longitude-180 projection for antimeridian-crossing diagnostics map"
```

---

### Task 11: Update `waves_pacific.yaml` and add an orchestrator-level integration test

**Files:**
- Modify: `recipes/waves_pacific.yaml`
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: everything from Tasks 1-10 (this is the end-to-end confirmation task).
- Produces: nothing new — this task only proves the pieces already built compose correctly for the recipe that motivated this plan.

- [ ] **Step 1: Update the recipe to use the crossing convention**

In `recipes/waves_pacific.yaml`, change the `geographic_bounds` block:

```yaml
geographic_bounds:
  min_lon: 135.0
  max_lon: -120.0
  min_lat: -15.0
  max_lat: 30.0
```

(only `max_lon` changes, from `240.0` to `-120.0`)

- [ ] **Step 2: Write the failing integration test**

Add to `tests/test_downloaders.py`, at the end of the file:

```python
# ---------------------------------------------------------------------------
# End-to-end: orchestrator wiring for a Pacific-crossing recipe
# ---------------------------------------------------------------------------

class TestOrchestratorAntimeridianDryRun:
    def test_pacific_crossing_recipe_wires_through_without_error(self, tmp_path):
        from unittest.mock import patch
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            Recipe, RecipeConfig, GeographicBounds, TemporalBounds,
            SARDataSpec, ValidationDataSource,
        )

        cfg = RecipeConfig(
            name="pacific_dry_run_test",
            variable="waves",
            geographic_bounds=GeographicBounds(min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0),
            temporal_bounds=TemporalBounds(start="2026-07-02", end="2026-07-03"),
            sar_data=SARDataSpec(swath_mode=["WV", "SM"]),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="tidal_gauge"),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="altimeter"),
            ],
            output_dir=str(tmp_path),
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls, patch(
            "sar_validation.downloaders.altimeter_downloader.AltimeterDownloader"
        ) as mock_alt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_insitu_cls.return_value.download.return_value = []
            mock_alt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        _, sar_kwargs = mock_sar_cls.return_value.download.call_args
        assert (sar_kwargs["min_lon"], sar_kwargs["max_lon"]) == (135.0, -120.0)
        _, insitu_kwargs = mock_insitu_cls.return_value.download.call_args
        assert (insitu_kwargs["min_lon"], insitu_kwargs["max_lon"]) == (135.0, -120.0)
        _, alt_kwargs = mock_alt_cls.return_value.download.call_args
        assert (alt_kwargs["min_lon"], alt_kwargs["max_lon"]) == (135.0, -120.0)

    def test_waves_pacific_recipe_loads_with_crossing_convention(self):
        from sar_validation.core.recipe import Recipe

        recipe = Recipe.from_yaml("recipes/waves_pacific.yaml")
        bounds = recipe.config.geographic_bounds
        assert bounds.min_lon == 135.0
        assert bounds.max_lon == -120.0
        assert bounds.min_lon > bounds.max_lon  # crossing convention
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_downloaders.py::TestOrchestratorAntimeridianDryRun -v`
Expected: `test_waves_pacific_recipe_loads_with_crossing_convention` FAILs (`bounds.max_lon == 240.0`, the recipe hasn't been updated yet if Step 1 wasn't applied — confirm it passes once Step 1 is done). `test_pacific_crossing_recipe_wires_through_without_error` should already PASS at this point since it doesn't depend on any of Tasks 1-10 (the orchestrator never needed changes — splitting is entirely internal to each downloader) — running it now is a sanity check that orchestrator wiring was never the problem.

- [ ] **Step 4: Confirm both tests pass**

Run: `pytest tests/test_downloaders.py::TestOrchestratorAntimeridianDryRun -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the entire test suite**

Run: `pytest -v`
Expected: PASS (no regressions anywhere in the suite)

- [ ] **Step 6: Manual smoke test with the real CLI**

Run: `python -m sar_validation.cli --recipe recipes/waves_pacific.yaml --dry-run`
Expected: the command prints per-source dry-run plans (SAR query, altimeter windows, in-situ region) that each show two lon windows (`[135.0, 180.0]` and `[-180.0, -120.0]`) somewhere in the output, and the command exits 0 with no traceback. This confirms the fix end-to-end for the exact recipe that motivated this plan.

- [ ] **Step 7: Commit**

```bash
git add recipes/waves_pacific.yaml tests/test_downloaders.py
git commit -m "test: verify waves_pacific.yaml and orchestrator wiring for antimeridian crossing"
```

---

## Post-plan verification checklist

- [ ] `pytest -v` passes in full (all tasks' tests plus the pre-existing suite).
- [ ] `python -m sar_validation.cli --recipe recipes/waves_pacific.yaml --dry-run` runs clean (Task 11, Step 6).
- [ ] Re-read `docs/superpowers/specs/2026-07-17-antimeridian-crossing-design.md` and confirm every section (`Convention`, `Design §1-4`, `Testing`) has a corresponding task above.
