# Optional min_depth/max_depth in Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `min_depth`/`max_depth` optional on `ValidationDataSource` so recipes that don't need a custom depth window never mention them, while the Copernicus Marine download still defaults to -20/20 when unspecified.

**Architecture:** `ValidationDataSource.min_depth`/`max_depth` become `Optional[float] = None`. Two new module constants (`DEFAULT_MIN_DEPTH`/`DEFAULT_MAX_DEPTH`) hold the fallback, and two new properties (`resolved_min_depth`/`resolved_max_depth`) apply it. Serialization (`to_dict`) omits the keys when `None`; deserialization (`_from_dict`) no longer forces a default. `orchestrator.py`'s two call sites switch from the raw fields to the resolved properties.

**Tech Stack:** Python 3, `dataclasses`, `pyyaml`, `pytest`, `unittest.mock`.

## Global Constraints

- Depth fallback value is exactly `-20.0` / `20.0` (from the spec).
- `HFRadarDownloader`/`InSituDownloader` constructor defaults and their standalone CLIs (`_parse_args`/`main`) are NOT touched — out of scope per spec.
- `cli.py`'s recipe-template builders (`_build_currents_config` etc.) are NOT touched — they already omit `min_depth`/`max_depth` for sources that don't override it.
- No new dependencies.

---

### Task 1: Optional depth fields, resolution properties, and serialization in `recipe.py`

**Files:**
- Modify: `sar_validation/core/recipe.py:50-76` (`ValidationDataSource` dataclass), `sar_validation/core/recipe.py:345-354` (`Recipe._from_dict` validation_sources list)
- Test: `tests/test_recipe.py`

**Interfaces:**
- Produces: `sar_validation.core.recipe.DEFAULT_MIN_DEPTH: float = -20.0`, `sar_validation.core.recipe.DEFAULT_MAX_DEPTH: float = 20.0`; `ValidationDataSource.min_depth: Optional[float]`, `ValidationDataSource.max_depth: Optional[float]`; `ValidationDataSource.resolved_min_depth -> float`, `ValidationDataSource.resolved_max_depth -> float` (properties); `ValidationDataSource.to_dict() -> Dict[str, Any]` (now omits `min_depth`/`max_depth` keys when `None`).
- Consumes: nothing new (pure change to existing `recipe.py`).

- [ ] **Step 1: Write failing tests for the dataclass defaults, resolution properties, and serialization**

Replace the existing `test_depth_defaults` test and add new tests in `tests/test_recipe.py`. Find this block (around line 134-146):

```python
class TestValidationSourceParsing:
    def test_depth_defaults(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.min_depth == -20.0
        assert src.max_depth == 20.0

    def test_custom_depth(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.min_depth == -2.0

    def test_collocation_kwargs_default_empty(self):
        src = ValidationDataSource(source_type="buoy")
        assert src.collocation_kwargs == {}
```

Replace it with:

```python
class TestValidationSourceParsing:
    def test_depth_defaults_to_none(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.min_depth is None
        assert src.max_depth is None

    def test_resolved_depth_falls_back_to_defaults(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.resolved_min_depth == -20.0
        assert src.resolved_max_depth == 20.0

    def test_resolved_depth_uses_explicit_value(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.resolved_min_depth == -2.0
        assert src.resolved_max_depth == 2.0

    def test_custom_depth(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.min_depth == -2.0

    def test_collocation_kwargs_default_empty(self):
        src = ValidationDataSource(source_type="buoy")
        assert src.collocation_kwargs == {}

    def test_to_dict_omits_unspecified_depth(self):
        src = ValidationDataSource(source_type="scatterometer")
        d = src.to_dict()
        assert "min_depth" not in d
        assert "max_depth" not in d

    def test_to_dict_includes_explicit_depth(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        d = src.to_dict()
        assert d["min_depth"] == -2.0
        assert d["max_depth"] == 2.0

    def test_yaml_roundtrip_omits_depth_when_unspecified(self, tmp_path):
        cfg = RecipeConfig(
            name="Depth Omission Test",
            variable="wind",
            validation_sources=[ValidationDataSource(source_type="scatterometer")],
        )
        recipe = Recipe(cfg)
        out = tmp_path / "recipe.yaml"
        recipe.to_yaml(out)

        raw_text = out.read_text()
        assert "min_depth" not in raw_text
        assert "max_depth" not in raw_text

        loaded = Recipe.from_yaml(out)
        loaded_src = loaded.config.validation_sources[0]
        assert loaded_src.min_depth is None
        assert loaded_src.max_depth is None
        assert loaded_src.resolved_min_depth == -20.0
        assert loaded_src.resolved_max_depth == 20.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_recipe.py -k "TestValidationSourceParsing" -v`
Expected: FAIL — `test_depth_defaults_to_none` fails because `src.min_depth == -20.0` currently (not `None`); `test_resolved_depth_falls_back_to_defaults`/`test_resolved_depth_uses_explicit_value` fail with `AttributeError: 'ValidationDataSource' object has no attribute 'resolved_min_depth'`; `test_to_dict_omits_unspecified_depth` fails because the key is present; `test_yaml_roundtrip_omits_depth_when_unspecified` fails because `min_depth`/`max_depth` appear in the YAML text.

- [ ] **Step 3: Implement the dataclass change, resolution properties, and serialization**

In `sar_validation/core/recipe.py`, replace the `ValidationDataSource` dataclass (currently lines 50-76):

```python
#: Fallback depth window (metres; negative = below sea surface) applied when
#: a recipe's validation source doesn't specify min_depth/max_depth.
DEFAULT_MIN_DEPTH = -20.0
DEFAULT_MAX_DEPTH = 20.0


@dataclass
class ValidationDataSource:
    """One validation data source referenced in a recipe."""

    source_type: str
    """
    Platform / product type.

    Accepted values:
      in-situ   : mooring, buoy, ferrybox, drifter, tidal_gauge
      satellite : scatterometer, altimeter, radiometer
      coastal   : hf_radar
    """

    # Optional depth filter (only meaningful for in-situ and HF radar
    # sources). None means "use DEFAULT_MIN_DEPTH/DEFAULT_MAX_DEPTH" — see
    # resolved_min_depth/resolved_max_depth below. Left as None for source
    # types that don't use depth (e.g. scatterometer) so recipes don't
    # serialize a meaningless depth window for them.
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None

    # Extra keyword arguments forwarded to the downloader
    download_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Override per-source collocation tolerances
    collocation_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_min_depth(self) -> float:
        return self.min_depth if self.min_depth is not None else DEFAULT_MIN_DEPTH

    @property
    def resolved_max_depth(self) -> float:
        return self.max_depth if self.max_depth is not None else DEFAULT_MAX_DEPTH

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"source_type": self.source_type}
        if self.min_depth is not None:
            d["min_depth"] = self.min_depth
        if self.max_depth is not None:
            d["max_depth"] = self.max_depth
        d["download_kwargs"] = self.download_kwargs
        d["collocation_kwargs"] = self.collocation_kwargs
        return d
```

Then in `Recipe._from_dict()`, find (currently around line 345-354):

```python
            validation_sources=[
                ValidationDataSource(
                    source_type=src["source_type"],
                    min_depth=src.get("min_depth", -20.0),
                    max_depth=src.get("max_depth",  20.0),
                    download_kwargs=src.get("download_kwargs", {}),
                    collocation_kwargs=src.get("collocation_kwargs", {}),
                )
                for src in data.get("validation_sources", [])
            ],
```

Replace with:

```python
            validation_sources=[
                ValidationDataSource(
                    source_type=src["source_type"],
                    min_depth=src.get("min_depth"),
                    max_depth=src.get("max_depth"),
                    download_kwargs=src.get("download_kwargs", {}),
                    collocation_kwargs=src.get("collocation_kwargs", {}),
                )
                for src in data.get("validation_sources", [])
            ],
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_recipe.py -v`
Expected: PASS (all tests in the file, not just `TestValidationSourceParsing` — this confirms no other test in the file relied on the old `-20.0`/`20.0` dataclass default).

- [ ] **Step 5: Commit**

```bash
git add sar_validation/core/recipe.py tests/test_recipe.py
git commit -m "feat: make ValidationDataSource min_depth/max_depth optional"
```

---

### Task 2: Resolve depth in `orchestrator.py` at point of use

**Files:**
- Modify: `sar_validation/core/orchestrator.py:108-110` (in-situ batch aggregation), `sar_validation/core/orchestrator.py:259-260` (HF radar dispatch)
- Test: `tests/test_downloaders.py`

**Interfaces:**
- Consumes: `ValidationDataSource.resolved_min_depth`/`resolved_max_depth` (properties from Task 1).
- Produces: nothing new for later tasks — this is the final task.

- [ ] **Step 1: Write failing tests for orchestrator depth resolution**

Add a new test class to the end of `tests/test_downloaders.py` (after the existing `TestOrchestratorHFRadarNOAAWiring` class, matching its style):

```python
# ---------------------------------------------------------------------------
# Tests for DataOrchestrator depth resolution (optional min_depth/max_depth)
# ---------------------------------------------------------------------------

class TestOrchestratorDepthResolution:
    def test_hf_radar_dispatch_uses_default_depth_when_unspecified(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-depth-default",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar")

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_hf_radar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0

    def test_hf_radar_dispatch_honours_explicit_depth(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-depth-explicit",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_hf_radar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -2.0
        assert kwargs["max_depth"] == 2.0

    def test_insitu_batch_uses_default_depth_when_all_unspecified(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-insitu-depth-default",
            variable="wind",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="buoy"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator.download_all()

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0

    def test_insitu_batch_widens_window_around_explicit_override(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-insitu-depth-mixed",
            variable="wind",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="mooring", min_depth=-5.0, max_depth=5.0),
                ValidationDataSource(source_type="buoy"),  # unspecified -> -20/20
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator.download_all()

        _, kwargs = mock_cls.call_args
        # most permissive window across resolved depths: min(-5,-20)=-20, max(5,20)=20
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0
```

Also patch the `patch`/`SARDownloader` side effects don't interfere: `download_all()` also calls `_download_sar()`, which will attempt a real `SARDownloader` call. Check the top of `tests/test_downloaders.py` for an existing pattern that avoids this (e.g. an autouse fixture or a `sar_data` stub) before running — if none exists, wrap the `download_all()` call in the same `with patch(...)` block and additionally patch `"sar_validation.downloaders.sar_downloader.SARDownloader"` so `_download_sar()` doesn't attempt network access:

```python
        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.sar_downloader.SARDownloader"
        ) as mock_sar_cls:
            mock_cls.return_value.download.return_value = None
            mock_sar_cls.return_value.download.return_value = []
            orchestrator.download_all()
```

Apply this same two-patch pattern to both `test_insitu_batch_*` tests above.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_downloaders.py -k "TestOrchestratorDepthResolution" -v`
Expected: FAIL — with the current code, `ValidationDataSource(source_type="hf_radar")` has `min_depth=None` (after Task 1), and `orchestrator._download_hf_radar` still passes the raw `source.min_depth`/`source.max_depth` (i.e. `None`), so `kwargs["min_depth"] == -20.0` fails (`None != -20.0`). Similarly the in-situ tests fail with `TypeError: '<=' not supported between instances of 'NoneType' and 'float'` from the `min()`/`max()` call over raw `None` fields, or a `None` result depending on Python's `min`/`max` semantics with mixed types.

- [ ] **Step 3: Implement the orchestrator change**

In `sar_validation/core/orchestrator.py`, find (currently lines 106-112):

```python
        if insitu_sources:
            source_types = [s.source_type for s in insitu_sources]
            # Use the most permissive depth window across all in-situ sources
            min_depth = min(s.min_depth for s in insitu_sources)
            max_depth = max(s.max_depth for s in insitu_sources)
            if not self._download_insitu(source_types, min_depth, max_depth):
                ok = False
```

Replace with:

```python
        if insitu_sources:
            source_types = [s.source_type for s in insitu_sources]
            # Use the most permissive depth window across all in-situ sources
            min_depth = min(s.resolved_min_depth for s in insitu_sources)
            max_depth = max(s.resolved_max_depth for s in insitu_sources)
            if not self._download_insitu(source_types, min_depth, max_depth):
                ok = False
```

Then find, in `_download_hf_radar` (currently lines 256-261):

```python
            dl = HFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.min_depth,
                max_depth=source.max_depth,
            )
```

Replace with:

```python
            dl = HFRadarDownloader(
                output_dir=out_dir,
                dry_run=self.dry_run,
                min_depth=source.resolved_min_depth,
                max_depth=source.resolved_max_depth,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_downloaders.py -v`
Expected: PASS (whole file, to confirm no regression in the existing `TestOrchestratorHFRadarNOAAWiring` tests or any other test in the module).

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS — no test anywhere in the suite still depends on `ValidationDataSource.min_depth`/`max_depth` defaulting to a concrete float.

- [ ] **Step 6: Commit**

```bash
git add sar_validation/core/orchestrator.py tests/test_downloaders.py
git commit -m "feat: resolve recipe depth defaults at orchestrator dispatch time"
```

---

## Self-Review Notes

- **Spec coverage:** Data model change → Task 1 Step 3. Resolution properties → Task 1 Step 3. Serialization (`to_dict` omission, `_from_dict` no forced default) → Task 1 Step 3. In-situ aggregation resolution → Task 2 Step 3. HF radar dispatch resolution → Task 2 Step 3. Out-of-scope items (downloader defaults, CLI, `cli.py` templates) → explicitly left untouched, verified no task modifies them. Testing section of spec → covered by Task 1 Step 1 (dataclass/serialization tests) and Task 2 Step 1 (orchestrator resolution tests).
- **Placeholder scan:** No TBD/TODO; all code blocks are complete and copy-pasteable.
- **Type consistency:** `resolved_min_depth`/`resolved_max_depth` (Task 1) are the exact names consumed in Task 2. `DEFAULT_MIN_DEPTH`/`DEFAULT_MAX_DEPTH` values (-20.0/20.0) match between Task 1's implementation and Task 2's test assertions.
