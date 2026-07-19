# Lint Cleanup and Ruff Adoption — Design

**Date:** 2026-07-18
**Status:** Approved

## Problem

VS Code flags two problems in `sar_validation/core/statistics.py`:

1. `infer_variable_pairs` is imported (line 17) but never used.
2. `variable = recipe.config.variable` (line 268) is assigned but never used.

Both are leftovers from the refactor that replaced direct
`infer_variable_pairs(variable)` calls with
`filter_variable_pairs(recipe, collocation_ds)`. A ruff scan shows the same
class of problem across the package — 232 findings under rules E, F, I at a
120-character line length — because the repo has mypy configured but no
linter, so dead code accumulates silently.

## Goals

- Remove all dead code (unused imports/variables) package-wide.
- Adopt ruff so this class of problem is caught automatically from now on.
- Enforce via GitHub Actions CI (the repo works through PRs).
- Eliminate the six warnings the pytest suite currently emits.
- No user-visible behavior change beyond a plot-axis fix for
  constant-value data; test count stays at 414 passing.

## Non-goals

- No refactoring of `compute_statistics` internals.
- No typing modernization (`Optional[X]` → `X | None`).
- No formatter adoption (ruff *lint* only) — keeps diffs reviewable.
- No pre-commit hooks.

## Design

### 1. Original two warnings (`statistics.py`)

- Drop `infer_variable_pairs` from the `._variable_map` import.
- Delete the dead `variable = recipe.config.variable` assignment in
  `run_statistics`.
- Correct the `run_statistics` docstring: pairs come from
  `filter_variable_pairs`, not `infer_variable_pairs`.

### 2. Ruff configuration (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 120
target-version = "py310"   # matches requires-python

[tool.ruff.lint]
select = ["E", "F", "I"]
```

Add `ruff` to the `dev` optional-dependency group.

### 3. Package-wide fixes (~232 findings)

| Category | Count | Handling |
|---|---|---|
| I001 unsorted imports | 150 | `ruff --fix` |
| F811 duplicate `MagicMock` imports (tests) | 25 | `ruff --fix` |
| F401 unused imports | 23 | Manual; verify each is truly unused before deleting |
| F841 unused variables | 8 | Manual; verify each |
| E402 import not at top | 11 | Fix where trivial; `# noqa: E402` where the late import is deliberate (optional heavy deps) |
| E501 lines > 120 chars | 11 | Wrap |
| E731 lambda assignment | 2 | Convert to `def` |
| F541 empty f-string | 1 | `ruff --fix` |
| F821 undefined `plt` | 1 | See below |

Cases needing care rather than deletion:

- `cartopy.feature` in `visualization.py` is imported purely as an
  availability probe → replace with `importlib.util.find_spec`.
- `RecipeConfig`, `GeographicBounds`, `TemporalBounds` imported in
  `orchestrator.py` may be intentional re-exports → grep the repo
  (including CLI and tests) for downstream usage before removing; if used
  downstream, keep and mark with `# noqa: F401` or add to `__all__`.
- `Optional["plt.Figure"]` annotation at `visualization.py:1921` references
  `plt`, which is only imported inside function bodies → add a
  `TYPE_CHECKING` import of `matplotlib.figure.Figure` and annotate with
  that instead.

### 4. Pytest warnings (6)

The test suite currently emits six warnings; all are fixed at the source:

- **`Dataset.dims` FutureWarning** (`tests/test_datatree_converter.py:204`,
  `:448`): xarray is changing `Dataset.dims` to return a set. Replace
  `ds.dims["point"]` / `ds.dims["collocation"]` with `ds.sizes[...]`, the
  replacement the warning prescribes.
- **Singular axis-limits UserWarning ×2**
  (`sar_validation/core/visualization.py:450–451`): when every collocated
  value is identical, `vmin == vmax` and `set_xlim`/`set_ylim` get an empty
  range. Production fix in `plot_scatter`: when `vmin == vmax`, pad both
  limits symmetrically (e.g. ±0.5 or ±5 % of |value|, whichever is larger)
  before setting them. `test_constant_values_no_runtime_warning` already
  covers the constant-value path and will confirm the warning is gone.
- **Intentional fallback UserWarnings ×2**
  (`visualization.py:391` and `:1141`, triggered by
  `test_currents_recipe_passes_reduced_point_size_to_geographic`): the
  test's collocation fixture lacks the `temporal_distance_minutes` column,
  so the code under test warns and falls back. Add that column to the
  fixture — it makes the fixture match real collocation output and both
  warnings stop firing. The warnings themselves are correct behavior and
  stay in the production code.

### 5. CI (`.github/workflows/ci.yml`, new)

Single workflow on push and pull request:
checkout → set up Python 3.12 → `pip install -e .[dev]` →
`ruff check .` → `pytest`. The `dev` extra already includes the optional
plotting dependencies, and tests handle missing optional deps gracefully.

### 6. Verification

No new tests. The only behavior change is the axis-limit padding in
`plot_scatter` for constant-value data, which an existing test covers.
Done means:

- `ruff check .` exits clean.
- Full pytest suite passes with the same 414 tests as before, and the six
  warnings listed in section 4 no longer appear in the pytest output.
- `mypy` output is no worse than before the change.
- Any deletion that is not provably safe is grepped for usage across the
  repo first.
