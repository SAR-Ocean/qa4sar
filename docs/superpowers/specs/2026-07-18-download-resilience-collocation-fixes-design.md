# Download resilience, individual-collocation caching, and diagnostics-plot transparency — design

## Problem

Three independent, previously-deferred bugs from the original report list (the antimeridian-crossing bug was handled separately — see `2026-07-17-antimeridian-crossing-design.md`):

**B — download resilience & per-file resume.** `_execute_recipe()` ([cli.py:511-513](../../../sar_validation/cli.py#L511-L513)) calls `sys.exit(1)` if *any* single source's download fails, even though `DataOrchestrator.download_all()` ([orchestrator.py:84-123](../../../sar_validation/core/orchestrator.py#L84-L123)) already tries every source and records failures in `download_metadata.json`'s `errors` list — so a single flaky source (e.g. altimeter timing out) aborts conversion/collocation/stats/plot entirely, discarding everything else that succeeded. Separately, no downloader checks per-file existence before downloading, so `--force-download` (and even a plain re-run after a partial failure) re-downloads everything from scratch.

**Root-cause discovery made while designing B:** three downloaders (`hf_radar_downloader.py:213`, `insitu_downloader.py:319`, `altimeter_downloader.py:224`) call `copernicusmarine.subset(..., force_download=True/False)`. Verified via `inspect.signature(copernicusmarine.subset)` against the installed version (2.4.1, pinned as `>=2.0` in `pyproject.toml`) that `force_download` **is not a parameter of this function** — it has `overwrite`/`skip_existing` instead (mutually exclusive), and does not accept `**kwargs`. Every real (non-mocked) call to these three downloaders currently raises `TypeError: subset() got an unexpected keyword argument 'force_download'`. Tests never caught this because they all mock `copernicusmarine` entirely. Fixing per-file resume necessarily means replacing this broken kwarg with the real API, so this is folded into bug B's fix rather than tracked separately.

**C — individual collocation caching.** `_execute_recipe()` ([cli.py:525-530](../../../sar_validation/cli.py#L525-L530)): when `--layer-vs-layer-collocation-method individual` is passed (not `"both"`), `method_runs = [(layer_vs_layer_collocation_method, "")]` hardcodes `filename_suffix=""` regardless of method. Only `"both"` mode correctly maps `"individual"` → `"_individual"`. This means a lone `--layer-vs-layer-collocation-method individual` run reads/writes `collocation_results.nc` — the same filename cell-averaging uses — so (1) if a prior cell-averaging run already produced `collocation_results.nc`, the individual run's Step 3 is wrongly skipped as "already exists", and (2) re-running `--stats`/`--plot` with `individual` alone always resolves to whatever `collocation_results.nc` currently contains, never `collocation_results_individual.nc`.

**D — diagnostics-plot transparency.** `plot_collocation_diagnostics()`'s `matched_layer_alpha` ([visualization.py:1518](../../../sar_validation/core/visualization.py#L1518)) depends only on `recipe.config.variable` (wind → 0.65, else → 1.0), never on which collocation method produced the data. Individual-mode collocation produces roughly 100-500x more matched points than cell-averaging (one point per matched SAR pixel vs. one per validation-instrument location), so the same alpha value saturates into a solid-looking blob, losing the "per-pixel match" visual distinction the mode exists to show. Confirmed empirically: both `collocation_diagnostics_wind_europe_example.png` (983 points, cell-averaging) and `collocation_diagnostics_wind_europe_example_individual.png` (164,749 points, individual) already use the identical alpha value (0.65) today and are visually near-indistinguishable — the fix must lower individual mode's alpha specifically, not merely "match" cell-averaging's (which is already the case and insufficient on its own).

## Section 1 — Bug B: download resilience & per-file resume

### B1 — Continue past partial download failure

Replace the `elif not success: ... sys.exit(1)` branch in `_execute_recipe()` with a warning that lets execution continue into convert/collocate/stats/plot using whatever succeeded:

```python
elif not success:
    print("\nOne or more downloads failed — continuing with available data.")
    print("Check download_metadata.json for details.")
else:
    ...
```

No change to `download_all()`'s own per-source try/except — it already continues past failures and records them; only the CLI's reaction to a `False` return changes.

### B2 — Surface download failures on the PDF cover page

`_generate_plots()` ([cli.py:683](../../../sar_validation/cli.py#L683)) reads `download_metadata.json` from `base_dir` (if present) and extracts `errors`, passing them to `validation_report()` as a new `download_warnings: Optional[list[str]] = None` parameter. `validation_report()`'s cover-page block ([visualization.py:2201-2214](../../../sar_validation/core/visualization.py#L2201-L2214)) adds a short line under the existing title/metadata text when `download_warnings` is non-empty, e.g.:

```
⚠ 1 source failed to download: altimeter download failed: <error message>
```

(joined with `"; "` if there are multiple). This is a read of a file that already exists by the time `--plot` runs — no new state needs to be threaded through `download_all()` itself.

### B3 — Per-file download resume

Real per-file skip/overwrite behavior, replacing the broken `force_download` kwarg and adding the missing checks to the downloaders that never had one:

- **`DataOrchestrator.__init__`** gains `force_download: bool = False`, stored as `self.force_download`. Each `_download_*` method passes `force_download=self.force_download` into its downloader's constructor. `cli.py`'s existing `force_download` variable (already parsed from `--force-download`) is passed to `DataOrchestrator(recipe, dry_run=dry_run, force_download=force_download)`.
- **Copernicus-Marine downloaders with an explicit `output_filename`** (`AltimeterDownloader`, `HFRadarDownloader`): `copernicusmarine.subset(..., force_download=...)` → `copernicusmarine.subset(..., skip_existing=not self.force_download, overwrite=self.force_download)`. Default behavior (no `--force-download`) skips files that already exist; `--force-download` overwrites them.
- **`InSituDownloader`**: doesn't pass an explicit `output_filename` to `subset()` (downloads to CWD, then moves the file to `dest_path` — see [insitu_downloader.py:241-256](../../../sar_validation/downloaders/insitu_downloader.py#L241-L256)), so `skip_existing`/`overwrite` don't apply the same way. Instead: if `not self.force_download and dest_path.exists()`, skip the download entirely (print "Already downloaded", return `dest_path`) before calling `copernicusmarine.subset()` at all.
- **`HFRadarHistoricalDownloader`**: its raw-archive fetch already uses `copernicusmarine.get(..., skip_existing=True)` correctly ([hf_radar_historical_downloader.py:165-172](../../../sar_validation/downloaders/hf_radar_historical_downloader.py#L165-L172)) — no change needed there. Add a `dest_path.exists()` pre-check (skip local xarray re-subset/re-write when `not self.force_download`) purely as a minor efficiency improvement, not a correctness fix.
- **`SARDownloader`**: before calling `client.download_product()` for each queried row, check whether `output_dir` already contains a file/directory matching `row["Name"]` (the product's own unzipped-directory name); skip if `not self.force_download` and it exists.
- **`ScatterometerDownloader`**: EUMDAC doesn't expose the final filename before opening the product stream, so the check uses the product's own ID string: skip a product if `not self.force_download` and any file in `output_dir` already contains that product ID as a substring.
- **`NOAAHFRadarDownloader`**: filename is fully deterministic before the request (`{dataset_id}_{resolution_km}km_{date_str}.nc`); check `out_path.exists()` before `urllib.request.urlretrieve()`, skip if `not self.force_download` and present.

The existing whole-run `_is_already_downloaded()` gate ([cli.py:553](../../../sar_validation/cli.py#L553)) is unchanged — it stays as a fast-path that skips Step 1 entirely when a prior run had zero errors. When it doesn't skip (first run, or a prior run had errors), Step 1 now runs cheaply thanks to per-file skipping: sources that already succeeded re-verify their files exist and move on; only previously-failed or missing sources actually hit the network.

## Section 2 — Bug C: individual-collocation filename fix

Single-line root-cause fix. Replace the hardcoded `""` in the non-`"both"` branch of `_execute_recipe()` ([cli.py:529-530](../../../sar_validation/cli.py#L529-L530)) with a lookup:

```python
_METHOD_SUFFIX = {"cell-averaging": "", "individual": "_individual"}
...
if layer_vs_layer_collocation_method == "both":
    method_runs = [("cell-averaging", ""), ("individual", "_individual")]
else:
    method_runs = [(layer_vs_layer_collocation_method, _METHOD_SUFFIX[layer_vs_layer_collocation_method])]
```

This makes a lone `--layer-vs-layer-collocation-method individual` run consistently read/write `collocation_results_individual.nc` across collocate, stats, and plot steps — the same file `"both"` mode already produces for that method — so re-running `--stats`/`--plot` with `individual` finds the right file instead of silently falling back to (or being blocked by) the cell-averaging one.

## Section 3 — Bug D: density-aware transparency for individual-mode matched points

`plot_collocation_diagnostics()` gains an explicit `layer_vs_layer_collocation_method: str = "cell-averaging"` parameter. Both call sites already have the method name available and thread it through:

- `_collocate_data()` ([cli.py:584-633](../../../sar_validation/cli.py#L584-L633)) already receives `layer_vs_layer_collocation_method` as a parameter — passes it straight to its `plot_collocation_diagnostics(...)` call ([cli.py:628](../../../sar_validation/cli.py#L628)).
- `_generate_plots()` ([cli.py:683](../../../sar_validation/cli.py#L683)) gains a new `layer_vs_layer_collocation_method: str = "cell-averaging"` parameter, passed through to `validation_report(...)`, which gains the same new parameter and threads it to its own internal `plot_collocation_diagnostics(...)` call ([visualization.py:2154](../../../sar_validation/core/visualization.py#L2154)). `_execute_recipe()`'s `for method, suffix in method_runs:` loop ([cli.py:532](../../../sar_validation/cli.py#L532)) already has `method` in scope and passes it to both `_generate_plots(...)` and (already, via B2/C) other per-method calls.

Inside `plot_collocation_diagnostics()`, `matched_layer_alpha`'s computation ([visualization.py:1511-1518](../../../sar_validation/core/visualization.py#L1511-L1518)) becomes method-aware:

```python
if layer_vs_layer_collocation_method == "individual":
    matched_layer_alpha = 0.15
else:
    matched_layer_alpha = 0.65 if variable == "wind" else 1.0
```

`0.15` is a fixed, empirically-reasonable value for individual mode's typical point density (verified: at the observed ~170x density ratio between cell-averaging and individual collocation for the same recipe, 0.15 keeps dense clusters visibly denser than sparse ones without either vanishing or re-saturating to solid). No other part of the function changes — marker size, edge color, and every other tier's styling stay as-is.

## Testing

- **B1:** unit test on `_execute_recipe` (or a focused test of the branch itself) confirming a `download_all() == False` result no longer calls `sys.exit`, and that convert/collocate/stats/plot still run afterward.
- **B2:** test that `validation_report(..., download_warnings=["altimeter download failed: ..."])` renders a cover page containing that text (image-based assertion, matching this file's existing PDF-page test patterns), and that omitting/empty `download_warnings` leaves the cover page unchanged from today.
- **B3:** per-downloader tests (matching the antimeridian-crossing plan's per-downloader test style) verifying: (a) `skip_existing=True`/`overwrite=False` passed to `subset()`/`get()` by default, flipped when `force_download=True`; (b) a downloader with no built-in skip mechanism (SAR, scatterometer, NOAA) does not attempt a network call when the expected output already exists and `force_download=False`; (c) `force_download=True` re-attempts the network call regardless of existing files. Also a regression test proving the previously-broken `force_download=...` kwarg no longer reaches `copernicusmarine.subset()`/`.get()` at all (asserting the actual kwargs dict passed to the mocked function).
- **C:** unit test on the suffix-lookup fix directly: `--layer-vs-layer-collocation-method individual` alone produces `method_runs == [("individual", "_individual")]`; `"cell-averaging"` alone still produces `[("cell-averaging", "")]`; `"both"` unchanged.
- **D:** extend `tests/test_visualization.py`'s existing `TestPlotCollocationDiagnostics*` classes with a case asserting `matched_layer_alpha == 0.15` is used (via a spy/monkeypatch on the scatter call, similar to existing tests' approach) when `layer_vs_layer_collocation_method="individual"`, and that omitting the parameter (default `"cell-averaging"`) reproduces today's exact alpha values (0.65/1.0) — a non-regression case.

## Out of scope

- Any change to the antimeridian-crossing work (already shipped, PR #7).
- Any change to `download_all()`'s own per-source try/except structure — it already continues past failures; only the CLI's *reaction* to its `False` return changes.
- A density-adaptive (computed) alpha for bug D — a fixed value was chosen per the design discussion; revisit only if `0.15` proves wrong in practice for very different point-density recipes.
- Cleaning up any other latent `copernicusmarine` API mismatches beyond `force_download` — this design fixes exactly the one found while working on bug B; a broader audit of the `copernicusmarine` integration is a separate task if desired.
