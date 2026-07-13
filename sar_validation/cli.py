"""
Command-line interface for the SAR L2 Validation Toolbox.

Entry point: ``sar-validate``  (installed via pyproject.toml)
Or:          ``python -m sar_validation``

Usage
-----
  # List example recipes
  sar-validate --list-recipes

  # Create a recipe template (wind | currents | waves)
  sar-validate --create-recipe wind

  # Dry-run (see what would be downloaded)
  sar-validate --recipe examples/wind_validation_example.yaml --dry-run

  # Execute a recipe (download all data)
  sar-validate --recipe examples/wind_validation_example.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only import: binds ``xr`` for the string return annotations below
    # (e.g. ``"xr.DataTree | None"``). This module imports xarray lazily inside
    # the functions that use it rather than at module scope, so without this
    # guard a type checker can't resolve ``xr`` in those annotations.
    import xarray as xr

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="sar-validate",
        description="SAR L2 Ocean Data Validation Toolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sar-validate --list-recipes
  sar-validate --create-recipe wind
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65
  sar-validate --create-recipe wind --start 2026-03-01 --end 2026-03-31
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 --start 2026-03-01 --end 2026-03-31
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 --start 2026-03-01 --end 2026-03-31 --recipe-name north_sea_march_2026
  sar-validate --recipe recipes/wind_validation.yaml --dry-run #no data will be downloaded, just show what would be downloaded
  sar-validate --recipe recipes/wind_validation.yaml # for downloading the data if there is not already a download_metadata.json file in the data folder
  sar-validate --recipe recipes/wind_validation.yaml --force-download # overrides the download_metadata.json and redownloads the data
  sar-validate --recipe recipes/wind_validation.yaml --convert
  sar-validate --recipe recipes/wind_validation.yaml --convert --collocate
  sar-validate --recipe recipes/wind_validation.yaml --stats
  sar-validate --recipe recipes/wind_validation.yaml --plot
        """,
    )

    parser.add_argument(
        "--recipe",
        metavar="FILE",
        help="Path to a recipe YAML or JSON file to execute",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="List available example recipe files",
    )
    parser.add_argument(
        "--create-recipe",
        metavar="NAME",
        help="Create a recipe template: wind | currents | waves",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit SAR downloads to first N files (used with --create-recipe)",
    )
    parser.add_argument(
        "--min-lon",
        type=float,
        metavar="DEG",
        help="Western bound in decimal degrees (used with --create-recipe)",
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        metavar="DEG",
        help="Eastern bound in decimal degrees (used with --create-recipe)",
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        metavar="DEG",
        help="Southern bound in decimal degrees (used with --create-recipe)",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        metavar="DEG",
        help="Northern bound in decimal degrees (used with --create-recipe)",
    )
    parser.add_argument(
        "--start",
        metavar="DATE",
        help="Start of time window, ISO-8601 (used with --create-recipe), e.g. 2026-01-01",
    )
    parser.add_argument(
        "--end",
        metavar="DATE",
        help="End of time window, ISO-8601 (used with --create-recipe), e.g. 2026-01-02",
    )
    parser.add_argument(
        "--recipe-name",
        metavar="LABEL",
        help="Custom name for the recipe (sets the 'name' field and output filename)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what will be downloaded without actually downloading",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert downloaded data to xarray.DataTree (step 2)",
    )
    parser.add_argument(
        "--collocate",
        action="store_true",
        help="Run collocation between SAR and validation data (step 3); implies --convert",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute validation statistics (step 4b — bias, RMSE, correlation); implies --collocate",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate validation plots (step 5 — scatter, geographic, statistics, residuals); implies --stats",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Override the output directory specified in the recipe",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download data even if it already exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--collocation-log",
        action="store_true",
        help="Enable detailed per-point diagnostic logging during collocation "
             "(the diagnostics plot is always generated regardless of this flag)",
    )
    parser.add_argument(
        "--layer-vs-layer-collocation-method",
        metavar="METHOD",
        choices=["individual", "cell-averaging", "both"],
        default="cell-averaging",
        help="Collocation method for layer-vs-layer (scatterometer) data: "
             "'individual' matches each SAR pixel to the closest scatterometer point (reusable, many matches), "
             "'cell-averaging' aggregates SAR pixels within aggregation_window_km around each scatterometer "
             "point (default), 'both' runs both methods and writes suffixed outputs "
             "(collocation_results.nc + collocation_results_individual.nc, etc.) for comparison",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
        logging.getLogger("matplotlib").setLevel(logging.INFO)

    if args.list_recipes:
        _list_recipes()
    elif args.create_recipe:
        _create_recipe(
            args.create_recipe,
            limit=args.limit,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            start=args.start,
            end=args.end,
            recipe_name=args.recipe_name,
        )
    elif args.recipe:
        _execute_recipe(
            args.recipe,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            force_download=args.force_download,
            convert=args.convert or args.collocate or args.stats or args.plot,
            collocate=args.collocate or args.stats or args.plot,
            stats=args.stats or args.plot,
            plot=args.plot,
            collocation_log=args.collocation_log,
            layer_vs_layer_collocation_method=args.layer_vs_layer_collocation_method,
        )
    else:
        parser.print_help()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def _list_recipes() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    files = sorted(
        list(examples_dir.glob("*.yaml")) + list(examples_dir.glob("*.json"))
    )
    if not files:
        print("No example recipes found in examples/")
        return
    print("Available example recipes:")
    for f in files:
        print(f"  {f.name}")


def _create_recipe(
    name: str,
    limit: int = None,
    min_lon: float = None,
    max_lon: float = None,
    min_lat: float = None,
    max_lat: float = None,
    start: str = None,
    end: str = None,
    recipe_name: str = None,
) -> None:
    from .core.recipe import (
        Recipe, RecipeConfig, GeographicBounds, TemporalBounds,
        SARDataSpec, ValidationDataSource, CollocationType,
        PointVsLayerCollocation, LayerVsLayerCollocation,
    )

    templates = {
        "wind": RecipeConfig(
            name="Wind Validation",
            description=(
                "Validate Sentinel-1 IW/EW mode wind speed and direction\n"
                "against moorings, buoys, ASCAT scatterometer, 1 Hz altimeter,\n"
                "and RSS radiometer (AMSR2) ocean winds."
            ),
            variable="wind",
            variable_specs={"components": ["speed", "direction"]},
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            sar_data=SARDataSpec(swath_mode=["IW", "EW"], max_downloads=limit),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="buoy"),
                ValidationDataSource(source_type="ferrybox"),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="tidal_gauge"),
                ValidationDataSource(source_type="scatterometer"),
                ValidationDataSource(source_type="altimeter"),
                ValidationDataSource(source_type="radiometer"),
            ],
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(),
                layer_vs_layer=LayerVsLayerCollocation(
                    layer_type_specs={
                        "scatterometer": {
                            "time_tolerance_minutes": 180,
                            "aggregation_window_km": 12.5,
                            "distance_weighting": "equal",
                        },
                        # Wind recipes only ever download 1 Hz altimeter data
                        # (no WIND_SPEED at 5 Hz) — see DEFAULT_LAYER_TYPE_SPECS
                        # in recipe.py for the matching aggregation window.
                        "altimeter_1hz": {
                            "time_tolerance_minutes": 180,
                            "aggregation_window_km": 7.0,
                            "distance_weighting": "equal",
                        },
                        # RSS radiometer (AMSR2): all products share the 0.25°
                        # (~25 km) grid. Per-sensor key so each sensor stays
                        # individually tunable; collocation resolves a node's
                        # 'radiometer' layer type to 'radiometer_<sensor>'.
                        "radiometer_amsr2": {
                            "time_tolerance_minutes": 180,
                            "aggregation_window_km": 25.0,
                            "distance_weighting": "equal",
                        },
                    }
                ),
            ),
        ),
        "currents": RecipeConfig(
            name="Ocean Currents Validation",
            description=(
                "Validate Sentinel-1 WV mode ocean currents\n"
                "against HF radar and drifting buoys."
            ),
            variable="currents",
            variable_specs={"components": ["zonal", "meridional"]},
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            sar_data=SARDataSpec(swath_mode=["WV","IW","EW","SM"], max_downloads=limit),
            validation_sources=[
                ValidationDataSource(
                    source_type="hf_radar",
                    min_depth=-2.0, max_depth=2.0,
                ),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="ferrybox"),
                ValidationDataSource(source_type="mooring"),
            ],
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(),
                layer_vs_layer=LayerVsLayerCollocation(
                    layer_type_specs={
                        "hf_radar": {
                            "time_tolerance_minutes": 60,
                            "aggregation_window_km": 5.0,
                            "distance_weighting": "equal",
                        }
                    }
                ),
            ),
        ),
        "waves": RecipeConfig(
            name="Wave Height Validation",
            description=(
                "Validate Sentinel-1 significant wave height\n"
                "against moorings, tidal gauges, drifters, and altimeter."
            ),
            variable="waves",
            variable_specs={"components": ["significant_wave_height"]},
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            sar_data=SARDataSpec(swath_mode=["WV","SM"], max_downloads=limit),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="tidal_gauge"),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="altimeter"),
            ],
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(),
                layer_vs_layer=LayerVsLayerCollocation(
                    layer_type_specs={
                        # 1 Hz (~7km) and 5 Hz (~1.4km along-track) altimeter
                        # data need different aggregation windows — see
                        # DEFAULT_LAYER_TYPE_SPECS in recipe.py.
                        "altimeter_1hz": {
                            "time_tolerance_minutes": 180,
                            "aggregation_window_km": 7.0,
                            "distance_weighting": "equal",
                        },
                        "altimeter_5hz": {
                            "time_tolerance_minutes": 180,
                            "aggregation_window_km": 1.4,
                            "distance_weighting": "equal",
                        },
                    }
                ),
            ),
        ),
    }

    if name not in templates:
        print(f"Unknown recipe type '{name}'. Choose from: {', '.join(templates)}")
        sys.exit(1)

    cfg = templates[name]

    # Override geographic bounds if any bound flag was supplied
    if any(v is not None for v in (min_lon, max_lon, min_lat, max_lat)):
        cfg.geographic_bounds = GeographicBounds(
            min_lon=min_lon if min_lon is not None else cfg.geographic_bounds.min_lon,
            max_lon=max_lon if max_lon is not None else cfg.geographic_bounds.max_lon,
            min_lat=min_lat if min_lat is not None else cfg.geographic_bounds.min_lat,
            max_lat=max_lat if max_lat is not None else cfg.geographic_bounds.max_lat,
        )

    # Override temporal bounds if either date flag was supplied
    if start is not None or end is not None:
        cfg.temporal_bounds = TemporalBounds(
            start=start if start is not None else cfg.temporal_bounds.start,
            end=end   if end   is not None else cfg.temporal_bounds.end,
        )

    if recipe_name:
        cfg.name = recipe_name

    recipe = Recipe(cfg)
    slug = recipe_name.lower().replace(" ", "_") if recipe_name else f"{name}_validation"
    out_path = Path("recipes") / f"{slug}.yaml"
    recipe.to_yaml(out_path)
    print(f"Recipe created: {out_path}")
    print("Edit the file to adjust data sources and collocation settings.")


def _execute_recipe(
    recipe_path: str,
    dry_run: bool = False,
    output_dir: str = None,
    force_download: bool = False,
    convert: bool = False,
    collocate: bool = False,
    stats: bool = False,
    plot: bool = False,
    collocation_log: bool = False,
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> None:
    from .core.recipe import Recipe
    from .core.orchestrator import DataOrchestrator

    path = Path(recipe_path)
    if not path.exists():
        print(f"Recipe file not found: {path}")
        sys.exit(1)

    if path.suffix in (".yaml", ".yml"):
        recipe = Recipe.from_yaml(path)
    else:
        recipe = Recipe.from_json(path)

    if output_dir:
        recipe.config.output_dir = output_dir

    orchestrator = DataOrchestrator(recipe, dry_run=dry_run)

    # Skip download if data was already downloaded successfully and not forcing re-download
    if not dry_run and not force_download and _is_already_downloaded(orchestrator.base_dir):
        logger.info(
            "Data already downloaded in %s — skipping Step 1.",
            orchestrator.base_dir,
        )
        print(f"Step 1 skipped — data already present in {orchestrator.base_dir}")
        success = True
    else:
        success = orchestrator.download_all()

        if dry_run:
            print("\nDry run complete — no data was downloaded.")
            print("No data directories or files were created.")
            return
        elif not success:
            print("\nOne or more downloads failed. Check download_metadata.json for details.")
            sys.exit(1)
        else:
            print("\nAll downloads completed.")
            print(f"Data directory: {orchestrator.base_dir}")

    if convert:
        datatree_path = orchestrator.base_dir / "datatree.nc"
        if not datatree_path.exists():
            _convert_data(recipe, orchestrator.base_dir)
        else:
            print("Step 2 skipped — datatree.nc already exists")

    if layer_vs_layer_collocation_method == "both":
        # Run the full pipeline once per method, writing distinctly-suffixed
        # outputs so neither run overwrites the other.
        method_runs = [("cell-averaging", ""), ("individual", "_individual")]
    else:
        method_runs = [(layer_vs_layer_collocation_method, "")]

    for method, suffix in method_runs:
        if collocate:
            collocation_path = orchestrator.base_dir / f"collocation_results{suffix}.nc"
            if not collocation_path.exists():
                _collocate_data(
                    recipe,
                    orchestrator.base_dir,
                    emit_diagnostics=collocation_log,
                    layer_vs_layer_collocation_method=method,
                    filename_suffix=suffix,
                )
            else:
                print(f"Step 3 skipped — collocation_results{suffix}.nc already exists")

        if stats or plot:
            _compute_stats(recipe, orchestrator.base_dir, filename_suffix=suffix)

        if plot:
            _generate_plots(recipe, orchestrator.base_dir, filename_suffix=suffix)


def _is_already_downloaded(base_dir: Path) -> bool:
    """Return True if *base_dir* has a download_metadata.json with no errors."""
    import json as _json
    meta_path = base_dir / "download_metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
        return meta.get("errors", ["placeholder"]) == []
    except Exception:
        return False


def _convert_data(recipe, base_dir: Path) -> "xr.DataTree | None":
    """Run step 2: convert downloaded files to a DataTree."""
    from .core.datatree_converter import DataTreeConverter

    print("\nStep 2: Converting data to DataTree…")
    # Extract product type from recipe to guide SAR data extraction
    product_type = getattr(recipe.config, "variable", "wind")  # Default to "wind" for backward compatibility
    tree = DataTreeConverter.convert_downloaded_data(
        base_dir, product_type=product_type, recipe=recipe
    )
    if tree is None:
        print("  No data files found — nothing to convert.")
        return None
    print(f"  DataTree saved to {base_dir / 'datatree.nc'}")
    return tree


def _collocate_data(
    recipe,
    base_dir: Path,
    emit_diagnostics: bool = False,
    layer_vs_layer_collocation_method: str = "cell-averaging",
    filename_suffix: str = "",
) -> None:
    """Run step 3: load DataTree and run collocation."""
    import xarray as xr
    from .core.collocation import run_collocation
    from .core.visualization import plot_collocation_diagnostics

    datatree_path = base_dir / "datatree.nc"
    if not datatree_path.exists():
        # datatree wasn't produced yet (shouldn't happen since --collocate implies --convert)
        print("  DataTree not found — running conversion first.")
        tree = _convert_data(recipe, base_dir)
        if tree is None:
            print("  Conversion produced no output — collocation skipped.")
            return
    else:
        tree = xr.open_datatree(str(datatree_path), engine='netcdf4')

    print(f"\nStep 3: Running collocation (method={layer_vs_layer_collocation_method})…")
    result = run_collocation(
        recipe,
        tree,
        base_dir,
        emit_diagnostics=emit_diagnostics,
        layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
        filename_suffix=filename_suffix,
    )
    if result is None:
        print("  No collocated pairs found.")
        collocation_ds = None
    else:
        n = result.sizes.get("collocation", 0)
        print(f"  {n} collocated pair(s) saved to {base_dir / f'collocation_results{filename_suffix}.nc'}")
        collocation_ds = result

    # Always generate the collocation diagnostics plot — including when
    # there are zero collocated pairs, in which case every validation point
    # shows up as unmatched.
    try:
        diag_path = plot_collocation_diagnostics(tree, collocation_ds, recipe, base_dir, filename_suffix=filename_suffix)
        if diag_path:
            print(f"  Generated diagnostic plot: {diag_path.relative_to(base_dir.parent)}")
    except Exception as exc:
        print(f"  Could not generate diagnostic plot: {exc}")
        logger.debug("Diagnostic plot error: %s", exc, exc_info=True)


def _compute_stats(recipe, base_dir: Path, filename_suffix: str = "") -> None:
    """Run step 4b: compute validation statistics from collocation_results<suffix>.nc."""
    import xarray as xr
    from .core.statistics import run_statistics

    coll_path = base_dir / f"collocation_results{filename_suffix}.nc"
    if not coll_path.exists():
        print(f"  collocation_results{filename_suffix}.nc not found — statistics skipped.")
        return

    print("\nStep 4b: Computing validation statistics…")
    collocation_ds = xr.open_dataset(str(coll_path))
    results = run_statistics(collocation_ds, recipe, base_dir, filename_suffix=filename_suffix)
    if not results:
        print("  No statistics produced (check that variable names match the recipe).")
    else:
        for key in results:
            print(f"  Statistics saved: validation_statistics_{key}{filename_suffix}.nc/.csv")


def _generate_plots(recipe, base_dir: Path, filename_suffix: str = "") -> None:
    """Run step 5: generate validation plots and save to <base_dir>/plots/."""
    import xarray as xr
    from .core.visualization import validation_report

    coll_path = base_dir / f"collocation_results{filename_suffix}.nc"
    datatree_path = base_dir / "datatree.nc"

    if not coll_path.exists():
        print(f"  collocation_results{filename_suffix}.nc not found — plotting skipped.")
        return
    if not datatree_path.exists():
        print("  datatree.nc not found — plotting skipped.")
        return

    print("\nStep 5: Generating validation plots…")
    collocation_ds = xr.open_dataset(str(coll_path))
    datatree = xr.open_datatree(str(datatree_path), engine="netcdf4")

    # Load pre-computed statistics if available
    from .core._variable_map import infer_variable_pairs
    stats_ds_map = {}
    try:
        pairs = infer_variable_pairs(recipe.config.variable)
        for sar_var, val_var in pairs:
            key = f"{sar_var}_vs_{val_var}"
            stats_path = base_dir / f"validation_statistics_{key}{filename_suffix}.nc"
            if stats_path.exists():
                stats_ds_map[key] = xr.open_dataset(str(stats_path))
    except KeyError:
        pass

    validation_report(collocation_ds, datatree, recipe,
                      stats_ds_map=stats_ds_map or None,
                      out_dir=base_dir,
                      filename_suffix=filename_suffix)
    plots_dir = base_dir / "plots"
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
    print(f"  PNG plots saved to {plots_dir}")
    if pdf_path.exists():
        print(f"  PDF report saved to {pdf_path}")
