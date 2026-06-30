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

logging.basicConfig(
    level=logging.INFO,
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
  sar-validate --recipe recipes/wind_validation.yaml --dry-run
  sar-validate --recipe recipes/wind_validation.yaml
  sar-validate --recipe recipes/wind_validation.yaml --convert
  sar-validate --recipe recipes/wind_validation.yaml --convert --collocate
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
        "--output-dir",
        metavar="DIR",
        help="Override the output directory specified in the recipe",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
            convert=args.convert or args.collocate,
            collocate=args.collocate,
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
    )

    templates = {
        "wind": RecipeConfig(
            name="Wind Validation",
            description=(
                "Validate Sentinel-1 IW/EW mode wind speed and direction\n"
                "against moorings, buoys, and ASCAT scatterometer."
            ),
            variable="wind",
            variable_specs={"components": ["speed", "direction"]},
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            sar_data=SARDataSpec(swath_mode=["IW", "EW"], max_downloads=limit),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="buoy"),
                ValidationDataSource(
                    source_type="scatterometer",
                    collocation_kwargs={
                        "time_tolerance_minutes": 120,
                        "spatial_tolerance_km": 100,
                    },
                ),
            ],
            collocation=CollocationType("point_vs_layer"),
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
            sar_data=SARDataSpec(swath_mode=["WV"], max_downloads=limit),
            validation_sources=[
                ValidationDataSource(
                    source_type="hf_radar",
                    min_depth=-2.0, max_depth=2.0,
                ),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="ferrybox"),
            ],
            collocation=CollocationType("point_vs_layer"),
        ),
        "waves": RecipeConfig(
            name="Wave Height Validation",
            description=(
                "Validate Sentinel-1 significant wave height\n"
                "against moorings and altimeter."
            ),
            variable="waves",
            variable_specs={"components": ["significant_wave_height"]},
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            sar_data=SARDataSpec(swath_mode=["WV"], max_downloads=limit),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="altimeter"),
            ],
            collocation=CollocationType("point_vs_layer"),
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
    convert: bool = False,
    collocate: bool = False,
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

    # Skip download if data was already downloaded successfully
    if not dry_run and _is_already_downloaded(orchestrator.base_dir):
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
            print("Step 2 skipped — DataTree already exists")

    if collocate:
        _collocate_data(recipe, orchestrator.base_dir)


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
    tree = DataTreeConverter.convert_downloaded_data(base_dir)
    if tree is None:
        print("  No data files found — nothing to convert.")
        return None
    print(f"  DataTree saved to {base_dir / 'datatree.nc'}")
    return tree


def _collocate_data(recipe, base_dir: Path) -> None:
    """Run step 3: load DataTree and run collocation."""
    import xarray as xr
    from .core.collocation import run_collocation

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

    print("\nStep 3: Running collocation…")
    result = run_collocation(recipe, tree, base_dir)
    if result is None:
        print("  No collocated pairs found.")
    else:
        n = result.sizes.get("collocation", 0)
        print(f"  {n} collocated pair(s) saved to {base_dir / 'collocation_results.nc'}")
