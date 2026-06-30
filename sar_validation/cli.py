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
  sar-validate --recipe examples/wind_validation_example.yaml --dry-run
  sar-validate --recipe examples/wind_validation_example.yaml
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
        "--dry-run",
        action="store_true",
        help="Show what will be downloaded without actually downloading",
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
        _create_recipe(args.create_recipe, limit=args.limit)
    elif args.recipe:
        _execute_recipe(args.recipe, dry_run=args.dry_run, output_dir=args.output_dir)
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


def _create_recipe(name: str, limit: int = None) -> None:
    from .core.recipe import (
        RecipeConfig, GeographicBounds, TemporalBounds,
        SARDataSpec, ValidationDataSource, CollocationType, Recipe,
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

    recipe = Recipe(templates[name])
    out_path = Path("recipes") / f"{name}_validation.yaml"
    recipe.to_yaml(out_path)
    print(f"Recipe created: {out_path}")
    print("Edit the file to set your region, time window, and data sources.")


def _execute_recipe(
    recipe_path: str,
    dry_run: bool = False,
    output_dir: str = None,
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
    success = orchestrator.download_all()

    if success:
        print("\nAll downloads completed.")
        print(f"Data directory: {orchestrator.base_dir}")
    else:
        print("\nOne or more downloads failed. Check download_metadata.json for details.")
        sys.exit(1)
