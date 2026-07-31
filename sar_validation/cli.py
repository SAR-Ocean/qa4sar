"""
Command-line interface for the SAR L2 Validation Toolbox.

Entry point: ``sar-validate``  (installed via pyproject.toml)
Or:          ``python -m sar_validation``

Usage
-----
  # List example recipes
  sar-validate --list-recipes

  # Create a recipe template (wind | currents | waves | soil_moisture)
  sar-validate --create-recipe wind

  # Store credentials in the OS keyring (eumdac | osi_saf | gportal | smos)
  sar-validate --set-credential eumdac

  # Dry-run (see what would be downloaded)
  sar-validate --recipe recipes/wind_validation.yaml --dry-run

  # Execute a recipe (download all data)
  sar-validate --recipe recipes/wind_validation.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

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
  sar-validate --set-credential eumdac
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65
  sar-validate --create-recipe wind --start 2026-03-01 --end 2026-03-31
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 \\
      --start 2026-03-01 --end 2026-03-31
  sar-validate --create-recipe wind --min-lon -10 --max-lon 5 --min-lat 50 --max-lat 65 \\
      --start 2026-03-01 --end 2026-03-31 --recipe-name north_sea_march_2026
  # Dry run: no data downloaded, just show what would be downloaded
  sar-validate --recipe recipes/wind_validation.yaml --dry-run
  # Download the data (skipped when download_metadata.json already exists in the data folder)
  sar-validate --recipe recipes/wind_validation.yaml
  # Ignore download_metadata.json and redownload the data
  sar-validate --recipe recipes/wind_validation.yaml --force-download
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
        help="Create a recipe template: wind | currents | waves | soil_moisture",
    )
    parser.add_argument(
        "--set-credential",
        metavar="SERVICE",
        choices=["eumdac", "osi_saf", "gportal", "smos"],
        help="Prompt for a username/password and store them in the OS keyring "
             "for SERVICE (eumdac | osi_saf | gportal | smos)",
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
        help=(
            "Eastern bound in decimal degrees (used with --create-recipe). "
            "If less than --min-lon, the bbox wraps through 180 deg "
            "(e.g. --min-lon 135 --max-lon -120 covers the Pacific)."
        ),
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
        help="Compute validation statistics (step 4 — bias, RMSE, correlation); implies --collocate",
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
    elif args.set_credential:
        _set_credential(args.set_credential)
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

def _set_credential(name: str) -> None:
    """Prompt for a username/password and store them in the OS keyring.

    Backs ``sar-validate --set-credential {eumdac,osi_saf,gportal,smos}``.
    """
    import getpass

    from .downloaders import base

    username = input(f"Username for '{name}': ")
    password = getpass.getpass(f"Password for '{name}': ")
    try:
        base.set_credential(name, username, password)
    except Exception as exc:
        print(f"Failed to store '{name}' credentials in the OS keyring: {exc}")
        sys.exit(1)
    print(f"Stored '{name}' credentials in the OS keyring.")


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


def _build_currents_config(limit: Optional[int] = None):
    """Build the 'currents' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects.
    """
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        LayerVsLayerCollocation,
        PointVsLayerCollocation,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )

    return RecipeConfig(
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
            ValidationDataSource(source_type="hf_radar"),
            ValidationDataSource(source_type="hf_radar_historical"),
            ValidationDataSource(source_type="drifter"),
            ValidationDataSource(source_type="ferrybox"),
            ValidationDataSource(source_type="mooring"),
            # Delayed-mode (6mo+ old) current observations — Copernicus
            # Marine product 013_044, EWCT/NSCT only. Each individually
            # gated at download time by its own recency guard.
            ValidationDataSource(source_type="adcp_historical"),
            ValidationDataSource(source_type="argo_historical"),
            ValidationDataSource(source_type="drifter_historical"),
            ValidationDataSource(source_type="glider_historical"),
        ],
        collocation=CollocationType(
            point_vs_layer=PointVsLayerCollocation(),
            layer_vs_layer=LayerVsLayerCollocation(
                layer_type_specs={
                    "hf_radar_grid": {
                        "time_tolerance_minutes": 30,
                        "aggregation_window_km": 6.0,
                        "distance_weighting": "equal",
                        "dedup_nearest_in_time": True,
                    },
                }
            ),
        ),
    )


def _build_wind_config(limit: Optional[int] = None):
    """Build the 'wind' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects,
    mirroring ``_build_currents_config``.
    """
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        LayerVsLayerCollocation,
        PointVsLayerCollocation,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )

    return RecipeConfig(
        name="Wind Validation",
        description=(
            "Validate Sentinel-1 IW/EW mode wind speed and direction\n"
            "against moorings, buoys, ASCAT scatterometer, HY-2B/HY-2C/\n"
            "Oceansat-3 scatterometer, 1 Hz altimeter, and RSS radiometer\n"
            "(AMSR2) ocean winds."
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
            # KNMI OSI-SAF FTP, recent-only (3-day window), 25 km.
            ValidationDataSource(source_type="scatterometer_hy2b"),
            ValidationDataSource(source_type="scatterometer_hy2c"),
            ValidationDataSource(source_type="scatterometer_oceansat3"),
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
                    # 25 km grid, distinct from ASCAT's 12.5 km above.
                    "scatterometer_hy2b": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "scatterometer_hy2c": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "scatterometer_oceansat3": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
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
                    # RSS radiometers all share the 0.25° (~25 km) grid, so
                    # the specs default alike. Per-sensor keys keep each one
                    # individually tunable; collocation resolves a node's
                    # 'radiometer' layer type to 'radiometer_<sensor>'.
                    # AMSR2 is NetCDF; GMI/SSMIS/WindSat are RSS bytemaps.
                    "radiometer_amsr2": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "radiometer_gmi": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "radiometer_ssmis_f16": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "radiometer_ssmis_f17": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "radiometer_ssmis_f18": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                    "radiometer_windsat": {
                        "time_tolerance_minutes": 180,
                        "aggregation_window_km": 25.0,
                        "distance_weighting": "equal",
                    },
                }
            ),
        ),
    )


def _build_soil_moisture_config(limit: Optional[int] = None):
    """Build the 'soil_moisture' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects,
    mirroring ``_build_wind_config``/``_build_currents_config``.
    """
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        PointVsLayerCollocation,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )

    return RecipeConfig(
        name="Soil Moisture Validation",
        description=(
            "Validate Sentinel-1 CLMS Surface Soil Moisture (1 km, Europe,\n"
            "daily) against ISMN in-situ stations."
        ),
        variable="soil_moisture",
        variable_specs={"components": ["soil_moisture"]},
        # Narrower than the ocean templates' Europe default — matches the
        # CLMS SSM 1km product's documented extent more tightly.
        geographic_bounds=GeographicBounds(-10.0, 30.0, 35.0, 60.0),
        sar_data=SARDataSpec(satellite="Sentinel-1", product_level="L3_SSM", max_downloads=limit),
        validation_sources=[
            # download_kwargs left empty on purpose — first run prints ISMN
            # portal instructions rather than needing a placeholder path.
            # min_depth/max_depth here (~0-5 cm) match C-band Sentinel-1's
            # sensing depth; a future NISAR (L-band) recipe would instead
            # use a deeper window (~0-0.25 m), set the same way.
            ValidationDataSource(source_type="ismn", min_depth=0.0, max_depth=0.05),
            # Satellite soil-moisture sources -- ASCAT (scatterometer),
            # AMSR2/SMAP/SMOS (radiometer) -- default on alongside ISMN
            # rather than requiring a separate recipe, matching
            # recipes/soil_moisture_satellite_example.yaml's source list.
            ValidationDataSource(source_type="ascat_ssm"),
            ValidationDataSource(source_type="amsr_ssm"),
            ValidationDataSource(source_type="smap_ssm"),
            ValidationDataSource(source_type="smos_ssm"),
        ],
        collocation=CollocationType(
            # Pixel-scale tolerances, not the ocean buoy-footprint defaults:
            # a 1 km SAR pixel and a point ISMN station don't need a 25 km
            # buoy-scale aggregation window, and the product is daily (not
            # 30-minutes-tolerance) — only a calendar-day match matters.
            point_vs_layer=PointVsLayerCollocation(
                spatial_tolerance_km=2.0,
                aggregation_window_km=1.0,
                distance_weighting="equal",
                interpolation_method="nearest",
                time_tolerance_minutes=720,
            ),
        ),
    )


def _create_recipe(
    name: str,
    limit: Optional[int] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    recipe_name: Optional[str] = None,
) -> None:
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        LayerVsLayerCollocation,
        PointVsLayerCollocation,
        Recipe,
        RecipeConfig,
        SARDataSpec,
        TemporalBounds,
        ValidationDataSource,
    )

    templates = {
        "wind": _build_wind_config(limit),
        "currents": _build_currents_config(limit),
        "soil_moisture": _build_soil_moisture_config(limit),
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


# Filename suffix per layer-vs-layer collocation method. "both" mode writes
# both suffixes directly (see method_runs below); a single method must map
# to the SAME suffix "both" mode would use for it, so that e.g.
# --layer-vs-layer-collocation-method individual alone reads/writes
# collocation_results_individual.nc consistently across collocate/stats/plot
# instead of the unsuffixed collocation_results.nc cell-averaging uses.
_METHOD_SUFFIX = {"cell-averaging": "", "individual": "_individual"}


def _execute_recipe(
    recipe_path: str,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    force_download: bool = False,
    convert: bool = False,
    collocate: bool = False,
    stats: bool = False,
    plot: bool = False,
    collocation_log: bool = False,
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> None:
    from .core.orchestrator import DataOrchestrator
    from .core.recipe import Recipe

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

    orchestrator = DataOrchestrator(recipe, dry_run=dry_run, force_download=force_download)

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
            print("\nOne or more downloads failed — continuing with available data.")
            print("Check download_metadata.json for details.")
            print(f"Data directory: {orchestrator.base_dir}")
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
        method_runs = [
            (layer_vs_layer_collocation_method, _METHOD_SUFFIX[layer_vs_layer_collocation_method])
        ]

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
            if _stats_already_computed(recipe, orchestrator.base_dir, filename_suffix=suffix):
                print("Step 4 skipped — validation_statistics files already exist")
            else:
                _compute_stats(recipe, orchestrator.base_dir, filename_suffix=suffix)

        if plot:
            _generate_plots(
                recipe, orchestrator.base_dir, filename_suffix=suffix,
                layer_vs_layer_collocation_method=method,
            )

    # Reprint any warnings/notices at the very end of the run: they may
    # have fired during Step 1 (download), long before Steps 2/3/4/5's
    # own console output scrolled them out of view.
    warnings = _load_download_warnings(orchestrator.base_dir)
    if warnings:
        print("\nWarnings from this run:")
        for w in warnings:
            print(f"  - {w}")


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
        diag_result = plot_collocation_diagnostics(
            tree, collocation_ds, recipe, base_dir,
            filename_suffix=filename_suffix,
            layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
        )
        # soil_moisture recipes with multiple SAR files return one Path per
        # file (see plot_collocation_diagnostics); every other case returns
        # a single Path or None.
        diag_paths = diag_result if isinstance(diag_result, list) else ([diag_result] if diag_result else [])
        for diag_path in diag_paths:
            print(f"  Generated diagnostic plot: {diag_path.relative_to(base_dir.parent)}")
    except Exception as exc:
        print(f"  Could not generate diagnostic plot: {exc}")
        logger.debug("Diagnostic plot error: %s", exc, exc_info=True)


def _compute_stats(recipe, base_dir: Path, filename_suffix: str = "") -> None:
    """Run step 4: compute validation statistics from collocation_results<suffix>.nc."""
    import xarray as xr

    from .core.statistics import run_statistics, run_statistics_native_units

    coll_path = base_dir / f"collocation_results{filename_suffix}.nc"
    if not coll_path.exists():
        print(f"  collocation_results{filename_suffix}.nc not found — statistics skipped.")
        return

    print("\nStep 4: Computing validation statistics…")
    collocation_ds = xr.open_dataset(str(coll_path))
    results = run_statistics(collocation_ds, recipe, base_dir, filename_suffix=filename_suffix)
    if not results:
        print("  No statistics produced (check that variable names match the recipe).")
    else:
        for key in results:
            print(f"  Statistics saved: validation_statistics_{key}{filename_suffix}.nc/.csv")

    if recipe.config.variable == "soil_moisture":
        native_results = run_statistics_native_units(collocation_ds, recipe, base_dir, filename_suffix=filename_suffix)
        for key in native_results:
            print(f"  Native-units statistics saved: validation_statistics_{key}_native_units{filename_suffix}.nc/.csv")


def _stats_already_computed(recipe, base_dir: Path, filename_suffix: str = "") -> bool:
    """Return True if every ``validation_statistics_*<suffix>.nc`` file Step 4
    would produce already exists on disk, so ``_execute_recipe`` can skip
    recomputation the same way Steps 1-3 skip their own already-done work.

    Opens ``collocation_results<suffix>.nc`` (Step 3's output) and applies
    the same dataset-aware ``filter_variable_pairs`` selection ``run_statistics``
    uses to decide *which* files it writes — reusing ``_load_precomputed_stats``
    for the lookup so the pair-matching logic isn't duplicated. Also requires
    the ``_native_units`` companion file for soil_moisture recipes, since
    ``_compute_stats`` writes those too.

    Returns False (i.e. "recompute") whenever the collocation results are
    missing, no variable pairs apply, or anything is still absent — a safe
    default that falls through to the normal Step 4 behaviour.
    """
    import xarray as xr

    from .core._variable_map import filter_variable_pairs

    coll_path = base_dir / f"collocation_results{filename_suffix}.nc"
    if not coll_path.exists():
        return False

    collocation_ds = xr.open_dataset(str(coll_path))
    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError:
        return False
    if not pairs:
        return False

    precomputed = _load_precomputed_stats(recipe, collocation_ds, base_dir, filename_suffix=filename_suffix)
    if set(precomputed) != {f"{sar_var}_vs_{val_var}" for sar_var, val_var in pairs}:
        return False

    if recipe.config.variable == "soil_moisture":
        for sar_var, val_var in pairs:
            native_path = base_dir / f"validation_statistics_{sar_var}_vs_{val_var}_native_units{filename_suffix}.nc"
            if not native_path.exists():
                return False

    return True


def _load_precomputed_stats(recipe, collocation_ds, base_dir: Path, filename_suffix: str = "") -> dict:
    """Load ``validation_statistics_<sar_var>_vs_<val_var><suffix>.nc`` files already
    saved by step 4, keyed the same way ``run_statistics`` names them.

    Extracted from ``_generate_plots`` so the file-matching logic is
    unit-testable independent of the CLI's plotting/PDF side effects. Uses
    ``filter_variable_pairs`` — the same dataset-aware pair selection
    ``run_statistics`` used to *write* these files — rather than the static
    ``infer_variable_pairs`` list, so the keys this looks up always match
    the keys the files were actually saved under.
    """
    import xarray as xr

    from .core._variable_map import filter_variable_pairs

    stats_ds_map: dict[str, xr.Dataset] = {}
    try:
        pairs = filter_variable_pairs(recipe, collocation_ds)
    except KeyError:
        return stats_ds_map
    for sar_var, val_var in pairs:
        key = f"{sar_var}_vs_{val_var}"
        stats_path = base_dir / f"validation_statistics_{key}{filename_suffix}.nc"
        if stats_path.exists():
            stats_ds_map[key] = xr.open_dataset(str(stats_path))
    return stats_ds_map


def _generate_plots(
    recipe, base_dir: Path, filename_suffix: str = "",
    layer_vs_layer_collocation_method: str = "cell-averaging",
) -> None:
    """Run step 5: generate validation plots and save PDF to <base_dir>/, PNG to <base_dir>/plots/."""
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

    print("\nStep 5: Generating validation plots and validation report")
    # .load() eagerly reads both files into memory once instead of leaving
    # them backed by lazy netCDF4 handles: report generation re-reads the
    # same collocation columns and SAR scenes from many different sections
    # (scatter, geographic, residuals, stats table), and repeated on-disk
    # reads of small-enough-to-fit-in-memory files (collocation_results.nc,
    # datatree.nc are tens of MB here) were a measured contributor to slow
    # report generation on multi-source (soil moisture) recipes.
    collocation_ds = xr.open_dataset(str(coll_path)).load()
    datatree = xr.open_datatree(str(datatree_path), engine="netcdf4").load()

    stats_ds_map = _load_precomputed_stats(recipe, collocation_ds, base_dir, filename_suffix)
    download_warnings = _load_download_warnings(base_dir)

    native_units_stats_ds_map = None
    if recipe.config.variable == "soil_moisture":
        native_units_stats_ds_map = _load_precomputed_stats(
            recipe, collocation_ds, base_dir, filename_suffix=f"_native_units{filename_suffix}",
        )

    validation_report(collocation_ds, datatree, recipe,
                      stats_ds_map=stats_ds_map or None,
                      out_dir=base_dir,
                      filename_suffix=filename_suffix,
                      download_warnings=download_warnings,
                      layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
                      native_units_stats_ds_map=native_units_stats_ds_map or None)
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
    if pdf_path.exists():
        print(f"  PDF report saved to {pdf_path}")
    print(f"  Collocation diagnostics PNG saved to {base_dir / 'plots'}")


def _load_download_warnings(base_dir: Path) -> Optional[list[str]]:
    """Read download_metadata.json's ``errors`` and ``notices`` lists, if
    present, for surfacing on the PDF cover page. ``notices`` are
    non-failure observations (e.g. "no data found for this window") that
    still deserve a durable, easy-to-find spot rather than only a
    console-log line that scrolls past during a long run. Returns None if
    there's no metadata file, it can't be parsed, or it has neither."""
    import json as _json

    meta_path = base_dir / "download_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
    except Exception:
        return None
    warnings = (meta.get("errors") or []) + (meta.get("notices") or [])
    return warnings or None
