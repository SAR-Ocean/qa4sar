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

  # Store credentials in the OS keyring (eumdac | osi_saf | gportal | smos | earthdata | hsaf | space_track)
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
    # Type-only import, binding ``xr`` for the string return annotations
    # below (e.g. ``"xr.DataTree | None"``): this module imports xarray
    # lazily inside the functions that use it, so a type checker cannot
    # otherwise resolve ``xr`` in those annotations.
    import xarray as xr

    from .core.recipe import Recipe

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ismn's own file-collection logger ('ismn_meta_collector') logs one INFO line 
# per station file while building ISMN_Interface's metadata -- hundreds of
# lines for a real archive. Pinned to WARNING unconditionally (matching the 
# "matplotlib" cap below).
logging.getLogger("ismn_meta_collector").setLevel(logging.WARNING)


def main(argv=None) -> None:
    from .core.sar_sources import AVAILABLE_SATELLITES

    parser = argparse.ArgumentParser(
        prog="sar-validate",
        description="SAR L2 Ocean Data Validation Toolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available SAR sources (--sar-source): {', '.join(AVAILABLE_SATELLITES)}
  See sar_validation.core.sar_sources for which recipe variables each supports.

Examples:
  sar-validate --list-recipes
  sar-validate --create-recipe wind
  sar-validate --create-recipe waves --altimeter-freq 5hz
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
        choices=["eumdac", "osi_saf", "gportal", "smos", "earthdata", "hsaf", "space_track"],
        help="Prompt for a username/password and store them in the OS keyring "
             "for SERVICE (eumdac | osi_saf | gportal | smos | earthdata | hsaf | space_track)",
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
        "--sar-source",
        metavar="SATELLITE",
        default=None,
        help="Which satellite to validate against (used with --create-recipe), "
             f"e.g. {', '.join(AVAILABLE_SATELLITES)} -- defaults to the "
             "category's existing satellite if omitted. The matching product "
             "for the requested recipe category is picked automatically (e.g. "
             "sentinel1 resolves to L2 OCN for wind/waves/currents, CLMS SSM "
             "for soil_moisture). See sar_validation.core.sar_sources for details.",
    )
    parser.add_argument(
        "--hfradar-resolution",
        choices=["finest", "1km", "2km", "6km"],
        default=None,
        help="NOAA HF-radar grid resolution for --create-recipe currents "
             "(only valid with that template). 'finest' uses the best "
             "resolution available for the recipe's region.",
    )
    parser.add_argument(
        "--altimeter-freq",
        choices=["1hz", "5hz", "both"],
        default=None,
        help="Along-track altimeter frequency for --create-recipe waves "
             "(default: 1hz). Only valid with --create-recipe waves -- 5 Hz "
             "has no WIND_SPEED, so wind recipes always use 1hz regardless "
             "of this flag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what will be downloaded without actually downloading",
    )
    parser.add_argument(
        "--dry-collocation",
        action="store_true",
        help=(
            "Predict which validation sources would collocate with this "
            "recipe's SAR data, without downloading anything from any "
            "source. Prints a report and writes dry_collocation_report.json."
        ),
    )
    parser.add_argument(
        "--download-all-in-bbox",
        action="store_true",
        help=(
            "Disable the default collocation-based skip-gating: download "
            "everything in the recipe's bbox/window regardless of "
            "predicted collocation (today's pre-existing behavior)."
        ),
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

    if args.hfradar_resolution is not None and args.create_recipe != "currents":
        parser.error("--hfradar-resolution is only valid with --create-recipe currents")

    if args.altimeter_freq is not None and args.create_recipe != "waves":
        parser.error("--altimeter-freq is only valid with --create-recipe waves")

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
            sar_source=args.sar_source,
            hfradar_resolution=args.hfradar_resolution,
            altimeter_freq=args.altimeter_freq,
        )
    elif args.recipe:
        _execute_recipe(
            args.recipe,
            dry_run=args.dry_run,
            dry_collocation=args.dry_collocation,
            output_dir=args.output_dir,
            force_download=args.force_download,
            download_all_in_bbox=args.download_all_in_bbox,
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
    """
    Prompt for a username/password and store them in the OS keyring.

    Backs ``sar-validate --set-credential {eumdac,osi_saf,gportal,smos,earthdata,hsaf,space_track}``.
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




def _build_currents_config(
    limit: Optional[int] = None,
    sar_source: str = "sentinel1_l2_ocn",
    hfradar_resolution: Optional[str] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
):
    """
    Build the 'currents' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects.
    Resolves geographic_bounds from the bbox overrides (or the template's
    own default) *inside* this function -- unlike the other templates --
    because the HF-radar source choice below depends on knowing the final
    bbox before validation_sources is built.
    """
    from .core.sar_sources import resolve_sar_source
    sar_source = resolve_sar_source(sar_source, "currents")
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        LayerVsLayerCollocation,
        PointVsLayerCollocation,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )
    from .downloaders._noaa_hfr_regions import NOAA_HFR_REGIONS, region_bbox_overlaps

    default_bounds = GeographicBounds(-20.0, 0.0, 35.0, 60.0)
    bounds = GeographicBounds(
        min_lon=min_lon if min_lon is not None else default_bounds.min_lon,
        max_lon=max_lon if max_lon is not None else default_bounds.max_lon,
        min_lat=min_lat if min_lat is not None else default_bounds.min_lat,
        max_lat=max_lat if max_lat is not None else default_bounds.max_lat,
    )

    overlapping = [
        (name, region) for name, region in NOAA_HFR_REGIONS.items()
        if region_bbox_overlaps(region, bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat)
    ]

    _RESOLUTION_VALUES = {"1km": 1.0, "2km": 2.0, "6km": 6.0}

    if overlapping:
        download_kwargs: dict = {}
        if hfradar_resolution is not None:
            if hfradar_resolution == "finest":
                download_kwargs["resolution_km"] = "finest"
            else:
                requested_km = _RESOLUTION_VALUES[hfradar_resolution]
                download_kwargs["resolution_km"] = requested_km
                if len(overlapping) == 1:
                    _, region = overlapping[0]
                    available = set(region["thredds_resolutions_km"])
                    if region["erddap_datasets"] is not None:
                        available |= set(region["erddap_datasets"])
                    if requested_km not in available:
                        from .downloaders._noaa_hfr_regions import _resolution_token
                        names = ", ".join(sorted(_resolution_token(r) for r in available))
                        raise ValueError(
                            f"resolution {hfradar_resolution} not available for "
                            f"{overlapping[0][0]}; available: {names}"
                        )
        hf_radar_sources = [
            ValidationDataSource(source_type="hf_radar_us", download_kwargs=download_kwargs),
        ]
    else:
        if hfradar_resolution is not None:
            logger.warning(
                "--hfradar-resolution has no effect: the recipe's bounds don't "
                "overlap any NOAA HF-radar region."
            )
        hf_radar_sources = [
            ValidationDataSource(source_type="hf_radar"),
            ValidationDataSource(source_type="hf_radar_historical"),
        ]

    return RecipeConfig(
        name="Ocean Currents Validation",
        description=(
            "Validate Sentinel-1 WV mode ocean currents\n"
            "against HF radar and drifting buoys."
        ),
        variable="currents",
        variable_specs={"components": ["zonal", "meridional"]},
        geographic_bounds=bounds,
        sar_data=SARDataSpec(source=sar_source, swath_mode=["WV","IW","EW","SM"], max_downloads=limit),
        validation_sources=[
            *hf_radar_sources,
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
            # HYCOM ocean model (surface currents) -- unconditional, unlike
            # hf_radar_sources above: HYCOM has global coverage, so it
            # applies regardless of bbox (no NOAA-region-overlap branching
            # needed the way hf_radar_us/hf_radar differ).
            ValidationDataSource(source_type="hycom"),
        ],
        collocation=CollocationType(
            point_vs_layer=PointVsLayerCollocation(),
            layer_vs_layer=LayerVsLayerCollocation(
                layer_type_specs={
                    "hf_radar_grid": {
                        "time_tolerance_minutes": 30,
                        "distance_weighting": "equal",
                        "dedup_nearest_in_time": True,
                    },
                    # HYCOM ocean model -- tuning mirrors
                    # DEFAULT_LAYER_TYPE_SPECS's "hycom" entry (recipe.py).
                    # time_tolerance_minutes=360 (6h = 2x HYCOM's 3-hourly
                    # cadence) is the minimum that guarantees
                    # ModelLayerCollocation always finds a bracketing pair
                    # of granules -- see recipe.MODEL_CADENCE_HOURS.
                    "hycom": {
                        "time_tolerance_minutes": 360,
                        "aggregation_window_km": 4.6,
                        "distance_weighting": "equal",
                        "method": "cell-averaging",
                        "temporal_method": "hyperbolic",
                    },
                }
            ),
        ),
    )


def _build_wind_config(limit: Optional[int] = None, sar_source: str = "sentinel1_l2_ocn"):
    """Build the 'wind' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects,
    mirroring ``_build_currents_config``.
    """
    from .core.sar_sources import resolve_sar_source
    sar_source = resolve_sar_source(sar_source, "wind")
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        LayerVsLayerCollocation,
        PointVsLayerCollocation,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )

    if sar_source == "radarsat2":
        description = (
            "Validate RADARSAT-2 SAR-derived wind speed (NOAA NCEI,\n"
            "0.5 km, C-band) against moorings, buoys, ASCAT scatterometer,\n"
            "HY-2B/HY-2C/Oceansat-3 scatterometer, 1 Hz altimeter, and RSS\n"
            "radiometer (AMSR2) ocean winds. Speed only -- this product\n"
            "carries no independently SAR-retrieved wind direction."
        )
        # RADARSAT-2 has no SAR-retrieved direction component (see
        # description above), and swath_mode is Sentinel-1-specific
        # terminology.
        components = ["speed"]
        swath_mode: list[str] = []
    else:
        description = (
            "Validate Sentinel-1 IW/EW mode wind speed and direction\n"
            "against moorings, buoys, ASCAT scatterometer, HY-2B/HY-2C/\n"
            "Oceansat-3 scatterometer, 1 Hz altimeter, and RSS radiometer\n"
            "(AMSR2) ocean winds."
        )
        components = ["speed", "direction"]
        swath_mode = ["IW", "EW"]

    return RecipeConfig(
        name="Wind Validation",
        description=description,
        variable="wind",
        variable_specs={"components": components},
        geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
        sar_data=SARDataSpec(source=sar_source, swath_mode=swath_mode, max_downloads=limit),
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
            # ERA5 reanalysis (Copernicus CDS).
            ValidationDataSource(source_type="era5"),
        ],
        collocation=CollocationType(
            point_vs_layer=PointVsLayerCollocation(),
            layer_vs_layer=LayerVsLayerCollocation(
                layer_type_specs={
                    # tuning mirrors DEFAULT_LAYER_TYPE_SPECS's "era5_wind"
                    # entry (recipe.py). time_tolerance_minutes=120 (2h =
                    # 2x ERA5's 1-hourly cadence) is the minimum that
                    # guarantees ModelLayerCollocation always finds a
                    # bracketing pair of granules -- see
                    # recipe.MODEL_CADENCE_HOURS.
                    "era5_wind": {
                        "time_tolerance_minutes": 120,
                        "aggregation_window_km": 12.5,
                        "distance_weighting": "equal",
                        "method": "cell-averaging",
                        "temporal_method": "hyperbolic",
                    },
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


def _build_waves_config(
    limit: Optional[int] = None,
    sar_source: str = "sentinel1_l2_ocn",
    altimeter_freq: Optional[str] = None,
):
    """
    Build the 'waves' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects,
    mirroring ``_build_wind_config``/``_build_currents_config``.

    ``altimeter_freq`` selects which along-track altimeter frequency the
    generated recipe validates against: "1hz" (default), "5hz", or "both".
    1 Hz and 5 Hz sample the same ground track at different point spacing
    (~7km vs ~1.4km) — mixing both by default over-samples SAR pixels near
    the ground track, skewing validation statistics toward them. ``None`` 
    is treated as "1hz".
    """
    from .core.sar_sources import resolve_sar_source
    sar_source = resolve_sar_source(sar_source, "waves")
    from .core.recipe import (
        CollocationType,
        GeographicBounds,
        LayerVsLayerCollocation,
        PointVsLayerCollocation,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )

    _ALTIMETER_FREQ_LISTS = {
        "1hz": ["1hz"],
        "5hz": ["5hz"],
        "both": ["1hz", "5hz"],
    }
    altimeter_freqs = _ALTIMETER_FREQ_LISTS[altimeter_freq or "1hz"]

    _ALTIMETER_LAYER_SPECS = {
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
    altimeter_layer_specs = {
        f"altimeter_{freq}": _ALTIMETER_LAYER_SPECS[f"altimeter_{freq}"]
        for freq in altimeter_freqs
    }

    return RecipeConfig(
        name="Wave Height Validation",
        description=(
            "Validate Sentinel-1 significant wave height\n"
            "against moorings, tidal gauges, drifters, and altimeter."
        ),
        variable="waves",
        variable_specs={"components": ["significant_wave_height"]},
        geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
        sar_data=SARDataSpec(source=sar_source, swath_mode=["WV","SM"], max_downloads=limit),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="tidal_gauge"),
            ValidationDataSource(source_type="drifter"),
            ValidationDataSource(
                source_type="altimeter",
                download_kwargs={"frequencies": altimeter_freqs},
            ),
            # ERA5 reanalysis (Copernicus CDS).
            ValidationDataSource(source_type="era5"),
        ],
        collocation=CollocationType(
            point_vs_layer=PointVsLayerCollocation(),
            layer_vs_layer=LayerVsLayerCollocation(
                layer_type_specs={
                    # tuning mirrors DEFAULT_LAYER_TYPE_SPECS's "era5_waves"
                    # entry (recipe.py) -- see the identical rationale in
                    # _build_wind_config's "era5_wind" entry above.
                    "era5_waves": {
                        "time_tolerance_minutes": 120,
                        "aggregation_window_km": 12.5,
                        "distance_weighting": "equal",
                        "method": "cell-averaging",
                        "temporal_method": "hyperbolic",
                    },
                    **altimeter_layer_specs,
                }
            ),
        ),
    )


def _build_soil_moisture_config(limit: Optional[int] = None, sar_source: str = "sentinel1_clms_ssm"):
    """
    Build the 'soil_moisture' recipe template's RecipeConfig.

    Extracted from ``_create_recipe`` so the template content is
    unit-testable independent of the CLI's file-writing side effects,
    mirroring ``_build_wind_config``/``_build_currents_config``. Depth and
    collocation-tolerance defaults are read from the resolved
    ``SARSourceSpec`` (``sar_sources.SAR_SOURCES``), so a NISAR SME2
    recipe automatically gets its own template values instead of
    Sentinel-1 CLMS SSM's.
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
    from .core.sar_sources import SAR_SOURCES, resolve_sar_source

    sar_source = resolve_sar_source(sar_source, "soil_moisture")
    spec = SAR_SOURCES[sar_source]
    # SARSourceSpec's default_* fields are Optional[...] because sources
    # outside the "soil_moisture" variable set (e.g. sentinel1_l2_ocn)
    # leave them unset; every "soil_moisture" source populates them, so
    # narrow the Optional types for mypy here.
    assert spec.default_time_tolerance_minutes is not None
    assert spec.default_aggregation_window_km is not None
    assert spec.default_spatial_tolerance_km is not None
    time_tol = spec.default_time_tolerance_minutes
    agg_km = spec.default_aggregation_window_km
    spatial_km = spec.default_spatial_tolerance_km

    # era5_soil_moisture is always present (unlike the ssm-sensor overrides
    # below, which only apply for a non-default sar_source): tuning mirrors
    # DEFAULT_LAYER_TYPE_SPECS's own entry (recipe.py). 720 min (12h)
    # already exceeds the 120 min bracket-safety minimum (2x ERA5's
    # 1-hourly cadence -- see recipe.MODEL_CADENCE_HOURS), so no
    # cadence-driven bump is needed here the way hycom/era5_wind/
    # era5_waves needed one.
    layer_type_specs = {
        "era5_soil_moisture": {
            "time_tolerance_minutes": 720,
            "aggregation_window_km": 5.0,
            "distance_weighting": "equal",
            "method": "cell-averaging",
            "temporal_method": "hyperbolic",
        },
    }
    if sar_source != "sentinel1_clms_ssm":
        # Sentinel-1 CLMS SSM's ±12h default already matches
        # DEFAULT_LAYER_TYPE_SPECS's own fallback (recipe.py) -- no
        # per-recipe override needed. Any other source (e.g. NISAR's ±6h)
        # must set an explicit layer_type_specs override, since the global
        # fallback stays 720 for every recipe that doesn't override it.
        layer_type_specs.update({
            key: {"time_tolerance_minutes": time_tol}
            for key in (
                "scatterometer_ssm", "radiometer_ssm",
                "amsr_ssm", "smap_ssm", "smos_ssm", "cds_ssm",
            )
        })
    layer_vs_layer = LayerVsLayerCollocation(layer_type_specs=layer_type_specs)

    # C3S CDS product type: active (%) pairs with Sentinel-1 CLMS (%);
    # passive (m3 m-3) pairs with NISAR SME2.
    cds_product_type = "passive" if sar_source == "nisar_sme2" else "active"

    if sar_source == "nisar_sme2":
        description = (
            "Validate NISAR SME2 (beta) L-band Surface Soil Moisture\n"
            "against ISMN in-situ stations, ASCAT, AMSR2, SMAP, SMOS,\n"
            "and C3S CDS Passive SSM (multi-radiometer composite, 0.25°)."
        )
    else:
        description = (
            "Validate Sentinel-1 CLMS Surface Soil Moisture (1 km, Europe,\n"
            "daily) against ISMN in-situ stations and C3S CDS Active SSM\n"
            "(multi-ASCAT composite, 0.25°)."
        )

    return RecipeConfig(
        name="Soil Moisture Validation",
        description=description,
        variable="soil_moisture",
        variable_specs={"components": ["soil_moisture"]},
        geographic_bounds=GeographicBounds(-10.0, 30.0, 35.0, 60.0),
        sar_data=SARDataSpec(source=sar_source, max_downloads=limit),
        validation_sources=[
            ValidationDataSource(
                source_type="ismn", min_depth=spec.default_min_depth, max_depth=spec.default_max_depth,
            ),
            ValidationDataSource(source_type="ascat_ssm"),
            ValidationDataSource(source_type="amsr_ssm"),
            ValidationDataSource(source_type="smap_ssm"),
            ValidationDataSource(source_type="smos_ssm"),
            ValidationDataSource(
                source_type="cds_ssm",
                download_kwargs={"product_type": cds_product_type},
            ),
            # ERA5-Land reanalysis (Copernicus CDS) -- tuning comes from
            # DEFAULT_LAYER_TYPE_SPECS's "era5_soil_moisture" entry
            # (recipe.py), no per-recipe layer_vs_layer override needed.
            ValidationDataSource(source_type="era5"),
        ],
        collocation=CollocationType(
            point_vs_layer=PointVsLayerCollocation(
                spatial_tolerance_km=spatial_km,
                aggregation_window_km=agg_km,
                distance_weighting="equal",
                interpolation_method="nearest",
                time_tolerance_minutes=time_tol,
            ),
            layer_vs_layer=layer_vs_layer,
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
    sar_source: Optional[str] = None,
    hfradar_resolution: Optional[str] = None,
    altimeter_freq: Optional[str] = None,
) -> None:
    from .core.recipe import (
        GeographicBounds,
        Recipe,
        TemporalBounds,
    )

    defaults = {
        "wind": "sentinel1_l2_ocn", "currents": "sentinel1_l2_ocn",
        "waves": "sentinel1_l2_ocn", "soil_moisture": "sentinel1_clms_ssm",
    }
    resolved_sar_source = sar_source if sar_source is not None else defaults.get(name, "sentinel1_l2_ocn")

    # All four templates are built eagerly (see the `if name not in templates` 
    # check below), but an explicit --sar-source only applies to the *requested* 
    # category: e.g. `--create-recipe soil_moisture --sar-source sentinel1_clms_ssm` 
    # must not also try (and fail) to build the "wind" template with a source
    # that is only valid for soil_moisture. Every other, unrequested category
    # keeps building with its own default source.
    def _source_for(category: str) -> str:
        return resolved_sar_source if category == name else defaults[category]

    try:
        templates = {
            "wind": _build_wind_config(limit, _source_for("wind")),
            "currents": _build_currents_config(
                limit, _source_for("currents"), hfradar_resolution,
                min_lon, max_lon, min_lat, max_lat,
            ),
            "soil_moisture": _build_soil_moisture_config(limit, _source_for("soil_moisture")),
            "waves": _build_waves_config(limit, _source_for("waves"), altimeter_freq),
        }
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

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
    dry_collocation: bool = False,
    output_dir: Optional[str] = None,
    force_download: bool = False,
    download_all_in_bbox: bool = False,
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

    if dry_collocation:
        from .core.dry_collocation import (
            discover_sar_footprints_dry,
            predict_collocation,
            render_console_table,
            report_to_json,
        )
        from .downloaders.base import build_output_dir

        sar_footprints = discover_sar_footprints_dry(recipe.config.sar_data, recipe.config)
        report = predict_collocation(recipe.config, sar_footprints, recipe_path=recipe_path)
        print(render_console_table(report))
        if recipe.config.output_dir:
            output_base = Path(recipe.config.output_dir)
        else:
            b = recipe.config.geographic_bounds
            t = recipe.config.temporal_bounds
            output_base = build_output_dir(t.start, t.end, b.min_lon, b.max_lon, b.min_lat, b.max_lat)
        report_path = output_base / "dry_collocation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_to_json(report))
        print(f"\nWrote {report_path}")
        return

    orchestrator = DataOrchestrator(
        recipe, dry_run=dry_run, force_download=force_download,
        download_all_in_bbox=download_all_in_bbox,
    )

    # Skip download if data was already downloaded successfully and not forcing re-download
    download_step_ran = False
    if not dry_run and not force_download and _is_already_downloaded(orchestrator.base_dir, recipe):
        if not orchestrator.previous_sar_data_found():
            print(
                "\nNo SAR data found for this window — stopping before "
                "validation downloads and further pipeline steps."
            )
            return
        logger.info(
            "Data already downloaded in %s — skipping Step 1.",
            orchestrator.base_dir,
        )
        print(f"Step 1 skipped — data already present in {orchestrator.base_dir}")
        success = True
    else:
        download_step_ran = True
        success = orchestrator.download_all()

        if not orchestrator.metadata.get("sar_data_found", True):
            if dry_run:
                print(
                    "\nNo SAR data found for this window — stopping dry run "
                    "before validation sources."
                )
            else:
                print(
                    "\nNo SAR data found for this window — stopping before "
                    "validation downloads and further pipeline steps."
                )
            return

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
        # Never skip when Step 1 did fresh download work this run.
        if download_step_ran or not datatree_path.exists():
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
            # Same reasoning as Step 2 above: never skip when Step 1 ran fresh.
            if download_step_ran or not collocation_path.exists():
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


def _is_already_downloaded(base_dir: Path, recipe: Optional["Recipe"] = None) -> bool:
    """
    Return True if *base_dir* has a download_metadata.json recording a
    complete, reusable download for *recipe*.

    Checks: no errors recorded; no source still stick ``awaiting_manual_archive``
    (0 files collected without being an error); if *recipe* requests an 
    ``era5`` source, its ``variable`` matches the recorded run's (ERA5's
    downloaded file depends on ``variable``, unlike SAR's L2 OCN product
    which contains every field regardless of which recipe downloaded it);
    and every validation ``source_type`` *recipe* requests is present 
    among the recorded run's ``downloads`` keys (normalized via 
    ``_normalize_recorded_source_type`` so batched sources like ``mooring``/
    ``buoy`` recorded under ``insitu`` still match).
    """
    import json as _json

    from .core.orchestrator import _INSITU_TYPES

    def _normalize_recorded_source_type(source_type: str) -> str:
        return "insitu" if source_type in _INSITU_TYPES else source_type

    meta_path = base_dir / "download_metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
        if meta.get("errors", ["placeholder"]) != []:
            return False
        downloads = meta.get("downloads", {})
        recorded_variable = meta.get("variable")
        # A real download_metadata.json (written by DataOrchestrator) always
        # has "variable" set -- only a synthetic/legacy file could omit it,
        # in which case there is nothing to contradict *recipe* and the old
        # trust-it behavior applies (a minimal ``{"errors": [], ...}``
        # fixture, as several tests use to force this shortcut without
        # touching the network, must keep working).
        recipe_has_era5 = recipe is not None and any(
            s.source_type == "era5" for s in recipe.config.validation_sources
        )
        if recipe is not None and recipe_has_era5 and recorded_variable is not None \
                and recorded_variable != recipe.config.variable:
            return False
        if recipe is not None and recipe.config.validation_sources:
            requested_types = {
                _normalize_recorded_source_type(s.source_type)
                for s in recipe.config.validation_sources
            }
            recorded_types = set(downloads.keys())
            if not requested_types <= recorded_types:
                return False
        return all(
            entry.get("status") != "awaiting_manual_archive"
            for entry in downloads.values()
        )
    except Exception:
        return False


def _convert_data(recipe, base_dir: Path) -> "xr.DataTree | None":
    """
    Run step 2: convert downloaded files to a DataTree.
    """
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
    """
    Run step 3: load DataTree and run collocation.
    """
    import xarray as xr

    from .core.collocation import run_collocation
    from .core.visualization import plot_collocation_diagnostics

    datatree_path = base_dir / "datatree.nc"
    if not datatree_path.exists():
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
    """
    Run step 4: compute validation statistics from collocation_results<suffix>.nc.
    """
    import xarray as xr

    from .core.statistics import run_statistics, run_statistics_cds_ssm, run_statistics_native_units

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
        cds_results = run_statistics_cds_ssm(collocation_ds, recipe, base_dir, filename_suffix=filename_suffix)
        for key in cds_results:
            print(f"  C3S CDS SSM statistics saved: validation_statistics_{key}_cds_ssm{filename_suffix}.nc/.csv")


def _stats_already_computed(recipe, base_dir: Path, filename_suffix: str = "") -> bool:
    """
    Return True if every ``validation_statistics_*<suffix>.nc`` file Step 4
    would produce already exists on disk, so ``_execute_recipe`` can skip
    recomputation the same way Steps 1-3 skip their own already-done work.

    Opens ``collocation_results<suffix>.nc`` (Step 3's output) and applies
    the same dataset-aware ``filter_variable_pairs`` selection ``run_statistics``
    uses to decide *which* files it writes — reusing ``_load_precomputed_stats``
    for the lookup so the pair-matching logic is not duplicated. Also requires
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
    """
    Load ``validation_statistics_<sar_var>_vs_<val_var><suffix>.nc`` files already
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
    """
    Run step 5: generate validation plots and save PDF to <base_dir>/, PNG to <base_dir>/plots/.
    """
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
    # reads add up for files that easily fit in memory (tens of MB here).
    collocation_ds = xr.open_dataset(str(coll_path)).load()
    datatree = xr.open_datatree(str(datatree_path), engine="netcdf4").load()

    stats_ds_map = _load_precomputed_stats(recipe, collocation_ds, base_dir, filename_suffix)
    download_warnings = _load_download_warnings(base_dir)

    if recipe.config.variable == "soil_moisture":
        native_units_stats_ds_map = _load_precomputed_stats(
            recipe, collocation_ds, base_dir, filename_suffix=f"_native_units{filename_suffix}",
        )
        cds_ssm_stats_ds_map: Optional[dict] = _load_precomputed_stats(
            recipe, collocation_ds, base_dir, filename_suffix=f"_cds_ssm{filename_suffix}",
        ) or None
    else:
        native_units_stats_ds_map = None
        cds_ssm_stats_ds_map = None

    validation_report(collocation_ds, datatree, recipe,
                      stats_ds_map=stats_ds_map or None,
                      out_dir=base_dir,
                      filename_suffix=filename_suffix,
                      download_warnings=download_warnings,
                      layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
                      native_units_stats_ds_map=native_units_stats_ds_map or None,
                      cds_ssm_stats_ds_map=cds_ssm_stats_ds_map)
    pdf_path = base_dir / f"validation_report{filename_suffix}.pdf"
    if pdf_path.exists():
        print(f"  PDF report saved to {pdf_path}")
    print(f"  Collocation diagnostics PNG saved to {base_dir / 'plots'}")


def _load_download_warnings(base_dir: Path) -> Optional[list[str]]:
    """
    Read download_metadata.json's ``errors`` and ``notices`` lists, if
    present, for surfacing on the PDF cover page. ``notices`` are
    non-failure observations (e.g. "no data found for this window") that
    still deserve a durable, easy-to-find spot rather than only a
    console-log line that scrolls past during a long run. Returns None if
    there's no metadata file, it cannot be parsed, or it has neither.
    """
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
