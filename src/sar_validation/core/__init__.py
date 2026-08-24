"""sar_validation.core — recipe, orchestrator, converter, collocation, statistics, visualization."""

from ._cf_metadata import (
    COORD_ATTRS,
    INSITU_VARIABLE_ATTRS,
    PRODUCT_REFERENCES,
    annotate_collocation_ds,
    apply_cf_metadata,
    sanitize_raw_attrs,
)
from ._variable_map import VARIABLE_PAIRS, infer_variable_pairs
from .collocation import (
    CollocatedPoint,
    LayerLayerCollocation,
    PointLayerCollocation,
    run_collocation,
)
from .datatree_converter import DataTreeConverter
from .model_collocation import ModelLayerCollocation
from .orbit_coverage import (
    SATELLITE_ORBIT_SPECS,
    SatelliteOrbitSpec,
    TleFetchError,
    get_tle,
    orbit_overlaps_bbox,
)
from .orchestrator import DataOrchestrator
from .recipe import (
    CollocationType,
    GeographicBounds,
    Recipe,
    RecipeConfig,
    SARDataSpec,
    TemporalBounds,
    ValidationDataSource,
)
from .sar_sources import SAR_SOURCES, SARSourceSpec, resolve_sar_source
from .statistics import (
    MIN_N_FOR_CORRELATION,
    add_rescaled_sar_column,
    compute_statistics,
    compute_statistics_soil_moisture,
    fit_sar_to_val_transform,
    run_statistics,
    run_statistics_cds_ssm,
    run_statistics_native_units,
    save_statistics,
)
from .visualization import (
    plot_collocation_diagnostics,
    plot_geographic,
    plot_residuals,
    plot_scatter,
    plot_statistics,
    validation_report,
)

__all__ = [
    "Recipe",
    "RecipeConfig",
    "GeographicBounds",
    "TemporalBounds",
    "SARDataSpec",
    "ValidationDataSource",
    "CollocationType",
    "DataTreeConverter",
    "CollocatedPoint",
    "PointLayerCollocation",
    "LayerLayerCollocation",
    "ModelLayerCollocation",
    "run_collocation",
    "DataOrchestrator",
    "VARIABLE_PAIRS",
    "infer_variable_pairs",
    "SARSourceSpec",
    "SAR_SOURCES",
    "resolve_sar_source",
    "PRODUCT_REFERENCES",
    "COORD_ATTRS",
    "INSITU_VARIABLE_ATTRS",
    "sanitize_raw_attrs",
    "apply_cf_metadata",
    "annotate_collocation_ds",
    "SatelliteOrbitSpec",
    "SATELLITE_ORBIT_SPECS",
    "TleFetchError",
    "get_tle",
    "orbit_overlaps_bbox",
    "compute_statistics",
    "compute_statistics_soil_moisture",
    "add_rescaled_sar_column",
    "fit_sar_to_val_transform",
    "save_statistics",
    "run_statistics",
    "run_statistics_native_units",
    "run_statistics_cds_ssm",
    "MIN_N_FOR_CORRELATION",
    "plot_scatter",
    "plot_geographic",
    "plot_statistics",
    "plot_residuals",
    "plot_collocation_diagnostics",
    "validation_report",
]
