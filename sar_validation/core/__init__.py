"""sar_validation.core — recipe, orchestrator, converter, collocation, statistics, visualization."""

from ._variable_map import VARIABLE_PAIRS, infer_variable_pairs
from .collocation import (
    CollocatedPoint,
    LayerLayerCollocation,
    PointLayerCollocation,
)
from .datatree_converter import DataTreeConverter
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
from .statistics import compute_statistics, run_statistics, save_statistics
from .visualization import (
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
    "DataOrchestrator",
    "VARIABLE_PAIRS",
    "infer_variable_pairs",
    "compute_statistics",
    "save_statistics",
    "run_statistics",
    "plot_scatter",
    "plot_geographic",
    "plot_statistics",
    "plot_residuals",
    "validation_report",
]
