"""sar_validation.core — recipe, orchestrator, converter, collocation, statistics, visualization."""

from .recipe import (
    Recipe,
    RecipeConfig,
    GeographicBounds,
    TemporalBounds,
    SARDataSpec,
    ValidationDataSource,
    CollocationType,
)
from .datatree_converter import DataTreeConverter
from .collocation import (
    CollocatedPoint,
    PointLayerCollocation,
    TrajectoryLayerCollocation,
    LayerLayerCollocation,
)
from .orchestrator import DataOrchestrator
from ._variable_map import VARIABLE_PAIRS, infer_variable_pairs
from .statistics import compute_statistics, save_statistics, run_statistics
from .patch_extractor import extract_patches, add_patches_to_dataset, run_patch_extraction
from .visualization import (
    plot_scatter,
    plot_geographic,
    plot_statistics,
    plot_residuals,
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
    "TrajectoryLayerCollocation",
    "LayerLayerCollocation",
    "DataOrchestrator",
    "VARIABLE_PAIRS",
    "infer_variable_pairs",
    "compute_statistics",
    "save_statistics",
    "run_statistics",
    "extract_patches",
    "add_patches_to_dataset",
    "run_patch_extraction",
    "plot_scatter",
    "plot_geographic",
    "plot_statistics",
    "plot_residuals",
    "validation_report",
]
