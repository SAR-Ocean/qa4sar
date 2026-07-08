"""
Recipe dataclasses for the SAR validation toolbox.

A recipe (YAML or JSON) is the single configuration object that drives the
entire pipeline:

    step 1 — download    → DataOrchestrator reads the recipe
    step 3 — collocation → uses collocation parameters from the recipe
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Basic building-blocks
# ---------------------------------------------------------------------------

@dataclass
class GeographicBounds:
    """Bounding box in decimal degrees (WGS-84)."""
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TemporalBounds:
    """Inclusive time window (ISO-8601 strings)."""
    start: str   # e.g. "2026-01-01" or "2026-01-01T00:00:00"
    end: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ValidationDataSource:
    """One validation data source referenced in a recipe."""

    source_type: str
    """
    Platform / product type.

    Accepted values:
      in-situ   : mooring, buoy, ferrybox, drifter, tidal_gauge
      satellite : scatterometer, altimeter
      coastal   : hf_radar
      upper-air : radiosonde  (not yet implemented)
    """

    # Optional depth filter (for in-situ and HF radar sources)
    min_depth: float = -20.0
    max_depth: float = 20.0

    # Extra keyword arguments forwarded to the downloader
    download_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Override per-source collocation tolerances
    collocation_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SARDataSpec:
    """Specification for the SAR L2_OCN product to validate."""
    satellite: str = "Sentinel-1"
    product_level: str = "L2_OCN"
    swath_mode: List[str] = field(default_factory=lambda: ["IW", "EW"])
    max_downloads: Optional[int] = None   # None → download all found products

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PointVsLayerCollocation:
    """Collocation config for point vs gridded SAR data (buoys, moorings)."""
    time_tolerance_minutes: int = 30
    spatial_tolerance_km: float = 12.5
    interpolation_method: str = "nearest"
    aggregation_window_km: float = 5.0
    validation_temporal_averaging_minutes: int = 30
    distance_weighting: str = "gaussian"
    gaussian_sigma_km: float = 2.0
    patch_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerVsLayerCollocation:
    """Collocation config for gridded vs gridded SAR data (scatterometer, altimeter, HF radar).
    
    Maps data source types to their specific collocation parameters.
    Supports two collocation methods: 'cell-averaging' (clusters scatterometer into grid cells)
    or 'individual' (matches each scatterometer point to nearest SAR pixel).
    """
    layer_type_specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """
    Dict mapping layer data source type → collocation parameters.
    
    Example:
        layer_type_specs:
          scatterometer:
            time_tolerance_minutes: 180
            aggregation_window_km: 12.5
            distance_weighting: equal
          altimeter:
            time_tolerance_minutes: 120
            aggregation_window_km: 10.0
            distance_weighting: equal
    """
    method: str = "cell-averaging"
    """
    Collocation method for layer-vs-layer data:
    - 'cell-averaging': Cluster scatterometer into grid cells (~57 matches from ~3000 points)
    - 'individual': Match each scatterometer point to nearest SAR pixel (~3000+ matches)
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollocationType:
    """Collocation configuration for SAR validation.
    
    Supports both point_vs_layer (buoys, moorings) and layer_vs_layer
    (scatterometer, altimeter, HF radar) collocation strategies.
    Per-source overrides in ValidationDataSource.collocation_kwargs take
    precedence over all recipe defaults.
    """
    point_vs_layer: PointVsLayerCollocation = field(default_factory=PointVsLayerCollocation)
    layer_vs_layer: Optional[LayerVsLayerCollocation] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"point_vs_layer": self.point_vs_layer.to_dict()}
        if self.layer_vs_layer is not None:
            result["layer_vs_layer"] = self.layer_vs_layer.to_dict()
        return result


# ---------------------------------------------------------------------------
# Top-level recipe config
# ---------------------------------------------------------------------------

@dataclass
class RecipeConfig:
    """Complete, self-contained recipe configuration."""

    # --- Required fields (no default) ---
    name: str
    variable: str       # "wind" | "currents" | "waves"

    # --- Optional metadata ---
    description: str = ""
    version: str = "1.0"
    variable_specs: Dict[str, Any] = field(default_factory=dict)

    # --- Domain ---
    geographic_bounds: GeographicBounds = field(
        default_factory=lambda: GeographicBounds(-20.0, 0.0, 35.0, 60.0)
    )
    temporal_bounds: TemporalBounds = field(
        default_factory=lambda: TemporalBounds("2026-01-01", "2026-01-02")
    )

    # --- SAR product ---
    sar_data: SARDataSpec = field(default_factory=SARDataSpec)

    # --- Validation sources ---
    validation_sources: List[ValidationDataSource] = field(default_factory=list)

    # --- Collocation ---
    collocation: CollocationType = field(
        default_factory=CollocationType
    )

    # --- Output ---
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "variable": self.variable,
            "variable_specs": self.variable_specs,
            "geographic_bounds": self.geographic_bounds.to_dict(),
            "temporal_bounds": self.temporal_bounds.to_dict(),
            "sar_data": self.sar_data.to_dict(),
            "validation_sources": [s.to_dict() for s in self.validation_sources],
            "collocation": self.collocation.to_dict(),
            "output_dir": self.output_dir,
        }


# ---------------------------------------------------------------------------
# Recipe loader / writer
# ---------------------------------------------------------------------------

class Recipe:
    """Load, save, and manage a RecipeConfig."""

    def __init__(self, config: RecipeConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Recipe":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Recipe":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        geo = data.get("geographic_bounds", {})
        temporal = data.get("temporal_bounds", {})
        sar = data.get("sar_data", {})
        coll = data.get("collocation", {})

        # Parse point_vs_layer section
        pvl_config = coll.get("point_vs_layer", {})
        point_vs_layer = PointVsLayerCollocation(
            time_tolerance_minutes=pvl_config.get("time_tolerance_minutes", 30),
            spatial_tolerance_km=pvl_config.get("spatial_tolerance_km", 12.5),
            interpolation_method=pvl_config.get("interpolation_method", "nearest"),
            aggregation_window_km=pvl_config.get("aggregation_window_km", 5.0),
            validation_temporal_averaging_minutes=pvl_config.get("validation_temporal_averaging_minutes", 30),
            distance_weighting=pvl_config.get("distance_weighting", "gaussian"),
            gaussian_sigma_km=pvl_config.get("gaussian_sigma_km", 2.0),
            patch_size=pvl_config.get("patch_size", 0),
        )

        # Parse layer_vs_layer section if present
        layer_vs_layer = None
        if "layer_vs_layer" in coll:
            lvl_config = coll["layer_vs_layer"]
            layer_vs_layer = LayerVsLayerCollocation(
                layer_type_specs=lvl_config.get("layer_type_specs", {}),
            )

        config = RecipeConfig(
            name=data["name"],
            variable=data["variable"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            variable_specs=data.get("variable_specs", {}),
            geographic_bounds=GeographicBounds(
                min_lon=geo.get("min_lon", -20.0),
                max_lon=geo.get("max_lon",   0.0),
                min_lat=geo.get("min_lat",  35.0),
                max_lat=geo.get("max_lat",  60.0),
            ),
            temporal_bounds=TemporalBounds(
                start=temporal.get("start", "2026-01-01"),
                end=temporal.get("end",   "2026-01-02"),
            ),
            sar_data=SARDataSpec(
                satellite=sar.get("satellite", "Sentinel-1"),
                product_level=sar.get("product_level", "L2_OCN"),
                swath_mode=sar.get("swath_mode", sar.get("swath_modes", ["IW", "EW"])),
                max_downloads=sar.get("max_downloads"),
            ),
            validation_sources=[
                ValidationDataSource(
                    source_type=src["source_type"],
                    min_depth=src.get("min_depth", -20.0),
                    max_depth=src.get("max_depth",  20.0),
                    download_kwargs=src.get("download_kwargs", {}),
                    collocation_kwargs=src.get("collocation_kwargs", {}),
                )
                for src in data.get("validation_sources", [])
            ],
            collocation=CollocationType(
                point_vs_layer=point_vs_layer,
                layer_vs_layer=layer_vs_layer,
            ),
            output_dir=data.get("output_dir"),
        )
        return cls(config)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
