"""
Recipe dataclasses for the SAR validation toolbox.

A recipe (YAML or JSON) is the single configuration object that drives the
entire pipeline:

    step 1 — download    → DataOrchestrator reads the recipe
    step 3 — collocation → uses collocation parameters from the recipe
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic building-blocks
# ---------------------------------------------------------------------------

@dataclass
class GeographicBounds:
    """Bounding box in decimal degrees (WGS-84).

    ``min_lon > max_lon`` means the box wraps through the antimeridian
    (180 degrees) rather than being invalid — e.g. ``min_lon=135,
    max_lon=-120`` covers the Pacific from 135E to 120W. Downloaders split
    such a bbox into two non-crossing windows internally via
    ``sar_validation.downloaders.base.split_antimeridian_bbox``.
    """
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


#: Fallback depth window (metres; negative = below sea surface) applied when
#: a recipe's validation source doesn't specify min_depth/max_depth.
DEFAULT_MIN_DEPTH = -20.0
DEFAULT_MAX_DEPTH = 20.0


@dataclass
class ValidationDataSource:
    """One validation data source referenced in a recipe."""

    source_type: str
    """
    Platform / product type.

    Accepted values:
      in-situ   : mooring, buoy, ferrybox, drifter, tidal_gauge
      satellite : scatterometer, altimeter, radiometer
      coastal   : hf_radar
    """

    # Optional depth filter (only meaningful for in-situ and HF radar
    # sources). None means "use DEFAULT_MIN_DEPTH/DEFAULT_MAX_DEPTH" — see
    # resolved_min_depth/resolved_max_depth below. Left as None for source
    # types that don't use depth (e.g. scatterometer) so recipes don't
    # serialize a meaningless depth window for them.
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None

    # Extra keyword arguments forwarded to the downloader
    download_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Override per-source collocation tolerances
    collocation_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_min_depth(self) -> float:
        return self.min_depth if self.min_depth is not None else DEFAULT_MIN_DEPTH

    @property
    def resolved_max_depth(self) -> float:
        return self.max_depth if self.max_depth is not None else DEFAULT_MAX_DEPTH

    def to_dict(self) -> Dict[str, Any]:
        # Hand-built (not asdict()) so unset min_depth/max_depth are omitted
        # rather than serialized as null — keep in sync with the fields above.
        d: Dict[str, Any] = {"source_type": self.source_type}
        if self.min_depth is not None:
            d["min_depth"] = self.min_depth
        if self.max_depth is not None:
            d["max_depth"] = self.max_depth
        d["download_kwargs"] = self.download_kwargs
        d["collocation_kwargs"] = self.collocation_kwargs
        return d


@dataclass
class SARDataSpec:
    """Specification for the SAR product to validate."""
    satellite: str = "Sentinel-1"
    product_level: str = "L2_OCN"
    """
    ``"L2_OCN"`` (wind/currents/waves, via ``SARDownloader``) or
    ``"L3_SSM"`` (soil moisture, via ``SoilMoistureDownloader``).
    """
    swath_mode: List[str] = field(default_factory=lambda: ["IW", "EW"])
    """SAR beam mode(s), e.g. ``["IW", "EW"]`` or ``["WV"]``. Unused for
    ``product_level="L3_SSM"`` (no WV/IW/EW mode concept for a daily
    merged raster)."""
    max_downloads: Optional[int] = None   # None → download all found products

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PointVsLayerCollocation:
    """Collocation config for point vs gridded SAR data (buoys, moorings)."""
    time_tolerance_minutes: int = 30
    spatial_tolerance_km: float = 25
    interpolation_method: str = "nearest"
    aggregation_window_km: float = 5.0
    validation_temporal_averaging_minutes: int = 30  # unused for point_vs_layer; kept for schema/API compatibility
    distance_weighting: str = "gaussian"
    gaussian_sigma_km: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Built-in fallback collocation parameters per layer-type, applied
#: regardless of whether a recipe declares a ``layer_vs_layer`` section at
#: all. A recipe's own ``layer_type_specs`` entries (if any) override these
#: per-key. Altimeter is split into ``altimeter_1hz``/``altimeter_5hz``
#: because the two frequencies have very different along-track point
#: spacing (7km vs 1.4km — see WAVE_GLO_PHY_SWH_L3_NRT_014_001), so a single
#: aggregation window would be wrong for at least one of them.
#: RSS radiometers are all distributed on a common 0.25° (~25 km) grid, so
#: every ``radiometer_<sensor>`` entry defaults to the same aggregation window.
#: They are kept as separate per-sensor keys (mirroring the altimeter split)
#: purely so each can be tuned individually in a recipe — e.g. a different time
#: tolerance, or down-weighting a coarser-footprint sensor. Collocation refines
#: ``radiometer`` to ``radiometer_<sensor>`` from each node's ``sensor`` attr.
_RADIOMETER_DEFAULT = {"time_tolerance_minutes": 180, "aggregation_window_km": 25.0, "distance_weighting": "equal"}

DEFAULT_LAYER_TYPE_SPECS: Dict[str, Dict[str, Any]] = {
    "scatterometer":  {"time_tolerance_minutes": 180, "aggregation_window_km": 12.5, "distance_weighting": "equal"},
    # HY-2B/HY-2C/Oceansat-3 (KNMI OSI-SAF FTP, recent-only) are 25 km
    # products, distinct from ASCAT's 12.5 km coastal-wind grid above.
    "scatterometer_hy2b": {"time_tolerance_minutes": 180, "aggregation_window_km": 25.0, "distance_weighting": "equal"},
    "scatterometer_hy2c": {"time_tolerance_minutes": 180, "aggregation_window_km": 25.0, "distance_weighting": "equal"},
    "scatterometer_oceansat3": {
        "time_tolerance_minutes": 180, "aggregation_window_km": 25.0, "distance_weighting": "equal"
    },
    "altimeter_1hz":  {"time_tolerance_minutes": 180, "aggregation_window_km": 7.0,  "distance_weighting": "equal"},
    "altimeter_5hz":  {"time_tolerance_minutes": 180, "aggregation_window_km": 1.4,  "distance_weighting": "equal"},
    "hf_radar_grid":  {
        "time_tolerance_minutes": 30, "aggregation_window_km": 6.0,
        "distance_weighting": "equal", "dedup_nearest_in_time": True,
    },
    # Bare key is the fallback when a node's sensor is unknown.
    "radiometer":       dict(_RADIOMETER_DEFAULT),
    "radiometer_amsr2": dict(_RADIOMETER_DEFAULT),
    "radiometer_gmi":   dict(_RADIOMETER_DEFAULT),
    "radiometer_ssmis_f16": dict(_RADIOMETER_DEFAULT),
    "radiometer_ssmis_f17": dict(_RADIOMETER_DEFAULT),
    "radiometer_ssmis_f18": dict(_RADIOMETER_DEFAULT),
    "radiometer_windsat":   dict(_RADIOMETER_DEFAULT),
    "radiometer_amsre":     dict(_RADIOMETER_DEFAULT),
}


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
            time_tolerance_minutes: 180
            aggregation_window_km: 5
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

    #: Search radius (km) around each sparse SAR WV-mode OSW imagette point
    #: used to gather validation observations. A WV imagette covers ~20×20 km,
    #: so 14 km ≈ its center-to-corner distance (fully covers the footprint).
    #: Only affects WV/point-mode SAR; IW/EW grid collocation is unaffected.
    sar_footprint_radius_km: float = 14.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "point_vs_layer": self.point_vs_layer.to_dict(),
            "sar_footprint_radius_km": self.sar_footprint_radius_km,
        }
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
    variable: str       # "wind" | "currents" | "waves" | "soil_moisture"

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

        unknown_coll_keys = set(coll) - {"point_vs_layer", "layer_vs_layer", "sar_footprint_radius_km"}
        if unknown_coll_keys:
            logger.warning(
                "Recipe '%s': collocation block has unrecognized key(s) %s — "
                "expected a nested 'point_vs_layer:'/'layer_vs_layer:' schema. "
                "These keys are ignored; defaults apply instead.",
                data.get("name", "<unnamed>"), sorted(unknown_coll_keys),
            )

        # Parse point_vs_layer section
        pvl_config = coll.get("point_vs_layer", {})
        point_vs_layer = PointVsLayerCollocation(
            time_tolerance_minutes=pvl_config.get("time_tolerance_minutes", 30),
            spatial_tolerance_km=pvl_config.get("spatial_tolerance_km", 25),
            interpolation_method=pvl_config.get("interpolation_method", "nearest"),
            aggregation_window_km=pvl_config.get("aggregation_window_km", 5.0),
            validation_temporal_averaging_minutes=pvl_config.get("validation_temporal_averaging_minutes", 30),
            distance_weighting=pvl_config.get("distance_weighting", "gaussian"),
            gaussian_sigma_km=pvl_config.get("gaussian_sigma_km", 2.0),
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
                    min_depth=src.get("min_depth"),
                    max_depth=src.get("max_depth"),
                    download_kwargs=src.get("download_kwargs", {}),
                    collocation_kwargs=src.get("collocation_kwargs", {}),
                )
                for src in data.get("validation_sources", [])
            ],
            collocation=CollocationType(
                point_vs_layer=point_vs_layer,
                layer_vs_layer=layer_vs_layer,
                sar_footprint_radius_km=coll.get("sar_footprint_radius_km", 14.0),
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
