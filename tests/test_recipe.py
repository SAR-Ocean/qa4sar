"""Tests for Recipe loading, saving, and defaults."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from sar_validation.core.recipe import (
    Recipe,
    RecipeConfig,
    GeographicBounds,
    TemporalBounds,
    SARDataSpec,
    ValidationDataSource,
    CollocationType,
    PointVsLayerCollocation,
    LayerVsLayerCollocation,
)


# ---------------------------------------------------------------------------
# RecipeConfig defaults
# ---------------------------------------------------------------------------

class TestRecipeConfigDefaults:
    def test_required_fields(self):
        cfg = RecipeConfig(name="test", variable="wind")
        assert cfg.name == "test"
        assert cfg.variable == "wind"

    def test_defaults(self):
        cfg = RecipeConfig(name="x", variable="wind")
        assert cfg.description == ""
        assert cfg.version == "1.0"
        assert cfg.variable_specs == {}
        assert isinstance(cfg.geographic_bounds, GeographicBounds)
        assert isinstance(cfg.temporal_bounds, TemporalBounds)
        assert isinstance(cfg.sar_data, SARDataSpec)
        assert cfg.validation_sources == []
        assert cfg.output_dir is None

    def test_to_dict_roundtrip(self):
        cfg = RecipeConfig(
            name="Wind Test",
            variable="wind",
            variable_specs={"components": ["speed", "direction"]},
        )
        d = cfg.to_dict()
        assert d["name"] == "Wind Test"
        assert d["variable"] == "wind"
        assert d["variable_specs"]["components"] == ["speed", "direction"]


# ---------------------------------------------------------------------------
# YAML roundtrip
# ---------------------------------------------------------------------------

class TestYAMLRoundtrip:
    def test_save_and_load(self, tmp_path):
        cfg = RecipeConfig(
            name="Wave Test",
            variable="waves",
            geographic_bounds=GeographicBounds(-10, 5, 40, 60),
            temporal_bounds=TemporalBounds("2026-03-01", "2026-03-02"),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="altimeter"),
            ],
        )
        recipe = Recipe(cfg)
        out = tmp_path / "recipe.yaml"
        recipe.to_yaml(out)

        loaded = Recipe.from_yaml(out)
        assert loaded.config.name == "Wave Test"
        assert loaded.config.variable == "waves"
        assert loaded.config.geographic_bounds.min_lon == -10
        assert len(loaded.config.validation_sources) == 2
        assert loaded.config.validation_sources[0].source_type == "mooring"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Recipe.from_yaml(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------

class TestJSONRoundtrip:
    def test_save_and_load(self, tmp_path):
        cfg = RecipeConfig(
            name="Currents Test",
            variable="currents",
            sar_data=SARDataSpec(swath_mode=["WV"], max_downloads=5),
            collocation=CollocationType(
                point_vs_layer=PointVsLayerCollocation(
                    time_tolerance_minutes=120,
                    spatial_tolerance_km=25.0,
                ),
                layer_vs_layer=LayerVsLayerCollocation(
                    layer_type_specs={"altimeter": {"time_tolerance_minutes": 90}},
                ),
                sar_footprint_radius_km=18.0,
            ),
        )
        recipe = Recipe(cfg)
        out = tmp_path / "recipe.json"
        recipe.to_json(out)

        loaded = Recipe.from_json(out)
        assert loaded.config.name == "Currents Test"
        assert loaded.config.sar_data.swath_mode == ["WV"]
        assert loaded.config.sar_data.max_downloads == 5
        assert loaded.config.collocation.point_vs_layer.time_tolerance_minutes == 120
        assert loaded.config.collocation.point_vs_layer.spatial_tolerance_km == 25.0
        assert loaded.config.collocation.sar_footprint_radius_km == 18.0
        assert (
            loaded.config.collocation.layer_vs_layer.layer_type_specs["altimeter"][
                "time_tolerance_minutes"
            ]
            == 90
        )


# ---------------------------------------------------------------------------
# Validation source parsing
# ---------------------------------------------------------------------------

class TestValidationSourceParsing:
    def test_depth_defaults(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.min_depth == -20.0
        assert src.max_depth == 20.0

    def test_custom_depth(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.min_depth == -2.0

    def test_collocation_kwargs_default_empty(self):
        src = ValidationDataSource(source_type="buoy")
        assert src.collocation_kwargs == {}


# ---------------------------------------------------------------------------
# from_dict with edge-cases
# ---------------------------------------------------------------------------

class TestFromDict:
    def _minimal_dict(self):
        return {"name": "My Recipe", "variable": "wind"}

    def test_minimal_dict(self):
        recipe = Recipe._from_dict(self._minimal_dict())
        assert recipe.config.name == "My Recipe"

    def test_swath_mode_alias(self):
        """Legacy YAML key 'swath_modes' should still be parsed."""
        d = {
            "name": "x", "variable": "wind",
            "sar_data": {"swath_modes": ["IW"]},
        }
        recipe = Recipe._from_dict(d)
        assert recipe.config.sar_data.swath_mode == ["IW"]
