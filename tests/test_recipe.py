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
    def test_depth_defaults_to_none(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.min_depth is None
        assert src.max_depth is None

    def test_resolved_depth_falls_back_to_defaults(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.resolved_min_depth == -20.0
        assert src.resolved_max_depth == 20.0

    def test_resolved_depth_uses_explicit_value(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.resolved_min_depth == -2.0
        assert src.resolved_max_depth == 2.0

    def test_custom_depth(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.min_depth == -2.0

    def test_collocation_kwargs_default_empty(self):
        src = ValidationDataSource(source_type="buoy")
        assert src.collocation_kwargs == {}

    def test_to_dict_omits_unspecified_depth(self):
        src = ValidationDataSource(source_type="scatterometer")
        d = src.to_dict()
        assert "min_depth" not in d
        assert "max_depth" not in d

    def test_to_dict_includes_explicit_depth(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        d = src.to_dict()
        assert d["min_depth"] == -2.0
        assert d["max_depth"] == 2.0

    def test_yaml_roundtrip_omits_depth_when_unspecified(self, tmp_path):
        cfg = RecipeConfig(
            name="Depth Omission Test",
            variable="wind",
            validation_sources=[ValidationDataSource(source_type="scatterometer")],
        )
        recipe = Recipe(cfg)
        out = tmp_path / "recipe.yaml"
        recipe.to_yaml(out)

        raw_text = out.read_text()
        assert "min_depth" not in raw_text
        assert "max_depth" not in raw_text

        loaded = Recipe.from_yaml(out)
        loaded_src = loaded.config.validation_sources[0]
        assert loaded_src.min_depth is None
        assert loaded_src.max_depth is None
        assert loaded_src.resolved_min_depth == -20.0
        assert loaded_src.resolved_max_depth == 20.0


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


# ---------------------------------------------------------------------------
# CLI 'currents' recipe template
# ---------------------------------------------------------------------------

class TestCurrentsTemplate:
    def test_includes_noaa_hf_radar_source(self):
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        source_types = {s.source_type for s in recipe.validation_sources}
        assert "hf_radar_noaa" in source_types

        noaa_src = next(
            s for s in recipe.validation_sources if s.source_type == "hf_radar_noaa"
        )
        assert noaa_src.min_depth == -2.0
        assert noaa_src.max_depth == 2.0
        assert noaa_src.download_kwargs == {"resolution_km": 6}

    def test_includes_hf_radar_grid_layer_spec(self):
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        specs = recipe.collocation.layer_vs_layer.layer_type_specs
        assert "hf_radar_grid" in specs
        assert specs["hf_radar_grid"] == {
            "time_tolerance_minutes": 20,
            "aggregation_window_km": 6.0,
            "distance_weighting": "equal",
        }

    def test_preserves_existing_currents_content(self):
        """The extraction into a builder must not drop any existing sources/specs."""
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=7)
        source_types = [s.source_type for s in recipe.validation_sources]
        assert source_types == [
            "hf_radar", "hf_radar_historical", "hf_radar_noaa",
            "drifter", "ferrybox", "mooring",
        ]

        assert recipe.sar_data.max_downloads == 7

    def test_hf_radar_source_has_no_leftover_depth_kwargs(self):
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        hf_radar_src = next(
            s for s in recipe.validation_sources if s.source_type == "hf_radar"
        )
        # The gridded product has no depth axis; the recipe shouldn't imply one.
        assert hf_radar_src.min_depth is None
        assert hf_radar_src.max_depth is None
