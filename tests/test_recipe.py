"""Tests for Recipe loading, saving, and defaults."""

from __future__ import annotations

import pytest

from sar_validation.core.recipe import (
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

# ---------------------------------------------------------------------------
# RecipeConfig defaults
# ---------------------------------------------------------------------------

class TestRecipeConfigDefaults:
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
# SARDataSpec.source
# ---------------------------------------------------------------------------

class TestSARDataSpecSource:
    def test_to_dict_has_source_not_satellite_or_product_level(self):
        spec = SARDataSpec(source="sentinel1_clms_ssm")
        d = spec.to_dict()
        assert d["source"] == "sentinel1_clms_ssm"
        assert "satellite" not in d
        assert "product_level" not in d

    @pytest.mark.parametrize(
        "yaml_body,expected_source,expected_max_downloads",
        [
            pytest.param(
                "name: t\nvariable: soil_moisture\n"
                "sar_data:\n  source: sentinel1_clms_ssm\n  max_downloads: 3\n",
                "sentinel1_clms_ssm",
                3,
                id="reads_source_field",
            ),
            pytest.param(
                "name: t\nvariable: wind\n",
                "sentinel1_l2_ocn",
                None,
                id="missing_source_defaults_to_sentinel1_l2_ocn",
            ),
        ],
    )
    def test_from_yaml_source_field(
        self, tmp_path, yaml_body, expected_source, expected_max_downloads
    ):
        yaml_path = tmp_path / "r.yaml"
        yaml_path.write_text(yaml_body)
        recipe = Recipe.from_yaml(yaml_path)
        assert recipe.config.sar_data.source == expected_source
        if expected_max_downloads is not None:
            assert recipe.config.sar_data.max_downloads == expected_max_downloads

    @pytest.mark.parametrize(
        "yaml_body,expected_match",
        [
            pytest.param(
                "name: t\nvariable: wind\nsar_data:\n  source: not_a_real_source\n",
                "Unknown SAR source 'not_a_real_source'",
                id="unknown_source_raises_value_error",
            ),
            pytest.param(
                "name: t\nvariable: wind\nsar_data:\n  source: sentinel1_clms_ssm\n",
                "only valid for",
                id="source_not_valid_for_this_variable_raises_value_error",
            ),
        ],
    )
    def test_from_yaml_invalid_source_raises(self, tmp_path, yaml_body, expected_match):
        yaml_path = tmp_path / "r.yaml"
        yaml_path.write_text(yaml_body)
        with pytest.raises(ValueError, match=expected_match):
            Recipe.from_yaml(yaml_path)


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
    def test_resolved_depth_falls_back_to_defaults(self):
        src = ValidationDataSource(source_type="mooring")
        assert src.resolved_min_depth == -20.0
        assert src.resolved_max_depth == 20.0

    def test_resolved_depth_uses_explicit_value(self):
        src = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)
        assert src.resolved_min_depth == -2.0
        assert src.resolved_max_depth == 2.0

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

class TestCurrentsRecipeYamlFilesHfRadarGridSpec:
    """Every packaged currents_*.yaml recipe must carry the same
    hf_radar_grid tolerance/dedup fix as the cli.py template it was
    generated from -- a recipe's own layer_type_specs override wins over
    the Python-side default, so fixing only the default wouldn't have
    fixed the Finnmark bug for any already-written recipe file."""

    def test_all_currents_recipes_have_updated_hf_radar_grid_spec(self):
        import pathlib

        recipes_dir = pathlib.Path(__file__).resolve().parent.parent / "recipes"
        paths = sorted(recipes_dir.glob("currents_*.yaml"))
        # Not an exact/minimum count -- how many currents recipes exist is
        # up to the toolbox user. Just guard against the glob itself being
        # broken (e.g. a typo'd pattern) and silently testing nothing.
        assert len(paths) > 0

        for path in paths:
            recipe = Recipe.from_yaml(path)
            specs = recipe.config.collocation.layer_vs_layer.layer_type_specs
            spec = specs["hf_radar_grid"]
            assert spec["time_tolerance_minutes"] == 30, path
            assert spec["dedup_nearest_in_time"] is True, path
            # Bare "hf_radar" is dead config (data_type is always
            # "hf_radar_grid" for every HF-radar source) -- removed from
            # every packaged recipe rather than left as confusing,
            # unreachable tuning.
            assert "hf_radar" not in specs, path


class TestWindTemplate:
    def test_ftp_scatterometer_sources_have_25km_layer_specs(self):
        from sar_validation import cli

        recipe = cli._build_wind_config(limit=None)
        specs = recipe.collocation.layer_vs_layer.layer_type_specs
        for key in ("scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3"):
            assert specs[key]["aggregation_window_km"] == 25.0
        assert specs["scatterometer"]["aggregation_window_km"] == 12.5


