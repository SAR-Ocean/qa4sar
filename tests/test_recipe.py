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
# SARDataSpec.source
# ---------------------------------------------------------------------------

class TestSARDataSpecSource:
    def test_default_source_is_sentinel1_l2_ocn(self):
        spec = SARDataSpec()
        assert spec.source == "sentinel1_l2_ocn"

    def test_to_dict_has_source_not_satellite_or_product_level(self):
        spec = SARDataSpec(source="sentinel1_clms_ssm")
        d = spec.to_dict()
        assert d["source"] == "sentinel1_clms_ssm"
        assert "satellite" not in d
        assert "product_level" not in d

    def test_download_kwargs_default_empty_dict(self):
        spec = SARDataSpec()
        assert spec.download_kwargs == {}

    def test_from_yaml_reads_source_field(self, tmp_path):
        yaml_path = tmp_path / "r.yaml"
        yaml_path.write_text(
            "name: t\nvariable: soil_moisture\n"
            "sar_data:\n  source: sentinel1_clms_ssm\n  max_downloads: 3\n"
        )
        recipe = Recipe.from_yaml(yaml_path)
        assert recipe.config.sar_data.source == "sentinel1_clms_ssm"
        assert recipe.config.sar_data.max_downloads == 3

    def test_from_yaml_missing_source_defaults_to_sentinel1_l2_ocn(self, tmp_path):
        yaml_path = tmp_path / "r.yaml"
        yaml_path.write_text("name: t\nvariable: wind\n")
        recipe = Recipe.from_yaml(yaml_path)
        assert recipe.config.sar_data.source == "sentinel1_l2_ocn"

    def test_unknown_source_raises_value_error(self, tmp_path):
        yaml_path = tmp_path / "r.yaml"
        yaml_path.write_text(
            "name: t\nvariable: wind\nsar_data:\n  source: not_a_real_source\n"
        )
        with pytest.raises(ValueError, match="Unknown SAR source 'not_a_real_source'"):
            Recipe.from_yaml(yaml_path)

    def test_source_not_valid_for_this_variable_raises_value_error(self, tmp_path):
        yaml_path = tmp_path / "r.yaml"
        yaml_path.write_text(
            "name: t\nvariable: wind\nsar_data:\n  source: sentinel1_clms_ssm\n"
        )
        with pytest.raises(ValueError, match="only valid for"):
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
    def test_does_not_include_noaa_hf_radar_source(self):
        """NOAA HF-radar is retired from active recipe use (2026-07-30):
        Copernicus's US-region HF-radar-total product is sourced from the
        same IOOS/HFRNet network NOAA distributes directly, so stacking both
        double-counts the same stations. The downloader itself is kept,
        untouched, for a future revert — it's just no longer in the default
        template."""
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        source_types = {s.source_type for s in recipe.validation_sources}
        assert "hf_radar_noaa" not in source_types

    def test_includes_hf_radar_grid_layer_spec(self):
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        specs = recipe.collocation.layer_vs_layer.layer_type_specs
        assert "hf_radar_grid" in specs
        assert specs["hf_radar_grid"] == {
            "time_tolerance_minutes": 30,
            "aggregation_window_km": 6.0,
            "distance_weighting": "equal",
            "dedup_nearest_in_time": True,
        }
        # The bare "hf_radar" key is dead config: every HF-radar source is
        # tagged data_type="hf_radar_grid" by the one converter that
        # produces this data, so "hf_radar_grid" is the only key collocation
        # ever actually resolves to. Removed rather than left as confusing,
        # unreachable tuning.
        assert "hf_radar" not in specs

    def test_preserves_existing_currents_content(self):
        """The extraction into a builder must not drop any existing sources/specs."""
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=7)
        source_types = [s.source_type for s in recipe.validation_sources]
        assert source_types == [
            "hf_radar", "hf_radar_historical",
            "drifter", "ferrybox", "mooring",
            "adcp_historical", "argo_historical", "drifter_historical", "glider_historical",
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


class TestCurrentsTemplateDelayedInstruments:
    def test_all_four_delayed_instruments_present_by_default(self):
        from sar_validation import cli

        recipe = cli._build_currents_config(limit=None)
        source_types = {s.source_type for s in recipe.validation_sources}
        assert {
            "adcp_historical", "argo_historical", "drifter_historical", "glider_historical",
        }.issubset(source_types)


class TestWindTemplate:
    def test_includes_ftp_scatterometer_sources(self):
        from sar_validation import cli

        recipe = cli._build_wind_config(limit=None)
        source_types = {s.source_type for s in recipe.validation_sources}
        assert {
            "scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3",
        }.issubset(source_types)
        assert "scatterometer" in source_types  # ASCAT/EUMDAC untouched

    def test_ftp_scatterometer_sources_have_25km_layer_specs(self):
        from sar_validation import cli

        recipe = cli._build_wind_config(limit=None)
        specs = recipe.collocation.layer_vs_layer.layer_type_specs
        for key in ("scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3"):
            assert specs[key]["aggregation_window_km"] == 25.0
        assert specs["scatterometer"]["aggregation_window_km"] == 12.5

    def test_preserves_existing_wind_content(self):
        """The extraction into a builder must not drop any existing sources/specs."""
        from sar_validation import cli

        recipe = cli._build_wind_config(limit=3)
        source_types = [s.source_type for s in recipe.validation_sources]
        assert source_types == [
            "mooring", "buoy", "ferrybox", "drifter", "tidal_gauge",
            "scatterometer", "altimeter", "radiometer",
            "scatterometer_hy2b", "scatterometer_hy2c", "scatterometer_oceansat3",
        ]
        assert recipe.sar_data.max_downloads == 3


def test_default_layer_type_specs_has_scatterometer_ssm():
    from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

    # Keyed "scatterometer_ssm" (matching from_ascat_ssm's data_type tag),
    # not "ascat_ssm" (the source_type/platform_type/output-subfolder name)
    # -- there's only one scatterometer_ssm source today, so no
    # _resolve_layer_type refinement branch renames it, unlike radiometer_ssm.
    assert DEFAULT_LAYER_TYPE_SPECS["scatterometer_ssm"] == {
        "time_tolerance_minutes": 720,
        "aggregation_window_km": 12.5,
        "distance_weighting": "equal",
    }


def test_default_layer_type_specs_has_amsr_and_smap_ssm():
    from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

    assert DEFAULT_LAYER_TYPE_SPECS["amsr_ssm"] == {
        "time_tolerance_minutes": 720, "aggregation_window_km": 25.0, "distance_weighting": "equal",
    }
    assert DEFAULT_LAYER_TYPE_SPECS["smap_ssm"] == {
        "time_tolerance_minutes": 720, "aggregation_window_km": 9.0, "distance_weighting": "equal",
    }


def test_default_layer_type_specs_has_smos_ssm():
    from sar_validation.core.recipe import DEFAULT_LAYER_TYPE_SPECS

    assert DEFAULT_LAYER_TYPE_SPECS["smos_ssm"] == {
        "time_tolerance_minutes": 720, "aggregation_window_km": 35.0, "distance_weighting": "equal",
    }
