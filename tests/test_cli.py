"""Tests for sar_validation.cli."""

from __future__ import annotations

import xarray as xr

from sar_validation import cli
from sar_validation.core.recipe import Recipe, RecipeConfig, SARDataSpec


def _waves_recipe(swath_mode):
    return Recipe(RecipeConfig(
        name="waves_test",
        variable="waves",
        sar_data=SARDataSpec(swath_mode=swath_mode),
    ))


class TestLoadPrecomputedStats:
    def test_finds_files_saved_under_filter_variable_pairs_keys(self, tmp_path):
        """Regression test: run_statistics saves files keyed by
        filter_variable_pairs (dataset-aware), so _load_precomputed_stats
        must look them up the same way — not via the static
        infer_variable_pairs list, which used the wrong key
        (oswTotalHs_vs_VHM0) and silently found nothing for a mixed-mode
        WV/SM recipe where only sar_oswTotalHs/val_VAVH exist."""
        recipe = _waves_recipe(["WV", "SM"])
        collocation_ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })

        stats_ds = xr.Dataset({"bias": ("source", [0.02])}, coords={"source": ["altimeter"]})
        stats_ds.to_netcdf(tmp_path / "validation_statistics_oswTotalHs_vs_VAVH.nc")

        result = cli._load_precomputed_stats(recipe, collocation_ds, tmp_path)

        assert set(result.keys()) == {"oswTotalHs_vs_VAVH"}
        assert float(result["oswTotalHs_vs_VAVH"]["bias"].values[0]) == 0.02

    def test_missing_stats_file_is_skipped(self, tmp_path):
        """A pair that filter_variable_pairs selects but has no saved .nc
        file (e.g. run_statistics found no valid pairs for it) is simply
        left out of the map, not an error."""
        recipe = _waves_recipe(["WV", "SM"])
        collocation_ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })

        result = cli._load_precomputed_stats(recipe, collocation_ds, tmp_path)

        assert result == {}

    def test_respects_filename_suffix(self, tmp_path):
        recipe = _waves_recipe(["WV", "SM"])
        collocation_ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })

        stats_ds = xr.Dataset({"bias": ("source", [0.02])}, coords={"source": ["altimeter"]})
        stats_ds.to_netcdf(tmp_path / "validation_statistics_oswTotalHs_vs_VAVH_individual.nc")

        result = cli._load_precomputed_stats(recipe, collocation_ds, tmp_path, filename_suffix="_individual")

        assert set(result.keys()) == {"oswTotalHs_vs_VAVH"}


class TestComputeStatsWritesNativeUnitsForSoilMoisture:
    def test_compute_stats_writes_native_units_for_soil_moisture(self, tmp_path, capsys):
        import numpy as np

        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds

        cfg = RecipeConfig(
            name="test_native_units_cli", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        recipe = Recipe(config=cfg)

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 30.0, 40.0, 50.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 35.0, 0.15, 0.20])),
                "val_source": ("collocation", np.array(["ascat_ssm", "ascat_ssm", "ismn", "ismn"])),
            },
        )
        collocation_ds.to_netcdf(tmp_path / "collocation_results.nc")

        cli._compute_stats(recipe, tmp_path)

        assert (tmp_path / "validation_statistics_sarSSM_vs_SOIL_MOISTURE.nc").exists()
        assert (tmp_path / "validation_statistics_sarSSM_vs_SOIL_MOISTURE_native_units.nc").exists()
        out = capsys.readouterr().out
        assert "Native-units statistics saved" in out


class TestExecuteRecipeContinuesPastDownloadFailure:
    def test_does_not_exit_when_download_all_returns_false(self, tmp_path, capsys):
        from unittest.mock import patch

        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-partial-failure",
            variable="wind",
            output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        with patch(
            "sar_validation.core.orchestrator.DataOrchestrator.download_all",
            return_value=False,
        ):
            # Must not raise SystemExit.
            cli._execute_recipe(str(recipe_path), force_download=True)

        out = capsys.readouterr().out
        assert "continuing with available data" in out


class TestExecuteRecipePrintsFinalWarningsSummary:
    def test_prints_notices_and_errors_at_the_end_of_the_run(self, tmp_path, capsys):
        """A notice fired early (e.g. during Step 1's download phase) must
        still be visible at the very end of the run, after Steps 2/3/5a/5b
        have printed a lot more output -- not just as a log line that
        scrolls past mid-run."""
        import json

        from sar_validation.core.recipe import Recipe, RecipeConfig

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "notices": ["No delayed-mode in-situ current data found (adcp, argo) for this window."],
        }))

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-final-summary", variable="wind", output_dir=str(run_dir),
        )).to_yaml(recipe_path)

        cli._execute_recipe(str(recipe_path))

        out = capsys.readouterr().out
        assert "Warnings from this run:" in out
        assert "No delayed-mode in-situ current data found (adcp, argo) for this window." in out

    def test_no_warnings_prints_no_summary_block(self, tmp_path, capsys):
        import json

        from sar_validation.core.recipe import Recipe, RecipeConfig

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "download_metadata.json").write_text(json.dumps({"errors": [], "notices": []}))

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-no-warnings", variable="wind", output_dir=str(run_dir),
        )).to_yaml(recipe_path)

        cli._execute_recipe(str(recipe_path))

        out = capsys.readouterr().out
        assert "Warnings from this run:" not in out


class TestLoadDownloadWarnings:
    def test_no_metadata_file_returns_none(self, tmp_path):
        assert cli._load_download_warnings(tmp_path) is None

    def test_empty_errors_returns_none(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(json.dumps({"errors": []}))
        assert cli._load_download_warnings(tmp_path) is None

    def test_returns_errors_list(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(
            json.dumps({"errors": ["altimeter download failed: timeout"]})
        )
        assert cli._load_download_warnings(tmp_path) == ["altimeter download failed: timeout"]

    def test_returns_notices_list_alongside_errors(self, tmp_path):
        """notices (e.g. "no delayed-mode currents data found") must also
        surface on the PDF cover page, same as errors -- a notice isn't a
        failure, but the user still needs to see it without scrolling back
        through the whole run's console output."""
        import json

        (tmp_path / "download_metadata.json").write_text(
            json.dumps({
                "errors": ["altimeter download failed: timeout"],
                "notices": ["No delayed-mode in-situ current data found (adcp, argo) for this window."],
            })
        )
        assert cli._load_download_warnings(tmp_path) == [
            "altimeter download failed: timeout",
            "No delayed-mode in-situ current data found (adcp, argo) for this window.",
        ]

    def test_notices_only_still_returns_a_list(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(
            json.dumps({
                "errors": [],
                "notices": ["No delayed-mode in-situ current data found (adcp, argo) for this window."],
            })
        )
        assert cli._load_download_warnings(tmp_path) == [
            "No delayed-mode in-situ current data found (adcp, argo) for this window.",
        ]

    def test_missing_notices_key_is_backward_compatible(self, tmp_path):
        import json

        (tmp_path / "download_metadata.json").write_text(
            json.dumps({"errors": ["altimeter download failed: timeout"]})
        )
        assert cli._load_download_warnings(tmp_path) == ["altimeter download failed: timeout"]


class TestMethodRunsSuffixMapping:
    def _write_recipe_with_skippable_download(self, tmp_path):
        import json

        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "download_metadata.json").write_text(json.dumps({"errors": []}))
        return recipe_path

    def test_individual_alone_maps_to_individual_suffix(self, tmp_path):
        from unittest.mock import patch

        recipe_path = self._write_recipe_with_skippable_download(tmp_path)

        with patch("sar_validation.cli._collocate_data") as mock_collocate:
            cli._execute_recipe(
                str(recipe_path), collocate=True,
                layer_vs_layer_collocation_method="individual",
            )

        _, kwargs = mock_collocate.call_args
        assert kwargs["filename_suffix"] == "_individual"
        assert kwargs["layer_vs_layer_collocation_method"] == "individual"

    def test_cell_averaging_alone_still_maps_to_empty_suffix(self, tmp_path):
        from unittest.mock import patch

        recipe_path = self._write_recipe_with_skippable_download(tmp_path)

        with patch("sar_validation.cli._collocate_data") as mock_collocate:
            cli._execute_recipe(
                str(recipe_path), collocate=True,
                layer_vs_layer_collocation_method="cell-averaging",
            )

        _, kwargs = mock_collocate.call_args
        assert kwargs["filename_suffix"] == ""


class TestExecuteRecipePassesForceDownloadToOrchestrator:
    def test_force_download_flag_reaches_orchestrator_constructor(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        with patch("sar_validation.core.orchestrator.DataOrchestrator") as mock_cls:
            mock_cls.return_value.download_all.return_value = True
            mock_cls.return_value.base_dir = tmp_path / "run"
            cli._execute_recipe(str(recipe_path), force_download=True)

        assert mock_cls.call_args.kwargs["force_download"] is True


class TestExecuteRecipeSkipsStatsWhenAlreadyComputed:
    """Step 4 (compute statistics) should be resumable just like steps 1-3:
    if validation_statistics_*.nc files for every pair already exist on
    disk, --stats/--plot must not recompute them from scratch."""

    def _write_recipe_with_collocation(self, tmp_path):
        import json

        from sar_validation.core.recipe import Recipe, RecipeConfig

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "download_metadata.json").write_text(json.dumps({"errors": []}))

        collocation_ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [7.0, 8.0]),
            "val_WSPD":         ("collocation", [7.2, 7.9]),
        })
        collocation_ds.to_netcdf(run_dir / "collocation_results.nc")

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-stats-resume", variable="wind", output_dir=str(run_dir),
        )).to_yaml(recipe_path)
        return recipe_path, run_dir

    def test_does_not_recompute_stats_when_files_already_exist(self, tmp_path, capsys):
        from unittest.mock import patch

        recipe_path, run_dir = self._write_recipe_with_collocation(tmp_path)

        stats_ds = xr.Dataset({"bias": ("source", [0.1])}, coords={"source": ["scatterometer"]})
        stats_ds.to_netcdf(run_dir / "validation_statistics_owiWindSpeed_vs_WSPD.nc")

        with patch("sar_validation.cli._compute_stats") as mock_compute_stats:
            cli._execute_recipe(str(recipe_path), stats=True)

        mock_compute_stats.assert_not_called()
        out = capsys.readouterr().out
        assert "Step 4 skipped" in out

    def test_still_computes_stats_when_files_missing(self, tmp_path):
        from unittest.mock import patch

        recipe_path, run_dir = self._write_recipe_with_collocation(tmp_path)
        # No pre-existing validation_statistics_*.nc file this time.

        with patch("sar_validation.cli._compute_stats") as mock_compute_stats:
            cli._execute_recipe(str(recipe_path), stats=True)

        mock_compute_stats.assert_called_once()


class TestBuildSoilMoistureConfig:
    def test_recipe_shape(self):
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config()

        assert cfg.variable == "soil_moisture"
        assert cfg.sar_data.satellite == "Sentinel-1"
        assert cfg.sar_data.product_level == "L3_SSM"
        assert len(cfg.validation_sources) == 5
        source_types = [s.source_type for s in cfg.validation_sources]
        assert source_types == ["ismn", "ascat_ssm", "amsr_ssm", "smap_ssm", "smos_ssm"]
        ismn_source = cfg.validation_sources[0]
        assert ismn_source.min_depth == 0.0
        assert ismn_source.max_depth == 0.05
        assert ismn_source.download_kwargs == {}
        for satellite_source in cfg.validation_sources[1:]:
            assert satellite_source.download_kwargs == {}

    def test_default_geographic_bounds(self):
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config()
        bounds = cfg.geographic_bounds
        assert (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat) == (
            -10.0, 30.0, 35.0, 60.0,
        )

    def test_collocation_defaults_are_pixel_scale(self):
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config()
        pvl = cfg.collocation.point_vs_layer
        assert pvl.spatial_tolerance_km == 2.0
        assert pvl.aggregation_window_km == 1.0
        assert pvl.distance_weighting == "equal"
        assert pvl.interpolation_method == "nearest"
        assert pvl.time_tolerance_minutes == 720

    def test_limit_forwarded_to_max_downloads(self):
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config(limit=5)
        assert cfg.sar_data.max_downloads == 5

    def test_registered_under_create_recipe(self, tmp_path, monkeypatch):
        from sar_validation.cli import _create_recipe

        monkeypatch.chdir(tmp_path)
        _create_recipe("soil_moisture")

        recipe_path = tmp_path / "recipes" / "soil_moisture_validation.yaml"
        assert recipe_path.exists()
