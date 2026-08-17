"""Tests for sar_validation.cli."""

from __future__ import annotations

import pytest
import xarray as xr

from sar_validation import cli
from sar_validation.core.recipe import (
    Recipe,
    RecipeConfig,
    SARDataSpec,
    ValidationDataSource,
)


def _waves_recipe(swath_mode):
    return Recipe(RecipeConfig(
        name="waves_test",
        variable="waves",
        sar_data=SARDataSpec(swath_mode=swath_mode),
    ))


class TestBuildCurrentsConfig:
    def test_default_bbox_uses_copernicus_pair(self):
        from sar_validation.cli import _build_currents_config

        cfg = _build_currents_config()
        source_types = [s.source_type for s in cfg.validation_sources]
        assert "hf_radar" in source_types
        assert "hf_radar_historical" in source_types
        assert "hf_radar_us" not in source_types

    def test_no_hfradar_resolution_warning_for_non_us_bbox(self, caplog):
        import logging

        from sar_validation.cli import _build_currents_config

        with caplog.at_level(logging.WARNING):
            _build_currents_config()
        assert not any("has no effect" in r.message for r in caplog.records)

    def test_us_west_bbox_selects_hf_radar_us(self):
        from sar_validation.cli import _build_currents_config

        cfg = _build_currents_config(min_lon=-130.0, max_lon=-117.0, min_lat=32.0, max_lat=42.0)
        source_types = [s.source_type for s in cfg.validation_sources]
        assert source_types.count("hf_radar_us") == 1
        assert "hf_radar" not in source_types
        assert "hf_radar_historical" not in source_types
        hf_source = next(s for s in cfg.validation_sources if s.source_type == "hf_radar_us")
        assert hf_source.download_kwargs == {}

    @pytest.mark.parametrize(
        "bbox,resolution,expected_download_kwargs",
        [
            pytest.param(
                dict(min_lon=-130.0, max_lon=-117.0, min_lat=32.0, max_lat=42.0),
                "1km",
                {"resolution_km": 1.0},
                id="us_west_1km",
            ),
            pytest.param(
                dict(min_lon=-130.0, max_lon=-117.0, min_lat=32.0, max_lat=42.0),
                "finest",
                {"resolution_km": "finest"},
                id="us_west_finest_passes_sentinel_through",
            ),
            pytest.param(
                dict(min_lon=-68.0, max_lon=-64.0, min_lat=16.0, max_lat=20.0),
                "2km",
                {"resolution_km": 2.0},
                # US_PRVI's combined ERDDAP union THREDDS resolutions are
                # {2, 6} -- 2km succeeds here, unlike the 1km case below
                # which is kept separate because it raises instead.
                id="prvi_2km",
            ),
        ],
    )
    def test_hfradar_resolution_sets_download_kwargs(
        self, bbox, resolution, expected_download_kwargs
    ):
        from sar_validation.cli import _build_currents_config

        cfg = _build_currents_config(hfradar_resolution=resolution, **bbox)
        hf_source = next(s for s in cfg.validation_sources if s.source_type == "hf_radar_us")
        assert hf_source.download_kwargs == expected_download_kwargs

    def test_prvi_only_bbox_with_1km_raises_value_error(self):
        # US_PRVI's combined ERDDAP union THREDDS resolutions are {2, 6} --
        # 1km is available on neither backend for this region, unlike
        # Hawaii (see next test) where THREDDS covers what ERDDAP lacks.
        from sar_validation.cli import _build_currents_config

        with pytest.raises(ValueError, match="1km"):
            _build_currents_config(
                min_lon=-68.0, max_lon=-64.0, min_lat=16.0, max_lat=20.0,
                hfradar_resolution="1km",
            )

    def test_hawaii_only_bbox_with_6km_succeeds_via_thredds_even_though_erddap_lacks_it(self):
        # ERDDAP only ever publishes 1km for Hawaii, but THREDDS has
        # 1/2/6km -- the early creation-time check must consider both
        # backends' union, not just ERDDAP's, or this would wrongly reject
        # a request the waterfall could actually satisfy at run time.
        from sar_validation.cli import _build_currents_config

        cfg = _build_currents_config(
            min_lon=-159.0, max_lon=-154.0, min_lat=19.0, max_lat=22.0,
            hfradar_resolution="6km",
        )
        hf_source = next(s for s in cfg.validation_sources if s.source_type == "hf_radar_us")
        assert hf_source.download_kwargs == {"resolution_km": 6.0}

    def test_non_us_bbox_with_hfradar_resolution_warns_and_keeps_copernicus(self, caplog):
        import logging

        from sar_validation.cli import _build_currents_config

        with caplog.at_level(logging.WARNING):
            cfg = _build_currents_config(
                min_lon=-10.0, max_lon=5.0, min_lat=50.0, max_lat=65.0,
                hfradar_resolution="1km",
            )
        source_types = [s.source_type for s in cfg.validation_sources]
        assert "hf_radar" in source_types
        assert "hf_radar_us" not in source_types
        assert any("has no effect" in r.message for r in caplog.records)

    def test_hycom_always_included_regardless_of_bbox(self):
        from sar_validation.cli import _build_currents_config

        # Default (non-US) bbox
        cfg = _build_currents_config()
        assert "hycom" in [s.source_type for s in cfg.validation_sources]

        # US-West bbox (exercises the hf_radar_us branch)
        cfg_us = _build_currents_config(min_lon=-130.0, max_lon=-117.0, min_lat=32.0, max_lat=42.0)
        assert "hycom" in [s.source_type for s in cfg_us.validation_sources]

    def test_hycom_source_has_no_download_kwargs_by_default(self):
        from sar_validation.cli import _build_currents_config

        cfg = _build_currents_config()
        hycom_source = next(s for s in cfg.validation_sources if s.source_type == "hycom")
        assert hycom_source.download_kwargs == {}
        assert hycom_source.collocation_kwargs == {}

class TestHfradarResolutionCliFlag:
    def test_rejected_for_non_currents_template(self, tmp_path, monkeypatch, capsys):
        from sar_validation.cli import main

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["--create-recipe", "wind", "--hfradar-resolution", "1km"])
        captured = capsys.readouterr()
        assert "currents" in (captured.out + captured.err)

    def test_accepted_for_currents_template(self, tmp_path, monkeypatch):
        from sar_validation.cli import main
        from sar_validation.core.recipe import Recipe

        monkeypatch.chdir(tmp_path)
        main([
            "--create-recipe", "currents", "--hfradar-resolution", "finest",
            "--min-lon", "-130.0", "--max-lon", "-117.0",
            "--min-lat", "32.0", "--max-lat", "42.0",
        ])
        recipe = Recipe.from_yaml(tmp_path / "recipes" / "currents_validation.yaml")
        hf_source = next(
            s for s in recipe.config.validation_sources if s.source_type == "hf_radar_us"
        )
        assert hf_source.download_kwargs == {"resolution_km": "finest"}


class TestAltimeterFreqCliFlag:
    @pytest.mark.parametrize("category", ["wind", "currents", "soil_moisture"])
    def test_rejected_for_non_waves_template(self, category, tmp_path, monkeypatch, capsys):
        from sar_validation.cli import main

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["--create-recipe", category, "--altimeter-freq", "5hz"])
        captured = capsys.readouterr()
        assert "waves" in (captured.out + captured.err)

    def test_accepted_for_waves_template_and_written_to_yaml(self, tmp_path, monkeypatch):
        from sar_validation.cli import main
        from sar_validation.core.recipe import Recipe

        monkeypatch.chdir(tmp_path)
        main(["--create-recipe", "waves", "--altimeter-freq", "5hz"])
        recipe = Recipe.from_yaml(tmp_path / "recipes" / "waves_validation.yaml")
        alt_source = next(
            s for s in recipe.config.validation_sources if s.source_type == "altimeter"
        )
        assert alt_source.download_kwargs == {"frequencies": ["5hz"]}

    def test_omitted_flag_defaults_waves_recipe_to_1hz(self, tmp_path, monkeypatch):
        from sar_validation.cli import main
        from sar_validation.core.recipe import Recipe

        monkeypatch.chdir(tmp_path)
        main(["--create-recipe", "waves"])
        recipe = Recipe.from_yaml(tmp_path / "recipes" / "waves_validation.yaml")
        alt_source = next(
            s for s in recipe.config.validation_sources if s.source_type == "altimeter"
        )
        assert alt_source.download_kwargs == {"frequencies": ["1hz"]}

    def test_both_value_written_to_yaml(self, tmp_path, monkeypatch):
        from sar_validation.cli import main
        from sar_validation.core.recipe import Recipe

        monkeypatch.chdir(tmp_path)
        main(["--create-recipe", "waves", "--altimeter-freq", "both"])
        recipe = Recipe.from_yaml(tmp_path / "recipes" / "waves_validation.yaml")
        alt_source = next(
            s for s in recipe.config.validation_sources if s.source_type == "altimeter"
        )
        assert alt_source.download_kwargs == {"frequencies": ["1hz", "5hz"]}

    def test_invalid_value_rejected_by_argparse(self, tmp_path, monkeypatch, capsys):
        from sar_validation.cli import main

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["--create-recipe", "waves", "--altimeter-freq", "10hz"])

    def test_rejected_when_combined_with_recipe_flag(self, tmp_path, monkeypatch, capsys):
        """--altimeter-freq only makes sense at recipe-creation time
        (--create-recipe waves); combining it with --recipe (executing an
        *existing* recipe file) must be rejected the same way as combining
        it with a non-waves --create-recipe category. args.create_recipe is
        None here, so the guard's `!= "waves"` check fires regardless of
        whether the recipe file exists -- parser.error() must run before
        the file is ever touched."""
        from sar_validation.cli import main

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["--recipe", "some_recipe.yaml", "--altimeter-freq", "5hz"])
        captured = capsys.readouterr()
        assert "waves" in (captured.out + captured.err) or "--recipe" in (
            captured.out + captured.err
        )


class TestLoadPrecomputedStats:
    def test_finds_files_saved_under_filter_variable_pairs_keys(self, tmp_path):
        """Regression test: run_statistics saves files keyed by
        filter_variable_pairs (dataset-aware), so _load_precomputed_stats
        must look them up the same way — not via the static
        infer_variable_pairs list, which used the wrong key
        (oswTotalHs_vs_VHM0) and silently found nothing for a mixed-mode
        WV/SM recipe where only sar_oswTotalHs/val_VAVH exist. VAVH/VHM0
        merge into one "SWH" pair (docs/design-choices.md §5.8), so the
        key/filename is oswTotalHs_vs_SWH even though this fixture only
        ever populates val_VAVH."""
        recipe = _waves_recipe(["WV", "SM"])
        collocation_ds = xr.Dataset({
            "sar_oswTotalHs": ("collocation", [1.4, 1.5]),
            "val_VAVH":       ("collocation", [1.42, 1.48]),
            "val_source":     ("collocation", ["altimeter", "altimeter"]),
        })

        stats_ds = xr.Dataset({"bias": ("source", [0.02])}, coords={"source": ["altimeter"]})
        stats_ds.to_netcdf(tmp_path / "validation_statistics_oswTotalHs_vs_SWH.nc")

        result = cli._load_precomputed_stats(recipe, collocation_ds, tmp_path)

        assert set(result.keys()) == {"oswTotalHs_vs_SWH"}
        assert float(result["oswTotalHs_vs_SWH"]["bias"].values[0]) == 0.02

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
        stats_ds.to_netcdf(tmp_path / "validation_statistics_oswTotalHs_vs_SWH_individual.nc")

        result = cli._load_precomputed_stats(recipe, collocation_ds, tmp_path, filename_suffix="_individual")

        assert set(result.keys()) == {"oswTotalHs_vs_SWH"}


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
    @pytest.mark.parametrize(
        "json_content,expected_result",
        [
            pytest.param(None, None, id="no_metadata_file"),
            pytest.param(
                {"errors": ["altimeter download failed: timeout"]},
                ["altimeter download failed: timeout"],
                id="errors_list_missing_notices_key",
            ),
            pytest.param(
                {
                    "errors": ["altimeter download failed: timeout"],
                    "notices": ["No delayed-mode in-situ current data found (adcp, argo) for this window."],
                },
                [
                    "altimeter download failed: timeout",
                    "No delayed-mode in-situ current data found (adcp, argo) for this window.",
                ],
                # notices (e.g. "no delayed-mode currents data found") must
                # also surface on the PDF cover page, same as errors -- a
                # notice isn't a failure, but the user still needs to see it
                # without scrolling back through the whole run's console
                # output.
                id="errors_and_notices_combined",
            ),
            pytest.param(
                {"errors": ["altimeter download failed: timeout"]},
                ["altimeter download failed: timeout"],
                # Same input/output as errors_list_missing_notices_key
                # above -- kept as its own row because it documents a
                # distinct guarantee (backward compatibility with metadata
                # files written before "notices" existed).
                id="missing_notices_key_is_backward_compatible",
            ),
        ],
    )
    def test_returns_combined_warnings_or_none(self, tmp_path, json_content, expected_result):
        import json

        if json_content is not None:
            (tmp_path / "download_metadata.json").write_text(json.dumps(json_content))

        assert cli._load_download_warnings(tmp_path) == expected_result


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

    @pytest.mark.parametrize(
        "layer_vs_layer_collocation_method,expected_suffix",
        [
            pytest.param("individual", "_individual", id="individual_maps_to_individual_suffix"),
            pytest.param("cell-averaging", "", id="cell_averaging_maps_to_empty_suffix"),
        ],
    )
    def test_method_maps_to_expected_filename_suffix(
        self, tmp_path, layer_vs_layer_collocation_method, expected_suffix
    ):
        from unittest.mock import patch

        recipe_path = self._write_recipe_with_skippable_download(tmp_path)

        with patch("sar_validation.cli._collocate_data") as mock_collocate:
            cli._execute_recipe(
                str(recipe_path), collocate=True,
                layer_vs_layer_collocation_method=layer_vs_layer_collocation_method,
            )

        _, kwargs = mock_collocate.call_args
        assert kwargs["filename_suffix"] == expected_suffix
        if layer_vs_layer_collocation_method == "individual":
            assert kwargs["layer_vs_layer_collocation_method"] == "individual"


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

    @pytest.mark.parametrize(
        "stats_file_exists",
        [
            pytest.param(True, id="skips_when_files_already_exist"),
            pytest.param(False, id="computes_when_files_missing"),
        ],
    )
    def test_stats_recompute_depends_on_existing_files(self, tmp_path, capsys, stats_file_exists):
        from unittest.mock import patch

        recipe_path, run_dir = self._write_recipe_with_collocation(tmp_path)

        if stats_file_exists:
            stats_ds = xr.Dataset({"bias": ("source", [0.1])}, coords={"source": ["scatterometer"]})
            stats_ds.to_netcdf(run_dir / "validation_statistics_owiWindSpeed_vs_WSPD.nc")

        with patch("sar_validation.cli._compute_stats") as mock_compute_stats:
            cli._execute_recipe(str(recipe_path), stats=True)

        if stats_file_exists:
            mock_compute_stats.assert_not_called()
            out = capsys.readouterr().out
            assert "Step 4 skipped" in out
        else:
            mock_compute_stats.assert_called_once()


class TestGeneratePlotsNonSoilMoistureVariables:
    """Regression test: _generate_plots's soil_moisture-only branch used to
    leave native_units_stats_ds_map completely unassigned for every other
    variable (wind/waves/currents), so the unconditional
    validation_report(..., native_units_stats_ds_map=...) call a few lines
    down raised UnboundLocalError for every non-soil_moisture --plot run --
    a regression introduced alongside the cds_ssm feature's own (correctly
    initialized) cds_ssm_stats_ds_map branch, undetected because nothing in
    the suite exercised _generate_plots for any variable."""

    def test_wind_plot_does_not_raise_and_passes_none_for_soil_moisture_only_maps(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from sar_validation.core.datatree_converter import DataTreeConverter

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        collocation_ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [7.0, 8.0]),
            "val_WSPD":         ("collocation", [7.2, 7.9]),
        })
        collocation_ds.to_netcdf(run_dir / "collocation_results.nc")

        sar_ds = xr.Dataset(
            {"owiWindSpeed": ("point", [7.0, 8.0])},
            coords={"lon": ("point", [-9.0, -8.5]), "lat": ("point", [50.0, 50.5])},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})
        datatree.to_netcdf(run_dir / "datatree.nc")

        recipe = Recipe(RecipeConfig(name="wind_test", variable="wind", output_dir=str(run_dir)))

        with patch("sar_validation.core.visualization.validation_report") as mock_report:
            mock_report.return_value = {}
            cli._generate_plots(recipe, run_dir)

        mock_report.assert_called_once()
        assert mock_report.call_args.kwargs["native_units_stats_ds_map"] is None
        assert mock_report.call_args.kwargs["cds_ssm_stats_ds_map"] is None


class TestIsAlreadyDownloaded:
    """_is_already_downloaded() gates the top-level "Step 1 skipped" shortcut
    -- it must not be fooled by a source that recorded no *error* but also
    never actually got real data. ISMN's "awaiting_manual_archive" status is
    exactly that case (see ISMNDownloader/_download_ismn): the user hasn't
    placed the shared archive yet, 0 files were collected, but that's
    deliberately not an "error" (no notice/error is appended) -- confirmed
    against a real run of recipes/soil_moisture_cds_nisar_test.yaml where
    Step 1 kept getting skipped on every rerun despite ISMN never actually
    downloading anything."""

    def test_true_when_no_errors_and_every_source_succeeded(self, tmp_path):
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "downloads": {
                "ismn": {"status": "success"},
                "scatterometer": {"status": "success"},
            },
        }))
        assert _is_already_downloaded(tmp_path) is True

    def test_false_when_ismn_still_awaiting_manual_archive(self, tmp_path):
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "downloads": {"ismn": {"status": "awaiting_manual_archive"}},
        }))
        assert _is_already_downloaded(tmp_path) is False

    def test_false_when_errors_present(self, tmp_path):
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({"errors": ["boom"]}))
        assert _is_already_downloaded(tmp_path) is False

    def test_false_when_metadata_file_missing(self, tmp_path):
        from sar_validation.cli import _is_already_downloaded

        assert _is_already_downloaded(tmp_path) is False

    def test_true_when_era5_recipe_variable_matches_recorded(self, tmp_path):
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "wind",
            "downloads": {"era5": {"status": "success"}},
        }))
        recipe = Recipe(RecipeConfig(
            name="wind_era5_test",
            variable="wind",
            validation_sources=[ValidationDataSource(source_type="era5")],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is True

    def test_false_when_era5_recipe_variable_differs_from_recorded(self, tmp_path):
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "wind",
            "downloads": {"era5": {"status": "success"}},
        }))
        recipe = Recipe(RecipeConfig(
            name="waves_era5_test",
            variable="waves",
            validation_sources=[ValidationDataSource(source_type="era5")],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is False

    def test_true_when_recorded_variable_missing_legacy_metadata(self, tmp_path):
        """Legacy/synthetic download_metadata.json without a top-level
        "variable" key must keep the old trust-it behavior -- this is the
        exact case that caused a live, unmocked network download during
        Task 15 when a naive first version of the mismatch check ignored
        the "recorded_variable is not None" guard."""
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "downloads": {"era5": {"status": "success"}},
        }))
        recipe = Recipe(RecipeConfig(
            name="wind_era5_test",
            variable="wind",
            validation_sources=[ValidationDataSource(source_type="era5")],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is True

    def test_true_when_non_era5_recipe_variable_differs_from_recorded(self, tmp_path):
        """A non-ERA5 recipe/source pair sharing a base_dir with a
        differing recorded ``variable`` must behave exactly as it did
        before commit 7fcba5b -- i.e. the top-level variable mismatch is
        irrelevant to sources other than era5, matching
        DataOrchestrator._already_succeeded's own
        ``source_type == "era5"`` scoping. Without that scoping, this
        recipe/base_dir pair would wrongly bypass "Step 1 skipped" and
        re-dispatch every _HISTORICAL_FIRST_TYPES source unconditionally
        (they have no _already_succeeded gate of their own in
        download_all()'s step 2)."""
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "currents",
            "downloads": {
                "sar": {"status": "success"},
                "adcp_historical": {"status": "success"},
            },
        }))
        recipe = Recipe(RecipeConfig(
            name="currents_test",
            variable="waves",
            validation_sources=[ValidationDataSource(source_type="adcp_historical")],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is True

    def test_false_when_recipe_requests_source_type_missing_from_recorded_downloads(self, tmp_path):
        """C2 regression: recipes/wind_era5.yaml (validation_sources=[era5])
        and recipes/wind_example.yaml (validation_sources=[mooring, ...,
        scatterometer, ..., NOT era5]) share identical geographic/temporal
        bounds and therefore the same auto-derived base_dir. Running
        wind_example.yaml first records downloads with no "era5" key;
        running wind_era5.yaml next must NOT be treated as
        already-downloaded (the era5-variable-mismatch check alone misses
        this, since era5 never appears in either set)."""
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "wind",
            "downloads": {
                "sar": {"status": "success"},
                "scatterometer": {"status": "success"},
            },
        }))
        recipe = Recipe(RecipeConfig(
            name="wind_era5_test",
            variable="wind",
            validation_sources=[ValidationDataSource(source_type="era5")],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is False

    def test_false_when_recorded_downloads_lack_requested_scatterometer(self, tmp_path):
        """Reverse of the case above: recorded downloads had era5 (not
        scatterometer); the current recipe wants scatterometer -- must
        also trigger re-download."""
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "wind",
            "downloads": {
                "sar": {"status": "success"},
                "era5": {"status": "success"},
            },
        }))
        recipe = Recipe(RecipeConfig(
            name="wind_example_test",
            variable="wind",
            validation_sources=[ValidationDataSource(source_type="scatterometer")],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is False

    def test_true_when_recorded_and_requested_source_types_match_exactly(self, tmp_path):
        """No-regression case: recorded downloads and the current recipe's
        requested source_types match exactly -- the skip must still fire,
        no wasted re-download for the common case."""
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "wind",
            "downloads": {
                "sar": {"status": "success"},
                "scatterometer": {"status": "success"},
                "altimeter": {"status": "success"},
            },
        }))
        recipe = Recipe(RecipeConfig(
            name="wind_example_test",
            variable="wind",
            validation_sources=[
                ValidationDataSource(source_type="scatterometer"),
                ValidationDataSource(source_type="altimeter"),
            ],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is True

    def test_true_when_recorded_insitu_key_covers_individual_insitu_source_types(self, tmp_path):
        """DataOrchestrator._download_insitu batches mooring/buoy/drifter/
        ferrybox/tidal_gauge under a single "insitu" downloads key, not one
        key per source_type -- a recipe requesting e.g. just "mooring" must
        still match against a recorded "insitu" key (no false re-download
        for the common in-situ case)."""
        import json

        from sar_validation.cli import _is_already_downloaded

        (tmp_path / "download_metadata.json").write_text(json.dumps({
            "errors": [],
            "variable": "wind",
            "downloads": {
                "sar": {"status": "success"},
                "insitu": {"status": "success", "source_types": ["mooring", "buoy"]},
            },
        }))
        recipe = Recipe(RecipeConfig(
            name="wind_insitu_test",
            variable="wind",
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="buoy"),
            ],
        ))
        assert _is_already_downloaded(tmp_path, recipe) is True


class TestExecuteRecipeForcesConvertCollocateWhenDownloadActuallyRan:
    """Steps 2/3 (convert/collocate) must not skip regenerating
    datatree.nc/collocation_results.nc just because those files already
    exist on disk from a *previous* run, if Step 1 actually did fresh
    download work this run (e.g. a previously-incomplete source like ISMN
    finally got real data) -- otherwise the stale datatree/collocation
    never pick up the newly downloaded files. Confirmed against a real
    rerun of recipes/soil_moisture_cds_nisar_test.yaml."""

    def _recipe_with_stale_outputs(self, tmp_path):
        from sar_validation.core.recipe import Recipe, RecipeConfig

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "datatree.nc").write_bytes(b"stale")
        (run_dir / "collocation_results.nc").write_bytes(b"stale")

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-force-regen", variable="wind", output_dir=str(run_dir),
        )).to_yaml(recipe_path)
        return recipe_path, run_dir

    def test_regenerates_when_download_step_actually_ran(self, tmp_path):
        from unittest.mock import patch

        recipe_path, run_dir = self._recipe_with_stale_outputs(tmp_path)

        with patch("sar_validation.core.orchestrator.DataOrchestrator") as mock_cls, \
                patch("sar_validation.cli._convert_data") as mock_convert, \
                patch("sar_validation.cli._collocate_data") as mock_collocate:
            mock_cls.return_value.download_all.return_value = True
            mock_cls.return_value.base_dir = run_dir
            cli._execute_recipe(str(recipe_path), convert=True, collocate=True)

        mock_convert.assert_called_once()
        mock_collocate.assert_called_once()

    def test_still_skips_when_step1_itself_was_skipped(self, tmp_path):
        import json
        from unittest.mock import patch

        recipe_path, run_dir = self._recipe_with_stale_outputs(tmp_path)
        (run_dir / "download_metadata.json").write_text(json.dumps({"errors": []}))

        with patch("sar_validation.core.orchestrator.DataOrchestrator") as mock_cls, \
                patch("sar_validation.cli._convert_data") as mock_convert, \
                patch("sar_validation.cli._collocate_data") as mock_collocate:
            mock_cls.return_value.base_dir = run_dir
            cli._execute_recipe(str(recipe_path), convert=True, collocate=True)

        mock_convert.assert_not_called()
        mock_collocate.assert_not_called()
        mock_cls.return_value.download_all.assert_not_called()


class TestBuildWindConfigEra5:
    def test_includes_era5_alongside_observational_sources(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config()
        source_types = [s.source_type for s in cfg.validation_sources]
        assert "era5" in source_types
        # era5 has no download_kwargs/collocation_kwargs of its own here --
        # ModelLayerCollocation's tuning comes from DEFAULT_LAYER_TYPE_SPECS's
        # "era5_wind" entry (recipe.py), not a per-recipe override.
        era5_source = next(s for s in cfg.validation_sources if s.source_type == "era5")
        assert era5_source.download_kwargs == {}

    def test_radarsat2_also_gets_era5(self):
        """validation_sources isn't conditioned on sar_source elsewhere in
        this template (mooring/buoy/etc. are shared by every source) --
        era5 follows the same pattern."""
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="radarsat2")
        source_types = [s.source_type for s in cfg.validation_sources]
        assert "era5" in source_types


class TestBuildWavesConfigEra5:
    def test_includes_era5_alongside_observational_sources(self):
        from sar_validation.cli import _build_waves_config

        cfg = _build_waves_config()
        source_types = [s.source_type for s in cfg.validation_sources]
        assert "era5" in source_types
        era5_source = next(s for s in cfg.validation_sources if s.source_type == "era5")
        assert era5_source.download_kwargs == {}


class TestBuildWavesConfigAltimeterFrequency:
    def test_default_is_1hz_only(self):
        from sar_validation.cli import _build_waves_config

        cfg = _build_waves_config()
        alt_source = next(s for s in cfg.validation_sources if s.source_type == "altimeter")
        assert alt_source.download_kwargs == {"frequencies": ["1hz"]}
        specs = cfg.collocation.layer_vs_layer.layer_type_specs
        assert "altimeter_1hz" in specs
        assert "altimeter_5hz" not in specs

    def test_none_is_treated_as_1hz(self):
        from sar_validation.cli import _build_waves_config

        cfg = _build_waves_config(altimeter_freq=None)
        alt_source = next(s for s in cfg.validation_sources if s.source_type == "altimeter")
        assert alt_source.download_kwargs == {"frequencies": ["1hz"]}

    def test_explicit_1hz(self):
        from sar_validation.cli import _build_waves_config

        cfg = _build_waves_config(altimeter_freq="1hz")
        alt_source = next(s for s in cfg.validation_sources if s.source_type == "altimeter")
        assert alt_source.download_kwargs == {"frequencies": ["1hz"]}
        specs = cfg.collocation.layer_vs_layer.layer_type_specs
        assert "altimeter_1hz" in specs
        assert "altimeter_5hz" not in specs

    def test_5hz(self):
        from sar_validation.cli import _build_waves_config

        cfg = _build_waves_config(altimeter_freq="5hz")
        alt_source = next(s for s in cfg.validation_sources if s.source_type == "altimeter")
        assert alt_source.download_kwargs == {"frequencies": ["5hz"]}
        specs = cfg.collocation.layer_vs_layer.layer_type_specs
        assert "altimeter_5hz" in specs
        assert "altimeter_1hz" not in specs

    def test_both(self):
        from sar_validation.cli import _build_waves_config

        cfg = _build_waves_config(altimeter_freq="both")
        alt_source = next(s for s in cfg.validation_sources if s.source_type == "altimeter")
        assert alt_source.download_kwargs == {"frequencies": ["1hz", "5hz"]}
        specs = cfg.collocation.layer_vs_layer.layer_type_specs
        assert "altimeter_1hz" in specs
        assert "altimeter_5hz" in specs

    def test_era5_layer_type_spec_still_present_regardless_of_frequency(self):
        # Guard against the layer_type_specs trim accidentally dropping the
        # unrelated era5_waves entry alongside the altimeter ones.
        from sar_validation.cli import _build_waves_config

        for freq in ("1hz", "5hz", "both"):
            cfg = _build_waves_config(altimeter_freq=freq)
            assert "era5_waves" in cfg.collocation.layer_vs_layer.layer_type_specs


class TestBuildSoilMoistureConfig:
    def test_recipe_shape(self):
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config()

        assert cfg.sar_data.source == "sentinel1_clms_ssm"
        source_types = [s.source_type for s in cfg.validation_sources]
        assert source_types == ["ismn", "ascat_ssm", "amsr_ssm", "smap_ssm", "smos_ssm", "cds_ssm", "era5"]
        for satellite_source in cfg.validation_sources[1:6]:
            assert satellite_source.download_kwargs == {} or satellite_source.source_type == "cds_ssm"
        # cds_ssm has product_type in download_kwargs
        cds_ssm_source = cfg.validation_sources[5]
        assert cds_ssm_source.source_type == "cds_ssm"
        assert cds_ssm_source.download_kwargs == {"product_type": "active"}

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


class TestSarSourceCliOption:
    def test_wind_rejects_soil_moisture_only_source(self):
        from sar_validation.cli import _build_wind_config

        with pytest.raises(ValueError, match="only valid for"):
            _build_wind_config(sar_source="sentinel1_clms_ssm")

    def test_wind_accepts_its_own_default_source_explicitly(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="sentinel1_l2_ocn")
        assert cfg.sar_data.source == "sentinel1_l2_ocn"

    def test_cli_sar_source_flag_rejects_invalid_key(self, tmp_path, monkeypatch, capsys):
        from sar_validation.cli import main

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            main(["--create-recipe", "wind", "--sar-source", "sentinel1_clms_ssm"])
        captured = capsys.readouterr()
        assert "only valid for" in captured.out or "only valid for" in captured.err

    def test_cli_sar_source_flag_writes_to_recipe(self, tmp_path, monkeypatch):
        from sar_validation.cli import main
        from sar_validation.core.recipe import Recipe

        monkeypatch.chdir(tmp_path)
        main(["--create-recipe", "soil_moisture", "--sar-source", "sentinel1_clms_ssm"])
        recipe = Recipe.from_yaml(tmp_path / "recipes" / "soil_moisture_validation.yaml")
        assert recipe.config.sar_data.source == "sentinel1_clms_ssm"


class TestBuildSoilMoistureConfigNisarSme2:
    def test_satellite_ssm_sources_get_360_minute_layer_type_spec_override(self):
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config(sar_source="nisar_sme2")
        specs = cfg.collocation.layer_vs_layer.layer_type_specs
        for key in ("scatterometer_ssm", "radiometer_ssm", "amsr_ssm", "smap_ssm", "smos_ssm"):
            assert specs[key]["time_tolerance_minutes"] == 360

    def test_sentinel1_clms_ssm_source_unaffected_no_ssm_sensor_overrides(self):
        """No per-sensor 360-min override (that's NISAR-only, see the test
        above) -- but layer_vs_layer itself is no longer None: it always
        carries an "era5_soil_moisture" entry now (see
        test_era5_soil_moisture_layer_type_spec_always_present below)."""
        from sar_validation.cli import _build_soil_moisture_config

        cfg = _build_soil_moisture_config(sar_source="sentinel1_clms_ssm")
        specs = cfg.collocation.layer_vs_layer.layer_type_specs
        for key in ("scatterometer_ssm", "radiometer_ssm", "amsr_ssm", "smap_ssm", "smos_ssm", "cds_ssm"):
            assert key not in specs
        pvl = cfg.collocation.point_vs_layer
        assert pvl.time_tolerance_minutes == 720
        assert pvl.aggregation_window_km == 1.0

    def test_era5_soil_moisture_layer_type_spec_always_present(self):
        """The model_vs_layer collocation type (era5_soil_moisture) must
        be explicit in every newly-created soil_moisture recipe, including
        time_tolerance_minutes -- previously this recipe's YAML gave no
        indication of what tolerance era5 downloads actually used,
        silently inheriting a code-level default the recipe never showed."""
        from sar_validation.cli import _build_soil_moisture_config

        for sar_source in ("sentinel1_clms_ssm", "nisar_sme2"):
            cfg = _build_soil_moisture_config(sar_source=sar_source)
            spec = cfg.collocation.layer_vs_layer.layer_type_specs["era5_soil_moisture"]
            assert spec["time_tolerance_minutes"] == 720


class TestSetCredentialCli:
    """--set-credential prompts for username/password and stores them in
    the OS keyring via sar_validation.downloaders.base.set_credential."""

    def test_prompts_and_stores_via_set_credential(self, monkeypatch, capsys):
        from sar_validation import cli

        monkeypatch.setattr("builtins.input", lambda prompt: "alice")
        monkeypatch.setattr("getpass.getpass", lambda prompt: "secret")

        calls = []
        monkeypatch.setattr(
            "sar_validation.downloaders.base.set_credential",
            lambda name, username, password: calls.append((name, username, password)),
        )

        cli._set_credential("eumdac")

        assert calls == [("eumdac", "alice", "secret")]
        assert "eumdac" in capsys.readouterr().out.lower()

    def test_reports_failure_and_exits_nonzero_when_keyring_unavailable(
        self, monkeypatch, capsys
    ):
        import keyring.errors

        from sar_validation import cli

        monkeypatch.setattr("builtins.input", lambda prompt: "alice")
        monkeypatch.setattr("getpass.getpass", lambda prompt: "secret")

        def _raise(name, username, password):
            raise keyring.errors.NoKeyringError("no backend")

        monkeypatch.setattr("sar_validation.downloaders.base.set_credential", _raise)

        with pytest.raises(SystemExit) as exc_info:
            cli._set_credential("eumdac")

        assert exc_info.value.code != 0
        assert "no backend" in capsys.readouterr().out.lower()

    def test_main_wires_set_credential_flag(self, monkeypatch):
        from sar_validation import cli

        called = []
        monkeypatch.setattr(cli, "_set_credential", lambda name: called.append(name))

        cli.main(["--set-credential", "smos"])

        assert called == ["smos"]


class TestBuildWindConfigRadarsat2:
    def test_radarsat2_source_recorded(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="radarsat2")
        assert cfg.sar_data.source == "radarsat2"

    def test_description_states_speed_only(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="radarsat2")
        assert "Speed only" in cfg.description

    def test_default_source_description_unchanged(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="sentinel1_l2_ocn")
        assert "Sentinel-1 IW/EW mode wind speed and direction" in cfg.description

    def test_components_speed_only(self):
        """RADARSAT-2 carries no SAR-retrieved wind direction (see the
        description text) -- variable_specs.components must match, not
        list "direction" as if it were validated too."""
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="radarsat2")
        assert cfg.variable_specs["components"] == ["speed"]

    def test_default_source_components_unchanged(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="sentinel1_l2_ocn")
        assert cfg.variable_specs["components"] == ["speed", "direction"]

    def test_swath_mode_empty(self):
        """swath_mode is Sentinel-1-specific terminology (SARDataSpec.
        swath_mode's own docstring: "ignored ... by every other source")
        -- must not carry over Sentinel-1's IW/EW values for a source
        that doesn't use them at all."""
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="radarsat2")
        assert cfg.sar_data.swath_mode == []

    def test_default_source_swath_mode_unchanged(self):
        from sar_validation.cli import _build_wind_config

        cfg = _build_wind_config(sar_source="sentinel1_l2_ocn")
        assert cfg.sar_data.swath_mode == ["IW", "EW"]

    def test_rejects_radarsat2_for_currents(self):
        from sar_validation.cli import _build_currents_config

        with pytest.raises(ValueError, match="only valid for"):
            _build_currents_config(sar_source="radarsat2")

    def test_rejects_radarsat2_for_soil_moisture(self):
        from sar_validation.cli import _build_soil_moisture_config

        with pytest.raises(ValueError, match="only valid for"):
            _build_soil_moisture_config(sar_source="radarsat2")

    def test_cli_create_recipe_wind_radarsat2_writes_source(self, tmp_path, monkeypatch):
        from sar_validation.cli import main
        from sar_validation.core.recipe import Recipe

        monkeypatch.chdir(tmp_path)
        main(["--create-recipe", "wind", "--sar-source", "radarsat2"])
        recipe = Recipe.from_yaml(tmp_path / "recipes" / "wind_validation.yaml")
        assert recipe.config.sar_data.source == "radarsat2"


class TestExecuteRecipeStopsWhenNoSarData:
    def test_real_run_stops_before_convert_when_sar_data_found_false(self, tmp_path, capsys):
        from unittest.mock import patch

        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-no-sar", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        def fake_download_all(self):
            self.metadata["sar_data_found"] = False
            return True

        with patch(
            "sar_validation.core.orchestrator.DataOrchestrator.download_all",
            fake_download_all,
        ), patch("sar_validation.cli._convert_data") as mock_convert:
            cli._execute_recipe(str(recipe_path), force_download=True, convert=True)

        mock_convert.assert_not_called()
        out = capsys.readouterr().out
        assert "No SAR data found" in out

    def test_dry_run_stops_before_dry_run_complete_message(self, tmp_path, capsys):
        from unittest.mock import patch

        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-no-sar-dry", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        def fake_download_all(self):
            self.metadata["sar_data_found"] = False
            return True

        with patch(
            "sar_validation.core.orchestrator.DataOrchestrator.download_all",
            fake_download_all,
        ):
            cli._execute_recipe(str(recipe_path), dry_run=True)

        out = capsys.readouterr().out
        assert "No SAR data found" in out
        assert "Dry run complete" not in out

    def test_real_run_proceeds_when_sar_data_found_true(self, tmp_path, capsys):
        """Sanity check the gate doesn't fire on a normal successful run."""
        from unittest.mock import patch

        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-has-sar", variable="wind", output_dir=str(tmp_path / "run"),
        )).to_yaml(recipe_path)

        def fake_download_all(self):
            self.metadata["sar_data_found"] = True
            return True

        with patch(
            "sar_validation.core.orchestrator.DataOrchestrator.download_all",
            fake_download_all,
        ):
            cli._execute_recipe(str(recipe_path), force_download=True)

        out = capsys.readouterr().out
        assert "No SAR data found" not in out
        assert "All downloads completed" in out

    def test_resume_shortcut_stops_when_cached_sar_found_count_is_zero(self, tmp_path, capsys):
        import json
        from unittest.mock import patch

        from sar_validation.core.recipe import Recipe, RecipeConfig

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "download_metadata.json").write_text(json.dumps({
            "variable": "wind",
            "errors": [],
            "notices": [],
            "downloads": {"sar": {"status": "success", "found_count": 0, "files": []}},
        }))

        recipe_path = tmp_path / "recipe.yaml"
        Recipe(RecipeConfig(
            name="test-resume-no-sar", variable="wind", output_dir=str(run_dir),
        )).to_yaml(recipe_path)

        with patch("sar_validation.cli._convert_data") as mock_convert:
            cli._execute_recipe(str(recipe_path), convert=True)

        mock_convert.assert_not_called()
        out = capsys.readouterr().out
        assert "No SAR data found" in out
