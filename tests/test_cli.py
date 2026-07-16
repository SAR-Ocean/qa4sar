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
