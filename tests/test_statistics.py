"""Tests for sar_validation.core.statistics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from sar_validation.core.statistics import compute_statistics, save_statistics, run_statistics
from sar_validation.core._variable_map import infer_variable_pairs
from sar_validation.core.recipe import Recipe, RecipeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collocation_ds():
    """Synthetic collocation dataset with two sources."""
    n = 40
    rng = np.random.default_rng(42)
    sar_vals = rng.uniform(2, 14, size=n)
    # Correlated validation values with small noise
    val_vals = sar_vals + rng.normal(0, 0.5, size=n)

    sources = ["mooring"] * 20 + ["buoy"] * 20

    ds = xr.Dataset(
        {
            "sar_owiWindSpeed": ("collocation", sar_vals),
            "val_WSPD":         ("collocation", val_vals),
            "val_source":       ("collocation", sources),
            "sar_lon":          ("collocation", rng.uniform(-10, 5, n)),
            "sar_lat":          ("collocation", rng.uniform(50, 65, n)),
            "val_lon":          ("collocation", rng.uniform(-10, 5, n)),
            "val_lat":          ("collocation", rng.uniform(50, 65, n)),
        }
    )
    return ds


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------

class TestComputeStatistics:
    def test_returns_dataset(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        assert isinstance(ds, xr.Dataset)

    def test_expected_metrics(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        for metric in ("N", "bias", "std", "rmse", "correlation", "scatter_index"):
            assert metric in ds.data_vars, f"Missing metric: {metric}"

    def test_sources_dimension(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        assert "source" in ds.dims
        assert set(ds["source"].values) == {"mooring", "buoy"}

    def test_bias_near_zero(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        # Synthetic data has small noise so bias should be small
        assert abs(float(ds["bias"].mean())) < 1.0

    def test_rmse_positive(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        assert (ds["rmse"].values >= 0).all()

    def test_correlation_in_range(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        corr = ds["correlation"].values
        assert np.all((corr >= -1) & (corr <= 1))

    def test_missing_var_returns_none(self, collocation_ds):
        result = compute_statistics(collocation_ds, "owiWaveHeight", "VHM0")
        assert result is None

    def test_n_correct(self, collocation_ds):
        ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        total_n = int(ds["N"].sum())
        assert total_n == 40


# ---------------------------------------------------------------------------
# Circular statistics (wind direction)
# ---------------------------------------------------------------------------

@pytest.fixture
def direction_collocation_ds():
    """Direction pairs that straddle the 0°/360° wrap boundary."""
    sar_deg = np.array([10.0, 90.0, 180.0, 270.0, 359.0])
    val_deg = (sar_deg + 2.0) % 360.0  # sar - val should wrap to ~-2° everywhere

    ds = xr.Dataset(
        {
            "sar_owiWindDirection": ("collocation", sar_deg),
            "val_WDIR":             ("collocation", val_deg),
            "val_source":           ("collocation", ["buoy"] * len(sar_deg)),
        }
    )
    return ds


class TestCircularStatistics:
    def test_bias_uses_wrapped_difference(self, direction_collocation_ds):
        ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")
        # A naive (sar - val) mean would be dominated by the 358° outlier at
        # the wrap boundary; the correct wrapped bias is close to -2°.
        assert abs(float(ds["bias"].values[0]) - (-2.0)) < 1e-6

    def test_rmse_small_despite_wrap(self, direction_collocation_ds):
        ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")
        assert float(ds["rmse"].values[0]) < 5.0

    def test_correlation_near_one_for_rotated_series(self, direction_collocation_ds):
        ds = compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")
        assert float(ds["correlation"].values[0]) > 0.9

    def test_no_runtime_warning(self, direction_collocation_ds):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            compute_statistics(direction_collocation_ds, "owiWindDirection", "WDIR")

    def test_constant_direction_group_no_warning(self):
        """A group with zero angular spread must not raise a RuntimeWarning."""
        ds = xr.Dataset(
            {
                "sar_owiWindDirection": ("collocation", [180.0, 180.0]),
                "val_WDIR":             ("collocation", [180.0, 180.0]),
                "val_source":           ("collocation", ["buoy", "buoy"]),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = compute_statistics(ds, "owiWindDirection", "WDIR")
        assert np.isnan(float(result["correlation"].values[0]))


# ---------------------------------------------------------------------------
# run_statistics — platform-type grouping
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_station_collocation_ds():
    """Multiple stations per platform type, plus a scatterometer source."""
    rng = np.random.default_rng(7)

    def _block(n, source, val_ids):
        sar_vals = rng.uniform(2, 14, size=n)
        val_vals = sar_vals + rng.normal(0, 0.5, size=n)
        return sar_vals, val_vals, [source] * n, val_ids

    sar1, val1, src1, id1 = _block(6, "mooring", ["MO_A"] * 3 + ["MO_B"] * 3)
    sar2, val2, src2, id2 = _block(4, "buoy", ["BUOY_X"] * 2 + ["BUOY_Y"] * 2)
    sar3, val3, src3, id3 = _block(50, "scatterometer", ["unknown"] * 50)

    sar_vals = np.concatenate([sar1, sar2, sar3])
    val_vals = np.concatenate([val1, val2, val3])
    sources = src1 + src2 + src3
    val_ids = id1 + id2 + id3

    return xr.Dataset(
        {
            "sar_owiWindSpeed": ("collocation", sar_vals),
            "val_WSPD":         ("collocation", val_vals),
            "val_source":       ("collocation", sources),
            "val_id":           ("collocation", val_ids),
        }
    )


class TestRunStatisticsGrouping:
    def _recipe(self):
        return Recipe(RecipeConfig(name="test", variable="wind"))

    def test_groups_by_platform_type_not_station(self, tmp_path, multi_station_collocation_ds):
        results = run_statistics(multi_station_collocation_ds, self._recipe(), tmp_path)
        stats_ds = results["owiWindSpeed_vs_WSPD"]
        assert set(stats_ds["source"].values) == {"mooring", "buoy", "scatterometer"}

    def test_scatterometer_row_present_with_full_count(self, tmp_path, multi_station_collocation_ds):
        results = run_statistics(multi_station_collocation_ds, self._recipe(), tmp_path)
        stats_ds = results["owiWindSpeed_vs_WSPD"]
        df = stats_ds.to_dataframe()
        assert int(df.loc["scatterometer", "N"]) == 50


# ---------------------------------------------------------------------------
# save_statistics
# ---------------------------------------------------------------------------

class TestSaveStatistics:
    def test_writes_nc_and_csv(self, tmp_path, collocation_ds):
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        out_path = tmp_path / "stats.nc"
        save_statistics(stats_ds, out_path)
        assert out_path.exists()
        assert (tmp_path / "stats.csv").exists()

    def test_nc_roundtrip(self, tmp_path, collocation_ds):
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        out_path = tmp_path / "stats.nc"
        save_statistics(stats_ds, out_path)
        loaded = xr.open_dataset(out_path)
        assert "bias" in loaded.data_vars
        assert "rmse" in loaded.data_vars


# ---------------------------------------------------------------------------
# _variable_map
# ---------------------------------------------------------------------------

class TestVariableMap:
    def test_wind_pairs(self):
        pairs = infer_variable_pairs("wind")
        assert ("owiWindSpeed", "WSPD") in pairs
        assert ("owiWindDirection", "WDIR") in pairs

    def test_currents_pairs(self):
        pairs = infer_variable_pairs("currents")
        assert ("rvlRadVel", "rvlRadVel_projection") in pairs

    def test_waves_pairs(self):
        pairs = infer_variable_pairs("waves")
        assert ("oswHs", "VHM0") in pairs

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            infer_variable_pairs("invalid_variable")
