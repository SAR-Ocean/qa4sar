"""Tests for sar_validation.core.statistics."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sar_validation.core.statistics import compute_statistics, save_statistics
from sar_validation.core._variable_map import infer_variable_pairs


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
        assert ("owiEastwardCurrent", "EWCT") in pairs

    def test_waves_pairs(self):
        pairs = infer_variable_pairs("waves")
        assert ("owiSignificantWaveHeight", "VHM0") in pairs

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            infer_variable_pairs("invalid_variable")
