"""Tests for sar_validation.core.visualization and patch_extractor."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sar_validation.core.patch_extractor import (
    extract_patches,
    add_patches_to_dataset,
    _validate_patch_size,
)
from sar_validation.core.visualization import (
    plot_scatter,
    plot_residuals,
    plot_statistics,
)
from sar_validation.core.statistics import compute_statistics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collocation_ds():
    """Minimal synthetic collocation dataset."""
    n = 30
    rng = np.random.default_rng(0)
    sar_vals = rng.uniform(3, 12, n)
    val_vals = sar_vals + rng.normal(0, 0.4, n)
    sources = ["mooring"] * 15 + ["scatterometer"] * 15
    scene_names = ["sceneA"] * 15 + ["sceneB"] * 15

    return xr.Dataset({
        "sar_owiWindSpeed": ("collocation", sar_vals),
        "val_WSPD":         ("collocation", val_vals),
        "val_source":       ("collocation", sources),
        "sar_scene_name":   ("collocation", scene_names),
        "sar_y_idx":        ("collocation", rng.integers(0, 10, n)),
        "sar_x_idx":        ("collocation", rng.integers(0, 10, n)),
        "sar_lon":          ("collocation", rng.uniform(-10, 5, n)),
        "sar_lat":          ("collocation", rng.uniform(50, 65, n)),
        "val_lon":          ("collocation", rng.uniform(-10, 5, n)),
        "val_lat":          ("collocation", rng.uniform(50, 65, n)),
    })


@pytest.fixture
def mock_datatree():
    """Minimal DataTree with two SAR scenes."""
    rng = np.random.default_rng(1)
    ny, nx = 20, 20

    def make_scene_ds():
        lon = np.linspace(-10, 5, nx)
        lat = np.linspace(50, 65, ny)
        lon2d, lat2d = np.meshgrid(lon, lat)
        return xr.Dataset(
            {"owiWindSpeed": (("y", "x"), rng.uniform(3, 15, (ny, nx)))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d)},
        )

    scene_a = make_scene_ds()
    scene_b = make_scene_ds()

    dt = xr.DataTree.from_dict({
        "/sar/sceneA": scene_a,
        "/sar/sceneB": scene_b,
    })
    return dt


# ---------------------------------------------------------------------------
# patch_extractor
# ---------------------------------------------------------------------------

class TestValidatePatchSize:
    def test_valid_odd(self):
        assert _validate_patch_size(5) == 5

    def test_even_rounds_up(self):
        with pytest.warns(UserWarning, match="rounding up"):
            result = _validate_patch_size(4)
        assert result == 5

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _validate_patch_size(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _validate_patch_size(-3)


class TestExtractPatches:
    def test_output_shape(self, collocation_ds, mock_datatree):
        patches = extract_patches(collocation_ds, mock_datatree, patch_size=5)
        assert "sar_patch_owiWindSpeed" in patches
        arr = patches["sar_patch_owiWindSpeed"]
        assert arr.shape == (30, 5, 5)

    def test_patch_size_3(self, collocation_ds, mock_datatree):
        patches = extract_patches(collocation_ds, mock_datatree, patch_size=3)
        arr = patches["sar_patch_owiWindSpeed"]
        assert arr.shape == (30, 3, 3)

    def test_no_sar_group_raises(self, collocation_ds):
        empty_dt = xr.DataTree()
        with pytest.raises(ValueError, match="no '/sar' group"):
            extract_patches(collocation_ds, empty_dt, patch_size=5)


class TestAddPatchesToDataset:
    def test_adds_variables(self, collocation_ds, mock_datatree):
        patches = extract_patches(collocation_ds, mock_datatree, patch_size=5)
        augmented = add_patches_to_dataset(collocation_ds, patches, patch_size=5)
        assert "sar_patch_owiWindSpeed" in augmented.data_vars

    def test_patch_dimensions(self, collocation_ds, mock_datatree):
        patches = extract_patches(collocation_ds, mock_datatree, patch_size=5)
        augmented = add_patches_to_dataset(collocation_ds, patches, patch_size=5)
        var = augmented["sar_patch_owiWindSpeed"]
        assert var.dims == ("collocation", "patch_y", "patch_x")
        assert var.sizes["patch_y"] == 5
        assert var.sizes["patch_x"] == 5

    def test_patch_offsets(self, collocation_ds, mock_datatree):
        patches = extract_patches(collocation_ds, mock_datatree, patch_size=5)
        augmented = add_patches_to_dataset(collocation_ds, patches, patch_size=5)
        offsets = augmented["patch_y"].values.tolist()
        assert offsets == [-2, -1, 0, 1, 2]


# ---------------------------------------------------------------------------
# visualization (smoke tests — just check no exception and correct type)
# ---------------------------------------------------------------------------

class TestPlotScatter:
    def test_returns_figure(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD")
        assert fig is not None
        plt.close("all")

    def test_missing_var_returns_none(self, collocation_ds):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_scatter(collocation_ds, "owiWaveHeight", "VHM0")
        assert result is None

    def test_by_source_false(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", by_source=False)
        assert fig is not None
        plt.close("all")

    def test_constant_values_no_runtime_warning(self):
        import warnings
        import matplotlib.pyplot as plt

        n = 5
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [8.0] * n),
            "val_WSPD":         ("collocation", [8.0] * n),
            "val_source":       ("collocation", ["buoy"] * n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fig = plot_scatter(ds, "owiWindSpeed", "WSPD")
        assert fig is not None
        plt.close("all")


class TestPlotResiduals:
    def test_returns_figure(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD")
        assert fig is not None
        plt.close("all")

    def test_missing_var_returns_none(self, collocation_ds):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_residuals(collocation_ds, "owiWaveHeight", "VHM0")
        assert result is None


class TestPlotStatistics:
    def test_returns_figure(self, collocation_ds):
        import matplotlib.pyplot as plt
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        fig = plot_statistics(stats_ds)
        assert fig is not None
        plt.close("all")

    def test_custom_metrics(self, collocation_ds):
        import matplotlib.pyplot as plt
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        fig = plot_statistics(stats_ds, metrics=["bias", "rmse"])
        assert fig is not None
        plt.close("all")

    def test_no_valid_metrics_warns(self, collocation_ds):
        import warnings
        stats_ds = compute_statistics(collocation_ds, "owiWindSpeed", "WSPD")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = plot_statistics(stats_ds, metrics=["nonexistent_metric"])
        assert result is None
        assert any("nonexistent_metric" in str(warning.message) for warning in w)
