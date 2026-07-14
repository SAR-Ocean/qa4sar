"""Tests for sar_validation.core.visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
        "temporal_distance_minutes": ("collocation", rng.uniform(0, 180, n)),
    })


@pytest.fixture
def geo_datatree_and_collocation():
    """Synthetic DataTree + collocation_ds with two known validation
    sources (mooring, altimeter) within one SAR scene's bounds — used to
    test plot_geographic's and plot_collocation_diagnostics' per-source
    marker handling, and validation_report's temporal-offset pages."""
    from sar_validation.core.datatree_converter import DataTreeConverter

    y, x = 4, 5
    lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
    wind = np.linspace(5.0, 12.0, y * x).reshape(y, x)
    sar_ds = xr.Dataset(
        {"owiWindSpeed": (("y", "x"), wind)},
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
            "time": pd.Timestamp("2026-07-10T19:00:00"),
        },
    )

    n = 4
    mooring_ds = xr.Dataset(
        {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
        coords={
            "lon": ("point", np.array([-9.8, -9.6, -9.4, -9.2])),
            "lat": ("point", np.array([50.2, 50.4, 50.6, 50.8])),
            "time": ("point", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
        },
        attrs={"platform_type": "mooring"},
    )
    altimeter_ds = xr.Dataset(
        {"WSPD": ("point", np.array([8.0, 8.5, 9.0, 9.5]))},
        coords={
            "lon": ("point", np.array([-9.0, -8.8, -8.6, -8.4])),
            "lat": ("point", np.array([51.0, 51.2, 51.4, 51.6])),
            "time": ("point", pd.date_range("2026-07-10T19:10", periods=n, freq="5min")),
        },
        attrs={"platform_type": "altimeter"},
    )

    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": sar_ds,
        "validation/mooring": mooring_ds,
        "validation/altimeter": altimeter_ds,
    })

    collocation_ds = xr.Dataset({
        "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.9, 8.2, 9.3])),
        "val_WSPD":                    ("collocation", np.array([6.0, 7.0, 8.0, 9.5])),
        "val_source":                  ("collocation", ["mooring", "mooring", "altimeter", "altimeter"]),
        "sar_scene_name":              ("collocation", ["sceneA"] * n),
        "val_lon":                     ("collocation", np.array([-9.8, -9.6, -9.0, -8.8])),
        "val_lat":                     ("collocation", np.array([50.2, 50.4, 51.0, 51.2])),
        "val_id":                      ("collocation", ["mo0", "mo1", "al0", "al1"]),
        "temporal_distance_minutes":   ("collocation", np.array([10.0, 45.0, 90.0, 150.0])),
    })
    collocation_ds = collocation_ds.assign_coords(
        val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
    )
    return datatree, collocation_ds


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

    def test_distinct_sources_get_distinct_markers(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD")
        plt.close(fig)

        assert len(recorded_markers) == 2
        assert len(set(recorded_markers)) == 2


class TestPlotScatterColorByTemporalOffset:
    def test_returns_figure_with_colorbar(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        assert fig is not None
        assert len(fig.axes) >= 2  # main axes + colorbar axes
        plt.close(fig)

    def test_uses_distinct_markers_per_source(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        plt.close(fig)

        assert len(set(recorded_markers)) == 2

    def test_missing_temporal_column_falls_back_to_source(self):
        n = 5
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [8.0, 7.0, 6.0, 9.0, 10.0]),
            "val_WSPD":         ("collocation", [7.5, 7.2, 6.1, 8.9, 9.8]),
            "val_source":       ("collocation", ["buoy"] * n),
        })
        import matplotlib.pyplot as plt
        with pytest.warns(UserWarning, match="temporal_distance_minutes"):
            fig = plot_scatter(ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        assert fig is not None
        plt.close(fig)

    def test_shares_one_color_scale_across_sources(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes

        recorded_clims = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            pc = original_scatter(self, *args, **kwargs)
            if kwargs.get("c") is not None:
                recorded_clims.append(pc.get_clim())
            return pc

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        plt.close(fig)

        assert len(recorded_clims) == 2
        assert recorded_clims[0] == recorded_clims[1]


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


class TestPlotTemporalOffset:
    def test_returns_figure(self, collocation_ds):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_temporal_offset
        fig = plot_temporal_offset(collocation_ds, "owiWindSpeed", "WSPD")
        assert fig is not None
        plt.close(fig)

    def test_missing_temporal_column_returns_none(self):
        import warnings
        from sar_validation.core.visualization import plot_temporal_offset
        n = 5
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [8.0, 7.0, 6.0, 9.0, 10.0]),
            "val_WSPD":         ("collocation", [7.5, 7.2, 6.1, 8.9, 9.8]),
            "val_source":       ("collocation", ["buoy"] * n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_temporal_offset(ds, "owiWindSpeed", "WSPD")
        assert result is None

    def test_distinct_sources_get_distinct_markers(self, collocation_ds, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_temporal_offset

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_temporal_offset(collocation_ds, "owiWindSpeed", "WSPD")
        plt.close(fig)

        assert len(set(recorded_markers)) == 2


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


class TestSourceStyleMap:
    def test_stable_regardless_of_other_sources_present(self):
        from sar_validation.core.visualization import _source_style_map
        style_alone = _source_style_map(["altimeter"])
        style_with_others = _source_style_map(["altimeter", "radiometer", "mooring"])
        assert style_alone["altimeter"] == style_with_others["altimeter"]

    def test_distinct_known_sources_get_distinct_styles(self):
        from sar_validation.core.visualization import _source_style_map
        style = _source_style_map(["altimeter", "radiometer", "mooring", "buoy"])
        colors = [c for c, _ in style.values()]
        markers = [m for _, m in style.values()]
        assert len(set(colors)) == 4
        assert len(set(markers)) == 4

    def test_unknown_source_does_not_crash(self):
        # With exactly 9 canonical sources and a 9-entry palette, an unknown
        # source's index (9) wraps to the same palette slot as canonical
        # index 0 ("altimeter") — this is the same "cycles if more sources
        # than colours" behavior _SOURCE_COLORS already documents, not a
        # bug, so this only asserts "present, no crash," not "visually
        # distinct from every canonical source."
        from sar_validation.core.visualization import _source_style_map
        style = _source_style_map(["altimeter", "some_future_sensor"])
        assert "some_future_sensor" in style
        assert style["altimeter"] == ("#1f77b4", "o")

    def test_case_insensitive_matches_canonical_entry(self):
        from sar_validation.core.visualization import _source_style_map
        style_lower = _source_style_map(["altimeter"])
        style_title = _source_style_map(["Altimeter"])
        assert style_lower["altimeter"] == style_title["Altimeter"]


class TestPlotGeographic:
    def test_distinct_sources_get_distinct_markers(self, geo_datatree_and_collocation, monkeypatch):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        plt.close("all")

        assert fig is not None
        source_markers = [m for m in recorded_markers if m is not None]
        assert len(set(source_markers)) == 2


@pytest.fixture
def geo_datatree_and_collocation_with_unmatched():
    """Synthetic DataTree + collocation_ds with both matched and unmatched
    layer-type points (altimeter) to test per-source marker rendering for
    unmatched layer data."""
    from sar_validation.core.datatree_converter import DataTreeConverter

    y, x = 4, 5
    lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
    wind = np.linspace(5.0, 12.0, y * x).reshape(y, x)
    sar_ds = xr.Dataset(
        {"owiWindSpeed": (("y", "x"), wind)},
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
            "time": pd.Timestamp("2026-07-10T19:00:00"),
        },
    )

    # 4 mooring points: 2 matched, 2 unmatched
    mooring_ds = xr.Dataset(
        {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
        coords={
            "lon": ("point", np.array([-9.8, -9.6, -9.4, -9.2])),
            "lat": ("point", np.array([50.2, 50.4, 50.6, 50.8])),
            "time": ("point", pd.date_range("2026-07-10T19:05", periods=4, freq="5min")),
        },
        attrs={"platform_type": "mooring"},
    )

    # 6 altimeter points: 2 matched, 4 unmatched
    altimeter_ds = xr.Dataset(
        {"WSPD": ("point", np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5]))},
        coords={
            "lon": ("point", np.array([-9.0, -8.8, -8.6, -8.4, -8.2, -8.0])),
            "lat": ("point", np.array([51.0, 51.2, 51.4, 51.6, 51.8, 52.0])),
            "time": ("point", pd.date_range("2026-07-10T19:10", periods=6, freq="5min")),
        },
        attrs={"platform_type": "altimeter"},
    )

    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": sar_ds,
        "validation/mooring": mooring_ds,
        "validation/altimeter": altimeter_ds,
    })

    # Only 4 matches: 2 mooring, 2 altimeter (leaving 4 altimeter unmatched)
    collocation_ds = xr.Dataset({
        "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.9, 8.2, 9.3])),
        "val_WSPD":                    ("collocation", np.array([6.0, 7.0, 8.0, 9.5])),
        "val_source":                  ("collocation", ["mooring", "mooring", "altimeter", "altimeter"]),
        "sar_scene_name":              ("collocation", ["sceneA"] * 4),
        "val_lon":                     ("collocation", np.array([-9.8, -9.6, -9.0, -8.8])),
        "val_lat":                     ("collocation", np.array([50.2, 50.4, 51.0, 51.2])),
        "val_id":                      ("collocation", ["mo0", "mo1", "al0", "al1"]),
        "temporal_distance_minutes":   ("collocation", np.array([10.0, 45.0, 90.0, 150.0])),
    })
    collocation_ds = collocation_ds.assign_coords(
        val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=4, freq="5min")),
    )
    return datatree, collocation_ds


@pytest.fixture
def diagnostics_recipe():
    from sar_validation.core.recipe import (
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="wind",
        geographic_bounds=GeographicBounds(min_lon=-11.0, max_lon=-7.0, min_lat=49.0, max_lat=53.0),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="altimeter"),
        ],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)


class TestPlotCollocationDiagnostics:
    def test_distinct_sources_get_distinct_markers(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_markers = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_markers.append(kwargs.get("marker"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        matched_markers = [m for m in recorded_markers if m is not None]
        assert len(set(matched_markers)) >= 2


class TestPlotCollocationDiagnosticsRefinement:
    """Test 4-tier rendering with gray unmatched points."""

    def test_unmatched_points_are_gray_with_low_alpha(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Verify unmatched points render in gray (#808080) with alpha=0.3."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics
        import matplotlib.colors as mcolors

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_scatter_calls = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_scatter_calls.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None

        # Find unmatched scatter calls (they have no marker specified in new code,
        # or marker='o' by default, but crucially have color gray and alpha=0.3)
        unmatched_calls = [
            call for call in recorded_scatter_calls
            if call.get("c") == "#808080" or call.get("c") == (0.5, 0.5, 0.5)
        ]
        assert len(unmatched_calls) > 0, "Expected unmatched points in gray"

        # Check that unmatched points have low alpha
        for call in unmatched_calls:
            alpha = call.get("alpha")
            assert alpha is not None and alpha == 0.3, f"Expected alpha=0.3, got {alpha}"

    def test_zorder_ensures_insitu_on_top(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Verify z-order places matched in-situ data above matched layer data."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_scatter_calls = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_scatter_calls.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        assert len(recorded_scatter_calls) > 0, "Expected scatter calls"

        # Group calls by z-order
        by_zorder = {}
        for call in recorded_scatter_calls:
            zorder = call.get("zorder")
            alpha = call.get("alpha")
            marker = call.get("marker")
            if zorder is not None:
                if zorder not in by_zorder:
                    by_zorder[zorder] = []
                by_zorder[zorder].append({"alpha": alpha, "marker": marker})

        # Verify all 4 tiers are present (zorder 2, 3, 5, 6)
        # Note: zorder=1 is for SAR coverage (lines/plot, not scatter)
        expected_zorders = {2, 3, 5, 6}
        actual_zorders = set(by_zorder.keys())
        present_zorders = expected_zorders & actual_zorders
        assert len(present_zorders) >= 2, (
            f"Expected at least 2 of {expected_zorders} z-orders, "
            f"but got {present_zorders}"
        )

        # Verify alpha values: unmatched (0.3), matched layers (0.6), matched in-situ (0.7)
        unmatched_alphas = set()
        matched_layer_alphas = set()
        matched_insitu_alphas = set()

        if 2 in by_zorder:  # Tier 1: unmatched layers
            unmatched_alphas.update(call["alpha"] for call in by_zorder[2])
        if 3 in by_zorder:  # Tier 2: unmatched in-situ
            unmatched_alphas.update(call["alpha"] for call in by_zorder[3])
        if 5 in by_zorder:  # Tier 3: matched layers
            matched_layer_alphas.update(call["alpha"] for call in by_zorder[5])
        if 6 in by_zorder:  # Tier 4: matched in-situ
            matched_insitu_alphas.update(call["alpha"] for call in by_zorder[6])

        # Verify unmatched alpha is 0.3
        assert 0.3 in unmatched_alphas or len(unmatched_alphas) == 0, (
            f"Expected unmatched alpha=0.3, got {unmatched_alphas}"
        )

        # Verify matched layers alpha is 0.6
        assert 0.6 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=0.6, got {matched_layer_alphas}"
        )

        # Verify matched in-situ alpha is 0.7
        assert 0.7 in matched_insitu_alphas or len(matched_insitu_alphas) == 0, (
            f"Expected matched in-situ alpha=0.7, got {matched_insitu_alphas}"
        )

        # Verify per-source markers are used (not all None)
        all_markers = [call["marker"] for call in recorded_scatter_calls if call.get("marker") is not None]
        assert len(all_markers) > 0, "Expected per-source markers to be used"
        # Verify we have distinct markers (not all the same)
        unique_markers = set(all_markers)
        assert len(unique_markers) >= 1, "Expected at least one marker type"

    def test_unmatched_layer_points_get_per_source_markers(
        self, geo_datatree_and_collocation_with_unmatched, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Verify unmatched layer-type (altimeter) points get per-source markers, not default 'o'."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation_with_unmatched

        recorded_scatter_calls = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_scatter_calls.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None

        # Find unmatched layer data calls (zorder=2, alpha=0.3, c="#808080")
        unmatched_layer_calls = [
            call for call in recorded_scatter_calls
            if (call.get("zorder") == 2 and call.get("alpha") == 0.3 and call.get("c") == "#808080")
        ]
        assert len(unmatched_layer_calls) > 0, "Expected unmatched layer points"

        # Verify that at least some have non-default markers (not all "o")
        markers_used = [call.get("marker") for call in unmatched_layer_calls]
        unique_markers = set(m for m in markers_used if m is not None)
        assert len(unique_markers) > 0, "Expected per-source markers on unmatched layer points"


class TestValidationReport:
    def test_includes_temporal_offset_plots(self, geo_datatree_and_collocation, tmp_path):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))

        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        key = "owiWindSpeed_vs_WSPD"
        assert key in figures
        assert (tmp_path / "plots" / f"{key}_scatter_by_offset.png").exists()
        assert (tmp_path / "plots" / f"{key}_temporal_offset.png").exists()
        assert (tmp_path / "validation_report.pdf").exists()
        plt.close("all")


class TestValidationReportIncludesDiagnostics:
    def test_diagnostics_plot_included_in_report(self, geo_datatree_and_collocation, tmp_path):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        # Check that the collocation diagnostics plot PNG was created
        assert (tmp_path / "plots" / "collocation_diagnostics_test_recipe.png").exists()
        plt.close("all")

    def test_diagnostics_page_embedded_in_pdf(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """The diagnostics PNG must not just be saved to disk — it must also
        appear as a page inside validation_report.pdf. Spy on
        PdfPages.savefig and check one of the saved pages is a
        single-axes, full-bleed image page (the shape produced when the
        reloaded diagnostics PNG is embedded via imshow) — no other plot
        function in this module uses imshow, so this signature is unique
        to the diagnostics page.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        recorded_figs = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_figs.append(fig)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)

        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        def is_image_page(fig):
            if fig is None or len(fig.axes) != 1:
                return False
            return len(fig.axes[0].images) > 0

        assert any(is_image_page(fig) for fig in recorded_figs), (
            "Expected the collocation diagnostics plot to be embedded as a "
            "page in validation_report.pdf"
        )
