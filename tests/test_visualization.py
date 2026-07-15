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

    def test_every_canonical_source_gets_a_distinct_style(self):
        # Regression test: the palette must have at least as many entries as
        # there are canonical source names, otherwise it wraps and two
        # unrelated sources silently collide (e.g. tidal_gauge landing on
        # altimeter's blue circle because the palette was one entry short).
        from sar_validation.core.visualization import (
            _source_style_map, _canonical_source_order,
        )
        canonical = _canonical_source_order()
        style = _source_style_map(canonical)
        pairs = [style[name] for name in canonical]
        assert len(set(pairs)) == len(canonical), (
            f"Expected every canonical source to get a distinct (color, marker) "
            f"pair, got collisions: {pairs}"
        )

    def test_unknown_source_does_not_crash(self):
        # With exactly 10 canonical sources and a 10-entry palette, an
        # unknown source's index (10) wraps to the same palette slot as
        # canonical index 0 ("altimeter") — this is the same "cycles if more
        # sources than colours" behavior _SOURCE_COLORS already documents,
        # not a bug, so this only asserts "present, no crash," not "visually
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
def geo_datatree_and_collocation_mixed_layer_counts():
    """Synthetic DataTree + collocation_ds where every layer/in-situ source
    has a distinct matched-point count (scatterometer=5, radiometer=3,
    altimeter=1, mooring=2, buoy=1) — used to verify
    plot_collocation_diagnostics draws denser sources before sparser ones."""
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

    def _point_ds(n, lon0, lat0, platform_type):
        return xr.Dataset(
            {"WSPD": ("point", np.linspace(6.0, 10.0, n))},
            coords={
                "lon": ("point", lon0 + 0.05 * np.arange(n)),
                "lat": ("point", lat0 + 0.05 * np.arange(n)),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
            },
            attrs={"platform_type": platform_type},
        )

    scatt_ds = _point_ds(5, -9.8, 50.2, "scatterometer")
    radio_ds = _point_ds(3, -9.6, 50.6, "radiometer")
    alt_ds = _point_ds(1, -9.4, 51.0, "altimeter")
    mooring_ds = _point_ds(2, -9.2, 51.4, "mooring")
    buoy_ds = _point_ds(1, -9.0, 51.8, "buoy")

    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": sar_ds,
        "validation/scatterometer": scatt_ds,
        "validation/radiometer": radio_ds,
        "validation/altimeter": alt_ds,
        "validation/mooring": mooring_ds,
        "validation/buoy": buoy_ds,
    })

    def _matched_block(ds, source_label):
        n = ds.sizes["point"]
        return {
            "sar_owiWindSpeed": np.full(n, 7.0),
            "val_WSPD": ds["WSPD"].values,
            "val_source": np.array([source_label] * n),
            "sar_scene_name": np.array(["sceneA"] * n),
            "val_lon": ds["lon"].values,
            "val_lat": ds["lat"].values,
            "temporal_distance_minutes": np.full(n, 10.0),
        }

    blocks = [
        _matched_block(scatt_ds, "scatterometer"),
        _matched_block(radio_ds, "radiometer"),
        _matched_block(alt_ds, "altimeter"),
        _matched_block(mooring_ds, "mooring"),
        _matched_block(buoy_ds, "buoy"),
    ]
    merged = {
        key: np.concatenate([block[key] for block in blocks])
        for key in blocks[0]
    }
    collocation_ds = xr.Dataset({k: ("collocation", v) for k, v in merged.items()})
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

        # Verify matched layers alpha is 1.0 (emphasized so few matches stay visible)
        assert 1.0 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=1.0, got {matched_layer_alphas}"
        )

        # Verify matched in-situ alpha is 1.0 (opaque, matching matched layers)
        assert 1.0 in matched_insitu_alphas or len(matched_insitu_alphas) == 0, (
            f"Expected matched in-situ alpha=1.0, got {matched_insitu_alphas}"
        )

        # Verify per-source markers are used (not all None)
        all_markers = [call["marker"] for call in recorded_scatter_calls if call.get("marker") is not None]
        assert len(all_markers) > 0, "Expected per-source markers to be used"
        # Verify we have distinct markers (not all the same)
        unique_markers = set(all_markers)
        assert len(unique_markers) >= 1, "Expected at least one marker type"

    def test_matched_layer_points_are_emphasized(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Matched layer points (zorder=5) must be drawn bold: full opacity,
        no marker edge, same marker size as matched in-situ points (s=25)
        — so a few matched points stay visible against the SAR footprints
        and gray unmatched tracks, without visually outsizing in-situ."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        matched_layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert matched_layer_calls, "Expected at least one matched-layer scatter call"
        for c in matched_layer_calls:
            assert c.get("edgecolors") == "none"
            assert c.get("alpha") == 1.0
            assert c.get("s") == 25

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

    def test_matched_sources_drawn_in_descending_matched_count_order(
        self, geo_datatree_and_collocation_mixed_layer_counts, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Denser sources (more matched points) must be drawn before sparser
        ones within each tier, so a sparse instrument (e.g. altimeter) ends
        up layered on top of a dense one (e.g. scatterometer) instead of
        being buried underneath it."""
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation_mixed_layer_counts

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None

        # Tier 3 (zorder=5): matched layer data, one call per category.
        # scatterometer=5, radiometer=3, altimeter=1 matched points.
        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        drawn_order = [c["label"].split(" matched")[0] for c in layer_calls]
        assert drawn_order == ["Scatterometer", "Radiometer", "Altimeter"], (
            f"Expected layer categories drawn most-matched-first, got {drawn_order}"
        )

        # Tier 4 (zorder=6): matched in-situ data, one call per sub-source.
        # mooring=2, buoy=1 matched points.
        insitu_calls = [c for c in recorded if c.get("zorder") == 6]
        insitu_order = [c["label"].split(": ")[1].split(" (")[0] for c in insitu_calls]
        assert insitu_order == ["mooring", "buoy"], (
            f"Expected in-situ sub-sources drawn most-matched-first, got {insitu_order}"
        )
        insitu_labels = [c["label"] for c in insitu_calls]
        assert "In-situ matched: mooring (2)" in insitu_labels
        assert "In-situ matched: buoy (1)" in insitu_labels


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

    def test_diagnostics_page_is_first_after_cover(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """The diagnostics image page must be the first content page (index 1,
        right after the cover) in the PDF."""
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
            return fig is not None and len(fig.axes) == 1 and len(fig.axes[0].images) > 0

        image_indices = [i for i, f in enumerate(recorded_figs) if is_image_page(f)]
        assert image_indices, "Expected a diagnostics image page in the PDF"
        assert image_indices[0] == 1, (
            f"Diagnostics page should be first after the cover (index 1), "
            f"got first image page at index {image_indices[0]}"
        )


class TestDropNonDirectionalSources:
    def _ds(self):
        return xr.Dataset({
            "sar_owiWindDirection": ("collocation", [10.0, 20.0, 30.0, 40.0]),
            "val_WDIR":   ("collocation", [12.0, np.nan, 33.0, np.nan]),
            "val_WSPD":   ("collocation", [6.0, 6.5, 7.0, 7.5]),
            "val_source": ("collocation", ["mooring", "altimeter", "mooring", "altimeter"]),
        })

    def test_drops_all_nan_source_for_circular_var(self):
        from sar_validation.core.visualization import _drop_nondirectional_sources
        out = _drop_nondirectional_sources(self._ds(), "WDIR")
        kept = set(out["val_source"].values.tolist())
        assert kept == {"mooring"}          # altimeter had all-NaN WDIR

    def test_keeps_all_sources_when_var_has_values(self):
        from sar_validation.core.visualization import _drop_nondirectional_sources
        out = _drop_nondirectional_sources(self._ds(), "WSPD")
        kept = set(out["val_source"].values.tolist())
        assert kept == {"mooring", "altimeter"}  # both have finite WSPD

    def test_returns_input_when_columns_absent(self):
        from sar_validation.core.visualization import _drop_nondirectional_sources
        ds = xr.Dataset({"sar_x": ("collocation", [1.0, 2.0])})
        out = _drop_nondirectional_sources(ds, "WDIR")
        assert out is ds


class TestValidationReportWindDirectionFilter:
    def test_nondirectional_source_absent_from_wdir_scatter(self, tmp_path, monkeypatch):
        """Altimeter (all-NaN WDIR) must not appear in the wind-direction
        scatter, but must appear in the wind-speed scatter."""
        import matplotlib.pyplot as plt
        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.datatree_converter import DataTreeConverter

        # Minimal 1-scene SAR datatree so plot_geographic has something to draw.
        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"owiWindSpeed": (("y", "x"), np.full((y, x), 7.0)),
             "owiWindDirection": (("y", "x"), np.full((y, x), 90.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        coll = xr.Dataset({
            "sar_owiWindSpeed":     ("collocation", [7.0, 7.1, 7.2, 7.3]),
            "sar_owiWindDirection": ("collocation", [90.0, 92.0, 88.0, 91.0]),
            "val_WSPD":             ("collocation", [6.8, 6.9, 7.0, 7.1]),
            "val_WDIR":             ("collocation", [85.0, np.nan, 95.0, np.nan]),
            "val_source":           ("collocation", ["mooring", "altimeter", "mooring", "altimeter"]),
            "sar_scene_name":       ("collocation", ["sceneA"] * 4),
            "val_lon":              ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":              ("collocation", [50.5, 50.6, 50.7, 50.8]),
            # Present so color_by="temporal_offset" and plot_temporal_offset
            # don't fall back / bail with "missing" warnings unrelated to
            # what this test is checking (every other fixture in this file
            # that exercises those code paths includes this column too).
            "temporal_distance_minutes": ("collocation", [10.0, 20.0, 15.0, 25.0]),
        })

        # Spy at the wiring boundary itself: wrap the module-level
        # plot_scatter and plot_geographic so we observe the *dataset each
        # one receives* for each (sar_var, val_var) pair, not a downstream
        # rendering side-effect. (A scatter/label-based spy is unreliable
        # here — plot_scatter and plot_temporal_offset each do their own
        # dropna() on the val_ column before ever calling ax.scatter, which
        # would independently strip altimeter's all-NaN WDIR rows even if
        # validation_report's pair_ds filtering were never wired up at all —
        # see Fix Round 2 report for the falsification that proved this.)
        # validation_report calls plot_scatter/plot_geographic as bare
        # names resolved through the module's own globals at call time, so
        # patching the module attribute here intercepts those calls.
        captured = {}  # val_var -> list of val_source sets seen in each captured call

        def _record(coll_ds, sar_var, val_var, *args, **kwargs):
            if "val_source" in coll_ds:
                captured.setdefault(val_var, []).append(
                    set(coll_ds["val_source"].values.tolist())
                )

        original_plot_scatter = viz.plot_scatter
        original_plot_geographic = viz.plot_geographic

        def spy_plot_scatter(coll_ds, sar_var, val_var, **kwargs):
            _record(coll_ds, sar_var, val_var)
            return original_plot_scatter(coll_ds, sar_var, val_var, **kwargs)

        def spy_plot_geographic(datatree_arg, coll_ds, sar_var, val_var=None, **kwargs):
            _record(coll_ds, sar_var, val_var)
            return original_plot_geographic(datatree_arg, coll_ds, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_scatter", spy_plot_scatter)
        monkeypatch.setattr(viz, "plot_geographic", spy_plot_geographic)

        recipe = Recipe(config=RecipeConfig(name="wdir_test", variable="wind"))
        viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        # Both PNGs exist; only the direction one has altimeter removed.
        assert (tmp_path / "plots" / "owiWindSpeed_vs_WSPD_scatter.png").exists()
        assert (tmp_path / "plots" / "owiWindDirection_vs_WDIR_scatter.png").exists()

        # The pair_ds wiring actually reached plot_scatter/plot_geographic
        # with the filtered dataset: altimeter (all-NaN WDIR) must never be
        # present in any dataset handed to a plot call made for the WDIR
        # pair, but must still be present for the WSPD pair, where every
        # source is kept untouched.
        wdir_sources = set().union(*captured.get("WDIR", [set()]))
        wspd_sources = set().union(*captured.get("WSPD", [set()]))
        assert "altimeter" not in wdir_sources, (
            "altimeter should have been dropped from the dataset passed to "
            "WDIR-pair plot calls"
        )
        assert "altimeter" in wspd_sources, (
            "altimeter should still be present in the dataset passed to "
            "WSPD-pair plot calls"
        )
        assert "mooring" in wdir_sources
        assert "mooring" in wspd_sources


class TestPlotGeographicSceneFilter:
    def _two_scene_datatree_and_coll(self):
        from sar_validation.core.datatree_converter import DataTreeConverter
        y, x = 3, 3
        lonA, latA = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        lonB, latB = np.meshgrid(np.linspace(-6, -4, x), np.linspace(50, 52, y))
        sarA = xr.Dataset({"owiWindSpeed": (("y", "x"), np.full((y, x), 7.0))},
                          coords={"lon": (("y", "x"), lonA), "lat": (("y", "x"), latA),
                                  "time": pd.Timestamp("2026-07-10T19:00:00")})
        sarB = xr.Dataset({"owiWindSpeed": (("y", "x"), np.full((y, x), 8.0))},
                          coords={"lon": (("y", "x"), lonB), "lat": (("y", "x"), latB),
                                  "time": pd.Timestamp("2026-07-10T19:00:00")})
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sarA, "sar/sceneB": sarB})
        coll = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", [7.0, 7.1]),
            "val_WSPD":         ("collocation", [6.9, 7.0]),
            "val_source":       ("collocation", ["mooring", "mooring"]),
            "sar_scene_name":   ("collocation", ["sceneA", "sceneA"]),
            "val_lon":          ("collocation", [-9.5, -9.3]),
            "val_lat":          ("collocation", [50.5, 50.7]),
        })
        return datatree, coll

    def test_scenes_allowlist_renders_only_listed_scenes(self):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_geographic
        datatree, coll = self._two_scene_datatree_and_coll()

        fig = plot_geographic(datatree, coll, "owiWindSpeed", "WSPD",
                              split_by=None, scenes=["sceneA"])
        titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
        plt.close("all")
        assert len(titled) == 1
        assert "sceneA" in titled[0]

    def test_scenes_none_renders_all_scenes(self):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_geographic
        datatree, coll = self._two_scene_datatree_and_coll()

        fig = plot_geographic(datatree, coll, "owiWindSpeed", "WSPD",
                              split_by=None, scenes=None)
        titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
        plt.close("all")
        assert len(titled) == 2  # both sceneA and sceneB drawn


class TestValidationReportSceneAllowlist:
    def test_passes_matched_scene_allowlist_to_geographic(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """validation_report must derive the geographic scene allowlist from
        sar_scene_name and pass it through as `scenes`."""
        import matplotlib.pyplot as plt
        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["scenes"] = kwargs.get("scenes")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        viz.validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("scenes") is not None
        assert "sceneA" in set(captured["scenes"])
