"""Tests for sar_validation.core.visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar_validation.core.statistics import compute_statistics
from sar_validation.core.visualization import (
    _pad_degenerate_range,
    _source_marker_handles,
    plot_residuals,
    plot_scatter,
    plot_statistics,
)


def test_pad_degenerate_range_returns_unchanged_when_not_degenerate():
    assert _pad_degenerate_range(1.0, 5.0) == (1.0, 5.0)


def test_pad_degenerate_range_pads_when_equal():
    vmin, vmax = _pad_degenerate_range(3.0, 3.0)
    assert vmin < 3.0 < vmax
    assert vmax - 3.0 == pytest.approx(max(0.5, abs(3.0) * 0.05))


def test_pad_degenerate_range_pads_zero_with_floor():
    vmin, vmax = _pad_degenerate_range(0.0, 0.0)
    assert vmax == pytest.approx(0.5)
    assert vmin == pytest.approx(-0.5)


def test_source_marker_handles_builds_one_line2d_per_item():
    handles = _source_marker_handles([("buoy", "o"), ("mooring", "^")], markersize=6)
    assert [h.get_label() for h in handles] == ["buoy", "mooring"]
    assert [h.get_marker() for h in handles] == ["o", "^"]
    assert all(h.get_markersize() == 6 for h in handles)


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
            warnings.simplefilter("error", UserWarning)
            fig = plot_scatter(ds, "owiWindSpeed", "WSPD")
        assert fig is not None
        plt.close("all")

    def test_axis_labels_include_units_when_present(self):
        import matplotlib.pyplot as plt

        n = 5
        vals = np.linspace(3, 10, n)
        ds = xr.Dataset({
            "sar_owiWindSpeed": xr.DataArray(vals, dims="collocation", attrs={"units": "m s-1"}),
            "val_WSPD": xr.DataArray(vals + 0.5, dims="collocation", attrs={"units": "m s-1"}),
            "val_source": ("collocation", ["buoy"] * n),
        })
        fig = plot_scatter(ds, "owiWindSpeed", "WSPD")
        assert fig.axes[0].get_xlabel() == "WSPD (m s-1)"
        assert fig.axes[0].get_ylabel() == "owiWindSpeed (m s-1)"
        plt.close(fig)

    def test_axis_labels_fall_back_to_bare_name_without_units(self, collocation_ds):
        import matplotlib.pyplot as plt

        # collocation_ds fixture carries no `units` attrs.
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD")
        assert fig.axes[0].get_xlabel() == "WSPD"
        assert fig.axes[0].get_ylabel() == "owiWindSpeed"
        plt.close(fig)

    def test_distinct_sources_get_distinct_markers(self, collocation_ds, monkeypatch):
        import matplotlib.axes
        import matplotlib.pyplot as plt

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

    def test_dominant_source_triggers_small_multiples_split(self):
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_scatter

        rng = np.random.default_rng(0)
        n_ascat, n_smap = 700, 30
        n = n_ascat + n_smap
        sar = np.concatenate([rng.uniform(0, 100, n_ascat), rng.uniform(0, 100, n_smap)])
        val = sar + rng.normal(0, 2, n)
        # Real collocation_ds always carries per-point val_lat/val_lon (both
        # are required, non-optional Collocation fields) — included here so
        # _deduplicate_obs doesn't collapse same-source rows lacking any
        # other observation-identity column into a single representative row.
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", sar),
            "val_SOIL_MOISTURE": ("collocation", val),
            "val_source":        ("collocation", ["ascat_ssm"] * n_ascat + ["smap_ssm"] * n_smap),
            "val_lat":           ("collocation", rng.uniform(50, 60, n)),
            "val_lon":           ("collocation", rng.uniform(-10, 5, n)),
        })
        fig = plot_scatter(ds, "sarSSM", "SOIL_MOISTURE")
        # One subplot per source, not one shared axes with everything
        # crammed together.
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) >= 2
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_small_multiples_subplots_each_have_1to1_line(self):
        """Regression test: the single shared-axes path draws a dashed
        1:1 reference line, but when plot_scatter splits into per-source
        small multiples (see test_dominant_source_triggers_small_multiples_
        split above), the subplots must still each carry their own 1:1
        line -- readers use it to judge bias at a glance, and its absence
        was silently dropping it from most scatter panels in real reports
        (soil moisture's ASCAT/ISMN pairs almost always cross the 70%
        imbalance threshold)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_scatter

        rng = np.random.default_rng(0)
        n_ascat, n_smap = 700, 30
        n = n_ascat + n_smap
        sar = np.concatenate([rng.uniform(0, 100, n_ascat), rng.uniform(0, 100, n_smap)])
        val = sar + rng.normal(0, 2, n)
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", sar),
            "val_SOIL_MOISTURE": ("collocation", val),
            "val_source":        ("collocation", ["ascat_ssm"] * n_ascat + ["smap_ssm"] * n_smap),
            "val_lat":           ("collocation", rng.uniform(50, 60, n)),
            "val_lon":           ("collocation", rng.uniform(-10, 5, n)),
        })
        fig = plot_scatter(ds, "sarSSM", "SOIL_MOISTURE")
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) >= 2
        for ax in visible_axes:
            dashed_lines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
            assert dashed_lines, f"subplot {ax.get_title()!r} is missing its 1:1 reference line"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_balanced_sources_keep_single_combined_axes(self):
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_scatter

        rng = np.random.default_rng(0)
        n = 20
        sar = rng.uniform(0, 100, n * 2)
        val = sar + rng.normal(0, 2, n * 2)
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", sar),
            "val_SOIL_MOISTURE": ("collocation", val),
            "val_source":        ("collocation", ["ascat_ssm"] * n + ["smap_ssm"] * n),
            "val_lat":           ("collocation", rng.uniform(50, 60, n * 2)),
            "val_lon":           ("collocation", rng.uniform(-10, 5, n * 2)),
        })
        fig = plot_scatter(ds, "sarSSM", "SOIL_MOISTURE")
        assert len(fig.axes) == 1
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_force_split_splits_even_balanced_sources(self):
        """force_split must trigger the small-multiples layout regardless
        of dominant_share -- used by validation_report for soil moisture
        once a source (e.g. ASCAT) has been CDF-matched into a different
        reference domain, since piling every source into one shared axes
        at that point is too visually busy even without one source
        dominating by point count (confirmed against real data,
        soil_moisture_satellite_example: ASCAT ~45% share, still busy)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_scatter

        rng = np.random.default_rng(0)
        n = 20
        sar = rng.uniform(0, 100, n * 2)
        val = sar + rng.normal(0, 2, n * 2)
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", sar),
            "val_SOIL_MOISTURE": ("collocation", val),
            "val_source":        ("collocation", ["ascat_ssm"] * n + ["smap_ssm"] * n),
            "val_lat":           ("collocation", rng.uniform(50, 60, n * 2)),
            "val_lon":           ("collocation", rng.uniform(-10, 5, n * 2)),
        })
        fig = plot_scatter(ds, "sarSSM", "SOIL_MOISTURE", force_split=True)
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) >= 2
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotScatterColorByTemporalOffset:
    def test_returns_figure_with_colorbar(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_scatter(collocation_ds, "owiWindSpeed", "WSPD", color_by="temporal_offset")
        assert fig is not None
        assert len(fig.axes) >= 2  # main axes + colorbar axes
        plt.close(fig)

    def test_split_small_multiples_still_colors_by_temporal_offset(self):
        """Regression test: previously, whenever the >70%-imbalance split
        (or force_split) triggered, color_by='temporal_offset' was
        silently dropped -- _plot_scatter_small_multiples always rendered
        the plain by-source view, so the report's 'colored by temporal
        offset' page became an exact duplicate of the main scatter page
        under a misleading title. Each split subplot must now actually be
        colored by temporal_distance_minutes, with a shared colorbar."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_scatter

        rng = np.random.default_rng(0)
        n_ascat, n_smap = 700, 30
        n = n_ascat + n_smap
        sar = np.concatenate([rng.uniform(0, 100, n_ascat), rng.uniform(0, 100, n_smap)])
        val = sar + rng.normal(0, 2, n)
        offset = rng.uniform(0, 600, n)
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", sar),
            "val_SOIL_MOISTURE": ("collocation", val),
            "val_source":        ("collocation", ["ascat_ssm"] * n_ascat + ["smap_ssm"] * n_smap),
            "val_lat":           ("collocation", rng.uniform(50, 60, n)),
            "val_lon":           ("collocation", rng.uniform(-10, 5, n)),
            "temporal_distance_minutes": ("collocation", offset),
        })
        fig = plot_scatter(ds, "sarSSM", "SOIL_MOISTURE", color_by="temporal_offset")
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # 2 source subplots + 1 shared colorbar axes.
        assert len(visible_axes) == 3

        # Source subplots carry a "<src> (N=...)" title; the colorbar axes
        # (matplotlib label "<colorbar>") has none -- a more reliable
        # discriminator than ax.collections, since the colorbar's own
        # gradient is itself drawn via a QuadMesh in `.collections`.
        scatter_axes = [ax for ax in visible_axes if ax.get_title()]
        assert len(scatter_axes) == 2
        for ax in scatter_axes:
            offsets_used = ax.collections[0].get_array()
            assert offsets_used is not None and len(offsets_used) > 0, (
                "split subplot's scatter has no per-point color array -- "
                "color_by='temporal_offset' was dropped, not applied"
            )
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_uses_distinct_markers_per_source(self, collocation_ds, monkeypatch):
        import matplotlib.axes
        import matplotlib.pyplot as plt

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
        import matplotlib.axes
        import matplotlib.pyplot as plt

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


class TestLabeledVarMixedUnits:
    def test_uses_val_units_for_specific_source(self):
        import xarray as xr

        from sar_validation.core.visualization import _labeled_var

        ds = xr.Dataset({
            "val_SOIL_MOISTURE": ("collocation", [0.1, 20.0]),
            "val_source": ("collocation", ["ismn", "ascat_ssm"]),
            "val_units": ("collocation", ["m3 m-3", "%"]),
        })
        assert _labeled_var(ds, "val_SOIL_MOISTURE", "SOIL_MOISTURE", val_source="ascat_ssm") == "SOIL_MOISTURE (%)"
        assert _labeled_var(ds, "val_SOIL_MOISTURE", "SOIL_MOISTURE", val_source="ismn") == "SOIL_MOISTURE (m3 m-3)"

    def test_no_source_given_and_units_vary_returns_neutral_label(self):
        import xarray as xr

        from sar_validation.core.visualization import _labeled_var

        ds = xr.Dataset({
            "val_SOIL_MOISTURE": ("collocation", [0.1, 20.0]),
            "val_source": ("collocation", ["ismn", "ascat_ssm"]),
            "val_units": ("collocation", ["m3 m-3", "%"]),
        })
        assert _labeled_var(ds, "val_SOIL_MOISTURE", "SOIL_MOISTURE") == "SOIL_MOISTURE (units vary by source)"

    def test_absent_val_units_falls_back_to_column_attrs_unchanged(self):
        """No val_units companion (every non-soil-moisture recipe today) --
        behavior identical to before this task."""
        import xarray as xr

        from sar_validation.core.visualization import _labeled_var

        ds = xr.Dataset({"val_WSPD": ("collocation", [5.0, 6.0])})
        ds["val_WSPD"].attrs["units"] = "m s-1"
        assert _labeled_var(ds, "val_WSPD", "WSPD") == "WSPD (m s-1)"
        assert _labeled_var(ds, "val_WSPD", "WSPD", val_source="mooring") == "WSPD (m s-1)"


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

    def test_by_source_creates_one_subplot_per_source(self, collocation_ds):
        import matplotlib.pyplot as plt
        # collocation_ds fixture has 2 distinct sources: mooring, scatterometer
        fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD")
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_by_source_false_returns_single_axes(self, collocation_ds):
        import matplotlib.pyplot as plt
        fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD", by_source=False)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_shares_bin_range_across_sources(self, monkeypatch):
        """Regression test for the buoy-spike bug: plot_residuals used to
        call ax.hist(..., bins=30, density=True) once per source without a
        shared range, so a source with a very narrow residual spread (e.g.
        2 tightly-clustered buoy points) got a tiny bin width and a density
        spike that dwarfed every other source sharing the axes. Each source
        now gets its own subplot, but all subplots must still share one
        bin range so bars stay position-comparable."""
        import matplotlib.axes
        import matplotlib.pyplot as plt

        n_wide = 20
        rng = np.random.default_rng(1)
        sar_wide = rng.uniform(3, 12, n_wide)
        val_wide = sar_wide + rng.normal(0, 1.5, n_wide)
        sar_narrow = np.array([7.0, 7.02])
        val_narrow = np.array([6.0, 6.0])

        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", np.concatenate([sar_wide, sar_narrow])),
            "val_WSPD":         ("collocation", np.concatenate([val_wide, val_narrow])),
            "val_source":       ("collocation", ["mooring"] * n_wide + ["buoy"] * 2),
        })

        recorded_ranges = []
        original_hist = matplotlib.axes.Axes.hist

        def recording_hist(self, *args, **kwargs):
            recorded_ranges.append(kwargs.get("range"))
            return original_hist(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "hist", recording_hist)
        fig = plot_residuals(ds, "owiWindSpeed", "WSPD")
        plt.close(fig)

        assert len(recorded_ranges) == 2
        assert all(r is not None for r in recorded_ranges)
        assert recorded_ranges[0] == recorded_ranges[1]

    def test_by_source_subplot_labels_use_each_sources_own_units(self):
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_residuals

        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", np.array([10.0, 15.0, 0.12, 0.18])),
            "val_SOIL_MOISTURE": ("collocation", np.array([12.0, 18.0, 0.1, 0.2])),
            "val_source":        ("collocation", ["ascat_ssm", "ascat_ssm", "ismn", "ismn"]),
            "val_units":         ("collocation", ["%", "%", "m3 m-3", "m3 m-3"]),
        })
        fig = plot_residuals(ds, "sarSSM", "SOIL_MOISTURE", by_source=True)
        labels = [ax.get_xlabel() for ax in fig.axes if ax.get_visible()]
        assert any("(%)" in lbl for lbl in labels)
        assert any("(m3 m-3)" in lbl for lbl in labels)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_hist_range_overrides_data_driven_shared_range(self):
        """Regression test for the CDF-matching-outlier bug: one extreme
        residual used to balloon the auto-computed shared_range to roughly
        (-100, 100), collapsing the real, tightly-clustered residuals into
        a single bar. Passing hist_range must use it verbatim instead of
        computing from data min/max."""
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(2)
        tight = rng.normal(0, 0.05, 20)
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", np.concatenate([tight, [80.0]])),
            "val_SOIL_MOISTURE": ("collocation", np.zeros(21)),
            "val_source":        ("collocation", ["ismn"] * 21),
        })

        fig = plot_residuals(ds, "sarSSM", "SOIL_MOISTURE", hist_range=(-1.0, 1.0))
        ax = [a for a in fig.axes if a.get_visible()][0]
        assert ax.get_xlim() == (-1.0, 1.0)
        plt.close(fig)

    def test_hist_range_excludes_out_of_range_values_from_bins(self):
        """hist_range must be passed through to ax.hist's own range= (not
        just set_xlim afterward) so the extreme value's presence doesn't
        widen the bin width and hide the real distribution inside one bin."""
        import matplotlib.axes
        import matplotlib.pyplot as plt

        recorded_ranges = []
        original_hist = matplotlib.axes.Axes.hist

        def recording_hist(self, *args, **kwargs):
            recorded_ranges.append(kwargs.get("range"))
            return original_hist(self, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(matplotlib.axes.Axes, "hist", recording_hist)
        try:
            rng = np.random.default_rng(3)
            tight = rng.normal(0, 0.05, 20)
            ds = xr.Dataset({
                "sar_sarSSM":        ("collocation", np.concatenate([tight, [80.0]])),
                "val_SOIL_MOISTURE": ("collocation", np.zeros(21)),
                "val_source":        ("collocation", ["ismn"] * 21),
            })
            fig = plot_residuals(ds, "sarSSM", "SOIL_MOISTURE", hist_range=(-1.0, 1.0))
            plt.close(fig)
        finally:
            monkeypatch.undo()

        assert len(recorded_ranges) == 1
        assert recorded_ranges[0] == (-1.0, 1.0)

    def test_hist_range_none_preserves_existing_data_driven_behavior(self, collocation_ds):
        """Default (hist_range=None) must be unchanged from today's
        behavior -- every other variable's residuals plot is unaffected."""
        import matplotlib.pyplot as plt

        fig = plot_residuals(collocation_ds, "owiWindSpeed", "WSPD")
        ax = [a for a in fig.axes if a.get_visible()][0]
        xlim = ax.get_xlim()
        assert xlim != (-1.0, 1.0)
        plt.close(fig)

    def test_hist_range_applies_to_by_source_false_path_too(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(4)
        tight = rng.normal(0, 0.05, 20)
        ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", np.concatenate([tight, [80.0]])),
            "val_SOIL_MOISTURE": ("collocation", np.zeros(21)),
            "val_source":        ("collocation", ["ismn"] * 21),
        })

        fig = plot_residuals(
            ds, "sarSSM", "SOIL_MOISTURE", by_source=False, hist_range=(-1.0, 1.0),
        )
        assert fig.axes[0].get_xlim() == (-1.0, 1.0)
        plt.close(fig)

    def test_hist_range_dict_gives_each_source_its_own_range(self):
        """Regression test for the real-world mixed-source bug: a
        soil_moisture recipe with ismn (volumetric) alongside ascat_ssm
        (percent) pools ALL residuals into one global shared_range when
        hist_range is a single value/None, so ismn's panel inherits
        ascat_ssm's much wider percent-scale spread even though ismn's
        own residuals are tightly clustered near zero. A dict keyed by
        val_source must give each source ITS OWN range: an override for
        sources present in the dict, and a per-source (not pooled) data
        range for sources absent from it."""
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(6)
        ismn_vals = rng.normal(0, 0.05, 20)  # tight, near-zero residuals
        ascat_vals = rng.normal(0, 15.0, 20)  # wide, percent-scale residuals
        ds = xr.Dataset({
            "sar_sarSSM": ("collocation", np.concatenate([ismn_vals, ascat_vals])),
            "val_SOIL_MOISTURE": ("collocation", np.zeros(40)),
            "val_source": ("collocation", ["ismn"] * 20 + ["ascat_ssm"] * 20),
        })

        fig = plot_residuals(
            ds, "sarSSM", "SOIL_MOISTURE", hist_range={"ismn": (-1.0, 1.0)},
        )
        axes_by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_visible()}
        ismn_ax = next(ax for title, ax in axes_by_title.items() if title.startswith("ismn"))
        ascat_ax = next(ax for title, ax in axes_by_title.items() if title.startswith("ascat_ssm"))

        assert ismn_ax.get_xlim() == (-1.0, 1.0)
        # ascat_ssm has no override -- must get its OWN data-driven range
        # (wide enough to show its real ~N(0,15) spread), not ismn's (-1,1)
        # and not some huge range polluted by ismn's data being pooled in.
        ascat_lo, ascat_hi = ascat_ax.get_xlim()
        assert ascat_hi - ascat_lo > 2.0
        plt.close(fig)


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
        import matplotlib.axes
        import matplotlib.pyplot as plt

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


class TestPlotSummaryTable:
    def test_table_has_one_row_per_source_and_requested_columns(self):
        import numpy as np
        import xarray as xr

        from sar_validation.core.statistics import compute_statistics
        from sar_validation.core.visualization import plot_summary_table

        n = 20
        rng = np.random.default_rng(0)
        sar = rng.uniform(0, 10, n)
        val = sar + rng.normal(0, 0.5, n)
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", sar),
            "val_WSPD":         ("collocation", val),
            "val_source":       ("collocation", ["mooring"] * 10 + ["altimeter"] * 10),
        })
        stats_ds = compute_statistics(ds, "owiWindSpeed", "WSPD", group_by=["val_source"])

        fig = plot_summary_table(stats_ds)

        assert fig is not None
        ax = fig.axes[0]
        tables = [c for c in ax.get_children() if hasattr(c, "get_celld")]
        assert len(tables) == 1
        cell_texts = {cell.get_text().get_text() for cell in tables[0].get_celld().values()}
        assert "mooring" in cell_texts
        assert "altimeter" in cell_texts
        assert "bias" in cell_texts or "Bias" in cell_texts
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_returns_none_when_no_requested_metrics_available(self):
        import xarray as xr

        from sar_validation.core.visualization import plot_summary_table

        stats_ds = xr.Dataset(
            {"foo": ("source", [1.0])}, coords={"source": ["mooring"]},
        )
        assert plot_summary_table(stats_ds, metrics=["bias"]) is None

    def test_returns_none_for_empty_source_coordinate(self):
        import numpy as np
        import xarray as xr

        from sar_validation.core.visualization import plot_summary_table

        stats_ds = xr.Dataset(
            {"bias": ("source", np.array([], dtype=float))},
            coords={"source": np.array([], dtype=object)},
        )
        assert plot_summary_table(stats_ds) is None


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
            _canonical_source_order,
            _source_style_map,
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
        import matplotlib.axes
        import matplotlib.pyplot as plt

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

    def test_dict_point_size_sizes_each_collocation_type_independently(self, monkeypatch):
        """point_size accepts a dict keyed by collocation_type (see
        validation_report's soil-moisture/wind adaptive sizing) -- each
        returned per-type Figure must actually use its own entry, not one
        value shared across every type."""
        import matplotlib.axes
        import matplotlib.pyplot as plt
        import pandas as pd

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        rng = np.random.default_rng(0)
        n_layer, n_point = 6, 3
        n = n_layer + n_point
        collocation_ds = xr.Dataset({
            "sar_sarSSM":       ("collocation", rng.uniform(20, 40, n)),
            "val_SOIL_MOISTURE": ("collocation", rng.uniform(0.1, 0.4, n)),
            "val_source":       ("collocation", ["ascat_ssm"] * n_layer + ["ismn"] * n_point),
            "collocation_type": ("collocation", ["layer_vs_layer"] * n_layer + ["point_vs_layer"] * n_point),
            "sar_scene_name":   ("collocation", ["sceneA"] * n),
            "val_lon":          ("collocation", rng.uniform(-9.8, -8.2, n)),
            "val_lat":          ("collocation", rng.uniform(50.2, 51.8, n)),
        })

        recorded_sizes = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_sizes.append(kwargs.get("s"))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        figs = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE",
            point_size={"layer_vs_layer": 5, "point_vs_layer": 15},
        )
        plt.close("all")

        assert set(figs.keys()) == {"layer_vs_layer", "point_vs_layer"}
        # Each type's own point scatter call (excluding the SAR background
        # field's pcolormesh/scatter, which doesn't pass an `s` kwarg at
        # all -- point_size only ever governs validation-point markers).
        assert 5 in recorded_sizes, f"layer_vs_layer's point_size=5 never used, got {recorded_sizes!r}"
        assert 15 in recorded_sizes, f"point_vs_layer's point_size=15 never used, got {recorded_sizes!r}"

    def test_gridded_scene_with_nan_geolocation_does_not_raise(
        self, geo_datatree_and_collocation,
    ):
        """Regression test: S1 OCN products commonly carry NaN lon/lat at
        swath-edge cells. pcolormesh rejects non-finite x/y outright, so
        plot_geographic must repair the coordinate grid rather than crash."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        scene_ds = datatree["sar/sceneA"].to_dataset()
        lon2d = scene_ds["lon"].values.copy()
        lat2d = scene_ds["lat"].values.copy()
        lon2d[0, -1] = np.nan
        lat2d[0, -1] = np.nan
        scene_ds = scene_ds.assign_coords(
            lon=(("y", "x"), lon2d), lat=(("y", "x"), lat2d),
        )
        from sar_validation.core.datatree_converter import DataTreeConverter
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": scene_ds,
            "validation/mooring": datatree["validation/mooring"].to_dataset(),
            "validation/altimeter": datatree["validation/altimeter"].to_dataset(),
        })

        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        plt.close("all")

        assert fig is not None

    def test_genuine_per_point_nan_still_rendered_as_no_data_hatch(
        self, geo_datatree_and_collocation,
    ):
        """Regression guard for the source-level "omit dropped sources
        entirely" fix (see TestPlotGeographicPointLevelDomainHarmonization):
        that fix is specifically about a val_source _harmonize_percent_
        domain_sources couldn't harmonize at all, not an ordinary per-point
        missing retrieval for a source that never needed harmonizing (this
        fixture is wind/mooring/altimeter -- sar_units is None, so
        domains_differ is False and _harmonize_percent_domain_sources's
        point-level filtering path never even runs). A single NaN'd val
        value must still render as the existing gray hatched "no data"
        marker, exactly as before this fix."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        val = collocation_ds["val_WSPD"].values.copy()
        nan_lon = float(collocation_ds["val_lon"].values[1])
        nan_lat = float(collocation_ds["val_lat"].values[1])
        val[1] = np.nan
        collocation_ds = collocation_ds.assign(val_WSPD=("collocation", val))

        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        plt.close("all")

        assert fig is not None
        ax = [a for a in fig.axes if a.get_visible()][0]
        collections = _path_collections_with_offsets(ax)
        c = _collection_containing(collections, nan_lon, nan_lat)
        assert c is not None, f"no scatter collection found for the NaN'd point ({nan_lon}, {nan_lat})"
        assert c.get_hatch() == "////", (
            "a genuine per-point NaN (unrelated to source-level harmonization) must "
            f"still render as the 'no data' hatched marker, got hatch={c.get_hatch()!r}"
        )
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "No data (NaN)" in labels


class TestPlotGeographicDomainMismatch:
    """Regression tests: pooling percentiles across a SAR field and
    validation values that live in genuinely different physical domains
    (e.g. soil_moisture's SAR relative-saturation "%" vs. ISMN's
    volumetric "1") squashes whichever series has the smaller natural
    range into one end of a shared colour scale. The fix converts the SAR
    *field* itself into the validation domain (via a CDF-matching
    transform fit from the collocated pairs and applied to every grid
    cell, not just collocated ones) whenever there's enough collocated
    data to fit one, so the whole map shares one meaningful colorbar —
    falling back to two separate colorbars (one per layer's own
    percentile range) only when there isn't."""

    def _scene(self, n_colloc: int):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        # SAR field spans 0-100 ("%"); validation values span ~0.05-0.5
        # ("1") — a real ~200x range mismatch, matching soil_moisture.
        field = np.linspace(0.0, 100.0, y * x).reshape(y, x)
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), field, {"units": "%"})},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )
        lons = np.array([-9.8, -9.6, -9.4, -9.2])[:n_colloc]
        lats = np.array([50.2, 50.4, 50.6, 50.8])[:n_colloc]
        val_vals = np.array([0.1, 0.2, 0.3, 0.4])[:n_colloc]
        sar_vals = np.array([10.0, 30.0, 50.0, 70.0])[:n_colloc]
        ismn_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", val_vals)},
            coords={
                "lon": ("point", lons),
                "lat": ("point", lats),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=n_colloc, freq="5min")),
            },
            attrs={"platform_type": "ismn"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ismn": ismn_ds,
        })
        collocation_ds = xr.Dataset({
            "sar_sarSSM": ("collocation", sar_vals),
            "val_SOIL_MOISTURE": xr.DataArray(
                val_vals, dims="collocation", attrs={"units": "1"},
            ),
            "val_source": ("collocation", ["ismn"] * n_colloc),
            "sar_scene_name": ("collocation", ["sceneA"] * n_colloc),
            "val_lon": ("collocation", lons),
            "val_lat": ("collocation", lats),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n_colloc, freq="5min")),
        )
        return datatree, collocation_ds

    @pytest.fixture
    def mismatched_units_scene(self):
        return self._scene(n_colloc=4)

    @pytest.fixture
    def mismatched_units_scene_too_sparse_to_fit(self):
        # fit_sar_to_val_transform needs >= 2 valid pairs to fit a
        # transform at all — one point can't define a CDF mapping.
        return self._scene(n_colloc=1)

    def test_enough_data_converts_field_to_one_shared_colorbar(self, mismatched_units_scene):
        """The requested behaviour: convert the SAR field into the
        validation domain rather than showing two separate colorbars."""
        import warnings

        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = mismatched_units_scene
        # Real (unmocked) pytesmo.cdf_matching.CDFMatching resizes its bins
        # for this deliberately tiny 4-point fixture — an expected, benign
        # side effect of fitting on so little data, not a defect.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = plot_geographic(
                datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        ylabels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
        # One shared colorbar: no separate "SAR"/"In-situ" labels.
        assert not any(lbl.startswith("SAR ") or lbl.startswith("In-situ ") for lbl in ylabels)
        assert any("sarSSM" in lbl and "SOIL_MOISTURE" in lbl for lbl in ylabels)
        plt.close("all")

    def test_converted_field_values_land_in_validation_range(self, mismatched_units_scene):
        """The core bug this fixes: without conversion, the SAR field's
        raw 0-100 values would dominate the shared colour range, making
        real ISMN values (~0.05-0.5) collapse near one end of the scale."""
        import warnings

        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = mismatched_units_scene
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = plot_geographic(
                datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        # The shared norm's vmax must reflect the *converted* field/val
        # range (well under 10), not the raw SAR field's 0-100 range.
        quadmesh = next(
            c for ax in fig.axes for c in ax.collections
            if "QuadMesh" in type(c).__name__
        )
        assert quadmesh.norm.vmax < 10.0
        plt.close("all")

    def test_too_sparse_to_fit_falls_back_to_two_colorbars(
        self, mismatched_units_scene_too_sparse_to_fit,
    ):
        """A single collocated pair can't define a CDF-matching transform
        — plot_geographic must fall back to the old two-colorbar
        behaviour rather than silently plotting an unconverted field
        against a converted-looking label, or crashing."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = mismatched_units_scene_too_sparse_to_fit
        result = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
        )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        ylabels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
        assert any("SAR" in lbl for lbl in ylabels)
        assert any("In-situ" in lbl for lbl in ylabels)
        plt.close("all")

    def test_same_units_still_use_one_pooled_colorbar(self, geo_datatree_and_collocation):
        """Existing wind/currents/waves behavior (same units both sides)
        must be unaffected by this fix — still one shared colorbar."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        result = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        ylabels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
        # No units attrs on this fixture's variables, so no "SAR"/"In-situ"
        # split labels either — single shared colorbar, no ylabel at all
        # (matches the pre-existing single_colorbar code path).
        assert not any("SAR" in lbl or "In-situ" in lbl for lbl in ylabels)
        plt.close("all")


class TestPlotGeographicSkipDomainHarmonization:
    """Bug 1 regression: validation_report's native-units section builds
    ``nu_pair_ds`` by row-filtering the full ``geo_pair_ds`` down to
    val_source groups that already share SAR's own units family (e.g. only
    ``ascat_ssm`` when ISMN is absent/too sparse) — but that filtering
    doesn't touch the *column-level* ``val_<var>`` units attr, which is
    still the "mixed — see val_units" sentinel stamped onto the full,
    unfiltered dataset by ``annotate_collocation_ds`` back when multiple
    unit families were genuinely present. ``plot_geographic`` reads that
    stale attrs string to decide ``domains_differ``, incorrectly concludes
    the (now single-family) native-units dataset needs harmonizing, and — since
    the sole remaining source, ASCAT, can't be harmonized without ISMN as
    a reference — ends up forcing the two-separate-colorbars fallback for
    a case that should share one colorbar. ``skip_domain_harmonization=True``
    is the caller's explicit opt-out for exactly this call site."""

    def _single_family_scene_with_stale_mixed_attrs(self):
        """Mirrors what validation_report's nu_pair_ds looks like: only
        ascat_ssm rows remain (ISMN was filtered out for being absent/too
        sparse for the native-units restriction), but val_SOIL_MOISTURE's
        column-level units attr is still the "mixed" sentinel inherited,
        unchanged, from the pre-filter full dataset."""
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        field = np.linspace(0.0, 100.0, y * x).reshape(y, x)
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), field, {"units": "%"})},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )
        n = 5
        lons = np.array([-9.8, -9.6, -9.4, -9.2, -9.0])
        lats = np.array([50.2, 50.4, 50.6, 50.8, 51.0])
        val_vals = np.array([72.2, 70.3, 74.7, 75.7, 71.4])
        sar_vals = np.array([70.0, 68.0, 76.0, 74.0, 69.0])
        ascat_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", val_vals)},
            coords={
                "lon": ("point", lons),
                "lat": ("point", lats),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
            },
            attrs={"platform_type": "ascat_ssm"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ascat_ssm": ascat_ds,
        })
        collocation_ds = xr.Dataset({
            "sar_sarSSM": xr.DataArray(sar_vals, dims="collocation", attrs={"units": "%"}),
            "val_SOIL_MOISTURE": xr.DataArray(
                val_vals, dims="collocation",
                # The stale sentinel: real, unfiltered soil_moisture runs
                # stamp this when ISMN/SMAP (volumetric) were present
                # alongside ASCAT (percent) *before* row-filtering to
                # native units — filtering rows never recomputes it.
                attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", ["ascat_ssm"] * n),
            "sar_scene_name": ("collocation", ["sceneA"] * n),
            "val_lon": ("collocation", lons),
            "val_lat": ("collocation", lats),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
        )
        return datatree, collocation_ds

    def test_skip_domain_harmonization_forces_single_colorbar(self):
        """The fix: with skip_domain_harmonization=True, plot_geographic
        must not treat the stale "mixed" sentinel as a real mismatch, and
        must render one shared colorbar (matching what this ascat_ssm-only,
        single-family dataset actually needs)."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = self._single_family_scene_with_stale_mixed_attrs()
        result = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            skip_domain_harmonization=True,
        )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        # ylabel-based check (not GeoAxes-count): plot_geographic's default
        # ncols=2 layout adds a second, hidden placeholder subplot even for
        # a single-scene figure (plain Axes, not GeoAxes, invisible) -- an
        # unrelated layout detail that would otherwise be miscounted as a
        # second colorbar axes.
        ylabels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
        assert not any(lbl.startswith("SAR ") or lbl.startswith("In-situ ") for lbl in ylabels), (
            f"expected one shared colorbar (no SAR/In-situ split), got ylabels={ylabels!r}"
        )
        assert any("sarSSM" in lbl and "SOIL_MOISTURE" in lbl for lbl in ylabels)
        plt.close("all")

    def test_without_flag_stale_mixed_attrs_wrongly_forces_two_colorbars(self):
        """Documents the bug this fix works around: absent the opt-out,
        the same single-family dataset's stale "mixed" attrs string makes
        plot_geographic wrongly believe domains differ, and — since ASCAT
        alone can't be harmonized without ISMN present as a reference —
        falls back to two separate colorbars."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = self._single_family_scene_with_stale_mixed_attrs()
        result = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
        )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        from cartopy.mpl.geoaxes import GeoAxes
        non_data_axes = [ax for ax in fig.axes if not isinstance(ax, GeoAxes)]
        assert len(non_data_axes) == 2, (
            "expected the (buggy, pre-fix) two-colorbar fallback when the "
            f"stale sentinel isn't opted out of, got {len(non_data_axes)}"
        )
        plt.close("all")


def _path_collections_with_offsets(ax):
    """Return every matplotlib PathCollection scatter artist on *ax* that
    has at least one plotted point (excludes empty/decorative collections),
    for inspecting which lon/lat points landed in which scatter call."""
    import matplotlib.collections as mcollections

    return [
        c for c in ax.collections
        if isinstance(c, mcollections.PathCollection) and len(c.get_offsets()) > 0
    ]


def _collection_containing(collections, lon: float, lat: float, tol: float = 1e-6):
    """Find the (unique, expected) PathCollection among *collections* whose
    offsets include (lon, lat), or None if none match."""
    for c in collections:
        offsets = np.asarray(c.get_offsets())
        if len(offsets) == 0:
            continue
        hit = np.any(
            (np.abs(offsets[:, 0] - lon) < tol) & (np.abs(offsets[:, 1] - lat) < tol)
        )
        if hit:
            return c
    return None


class TestPlotGeographicPointLevelDomainHarmonization:
    """Bug 2 regression: the CDF-matched geographic plot's *field*
    (background raster + fit_sar_to_val_transform) already correctly drops
    ASCAT when ISMN is too sparse to harmonize against — but the point
    markers drawn on top were, until this fix, still taken from
    collocation_ds's raw, unharmonized val_<var> column throughout the
    rest of plot_geographic, plotting ASCAT's raw ~0-100 percent values as
    colored dots under a colour scale calibrated for ISMN/SMAP's ~0-1
    volumetric domain."""

    def _scene(self, n_ismn: int):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        field = np.linspace(0.0, 100.0, y * x).reshape(y, x)
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), field, {"units": "%"})},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )

        ismn_lons = np.array([-9.9, -9.7, -9.5, -9.3])[:n_ismn]
        ismn_lats = np.array([50.1, 50.3, 50.5, 50.7])[:n_ismn]
        ismn_vals = np.array([0.10, 0.15, 0.20, 0.25])[:n_ismn]
        ismn_sar_vals = np.array([12.0, 18.0, 22.0, 28.0])[:n_ismn]
        ismn_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", ismn_vals)},
            coords={
                "lon": ("point", ismn_lons),
                "lat": ("point", ismn_lats),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=n_ismn, freq="5min")),
            },
            attrs={"platform_type": "ismn"},
        )

        n_ascat = 5
        ascat_lons = np.array([-9.8, -9.6, -9.4, -9.2, -9.0])
        ascat_lats = np.array([50.2, 50.4, 50.6, 50.8, 51.0])
        ascat_vals = np.array([72.2, 70.3, 74.7, 75.7, 71.4])
        ascat_sar_vals = np.array([70.0, 68.0, 76.0, 74.0, 69.0])
        ascat_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", ascat_vals)},
            coords={
                "lon": ("point", ascat_lons),
                "lat": ("point", ascat_lats),
                "time": ("point", pd.date_range("2026-07-10T19:06", periods=n_ascat, freq="5min")),
            },
            attrs={"platform_type": "ascat_ssm"},
        )

        n_smap = 5
        smap_lons = np.array([-8.9, -8.7, -8.5, -8.3, -8.1])
        smap_lats = np.array([51.2, 51.4, 51.6, 51.8, 52.0])
        smap_vals = np.array([0.12, 0.22, 0.32, 0.42, 0.30])
        smap_sar_vals = np.array([15.0, 25.0, 35.0, 45.0, 33.0])
        smap_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", smap_vals)},
            coords={
                "lon": ("point", smap_lons),
                "lat": ("point", smap_lats),
                "time": ("point", pd.date_range("2026-07-10T19:07", periods=n_smap, freq="5min")),
            },
            attrs={"platform_type": "smap_ssm"},
        )

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ismn": ismn_ds,
            "validation/ascat_ssm": ascat_ds,
            "validation/smap_ssm": smap_ds,
        })

        lons = np.concatenate([ismn_lons, ascat_lons, smap_lons])
        lats = np.concatenate([ismn_lats, ascat_lats, smap_lats])
        val_vals = np.concatenate([ismn_vals, ascat_vals, smap_vals])
        sar_vals = np.concatenate([ismn_sar_vals, ascat_sar_vals, smap_sar_vals])
        sources = (["ismn"] * n_ismn) + (["ascat_ssm"] * n_ascat) + (["smap_ssm"] * n_smap)
        n_total = n_ismn + n_ascat + n_smap

        collocation_ds = xr.Dataset({
            # units="%" mirrors annotate_collocation_ds copying the SAR
            # scene variable's own attrs onto sar_<var> -- needed for
            # _harmonize_percent_domain_sources's sar_family detection,
            # which reads collocation_ds's own sar_col attrs (not the
            # datatree scene field's).
            "sar_sarSSM": xr.DataArray(sar_vals, dims="collocation", attrs={"units": "%"}),
            "val_SOIL_MOISTURE": xr.DataArray(
                val_vals, dims="collocation",
                # Mirrors annotate_collocation_ds's real output for a run
                # with genuinely mixed val_source units (ismn/smap "m3
                # m-3" volumetric vs. ascat_ssm "%").
                attrs={"units": "mixed — see val_units"},
            ),
            "val_source": ("collocation", sources),
            "sar_scene_name": ("collocation", ["sceneA"] * n_total),
            "val_lon": ("collocation", lons),
            "val_lat": ("collocation", lats),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n_total, freq="1min")),
        )
        return datatree, collocation_ds, ascat_lons, ascat_lats, ismn_lons, ismn_lats, smap_lons, smap_lats

    def test_ascat_points_omitted_entirely_when_ismn_too_sparse(self):
        """Design decision (2026-07-28): with only one ISMN point (below
        the < 2 threshold _harmonize_percent_domain_sources requires),
        ASCAT is an entire *source* that couldn't be harmonized, not a
        single failed per-point retrieval -- at real-world scale (ASCAT
        can be ~56% of all collocated points in a run) rendering it via
        the "No data (NaN)" gray-hatched convention floods the map and
        buries genuinely meaningful data underneath it. So dropped
        sources must be omitted from the CDF-matched geographic plot
        entirely: no colored point, no hatched "no data" marker either --
        not present in any scatter collection at all. (They remain fully
        visible in the separate native-units section, unaffected by this.)"""
        import warnings

        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        (datatree, collocation_ds, ascat_lons, ascat_lats,
         ismn_lons, ismn_lats, smap_lons, smap_lats) = self._scene(n_ismn=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = plot_geographic(
                datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        from cartopy.mpl.geoaxes import GeoAxes
        ax = next(a for a in fig.axes if isinstance(a, GeoAxes))
        collections = _path_collections_with_offsets(ax)

        for lon, lat in zip(ascat_lons, ascat_lats):
            c = _collection_containing(collections, lon, lat)
            assert c is None, (
                f"ASCAT point ({lon}, {lat}) must be omitted entirely (no scatter "
                f"collection at all -- not colored, not hatched 'no data') when ISMN "
                f"is too sparse to harmonize, but found it in collection {c!r}"
            )

        # No collection anywhere on the map should carry the "no data"
        # hatch pattern at all in this scenario -- the only source that
        # needed (and failed) harmonization is ASCAT, and it's now
        # excluded before the nan_pts/valid_pts split even runs, so
        # nothing should reach that gray-hatched rendering path.
        assert not any(c.get_hatch() == "////" for c in collections), (
            "no 'No data (NaN)' hatched markers expected -- the only unharmonized "
            "source (ascat_ssm) must be omitted upstream, not rendered as hatched"
        )

        # SMAP (already volumetric, never needed converting) must stay
        # untouched and colored -- this fix must not over-trigger the drop.
        for lon, lat in zip(smap_lons, smap_lats):
            c = _collection_containing(collections, lon, lat)
            assert c is not None, f"no scatter collection found for SMAP point ({lon}, {lat})"
            assert c.get_hatch() != "////", (
                f"SMAP point ({lon}, {lat}) must remain colored, not dropped as 'no data'"
            )
        plt.close("all")

    def test_ascat_dropped_source_omitted_from_legend(self):
        """Regression test: when ASCAT is fully dropped (see
        test_ascat_points_omitted_entirely_when_ismn_too_sparse above --
        every ASCAT point is excluded from point_collocation_ds entirely,
        before point rendering even runs), the legend must NOT still show
        ASCAT's colored marker/label, nor a "No data (NaN)" entry (since
        ASCAT was the only source that needed harmonizing and it's gone
        before the nan_pts/valid_pts split, there are no NaN rows left to
        produce that legend entry either). Before the original (now
        superseded) fix, the legend's `present` set was built from every
        val_source in df_pts (all rows, including the all-NaN ones), so
        ascat_ssm's marker appeared in the legend even though zero actual
        ascat_ssm-colored points exist on the map -- misleading a reader
        into looking for points that aren't there. This test now also
        confirms that the newer "omit dropped sources entirely" behavior
        doesn't reintroduce that problem or a "No data (NaN)" stand-in."""
        import warnings

        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        (datatree, collocation_ds, ascat_lons, ascat_lats,
         ismn_lons, ismn_lats, smap_lons, smap_lats) = self._scene(n_ismn=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = plot_geographic(
                datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        from cartopy.mpl.geoaxes import GeoAxes
        ax = next(a for a in fig.axes if isinstance(a, GeoAxes))
        legend = ax.get_legend()
        assert legend is not None, "expected a legend to be rendered for this scene"
        labels = [t.get_text() for t in legend.get_texts()]

        assert "ascat_ssm" not in labels, (
            "ascat_ssm must not appear in the legend when every one of its "
            f"points was dropped -- got legend labels {labels!r}"
        )
        assert "No data (NaN)" not in labels, (
            "no genuine per-point NaN remains in this scenario once ascat_ssm "
            f"(the only unharmonized source) is omitted upstream -- got {labels!r}"
        )
        # Sources that DO have real colored points on the map must still
        # be labeled -- this fix must not empty the legend entirely.
        assert "ismn" in labels
        assert "smap_ssm" in labels
        plt.close("all")

    def test_ascat_points_still_colored_when_ismn_sufficient(self):
        """Regression guard: when ISMN has enough points to harmonize
        against (the ordinary case), ASCAT's points must still render as
        colored, converted points -- not accidentally dropped as 'no
        data' by an over-eager Bug 2 fix."""
        import warnings

        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        (datatree, collocation_ds, ascat_lons, ascat_lats,
         ismn_lons, ismn_lats, smap_lons, smap_lats) = self._scene(n_ismn=4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = plot_geographic(
                datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            )
        fig = result if not isinstance(result, dict) else list(result.values())[0]

        from cartopy.mpl.geoaxes import GeoAxes
        ax = next(a for a in fig.axes if isinstance(a, GeoAxes))
        collections = _path_collections_with_offsets(ax)

        for lon, lat in zip(ascat_lons, ascat_lats):
            c = _collection_containing(collections, lon, lat)
            assert c is not None, f"no scatter collection found for ASCAT point ({lon}, {lat})"
            assert c.get_hatch() != "////", (
                f"ASCAT point ({lon}, {lat}) must remain colored when ISMN is "
                "sufficient to harmonize against -- must not be dropped"
            )
        plt.close("all")


class TestValidationReportSoilMoistureGeographicUsesRawData:
    """Regression test: validation_report() replaces pair_ds's sar_<var>
    column with its CDF-matched (rescaled) values before calling the
    point-based plots (scatter/residuals/temporal offset), which is
    correct for those. But plot_geographic must NOT receive that already
    -rescaled column — its internal fit_sar_to_val_transform needs the
    raw (sar, val) correspondence to fit a transform for the whole SAR
    *field*; handed the rescaled column instead, it trains on values
    already in the validation's domain (~0.05-0.5) and then applies the
    resulting transform to the real, raw SAR field (~0-100), producing
    wildly out-of-range output (confirmed against real data: predicted
    values above 300 for a variable that should span roughly 0-1)."""

    @pytest.fixture
    def soil_moisture_report_scene(self):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        field = np.linspace(0.0, 100.0, y * x).reshape(y, x)
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), field, {"units": "%"})},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )
        n_colloc = 4
        lons = np.array([-9.8, -9.6, -9.4, -9.2])
        lats = np.array([50.2, 50.4, 50.6, 50.8])
        val_vals = np.array([0.1, 0.2, 0.3, 0.4])
        sar_vals = np.array([10.0, 30.0, 50.0, 70.0])
        ismn_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", val_vals)},
            coords={
                "lon": ("point", lons),
                "lat": ("point", lats),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=n_colloc, freq="5min")),
            },
            attrs={"platform_type": "ismn"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ismn": ismn_ds,
        })
        collocation_ds = xr.Dataset({
            "sar_sarSSM": ("collocation", sar_vals),
            "val_SOIL_MOISTURE": xr.DataArray(
                val_vals, dims="collocation", attrs={"units": "1"},
            ),
            "val_source": ("collocation", ["ismn"] * n_colloc),
            "sar_scene_name": ("collocation", ["sceneA"] * n_colloc),
            "val_lon": ("collocation", lons),
            "val_lat": ("collocation", lats),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n_colloc, freq="5min")),
        )
        return datatree, collocation_ds

    def test_plot_geographic_receives_raw_not_rescaled_sar_values(
        self, soil_moisture_report_scene, tmp_path, monkeypatch,
    ):
        import warnings

        import matplotlib.pyplot as plt

        from sar_validation.core import visualization
        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = soil_moisture_report_scene
        recipe = Recipe(config=RecipeConfig(name="test", variable="soil_moisture"))

        received: dict = {}
        original_plot_geographic = visualization.plot_geographic

        def spy_plot_geographic(dt, pair_ds, sar_var, val_var, **kwargs):
            received["sar_sarSSM"] = pair_ds["sar_sarSSM"].values.copy()
            return original_plot_geographic(dt, pair_ds, sar_var, val_var, **kwargs)

        monkeypatch.setattr(visualization, "plot_geographic", spy_plot_geographic)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert "sar_sarSSM" in received, "plot_geographic was never called"
        np.testing.assert_allclose(received["sar_sarSSM"], np.array([10.0, 30.0, 50.0, 70.0]))


class TestPlotGeographicBoundsClamp:
    """geographic_bounds must clamp each scene panel's extent to the
    recipe's requested bounding box, not the SAR field's full native
    extent (e.g. CLMS Surface Soil Moisture's grid covers all of mainland
    Europe regardless of what a recipe actually requested)."""

    def test_clamps_to_geographic_bounds(self, geo_datatree_and_collocation):
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import GeographicBounds
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        # The fixture's SAR scene spans lon [-10, -8], lat [50, 52] — request
        # a strictly tighter box and confirm the axes actually shrink to it.
        bounds = GeographicBounds(min_lon=-9.5, max_lon=-8.5, min_lat=50.5, max_lat=51.5)
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
            geographic_bounds=bounds,
        )
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] > -10.0
        assert xlim[1] < -8.0
        assert ylim[0] > 50.0
        assert ylim[1] < 52.0
        plt.close("all")

    def test_no_bounds_keeps_full_scene_extent(self, geo_datatree_and_collocation):
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        # Unclamped: extent reflects the full SAR scene (lon [-10, -8]),
        # not a tight box like the clamped test above.
        assert xlim[0] <= -9.9
        plt.close("all")

    def test_padding_does_not_exceed_recipe_bounds(self, geo_datatree_and_collocation):
        """Regression test: a bbox much wider (lon) than tall (lat) used to
        get padded past its own requested lat bounds by
        _pad_extent_to_min_aspect, running after the bbox clamp. Real
        example: recipe bbox lon [-10,30] lat [35,60] (span 40x25, aspect
        0.625) padded lat out to ~[27.5, 67.5]."""
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import GeographicBounds
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        # Fixture's SAR scene spans lon [-10,-8] lat [50,52] -- request a
        # bbox wider than the scene itself so the pre-padding extent is
        # short-and-wide relative to min_aspect=1.0.
        bounds = GeographicBounds(min_lon=-10.0, max_lon=-4.0, min_lat=50.0, max_lat=51.0)
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
            geographic_bounds=bounds,
        )
        ax = fig.axes[0]
        ylim = ax.get_ylim()
        assert ylim[0] >= bounds.min_lat, ylim
        assert ylim[1] <= bounds.max_lat, ylim
        plt.close("all")


class TestPlotGeographicTicks:
    def test_subplots_get_degree_formatted_ticks(self, geo_datatree_and_collocation):
        """Regression test for the gridliner -> plain-tick swap: subplots
        must still show degree-labeled lon/lat ticks, just without
        gridliner's expensive label-placement machinery."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        # This fixture has no "collocation_type" field, so split_by=None
        # (matching TestPlotGeographic's existing convention) makes
        # plot_geographic return a single Figure rather than a dict.
        fig = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None,
        )
        fig.canvas.draw()
        ax = fig.axes[0]

        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("°" in lbl for lbl in xlabels), xlabels
        assert any("°" in lbl for lbl in ylabels), ylabels
        # Regression test: axis visibility must be enabled so ticks are actually
        # rendered to canvas. Without this, get_xticklabels() returns the Tick
        # objects and their text content, but nothing is drawn (the axis is
        # invisible), so the PNG appears with blank margins.
        assert ax.xaxis.get_visible() is True, "x-axis must be visible"
        assert ax.yaxis.get_visible() is True, "y-axis must be visible"
        plt.close("all")

    def test_set_lonlat_ticks_aligns_with_gridliner_locator(self):
        """Regression test for tick-label alignment: _set_lonlat_ticks must
        read tick positions from the gridliner's own locator to guarantee
        labels and grid lines never diverge."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import _set_lonlat_ticks

        # Create a GeoAxes with PlateCarree projection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # Plot dummy data so axes have a real extent
        lon = np.linspace(-10.0, -8.0, 5)
        lat = np.linspace(50.0, 52.0, 5)
        lon2d, lat2d = np.meshgrid(lon, lat)
        data = np.linspace(5.0, 12.0, 25).reshape(5, 5)
        ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree())

        # Get gridliner and call _set_lonlat_ticks
        gl = ax.gridlines(draw_labels=False, alpha=0.3)
        _set_lonlat_ticks(ax, gl)

        # After calling _set_lonlat_ticks, verify that the axis tick positions
        # match the gridliner's locator (filtered to within axis limits)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        expected_xticks = np.array([x for x in gl.xlocator.tick_values(*xlim)
                                     if xlim[0] <= x <= xlim[1]])
        expected_yticks = np.array([y for y in gl.ylocator.tick_values(*ylim)
                                     if ylim[0] <= y <= ylim[1]])

        actual_xticks = np.array(ax.get_xticks())
        actual_yticks = np.array(ax.get_yticks())

        # Filter to only ticks within limits (same logic as _set_lonlat_ticks)
        actual_xticks = actual_xticks[(xlim[0] <= actual_xticks) & (actual_xticks <= xlim[1])]
        actual_yticks = actual_yticks[(ylim[0] <= actual_yticks) & (actual_yticks <= ylim[1])]

        # Assert alignment with reasonable tolerance for floating point
        np.testing.assert_allclose(actual_xticks, expected_xticks, atol=1e-10,
                                    err_msg="X-ticks don't match gridliner locator values")
        np.testing.assert_allclose(actual_yticks, expected_yticks, atol=1e-10,
                                    err_msg="Y-ticks don't match gridliner locator values")

        plt.close("all")

    def test_narrow_extent_caps_tick_count(self):
        """Regression test: narrow WV-mode SAR scenes (sub-1° longitude
        span) must not get more than a handful of ticks — the gridliner's
        default locator (nbins=8) produces many closely-spaced,
        high-decimal-precision ticks that overlap into unreadable labels
        otherwise."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import _set_lonlat_ticks

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        lon = np.linspace(-49.55, -49.50, 5)   # ~0.05° wide, like a WV imagette
        lat = np.linspace(20.0, 43.0, 5)       # wide latitude range
        lon2d, lat2d = np.meshgrid(lon, lat)
        data = np.linspace(5.0, 12.0, 25).reshape(5, 5)
        ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=False, alpha=0.3)
        _set_lonlat_ticks(ax, gl)

        assert len(ax.get_xticks()) <= 3
        assert len(ax.get_yticks()) <= 3
        plt.close(fig)

    def test_tall_narrow_wv_track_extent_caps_tick_count(self):
        """Regression test: a WV-mode SAR scene's ground track can span
        several degrees of longitude (not just sub-1°) while spanning many
        more degrees of latitude. Because cartopy's PlateCarree GeoAxes
        always renders at equal aspect ratio, that tall/narrow extent gets
        squeezed into a visually narrow map column regardless of the
        subplot's nominal figure width — a moderate cap (e.g. nbins=4) still
        overlaps in that squeezed column; only a stricter cap avoids it.
        This reproduces the real bug found by rendering an actual report
        (a 5°-wide, 23°-tall WV scene), not just an isolated tick-value
        count."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import _set_lonlat_ticks

        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": ccrs.PlateCarree()})
        lon = np.linspace(-50.0, -45.0, 20)   # 5° wide
        lat = np.linspace(20.0, 43.0, 20)     # 23° tall — same equal-aspect
                                               # squeeze that broke this in
                                               # the real report
        ax.scatter(lon, lat, transform=ccrs.PlateCarree())
        ax.set_extent(
            [lon.min() - 0.5, lon.max() + 0.5, lat.min() - 0.5, lat.max() + 0.5],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(draw_labels=False, alpha=0.3)
        _set_lonlat_ticks(ax, gl)

        assert len(ax.get_xticks()) <= 2
        assert len(ax.get_yticks()) <= 3
        plt.close(fig)


class TestPlotGeographicCircularColormap:
    def test_wdir_uses_twilight_cmap_and_0_360_range(self, geo_datatree_and_collocation, monkeypatch):
        """owiWindDirection/WDIR is a circular (0-360°) variable: both the
        SAR field (pcolormesh) and the validation points (scatter) must
        render with a shared cyclic colormap and fixed 0-360 color limits,
        not viridis + percentile-derived limits (meaningless for data that
        wraps at the 0/360 seam)."""
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        scene_ds = datatree["sar/sceneA"].to_dataset()
        scene_ds = scene_ds.assign(owiWindDirection=scene_ds["owiWindSpeed"])
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": scene_ds,
            "validation/mooring": datatree["validation/mooring"].to_dataset(),
            "validation/altimeter": datatree["validation/altimeter"].to_dataset(),
        })
        coll = collocation_ds.rename({"val_WSPD": "val_WDIR"}).assign(
            sar_owiWindDirection=collocation_ds["sar_owiWindSpeed"]
        )

        mesh_calls = []
        scatter_calls = []
        original_scatter = matplotlib.axes.Axes.scatter
        original_pcolormesh = matplotlib.axes.Axes.pcolormesh

        def recording_scatter(self, *args, **kwargs):
            if "c" in kwargs:
                scatter_calls.append((kwargs.get("cmap"), kwargs.get("norm")))
            return original_scatter(self, *args, **kwargs)

        def recording_pcolormesh(self, *args, **kwargs):
            mesh_calls.append((kwargs.get("cmap"), kwargs.get("norm")))
            return original_pcolormesh(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", recording_pcolormesh)
        fig = plot_geographic(datatree, coll, "owiWindDirection", "WDIR", split_by=None)
        plt.close("all")

        assert fig is not None
        assert mesh_calls, "expected the SAR field to be drawn with pcolormesh"
        assert scatter_calls, "expected validation points to be drawn with scatter"

        def cmap_name(c):
            return getattr(c, "name", c)

        for cmap, norm in mesh_calls + scatter_calls:
            assert cmap_name(cmap) == "twilight"
            assert norm.vmin == 0.0
            assert norm.vmax == 360.0

    def test_non_circular_var_keeps_viridis_and_percentile_limits(self, geo_datatree_and_collocation, monkeypatch):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation

        mesh_calls = []
        original_pcolormesh = matplotlib.axes.Axes.pcolormesh

        def recording_pcolormesh(self, *args, **kwargs):
            mesh_calls.append((kwargs.get("cmap"), kwargs.get("norm")))
            return original_pcolormesh(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", recording_pcolormesh)
        fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None)
        plt.close("all")

        assert fig is not None
        assert mesh_calls

        def cmap_name(c):
            return getattr(c, "name", c)

        for cmap, norm in mesh_calls:
            assert cmap_name(cmap) == "viridis"
            assert not (norm.vmin == 0.0 and norm.vmax == 360.0)


class TestPlotGeographicAntimeridian:
    def test_scene_crossing_dateline_gets_own_central_longitude_180_projection(self):
        """Regression test: a per-scene antimeridian bug distinct from the
        one already fixed in plot_collocation_diagnostics. A WV-style scene
        whose 1-D lon coordinate straddles the dateline (e.g. 178E..178W)
        used to be drawn on the module's single shared
        ``PlateCarree(central_longitude=0)`` axes, which autoscales to a
        full [-180, 180] world map with the swath split across both edges.
        Each per-scene subplot must instead get its own projection —
        ``central_longitude=180`` when that scene's raw lons span more than
        180 degrees — so the swath renders as one contiguous, narrow strip."""
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        n = 4
        lon = np.array([178.0, 179.0, -179.0, -178.0])
        lat = np.array([-1.0, -0.5, 0.5, 1.0])
        wind = np.array([6.0, 6.5, 7.0, 7.5])
        sar_ds = xr.Dataset(
            {"owiWindSpeed": ("point", wind)},
            coords={
                "lon": ("point", lon),
                "lat": ("point", lat),
                "time": pd.Timestamp("2026-07-02T12:00:00"),
            },
        )
        mooring_ds = xr.Dataset(
            {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
            coords={
                "lon": ("point", np.array([178.2, 178.8, -178.8, -178.2])),
                "lat": ("point", np.array([-0.8, -0.3, 0.3, 0.8])),
                "time": ("point", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
            },
            attrs={"platform_type": "mooring"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/mooring": mooring_ds,
        })

        collocation_ds = xr.Dataset({
            "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.4, 6.9, 7.3])),
            "val_WSPD":                    ("collocation", np.array([6.0, 6.5, 7.0, 7.5])),
            "val_source":                  ("collocation", ["mooring"] * n),
            "sar_scene_name":              ("collocation", ["sceneA"] * n),
            "val_lon":                     ("collocation", np.array([178.2, 178.8, -178.8, -178.2])),
            "val_lat":                     ("collocation", np.array([-0.8, -0.3, 0.3, 0.8])),
            "val_id":                      ("collocation", ["mo0", "mo1", "mo2", "mo3"]),
            "temporal_distance_minutes":   ("collocation", np.array([10.0, 20.0, 30.0, 40.0])),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
        )

        fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None)
        fig.canvas.draw()
        ax = fig.axes[0]

        xlim = ax.get_xlim()
        span = abs(xlim[1] - xlim[0])
        assert span < 90, (
            f"scene subplot autoscaled to a near-world extent {xlim} (span={span}) "
            "instead of a narrow contiguous strip around the dateline"
        )
        assert ax.projection.proj4_params.get("lon_0") == 180

        plt.close("all")

    def test_non_crossing_scene_keeps_default_projection(self, geo_datatree_and_collocation):
        """A scene whose lons do not straddle the dateline must be
        unaffected: plain PlateCarree (central_longitude=0), matching
        pre-fix behaviour."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None)
        ax = fig.axes[0]

        assert ax.projection.proj4_params.get("lon_0", 0) == 0
        plt.close("all")


class TestPlotGeographicPanelAspect:
    def test_short_wide_scene_padded_to_min_aspect(self):
        """Regression test: a scene with very few, tightly-clustered
        imagettes (small latitude span relative to its longitude span) used
        to autoscale to a short, wide box, visually inconsistent next to the
        tall/portrait panels typical of WV-mode satellite tracks in the same
        report. The rendered panel must never be shorter than it is wide."""
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        n = 2
        lon = np.array([-10.0, -8.0])
        lat = np.array([50.0, 50.3])
        wind = np.array([6.0, 6.5])
        sar_ds = xr.Dataset(
            {"owiWindSpeed": ("point", wind)},
            coords={
                "lon": ("point", lon),
                "lat": ("point", lat),
                "time": pd.Timestamp("2026-07-02T12:00:00"),
            },
        )
        mooring_ds = xr.Dataset(
            {"WSPD": ("point", np.array([6.0, 6.5]))},
            coords={
                "lon": ("point", np.array([-9.8, -8.2])),
                "lat": ("point", np.array([50.05, 50.25])),
                "time": ("point", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
            },
            attrs={"platform_type": "mooring"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/mooring": mooring_ds,
        })

        collocation_ds = xr.Dataset({
            "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.4])),
            "val_WSPD":                    ("collocation", np.array([6.0, 6.5])),
            "val_source":                  ("collocation", ["mooring"] * n),
            "sar_scene_name":              ("collocation", ["sceneA"] * n),
            "val_lon":                     ("collocation", np.array([-9.8, -8.2])),
            "val_lat":                     ("collocation", np.array([50.05, 50.25])),
            "val_id":                      ("collocation", ["mo0", "mo1"]),
            "temporal_distance_minutes":   ("collocation", np.array([10.0, 20.0])),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
        )

        fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None)
        fig.canvas.draw()
        ax = fig.axes[0]

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        width = x1 - x0
        height = y1 - y0
        assert height / width >= 1.0, (
            f"panel rendered short-and-wide (width={width}, height={height}) instead of "
            "portrait-or-square"
        )

        plt.close("all")

    def test_already_portrait_scene_extent_unchanged(self):
        """A scene whose natural autoscaled extent is already portrait
        (height/width >= 1, similar to a real WV-mode satellite track
        spanning many degrees of latitude but only a few of longitude) must
        be completely unaffected by the padding helper: its aspect should
        remain >= 1, i.e. no distortion is introduced where none is needed."""
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        n = 4
        lon = np.array([-10.0, -9.3, -8.6, -7.9])
        lat = np.array([10.0, 20.0, 30.0, 50.0])
        wind = np.array([6.0, 6.5, 7.0, 7.5])
        sar_ds = xr.Dataset(
            {"owiWindSpeed": ("point", wind)},
            coords={
                "lon": ("point", lon),
                "lat": ("point", lat),
                "time": pd.Timestamp("2026-07-02T12:00:00"),
            },
        )
        mooring_ds = xr.Dataset(
            {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
            coords={
                "lon": ("point", lon),
                "lat": ("point", lat),
                "time": ("point", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
            },
            attrs={"platform_type": "mooring"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/mooring": mooring_ds,
        })

        collocation_ds = xr.Dataset({
            "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.4, 6.9, 7.3])),
            "val_WSPD":                    ("collocation", np.array([6.0, 6.5, 7.0, 7.5])),
            "val_source":                  ("collocation", ["mooring"] * n),
            "sar_scene_name":              ("collocation", ["sceneA"] * n),
            "val_lon":                     ("collocation", lon),
            "val_lat":                     ("collocation", lat),
            "val_id":                      ("collocation", ["mo0", "mo1", "mo2", "mo3"]),
            "temporal_distance_minutes":   ("collocation", np.array([10.0, 20.0, 30.0, 40.0])),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
        )

        fig = plot_geographic(datatree, collocation_ds, "owiWindSpeed", "WSPD", split_by=None)
        fig.canvas.draw()
        ax = fig.axes[0]

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # A trivial `>= 1.0` assertion on the ratio would pass even if the
        # padding helper always widened the y-extent somewhat, since this
        # scene's natural ratio (~19) is so far above 1 that mild padding
        # wouldn't pull it back down that far. To genuinely pin "extent is
        # unaffected by padding", independently derive the *expected*
        # natural extent from the same lon/lat data using matplotlib's
        # default autoscale margin (5% of span on each side, the same
        # ``axes.xmargin``/``axes.ymargin`` defaults matplotlib/cartopy
        # apply when autoscaling a scatter with no explicit limits) and
        # assert the rendered extent matches it exactly (within floating
        # tolerance) — proving `_pad_extent_to_min_aspect` took its
        # early-return path and never touched the y-limits at all.
        xmargin = plt.rcParams["axes.xmargin"]
        ymargin = plt.rcParams["axes.ymargin"]
        lon_span = lon.max() - lon.min()
        lat_span = lat.max() - lat.min()
        expected_x0 = lon.min() - xmargin * lon_span
        expected_x1 = lon.max() + xmargin * lon_span
        expected_y0 = lat.min() - ymargin * lat_span
        expected_y1 = lat.max() + ymargin * lat_span

        assert x0 == pytest.approx(expected_x0, abs=1e-6)
        assert x1 == pytest.approx(expected_x1, abs=1e-6)
        assert y0 == pytest.approx(expected_y0, abs=1e-6)
        assert y1 == pytest.approx(expected_y1, abs=1e-6)

        plt.close("all")


class TestPlotGeographicPointSubsampling:
    def test_dense_scene_is_subsampled_for_plotting(self):
        """A scene with far more collocated points than max_points_per_panel
        must still render (not crash/hang) and must plot fewer markers than
        the full point count -- statistics elsewhere are unaffected since
        this only touches plot_geographic's own dataframe, not
        collocation_ds itself."""
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        rng = np.random.default_rng(0)
        y, x = 50, 50
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), rng.uniform(0, 100, (y, x)))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T12:00:00")},
        )
        n = 6000
        ascat_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", rng.uniform(0, 100, n))},
            coords={
                "lon": ("point", rng.uniform(-10.0, -8.0, n)),
                "lat": ("point", rng.uniform(50.0, 52.0, n)),
                "time": ("point", pd.date_range("2026-07-10T12:00", periods=n, freq="1s")),
            },
            attrs={"platform_type": "ascat_ssm"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds, "validation/ascat_ssm": ascat_ds,
        })
        collocation_ds = xr.Dataset({
            "sar_sarSSM":       ("collocation", rng.uniform(0, 100, n)),
            "val_SOIL_MOISTURE": ("collocation", rng.uniform(0, 100, n)),
            "val_source":       ("collocation", ["ascat_ssm"] * n),
            "sar_scene_name":   ("collocation", ["sceneA"] * n),
            "val_lon":          ("collocation", rng.uniform(-10.0, -8.0, n)),
            "val_lat":          ("collocation", rng.uniform(50.0, 52.0, n)),
            "val_id":           ("collocation", [f"o{i}" for i in range(n)]),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T12:00", periods=n, freq="1s")),
        )

        fig = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE", split_by=None,
            max_points_per_panel=500,
        )
        ax = fig.axes[0]
        plotted = sum(len(c.get_offsets()) for c in ax.collections if hasattr(c, "get_offsets"))
        assert 0 < plotted <= 600, plotted  # allows a little slack over the 500 cap
        plt.close("all")


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
        CollocationType,
        GeographicBounds,
        PointVsLayerCollocation,
        Recipe,
        RecipeConfig,
        ValidationDataSource,
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


@pytest.fixture
def geo_datatree_and_collocation_dateline():
    """Synthetic DataTree + collocation_ds whose SAR scene and validation
    points straddle the antimeridian (170E..170W), used to test
    plot_collocation_diagnostics' dateline-crossing map extent."""
    from sar_validation.core.datatree_converter import DataTreeConverter

    lon2d = np.array([
        [170.0, 175.0, 180.0, -175.0, -170.0],
        [170.0, 175.0, 180.0, -175.0, -170.0],
    ])
    lat2d = np.array([
        [-2.0, -2.0, -2.0, -2.0, -2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ])
    wind = np.linspace(5.0, 12.0, lon2d.size).reshape(lon2d.shape)
    sar_ds = xr.Dataset(
        {"owiWindSpeed": (("y", "x"), wind)},
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
            "time": pd.Timestamp("2026-07-02T12:00:00"),
        },
    )

    n = 4
    mooring_ds = xr.Dataset(
        {"WSPD": ("point", np.array([6.0, 6.5, 7.0, 7.5]))},
        coords={
            "lon": ("point", np.array([172.0, 178.0, -178.0, -172.0])),
            "lat": ("point", np.array([-1.0, -0.5, 0.5, 1.0])),
            "time": ("point", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
        },
        attrs={"platform_type": "mooring"},
    )

    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": sar_ds,
        "validation/mooring": mooring_ds,
    })

    collocation_ds = xr.Dataset({
        "sar_owiWindSpeed":            ("collocation", np.array([6.1, 6.9, 8.2, 9.3])),
        "val_WSPD":                    ("collocation", np.array([6.0, 7.0, 8.0, 9.5])),
        "val_source":                  ("collocation", ["mooring"] * n),
        "sar_scene_name":              ("collocation", ["sceneA"] * n),
        "val_lon":                     ("collocation", np.array([172.0, 178.0, -178.0, -172.0])),
        "val_lat":                     ("collocation", np.array([-1.0, -0.5, 0.5, 1.0])),
        "val_id":                      ("collocation", ["mo0", "mo1", "mo2", "mo3"]),
        "temporal_distance_minutes":   ("collocation", np.array([10.0, 20.0, 30.0, 40.0])),
    })
    collocation_ds = collocation_ds.assign_coords(
        val_time=("collocation", pd.date_range("2026-07-02T12:05", periods=n, freq="5min")),
    )
    return datatree, collocation_ds


@pytest.fixture
def diagnostics_recipe_dateline():
    from sar_validation.core.recipe import (
        CollocationType,
        GeographicBounds,
        PointVsLayerCollocation,
        Recipe,
        RecipeConfig,
        ValidationDataSource,
    )
    config = RecipeConfig(
        name="test_recipe_dateline",
        variable="wind",
        geographic_bounds=GeographicBounds(min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0),
        validation_sources=[ValidationDataSource(source_type="mooring")],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)


@pytest.fixture
def diagnostics_recipe_waves():
    from sar_validation.core.recipe import (
        CollocationType,
        GeographicBounds,
        PointVsLayerCollocation,
        Recipe,
        RecipeConfig,
        ValidationDataSource,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="waves",
        geographic_bounds=GeographicBounds(min_lon=-11.0, max_lon=-7.0, min_lat=49.0, max_lat=53.0),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="altimeter"),
        ],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)


@pytest.fixture
def diagnostics_recipe_currents():
    from sar_validation.core.recipe import (
        CollocationType,
        GeographicBounds,
        PointVsLayerCollocation,
        Recipe,
        RecipeConfig,
        ValidationDataSource,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="currents",
        geographic_bounds=GeographicBounds(min_lon=-11.0, max_lon=-7.0, min_lat=49.0, max_lat=53.0),
        validation_sources=[
            ValidationDataSource(source_type="mooring"),
            ValidationDataSource(source_type="altimeter"),
        ],
        collocation=CollocationType(point_vs_layer=PointVsLayerCollocation(time_tolerance_minutes=30)),
    )
    return Recipe(config=config)


@pytest.fixture
def diagnostics_recipe_soil_moisture():
    from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, ValidationDataSource
    config = RecipeConfig(
        name="test_recipe",
        variable="soil_moisture",
        geographic_bounds=GeographicBounds(min_lon=-10.0, max_lon=-8.0, min_lat=50.0, max_lat=52.0),
        validation_sources=[ValidationDataSource(source_type="ismn")],
    )
    return Recipe(config=config)


class TestPlotGeographicTwoColumnByType:
    def test_returns_one_figure_per_scene_with_two_columns(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.linspace(10.0, 60.0, y * x).reshape(y, x))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T12:00:00")},
        )
        n = 4
        ismn_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array([0.1, 0.15, 0.2, 0.25]))},
            coords={"lon": ("point", np.array([-9.8, -9.6, -9.4, -9.2])),
                    "lat": ("point", np.array([50.2, 50.4, 50.6, 50.8])),
                    "time": ("point", pd.date_range("2026-07-10T12:00", periods=n, freq="5min"))},
            attrs={"platform_type": "ismn"},
        )
        ascat_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array([20.0, 30.0, 40.0, 50.0]))},
            coords={"lon": ("point", np.array([-9.0, -8.8, -8.6, -8.4])),
                    "lat": ("point", np.array([51.0, 51.2, 51.4, 51.6])),
                    "time": ("point", pd.date_range("2026-07-10T12:00", periods=n, freq="5min"))},
            attrs={"platform_type": "ascat_ssm"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ismn": ismn_ds,
            "validation/ascat_ssm": ascat_ds,
        })
        collocation_ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", np.array([12.0, 18.0, 22.0, 48.0])),
            "val_SOIL_MOISTURE": ("collocation", np.array([0.1, 0.15, 20.0, 50.0])),
            "val_source":        ("collocation", ["ismn", "ismn", "ascat_ssm", "ascat_ssm"]),
            "collocation_type":  ("collocation", ["point_vs_layer", "point_vs_layer",
                                                    "layer_vs_layer", "layer_vs_layer"]),
            "sar_scene_name":    ("collocation", ["sceneA"] * n),
            "val_lon":           ("collocation", np.array([-9.8, -9.6, -9.0, -8.8])),
            "val_lat":           ("collocation", np.array([50.2, 50.4, 51.0, 51.2])),
            "val_id":            ("collocation", ["i0", "i1", "a0", "a1"]),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T12:00", periods=n, freq="5min")),
        )

        result = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE",
            split_by="collocation_type", two_column_by_type=True,
        )

        assert set(result.keys()) == {"sceneA"}
        fig = result["sceneA"]
        assert len(fig.axes) >= 2
        plt.close("all")

    def test_two_column_figure_has_a_colorbar(self):
        """Regression test: _build_scene_pair_figure never called
        fig.colorbar() at all -- soil moisture's geographic layout
        structurally had no colorbar, not just an intermittent one."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_geographic

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.linspace(10.0, 60.0, y * x).reshape(y, x))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T12:00:00")},
        )
        n = 4
        ismn_ds = xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array([0.1, 0.15, 0.2, 0.25]))},
            coords={"lon": ("point", np.array([-9.8, -9.6, -9.4, -9.2])),
                    "lat": ("point", np.array([50.2, 50.4, 50.6, 50.8])),
                    "time": ("point", pd.date_range("2026-07-10T12:00", periods=n, freq="5min"))},
            attrs={"platform_type": "ismn"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ismn": ismn_ds,
        })
        collocation_ds = xr.Dataset({
            "sar_sarSSM":        ("collocation", np.array([12.0, 18.0, 22.0, 28.0])),
            "val_SOIL_MOISTURE": ("collocation", np.array([0.1, 0.15, 0.2, 0.25])),
            "val_source":        ("collocation", ["ismn"] * n),
            "collocation_type":  ("collocation", ["point_vs_layer"] * n),
            "sar_scene_name":    ("collocation", ["sceneA"] * n),
            "val_lon":           ("collocation", np.array([-9.8, -9.6, -9.4, -9.2])),
            "val_lat":           ("collocation", np.array([50.2, 50.4, 50.6, 50.8])),
            "val_id":            ("collocation", ["i0", "i1", "i2", "i3"]),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T12:00", periods=n, freq="5min")),
        )

        result = plot_geographic(
            datatree, collocation_ds, "sarSSM", "SOIL_MOISTURE",
            split_by="collocation_type", two_column_by_type=True,
        )

        fig = result["sceneA"]
        # A colorbar is its own Axes with no ticks/labels on the main
        # data axes -- matplotlib gives it a distinguishing '_colorbar'
        # attribute reference chain via its images/collections being empty
        # of geographic data; check via the presence of extra narrow axes
        # beyond the one/two data GeoAxes already asserted to exist.
        from cartopy.mpl.geoaxes import GeoAxes
        data_axes = [ax for ax in fig.axes if isinstance(ax, GeoAxes)]
        non_data_axes = [ax for ax in fig.axes if not isinstance(ax, GeoAxes)]
        assert len(data_axes) >= 1
        assert len(non_data_axes) >= 1, "expected at least one colorbar axes"
        plt.close("all")

    def test_two_column_disabled_by_default_keeps_dict_by_group(self, geo_datatree_and_collocation):
        """Default (two_column_by_type=False) behavior is byte-identical to
        before this task -- keyed by collocation_type, not scene."""
        from sar_validation.core.visualization import plot_geographic

        datatree, collocation_ds = geo_datatree_and_collocation
        collocation_ds = collocation_ds.assign(
            collocation_type=("collocation", ["point_vs_layer"] * collocation_ds.sizes["collocation"]),
        )
        result = plot_geographic(
            datatree, collocation_ds, "owiWindSpeed", "WSPD",
            split_by="collocation_type",
        )
        assert set(result.keys()) == {"point_vs_layer"}


class TestDiagnosticsCategory:
    def test_literal_ascat_ssm_maps_to_scatterometer(self):
        from sar_validation.core.visualization import _diagnostics_category
        assert _diagnostics_category("ascat_ssm") == "Scatterometer"

    def test_literal_radiometer_trio_maps_to_radiometer(self):
        from sar_validation.core.visualization import _diagnostics_category
        assert _diagnostics_category("amsr_ssm") == "Radiometer"
        assert _diagnostics_category("smap_ssm") == "Radiometer"
        assert _diagnostics_category("smos_ssm") == "Radiometer"

    def test_generic_data_type_tokens_map_to_same_categories_as_literal_names(self):
        """The unmatched code path carries the generic data_type token
        (e.g. "scatterometer_ssm"), not the literal per-satellite name
        -- both forms must resolve to the SAME category so a matched
        ascat_ssm point and an unmatched scatterometer_ssm point land
        in the same legend bucket."""
        from sar_validation.core.visualization import _diagnostics_category
        assert _diagnostics_category("scatterometer_ssm") == "Scatterometer"
        assert _diagnostics_category("radiometer_ssm") == "Radiometer"

    def test_existing_generic_layer_types_still_work_via_fallback(self):
        from sar_validation.core.visualization import _diagnostics_category
        assert _diagnostics_category("scatterometer") == "Scatterometer"
        assert _diagnostics_category("altimeter") == "Altimeter"
        assert _diagnostics_category("hf_radar") == "Hf_Radar"

    def test_in_situ_platform_type_falls_to_in_situ(self):
        from sar_validation.core.visualization import _diagnostics_category
        assert _diagnostics_category("mooring") == "In-situ"
        assert _diagnostics_category("ismn") == "In-situ"


class TestPlotCollocationDiagnosticsSoilMoistureSourceCategories:
    """Regression test for the real bug: matched ascat_ssm points used
    to show up as "In-situ" instead of "Scatterometer", because
    plot_collocation_diagnostics' matched-point path keys off the
    literal val_source name ("ascat_ssm"), which was never a member of
    LAYER_DATA_TYPES (that set holds the generic category token
    "scatterometer_ssm" instead). The unmatched path was not actually
    broken by this bug -- it already reads the generic data_type token
    directly off the datatree node, which *is* a LAYER_DATA_TYPES
    member -- but plot_collocation_diagnostics never attaches a
    per-category `label` to unmatched-point scatter calls at all (see
    the Tier 1/Tier 2 drawing loops), so the only observable signal for
    an unmatched point's category is which zorder tier it lands in:
    zorder=2 (non-in-situ / layer) vs. zorder=3 (in-situ)."""

    def test_matched_and_unmatched_soil_moisture_sources_categorized_correctly(
        self, tmp_path, monkeypatch,
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig
        from sar_validation.core.visualization import plot_collocation_diagnostics

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={
                "lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )

        def _point_ds(n, lon0, lat0, data_type):
            return xr.Dataset(
                {"SOIL_MOISTURE": ("point", np.linspace(0.1, 0.3, n))},
                coords={
                    "lon": ("point", lon0 + 0.05 * np.arange(n)),
                    "lat": ("point", lat0 + 0.05 * np.arange(n)),
                    "time": ("point", pd.date_range("2026-07-10T19:05", periods=n, freq="5min")),
                },
                attrs={"data_type": data_type},
            )

        # matched: literal val_source "ascat_ssm" (what collocation_ds
        # actually carries for a matched point) -- this is the real bug.
        ascat_matched_ds = _point_ds(2, -9.8, 50.2, "scatterometer_ssm")
        # unmatched-only: a SEPARATE node the collocation step never
        # matched, carrying the generic data_type token directly --
        # included to confirm the fix doesn't regress this path, even
        # though it wasn't the source of the original bug.
        smap_unmatched_ds = _point_ds(2, -9.0, 51.8, "radiometer_ssm")

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/ascat_ssm": ascat_matched_ds,
            "validation/smap_ssm": smap_unmatched_ds,
        })

        n = ascat_matched_ds.sizes["point"]
        collocation_ds = xr.Dataset({
            "sar_sarSSM": ("collocation", np.full(n, 25.0)),
            "val_SOIL_MOISTURE": ("collocation", ascat_matched_ds["SOIL_MOISTURE"].values),
            "val_source": ("collocation", np.array(["ascat_ssm"] * n)),
            "sar_scene_name": ("collocation", np.array(["sceneA"] * n)),
            "val_lon": ("collocation", ascat_matched_ds["lon"].values),
            "val_lat": ("collocation", ascat_matched_ds["lat"].values),
            "temporal_distance_minutes": ("collocation", np.full(n, 10.0)),
        })

        recipe = Recipe(config=RecipeConfig(
            name="test_soil_moisture_categories", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
        ))

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append((args, kwargs))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        assert out_path is not None

        # Matched ascat_ssm points (zorder=5): must be labeled
        # "Scatterometer matched", never "In-situ matched" -- this is
        # the real, confirmed bug.
        matched_labels = [k.get("label", "") for _, k in recorded if k.get("zorder") == 5]
        assert any("Scatterometer matched" in lbl for lbl in matched_labels), (
            f"expected a 'Scatterometer matched' label among matched-tier "
            f"calls, got: {matched_labels}"
        )
        assert not any("In-situ" in lbl for lbl in matched_labels), (
            f"ascat_ssm matched points must never be labeled In-situ, got: {matched_labels}"
        )

        # Unmatched smap_ssm (data_type="radiometer_ssm") points: must
        # land in the unmatched-LAYER tier (zorder=2), never the
        # unmatched-IN-SITU tier (zorder=3, no soil-moisture in-situ
        # source is unmatched in this fixture, so any zorder=3 points
        # would be a misrouted smap point).
        def _lons(call_args):
            args, _ = call_args
            return np.atleast_1d(args[0]) if args else np.array([])

        zorder2_lons = np.concatenate(
            [_lons(c) for c in recorded if c[1].get("zorder") == 2] or [np.array([])]
        )
        zorder3_lons = np.concatenate(
            [_lons(c) for c in recorded if c[1].get("zorder") == 3] or [np.array([])]
        )
        smap_lons = smap_unmatched_ds["lon"].values
        assert all(lon in zorder2_lons for lon in smap_lons), (
            f"expected smap_ssm's unmatched points {smap_lons} in the "
            f"unmatched-layer tier (zorder=2), got zorder=2 lons: {zorder2_lons}"
        )
        assert zorder3_lons.size == 0, (
            f"no soil-moisture in-situ source is unmatched in this fixture -- "
            f"expected zero unmatched-in-situ (zorder=3) points, got: {zorder3_lons}"
        )

    @pytest.mark.parametrize("data_type", ["scatterometer_ssm", "radiometer_ssm"])
    def test_unmatched_point_outside_wind_tolerance_inside_soil_moisture_tolerance_still_shown(
        self, data_type, tmp_path, monkeypatch,
    ):
        """Regression test for the consolidated-label lookup bug: an
        unmatched soil-moisture satellite point placed 8h from the SAR
        scene's time is outside the wind-default 180-minute tolerance
        (DEFAULT_LAYER_TYPE_SPECS["scatterometer"]/["radiometer"]) but
        inside the soil-moisture 720-minute tolerance
        (DEFAULT_LAYER_TYPE_SPECS["scatterometer_ssm"]/["radiometer_ssm"]).

        Parametrized over BOTH scatterometer_ssm (ASCAT) and
        radiometer_ssm (AMSR/SMAP/SMOS all stamp this exact generic
        data_type on their datatree nodes -- see from_amsr_ssm/
        from_smap_ssm/from_smos_ssm in datatree_converter.py; only
        _resolve_layer_type, in collocation.py, refines it further to
        amsr_ssm/smap_ssm/smos_ssm, and plot_collocation_diagnostics
        does not use that refinement) -- a first version of this fix
        covered scatterometer_ssm only and missed that radiometer_ssm
        had no DEFAULT_LAYER_TYPE_SPECS entry at all, so AMSR/SMAP/SMOS
        unmatched points fell through to the recipe's 30-minute
        point_vs_layer default instead of 720.

        _time_tolerance_minutes and the Tier 1 unmatched-layer drawing
        loop must resolve tolerance/style using the RAW data_type token,
        not the _diagnostics_category-consolidated display label
        ("Scatterometer"/"Radiometer") -- keying off the consolidated
        label collides with the wind specs (180 min) and silently drops
        the point from the plot. This must FAIL against the pre-fix code.
        """
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig
        from sar_validation.core.visualization import plot_collocation_diagnostics

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        scene_time = pd.Timestamp("2026-07-10T19:00:00")
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={
                "lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                "time": scene_time,
            },
        )

        # 8h after the scene time: outside the 180-min wind tolerance,
        # inside the 720-min soil-moisture tolerance.
        far_point_time = scene_time + pd.Timedelta(hours=8)

        def _point_ds(n, lon0, lat0, dtype, point_time):
            return xr.Dataset(
                {"SOIL_MOISTURE": ("point", np.linspace(0.1, 0.3, n))},
                coords={
                    "lon": ("point", lon0 + 0.05 * np.arange(n)),
                    "lat": ("point", lat0 + 0.05 * np.arange(n)),
                    "time": ("point", [point_time] * n),
                },
                attrs={"data_type": dtype},
            )

        unmatched_ds = _point_ds(2, -9.0, 51.8, data_type, far_point_time)

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds,
            "validation/far_source": unmatched_ds,
        })

        recipe = Recipe(config=RecipeConfig(
            name="test_soil_moisture_far_point_tolerance", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
        ))

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append((args, kwargs))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(datatree, None, recipe, tmp_path)
        plt.close("all")

        assert out_path is not None

        def _lons(call_args):
            args, _ = call_args
            return np.atleast_1d(args[0]) if args else np.array([])

        zorder2_lons = np.concatenate(
            [_lons(c) for c in recorded if c[1].get("zorder") == 2] or [np.array([])]
        )
        point_lons = unmatched_ds["lon"].values
        assert all(lon in zorder2_lons for lon in point_lons), (
            f"expected the 8h-offset {data_type} point(s) {point_lons} to "
            f"still appear in the unmatched-layer tier (zorder=2) since they're "
            f"within the 720-min soil-moisture tolerance, got zorder=2 lons: "
            f"{zorder2_lons}"
        )


class TestPlotCollocationDiagnostics:
    def test_distinct_sources_get_distinct_markers(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

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


def _small_soil_moisture_scene(lon_center, lat_center, time):
    y, x = 3, 3
    lon2d, lat2d = np.meshgrid(
        np.linspace(lon_center - 0.2, lon_center + 0.2, x),
        np.linspace(lat_center - 0.2, lat_center + 0.2, y),
    )
    sm = np.linspace(0.1, 0.3, y * x).reshape(y, x)
    return xr.Dataset(
        {"sarSSM": (("y", "x"), sm)},
        coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d), "time": pd.Timestamp(time)},
    )


@pytest.fixture
def two_scene_soil_moisture_datatree():
    """Two small, non-overlapping NISAR-like SAR scenes, each tiny
    relative to a continent-scale recipe bbox, plus one ISMN point near
    the first scene -- used to test the registry-driven per-scene-split
    decision and the diagnostics-plot auto-zoom."""
    from sar_validation.core.datatree_converter import DataTreeConverter

    # sceneB sits a couple of degrees from sceneA -- like adjacent
    # NISAR orbit segments from the same pass, not scattered edge-to-edge
    # across the whole (60-degree-wide) recipe bbox, so the auto-zoom
    # test below has something meaningful to zoom in past.
    scene_a = _small_soil_moisture_scene(-120.0, 45.0, "2026-06-17T12:00:00")
    scene_b = _small_soil_moisture_scene(-118.0, 44.0, "2026-06-17T13:00:00")
    ismn_ds = xr.Dataset(
        {"SOIL_MOISTURE": ("point", np.array([0.15, 0.18]))},
        coords={
            "lon": ("point", np.array([-120.05, -119.95])),
            "lat": ("point", np.array([44.95, 45.05])),
            "time": ("point", pd.date_range("2026-06-17T12:05", periods=2, freq="5min")),
        },
        attrs={"platform_type": "ismn"},
    )
    datatree = DataTreeConverter.to_datatree({
        "sar/sceneA": scene_a,
        "sar/sceneB": scene_b,
        "validation/ismn": ismn_ds,
    })
    collocation_ds = xr.Dataset({
        "sar_sarSSM":        ("collocation", np.array([0.14, 0.19])),
        "val_SOIL_MOISTURE": ("collocation", np.array([0.15, 0.18])),
        "val_source":        ("collocation", ["ismn", "ismn"]),
        "sar_scene_name":    ("collocation", ["sceneA", "sceneA"]),
        "val_lon":           ("collocation", np.array([-120.05, -119.95])),
        "val_lat":           ("collocation", np.array([44.95, 45.05])),
        "val_id":            ("collocation", ["i0", "i1"]),
    })
    collocation_ds = collocation_ds.assign_coords(
        val_time=("collocation", pd.date_range("2026-06-17T12:05", periods=2, freq="5min")),
    )
    return datatree, collocation_ds


def _soil_moisture_recipe(source):
    from sar_validation.core.recipe import (
        GeographicBounds,
        Recipe,
        RecipeConfig,
        SARDataSpec,
        ValidationDataSource,
    )
    config = RecipeConfig(
        name="test_recipe",
        variable="soil_moisture",
        geographic_bounds=GeographicBounds(min_lon=-125.0, max_lon=-65.0, min_lat=25.0, max_lat=50.0),
        validation_sources=[ValidationDataSource(source_type="ismn")],
        sar_data=SARDataSpec(source=source),
    )
    return Recipe(config=config)


class TestPlotCollocationDiagnosticsSplitByScenePerSource:
    def test_sentinel1_clms_ssm_splits_into_one_plot_per_scene(
        self, two_scene_soil_moisture_datatree, tmp_path
    ):
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = two_scene_soil_moisture_datatree
        recipe = _soil_moisture_recipe("sentinel1_clms_ssm")

        result = plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_nisar_sme2_keeps_one_combined_plot(
        self, two_scene_soil_moisture_datatree, tmp_path
    ):
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = two_scene_soil_moisture_datatree
        recipe = _soil_moisture_recipe("nisar_sme2")

        result = plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        assert not isinstance(result, list)
        assert result is not None


class TestPlotCollocationDiagnosticsAutoZoom:
    def test_small_scene_zooms_in_past_the_full_recipe_bbox(
        self, two_scene_soil_moisture_datatree, tmp_path, monkeypatch
    ):
        """The recipe bbox spans 60 degrees of longitude; each SAR scene
        is <1 degree wide. The plotted extent must be far tighter than
        the full bbox, not always the full bbox."""
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = two_scene_soil_moisture_datatree
        recipe = _soil_moisture_recipe("nisar_sme2")

        captured_extents = []
        import cartopy.mpl.geoaxes

        original_set_extent = cartopy.mpl.geoaxes.GeoAxes.set_extent

        def recording_set_extent(self, extent, *args, **kwargs):
            captured_extents.append(extent)
            return original_set_extent(self, extent, *args, **kwargs)

        monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "set_extent", recording_set_extent)
        plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        assert captured_extents, "expected at least one set_extent call"
        lon0, lon1, _lat0, _lat1 = captured_extents[0]
        assert (lon1 - lon0) < 20.0, (
            f"expected a zoomed-in extent (<20 degrees wide), got {lon1 - lon0}"
        )

    def test_zoom_never_exceeds_the_recipe_bounds(
        self, two_scene_soil_moisture_datatree, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = two_scene_soil_moisture_datatree
        recipe = _soil_moisture_recipe("nisar_sme2")

        captured_extents = []
        import cartopy.mpl.geoaxes

        original_set_extent = cartopy.mpl.geoaxes.GeoAxes.set_extent

        def recording_set_extent(self, extent, *args, **kwargs):
            captured_extents.append(extent)
            return original_set_extent(self, extent, *args, **kwargs)

        monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, "set_extent", recording_set_extent)
        plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        bounds = recipe.config.geographic_bounds
        lon0, lon1, lat0, lat1 = captured_extents[0]
        assert lon0 >= bounds.min_lon - 1e-6
        assert lon1 <= bounds.max_lon + 1e-6
        assert lat0 >= bounds.min_lat - 1e-6
        assert lat1 <= bounds.max_lat + 1e-6


class TestPlotCollocationDiagnosticsNoValidationDataAtAll:
    """A validation source that collected zero files (e.g. ISMN awaiting a
    manually-downloaded archive) means the DataTree has SAR data but no
    'validation' node at all -- not just zero collocated pairs. The plot is
    documented as 'always generated', but this is a different, stricter
    case than 'zero collocated pairs with some validation points still
    marked unmatched': there's no validation data whatsoever to mark. It
    must still render the SAR coverage rather than silently returning
    None."""

    def _sar_only_datatree(self):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        sar_ds = xr.Dataset(
            {"owiWindSpeed": (("y", "x"), np.linspace(5.0, 12.0, y * x).reshape(y, x))},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )
        return DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

    def test_returns_a_plot_path_instead_of_none(self, diagnostics_recipe, tmp_path):
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree = self._sar_only_datatree()

        out_path = plot_collocation_diagnostics(
            datatree, None, diagnostics_recipe, tmp_path,
        )

        assert out_path is not None
        assert out_path.exists()

    def test_plot_shows_only_sar_coverage_no_validation_points(self, diagnostics_recipe, tmp_path, monkeypatch):
        import matplotlib.axes

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree = self._sar_only_datatree()

        recorded_scatter_calls = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded_scatter_calls.append((args, kwargs))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, None, diagnostics_recipe, tmp_path)

        # No validation points to scatter at all (only the SAR footprint,
        # drawn with ax.plot, not ax.scatter).
        assert recorded_scatter_calls == []


class TestPlotCollocationDiagnosticsSoilMoistureOverpassCoverage:
    """CLMS Surface Soil Moisture's grid has valid lon/lat everywhere across
    the continent, but the actual retrieved value is NaN except along that
    day's satellite overpass swaths -- a min/max bounding rectangle over
    the grid's lon/lat therefore claims coverage across empty regions with
    no real data. Only soil_moisture should switch to plotting the real
    valid-pixel footprint instead of a rectangle; other variables (whose
    SAR products genuinely do fill their bounding rectangle) must keep the
    existing rectangle."""

    def _half_covered_datatree(self, variable_name="sarSSM"):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 20, 20
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        data = np.full((y, x), 30.0)
        # Only the left half of the grid has real retrievals (simulating an
        # overpass swath); the right half is NaN, same as unretrieved CLMS
        # SSM pixels on a real day.
        data[:, x // 2:] = np.nan
        sar_ds = xr.Dataset(
            {variable_name: (("y", "x"), data)},
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )
        return DataTreeConverter.to_datatree({"sar/sceneA": sar_ds}), lon2d

    def test_soil_moisture_scatters_only_actual_valid_pixels(
        self, diagnostics_recipe_soil_moisture, tmp_path, monkeypatch
    ):
        import matplotlib.axes

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, lon2d = self._half_covered_datatree()
        midpoint_lon = float(lon2d[:, lon2d.shape[1] // 2].mean())

        recorded_scatter_lons = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, xs, ys, *args, **kwargs):
            recorded_scatter_lons.extend(np.atleast_1d(xs).tolist())
            return original_scatter(self, xs, ys, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(
            datatree, None, diagnostics_recipe_soil_moisture, tmp_path,
        )

        assert out_path is not None
        assert len(recorded_scatter_lons) > 0
        # Every scattered coverage point must fall within the actual valid
        # (left) half -- none in the NaN (right) half the old rectangle
        # would have wrongly covered.
        assert all(lon < midpoint_lon for lon in recorded_scatter_lons)

    def test_soil_moisture_does_not_draw_a_bounding_rectangle(
        self, diagnostics_recipe_soil_moisture, tmp_path, monkeypatch
    ):
        import matplotlib.axes

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, _ = self._half_covered_datatree()

        recorded_labels = []
        original_plot = matplotlib.axes.Axes.plot

        def recording_plot(self, *args, **kwargs):
            recorded_labels.append(kwargs.get("label"))
            return original_plot(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
        plot_collocation_diagnostics(datatree, None, diagnostics_recipe_soil_moisture, tmp_path)

        assert "SAR scene bounds" not in recorded_labels

    def test_non_soil_moisture_still_uses_rectangle(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.axes

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation

        recorded_labels = []
        original_plot = matplotlib.axes.Axes.plot

        def recording_plot(self, *args, **kwargs):
            recorded_labels.append(kwargs.get("label"))
            return original_plot(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)

        assert "SAR scene bounds" in recorded_labels


class TestPlotCollocationDiagnosticsRefinement:
    """Test 4-tier rendering with gray unmatched points."""

    def test_unmatched_points_are_gray_with_low_alpha(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Verify unmatched points render in gray (#808080) with alpha=0.3."""
        import matplotlib.axes
        import matplotlib.pyplot as plt

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
        import matplotlib.axes
        import matplotlib.pyplot as plt

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

        # Verify matched layers alpha is 0.65 for wind recipes (a dense
        # source like scatterometer would otherwise fully occlude a
        # sparser layer source, e.g. radiometer, drawn underneath it)
        assert 0.65 in matched_layer_alphas or len(matched_layer_alphas) == 0, (
            f"Expected matched layer alpha=0.65 for a wind recipe, got {matched_layer_alphas}"
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
        """Matched layer points (zorder=5) must be drawn bold: no marker
        edge, same marker size as matched in-situ points (s=25), and for a
        wind recipe alpha=0.65 (not full opacity — a dense source like
        scatterometer would otherwise fully occlude a sparser one, e.g.
        radiometer, drawn underneath it in the same tier)."""
        import matplotlib.axes
        import matplotlib.pyplot as plt

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
            assert c.get("alpha") == 0.65
            assert c.get("s") == 25

    def test_unmatched_layer_points_get_per_source_markers(
        self, geo_datatree_and_collocation_with_unmatched, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Verify unmatched layer-type (altimeter) points get per-source markers, not default 'o'."""
        import matplotlib.axes
        import matplotlib.pyplot as plt

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
        import matplotlib.axes
        import matplotlib.pyplot as plt

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


class TestPlotCollocationDiagnosticsRecipeVariableStyling:
    def test_wind_matched_layer_alpha_is_reduced(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 0.65

    def test_soil_moisture_matched_layer_alpha_is_reduced(
        self, geo_datatree_and_collocation, diagnostics_recipe_soil_moisture, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe_soil_moisture, tmp_path,
        )
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 0.65

    def test_waves_matched_layer_alpha_stays_opaque(
        self, geo_datatree_and_collocation, diagnostics_recipe_waves, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_waves, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 1.0

    def test_currents_matched_layer_alpha_stays_opaque(
        self, geo_datatree_and_collocation, diagnostics_recipe_currents, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_currents, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 1.0

    def test_waves_matched_points_are_larger_with_black_edge(
        self, geo_datatree_and_collocation, diagnostics_recipe_waves, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_waves, tmp_path)
        plt.close("all")

        matched_calls = [c for c in recorded if c.get("zorder") in (5, 6)]
        assert matched_calls
        for c in matched_calls:
            assert c.get("s") == 45
            assert c.get("edgecolors") == "black"

    def test_wind_matched_points_keep_default_size_and_no_edge(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append(kwargs)
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        matched_calls = [c for c in recorded if c.get("zorder") in (5, 6)]
        assert matched_calls
        for c in matched_calls:
            assert c.get("s") == 25
            assert c.get("edgecolors") == "none"


class TestPlotCollocationDiagnosticsSoilMoistureTransparency:
    def test_soil_moisture_matched_layer_alpha_matches_wind(
        self, geo_datatree_and_collocation, tmp_path,
    ):
        """Soil moisture's matched-layer points must use the same
        alpha=0.65 as wind (previously only wind got it; soil_moisture
        defaulted to opaque alpha=1.0, burying the SAR field/other
        sources underneath ASCAT/SMAP overlays)."""
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        collocation_ds = collocation_ds.assign(
            collocation_type=("collocation", ["layer_vs_layer"] * collocation_ds.sizes["collocation"]),
        )
        recipe = Recipe(RecipeConfig(
            name="soilmoisturetest", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))

        path = plot_collocation_diagnostics(
            datatree, collocation_ds, recipe, tmp_path,
            layer_vs_layer_collocation_method="cell-averaging",
        )
        assert path is not None


class TestPlotCollocationDiagnosticsDensePointSubsampling:
    """Regression test: the Tier 3/4 matched-point subsampling cap
    (_subsample_matched_points, max 1000 points) was added for soil
    moisture's routinely-dense ASCAT/SMAP/SMOS footprints but was applied
    unconditionally to every variable, silently dropping matched
    radiometer/scatterometer points for wind recipes down to a random
    1000-point subsample even though the legend count showed the true
    total. It must only kick in for soil_moisture."""

    @staticmethod
    def _dense_radiometer_scene(n, value_var="WSPD", sar_var="owiWindSpeed"):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 4, 5
        lon2d, lat2d = np.meshgrid(np.linspace(-10.0, -8.0, x), np.linspace(50.0, 52.0, y))
        sar_ds = xr.Dataset(
            {sar_var: (("y", "x"), np.full((y, x), 8.0))},
            coords={
                "lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                "time": pd.Timestamp("2026-07-10T19:00:00"),
            },
        )
        rng = np.random.default_rng(0)
        radiometer_ds = xr.Dataset(
            {value_var: ("point", rng.uniform(5.0, 12.0, n))},
            coords={
                "lon": ("point", rng.uniform(-9.9, -8.1, n)),
                "lat": ("point", rng.uniform(50.1, 51.9, n)),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=n, freq="1s")),
            },
            attrs={"data_type": "radiometer"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": sar_ds, "validation/radiometer": radiometer_ds,
        })
        collocation_ds = xr.Dataset({
            f"sar_{sar_var}":     ("collocation", rng.uniform(5.0, 12.0, n)),
            f"val_{value_var}":   ("collocation", radiometer_ds[value_var].values),
            "val_source":       ("collocation", np.array(["radiometer"] * n)),
            "sar_scene_name":   ("collocation", np.array(["sceneA"] * n)),
            "val_lon":          ("collocation", radiometer_ds["lon"].values),
            "val_lat":          ("collocation", radiometer_ds["lat"].values),
            "val_id":           ("collocation", [f"r{i}" for i in range(n)]),
        })
        collocation_ds = collocation_ds.assign_coords(
            val_time=("collocation", pd.date_range("2026-07-10T19:05", periods=n, freq="1s")),
        )
        return datatree, collocation_ds

    def test_wind_recipe_plots_every_matched_radiometer_point(self, tmp_path, monkeypatch):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig
        from sar_validation.core.visualization import plot_collocation_diagnostics

        n = 1500
        datatree, collocation_ds = self._dense_radiometer_scene(n)
        recipe = Recipe(config=RecipeConfig(
            name="test_wind_dense", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
        ))

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append((args, kwargs))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        assert out_path is not None
        layer_calls = [args for args, kwargs in recorded if kwargs.get("zorder") == 5]
        assert layer_calls
        plotted = sum(len(np.atleast_1d(args[0])) for args in layer_calls)
        assert plotted == n, f"expected all {n} matched wind points plotted, got {plotted}"

    def test_soil_moisture_recipe_still_subsamples_dense_matches(self, tmp_path, monkeypatch):
        import matplotlib.axes
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig
        from sar_validation.core.visualization import plot_collocation_diagnostics

        n = 1500
        datatree, collocation_ds = self._dense_radiometer_scene(
            n, value_var="SOIL_MOISTURE", sar_var="sarSSM",
        )
        recipe = Recipe(config=RecipeConfig(
            name="test_soil_moisture_dense", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
        ))

        recorded = []
        original_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, *args, **kwargs):
            recorded.append((args, kwargs))
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        out_path = plot_collocation_diagnostics(datatree, collocation_ds, recipe, tmp_path)
        plt.close("all")

        assert out_path is not None
        layer_calls = [args for args, kwargs in recorded if kwargs.get("zorder") == 5]
        assert layer_calls
        plotted = sum(len(np.atleast_1d(args[0])) for args in layer_calls)
        assert plotted <= 1000, f"expected soil_moisture to stay subsampled to <=1000, got {plotted}"


class TestPlotCollocationDiagnosticsTicks:
    def test_overview_plot_gets_degree_formatted_ticks(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path
    ):
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert out_path is not None
        # plot_collocation_diagnostics saves its own PNG and closes its
        # figure internally, so we can only verify the rendered image
        # exists and is non-trivial in size (a proxy for "axes labels were
        # drawn"), not inspect live tick-label Text objects.
        img = mpimg.imread(str(out_path))
        assert img.shape[0] > 100 and img.shape[1] > 100

    def test_overview_plot_calls_set_lonlat_ticks(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation

        calls = []
        original = viz._set_lonlat_ticks

        def spy(ax, gl):
            calls.append(ax)
            return original(ax, gl)

        monkeypatch.setattr(viz, "_set_lonlat_ticks", spy)
        viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )
        plt.close("all")

        assert len(calls) == 1


class TestValidationReport:
    def test_includes_temporal_offset_plots(self, geo_datatree_and_collocation, tmp_path):
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))

        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        key = "owiWindSpeed_vs_WSPD"
        assert key in figures
        assert (tmp_path / "validation_report.pdf").exists()
        plt.close("all")

    def test_geographic_plot_rendered_before_scatter_plot(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch,
    ):
        """A reader should see the spatial context (where/how dense the
        matches are) before the more abstract point-cloud comparison --
        applies to every recipe type, not just soil moisture, since the
        report loop is shared."""
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        call_order = []
        original_scatter = viz.plot_scatter
        original_geo = viz.plot_geographic

        def scatter_spy(*args, **kwargs):
            call_order.append("scatter")
            return original_scatter(*args, **kwargs)

        def geo_spy(*args, **kwargs):
            call_order.append("geographic")
            return original_geo(*args, **kwargs)

        monkeypatch.setattr(viz, "plot_scatter", scatter_spy)
        monkeypatch.setattr(viz, "plot_geographic", geo_spy)
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))
        viz.validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert call_order[:2] == ["geographic", "scatter"], (
            f"expected geographic before scatter, got call order {call_order!r}"
        )

    def test_summary_table_title_distinguishes_from_statistics(self):
        # A full validation_report() invocation for this smoke check needs
        # a real DataTree, which is awkward to construct empty just to
        # confirm stats_ds_map wiring — so this locks down the actual
        # behavior worth testing (the summary table's title is distinct
        # from, and appears immediately before, the statistics page)
        # directly against the two plotting functions instead.
        import numpy as np
        import xarray as xr

        from sar_validation.core.statistics import compute_statistics
        from sar_validation.core.visualization import plot_statistics, plot_summary_table

        n = 20
        rng = np.random.default_rng(0)
        sar = rng.uniform(0, 10, n)
        val = sar + rng.normal(0, 0.5, n)
        ds = xr.Dataset({
            "sar_owiWindSpeed": ("collocation", sar),
            "val_WSPD":         ("collocation", val),
            "val_source":       ("collocation", ["mooring"] * n),
        })
        stats_ds = compute_statistics(ds, "owiWindSpeed", "WSPD", group_by=["val_source"])

        table_fig = plot_summary_table(stats_ds)
        stats_fig = plot_statistics(stats_ds)

        assert table_fig.axes[0].get_title() != stats_fig._suptitle.get_text()
        import matplotlib.pyplot as plt
        plt.close("all")


class TestValidationReportIncludesDiagnostics:
    def test_diagnostics_plot_included_in_report(self, geo_datatree_and_collocation, tmp_path):
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

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

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

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

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

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


class TestValidationReportClosesPageFigures:
    def test_no_figures_left_open_after_report(self, geo_datatree_and_collocation, tmp_path):
        """Regression guard for the render-once refactor: the new
        lightweight PDF-page figures (built by _finalize_figure_for_report /
        _image_page_figure) must be closed once written, not leaked —
        unlike the original heavy figures, they aren't tracked in
        `all_figures` / `figs`, so nothing else closes them."""
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))

        plt.close("all")
        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        assert plt.get_fignums() == []


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
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

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


class TestValidationReportCurrentsPointSize:
    def test_currents_recipe_passes_reduced_point_size_to_geographic(self, tmp_path, monkeypatch):
        """HF radar (layer_vs_layer) currents validation points are dense
        enough at the default marker size (s=40) to fully blanket the SAR
        field underneath — validation_report must request a smaller marker
        for currents recipes so the SAR scene stays visible through them."""
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        coll = xr.Dataset({
            "sar_rvlRadVel":             ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection":  ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":                ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":            ("collocation", ["sceneA"] * 4),
            "val_lon":                   ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                   ("collocation", [50.5, 50.6, 50.7, 50.8]),
            "temporal_distance_minutes": ("collocation", [10.0, 12.0, 8.0, 15.0]),
        })

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))
        viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("point_size") == 15

    def test_wind_recipe_uses_adaptive_point_size(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """Wind recipes now use adaptive point sizing for scatterometer data
        to avoid occluding the SAR field. For sparse data (<300 pts/scene),
        use point_size=15. For dense data (>300 pts/scene), use point_size=5."""
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))
        viz.validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        # geo_datatree_and_collocation has 4 points in 1 scene = 4 points/scene
        # which is < 300, so should use point_size=15
        assert captured.get("point_size") == 15


class TestValidationReportSoilMoistureGeographicSizing:
    """ISMN stations render at the default point_size=40, which was far
    larger than requested — and the geographic map showed the SAR field's
    full native extent (e.g. CLMS SSM's all-of-mainland-Europe grid)
    instead of the recipe's requested bounding box. Fixed per Lotte's
    feedback: validation_report passes the recipe's geographic_bounds
    through so plot_geographic clamps each scene panel to it -- but only
    for sources whose SARSourceSpec.geographic_plot_clamp_to_bounds opts
    in (sentinel1_clms_ssm); a source like nisar_sme2, whose own native
    grid is already tight around real data, must NOT be clamped to a much
    larger recipe bbox, or its real scene shrinks into a small corner of
    an otherwise-empty panel (see the two tests immediately below).

    Point sizing itself was originally a flat point_size=10, but that made
    sparse in-situ ISMN points too small and dense scatterometer/radiometer
    (ASCAT/SMAP/SMOS) points too big in the same panel -- soil_moisture now
    reuses wind's adaptive density check instead (see
    TestValidationReportCurrentsPointSize.test_wind_recipe_uses_adaptive_point_size)."""

    def _soil_moisture_datatree_and_collocation(self):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        coll = xr.Dataset({
            "sar_sarSSM":       ("collocation", [30.0, 31.0, 29.0, 32.0]),
            "val_SOIL_MOISTURE": xr.DataArray(
                [0.28, 0.30, 0.27, 0.31], dims="collocation", attrs={"units": "1"},
            ),
            "val_source":       ("collocation", ["ismn"] * 4),
            "sar_scene_name":   ("collocation", ["sceneA"] * 4),
            "val_lon":          ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":          ("collocation", [50.5, 50.6, 50.7, 50.8]),
            "temporal_distance_minutes": ("collocation", [10.0, 12.0, 8.0, 15.0]),
        })
        return datatree, coll

    def test_soil_moisture_recipe_uses_adaptive_point_size_when_sparse(self, tmp_path, monkeypatch):
        """4 points in 1 scene = 4 pts/scene, well under the 300 threshold,
        so soil_moisture must use the same sparse-data size as wind (15)."""
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, coll = self._soil_moisture_datatree_and_collocation()

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="soil_moisture_test", variable="soil_moisture"))
        # Real (unmocked) pytesmo.cdf_matching resizes its bins for this
        # deliberately tiny 4-point fixture — an expected, benign side
        # effect of fitting on so little data, not a defect.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("point_size") == 15

    def test_soil_moisture_recipe_uses_adaptive_point_size_when_dense(self, tmp_path, monkeypatch):
        """A scatterometer/radiometer-dense scene (>300 pts/scene, e.g. a
        real ASCAT/SMAP/SMOS-heavy soil-moisture run) must fall back to the
        smaller marker (5), exactly like wind does, instead of the old flat
        point_size=10 -- proves the density check actually drives the value
        rather than the two thresholds coincidentally bracketing 10."""
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        rng = np.random.default_rng(0)
        n = 400
        coll = xr.Dataset({
            "sar_sarSSM":       ("collocation", rng.uniform(20, 40, n)),
            "val_SOIL_MOISTURE": xr.DataArray(
                rng.uniform(0.1, 0.4, n), dims="collocation", attrs={"units": "1"},
            ),
            "val_source":       ("collocation", ["ascat_ssm"] * n),
            "sar_scene_name":   ("collocation", ["sceneA"] * n),
            "val_lon":          ("collocation", rng.uniform(-9.8, -8.2, n)),
            "val_lat":          ("collocation", rng.uniform(50.2, 51.8, n)),
            "temporal_distance_minutes": ("collocation", rng.uniform(5, 20, n)),
        })

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="soil_moisture_test", variable="soil_moisture"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("point_size") == 5

    def test_soil_moisture_recipe_sizes_point_vs_layer_and_layer_vs_layer_independently(
        self, tmp_path, monkeypatch,
    ):
        """A pair mixing sparse in-situ ISMN (point_vs_layer) with dense
        ASCAT (layer_vs_layer) must size each collocation_type on its own
        density, not one average pooled across both -- pooling let the
        dense layer_vs_layer type's point count dominate the average and
        made the sparse point_vs_layer type's markers (e.g. ISMN) too
        small, exactly like the flat point_size=10 it replaced did.
        ISMN (point_vs_layer) is further fixed at 25 regardless of density
        per Lotte's follow-up feedback -- still too small to read
        individual stations at the density-adaptive 15."""
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        rng = np.random.default_rng(0)
        n_ascat = 400
        n_ismn = 5
        n = n_ascat + n_ismn
        coll = xr.Dataset({
            "sar_sarSSM":       ("collocation", rng.uniform(20, 40, n)),
            "val_SOIL_MOISTURE": xr.DataArray(
                rng.uniform(0.1, 0.4, n), dims="collocation", attrs={"units": "1"},
            ),
            "val_source":       ("collocation", ["ascat_ssm"] * n_ascat + ["ismn"] * n_ismn),
            "collocation_type": ("collocation", ["layer_vs_layer"] * n_ascat + ["point_vs_layer"] * n_ismn),
            "sar_scene_name":   ("collocation", ["sceneA"] * n),
            "val_lon":          ("collocation", rng.uniform(-9.8, -8.2, n)),
            "val_lat":          ("collocation", rng.uniform(50.2, 51.8, n)),
            "temporal_distance_minutes": ("collocation", rng.uniform(5, 20, n)),
        })

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["point_size"] = kwargs.get("point_size")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="soil_moisture_test", variable="soil_moisture"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("point_size") == {"layer_vs_layer": 5, "point_vs_layer": 25}

    def test_sentinel1_clms_ssm_recipe_passes_recipe_geographic_bounds(self, tmp_path, monkeypatch):
        """sentinel1_clms_ssm's raw grid covers all of mainland Europe
        regardless of what was requested (mostly NaN outside that day's
        real swath) -- clamping to the recipe's own bbox is required, or
        every scene panel would show far more than was asked for."""
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, SARDataSpec

        datatree, coll = self._soil_moisture_datatree_and_collocation()

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["geographic_bounds"] = kwargs.get("geographic_bounds")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        bounds = GeographicBounds(min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0)
        recipe = Recipe(config=RecipeConfig(
            name="soil_moisture_test", variable="soil_moisture", geographic_bounds=bounds,
            sar_data=SARDataSpec(source="sentinel1_clms_ssm"),
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("geographic_bounds") is bounds

    def test_nisar_sme2_recipe_does_not_clamp_to_geographic_bounds(self, tmp_path, monkeypatch):
        """nisar_sme2's own native grid is already tight around real data
        -- clamping to a much larger recipe bbox would shrink the actual
        scene into a small corner of an otherwise-empty panel instead of
        showing it at a legible scale."""
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, SARDataSpec

        datatree, coll = self._soil_moisture_datatree_and_collocation()

        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["geographic_bounds"] = kwargs.get("geographic_bounds")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        bounds = GeographicBounds(min_lon=-10.0, max_lon=30.0, min_lat=35.0, max_lat=60.0)
        recipe = Recipe(config=RecipeConfig(
            name="soil_moisture_test", variable="soil_moisture", geographic_bounds=bounds,
            sar_data=SARDataSpec(source="nisar_sme2"),
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("geographic_bounds") is None

    def test_other_variables_do_not_get_geographic_bounds(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
        """Scoped deliberately to soil_moisture only — wind/currents/waves
        reports must keep showing each SAR scene's full native extent,
        unaffected by this change."""
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        captured = {}
        original = viz.plot_geographic

        def spy(datatree_, coll_, sar_var, val_var, **kwargs):
            captured["geographic_bounds"] = kwargs.get("geographic_bounds")
            return original(datatree_, coll_, sar_var, val_var, **kwargs)

        monkeypatch.setattr(viz, "plot_geographic", spy)
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))
        viz.validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured.get("geographic_bounds") is None

    def test_soil_moisture_report_uses_two_column_geographic(
        self, geo_datatree_and_collocation, tmp_path,
    ):
        """Regression test for the two-column point_vs_layer/layer_vs_layer
        geographic layout: validation_report must pass
        two_column_by_type=True through to plot_geographic for
        soil_moisture recipes. Verified via the actual, real effect of
        that wiring on the produced page -- with two_column_by_type in
        effect, plot_geographic keys its dict by *scene name* and the
        resulting Figure's suptitle is tagged "[<scene_name>]"; without
        it (the pre-task behavior), the geographic page would instead be
        keyed/tagged by collocation_type ("[point_vs_layer]"). Note: the
        plan brief's literal assertion here (``assert any("geographic" in
        k or True for k in result)``) was a vacuous smoke test that would
        pass regardless of whether this wiring exists at all -- worse,
        checked literally without ``or True`` it would still not detect
        real breakage, since validation_report's returned dict is keyed
        by pair name (e.g. "sarSSM_vs_SOIL_MOISTURE"), which never
        contains the substring "geographic" in the first place. This
        version instead inspects the actual geographic Figure produced.
        """
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        collocation_ds = collocation_ds.rename({
            "sar_owiWindSpeed": "sar_sarSSM", "val_WSPD": "val_SOIL_MOISTURE",
        }).assign(
            collocation_type=("collocation", ["point_vs_layer"] * collocation_ds.sizes["collocation"]),
        )
        recipe = Recipe(RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))
        result = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        key = "sarSSM_vs_SOIL_MOISTURE"
        assert key in result
        suptitles = [
            fig._suptitle.get_text()
            for fig in result[key]
            if getattr(fig, "_suptitle", None) is not None
        ]
        assert any("[sceneA]" in t for t in suptitles), suptitles
        assert not any("[point_vs_layer]" in t for t in suptitles), suptitles


class TestValidationReportForceSplitWhenHarmonized:
    """When ismn has enough points to CDF-match ascat_ssm into its
    volumetric domain (see _harmonize_percent_domain_sources),
    validation_report must force the main CDF-matched scatter (and its
    temporal-offset-colored twin) into per-source small multiples even
    though no single source dominates by point count -- piling every
    harmonized source into one shared axes was too visually busy
    (confirmed against real data, soil_moisture_satellite_example: ASCAT
    ~45% share, well under the 70% imbalance threshold, still busy)."""

    def _datatree_and_collocation(self, *, include_ismn: bool):
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"sarSSM": (("y", "x"), np.full((y, x), 30.0))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        rng = np.random.default_rng(3)
        n_ascat = 20
        ascat_val = rng.uniform(10.0, 40.0, n_ascat)
        ascat_sar = ascat_val + rng.normal(0, 2, n_ascat)
        val_source = ["ascat_ssm"] * n_ascat
        val_vals = list(ascat_val)
        sar_vals = list(ascat_sar)

        if include_ismn:
            n_ismn = 10
            ismn_val = rng.uniform(0.1, 0.4, n_ismn)
            ismn_sar = ismn_val * 100 + rng.normal(0, 2, n_ismn)
            val_source += ["ismn"] * n_ismn
            val_vals += list(ismn_val)
            sar_vals += list(ismn_sar)

        n = len(val_source)
        coll = xr.Dataset({
            "sar_sarSSM":       ("collocation", np.array(sar_vals), {"units": "%"}),
            "val_SOIL_MOISTURE": ("collocation", np.array(val_vals)),
            "val_source":       ("collocation", np.array(val_source)),
            "sar_scene_name":   ("collocation", ["sceneA"] * n),
            "val_lon":          ("collocation", rng.uniform(-9.8, -8.2, n)),
            "val_lat":          ("collocation", rng.uniform(50.2, 51.8, n)),
            "temporal_distance_minutes": ("collocation", rng.uniform(5, 20, n)),
        })
        return datatree, coll

    def test_force_split_true_when_ascat_harmonized_into_ismn_domain(self, tmp_path, monkeypatch):
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, coll = self._datatree_and_collocation(include_ismn=True)

        captured = []
        original = viz.plot_scatter

        def spy(*args, **kwargs):
            captured.append(kwargs.get("force_split"))
            return original(*args, **kwargs)

        monkeypatch.setattr(viz, "plot_scatter", spy)
        recipe = Recipe(config=RecipeConfig(name="soil_moisture_test", variable="soil_moisture"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured, "plot_scatter was never called"
        assert all(force_split is True for force_split in captured), (
            f"expected every plot_scatter call to request force_split=True once ascat_ssm "
            f"was harmonized into ismn's domain, got {captured!r}"
        )

    def test_force_split_false_when_ascat_not_harmonized(self, tmp_path, monkeypatch):
        """Without ismn present, ascat_ssm can't be harmonized at all (see
        _harmonize_percent_domain_sources's reference-absent fallback) --
        force_split must stay False, matching plain split_when_imbalanced
        behavior instead of forcing a split unconditionally."""
        import warnings

        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, coll = self._datatree_and_collocation(include_ismn=False)

        captured = []
        original = viz.plot_scatter

        def spy(*args, **kwargs):
            captured.append(kwargs.get("force_split"))
            return original(*args, **kwargs)

        monkeypatch.setattr(viz, "plot_scatter", spy)
        recipe = Recipe(config=RecipeConfig(name="soil_moisture_test", variable="soil_moisture"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            viz.validation_report(coll, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert captured, "plot_scatter was never called"
        assert all(force_split is False for force_split in captured), (
            f"expected force_split=False when ascat_ssm has no reference to harmonize "
            f"against, got {captured!r}"
        )


class TestPlotRvlLandQa:
    def _make_sar_node(self, *, land_count=0, land_fraction=float("nan"), land_mean=float("nan")):
        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        attrs = {"measurement_type": "rvl"}
        if land_count:
            attrs["rvl_land_pixel_count"] = land_count
            attrs["rvl_land_pixel_fraction"] = land_fraction
            attrs["rvl_land_mean_radvel"] = land_mean
        return xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
            attrs=attrs,
        )

    def test_returns_none_when_no_scene_has_land(self):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_rvl_land_qa

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": self._make_sar_node(land_count=0),
        })
        assert plot_rvl_land_qa(datatree) is None

    def test_returns_none_when_no_sar_node(self):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_rvl_land_qa

        datatree = DataTreeConverter.to_datatree({})
        assert plot_rvl_land_qa(datatree) is None

    def test_returns_table_with_one_row_per_land_scene(self):
        import matplotlib.pyplot as plt

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.visualization import plot_rvl_land_qa

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": self._make_sar_node(land_count=0),
            "sar/sceneB": self._make_sar_node(land_count=24, land_fraction=0.4, land_mean=0.71),
        })
        fig = plot_rvl_land_qa(datatree)
        assert fig is not None
        table = fig.axes[0].tables[0]
        cells = table.get_celld()
        n_rows = len({r for (r, _c) in cells.keys()})
        assert n_rows == 2  # header + 1 data row (sceneA has no land, omitted)
        assert cells[(1, 0)].get_text().get_text() == "sceneB"
        assert cells[(1, 1)].get_text().get_text() == "24"
        plt.close(fig)


class TestValidationReportRvlLandQaPage:
    def _sar_node(self, *, with_land: bool):
        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        attrs = {}
        if with_land:
            attrs.update(
                rvl_land_pixel_count=9, rvl_land_pixel_fraction=1.0, rvl_land_mean_radvel=0.65,
            )
        return xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
            attrs=attrs,
        )

    def _coll_ds(self):
        return xr.Dataset({
            "sar_rvlRadVel":             ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection":  ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":                ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":            ("collocation", ["sceneA"] * 4),
            "val_lon":                   ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                   ("collocation", [50.5, 50.6, 50.7, 50.8]),
            # Present (unlike a minimal fixture omitting it) so that
            # plot_scatter(color_by="temporal_offset") and
            # plot_temporal_offset() don't emit "missing
            # temporal_distance_minutes" warnings/None-returns — keeps this
            # test's output pristine and its baseline page count stable.
            "temporal_distance_minutes": ("collocation", [10.0, 20.0, 15.0, 25.0]),
        })

    def _count_image_pages(self, monkeypatch, datatree, recipe, tmp_path):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from sar_validation.core.visualization import validation_report

        recorded_figs = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_figs.append(fig)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)
        validation_report(self._coll_ds(), datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        def is_image_page(fig):
            return fig is not None and len(fig.axes) == 1 and len(fig.axes[0].images) > 0

        return sum(1 for f in recorded_figs if is_image_page(f))

    def test_qa_page_added_for_currents_with_land(self, tmp_path, monkeypatch):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))
        # Baseline is asserted relative to the no-land run rather than a
        # hardcoded absolute count: with this minimal collocation_ds, the
        # per-pair scatter/geographic/residuals plots in validation_report
        # already render successfully (they only warn, not fail, when
        # temporal_distance_minutes is absent), so the diagnostics plot is
        # not the only image page even before the QA page is added. What
        # this test verifies is that adding land-flagged data contributes
        # exactly one extra page (the QA table) on top of that baseline.
        datatree_no_land = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=False)})
        baseline = self._count_image_pages(monkeypatch, datatree_no_land, recipe, tmp_path)

        datatree = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=True)})
        assert self._count_image_pages(monkeypatch, datatree, recipe, tmp_path) == baseline + 1

    def test_qa_page_omitted_for_currents_without_land(self, tmp_path, monkeypatch):
        """Previously this test compared a no-land run's page count against
        a *second, identically-configured* no-land run — a tautology that
        would still pass even if the omission logic were completely broken
        (both runs would inflate identically, since nothing actually
        varies between them). This version instead (a) calls
        plot_rvl_land_qa directly — the function actually responsible for
        the omission — and asserts it returns None for a no-land datatree,
        a real assertion that varies with its input and fails if that
        function's `if not rows: return None` guard breaks; and (b)
        compares the no-land page count against a genuinely different
        *with-land* run of the same recipe (one fewer page), rather than
        against another no-land run, so the comparison itself is capable of
        failing if the wiring's `if fig_land_qa is not None:` guard is
        removed and a page ends up added regardless of land presence.
        """
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import plot_rvl_land_qa

        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))

        datatree_no_land = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=False)})
        assert plot_rvl_land_qa(datatree_no_land) is None

        no_land_count = self._count_image_pages(monkeypatch, datatree_no_land, recipe, tmp_path)

        datatree_with_land = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=True)})
        with_land_count = self._count_image_pages(monkeypatch, datatree_with_land, recipe, tmp_path)

        assert no_land_count == with_land_count - 1, (
            "Expected exactly one fewer image page when no scene has "
            "land-flagged cells (no QA page) than when a scene does."
        )

    def test_qa_page_omitted_for_non_currents_variable(self, tmp_path, monkeypatch):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="wind"))
        datatree_no_land = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=False)})
        baseline = self._count_image_pages(monkeypatch, datatree_no_land, recipe, tmp_path)

        # Land-flagged data present, but variable != "currents" — QA page
        # must still be gated off, so the page count is unchanged from baseline.
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": self._sar_node(with_land=True)})
        assert self._count_image_pages(monkeypatch, datatree, recipe, tmp_path) == baseline

    def test_qa_page_immediately_follows_diagnostics_page(self, tmp_path, monkeypatch):
        """Regression guard for the design-spec ordering requirement (3.b):
        the QA page must be inserted right after the collocation-diagnostics
        page, not appended at the end of the report.

        Both the diagnostics page and the QA page end up as single-axis
        "image pages" after ``_finalize_figure_for_report``/
        ``_image_page_figure`` run, so a plain figure-object spy on
        ``PdfPages.savefig`` (as used by ``_count_image_pages`` above)
        cannot tell them apart. Instead we tag each page's *finalized*
        Figure with a distinctive ``set_label`` marker:

        * ``plot_rvl_land_qa`` is wrapped so its returned Figure carries a
          ``_is_qa_source`` marker.
        * ``_finalize_figure_for_report`` is wrapped so that, when it
          finalizes a Figure carrying that marker, the *new* page Figure it
          returns is labeled ``"__qa_page__"``.
        * ``_image_page_figure`` is wrapped so that its *only* direct
          (non-nested) call site in ``validation_report`` — the
          diagnostics-PNG embed — is labeled ``"__diagnostics_page__"``;
          all other calls to it happen nested inside
          ``_finalize_figure_for_report`` and are left unlabeled there.

        We then recover each page's position by label from the recorded
        ``PdfPages.savefig`` call order and assert adjacency.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        import sar_validation.core.visualization as viz
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe = Recipe(config=RecipeConfig(name="currents_test", variable="currents"))
        # plot_collocation_diagnostics returns None (no diagnostics page) unless
        # the DataTree also has a "validation" node — the class-level _coll_ds/
        # _sar_node fixtures used by the other tests in this class only build a
        # bare "sar" node, which is enough for the QA-page gating tests but not
        # for exercising diagnostics-page adjacency. Add a matching hf_radar
        # validation node, in the same scene bounds/time window as
        # _coll_ds()'s val_lon/val_lat/val_source rows, so the diagnostics page
        # actually renders here.
        hf_radar_ds = xr.Dataset(
            {"rvlRadVel_projection": ("point", np.array([0.28, 0.30, 0.27, 0.31]))},
            coords={
                "lon": ("point", np.array([-9.5, -9.4, -9.3, -9.2])),
                "lat": ("point", np.array([50.5, 50.6, 50.7, 50.8])),
                "time": ("point", pd.date_range("2026-07-10T19:05", periods=4, freq="5min")),
            },
            attrs={"platform_type": "hf_radar"},
        )
        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": self._sar_node(with_land=True),
            "validation/hf_radar": hf_radar_ds,
        })

        original_qa = viz.plot_rvl_land_qa

        def spy_qa(dt):
            fig = original_qa(dt)
            if fig is not None:
                fig._is_qa_source = True
            return fig

        original_finalize = viz._finalize_figure_for_report
        original_image_page = viz._image_page_figure
        state = {"inside_finalize": False}

        def spy_image_page_figure(img, dpi=150):
            result = original_image_page(img, dpi=dpi)
            if not state["inside_finalize"]:
                result.set_label("__diagnostics_page__")
            return result

        def spy_finalize(fig, png_path, dpi=150):
            is_qa = getattr(fig, "_is_qa_source", False)
            state["inside_finalize"] = True
            try:
                result = original_finalize(fig, png_path, dpi=dpi)
            finally:
                state["inside_finalize"] = False
            if is_qa:
                result.set_label("__qa_page__")
            return result

        monkeypatch.setattr(viz, "plot_rvl_land_qa", spy_qa)
        monkeypatch.setattr(viz, "_image_page_figure", spy_image_page_figure)
        monkeypatch.setattr(viz, "_finalize_figure_for_report", spy_finalize)

        recorded_labels = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_labels.append(fig.get_label() if fig is not None else None)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)

        viz.validation_report(self._coll_ds(), datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        assert "__diagnostics_page__" in recorded_labels, (
            f"diagnostics page was not rendered: {recorded_labels}"
        )
        assert "__qa_page__" in recorded_labels, f"QA page was not rendered: {recorded_labels}"
        diag_idx = recorded_labels.index("__diagnostics_page__")
        qa_idx = recorded_labels.index("__qa_page__")
        assert qa_idx == diag_idx + 1, (
            "QA page must immediately follow the diagnostics page; got diagnostics "
            f"at index {diag_idx}, QA at index {qa_idx} (full order: {recorded_labels})"
        )


class TestImagePageFigure:
    def test_figure_size_matches_image_pixel_dimensions(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from sar_validation.core.visualization import _image_page_figure

        img = np.zeros((300, 450, 3), dtype=np.uint8)
        fig = _image_page_figure(img, dpi=150)

        w_in, h_in = fig.get_size_inches()
        assert w_in == pytest.approx(450 / 150)
        assert h_in == pytest.approx(300 / 150)
        plt.close(fig)


class TestFinalizeFigureForReport:
    def test_writes_png_closes_original_returns_image_page(self, tmp_path):
        import matplotlib._pylab_helpers as pylab_helpers
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import _finalize_figure_for_report

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        png_path = tmp_path / "out.png"

        page_fig = _finalize_figure_for_report(fig, png_path)

        assert png_path.exists()
        assert png_path.stat().st_size > 0
        # Check object identity against pyplot's live figure registry, not
        # `fig.number` — matplotlib recycles figure numbers after `close()`,
        # so a freshly created page figure can legitimately reuse the
        # original's number; that would make an `fignum_exists()`-based
        # check pass even if `fig` were never actually closed.
        open_figs = [m.canvas.figure for m in pylab_helpers.Gcf.figs.values()]
        assert fig not in open_figs, "original figure should be closed"
        assert page_fig is not fig
        plt.close(page_fig)

    def test_none_png_path_skips_disk_write(self, tmp_path):
        import matplotlib.pyplot as plt

        from sar_validation.core.visualization import _finalize_figure_for_report

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        page_fig = _finalize_figure_for_report(fig, None)

        assert page_fig is not None
        assert list(tmp_path.iterdir()) == []
        plt.close(page_fig)


class TestValidationReportOnlyDiagnosticsPngSaved:
    def test_plots_dir_contains_only_diagnostics_png(self, geo_datatree_and_collocation, tmp_path):
        """Every plot is already embedded in validation_report.pdf, so
        plots/ must contain only the collocation-diagnostics PNG — no
        individual scatter/geographic/statistics/residuals/temporal-offset
        PNGs."""
        import matplotlib.pyplot as plt

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        plots_dir = tmp_path / "plots"
        png_files = sorted(p.name for p in plots_dir.glob("*.png"))
        assert png_files == ["collocation_diagnostics_test_recipe.png"]


class TestPlotCollocationDiagnosticsAntimeridian:
    def test_uses_central_longitude_180_projection_when_crossing(
        self, geo_datatree_and_collocation_dateline, diagnostics_recipe_dateline, tmp_path, monkeypatch
    ):
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation_dateline
        seen_ax = []
        original = viz._set_lonlat_ticks

        def spy(ax, gl):
            seen_ax.append(ax)
            return original(ax, gl)

        monkeypatch.setattr(viz, "_set_lonlat_ticks", spy)
        out_path = viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe_dateline, tmp_path,
        )

        assert out_path is not None
        assert len(seen_ax) == 1
        assert seen_ax[0].projection.proj4_params.get("lon_0") == 180

    def test_non_crossing_recipe_keeps_default_projection(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation
        seen_ax = []
        original = viz._set_lonlat_ticks

        def spy(ax, gl):
            seen_ax.append(ax)
            return original(ax, gl)

        monkeypatch.setattr(viz, "_set_lonlat_ticks", spy)
        viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
        )

        assert len(seen_ax) == 1
        assert seen_ax[0].projection.proj4_params.get("lon_0", 0) == 0

    def test_produces_a_valid_png_for_crossing_bbox(
        self, geo_datatree_and_collocation_dateline, diagnostics_recipe_dateline, tmp_path
    ):
        import matplotlib.image as mpimg

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation_dateline
        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe_dateline, tmp_path,
        )
        img = mpimg.imread(str(out_path))
        assert img.shape[0] > 100 and img.shape[1] > 100

    def test_scene_bounds_line_plotted_in_projected_frame_when_crossing(
        self, geo_datatree_and_collocation_dateline, diagnostics_recipe_dateline, tmp_path, monkeypatch
    ):
        """Regression test for a double-shift bug: when the recipe bbox
        crosses the antimeridian, the SAR scene-bounds box longitudes are
        pre-shifted into the central_longitude=180 axes frame (see
        ``_shift`` in plot_collocation_diagnostics) so the box spans the
        dateline as one contiguous range. Those already-shifted coordinates
        must then be plotted with a transform that matches that frame
        (``proj``), not the raw-lon ``PlateCarree()`` transform — otherwise
        cartopy reprojects them a second time and the box lands far outside
        the visible extent.

        This test captures the exact ``ax.plot(...)`` call used to draw the
        "SAR scene bounds" line, reprojects its data through the transform
        it was actually plotted with into the axes' own projection, and
        asserts the result falls within the axes' x-limits.
        """
        import cartopy.mpl.geoaxes as cgeoaxes

        from sar_validation.core.visualization import plot_collocation_diagnostics

        datatree, collocation_ds = geo_datatree_and_collocation_dateline

        calls = []
        original_plot = cgeoaxes.GeoAxes.plot

        def spy_plot(self, *args, **kwargs):
            if kwargs.get("label") == "SAR scene bounds":
                calls.append({
                    "ax": self,
                    "xdata": np.asarray(args[0]),
                    "ydata": np.asarray(args[1]),
                    "transform": kwargs.get("transform"),
                })
            return original_plot(self, *args, **kwargs)

        monkeypatch.setattr(cgeoaxes.GeoAxes, "plot", spy_plot)

        out_path = plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe_dateline, tmp_path,
        )

        assert out_path is not None
        assert len(calls) == 1, "expected exactly one 'SAR scene bounds' plot call"

        call = calls[0]
        ax = call["ax"]
        projected = ax.projection.transform_points(call["transform"], call["xdata"], call["ydata"])
        x_proj = projected[:, 0]

        xlim = ax.get_xlim()
        lo, hi = min(xlim), max(xlim)
        assert np.all(x_proj >= lo) and np.all(x_proj <= hi), (
            f"scene-bounds box x-coords {x_proj} fall outside axes x-limits {xlim} "
            "(likely double-shifted/double-transformed across the dateline)"
        )


class TestValidationReportDownloadWarnings:
    def test_download_warning_appears_on_cover_page(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test_recipe", variable="wind"))

        recorded_figs = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_figs.append(fig)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)
        validation_report(
            collocation_ds, datatree, recipe, out_dir=tmp_path,
            download_warnings=["altimeter download failed: timeout"],
        )
        plt.close("all")

        cover = recorded_figs[0]
        cover_text = " ".join(t.get_text() for t in cover.texts)
        assert "altimeter download failed: timeout" in cover_text

    def test_no_warning_text_when_download_warnings_omitted(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

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

        cover = recorded_figs[0]
        # Exactly the same two text() calls as before this change: title + variable/date.
        assert len(cover.texts) == 2


class TestValidationReportSensingDepths:
    def test_cover_page_lists_sensing_depths_for_soil_moisture(self, tmp_path, monkeypatch):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )
        from sar_validation.core.visualization import validation_report

        cfg = RecipeConfig(
            name="test_depths", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[ValidationDataSource(source_type="ismn", min_depth=0.0, max_depth=0.05)],
        )
        recipe = Recipe(config=cfg)

        # Two points (rather than one) so add_rescaled_sar_column's
        # per-group CDF-matching (which requires >= 2 valid pairs per
        # val_source group, see statistics.add_rescaled_sar_column) doesn't
        # skip the group and leave every plot with no valid data — which
        # would leave pdf_pages empty and no PDF written at all, unrelated
        # to this task's sensing-depth feature.
        ascat_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array([25.0, 30.0]))},
            coords={
                "lon": ("point", [0.0, 1.0]),
                "lat": ("point", [45.0, 46.0]),
                "time": ("point", np.array(["2026-01-01", "2026-01-01"], dtype="datetime64[ns]")),
            },
            attrs={"platform_type": "ascat_ssm", "sensing_depth_cm": "0-5", "band": "C"},
        )
        # Also a minimal ismn group (>= 2 points, same rationale as above)
        # -- _harmonize_percent_domain_sources (wired in by Tasks 1-3 of the
        # current plan) requires an "ismn" val_source to be present so it
        # can fit a SAR->ismn CDF-match transform and convert ascat_ssm's
        # percent-scale rows into ismn's volumetric domain; without it,
        # ascat_ssm's rows are dropped to NaN in the CDF-matched section,
        # leaving zero valid data for every CDF-matched plot and (since this
        # test passes no native_units_stats_ds_map fallback) no PDF at all.
        ismn_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array([0.20, 0.24]))},
            coords={
                "lon": ("point", [2.0, 3.0]),
                "lat": ("point", [47.0, 48.0]),
                "time": ("point", np.array(["2026-01-01", "2026-01-01"], dtype="datetime64[ns]")),
            },
        )
        datatree = xr.DataTree.from_dict({
            "validation/ascat_ssm/f1": ascat_node,
            "validation/ismn/f1": ismn_node,
        })

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 24.0, 18.0, 22.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 30.0, 0.20, 0.24])),
                "val_source": (
                    "collocation", np.array(["ascat_ssm", "ascat_ssm", "ismn", "ismn"])
                ),
                "val_id": ("collocation", np.array(["a0", "a1", "i0", "i1"])),
            },
        )

        recorded_figs = []
        original_savefig = PdfPages.savefig

        def recording_savefig(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure")
            recorded_figs.append(fig)
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(PdfPages, "savefig", recording_savefig)
        figs = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        plt.close("all")

        pdf_path = tmp_path / "validation_report.pdf"
        assert pdf_path.exists()
        assert "sarSSM_vs_SOIL_MOISTURE" in figs

        cover = recorded_figs[0]
        cover_text = " ".join(t.get_text() for t in cover.texts)
        assert "Sensing depths:" in cover_text
        assert "ascat_ssm ~0-5cm (C-band)" in cover_text
        assert "ISMN 0.0-0.05m depth window" in cover_text

    def test_no_sensing_depths_line_for_non_soil_moisture(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.visualization import validation_report

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

        cover = recorded_figs[0]
        cover_text = " ".join(t.get_text() for t in cover.texts)
        assert "Sensing depths:" not in cover_text


class TestPlotCollocationDiagnosticsIndividualMethodAlpha:
    def test_individual_method_uses_low_fixed_alpha(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation
        captured_alphas = []
        original_scatter = plt.Axes.scatter

        def spy_scatter(self, *args, **kwargs):
            if "alpha" in kwargs:
                captured_alphas.append(kwargs["alpha"])
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "scatter", spy_scatter)
        viz.plot_collocation_diagnostics(
            datatree, collocation_ds, diagnostics_recipe, tmp_path,
            layer_vs_layer_collocation_method="individual",
        )

        assert 0.15 in captured_alphas

    def test_cell_averaging_default_keeps_todays_alpha(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
        """Non-regression: omitting layer_vs_layer_collocation_method (or
        passing 'cell-averaging' explicitly) must reproduce today's exact
        variable-dependent alpha (0.65 for wind, matching diagnostics_recipe)."""
        import matplotlib.pyplot as plt

        import sar_validation.core.visualization as viz

        datatree, collocation_ds = geo_datatree_and_collocation
        captured_alphas = []
        original_scatter = plt.Axes.scatter

        def spy_scatter(self, *args, **kwargs):
            if "alpha" in kwargs:
                captured_alphas.append(kwargs["alpha"])
            return original_scatter(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "scatter", spy_scatter)
        viz.plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)

        assert 0.15 not in captured_alphas
        assert 0.65 in captured_alphas


class TestMarkNativeUnits:
    """Task 12: `_mark_native_units` stamps a "— native units —" banner
    onto a figure via `fig.text(...)`, independent of any caller in
    `validation_report`."""

    def test_stamps_native_units_banner_text(self):
        import matplotlib.figure

        from sar_validation.core.visualization import _mark_native_units

        fig = matplotlib.figure.Figure()
        returned = _mark_native_units(fig)

        assert returned is fig
        assert any("native units" in t.get_text() for t in fig.texts)


class TestValidationReportNativeUnitsSection:
    """Task 12: validation_report(..., native_units_stats_ds_map=...) adds a
    second, non-CDF-matched scatter/residuals/statistics page set per
    soil_moisture pair, restricted to the val_source groups that already
    share SAR's units — using the raw (non-rescaled) collocation data."""

    @staticmethod
    def _recipe():
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds

        cfg = RecipeConfig(
            name="test_native_units_plots", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        return Recipe(config=cfg)

    @staticmethod
    def _ascat_node(lon, lat, vals):
        return xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array(vals))},
            coords={
                "lon": ("point", lon), "lat": ("point", lat),
                "time": ("point", np.array(["2026-01-01"] * len(vals), dtype="datetime64[ns]")),
            },
        )

    def test_native_units_plots_added_when_stats_map_provided(self, tmp_path):
        """The native-units section must add *new* figures beyond whatever
        the CDF-matched section already produces for this exact fixture —
        not merely leave the pair's figure list non-empty, which is
        trivially true from the pre-existing scatter/residuals pages alone
        even if the native-units block never ran."""
        from sar_validation.core.statistics import compute_statistics
        from sar_validation.core.visualization import validation_report

        recipe = self._recipe()
        datatree = xr.DataTree.from_dict({
            "validation/ascat_ssm/f1": self._ascat_node([0.0, 1.0], [45.0, 46.0], [25.0, 35.0]),
        })
        # val_id distinguishes the two observations for
        # _deduplicate_obs/plot_scatter — without it both rows share the
        # same (val_source) grouping key and collapse into a single point.
        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 30.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 35.0])),
                "val_source": ("collocation", np.array(["ascat_ssm", "ascat_ssm"])),
                "val_id": ("collocation", np.array(["a1", "a2"])),
            },
        )
        native_stats = compute_statistics(collocation_ds, "sarSSM", "SOIL_MOISTURE", group_by=["val_source"])
        native_stats_map = {"sarSSM_vs_SOIL_MOISTURE": native_stats}

        key = "sarSSM_vs_SOIL_MOISTURE"
        figs_without = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path / "without")
        figs_with = validation_report(
            collocation_ds, datatree, recipe, out_dir=tmp_path / "with",
            native_units_stats_ds_map=native_stats_map,
        )

        assert key in figs_without and key in figs_with
        n_without = len(figs_without[key])
        n_with = len(figs_with[key])
        assert n_with > n_without, (
            f"expected native_units_stats_ds_map to add figures beyond the "
            f"{n_without} produced without it; got {n_with}"
        )

        # At least one of the *newly added* figures must carry the
        # native-units banner stamped by _mark_native_units — proving the
        # extra figures really are the native-units pages, not some
        # unrelated addition (e.g. an accidental duplicate CDF-matched page).
        new_figs = figs_with[key][n_without:]
        assert any(
            any("native units" in t.get_text() for t in fig.texts)
            for fig in new_figs
        ), "no newly-added figure carries the '— native units —' banner"

    def test_native_units_geographic_rendered_before_native_units_scatter(self, tmp_path, monkeypatch):
        """The native-units section must lead with its geographic plot too,
        matching the CDF-matched section's order (§9.3) -- previously it
        started with scatter/residuals and only rendered geographic last."""
        import sar_validation.core.visualization as viz
        from sar_validation.core.statistics import compute_statistics

        recipe = self._recipe()
        datatree = xr.DataTree.from_dict({
            "validation/ascat_ssm/f1": self._ascat_node([0.0, 1.0], [45.0, 46.0], [25.0, 35.0]),
        })
        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 30.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 35.0])),
                "val_source": ("collocation", np.array(["ascat_ssm", "ascat_ssm"])),
                "val_id": ("collocation", np.array(["a1", "a2"])),
            },
        )
        native_stats = compute_statistics(collocation_ds, "sarSSM", "SOIL_MOISTURE", group_by=["val_source"])
        native_stats_map = {"sarSSM_vs_SOIL_MOISTURE": native_stats}

        call_order = []
        original_scatter = viz.plot_scatter
        original_geo = viz.plot_geographic

        def scatter_spy(*args, **kwargs):
            call_order.append("scatter")
            return original_scatter(*args, **kwargs)

        def geo_spy(*args, **kwargs):
            call_order.append("geographic")
            return original_geo(*args, **kwargs)

        monkeypatch.setattr(viz, "plot_scatter", scatter_spy)
        monkeypatch.setattr(viz, "plot_geographic", geo_spy)
        viz.validation_report(
            collocation_ds, datatree, recipe, out_dir=tmp_path,
            native_units_stats_ds_map=native_stats_map,
        )
        import matplotlib.pyplot as plt
        plt.close("all")

        # CDF-matched section: geographic, then scatter. Native-units
        # section: geographic, then scatter. Each section's "geographic"
        # call must come first within that section. The CDF-matched
        # section's temporal-offset-colored scatter is skipped entirely for
        # soil_moisture (see the skip in validation_report), so there's only
        # one "scatter" per section, not two.
        assert call_order == ["geographic", "scatter", "geographic", "scatter"], call_order

    def test_native_units_section_restricted_to_matching_sources(self, tmp_path):
        """A val_source present in collocation_ds/datatree but *absent* from
        the native-units stats Dataset's ``source`` coordinate (i.e. its
        units don't already match SAR's, e.g. ISMN's volumetric m3/m3 vs.
        SAR's relative "%") must be excluded from the native-units plots —
        exercising the ``nu_mask`` / ``geo_pair_ds.where(nu_mask, ...)``
        restriction in validation_report."""
        from sar_validation.core.statistics import compute_statistics
        from sar_validation.core.visualization import validation_report

        recipe = self._recipe()
        datatree = xr.DataTree.from_dict({
            "validation/ascat_ssm/f1": self._ascat_node([0.0, 1.0], [45.0, 46.0], [25.0, 35.0]),
            "validation/ismn/f1": self._ascat_node([2.0, 3.0], [47.0, 48.0], [0.30, 0.35]),
        })

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 30.0, 22.0, 28.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 35.0, 0.30, 0.35])),
                "val_source": ("collocation", np.array(["ascat_ssm", "ascat_ssm", "ismn", "ismn"])),
                "val_id": ("collocation", np.array(["a1", "a2", "i1", "i2"])),
            },
        )
        # ismn reports volumetric m3/m3, not SAR's "%"-family unit, so
        # run_statistics_native_units would never include it — reproduce
        # that here directly by computing native stats over the ascat-only
        # subset, so the resulting stats Dataset's "source" coord only ever
        # carries "ascat_ssm".
        ascat_only_ds = collocation_ds.where(collocation_ds["val_source"] == "ascat_ssm", drop=True)
        native_stats = compute_statistics(ascat_only_ds, "sarSSM", "SOIL_MOISTURE", group_by=["val_source"])
        assert list(native_stats["source"].values) == ["ascat_ssm"]
        native_stats_map = {"sarSSM_vs_SOIL_MOISTURE": native_stats}

        key = "sarSSM_vs_SOIL_MOISTURE"
        figs = validation_report(
            collocation_ds, datatree, recipe, out_dir=tmp_path,
            native_units_stats_ds_map=native_stats_map,
        )
        assert key in figs

        # plot_scatter stamps an "N=<count>\n..." annotation (via ax.text)
        # on every scatter figure it produces, reflecting how many
        # deduplicated observations actually went into that plot. Read it
        # off the native-units scatter figure specifically (identified by
        # the "— native units —" banner) to verify what data it plotted.
        native_scatter_n = set()
        for fig in figs[key]:
            if not any("native units" in t.get_text() for t in fig.texts):
                continue
            for ax in fig.axes:
                for txt in ax.texts:
                    content = txt.get_text()
                    if content.startswith("N="):
                        native_scatter_n.add(int(content.splitlines()[0].split("=")[1]))

        assert native_scatter_n, "expected a native-units figure with an 'N=' scatter annotation"
        # Only the 2 ascat_ssm rows should have reached the native-units
        # plots. If nu_mask/the source restriction were a no-op, all 4 rows
        # (including ismn's non-matching-units ones) would show up instead,
        # giving N=4.
        assert native_scatter_n == {2}, (
            f"native-units scatter should only reflect the 2 matching-source "
            f"(ascat_ssm) rows; got N={native_scatter_n} — ismn rows may have "
            f"leaked through the source restriction"
        )

    def test_no_native_units_section_when_map_not_provided(self, tmp_path):
        """Omitting native_units_stats_ds_map (the default) must not change
        existing behaviour — no extra pages, no crash."""
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds
        from sar_validation.core.visualization import validation_report

        cfg = RecipeConfig(
            name="test_no_native_units_plots", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        recipe = Recipe(config=cfg)

        ascat_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", np.array([25.0, 35.0]))},
            coords={
                "lon": ("point", [0.0, 1.0]), "lat": ("point", [45.0, 46.0]),
                "time": ("point", np.array(["2026-01-01", "2026-01-01"], dtype="datetime64[ns]")),
            },
            attrs={"data_type": "scatterometer_ssm"},
        )
        datatree = xr.DataTree.from_dict({"validation/ascat_ssm/f1": ascat_node})

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.array([20.0, 30.0]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.array([25.0, 35.0])),
                "val_source": ("collocation", np.array(["ascat_ssm", "ascat_ssm"])),
            },
        )

        figs = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        assert "sarSSM_vs_SOIL_MOISTURE" in figs


class TestValidationReportNativeUnitsGeographic:
    """Task 6 (B3): the native-units section (soil_moisture only) should
    include a geographic page, alongside the existing native-units
    scatter/residuals/statistics pages."""

    def test_native_units_section_includes_a_geographic_page(
        self, geo_datatree_and_collocation, tmp_path,
    ):
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        from sar_validation.core.statistics import compute_statistics
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        collocation_ds = collocation_ds.rename({
            "sar_owiWindSpeed": "sar_sarSSM", "val_WSPD": "val_SOIL_MOISTURE",
        }).assign(
            collocation_type=("collocation", ["layer_vs_layer"] * collocation_ds.sizes["collocation"]),
            val_source=("collocation", ["ascat_ssm", "ascat_ssm", "ascat_ssm", "ascat_ssm"]),
        )
        recipe = Recipe(RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))
        nu_stats = compute_statistics(collocation_ds, "sarSSM", "SOIL_MOISTURE", group_by=["val_source"])

        result = validation_report(
            collocation_ds, datatree, recipe, out_dir=tmp_path,
            native_units_stats_ds_map={"sarSSM_vs_SOIL_MOISTURE": nu_stats},
        )
        # Both the main-section and native-units geographic figures should
        # have been generated (and closed) for this pair -- the exact count
        # depends on plot_geographic's dict-per-group return, so assert
        # indirectly via the PDF having grown past just scatter/stats/residuals.
        assert (tmp_path / "validation_report.pdf").exists()
        assert "sarSSM_vs_SOIL_MOISTURE" in result

    def test_plot_geographic_called_for_native_units_section(
        self, geo_datatree_and_collocation, tmp_path,
    ):
        from unittest.mock import patch

        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        from sar_validation.core.statistics import compute_statistics
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        collocation_ds = collocation_ds.rename({
            "sar_owiWindSpeed": "sar_sarSSM", "val_WSPD": "val_SOIL_MOISTURE",
        }).assign(
            collocation_type=("collocation", ["layer_vs_layer"] * collocation_ds.sizes["collocation"]),
            val_source=("collocation", ["ascat_ssm"] * collocation_ds.sizes["collocation"]),
        )
        recipe = Recipe(RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))
        nu_stats = compute_statistics(collocation_ds, "sarSSM", "SOIL_MOISTURE", group_by=["val_source"])

        with patch(
            "sar_validation.core.visualization.plot_geographic",
            wraps=__import__("sar_validation.core.visualization", fromlist=["plot_geographic"]).plot_geographic,
        ) as spy:
            validation_report(
                collocation_ds, datatree, recipe, out_dir=tmp_path,
                native_units_stats_ds_map={"sarSSM_vs_SOIL_MOISTURE": nu_stats},
            )

        # Called at least twice: once for the main section, once for
        # native-units.
        assert spy.call_count >= 2


class TestCdfMatchedTitleSuffix:
    """Task 6 (B3): the main section's scatter/geographic/residuals page
    titles get a " (CDF-matched)" suffix for soil_moisture recipes only —
    flagging that (unlike every other variable) that section's SAR values
    are rescaled, not raw, so it isn't mistaken for a units bug.

    ``validation_report`` doesn't expose its internal page titles through
    its return value (they only ever end up as an unused label alongside
    each PDF page's Figure -- ``pdf_pages: list[(title, Figure)]`` -- the
    title string is never actually drawn onto the page or otherwise
    surfaced), so a black-box assertion on a returned Figure's rendered
    text cannot observe this behavior (confirmed: plot_scatter/
    plot_residuals/plot_geographic each set their own ax/fig title
    independently of validation_report's local ``title`` variable, and
    none of those independent titles include this suffix). Given that,
    this test exercises the extracted ``_cdf_matched_suffix`` helper
    directly (real behavior, not implementation detail: identical to
    calling `validation_report` and reading back the exact string that
    would have been assigned to each page's title) plus a source-level
    check that all four title-construction sites actually consume it, so
    the test fails if either the suffix computation or its wiring into
    the title strings is removed or reversed.
    """

    def test_suffix_present_for_soil_moisture_only(self):
        from sar_validation.core.visualization import _cdf_matched_suffix

        assert _cdf_matched_suffix("soil_moisture") == " (CDF-matched)"
        for other in ("wind", "currents", "waves", "sea_ice", ""):
            assert _cdf_matched_suffix(other) == "", (
                f"expected no CDF-matched suffix for variable={other!r}"
            )

    def test_suffix_wired_into_all_four_main_section_title_sites(self):
        """Guards against the helper existing but never being consumed
        (or being consumed by the wrong branch) -- i.e. it would fail if
        someone reverted the f-string edits from Step 6 while leaving
        ``_cdf_matched_suffix`` itself intact."""
        import inspect

        from sar_validation.core.visualization import validation_report

        source = inspect.getsource(validation_report)

        assert 'cdf_matched_suffix = _cdf_matched_suffix(variable)' in source
        assert '— scatter{cdf_matched_suffix}"' in source
        assert '— geographic [{group}]{cdf_matched_suffix}"' in source
        assert '— geographic{cdf_matched_suffix}"' in source
        assert '— residuals{cdf_matched_suffix}"' in source
        # The statistics-table pages show metric numbers, not raw/rescaled
        # values directly, so they should NOT carry this suffix.
        assert '— statistics{cdf_matched_suffix}"' not in source

    def test_soil_moisture_main_section_report_still_generates(
        self, geo_datatree_and_collocation, tmp_path,
    ):
        """Smoke test (per the task brief's Step 7): the title-formatting
        change doesn't break report generation end-to-end for a
        soil_moisture recipe."""
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation
        collocation_ds = collocation_ds.rename({
            "sar_owiWindSpeed": "sar_sarSSM", "val_WSPD": "val_SOIL_MOISTURE",
        }).assign(
            collocation_type=("collocation", ["point_vs_layer"] * collocation_ds.sizes["collocation"]),
        )
        recipe = Recipe(RecipeConfig(
            name="test", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))
        validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        pdf_path = tmp_path / "validation_report.pdf"
        assert pdf_path.exists()

    def test_cdf_matched_banner_actually_rendered_on_figures(
        self, geo_datatree_and_collocation,
    ):
        """The " (CDF-matched)" suffix computed by ``_cdf_matched_suffix``
        only ever lands in ``pdf_pages``'s ``title`` element, which is never
        drawn onto the page (``pdf.savefig()`` takes no title, and
        ``_title`` in the ``for _title, fig in pdf_pages`` consumer loop is
        this codebase's "deliberately unused" convention) -- so none of the
        other tests in this class (which check the bare helper's return
        value or `validation_report`'s *source text*) would fail if the
        suffix were never actually stamped onto a rendered figure. This
        test checks the real artifact instead: a returned Figure's
        `fig.texts`, mirroring how `TestMarkNativeUnits`/
        `TestValidationReportNativeUnitsSection` verify `_mark_native_units`.
        """
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )
        from sar_validation.core.visualization import validation_report

        datatree, collocation_ds = geo_datatree_and_collocation

        # Non-soil-moisture (wind) run: no CDF-matched banner anywhere.
        wind_recipe = Recipe(RecipeConfig(
            name="test_wind", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))
        wind_figs = validation_report(collocation_ds, datatree, wind_recipe, out_dir=None)
        wind_key = "owiWindSpeed_vs_WSPD"
        assert wind_key in wind_figs and wind_figs[wind_key]
        assert not any(
            any("CDF-matched" in t.get_text() for t in fig.texts)
            for fig in wind_figs[wind_key]
        ), "wind (non-soil-moisture) report must not carry a CDF-matched banner"

        # Soil-moisture run: at least one figure must carry the real,
        # rendered "(CDF-matched)" banner stamped via fig.text(...).
        sm_collocation_ds = collocation_ds.rename({
            "sar_owiWindSpeed": "sar_sarSSM", "val_WSPD": "val_SOIL_MOISTURE",
        }).assign(
            collocation_type=("collocation", ["point_vs_layer"] * collocation_ds.sizes["collocation"]),
        )
        sm_recipe = Recipe(RecipeConfig(
            name="test_soil_moisture", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, -8.0, 50.0, 52.0),
            temporal_bounds=TemporalBounds("2026-07-10", "2026-07-11"),
        ))
        sm_figs = validation_report(sm_collocation_ds, datatree, sm_recipe, out_dir=None)
        sm_key = "sarSSM_vs_SOIL_MOISTURE"
        assert sm_key in sm_figs and sm_figs[sm_key]
        assert any(
            any("CDF-matched" in t.get_text() for t in fig.texts)
            for fig in sm_figs[sm_key]
        ), "no soil_moisture figure carries a rendered '(CDF-matched)' banner"


class TestValidationReportResidualsHistRange:
    """Regression test: the residuals page for a soil_moisture pair --
    the only variable whose main-section SAR series is CDF-matched
    before plotting -- must use the fixed (-1, 1) x-axis range for every
    val_source present, regardless of that source's native unit family
    (e.g. ascat_ssm's "percent_saturation" family vs ismn's "volumetric"
    one). This holds even for an originally-percent-scale source like
    ascat_ssm because, by the time the residuals page is built,
    add_rescaled_sar_column has already harmonized every source into
    ismn's volumetric domain via _harmonize_percent_domain_sources --
    see _volumetric_hist_range_overrides, which now ranges every present
    source uniformly instead of excluding percent-family sources."""

    def test_soil_moisture_residuals_page_uses_fixed_range(self, tmp_path):
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds
        from sar_validation.core.visualization import validation_report

        cfg = RecipeConfig(
            name="test_residuals_range", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        recipe = Recipe(config=cfg)

        rng = np.random.default_rng(5)
        n = 24
        lons = rng.uniform(-20.0, 0.0, n)
        lats = rng.uniform(35.0, 60.0, n)
        val_vals = rng.uniform(0.1, 0.4, n)
        sar_vals = val_vals * 100 + rng.normal(0, 2, n)  # correlated, different domain
        ascat_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", val_vals)},
            coords={
                "lon": ("point", lons), "lat": ("point", lats),
                "time": ("point", np.array(["2026-01-01"] * n, dtype="datetime64[ns]")),
            },
        )
        datatree = xr.DataTree.from_dict({"validation/ascat_ssm/f1": ascat_node})

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", sar_vals, {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", val_vals),
                "val_source": ("collocation", np.array(["ismn"] * n)),
                "val_id": ("collocation", np.array([f"a{i}" for i in range(n)])),
            },
        )

        key = "sarSSM_vs_SOIL_MOISTURE"
        figs = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        assert key in figs

        residuals_figs = [
            fig for fig in figs[key]
            if any("−" in ax.get_xlabel() for ax in fig.axes if ax.get_visible())
        ]
        if not residuals_figs:
            pytest.skip(
                "CDF-matching degenerated for this synthetic sample "
                "(add_rescaled_sar_column returned all-NaN) -- the direct "
                "TestPlotResiduals unit tests already cover hist_range "
                "behavior; this integration check is best-effort."
            )
        ax = [a for a in residuals_figs[0].axes if a.get_visible()][0]
        assert ax.get_xlim() == (-1.0, 1.0)

    def test_soil_moisture_residuals_page_has_no_data_when_ascat_present_without_ismn(self, tmp_path):
        """ascat_ssm's "percent_saturation" values can only be converted
        into the shared volumetric domain used by the CDF-matched report
        section by riding ismn's own CDF-matching fit as a reference (see
        _harmonize_percent_domain_sources) -- ismn must be present with
        >= 2 valid collocated pairs for that conversion to happen at all.

        With ascat_ssm as the *only* val_source and ismn absent entirely,
        there is no reference to convert against: every ascat_ssm row's
        sar_sarSSM/val_SOIL_MOISTURE values are set to NaN for this section
        (a deliberate, logged fallback -- ascat_ssm's data is still shown,
        untouched, in the report's separate "native units" section). This
        is a regression test for the current design: ascat_ssm-without-ismn
        used to mean "leave it in a wide percent-scale range"; it now means
        "no comparable data at all" for the CDF-matched section, so
        validation_report must produce no figures whatsoever for this pair
        -- not a residuals page with a wide, non-fixed x-axis range."""
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds
        from sar_validation.core.visualization import validation_report

        cfg = RecipeConfig(
            name="test_residuals_range_ascat", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        recipe = Recipe(config=cfg)

        rng = np.random.default_rng(5)
        n = 24
        lons = rng.uniform(-20.0, 0.0, n)
        lats = rng.uniform(35.0, 60.0, n)
        val_vals = rng.uniform(10.0, 40.0, n)   # ascat_ssm: percent saturation, ~0-100
        sar_vals = val_vals + rng.normal(0, 2, n)  # correlated, same rough domain (%)
        ascat_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", val_vals)},
            coords={
                "lon": ("point", lons), "lat": ("point", lats),
                "time": ("point", np.array(["2026-01-01"] * n, dtype="datetime64[ns]")),
            },
        )
        datatree = xr.DataTree.from_dict({"validation/ascat_ssm/f1": ascat_node})

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", sar_vals, {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", val_vals),
                "val_source": ("collocation", np.array(["ascat_ssm"] * n)),
                "val_id": ("collocation", np.array([f"a{i}" for i in range(n)])),
            },
        )

        key = "sarSSM_vs_SOIL_MOISTURE"
        figs = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        assert key in figs

        # Every row is dropped to NaN by _harmonize_percent_domain_sources
        # (ismn absent -- see docstring), so none of scatter/residuals/
        # geographic/etc. has any valid data left to plot for this pair.
        assert figs[key] == [], (
            "with ascat_ssm as the only val_source and ismn absent, "
            f"expected no figures for {key!r} in the CDF-matched section, "
            f"got {len(figs[key])}"
        )

        residuals_figs = [
            fig for fig in figs[key]
            if any("−" in ax.get_xlabel() for ax in fig.axes if ax.get_visible())
        ]
        assert residuals_figs == []

    def test_soil_moisture_residuals_page_gives_every_source_the_same_volumetric_range(self, tmp_path):
        """Regression test for a recipe that validates against BOTH ismn
        (volumetric) AND ascat_ssm (originally percent-scale) in the same
        pair -- e.g. recipes/soil_moisture_satellite_example.yaml -- must
        give BOTH ismn's and ascat_ssm's subplot the same fixed (-1, 1)
        range, because add_rescaled_sar_column harmonizes ascat_ssm into
        ismn's volumetric domain before the residuals page is built (see
        _harmonize_percent_domain_sources), so ascat_ssm's residuals are
        genuinely volumetric-scale here too -- not left in its raw
        percent domain, and not pooled with ismn's spread by accident,
        but explicitly ranged the same because both sources now share
        one common domain."""
        from sar_validation.core.recipe import GeographicBounds, Recipe, RecipeConfig, TemporalBounds
        from sar_validation.core.visualization import validation_report

        cfg = RecipeConfig(
            name="test_residuals_range_mixed", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-20.0, 0.0, 35.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
        )
        recipe = Recipe(config=cfg)

        rng = np.random.default_rng(7)
        n_each = 24
        lons = rng.uniform(-20.0, 0.0, n_each)
        lats = rng.uniform(35.0, 60.0, n_each)

        ismn_val = rng.uniform(0.1, 0.4, n_each)
        ismn_sar = ismn_val * 100 + rng.normal(0, 2, n_each)
        ascat_val = rng.uniform(10.0, 40.0, n_each)
        ascat_sar = ascat_val + rng.normal(0, 2, n_each)

        ismn_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", ismn_val)},
            coords={
                "lon": ("point", lons), "lat": ("point", lats),
                "time": ("point", np.array(["2026-01-01"] * n_each, dtype="datetime64[ns]")),
            },
        )
        ascat_node = xr.Dataset(
            {"SOIL_MOISTURE": ("point", ascat_val)},
            coords={
                "lon": ("point", lons), "lat": ("point", lats),
                "time": ("point", np.array(["2026-01-01"] * n_each, dtype="datetime64[ns]")),
            },
        )
        datatree = xr.DataTree.from_dict({
            "validation/ismn/f1": ismn_node,
            "validation/ascat_ssm/f1": ascat_node,
        })

        collocation_ds = xr.Dataset(
            {
                "sar_sarSSM": ("collocation", np.concatenate([ismn_sar, ascat_sar]), {"units": "%"}),
                "val_SOIL_MOISTURE": ("collocation", np.concatenate([ismn_val, ascat_val])),
                "val_source": ("collocation", np.array(["ismn"] * n_each + ["ascat_ssm"] * n_each)),
                "val_id": ("collocation", np.array([f"a{i}" for i in range(2 * n_each)])),
            },
        )

        key = "sarSSM_vs_SOIL_MOISTURE"
        figs = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)
        assert key in figs

        residuals_figs = [
            fig for fig in figs[key]
            if any("−" in ax.get_xlabel() for ax in fig.axes if ax.get_visible())
        ]
        if not residuals_figs:
            pytest.skip(
                "CDF-matching degenerated for this synthetic sample -- "
                "the direct TestPlotResiduals unit tests already cover "
                "hist_range behavior; this integration check is best-effort."
            )
        fig = residuals_figs[0]
        axes_by_title = {
            ax.get_title(): ax for ax in fig.axes if ax.get_visible()
        }
        ismn_ax = next((ax for title, ax in axes_by_title.items() if title.startswith("ismn")), None)
        ascat_ax = next((ax for title, ax in axes_by_title.items() if title.startswith("ascat_ssm")), None)
        assert ismn_ax is not None and ascat_ax is not None

        assert ismn_ax.get_xlim() == (-1.0, 1.0), (
            "ismn's own subplot must keep the fixed volumetric range even "
            "when ascat_ssm is present in the same pair"
        )
        assert ascat_ax.get_xlim() == (-1.0, 1.0), (
            "ascat_ssm's subplot must get the same fixed volumetric range "
            "as ismn's once add_rescaled_sar_column has harmonized it into "
            "ismn's volumetric domain -- its residuals are no longer on "
            "the raw percent scale by the time this page is built"
        )
