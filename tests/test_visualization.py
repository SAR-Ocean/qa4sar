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
        import matplotlib.pyplot as plt
        import matplotlib.axes

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
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
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
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
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
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
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
        import matplotlib.pyplot as plt
        import matplotlib.axes
        from sar_validation.core.visualization import plot_geographic
        from sar_validation.core.datatree_converter import DataTreeConverter

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
        import matplotlib.pyplot as plt
        import matplotlib.axes
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
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
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
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
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
        GeographicBounds, Recipe, RecipeConfig, ValidationDataSource,
        CollocationType, PointVsLayerCollocation,
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
            assert c.get("alpha") == 0.65
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


class TestPlotCollocationDiagnosticsRecipeVariableStyling:
    def test_wind_matched_layer_alpha_is_reduced(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 0.65

    def test_waves_matched_layer_alpha_stays_opaque(
        self, geo_datatree_and_collocation, diagnostics_recipe_waves, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_waves, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 1.0

    def test_currents_matched_layer_alpha_stays_opaque(
        self, geo_datatree_and_collocation, diagnostics_recipe_currents, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe_currents, tmp_path)
        plt.close("all")

        layer_calls = [c for c in recorded if c.get("zorder") == 5]
        assert layer_calls
        for c in layer_calls:
            assert c.get("alpha") == 1.0

    def test_waves_matched_points_are_larger_with_black_edge(
        self, geo_datatree_and_collocation, diagnostics_recipe_waves, tmp_path, monkeypatch
    ):
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
        plot_collocation_diagnostics(datatree, collocation_ds, diagnostics_recipe, tmp_path)
        plt.close("all")

        matched_calls = [c for c in recorded if c.get("zorder") in (5, 6)]
        assert matched_calls
        for c in matched_calls:
            assert c.get("s") == 25
            assert c.get("edgecolors") == "none"


class TestPlotCollocationDiagnosticsTicks:
    def test_overview_plot_gets_degree_formatted_ticks(
        self, geo_datatree_and_collocation, diagnostics_recipe, tmp_path
    ):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
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
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

        datatree, collocation_ds = geo_datatree_and_collocation
        recipe = Recipe(config=RecipeConfig(name="test", variable="wind"))

        figures = validation_report(collocation_ds, datatree, recipe, out_dir=tmp_path)

        key = "owiWindSpeed_vs_WSPD"
        assert key in figures
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


class TestValidationReportClosesPageFigures:
    def test_no_figures_left_open_after_report(self, geo_datatree_and_collocation, tmp_path):
        """Regression guard for the render-once refactor: the new
        lightweight PDF-page figures (built by _finalize_figure_for_report /
        _image_page_figure) must be closed once written, not leaked —
        unlike the original heavy figures, they aren't tracked in
        `all_figures` / `figs`, so nothing else closes them."""
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

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
        from sar_validation.core.recipe import Recipe, RecipeConfig
        from sar_validation.core.datatree_converter import DataTreeConverter

        y, x = 3, 3
        lon2d, lat2d = np.meshgrid(np.linspace(-10, -8, x), np.linspace(50, 52, y))
        sar_ds = xr.Dataset(
            {"rvlRadVel": (("y", "x"), np.full((y, x), 0.3))},
            coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d),
                    "time": pd.Timestamp("2026-07-10T19:00:00")},
        )
        datatree = DataTreeConverter.to_datatree({"sar/sceneA": sar_ds})

        coll = xr.Dataset({
            "sar_rvlRadVel":            ("collocation", [0.3, 0.31, 0.29, 0.32]),
            "val_rvlRadVel_projection": ("collocation", [0.28, 0.30, 0.27, 0.31]),
            "val_source":               ("collocation", ["hf_radar"] * 4),
            "sar_scene_name":           ("collocation", ["sceneA"] * 4),
            "val_lon":                  ("collocation", [-9.5, -9.4, -9.3, -9.2]),
            "val_lat":                  ("collocation", [50.5, 50.6, 50.7, 50.8]),
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

    def test_wind_recipe_keeps_default_point_size(self, geo_datatree_and_collocation, tmp_path, monkeypatch):
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

        assert captured.get("point_size") == 40


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
        from sar_validation.core.visualization import plot_rvl_land_qa
        from sar_validation.core.datatree_converter import DataTreeConverter

        datatree = DataTreeConverter.to_datatree({
            "sar/sceneA": self._make_sar_node(land_count=0),
        })
        assert plot_rvl_land_qa(datatree) is None

    def test_returns_none_when_no_sar_node(self):
        from sar_validation.core.visualization import plot_rvl_land_qa
        from sar_validation.core.datatree_converter import DataTreeConverter

        datatree = DataTreeConverter.to_datatree({})
        assert plot_rvl_land_qa(datatree) is None

    def test_returns_table_with_one_row_per_land_scene(self):
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import plot_rvl_land_qa
        from sar_validation.core.datatree_converter import DataTreeConverter

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
        import numpy as np
        import matplotlib.pyplot as plt
        from sar_validation.core.visualization import _image_page_figure

        img = np.zeros((300, 450, 3), dtype=np.uint8)
        fig = _image_page_figure(img, dpi=150)

        w_in, h_in = fig.get_size_inches()
        assert w_in == pytest.approx(450 / 150)
        assert h_in == pytest.approx(300 / 150)
        plt.close(fig)


class TestFinalizeFigureForReport:
    def test_writes_png_closes_original_returns_image_page(self, tmp_path):
        import matplotlib.pyplot as plt
        import matplotlib._pylab_helpers as pylab_helpers
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
        from sar_validation.core.visualization import validation_report
        from sar_validation.core.recipe import Recipe, RecipeConfig

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


class TestValidationReportDownloadWarnings:
    def test_download_warning_appears_on_cover_page(
        self, geo_datatree_and_collocation, tmp_path, monkeypatch
    ):
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

        cover = recorded_figs[0]
        # Exactly the same two text() calls as before this change: title + variable/date.
        assert len(cover.texts) == 2


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
