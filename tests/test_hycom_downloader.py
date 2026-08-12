"""Tests for HycomDownloader."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


class TestResolveHycomSegments:
    def test_window_entirely_in_espc_d_v02(self):
        from sar_validation.downloaders.hycom_downloader import _resolve_hycom_segments

        segs = _resolve_hycom_segments(
            datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 2, 0, 0, 0),
        )
        assert segs == [("espc_d_v02", datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 2, 0, 0, 0))]

    def test_window_entirely_in_gofs31(self):
        from sar_validation.downloaders.hycom_downloader import _resolve_hycom_segments

        segs = _resolve_hycom_segments(
            datetime(2020, 6, 1, 0, 0, 0), datetime(2020, 6, 2, 0, 0, 0),
        )
        assert segs == [("gofs31_930", datetime(2020, 6, 1, 0, 0, 0), datetime(2020, 6, 2, 0, 0, 0))]

    def test_window_straddling_cutover_splits_into_two_segments(self):
        from sar_validation.downloaders.hycom_downloader import _resolve_hycom_segments

        segs = _resolve_hycom_segments(
            datetime(2024, 8, 9, 12, 0, 0), datetime(2024, 8, 10, 12, 0, 0),
        )
        assert segs == [
            ("gofs31_930", datetime(2024, 8, 9, 12, 0, 0), datetime(2024, 8, 10, 0, 0, 0)),
            ("espc_d_v02", datetime(2024, 8, 10, 0, 0, 0), datetime(2024, 8, 10, 12, 0, 0)),
        ]

    def test_window_ending_before_coverage_start_raises(self):
        from sar_validation.downloaders.hycom_downloader import _resolve_hycom_segments

        with pytest.raises(ValueError, match="2018-12-04"):
            _resolve_hycom_segments(
                datetime(2018, 3, 5, 0, 0, 0), datetime(2018, 3, 7, 0, 0, 0),
            )

    def test_window_starting_before_coverage_start_but_ending_after_is_clamped(self):
        from sar_validation.downloaders.hycom_downloader import _resolve_hycom_segments

        segs = _resolve_hycom_segments(
            datetime(2018, 11, 1, 0, 0, 0), datetime(2018, 12, 10, 0, 0, 0),
        )
        assert segs == [("gofs31_930", datetime(2018, 12, 4, 0, 0, 0), datetime(2018, 12, 10, 0, 0, 0))]


class TestDodscUrls:
    def test_espc_d_v02_returns_one_url_per_component(self, tmp_path):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        dl = HycomDownloader(output_dir=tmp_path)
        urls = dl._dodsc_urls("espc_d_v02", datetime(2025, 1, 1), datetime(2025, 1, 2))
        assert urls == {
            "u": "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z",
            "v": "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z",
        }

    def test_gofs31_returns_one_combined_url_per_touched_year(self, tmp_path):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        dl = HycomDownloader(output_dir=tmp_path)
        urls = dl._dodsc_urls("gofs31_930", datetime(2019, 12, 30), datetime(2020, 1, 2))
        assert urls == {
            "uv_2019": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/2019",
            "uv_2020": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/2020",
        }

    def test_gofs31_single_year_window_returns_one_url(self, tmp_path):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        dl = HycomDownloader(output_dir=tmp_path)
        urls = dl._dodsc_urls("gofs31_930", datetime(2020, 6, 1), datetime(2020, 6, 2))
        assert urls == {"uv_2020": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/2020"}


class TestNcPathForSegment:
    def test_nc_path_naming(self, tmp_path):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        dl = HycomDownloader(output_dir=tmp_path)
        path = dl._nc_path_for_segment("espc_d_v02", datetime(2025, 1, 1), datetime(2025, 1, 2))
        assert path.name == "hycom_espc_d_v02_20250101T000000_20250102T000000.nc"


class TestHycomDownloaderDownload:
    def test_dry_run_probes_time_coordinate_without_loading_full_grid(self, tmp_path, monkeypatch):
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        probe_calls = []

        class FakeLazyDataset:
            def __init__(self):
                self.time = xr.DataArray(
                    pd.date_range("2025-01-01", periods=4, freq="3h"), dims="time",
                )
            def close(self):
                pass

        def fake_open_dataset(url, *a, **kw):
            probe_calls.append(url)
            return FakeLazyDataset()

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

        dl = HycomDownloader(output_dir=tmp_path, dry_run=True)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2025-01-01T00:00:00", end="2025-01-01T06:00:00",
        )
        assert paths == []
        assert len(probe_calls) == 2  # u3z and v3z, ESPC-D-V02
        assert not list(tmp_path.glob("*.nc"))

    def test_existing_segment_file_is_skipped_and_returned(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        called = []
        monkeypatch.setattr(
            HycomDownloader, "_download_segment",
            lambda self, *a: called.append(a) or None,
        )

        dl = HycomDownloader(output_dir=tmp_path)
        existing = dl._nc_path_for_segment("espc_d_v02", datetime(2025, 1, 1), datetime(2025, 1, 1, 6))
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("fake")

        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2025-01-01T00:00:00", end="2025-01-01T06:00:00",
        )
        assert called == []
        assert paths == [existing]

    def test_straddling_window_downloads_both_segments(self, tmp_path, monkeypatch):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        requested_keys = []

        def fake_download_segment(self, dataset_key, seg_start, seg_end, *a):
            requested_keys.append(dataset_key)
            p = self._nc_path_for_segment(dataset_key, seg_start, seg_end)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fake")
            return p

        monkeypatch.setattr(HycomDownloader, "_download_segment", fake_download_segment)

        dl = HycomDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2024-08-09T12:00:00", end="2024-08-10T12:00:00",
        )
        assert requested_keys == ["gofs31_930", "espc_d_v02"]
        assert len(paths) == 2

    def test_pre_coverage_window_propagates_value_error(self, tmp_path):
        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        dl = HycomDownloader(output_dir=tmp_path)
        with pytest.raises(ValueError, match="2018-12-04"):
            dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2018-03-05T00:00:00", end="2018-03-07T00:00:00",
            )


class TestTauVariableDropped:
    """Live-verified 2026-08-11: both real HyCOM THREDDS datasets
    (GLBy0.08/expt_93.0 and ESPC-D-V02) carry a ``tau`` (forecast
    lead-time) data variable with a CF-noncompliant ``units`` attribute of
    ``"hours since analysis"`` -- "analysis" is not a parseable reference
    date, so xarray's default ``decode_times=True`` blows up on
    ``xr.open_dataset(url)`` with "unable to decode time units 'hours
    since analysis'". This broke every real network call (dry-run probe
    AND actual download) even though the mocked unit tests above never
    exercised real CF decoding and so never caught it. ``tau`` is never
    consumed anywhere downstream (only ``water_u``/``water_v``/``time``/
    ``lat``/``lon``/``depth`` are used) -- it must be excluded via
    ``drop_variables`` before xarray attempts CF decoding."""

    def test_probe_coverage_drops_tau_variable(self, tmp_path, monkeypatch):
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        calls = []

        class FakeLazyDataset:
            def __init__(self):
                self.time = xr.DataArray(
                    pd.date_range("2025-01-01", periods=4, freq="3h"), dims="time",
                )

            def close(self):
                pass

        def fake_open_dataset(url, *a, **kw):
            calls.append(kw)
            return FakeLazyDataset()

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

        dl = HycomDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2025-01-01T00:00:00", end="2025-01-01T06:00:00",
        )
        assert len(calls) == 2  # u3z and v3z, ESPC-D-V02
        for kw in calls:
            assert "tau" in kw.get("drop_variables", ()), (
                "xr.open_dataset must drop_variables=['tau'] -- the real "
                "HyCOM datasets' tau variable has units='hours since "
                "analysis', which is not CF-decodable and breaks the "
                "default decode_times=True open."
            )

    def test_download_segment_drops_tau_variable(self, tmp_path, monkeypatch):
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        calls = []

        class FakeDataset:
            def __getitem__(self, _keys):
                return self

            def close(self):
                pass

        def fake_open_dataset(url, *a, **kw):
            calls.append(kw)
            return FakeDataset()

        def fake_merge(_datasets, *a, **kw):
            return FakeDataset()

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        monkeypatch.setattr(xr, "merge", fake_merge)

        dl = HycomDownloader(output_dir=tmp_path)
        # sel()/load()/to_netcdf() chain on FakeDataset — stub minimally.
        FakeDataset.sel = lambda self, *a, **kw: self
        FakeDataset.load = lambda self: self
        FakeDataset.to_netcdf = lambda self, path: None

        dl._download_segment(
            "espc_d_v02", datetime(2025, 1, 1), datetime(2025, 1, 2),
            -10.0, 10.0, 40.0, 55.0,
        )
        assert len(calls) == 2  # u3z and v3z
        for kw in calls:
            assert "tau" in kw.get("drop_variables", ()), (
                "xr.open_dataset must drop_variables=['tau'] -- same real "
                "decode bug as the dry-run probe path."
            )


class TestDownloadSegmentLonConvention:
    """
    Live-confirmed 2026-08-11: HyCOM's REAL ``lon`` coordinate axis (both
    ``ESPC-D-V02`` and ``GLBy0.08/expt_93.0``) uses the 0-360 convention
    (0.0 .. 359.92), not this toolbox's standard -180..180 convention that
    every recipe bbox uses. ``_download_segment`` used to build ``west``/
    ``east`` directly from the (possibly negative) recipe bbox and
    ``.sel(lon=slice(west, east))`` against that native axis -- for any
    bbox with negative bounds (most of the Americas, Atlantic Europe,
    etc.) this matched nothing (zero-length ``lon`` axis, no error) or
    silently dropped part of the intended coverage (bbox straddling 0 deg).

    Unlike the ``FakeDataset``/``FakeLazyDataset`` stubs used elsewhere in
    this file (whose ``.sel()`` is a no-op passthrough or absent
    entirely), these tests build REAL ``xr.Dataset`` objects with a
    genuine global 0-360 ``lon`` axis and only monkeypatch
    ``xr.open_dataset`` to serve them in place of the network call --
    real ``xr.merge``/``.sel``/``.load``/``.to_netcdf`` all run for real,
    so a wrong longitude conversion actually shows up as wrong/missing
    data, exactly the class of bug that the old fully-stubbed tests could
    never have caught.
    """

    #: 0.5-degree global lon axis, 0.0 .. 359.5 -- coarse enough to keep
    #: these tests fast, fine-grained enough to assert real coverage
    #: bounds and cell-level correctness.
    _LON_STEP = 0.5

    @classmethod
    def _make_component_dataset(cls, var_name: str, seg_start):
        """One real in-memory ``xr.Dataset`` mimicking a single HyCOM
        component file (e.g. ESPC-D-V02's ``u3z``): a genuine global
        0-360 ``lon`` axis, a small ``lat`` range covering every bbox
        used below, two ``depth`` levels, two ``time`` steps straddling
        *seg_start*. The data pattern is simply ``value == lon`` at every
        point -- lets assertions verify not just "non-empty" but exactly
        which cells were selected."""
        import numpy as np
        import pandas as pd
        import xarray as xr

        lon = np.round(np.arange(0.0, 360.0, cls._LON_STEP), 4)
        lat = np.round(np.arange(20.0, 70.0, 0.5), 4)
        depth = np.array([0.0, 10.0])
        time = pd.date_range(seg_start, periods=2, freq="3h")

        arr = np.broadcast_to(
            lon[None, None, None, :],
            (len(time), len(depth), len(lat), len(lon)),
        ).astype("float64").copy()

        return xr.Dataset(
            {var_name: (("time", "depth", "lat", "lon"), arr)},
            coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        )

    def _patch_open_dataset(self, monkeypatch, seg_start):
        """Serve real fake ``water_u``/``water_v`` Datasets for the two
        ESPC-D-V02 URLs; delegate any other path (e.g. reading back the
        ``.nc`` file this test writes) to the REAL ``xr.open_dataset``."""
        import xarray as xr

        real_open_dataset = xr.open_dataset
        u_ds = self._make_component_dataset("water_u", seg_start)
        v_ds = self._make_component_dataset("water_v", seg_start)

        def fake_open_dataset(url, *a, **kw):
            if url == "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z":
                return u_ds
            if url == "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z":
                return v_ds
            return real_open_dataset(url, *a, **kw)

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    def test_negative_lon_bbox_us_east_coast_downloads_nonempty_correct_lon(
        self, tmp_path, monkeypatch,
    ):
        """The exact real-world failing case: a fully-negative-longitude
        bbox (US East Coast, min_lon=-77.0, max_lon=-68.0) used to
        ``.sel(lon=slice(-77.08, -67.92))`` against a 0-360-only axis,
        matching NOTHING -- a silent zero-length ``lon`` dimension. After
        the fix, the recipe bounds are converted to their 0-360
        equivalents (282.92 .. 292.08) before selecting."""
        import numpy as np
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        seg_start, seg_end = datetime(2025, 1, 1), datetime(2025, 1, 2)
        self._patch_open_dataset(monkeypatch, seg_start)

        dl = HycomDownloader(output_dir=tmp_path)
        nc_path = dl._download_segment(
            "espc_d_v02", seg_start, seg_end,
            -77.0, -68.0, 35.0, 44.0,
        )

        assert nc_path is not None and nc_path.exists()
        result = xr.open_dataset(nc_path)
        try:
            assert result.sizes["lon"] > 0, (
                "negative-longitude bbox produced a zero-length lon axis -- "
                "the exact live-confirmed bug (west/east never converted to "
                "HyCOM's native 0-360 convention)."
            )
            lon = result["lon"].values
            # west_360 = (-77 - 1/12) % 360 = 282.9166...
            # east_360 = (-68 + 1/12) % 360 = 292.0833...
            assert lon.min() >= 282.9
            assert lon.max() <= 292.1
            assert lon.min() < 283.5  # real coverage starts near the west edge
            assert lon.max() > 291.5  # real coverage extends to the east edge
            # data pattern is value == lon -- confirms these are genuinely
            # the correct source cells, not e.g. wrapped to the wrong
            # hemisphere.
            np.testing.assert_allclose(result["water_u"].isel(time=0, lat=0).values, lon)
        finally:
            result.close()

    def test_download_segment_logs_before_starting_the_network_fetch(
        self, tmp_path, monkeypatch, caplog,
    ):
        """A real OPeNDAP fetch can take a long time with zero output in
        between -- without a log line before it starts, a slow segment is
        indistinguishable from a hang (reported directly against a real
        run). ``_download_segment`` must ``logger.info(...)`` which
        dataset/segment it's about to request before the network calls
        begin."""
        import logging

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        seg_start, seg_end = datetime(2025, 1, 1), datetime(2025, 1, 2)
        self._patch_open_dataset(monkeypatch, seg_start)

        dl = HycomDownloader(output_dir=tmp_path)
        with caplog.at_level(logging.INFO, logger="sar_validation.downloaders.hycom_downloader"):
            nc_path = dl._download_segment(
                "espc_d_v02", seg_start, seg_end,
                -77.0, -68.0, 35.0, 44.0,
            )

        assert nc_path is not None and nc_path.exists()
        assert any(
            "espc_d_v02" in rec.message and "requesting HyCOM" in rec.message
            for rec in caplog.records
        ), f"no 'requesting HyCOM' notice logged; records were: {[r.message for r in caplog.records]}"

    def test_wrapping_bbox_straddling_zero_degrees_stitches_monotonic_lon(
        self, tmp_path, monkeypatch,
    ):
        """A bbox straddling 0 deg longitude (e.g. -10..10) converts to a
        0-360 window that itself straddles HyCOM's own 0/360 seam
        (349.92..10.08) -- must select two segments and shift the second
        by +360 so the combined axis stays monotonically increasing,
        mirroring ``DataTreeConverter._stitch_antimeridian_window_files``'s
        existing ERA5 antimeridian handling."""
        import numpy as np
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        seg_start, seg_end = datetime(2025, 1, 1), datetime(2025, 1, 2)
        self._patch_open_dataset(monkeypatch, seg_start)

        dl = HycomDownloader(output_dir=tmp_path)
        nc_path = dl._download_segment(
            "espc_d_v02", seg_start, seg_end,
            -10.0, 10.0, 40.0, 55.0,
        )

        assert nc_path is not None and nc_path.exists()
        result = xr.open_dataset(nc_path)
        try:
            lon = result["lon"].values
            assert len(lon) > 0
            # Strictly increasing -- required by RegularGridInterpolator
            # (see model_collocation.build_spatial_interpolator's
            # documented contract).
            assert np.all(np.diff(lon) > 0), (
                f"combined lon axis is not monotonically increasing: {lon}"
            )
            # west segment: near-360 values (originally 349.92..359.5).
            assert lon.min() >= 349.5
            assert lon.min() < 350.5
            # east segment: shifted by +360 (originally 0..10.08).
            assert lon.max() > 369.0
            assert lon.max() <= 370.5
            # No numeric wrap-around back down to small values -- the
            # whole point of the +360 shift.
            assert lon.max() > 360.0
            # Data-level correctness: the shifted segment's values must
            # still equal their ORIGINAL (pre-shift) longitude -- e.g. the
            # cell now at lon=365.0 came from the source's lon=5.0 cell.
            water_u = result["water_u"].isel(time=0, lat=0)
            # >= (not >): the shifted segment starts AT exactly 360.0
            # (the former 0.0), so that boundary point belongs to the
            # shifted comparison, not the unshifted one.
            shifted_mask = lon >= 360.0
            np.testing.assert_allclose(
                water_u.values[shifted_mask], lon[shifted_mask] - 360.0,
            )
            unshifted_mask = ~shifted_mask
            np.testing.assert_allclose(
                water_u.values[unshifted_mask], lon[unshifted_mask],
            )
        finally:
            result.close()

    def test_ordinary_positive_lon_bbox_still_works_no_regression(
        self, tmp_path, monkeypatch,
    ):
        """An ordinary positive-longitude bbox (Vestlandet recipe's
        min_lon=0.5, max_lon=6.0) must keep working exactly as before --
        it never touches HyCOM's 0/360 seam, so this is the non-wrapping
        common case that happened to "work" pre-fix (a purely positive
        bbox is unaffected by the -180..180-vs-0-360 mismatch)."""
        import numpy as np
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        seg_start, seg_end = datetime(2025, 1, 1), datetime(2025, 1, 2)
        self._patch_open_dataset(monkeypatch, seg_start)

        dl = HycomDownloader(output_dir=tmp_path)
        nc_path = dl._download_segment(
            "espc_d_v02", seg_start, seg_end,
            0.5, 6.0, 58.0, 62.5,
        )

        assert nc_path is not None and nc_path.exists()
        result = xr.open_dataset(nc_path)
        try:
            lon = result["lon"].values
            assert len(lon) > 0
            assert np.all(np.diff(lon) > 0)
            # west_360 = 0.5 - 1/12 = 0.4167; east_360 = 6.0 + 1/12 = 6.0833
            assert lon.min() >= 0.4
            assert lon.max() <= 6.1
            assert lon.min() < 1.0
            assert lon.max() > 5.5
            np.testing.assert_allclose(
                result["water_u"].isel(time=0, lat=0).values, lon,
            )
        finally:
            result.close()


class TestBracketBuffer:
    """Live-reproduced 2026-08-11 running recipes/currents_useastcoast.yaml
    end-to-end (~20-minute SAR window): the download succeeded, but
    ModelLayerCollocation logged "no bracketing ERA5 hour for scene ...
    -- skipping cell-averaging pass" for every SAR scene, producing ZERO
    HyCOM collocation matches. Root cause: HyCOM's real granule cadence is
    3 hours, but HycomDownloader requested exactly the recipe's own
    (generically, thinly-padded) window with no internal margin of its
    own -- unlike ERA5Downloader, which already carries its own
    ``_HOUR_BUFFER`` for exactly this reason (see that module's
    docstring). ModelLayerCollocation's hyperbolic bracket needs 3
    consecutive granules ``[t2 - cadence, t2, t2 + cadence]`` around each
    SAR scene time T; tracing its floor-hour/searchsorted logic shows the
    bracket CENTER t2 can be up to just-under-one-cadence-period (< 3h)
    before T, so the EARLIEST granule ever needed can be up to
    just-under-2*cadence (< 6h) before T, and the LATEST up to one
    cadence (3h) after T. HycomDownloader must buffer its own OPeNDAP
    request window by 2*cadence on both sides to guarantee a full bracket
    is always downloaded, even for a SAR scene at the very edge of the
    recipe's core window."""

    @staticmethod
    def _make_fake_dataset_classes(sel_calls):
        class FakeDataset:
            def __getitem__(self, _keys):
                return self

            def sel(self, *a, **kw):
                sel_calls.append(kw)
                return self

            def load(self):
                return self

            def to_netcdf(self, path):
                pass

            def close(self):
                pass

        def fake_open_dataset(url, *a, **kw):
            return FakeDataset()

        def fake_merge(_datasets, *a, **kw):
            return FakeDataset()

        return fake_open_dataset, fake_merge

    def test_download_segment_requests_buffered_time_slice(self, tmp_path, monkeypatch):
        """The real failing scenario: a narrow ~20-minute segment window
        must still produce a much wider OPeNDAP time-slice request so a
        full 3-granule bracket is available around the scene time."""
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import (
            _BRACKET_BUFFER_HOURS,
            HycomDownloader,
        )

        sel_calls: list = []
        fake_open_dataset, fake_merge = self._make_fake_dataset_classes(sel_calls)
        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        monkeypatch.setattr(xr, "merge", fake_merge)

        dl = HycomDownloader(output_dir=tmp_path)
        seg_start = datetime(2026, 7, 14, 10, 30, 0)
        seg_end = datetime(2026, 7, 14, 10, 50, 0)
        dl._download_segment("espc_d_v02", seg_start, seg_end, -77.0, -68.0, 35.0, 44.0)

        # Two .sel() calls happen (lon/lat/time window, then depth=0.0
        # nearest-match) -- find the one carrying the time bounds.
        time_calls = [kw for kw in sel_calls if "time" in kw]
        assert len(time_calls) == 1
        time_slice = time_calls[0]["time"]
        assert time_slice.start <= seg_start - timedelta(hours=_BRACKET_BUFFER_HOURS)
        assert time_slice.stop >= seg_end + timedelta(hours=_BRACKET_BUFFER_HOURS)

    def test_time_tolerance_minutes_constructor_arg_drives_the_buffer(self, tmp_path, monkeypatch):
        """The bracket margin is now driven by time_tolerance_minutes
        (recipe-resolved by the orchestrator, see
        orchestrator._resolve_temporal_padding_minutes), not the fixed
        _BRACKET_BUFFER_HOURS constant -- a caller passing an explicit
        value must see the actual OPeNDAP request widen/narrow to match,
        not just the same fixed default every time."""
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        sel_calls: list = []
        fake_open_dataset, fake_merge = self._make_fake_dataset_classes(sel_calls)
        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        monkeypatch.setattr(xr, "merge", fake_merge)

        dl = HycomDownloader(output_dir=tmp_path, time_tolerance_minutes=600)  # 10h, not the 6h default
        seg_start = datetime(2026, 7, 14, 10, 30, 0)
        seg_end = datetime(2026, 7, 14, 10, 50, 0)
        dl._download_segment("espc_d_v02", seg_start, seg_end, -77.0, -68.0, 35.0, 44.0)

        time_slice = [kw for kw in sel_calls if "time" in kw][0]["time"]
        assert time_slice.start == seg_start - timedelta(hours=10)
        assert time_slice.stop == seg_end + timedelta(hours=10)

    def test_buffered_start_clamped_at_hycom_min_date(self, tmp_path, monkeypatch):
        """A segment already clamped to _HYCOM_MIN_DATE by
        _resolve_hycom_segments must not have its actual OPeNDAP request
        pushed further back before that date by the bracket buffer."""
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import (
            _HYCOM_MIN_DATE,
            HycomDownloader,
        )

        sel_calls: list = []
        fake_open_dataset, fake_merge = self._make_fake_dataset_classes(sel_calls)
        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        monkeypatch.setattr(xr, "merge", fake_merge)

        dl = HycomDownloader(output_dir=tmp_path)
        seg_start = _HYCOM_MIN_DATE  # already clamped, as _resolve_hycom_segments produces
        seg_end = datetime(2018, 12, 10)
        dl._download_segment("gofs31_930", seg_start, seg_end, 0.5, 6.0, 40.0, 55.0)

        time_calls = [kw for kw in sel_calls if "time" in kw]
        assert len(time_calls) == 1
        time_slice = time_calls[0]["time"]
        assert time_slice.start >= _HYCOM_MIN_DATE

    def test_probe_coverage_uses_buffered_window(self, tmp_path, monkeypatch, caplog):
        """dry-run coverage reporting must reflect the same buffered
        window a real download would request -- otherwise dry-run output
        would misleadingly under-report what a real run actually
        fetches."""
        import logging

        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        class FakeLazyDataset:
            def __init__(self):
                # 5 hours before the raw requested start -- outside the
                # raw window, but inside the buffered (6h) one.
                self.time = xr.DataArray(
                    pd.to_datetime(["2025-01-01T19:00:00"]), dims="time",
                )

            def close(self):
                pass

        def fake_open_dataset(url, *a, **kw):
            return FakeLazyDataset()

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

        dl = HycomDownloader(output_dir=tmp_path, dry_run=True)
        with caplog.at_level(logging.INFO):
            dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2025-01-02T00:00:00", end="2025-01-02T01:00:00",
            )
        assert "1 granule(s)" in caplog.text


class TestCutoverBoundaryNoOverlap:
    """Regression tests for the ESPC-D-V02/GOFS 3.1 cutover-boundary
    overlap bug: for a recipe window straddling ``_HYCOM_CUTOVER_DATE``,
    ``_resolve_hycom_segments`` splits into two segments whose OWN
    ``[seg_start, seg_end]`` already touch at the cutover instant, and
    ``_buffered_bounds`` widens each segment's REAL OPeNDAP request by
    ``_BRACKET_BUFFER_HOURS`` on both sides -- pre-fix, this let the
    ``gofs31_930`` segment's request reach PAST the cutover and the
    ``espc_d_v02`` segment's request reach BEFORE it, so both datasets'
    real (but different, independently-run) data for the SAME real-world
    instants got downloaded -- ``DataTreeConverter.from_hycom``'s
    ``xr.concat(..., dim="time")`` then produced duplicate, non-monotonic
    timestamps that ``model_collocation.py``'s ``np.searchsorted``-based
    bracket search has no defined/correct behaviour for.

    Unlike ``TestBracketBuffer`` (which only ever exercises ONE segment
    in isolation) and ``test_straddling_window_downloads_both_segments``
    (which mocks ``_download_segment`` away entirely, so it never
    exercises real time-slicing or buffering at all), these tests run
    the REAL ``_download_segment`` for BOTH segments of a genuinely
    straddling window, with REAL (sentinel-valued) source data available
    on BOTH sides of the cutover in BOTH source datasets -- proving the
    fix actively clips the request rather than merely relying on the
    (real) source datasets happening to lack data there.
    """

    _LON_STEP = 0.5

    @classmethod
    def _grid(cls):
        import numpy as np

        lon = np.round(np.arange(0.0, 360.0, cls._LON_STEP), 4)
        lat = np.round(np.arange(20.0, 70.0, 0.5), 4)
        depth = np.array([0.0, 10.0])
        return lon, lat, depth

    @classmethod
    def _make_dataset(cls, time_index, value, var_names):
        import numpy as np
        import xarray as xr

        lon, lat, depth = cls._grid()
        shape = (len(time_index), len(depth), len(lat), len(lon))
        data = {
            v: (("time", "depth", "lat", "lon"), np.full(shape, value))
            for v in var_names
        }
        return xr.Dataset(
            data, coords={"time": time_index, "depth": depth, "lat": lat, "lon": lon},
        )

    def _patch_open_dataset(self, monkeypatch, full_time):
        import xarray as xr

        gofs_ds = self._make_dataset(full_time, -100.0, ["water_u", "water_v"])
        espc_u_ds = self._make_dataset(full_time, 100.0, ["water_u"])
        espc_v_ds = self._make_dataset(full_time, 200.0, ["water_v"])

        real_open_dataset = xr.open_dataset

        def fake_open_dataset(url, *a, **kw):
            if url == "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/2024":
                return gofs_ds
            if url == "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z":
                return espc_u_ds
            if url == "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z":
                return espc_v_ds
            return real_open_dataset(url, *a, **kw)

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        return real_open_dataset

    def test_straddling_window_segments_dont_overlap_in_real_downloaded_time(
        self, tmp_path, monkeypatch,
    ):
        import pandas as pd

        from sar_validation.downloaders.hycom_downloader import (
            _HYCOM_CUTOVER_DATE,
            HycomDownloader,
        )

        full_time = pd.date_range("2024-08-08T00:00:00", "2024-08-11T00:00:00", freq="3h")
        real_open_dataset = self._patch_open_dataset(monkeypatch, full_time)

        dl = HycomDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2024-08-09T22:00:00", end="2024-08-10T02:00:00",
        )
        assert len(paths) == 2
        gofs_path = next(p for p in paths if "gofs31_930" in p.name)
        espc_path = next(p for p in paths if "espc_d_v02" in p.name)

        gofs_result = real_open_dataset(gofs_path)
        espc_result = real_open_dataset(espc_path)
        try:
            gofs_times = pd.to_datetime(gofs_result["time"].values)
            espc_times = pd.to_datetime(espc_result["time"].values)

            assert len(gofs_times) > 0 and len(espc_times) > 0
            assert gofs_times.max() < _HYCOM_CUTOVER_DATE, (
                f"gofs31_930's buffered request reached the cutover or "
                f"beyond ({gofs_times.max()}) -- it must never overlap "
                f"with espc_d_v02's own preferred date range."
            )
            assert espc_times.min() >= _HYCOM_CUTOVER_DATE, (
                f"espc_d_v02's buffered request reached before the "
                f"cutover ({espc_times.min()}) -- it must never overlap "
                f"with gofs31_930's own preferred date range."
            )
            assert set(gofs_times) & set(espc_times) == set(), (
                "gofs31_930 and espc_d_v02 downloaded overlapping "
                "real-world timestamps -- xr.concat at from_hycom time "
                "would produce ambiguous duplicate timestamps."
            )

            # The two segments' granule sets must still be CONTIGUOUS on
            # the shared 3-hourly cadence grid across the cutover --
            # clipping must not open a bracket-finding gap at the
            # boundary (both real datasets are cadence-aligned and meet
            # exactly at the cutover instant).
            combined_times = sorted(set(gofs_times) | set(espc_times))
            gaps = [b - a for a, b in zip(combined_times, combined_times[1:])]
            assert all(g == pd.Timedelta(hours=3) for g in gaps), (
                f"combined granule series is not evenly 3-hourly across "
                f"the cutover boundary -- clipping introduced a real "
                f"bracket-finding gap: {combined_times}"
            )
        finally:
            gofs_result.close()
            espc_result.close()

    def test_straddling_window_from_hycom_produces_unique_monotonic_time_honoring_preference(
        self, tmp_path, monkeypatch,
    ):
        """End-to-end: DataTreeConverter.from_hycom's concatenation of
        the two now-disjoint segment files must produce a strictly
        monotonic, duplicate-free time axis, with ESPC-D-V02's sentinel
        value used at/after the cutover and GOFS 3.1's before it -- the
        toolbox's stated dataset preference actually reaching the
        collocation-ready data model_collocation.py consumes."""
        import numpy as np
        import pandas as pd

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.downloaders.hycom_downloader import (
            _HYCOM_CUTOVER_DATE,
            HycomDownloader,
        )

        full_time = pd.date_range("2024-08-08T00:00:00", "2024-08-11T00:00:00", freq="3h")
        self._patch_open_dataset(monkeypatch, full_time)

        dl = HycomDownloader(output_dir=tmp_path)
        paths = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2024-08-09T22:00:00", end="2024-08-10T02:00:00",
        )

        combined = DataTreeConverter.from_hycom(paths)
        assert combined is not None
        times = pd.to_datetime(combined["time"].values)
        assert times.is_unique, f"duplicate timestamps reached from_hycom's output: {times}"
        assert times.is_monotonic_increasing, (
            f"non-monotonic time axis reached from_hycom's output -- "
            f"np.searchsorted-based bracket search in model_collocation.py "
            f"has no correct behaviour for this: {times}"
        )

        before = combined.sel(time=slice(None, _HYCOM_CUTOVER_DATE - pd.Timedelta(seconds=1)))
        at_after = combined.sel(time=slice(_HYCOM_CUTOVER_DATE, None))
        assert before.sizes["time"] > 0 and at_after.sizes["time"] > 0
        np.testing.assert_allclose(before["EWCT"].values, -100.0)
        np.testing.assert_allclose(at_after["EWCT"].values, 100.0)
        np.testing.assert_allclose(at_after["NSCT"].values, 200.0)


class TestDownloadSegmentResourceCleanup:
    def test_all_opened_datasets_are_closed_when_a_later_step_fails(self, tmp_path, monkeypatch):
        """A failure in xr.merge (i.e. *after* every open_dataset call has
        already succeeded) must still close every dataset that was opened
        -- regression test for the resource leak where close() lived after
        the risky merge/sel/load statements inside the same try block."""
        import xarray as xr

        from sar_validation.downloaders.hycom_downloader import HycomDownloader

        closed = []

        class FakeDataset:
            def __init__(self, label):
                self.label = label

            def __getitem__(self, _keys):
                return self

            def close(self):
                closed.append(self.label)

        opened_labels = []

        def fake_open_dataset(url, *a, **kw):
            label = "u" if url.endswith("u3z") else "v"
            opened_labels.append(label)
            return FakeDataset(label)

        def fake_merge(_datasets, *a, **kw):
            raise RuntimeError("simulated merge failure")

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        monkeypatch.setattr(xr, "merge", fake_merge)

        dl = HycomDownloader(output_dir=tmp_path)
        result = dl._download_segment(
            "espc_d_v02", datetime(2025, 1, 1), datetime(2025, 1, 2),
            -10.0, 10.0, 40.0, 55.0,
        )

        assert result is None
        assert opened_labels == ["u", "v"]
        assert sorted(closed) == ["u", "v"]
