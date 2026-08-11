"""Tests for HycomDownloader."""

from __future__ import annotations

from datetime import datetime

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
