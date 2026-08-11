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
