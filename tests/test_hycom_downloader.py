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
