"""Tests for SMOSDownloader (ESA OADS SMOS soil moisture, HTTPS/SAML2)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from sar_validation.downloaders.smos_downloader import SMOSDownloader, _parse_sensing_window

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -10.0, 10.0, 40.0, 55.0

_REAL_FILENAME = (
    "W_XX-ESA,SMOS,NRTNN_C_LEMM_20260102131619_20260102103700_20260102123603_o_v300_l2sm.nc"
)


class TestParseSensingWindow:
    def test_parses_real_filename(self):
        assert _parse_sensing_window(_REAL_FILENAME) == (
            datetime(2026, 1, 2, 10, 37, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 36, 3, tzinfo=timezone.utc),
        )

    def test_unparseable_filename_returns_none(self):
        assert _parse_sensing_window("readme.txt") is None


class TestListCandidatesDry:
    def test_returns_matches_with_real_sensing_window(self, tmp_path):
        """Mirrors download()'s own day-by-day OADS browse loop, but never
        touches self.dry_run and never fetches a product's bytes."""
        with patch.object(SMOSDownloader, "_login", return_value=None), patch.object(
            SMOSDownloader,
            "_list_products_for_day",
            return_value=[{"filename": _REAL_FILENAME, "download_href": "/oads/data/NRT_Open/x.nc"}],
        ), patch(
            "sar_validation.downloaders.smos_downloader.authenticate_smos_ftp",
            return_value=("user", "pass"),
        ):
            dl = SMOSDownloader(output_dir=tmp_path, orbit_prefilter=False)
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-01-02", end="2026-01-02",
            )

        assert len(candidates) == 1
        name, sensing_start, sensing_end = candidates[0]
        assert name == _REAL_FILENAME
        assert sensing_start == datetime(2026, 1, 2, 10, 37, 0, tzinfo=timezone.utc)
        assert sensing_end == datetime(2026, 1, 2, 12, 36, 3, tzinfo=timezone.utc)

    def test_unparseable_filename_falls_back_to_whole_day_window(self, tmp_path):
        """Mirrors _filter_by_orbit_overlap's own fallback for a filename
        that doesn't match the expected OADS naming convention."""
        with patch.object(SMOSDownloader, "_login", return_value=None), patch.object(
            SMOSDownloader,
            "_list_products_for_day",
            return_value=[{"filename": "SM_MIR_weird_name.nc", "download_href": "/x"}],
        ), patch(
            "sar_validation.downloaders.smos_downloader.authenticate_smos_ftp",
            return_value=("user", "pass"),
        ):
            dl = SMOSDownloader(output_dir=tmp_path, orbit_prefilter=False)
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-01-02", end="2026-01-02",
            )

        assert len(candidates) == 1
        _name, sensing_start, sensing_end = candidates[0]
        assert sensing_start == datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert sensing_end == datetime(2026, 1, 2, 23, 59, 59, tzinfo=timezone.utc)

    def test_non_product_files_are_excluded(self, tmp_path):
        with patch.object(SMOSDownloader, "_login", return_value=None), patch.object(
            SMOSDownloader,
            "_list_products_for_day",
            return_value=[{"filename": "readme.txt", "download_href": "/x"}],
        ), patch(
            "sar_validation.downloaders.smos_downloader.authenticate_smos_ftp",
            return_value=("user", "pass"),
        ):
            dl = SMOSDownloader(output_dir=tmp_path, orbit_prefilter=False)
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-01-02", end="2026-01-02",
            )

        assert candidates == []

    def test_spans_multiple_days_without_fetching(self, tmp_path):
        """One _list_products_for_day call per day in [start, end], and
        list_candidates_dry never downloads a product's bytes."""
        days_seen = []

        def fake_list(self_, session, day):
            days_seen.append(day)
            return []

        with patch.object(SMOSDownloader, "_login", return_value=None), patch.object(
            SMOSDownloader, "_list_products_for_day", fake_list,
        ), patch(
            "sar_validation.downloaders.smos_downloader.authenticate_smos_ftp",
            return_value=("user", "pass"),
        ):
            dl = SMOSDownloader(output_dir=tmp_path, orbit_prefilter=False)
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-01-01", end="2026-01-03",
            )

        assert len(days_seen) == 3
        assert candidates == []
