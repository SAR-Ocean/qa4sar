"""Tests for HSAFDownloader (H-SAF ASCAT SSM NRT, H29 12.5km / H122 6.25km, FTP)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.downloaders.hsaf_downloader import (
    HSAFDownloader,
    _matches_ascat_nc,
    _parse_satellite,
    _parse_sensing_end,
    _parse_sensing_start,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -70.0, -30.0, 50.0, 67.0

_REAL_H29_FILENAME = (
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-12.5km-H29_C_LIIB_"
    "20260609001514_20260608231200_20260608231459____.nc"
)

_METOPA_H29_FILENAME = (
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-12.5km-H29_C_LIIB_"
    "20260610001514_20260609231200_20260609231459____.nc"
)

_REAL_H122_FILENAME = (
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-6.25km-H122_C_LIIB_"
    "20260609001515_20260608231200_20260608231459____.nc"
)


class TestMatchesAscatNc:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            pytest.param(_REAL_H29_FILENAME, True, id="real_h29_metopb_file_matches"),
            pytest.param(_METOPA_H29_FILENAME, True, id="h29_metopa_file_matches"),
            pytest.param(_REAL_H122_FILENAME, True, id="real_h122_metopb_file_matches"),
            pytest.param(_REAL_H29_FILENAME + ".md5", False, id="md5_sidecar_never_matches"),
            pytest.param("readme.txt", False, id="unrelated_file_never_matches"),
        ],
    )
    def test_matches_ascat_nc(self, filename, expected):
        assert _matches_ascat_nc(filename) is expected


class TestParseSensingStart:
    def test_parses_real_h29_filename(self):
        assert _parse_sensing_start(_REAL_H29_FILENAME) == datetime(
            2026, 6, 8, 23, 12, 0, tzinfo=timezone.utc
        )

    def test_parses_real_h122_filename(self):
        assert _parse_sensing_start(_REAL_H122_FILENAME) == datetime(
            2026, 6, 8, 23, 12, 0, tzinfo=timezone.utc
        )

    def test_unparseable_filename_returns_none(self):
        assert _parse_sensing_start("readme.txt") is None


class TestHSAFDownloaderProductSelection:
    def test_unknown_product_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown H-SAF product"):
            HSAFDownloader(output_dir=tmp_path, product="h999")

    def test_default_product_queries_h122_path(self, tmp_path):
        """H122 is now the default (higher resolution than H29) -- see
        spec §1b. Constructing HSAFDownloader with no explicit product
        must query H122's FTP directory."""
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H122_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path)
            assert dl.product == "h122"
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        fake_ftp.cwd.assert_called_once_with("/h122/h122_cur_mon_nc")
        assert len(result) == 1
        assert result[0].name == _REAL_H122_FILENAME

    def test_explicit_h29_product_queries_h29_path(self, tmp_path):
        """Recipes that need the legacy 12.5km product opt back in via
        product="h29" (recipe-level: download_kwargs: {hsaf_product: h29})."""
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H29_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path, product="h29")
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        fake_ftp.cwd.assert_called_once_with("/h29/h29_cur_mon_nc")
        assert len(result) == 1
        assert result[0].name == _REAL_H29_FILENAME


class TestHSAFDownloaderDownloadAll:
    def test_downloads_every_matching_file_in_window(self, tmp_path):
        """Explicit product="h29" preserves this test's original intent
        (verifying multi-file matching/sorting within one product's
        listing) now that h122 is the constructor default."""
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [
            _REAL_H29_FILENAME,
            _METOPA_H29_FILENAME,
            _REAL_H29_FILENAME + ".md5",
        ]

        def fake_retrbinary(cmd, callback):
            callback(b"fake nc bytes")

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path, product="h29")
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert len(result) == 2
        assert result[0].name == _METOPA_H29_FILENAME
        assert result[1].name == _REAL_H29_FILENAME
        fake_ftp.login.assert_called_once_with("user", "pass")
        fake_ftp.cwd.assert_called_once_with("/h29/h29_cur_mon_nc")

    def test_no_matching_files_returns_empty_without_crashing(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = ["readme.txt"]

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert result == []

    def test_force_download_refetches_existing_file(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H122_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")
        existing = tmp_path / _REAL_H122_FILENAME
        existing.write_text("stale")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path, force_download=True)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert fake_ftp.retrbinary.called
        assert len(result) == 1

    def test_dry_run_lists_matches_without_downloading(self, tmp_path, capsys):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H122_FILENAME]

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path, dry_run=True)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert result == []
        assert not fake_ftp.retrbinary.called
        assert _REAL_H122_FILENAME in capsys.readouterr().out


class TestParseSatellite:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            pytest.param(_REAL_H29_FILENAME, "metop-b", id="metopb_h29"),
            pytest.param(_REAL_H122_FILENAME, "metop-b", id="metopb_h122"),
            pytest.param(_METOPA_H29_FILENAME, "metop-a", id="metopa_h29"),
            pytest.param("readme.txt", None, id="unparseable"),
        ],
    )
    def test_parse_satellite(self, filename, expected):
        assert _parse_satellite(filename) == expected


class TestParseSensingEnd:
    def test_parses_real_h29_filename(self):
        assert _parse_sensing_end(_REAL_H29_FILENAME) == datetime(
            2026, 6, 8, 23, 14, 59, tzinfo=timezone.utc
        )

    def test_unparseable_filename_returns_none(self):
        assert _parse_sensing_end("readme.txt") is None


class TestHSAFDownloaderOrbitPrefilter:
    def test_default_orbit_prefilter_is_enabled(self, tmp_path):
        assert HSAFDownloader(output_dir=tmp_path).orbit_prefilter is True

    def test_dropped_files_are_excluded_from_download(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H29_FILENAME, _METOPA_H29_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        def fake_overlap(satellite, start, end, *bbox, **kwargs):
            return satellite == "metop-b"  # only let METOP-B's file through

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox", side_effect=fake_overlap,
        ):
            dl = HSAFDownloader(output_dir=tmp_path, product="h29")
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert len(result) == 1
        assert result[0].name == _REAL_H29_FILENAME

    def test_orbit_prefilter_false_reproduces_todays_behavior(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H29_FILENAME, _METOPA_H29_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ), patch("sar_validation.core.orbit_coverage.orbit_overlaps_bbox") as mock_overlap:
            dl = HSAFDownloader(output_dir=tmp_path, product="h29", orbit_prefilter=False)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert not mock_overlap.called
        assert len(result) == 2

    def test_unparseable_filename_is_kept_fail_open(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [_REAL_H29_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
            side_effect=AssertionError("must not be called for an unparseable satellite"),
        ), patch(
            "sar_validation.downloaders.hsaf_downloader._parse_satellite", return_value=None,
        ):
            dl = HSAFDownloader(output_dir=tmp_path, product="h29")
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert len(result) == 1
