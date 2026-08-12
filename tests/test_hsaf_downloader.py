"""Tests for HSAFDownloader (H-SAF ASCAT SSM NRT, 12.5km, FTP)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.downloaders.hsaf_downloader import (
    HSAFDownloader,
    _matches_h29_nc,
    _parse_sensing_start,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -70.0, -30.0, 50.0, 67.0

_REAL_FILENAME = (
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-12.5km-H29_C_LIIB_"
    "20260609001514_20260608231200_20260608231459____.nc"
)

_METOPA_FILENAME = (
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-12.5km-H29_C_LIIB_"
    "20260610001514_20260609231200_20260609231459____.nc"
)


class TestMatchesH29Nc:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            pytest.param(_REAL_FILENAME, True, id="real_metopb_file_matches"),
            pytest.param(
                (
                    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-12.5km-H29_C_LIIB_"
                    "20260609001514_20260608231200_20260608231459____.nc"
                ),
                True, id="metopa_file_matches",
            ),
            pytest.param(_REAL_FILENAME + ".md5", False, id="md5_sidecar_never_matches"),
            pytest.param("readme.txt", False, id="unrelated_file_never_matches"),
        ],
    )
    def test_matches_h29_nc(self, filename, expected):
        assert _matches_h29_nc(filename) is expected


class TestParseSensingStart:
    def test_parses_real_filename(self):
        assert _parse_sensing_start(_REAL_FILENAME) == datetime(
            2026, 6, 8, 23, 12, 0, tzinfo=timezone.utc
        )

    def test_unparseable_filename_returns_none(self):
        assert _parse_sensing_start("readme.txt") is None


class TestHSAFDownloaderDownloadAll:
    def test_downloads_every_matching_file_in_window(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [
            _REAL_FILENAME,
            _METOPA_FILENAME,
            _REAL_FILENAME + ".md5",
        ]

        def fake_retrbinary(cmd, callback):
            callback(b"fake nc bytes")

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.hsaf_downloader.authenticate_hsaf_ftp",
            return_value=("user", "pass"),
        ):
            dl = HSAFDownloader(output_dir=tmp_path)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-08", end="2026-06-09",
            )

        assert len(result) == 2
        assert result[0].name == _METOPA_FILENAME
        assert result[1].name == _REAL_FILENAME
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
        fake_ftp.nlst.return_value = [_REAL_FILENAME]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")
        existing = tmp_path / _REAL_FILENAME
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
        fake_ftp.nlst.return_value = [_REAL_FILENAME]

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
        assert _REAL_FILENAME in capsys.readouterr().out
