"""Tests for ScatterometerFTPDownloader (HY-2B/HY-2C/Oceansat-3, 25 km, FTP)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.downloaders.scatterometer_ftp_downloader import (
    ScatterometerFTPDownloader,
    _matches_25km,
    _parse_filename_timestamp,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -20.0, 0.0, 35.0, 60.0


class TestMatches25km:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            pytest.param(
                "oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc",
                True,
                id="oceansat3_25km_file_matches",
            ),
            pytest.param(
                "hscat_20260609_062235_hy_2c__28817_o_250_4006_ovw_l2.nc.gz",
                True,
                id="hy2c_25km_gz_file_matches",
            ),
            pytest.param(
                "oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc.md5",
                False,
                id="md5_sidecar_never_matches",
            ),
        ],
    )
    def test_matches_25km(self, filename, expected):
        assert _matches_25km(filename) is expected


class TestParseFilenameTimestamp:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            pytest.param(
                "oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc",
                datetime(2026, 7, 18, 4, 36, 6, tzinfo=timezone.utc),
                id="oceansat3_filename",
            ),
            pytest.param("readme.txt", None, id="no_timestamp_returns_none"),
        ],
    )
    def test_parse_filename_timestamp(self, filename, expected):
        assert _parse_filename_timestamp(filename) == expected


class TestScatterometerFTPDownloaderConstruction:
    def test_unknown_satellite_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown satellite"):
            ScatterometerFTPDownloader(satellite="ascat_b", output_dir=tmp_path)


class TestScatterometerFTPDownloaderRecencyGuard:
    @pytest.mark.parametrize(
        "days_offset",
        [
            pytest.param(4, id="old_end_date"),
            pytest.param(3, id="date_only_end_near_boundary"),
        ],
    )
    def test_stale_end_date_returns_empty_without_touching_network(self, tmp_path, caplog, days_offset):
        """Regression (days_offset=3): the matching-window expansion for
        date-only `end` strings (adding one day so files timestamped later
        than midnight on the end date still match) must never leak into the
        recency-guard comparison. If it did, a date-only `end` exactly 3
        calendar days before today would have its midnight timestamp bumped
        by the expansion to 2 days before today -- past the 3-day cutoff --
        and incorrectly slip past the guard to touch the network. Using the
        un-expanded end date, exactly-3-days-old is still older than
        `now - 3 days`, so the guard must fire and no FTP connection may be
        attempted."""
        import logging

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=days_offset)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=5)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        with patch("ftplib.FTP") as mock_ftp_cls, caplog.at_level(logging.WARNING):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        mock_ftp_cls.assert_not_called()
        assert any("days old" in r.message for r in caplog.records)

    def test_recent_end_date_proceeds_to_connect(self, tmp_path):
        now = datetime.now(timezone.utc)
        end = now.strftime("%Y-%m-%d")
        start = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = []
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        fake_ftp.login.assert_called_once_with("user", "pass")
        fake_ftp.cwd.assert_called_once_with("/scat/netcdf/oceansat3")


class TestScatterometerFTPDownloaderDownloadAll:
    def test_downloads_every_matching_file_not_just_the_last(self, tmp_path):
        """Regression: the tutorial notebook's loop only kept the last match
        in its file listing; this downloader must fetch every match."""
        now = datetime.now(timezone.utc)
        ts1 = now.strftime("%Y%m%d_%H%M%S")
        ts2 = (now - timedelta(hours=1)).strftime("%Y%m%d_%H%M%S")
        name1 = f"oscat_{ts1}_ocsat3_1_o_250_4007_ovw_l2.nc"
        name2 = f"oscat_{ts2}_ocsat3_2_o_250_4007_ovw_l2.nc"

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name1, name2, name1 + ".md5"]

        def fake_retrbinary(cmd, callback):
            callback(b"fake-bytes")

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        end = now.strftime("%Y-%m-%dT%H:%M:%S")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert len(out) == 2
        assert {p.name for p in out} == {name1, name2}
        assert fake_ftp.retrbinary.call_count == 2

    def test_no_matching_files_returns_empty_without_crashing(self, tmp_path):
        now = datetime.now(timezone.utc)
        dl = ScatterometerFTPDownloader(satellite="hy2b", output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = []

        start = (now - timedelta(hours=1)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        fake_ftp.retrbinary.assert_not_called()


class TestScatterometerFTPDownloaderGzip:
    @pytest.mark.parametrize(
        "satellite,is_gz,payload_bytes",
        [
            pytest.param(
                "hy2b", True, b"hello netcdf bytes",
                id="gz_match_is_gunzipped_and_original_removed",
            ),
            pytest.param(
                "oceansat3", False, b"raw-nc-bytes",
                id="non_gz_match_saved_as_is",
            ),
        ],
    )
    def test_gzip_handling(self, tmp_path, satellite, is_gz, payload_bytes):
        import gzip

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        if is_gz:
            name = f"hscat_{ts}_hy_2b__1_o_250_4006_ovw_l2.nc.gz"
            retrbinary_bytes = gzip.compress(payload_bytes)
        else:
            name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"
            retrbinary_bytes = payload_bytes

        dl = ScatterometerFTPDownloader(satellite=satellite, output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name]

        def fake_retrbinary(cmd, callback):
            callback(retrbinary_bytes)

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        start = (now - timedelta(hours=1)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert len(out) == 1
        expected_name = name[:-3] if is_gz else name
        assert out[0].name == expected_name
        assert out[0].read_bytes() == payload_bytes
        if is_gz:
            assert not (tmp_path / name).exists()


class TestScatterometerFTPDownloaderForceDownload:
    @pytest.mark.parametrize(
        "force_download,pre_existing_bytes,expect_retrbinary_called,expected_bytes",
        [
            pytest.param(False, b"already here", False, b"already here", id="skips_existing_file"),
            pytest.param(True, b"stale", True, b"fresh", id="force_download_redownloads_existing_file"),
        ],
    )
    def test_force_download(
        self, tmp_path, force_download, pre_existing_bytes, expect_retrbinary_called, expected_bytes,
    ):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"
        (tmp_path / name).write_bytes(pre_existing_bytes)

        dl = ScatterometerFTPDownloader(
            satellite="oceansat3", output_dir=tmp_path, force_download=force_download,
        )
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name]

        def fake_retrbinary(cmd, callback):
            callback(b"fresh")

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        start = (now - timedelta(hours=1)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert len(out) == 1
        if expect_retrbinary_called:
            fake_ftp.retrbinary.assert_called_once()
        else:
            fake_ftp.retrbinary.assert_not_called()
        assert (tmp_path / name).read_bytes() == expected_bytes


class TestScatterometerFTPDownloaderDryRun:
    def test_dry_run_prints_and_touches_no_network(self, tmp_path, capsys):
        now = datetime.now(timezone.utc)
        end = now.strftime("%Y-%m-%d")
        start = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="hy2c", output_dir=tmp_path, dry_run=True)
        with patch("ftplib.FTP") as mock_ftp_cls:
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        mock_ftp_cls.assert_not_called()
        assert "DRY RUN" in capsys.readouterr().out
