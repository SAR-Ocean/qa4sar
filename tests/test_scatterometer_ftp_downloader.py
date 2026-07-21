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
    def test_oceansat3_25km_file_matches(self):
        assert _matches_25km("oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc")

    def test_oceansat3_50km_file_does_not_match(self):
        assert not _matches_25km("oscat_20260718_043610_ocsat3_19234_o_500_4007_ovw_l2.nc")

    def test_hy2c_25km_gz_file_matches(self):
        assert _matches_25km("hscat_20260609_062235_hy_2c__28817_o_250_4006_ovw_l2.nc.gz")

    def test_md5_sidecar_never_matches(self):
        assert not _matches_25km("oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc.md5")


class TestParseFilenameTimestamp:
    def test_oceansat3_filename(self):
        ts = _parse_filename_timestamp("oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc")
        assert ts == datetime(2026, 7, 18, 4, 36, 6, tzinfo=timezone.utc)

    def test_hy2c_filename(self):
        ts = _parse_filename_timestamp("hscat_20260609_062235_hy_2c__28817_o_250_4006_ovw_l2.nc.gz")
        assert ts == datetime(2026, 6, 9, 6, 22, 35, tzinfo=timezone.utc)

    def test_no_timestamp_returns_none(self):
        assert _parse_filename_timestamp("readme.txt") is None


class TestScatterometerFTPDownloaderConstruction:
    def test_unknown_satellite_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown satellite"):
            ScatterometerFTPDownloader(satellite="ascat_b", output_dir=tmp_path)


class TestScatterometerFTPDownloaderRecencyGuard:
    def test_old_end_date_returns_empty_without_touching_network(self, tmp_path, caplog):
        import logging

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=4)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=5)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        with patch("ftplib.FTP") as mock_ftp_cls, caplog.at_level(logging.WARNING):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        mock_ftp_cls.assert_not_called()
        assert any("days old" in r.message for r in caplog.records)

    def test_date_only_end_near_boundary_still_guards_against_stale_data(self, tmp_path, caplog):
        """Regression: the matching-window expansion for date-only `end`
        strings (adding one day so files timestamped later than midnight on
        the end date still match) must never leak into the recency-guard
        comparison. If it did, a date-only `end` exactly 3 calendar days
        before today would have its midnight timestamp bumped by the
        expansion to 2 days before today -- past the 3-day cutoff -- and
        incorrectly slip past the guard to touch the network. Using the
        un-expanded end date, exactly-3-days-old is still older than
        `now - 3 days` (since `now` has a nonzero time-of-day), so the
        guard must fire and no FTP connection may be attempted. This
        deterministically reproduces the bug regardless of what time of
        day the test runs."""
        import logging

        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
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
    def test_gz_match_is_gunzipped_and_original_removed(self, tmp_path):
        import gzip

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        gz_name = f"hscat_{ts}_hy_2b__1_o_250_4006_ovw_l2.nc.gz"

        payload = b"hello netcdf bytes"
        gz_bytes = gzip.compress(payload)

        dl = ScatterometerFTPDownloader(satellite="hy2b", output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [gz_name]

        def fake_retrbinary(cmd, callback):
            callback(gz_bytes)

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        start = (now - timedelta(hours=1)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert len(out) == 1
        assert out[0].name == gz_name[:-3]
        assert out[0].read_bytes() == payload
        assert not (tmp_path / gz_name).exists()

    def test_non_gz_match_saved_as_is(self, tmp_path):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name]

        def fake_retrbinary(cmd, callback):
            callback(b"raw-nc-bytes")

        fake_ftp.retrbinary.side_effect = fake_retrbinary

        start = (now - timedelta(hours=1)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert len(out) == 1
        assert out[0].name == name
        assert out[0].read_bytes() == b"raw-nc-bytes"


class TestScatterometerFTPDownloaderForceDownload:
    def test_skips_existing_file(self, tmp_path):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"
        (tmp_path / name).write_bytes(b"already here")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name]

        start = (now - timedelta(hours=1)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert len(out) == 1
        fake_ftp.retrbinary.assert_not_called()

    def test_force_download_redownloads_existing_file(self, tmp_path):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"
        (tmp_path / name).write_bytes(b"stale")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path, force_download=True)
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
            dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        fake_ftp.retrbinary.assert_called_once()
        assert (tmp_path / name).read_bytes() == b"fresh"


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
