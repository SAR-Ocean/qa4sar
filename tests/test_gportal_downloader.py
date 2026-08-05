"""Tests for GPortalAMSR2Downloader (JAXA G-Portal SFTP, AMSR2 soil moisture)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

paramiko = pytest.importorskip("paramiko")

from sar_validation.downloaders.gportal_downloader import (  # noqa: E402
    GPortalAMSR2Downloader,
    _connect_with_retry,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -10.0, 10.0, 40.0, 55.0


class TestGPortalAMSR2DownloaderDryRun:
    def test_dry_run_prints_params_without_network(self, tmp_path, capsys):
        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, dry_run=True, username="u", password="p",
        )
        with patch("paramiko.Transport") as mock_transport_cls:
            out = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )
        assert out == []
        mock_transport_cls.assert_not_called()
        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert "ftp.gportal.jaxa.jp" in captured


class TestConnectWithRetry:
    """_connect_with_retry retries once on a transient connection-level
    failure (observed live: "Error reading SSH protocol banner" -- the
    TCP handshake succeeds but the server closes/goes silent before
    sending the SSH banner) rather than aborting this whole best-effort
    fallback source on a single blip."""

    def test_succeeds_on_first_attempt_without_sleeping(self):
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport") as mock_from_transport, \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            mock_transport = MagicMock()
            mock_transport_cls.return_value = mock_transport
            mock_from_transport.return_value = "sftp-instance"

            transport, sftp = _connect_with_retry("u", "p")

        assert transport is mock_transport
        assert sftp == "sftp-instance"
        mock_transport_cls.assert_called_once()
        mock_sleep.assert_not_called()

    def test_retries_once_after_transient_ssh_exception_then_succeeds(self):
        failing_transport = MagicMock()
        failing_transport.connect.side_effect = paramiko.SSHException(
            "Error reading SSH protocol banner"
        )
        working_transport = MagicMock()

        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value="sftp-instance"), \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            mock_transport_cls.side_effect = [failing_transport, working_transport]

            transport, sftp = _connect_with_retry("u", "p")

        assert transport is working_transport
        assert sftp == "sftp-instance"
        assert mock_transport_cls.call_count == 2
        failing_transport.close.assert_called_once()
        mock_sleep.assert_called_once_with(2.0)

    def test_gives_up_after_max_attempts_on_persistent_ssh_exception(self):
        always_failing = MagicMock()
        always_failing.connect.side_effect = paramiko.SSHException(
            "Error reading SSH protocol banner"
        )

        with patch("paramiko.Transport", return_value=always_failing), \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(paramiko.SSHException, match="SSH protocol banner"):
                _connect_with_retry("u", "p")

        assert always_failing.close.call_count == 2
        mock_sleep.assert_called_once()

    def test_non_retryable_exception_propagates_immediately(self):
        """A genuine auth failure (a plain Exception, not one of the
        transient connection-error types) must not be retried."""
        failing_transport = MagicMock()
        failing_transport.connect.side_effect = Exception("auth failed")

        with patch("paramiko.Transport", return_value=failing_transport), \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(Exception, match="auth failed"):
                _connect_with_retry("u", "p")

        failing_transport.close.assert_called_once()
        mock_sleep.assert_not_called()


class TestGPortalAMSR2DownloaderDiscovery:
    def _mock_sftp(self, listing_by_path: dict[str, list[str]]):
        """Build a fake paramiko.SFTPClient whose .listdir(path) returns
        listing_by_path[path], keyed by the exact path string passed in."""
        sftp = MagicMock()
        sftp.listdir.side_effect = lambda path: listing_by_path.get(path, [])
        return sftp

    def test_discovers_amsr2_product_directory_under_standard(self, tmp_path):
        listing = {
            "standard": ["GCOM-W", "GPM", "GCOM-C"],
            "standard/GCOM-W": ["GCOM-W.AMSR2", "GCOM-W.SGLI"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD", "L3.SST"],
            # Version directory (e.g. algorithm version "2210", matching the
            # trailing version code already embedded in the filenames below)
            # sits between the product directory and Year, per the G-Portal
            # manual's standard/.../[Product Name]/[Version]/[Year]/[Month]/
            # layout documented in the module docstring.
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07": [
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
                "GW1AM2_20260702_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)

        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection") as mock_create_connection:
            mock_create_connection.return_value = MagicMock()
            mock_transport_cls.return_value = MagicMock()
            sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        mock_create_connection.assert_called_once()
        assert len(downloaded) == 2
        assert all(p.exists() for p in downloaded)

    def test_no_confident_match_prints_listing_and_returns_empty(self, tmp_path, capsys):
        listing = {
            "standard": ["GPM", "GCOM-C"],
            "standard/GPM": ["GPM.DPR"],
            "standard/GCOM-C": ["GCOM-C.SGLI"],
            "nrt": ["GPM"],
            "nrt/GPM": ["GPM.DPR"],
        }
        sftp = self._mock_sftp(listing)

        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert downloaded == []
        captured = capsys.readouterr().out
        assert "GPM" in captured  # the raw listing it found got printed

    def test_falls_through_to_nrt_when_standard_matches_but_yields_zero_files_in_window(self, tmp_path):
        """Regression test: standard/ has a confidently-matched AMSR2
        product directory (as it always does on a real account), but no
        files there fall inside the requested date window (e.g. a very
        recent date not yet propagated to standard/'s Year/Month
        archive). nrt/ also has a confidently-matched product directory,
        with a file that IS in the window -- it must actually be reached
        and downloaded, not skipped just because standard/ matched first."""
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["06"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/06": [
                "GW1AM2_20260630_01D_EQMA_L3SGSMCLQ_2210.h5",  # before window
            ],
            "nrt": ["GCOM-W"],
            "nrt/GCOM-W": ["GCOM-W.AMSR2"],
            "nrt/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "nrt/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": [
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",  # in window
            ],
        }
        sftp = self._mock_sftp(listing)
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")

        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(downloaded) == 1
        assert "20260701" in downloaded[0].name
        sftp.get.assert_called_once()
        remote_path_used = sftp.get.call_args.args[0]
        assert remote_path_used.startswith("nrt/")

    def test_falls_through_past_decoy_sensor_to_real_sensor_within_same_tree(self, tmp_path):
        """Regression test against a real G-Portal account: standard/
        lists BOTH a decoy sensor directory whose name happens to match
        the AMSR2 pattern (AQUA/AQUA.AMSR-E_AMSR2Format -- real directory
        name; it's AMSR-E, a different, retired-since-2011 instrument,
        reformatted to look like AMSR2's file layout for cross-instrument
        continuity, not real AMSR2 data -- confirmed its own L3.SMC_10
        archive only spans 2002-2011) ahead of the genuine GCOM-W/
        GCOM-W.AMSR2 sensor (confirmed to hold real, current data). The
        real sensor must still be reached even though the decoy is
        listed first and matches the sensor-name pattern too."""
        listing = {
            "standard": ["AQUA", "GCOM-W"],
            "standard/AQUA": ["AQUA.AMSR-E", "AQUA.AMSR-E_AMSR2Format"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format": ["L3.SMC_10"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format/L3.SMC_10": ["8"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format/L3.SMC_10/8": ["2011"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07": [
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")

        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(downloaded) == 1
        assert "20260701" in downloaded[0].name

    def test_filters_files_by_embedded_date(self, tmp_path):
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07": [
                "GW1AM2_20260630_01D_EQMA_L3SGSMCLQ_2210.h5",  # before window
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",  # in window
                "GW1AM2_20260705_01D_EQMA_L3SGSMCLQ_2210.h5",  # after window
            ],
        }
        sftp = self._mock_sftp(listing)
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")

        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(downloaded) == 1
        assert "20260701" in downloaded[0].name


class TestGPortalAMSR2DownloaderForceDownload:
    def test_skips_already_downloaded_file(self, tmp_path):
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07": [
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        existing = tmp_path / "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5"
        existing.write_bytes(b"already here")
        sftp = MagicMock()
        sftp.listdir.side_effect = lambda path: listing.get(path, [])

        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        sftp.get.assert_not_called()
        assert downloaded == [existing]


class TestGPortalAMSR2DownloaderResourceCleanup:
    def test_closes_transport_on_connect_failure(self, tmp_path):
        """Verify that transport.close() is called even if transport.connect() raises."""
        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")

        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport = MagicMock()
            mock_transport_cls.return_value = mock_transport
            mock_transport.connect.side_effect = Exception("auth failed")

            with patch.object(dl, "_username", "u"), \
                 patch.object(dl, "_password", "p"):
                try:
                    dl.download(
                        min_lon=_MIN_LON, max_lon=_MAX_LON,
                        min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                        start="2026-07-01", end="2026-07-02",
                    )
                except Exception:
                    pass  # Expected: the auth failure should propagate

            # Even though connect() failed, transport.close() should have been called
            mock_transport.close.assert_called_once()

    def test_closes_sftp_on_from_transport_failure(self, tmp_path):
        """Verify that transport.close() is called even if from_transport() raises."""
        dl = GPortalAMSR2Downloader(output_dir=tmp_path, username="u", password="p")

        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport") as mock_from_transport, \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport = MagicMock()
            mock_transport_cls.return_value = mock_transport
            mock_from_transport.side_effect = Exception("from_transport failed")

            with patch.object(dl, "_username", "u"), \
                 patch.object(dl, "_password", "p"):
                try:
                    dl.download(
                        min_lon=_MIN_LON, max_lon=_MAX_LON,
                        min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                        start="2026-07-01", end="2026-07-02",
                    )
                except Exception:
                    pass  # Expected: the from_transport failure should propagate

            # Even though from_transport() failed, transport.close() should have been called
            mock_transport.close.assert_called_once()
