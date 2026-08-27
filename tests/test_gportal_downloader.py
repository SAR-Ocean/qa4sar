"""Tests for GPortalAMSR2Downloader (JAXA G-Portal SFTP, AMSR2 soil moisture)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

paramiko = pytest.importorskip("paramiko")

from sar_validation.core.orbit_coverage import TleFetchError  # noqa: E402
from sar_validation.downloaders import gportal_downloader  # noqa: E402
from sar_validation.downloaders.gportal_downloader import (  # noqa: E402
    GPortalAMSR2Downloader,
    _connect_with_retry,
)

_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -10.0, 10.0, 40.0, 55.0


@pytest.fixture(autouse=True)
def _reset_list_candidates_cache():
    gportal_downloader._list_candidates_cache.clear()
    gportal_downloader._list_candidates_locks.clear()
    gportal_downloader._product_dirs_cache.clear()
    yield
    gportal_downloader._list_candidates_cache.clear()
    gportal_downloader._list_candidates_locks.clear()
    gportal_downloader._product_dirs_cache.clear()


class TestGPortalAMSR2DownloaderDryRun:
    def test_dry_run_without_credentials_prints_setup_message_without_network(
        self, tmp_path, capsys, monkeypatch,
    ):
        """No credentials configured (constructor args, env vars, keyring,
        or legacy file) -- dry-run must never prompt interactively for
        one, so it prints a setup message and makes no connection
        attempt at all."""
        monkeypatch.delenv("GPORTAL_USERNAME", raising=False)
        monkeypatch.delenv("GPORTAL_PASSWORD", raising=False)
        dl = GPortalAMSR2Downloader(output_dir=tmp_path, dry_run=True)
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch(
                 "sar_validation.downloaders.base._resolve_from_keyring_or_legacy_file",
                 return_value=(None, None),
             ):
            out = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )
        assert out == []
        mock_transport_cls.assert_not_called()
        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert "not configured" in captured
        assert "--set-credential gportal" in captured

    def test_dry_run_with_credentials_reports_real_matches_without_downloading(
        self, tmp_path, capsys,
    ):
        """Credentials ARE configured -- dry-run connects and lists real
        matching files (so the user knows whether any exist for the
        requested window), but never calls sftp.get()."""
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
        sftp = MagicMock()
        sftp.listdir.side_effect = lambda path: listing.get(path, [])

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, dry_run=True, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            out = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert out == []
        sftp.get.assert_not_called()
        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5" in captured


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


def _assert_downloaded_from_nrt(downloaded, sftp):
    sftp.get.assert_called_once()
    remote_path_used = sftp.get.call_args.args[0]
    assert remote_path_used.startswith("nrt/")


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

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
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

    def test_amsr_e_decoy_never_listed_for_dates_after_retirement(self, tmp_path):
        """A query entirely after _AMSR_E_RETIREMENT_DATE must never even
        list the AMSR-E decoy directory, not just skip past it once found
        -- saving the wasted listdir() round trip."""
        listing = {
            "standard": ["AQUA", "GCOM-W"],
            "standard/AQUA": ["AQUA.AMSR-E_AMSR2Format"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format": ["L3.SMC_10"],
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

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(downloaded) == 1
        listed_paths = [call.args[0] for call in sftp.listdir.call_args_list]
        assert "standard/AQUA/AQUA.AMSR-E_AMSR2Format" not in listed_paths

    def test_amsr_e_decoy_still_reachable_for_dates_before_retirement(self, tmp_path):
        """A genuinely historical query (entirely before
        _AMSR_E_RETIREMENT_DATE) must still be able to reach the AMSR-E
        decoy directory, since it may hold real pre-retirement data."""
        listing = {
            "standard": ["AQUA"],
            "standard/AQUA": ["AQUA.AMSR-E_AMSR2Format"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format": ["L3.SMC_10"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format/L3.SMC_10": ["8"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format/L3.SMC_10/8": ["2010"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format/L3.SMC_10/8/2010": ["06"],
            "standard/AQUA/AQUA.AMSR-E_AMSR2Format/L3.SMC_10/8/2010/06": [
                "GW1AM2_20100601_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2010-06-01", end="2010-06-02",
            )

        assert len(downloaded) == 1
        assert "20100601" in downloaded[0].name

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

    def _run_discovery(self, tmp_path, listing, extra_check=None):
        """Shared harness for the discovery scenarios below: wires up a fake
        SFTP client from *listing*, patches paramiko/socket, writes fake
        bytes on sftp.get, and runs a single download() call over the
        standard 2026-07-01..2026-07-02 window. If *extra_check* is given,
        it's called with (downloaded, sftp) while the patches are still
        active. Returns the downloaded list."""
        sftp = self._mock_sftp(listing)
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )
            if extra_check is not None:
                extra_check(downloaded, sftp)
        return downloaded

    @pytest.mark.parametrize(
        "listing,extra_check",
        [
            pytest.param(
                {
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
                },
                _assert_downloaded_from_nrt,
                id="falls_through_to_nrt_when_standard_matches_but_yields_zero_files_in_window",
            ),
            pytest.param(
                {
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
                },
                None,
                id="falls_through_past_decoy_sensor_to_real_sensor_within_same_tree",
            ),
            pytest.param(
                {
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
                },
                None,
                id="filters_files_by_embedded_date",
            ),
        ],
    )
    def test_discovery_scenarios(self, tmp_path, listing, extra_check):
        """Regression coverage for three discovery-fallback/filtering
        scenarios: (1) standard/ has a confidently-matched AMSR2 product
        directory but no files in the requested window, so nrt/'s
        confidently-matched product directory must actually be reached;
        (2) standard/ lists both a decoy sensor directory ahead of the
        genuine sensor, and the real sensor must still be reached; (3)
        files outside the requested date window must be filtered out by
        their embedded date even when they sit alongside in-window files
        in the same directory listing."""
        downloaded = self._run_discovery(tmp_path, listing, extra_check=extra_check)
        assert len(downloaded) == 1
        assert "20260701" in downloaded[0].name

    def test_monthly_composite_file_excluded_even_when_date_falls_in_window(self, tmp_path):
        """Real G-Portal directories mix daily granules ("..._01D_...")
        with a whole-month composite file ("..._01M_...", dated
        "{year}{month}00" -- day "00" as a placeholder for "the whole
        month") in the same Year/Month listing. Once collocation-tolerance
        padding pushes the requested window's start back across a month
        boundary, the monthly file's "00"-day date can satisfy the plain
        start_date <= date <= end_date comparison just like a real day
        would -- it must still never be selected: from_amsr_ssm can't
        parse a monthly file's HDF5 layout (no "Time Information" group),
        so it silently drops it with a "Missing vsm/longitude/latitude
        field(s)" warning instead of using the real daily granule that
        was also available in the same window."""
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["3300300"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300/2026": ["06", "07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300/2026/06": [],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300/2026/07": [
                "GW1AM2_20260700_01M_EQMA_L3SGSMCHF3300300.h5",  # monthly composite
                "GW1AM2_20260701_01D_EQMA_L3SGSMCHF3300300.h5",  # real daily granule
            ],
        }
        sftp = self._mock_sftp(listing)
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-25", end="2026-07-02",
            )

        assert len(downloaded) == 1
        assert "01D" in downloaded[0].name
        assert "01M" not in downloaded[0].name


class TestGPortalAMSR2DownloaderListCandidatesDry:
    """list_candidates_dry, added for the dry-collocation predictor --
    same SFTP directory-discovery + date-matching logic
    _download_from_product_directory uses, without the orbit prefilter
    and without ever calling sftp.get."""

    def _mock_sftp(self, listing_by_path: "dict[str, list[str]]"):
        sftp = MagicMock()
        sftp.listdir.side_effect = lambda path: listing_by_path.get(path, [])
        return sftp

    def test_returns_matches_without_downloading(self, tmp_path):
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07": [
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
                "GW1AM2_20260702_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        sftp.get.assert_not_called()
        assert len(candidates) == 2
        names = {name for name, _start, _end in candidates}
        assert names == {
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07/GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07/GW1AM2_20260702_01D_EQMA_L3SGSMCLQ_2210.h5",
        }

    def test_sensing_window_is_the_whole_utc_day(self, tmp_path):
        """Filenames only embed a date, not a time-of-day -- each match's
        sensing window must be the whole UTC day, mirroring
        _filter_by_orbit_overlap's own whole-day construction."""
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
        sftp = self._mock_sftp(listing)

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(candidates) == 1
        _name, day_start, day_end = candidates[0]
        assert day_start == datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert day_end == datetime(2026, 7, 1, 23, 59, 59, tzinfo=timezone.utc)

    def test_monthly_composite_file_excluded(self, tmp_path):
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["3300300"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300/2026": ["07"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/3300300/2026/07": [
                "GW1AM2_20260700_01M_EQMA_L3SGSMCHF3300300.h5",  # monthly composite
                "GW1AM2_20260701_01D_EQMA_L3SGSMCHF3300300.h5",  # real daily granule
            ],
        }
        sftp = self._mock_sftp(listing)

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()):
            mock_transport_cls.return_value = MagicMock()
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-06-25", end="2026-07-02",
            )

        assert len(candidates) == 1
        assert "01D" in candidates[0][0]

    def test_never_prompts_interactively_even_when_allow_prompt_defaults_true(self, tmp_path, monkeypatch):
        """Unlike download() (which honors self._allow_prompt, defaulting
        to True for direct/CLI use), list_candidates_dry is a prediction
        call that must never block on an interactive password prompt --
        it always authenticates with allow_prompt=False."""
        monkeypatch.delenv("GPORTAL_USERNAME", raising=False)
        monkeypatch.delenv("GPORTAL_PASSWORD", raising=False)

        dl = GPortalAMSR2Downloader(output_dir=tmp_path)  # allow_prompt defaults True

        with patch("paramiko.Transport") as mock_transport_cls, \
             patch(
                 "sar_validation.downloaders.base._resolve_from_keyring_or_legacy_file",
                 return_value=(None, None),
             ):
            with pytest.raises(RuntimeError):
                dl.list_candidates_dry(
                    min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                    start="2026-07-01", end="2026-07-02",
                )

        mock_transport_cls.assert_not_called()  # never even attempted a connection


class TestGPortalAMSR2DownloaderListCandidatesDryCache:
    """list_candidates_dry's SFTP discovery is shared across calls whose
    start/end fall in the same calendar month(s) -- see
    _list_candidates_cache's own module-level comment. Without this, the
    --dry-collocation-detail exhaustive per-footprint loop reconnects and
    re-walks the whole product-directory tree once per SAR footprint."""

    def _mock_sftp(self, listing_by_path: "dict[str, list[str]]"):
        sftp = MagicMock()
        sftp.listdir.side_effect = lambda path: listing_by_path.get(path, [])
        return sftp

    _LISTING = {
        "standard": ["GCOM-W"],
        "standard/GCOM-W": ["GCOM-W.AMSR2"],
        "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
        "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
        "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
        "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["07"],
        "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/07": [
            "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
            "GW1AM2_20260702_01D_EQMA_L3SGSMCLQ_2210.h5",
        ],
    }

    def test_stops_at_first_directory_with_any_candidates_leaving_nrt_unqueried(self, tmp_path, capsys):
        """Mirrors download()'s own "try each until one yields files"
        loop: once standard/'s archive has any candidate in the cached
        window, nrt/'s ~1-week-retention tree is never even listed for
        that call. Also prints an explicit "stopping here" message, so
        the discovery phase's own "discovered ... nrt/..." line (printed
        for every matching directory regardless of whether it is later
        queried) is never mistaken for evidence that nrt was searched."""
        listing = {
            **self._LISTING,
            "nrt": ["GCOM-W"],
            "nrt/GCOM-W": ["GCOM-W.AMSR2"],
            "nrt/GCOM-W/GCOM-W.AMSR2": ["L2.SMC"],
            "nrt/GCOM-W/GCOM-W.AMSR2/L2.SMC": [
                "GW1AM2_20260703_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)

        def fake_connect(username, password):
            return MagicMock(), sftp

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("sar_validation.downloaders.gportal_downloader._connect_with_retry", side_effect=fake_connect):
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(candidates) == 2
        listed_paths = [call.args[0] for call in sftp.listdir.call_args_list]
        assert "nrt/GCOM-W/GCOM-W.AMSR2/L2.SMC" not in listed_paths

        captured = capsys.readouterr().out
        assert "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD has 2 candidate(s)" in captured
        assert "stopping here" in captured
        assert "nrt/GCOM-W/GCOM-W.AMSR2/L2.SMC has" not in captured

    def test_falls_through_to_nrt_when_standard_has_no_candidates_in_window(self, tmp_path, capsys):
        """The flip side of the "stop early" test above: when standard/'s
        archive has nothing at all for the cached window, nrt/ must still
        be reached, not silently skipped."""
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2025"],
            "nrt": ["GCOM-W"],
            "nrt/GCOM-W": ["GCOM-W.AMSR2"],
            "nrt/GCOM-W/GCOM-W.AMSR2": ["L2.SMC"],
            "nrt/GCOM-W/GCOM-W.AMSR2/L2.SMC": [
                "GW1AM2_20260701_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)

        def fake_connect(username, password):
            return MagicMock(), sftp

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("sar_validation.downloaders.gportal_downloader._connect_with_retry", side_effect=fake_connect):
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(candidates) == 1
        assert candidates[0][0].startswith("nrt/")

        captured = capsys.readouterr().out
        assert "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD has no candidates" in captured
        assert "trying next" in captured
        assert "nrt/GCOM-W/GCOM-W.AMSR2/L2.SMC has 1 candidate(s)" in captured

    def test_two_callers_with_identical_month_share_one_real_discovery(self, tmp_path):
        sftp = self._mock_sftp(self._LISTING)
        connect_calls = []

        def fake_connect(username, password):
            connect_calls.append((username, password))
            return MagicMock(), sftp

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("sar_validation.downloaders.gportal_downloader._connect_with_retry", side_effect=fake_connect):
            first = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )
            second = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        assert len(connect_calls) == 1
        assert len(first) == 2
        assert first == second

    def test_different_months_are_not_shared(self, tmp_path):
        sftp = self._mock_sftp(self._LISTING)
        connect_calls = []

        def fake_connect(username, password):
            connect_calls.append((username, password))
            return MagicMock(), sftp

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("sar_validation.downloaders.gportal_downloader._connect_with_retry", side_effect=fake_connect):
            dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )
            dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-08-01", end="2026-08-02",
            )

        assert len(connect_calls) == 2

    def test_product_directory_discovery_is_shared_across_different_months(self, tmp_path):
        """_discover_product_directory's own result (which Project/Sensor/
        Product directories hold AMSR2 data) does not depend on the
        requested start/end at all -- it must run at most once per
        process even though the per-month raw-candidate cache itself is
        legitimately not shared across different calendar months (see
        test_different_months_are_not_shared)."""
        sftp = self._mock_sftp(self._LISTING)
        discover_calls = []

        def fake_connect(username, password):
            return MagicMock(), sftp

        real_discover = GPortalAMSR2Downloader._discover_product_directory

        def counting_discover(self, sftp, exclude_amsr_e=False):
            discover_calls.append(1)
            return real_discover(self, sftp, exclude_amsr_e=exclude_amsr_e)

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("sar_validation.downloaders.gportal_downloader._connect_with_retry", side_effect=fake_connect), \
             patch.object(GPortalAMSR2Downloader, "_discover_product_directory", counting_discover):
            dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )
            dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-08-01", end="2026-08-02",
            )

        assert len(discover_calls) == 1

    def test_concurrent_callers_with_identical_month_still_share_one_discovery(self, tmp_path):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        sftp = self._mock_sftp(self._LISTING)
        connect_calls = []
        start_barrier = threading.Barrier(2)

        def fake_connect(username, password):
            connect_calls.append((username, password))
            return MagicMock(), sftp

        def call_with_synced_start():
            start_barrier.wait(timeout=5)
            return dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-02",
            )

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
        with patch("sar_validation.downloaders.gportal_downloader._connect_with_retry", side_effect=fake_connect):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(call_with_synced_start) for _ in range(2)]
                results = [f.result(timeout=5) for f in futures]

        assert len(connect_calls) == 1
        assert results[0] == results[1]


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

        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=False,
        )
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


class TestGPortalAMSR2DownloaderOrbitPrefilter:
    def _mock_sftp(self, listing_by_path: dict[str, list[str]]):
        sftp = MagicMock()
        sftp.listdir.side_effect = lambda path: listing_by_path.get(path, [])
        sftp.get.side_effect = lambda remote, local: Path(local).write_bytes(b"data")
        return sftp

    _LISTING = {
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

    def _run(self, tmp_path, orbit_prefilter=True, overlap_return=True):
        sftp = self._mock_sftp(self._LISTING)
        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=orbit_prefilter,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch(
                 "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                 return_value=overlap_return,
             ) as mock_overlap:
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-01",
            )
        return downloaded, mock_overlap

    def test_default_orbit_prefilter_is_enabled(self, tmp_path):
        dl = GPortalAMSR2Downloader(output_dir=tmp_path)
        assert dl.orbit_prefilter is True

    def test_dropped_files_are_excluded_from_download(self, tmp_path):
        downloaded, _ = self._run(tmp_path, orbit_prefilter=True, overlap_return=False)
        assert downloaded == []

    def test_orbit_prefilter_false_reproduces_todays_behavior(self, tmp_path):
        downloaded, mock_overlap = self._run(tmp_path, orbit_prefilter=False)
        assert not mock_overlap.called
        assert len(downloaded) == 1

    def test_orbit_overlaps_bbox_receives_whole_day_window(self, tmp_path):
        """AMSR2 filenames only embed a date, not a time -- the whole day
        must be used as the sensing window, not a degenerate single
        instant."""
        _downloaded, mock_overlap = self._run(tmp_path, orbit_prefilter=True, overlap_return=True)

        assert mock_overlap.call_count == 1
        satellite, start, end = mock_overlap.call_args[0][0:3]
        assert satellite == "gcom-w1"
        assert start == datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 7, 1, 23, 59, 59, tzinfo=timezone.utc)

    def test_tle_fetch_error_keeps_file_fail_open(self, tmp_path):
        """A mocked TleFetchError must keep the file (fail-open), matching
        H-SAF's equivalent test. This downloader's _filter_by_orbit_overlap
        uses `assert match is not None` on _FILENAME_DATE_RE -- a real
        invariant, since `matches` is pre-filtered upstream to only entries
        where that regex already matched -- so a parse-failure test (as
        used for H-SAF's satellite-name parse) isn't a valid failure
        surface here. Instead this exercises the *real*, unmocked
        orbit_overlaps_bbox by making the TLE fetch it depends on fail:
        patching get_tle (not orbit_overlaps_bbox itself) lets
        orbit_overlaps_bbox's own documented `except TleFetchError: return
        True` fail-open path run for real, which is the actual failure
        surface this downloader relies on."""
        sftp = self._mock_sftp(self._LISTING)
        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=True,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch(
                 "sar_validation.core.orbit_coverage.get_tle",
                 side_effect=TleFetchError("no TLE available"),
             ):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-01", end="2026-07-01",
            )

        assert len(downloaded) == 1

    def test_invalid_but_in_range_date_string_is_kept_not_crashed_on(self, tmp_path):
        """_FILENAME_DATE_RE is just an 8-digit run, not a real-calendar-
        date check -- "20260231" (Feb 31st doesn't exist) is
        lexicographically inside [start_date, end_date] and passes that
        string-comparison filter, but must not crash datetime.strptime
        and abort the whole download; it must be kept (fail-open),
        matching this module's philosophy elsewhere."""
        listing = {
            "standard": ["GCOM-W"],
            "standard/GCOM-W": ["GCOM-W.AMSR2"],
            "standard/GCOM-W/GCOM-W.AMSR2": ["L3.SM_STD"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD": ["2210"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210": ["2026"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026": ["02"],
            "standard/GCOM-W/GCOM-W.AMSR2/L3.SM_STD/2210/2026/02": [
                "GW1AM2_20260231_01D_EQMA_L3SGSMCLQ_2210.h5",
            ],
        }
        sftp = self._mock_sftp(listing)
        dl = GPortalAMSR2Downloader(
            output_dir=tmp_path, username="u", password="p", orbit_prefilter=True,
        )
        with patch("paramiko.Transport") as mock_transport_cls, \
             patch("paramiko.SFTPClient.from_transport", return_value=sftp), \
             patch("socket.create_connection", return_value=MagicMock()), \
             patch(
                 "sar_validation.core.orbit_coverage.orbit_overlaps_bbox",
                 side_effect=AssertionError("must not be called for an invalid date string"),
             ):
            mock_transport_cls.return_value = MagicMock()
            downloaded = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-02-01", end="2026-03-01",
            )

        assert len(downloaded) == 1
        assert "20260231" in downloaded[0].name
