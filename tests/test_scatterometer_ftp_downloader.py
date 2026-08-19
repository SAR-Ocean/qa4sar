"""Tests for ScatterometerFTPDownloader (HY-2B/HY-2C/Oceansat-3, 25 km, FTP)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from sar_validation.core.orbit_coverage import TleFetchError
from sar_validation.downloaders.scatterometer_ftp_downloader import (
    _ASSUMED_PASS_DURATION_BY_SATELLITE,
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

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path, orbit_prefilter=False)
        with patch("ftplib.FTP") as mock_ftp_cls, caplog.at_level(logging.WARNING):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        mock_ftp_cls.assert_not_called()
        assert any("days old" in r.message for r in caplog.records)

    def test_recent_end_date_proceeds_to_connect(self, tmp_path):
        now = datetime.now(timezone.utc)
        end = now.strftime("%Y-%m-%d")
        start = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path, orbit_prefilter=False)
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

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path, orbit_prefilter=False)
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
        dl = ScatterometerFTPDownloader(satellite="hy2b", output_dir=tmp_path, orbit_prefilter=False)
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

        dl = ScatterometerFTPDownloader(satellite=satellite, output_dir=tmp_path, orbit_prefilter=False)
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
            orbit_prefilter=False,
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
    def test_dry_run_without_credentials_prints_setup_message_without_network(
        self, tmp_path, capsys,
    ):
        """authenticate_osi_saf_ftp never prompts interactively -- it either
        resolves credentials or raises RuntimeError immediately -- so a
        dry-run with no credentials configured must print a setup message
        and never touch the FTP connection at all."""
        now = datetime.now(timezone.utc)
        end = now.strftime("%Y-%m-%d")
        start = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="hy2c", output_dir=tmp_path, dry_run=True)
        with patch("ftplib.FTP") as mock_ftp_cls, patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            side_effect=RuntimeError("OSI-SAF FTP credentials not found."),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        mock_ftp_cls.assert_not_called()
        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert "not configured" in captured
        assert "--set-credential osi_saf" in captured

    def test_dry_run_with_credentials_reports_real_matches_without_downloading(
        self, tmp_path, capsys,
    ):
        """Credentials ARE configured -- dry-run connects and lists real
        matching files, but never calls retrbinary()."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"
        end = now.strftime("%Y-%m-%d")
        start = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(
            satellite="oceansat3", output_dir=tmp_path, dry_run=True, orbit_prefilter=False,
        )
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name]
        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        fake_ftp.retrbinary.assert_not_called()
        assert not any(tmp_path.glob("*.nc"))
        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert name in captured


class TestScatterometerFTPDownloaderOrbitPrefilter:
    _OCEANSAT3_FILE = "oscat_20260718_043606_ocsat3_19234_o_250_4007_ovw_l2.nc"
    _HY2C_FILE = "hscat_20260718_050000_hy_2c__28900_o_250_4006_ovw_l2.nc.gz"

    @pytest.fixture(autouse=True)
    def _frozen_now(self):
        """The fixture filename/window dates in this class are hardcoded
        to 2026-07-18/19 to match the fixed embedded filename timestamp
        used in the padded-sensing-window assertion below. Freeze
        datetime.now() as seen by the downloader's recency guard so these
        tests keep exercising the orbit-filter logic regardless of the
        real wall-clock date the suite happens to run on (otherwise the
        3-day recency guard fires first once real time drifts past
        2026-07-22, short-circuiting before matches/orbit filtering ever
        run and making these tests pass/fail for the wrong reason)."""

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 7, 19, 12, 0, 0, tzinfo=tz)

        with patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.datetime",
            _FrozenDatetime,
        ):
            yield

    def test_default_orbit_prefilter_is_enabled(self, tmp_path):
        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
        assert dl.orbit_prefilter is True

    def test_dropped_files_are_excluded_from_download(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [self._OCEANSAT3_FILE]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox", return_value=False,
        ):
            dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-18", end="2026-07-19",
            )

        assert result == []

    def test_orbit_prefilter_false_reproduces_todays_behavior(self, tmp_path):
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [self._OCEANSAT3_FILE]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ), patch("sar_validation.core.orbit_coverage.orbit_overlaps_bbox") as mock_overlap:
            dl = ScatterometerFTPDownloader(
                satellite="oceansat3", output_dir=tmp_path, orbit_prefilter=False,
            )
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-18", end="2026-07-19",
            )

        assert not mock_overlap.called
        assert len(result) == 1

    def test_orbit_overlaps_bbox_receives_padded_sensing_window(self, tmp_path):
        """The single embedded timestamp must be padded into a
        [ts, ts+100min] window before being passed to orbit_overlaps_bbox
        -- not passed as a degenerate ts==ts window (which would silently
        disable filtering, since orbit_overlaps_bbox fails open on a
        single-sample window)."""
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [self._OCEANSAT3_FILE]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox", return_value=True,
        ) as mock_overlap:
            dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
            dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-18", end="2026-07-19",
            )

        assert mock_overlap.call_count == 1
        call_args = mock_overlap.call_args
        satellite, sensing_start, sensing_end = call_args[0][0], call_args[0][1], call_args[0][2]
        assert satellite == "oceansat3"
        assert sensing_start == datetime(2026, 7, 18, 4, 36, 6, tzinfo=timezone.utc)
        assert sensing_end == sensing_start + _ASSUMED_PASS_DURATION_BY_SATELLITE["oceansat3"]

    def test_assumed_pass_duration_covers_real_measured_cadence(self):
        """Real per-file cadence measured from live-downloaded files
        (revolution-counter deltas): HY-2B/HY-2C ~104.4 min/file,
        Oceansat-3 ~49.8 min/file (two half-orbit files per revolution).
        The assumed padding must exceed these real spans with a safety
        margin, or the orbit pre-filter risks a false negative -- this
        is what the original 100-minute-for-everyone bug got wrong."""
        assert _ASSUMED_PASS_DURATION_BY_SATELLITE["hy2b"] >= timedelta(minutes=105)
        assert _ASSUMED_PASS_DURATION_BY_SATELLITE["hy2c"] >= timedelta(minutes=105)
        assert _ASSUMED_PASS_DURATION_BY_SATELLITE["oceansat3"] >= timedelta(minutes=50)
        # Oceansat-3's real per-file span (~49.8 min) is roughly half of
        # HY-2B/HY-2C's (~104.4 min) because it ships two half-orbit files
        # per revolution instead of one -- the assumed durations should
        # reflect that, not use a single one-size-fits-all value.
        assert _ASSUMED_PASS_DURATION_BY_SATELLITE["oceansat3"] < _ASSUMED_PASS_DURATION_BY_SATELLITE["hy2b"]

    def test_tle_fetch_error_keeps_file_fail_open(self, tmp_path):
        """A mocked TleFetchError must keep the file (fail-open), matching
        H-SAF's equivalent test. This downloader has no per-file
        satellite-name parse step to fail (self.satellite is fixed at
        construction, validated in __init__), and its single embedded
        timestamp is shared between the date-window match (in download())
        and the orbit-window computation (in _filter_by_orbit_overlap) --
        an entry only reaches the orbit filter once that timestamp has
        already parsed successfully, so a parse-failure test isn't a valid
        failure surface here (mirroring GPortalAMSR2Downloader's analogous
        `assert match is not None` invariant). Instead this exercises the
        *real*, unmocked orbit_overlaps_bbox by making the TLE fetch it
        depends on fail: patching get_tle (not orbit_overlaps_bbox itself)
        lets orbit_overlaps_bbox's own documented `except TleFetchError:
        return True` fail-open path run for real."""
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [self._OCEANSAT3_FILE]
        fake_ftp.retrbinary.side_effect = lambda cmd, cb: cb(b"fake nc bytes")

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ), patch(
            "sar_validation.core.orbit_coverage.get_tle",
            side_effect=TleFetchError("no TLE available"),
        ):
            dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path)
            result = dl.download(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start="2026-07-18", end="2026-07-19",
            )

        assert len(result) == 1


class TestListCandidatesDry:
    def test_returns_matches_without_fetching(self, tmp_path):
        """Mirrors download()'s own FTP-listing/matching logic exactly, but
        never touches self.dry_run and never calls retrbinary. sensing_end
        is estimated from the satellite's assumed per-file pass duration,
        since each filename embeds only a single timestamp."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = f"oscat_{ts}_ocsat3_1_o_250_4007_ovw_l2.nc"

        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = [name, name + ".md5"]

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path, orbit_prefilter=False)
            start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            end = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start=start, end=end,
            )

        assert not fake_ftp.retrbinary.called
        assert len(candidates) == 1
        cand_name, sensing_start, sensing_end = candidates[0]
        assert cand_name == name
        expected_start = datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        assert sensing_start == expected_start
        assert sensing_end == expected_start + _ASSUMED_PASS_DURATION_BY_SATELLITE["oceansat3"]

    def test_no_matches_returns_empty(self, tmp_path):
        now = datetime.now(timezone.utc)
        fake_ftp = MagicMock()
        fake_ftp.nlst.return_value = ["readme.txt"]

        with patch("ftplib.FTP", return_value=fake_ftp), patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.authenticate_osi_saf_ftp",
            return_value=("user", "pass"),
        ):
            dl = ScatterometerFTPDownloader(satellite="hy2b", output_dir=tmp_path, orbit_prefilter=False)
            start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            end = now.strftime("%Y-%m-%dT%H:%M:%S")
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start=start, end=end,
            )

        assert candidates == []

    def test_stale_window_returns_empty_without_touching_network(self, tmp_path):
        """Mirrors download()'s own recency guard: a window entirely older
        than the FTP server's rolling retention must short-circuit before
        any FTP connection is attempted."""
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=4)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=5)).strftime("%Y-%m-%d")

        dl = ScatterometerFTPDownloader(satellite="oceansat3", output_dir=tmp_path, orbit_prefilter=False)
        with patch("ftplib.FTP") as mock_ftp_cls:
            candidates = dl.list_candidates_dry(
                min_lon=_MIN_LON, max_lon=_MAX_LON, min_lat=_MIN_LAT, max_lat=_MAX_LAT,
                start=start, end=end,
            )

        assert candidates == []
        mock_ftp_cls.assert_not_called()
