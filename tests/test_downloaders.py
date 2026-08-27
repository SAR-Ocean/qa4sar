"""Tests for downloader utilities: datetime parsing and dataset_part selection."""

from __future__ import annotations

import logging
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import keyring
import keyring.errors
import pytest

from sar_validation.downloaders.base import (
    copernicus_marine_download_kwargs,
    is_date_recent,
    normalize_datetime,
    prefer_ipv4_dns,
    set_credential,
    split_antimeridian_bbox,
)
from sar_validation.downloaders.insitu_downloader import (
    _resolve_platform_codes,
)

# ---------------------------------------------------------------------------
# Tests for normalize_datetime()
# ---------------------------------------------------------------------------

class TestNormalizeDatetime:
    """Test datetime normalization with various input formats."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            pytest.param("2026-01-01", "2026-01-01T00:00:00", id="date_only"),
            pytest.param("2026-01-01T12:34:56", "2026-01-01T12:34:56", id="date_with_iso_time"),
            pytest.param("2026-01-01 12:34:56", "2026-01-01T12:34:56", id="date_with_space_separator"),
            pytest.param("2026-06-24 000000", "2026-06-24T00:00:00", id="hhmmss_format_no_colons"),
            pytest.param("2026-06-24 120000", "2026-06-24T12:00:00", id="hhmmss_format_midday"),
            pytest.param("2026-06-24 235959", "2026-06-24T23:59:59", id="hhmmss_format_near_end_of_day"),
            pytest.param("2026-06-24 030000", "2026-06-24T03:00:00", id="hhmmss_with_three_hour_offset"),
            pytest.param("2026-01-01T12:34:56Z", "2026-01-01T12:34:56", id="trailing_z_removed"),
            pytest.param("2026-01-01T12:34:56.123Z", "2026-01-01T12:34:56", id="milliseconds_removed"),
            pytest.param("  2026-01-01  ", "2026-01-01T00:00:00", id="whitespace_stripped"),
            pytest.param("2026-01-01 120000", "2026-01-01T12:00:00", id="space_and_hhmmss"),
            pytest.param("2026-01-01T120000Z", "2026-01-01T12:00:00", id="iso_hhmmss_with_z"),
            # Regression: a tz-aware datetime's own .isoformat() (e.g. a
            # dry-collocation SarFootprint's sensing_start/sensing_end,
            # always aware) renders as "...+00:00", never "Z" -- the
            # offset must be stripped the same way "Z" already is, or
            # downstream fromisoformat(normalize_datetime(...)) callers
            # that compare against a naive "now" raise "can't compare
            # offset-naive and offset-aware datetimes". Found live against
            # noaa_hfradar_downloader.select_backend().
            pytest.param(
                "2026-01-01T12:34:56+00:00", "2026-01-01T12:34:56", id="utc_offset_removed",
            ),
            pytest.param(
                "2026-01-01T12:34:56.123456+00:00", "2026-01-01T12:34:56",
                id="utc_offset_and_microseconds_removed",
            ),
            pytest.param(
                "2026-01-01T12:34:56-05:00", "2026-01-01T12:34:56", id="negative_offset_removed",
            ),
        ],
    )
    def test_normalizes_various_formats(self, raw, expected):
        assert normalize_datetime(raw) == expected


# ---------------------------------------------------------------------------
# Tests for is_date_recent()
# ---------------------------------------------------------------------------

class TestIsDateRecent:
    """Test detection of recent dates."""

    @pytest.mark.parametrize(
        "raw_date,threshold_days,expected",
        [
            pytest.param("2026-07-02", 30, True, id="today_is_recent"),
            pytest.param("2026-07-01", 30, True, id="yesterday_is_recent"),
            pytest.param("2026-06-02", 30, True, id="30_days_ago_is_recent"),
            pytest.param("2026-06-01", 30, False, id="31_days_ago_is_not_recent"),
            pytest.param("2026-03-15", 30, False, id="old_date_is_not_recent"),
            pytest.param("2026-05-03", 30, False, id="custom_threshold_30_days_not_recent"),
            pytest.param("2026-05-03", 60, True, id="custom_threshold_60_days_recent"),
        ],
    )
    @patch("sar_validation.downloaders.base.datetime")
    def test_is_date_recent_relative_to_mocked_today(
        self, mock_datetime, raw_date, threshold_days, expected
    ):
        """`datetime.now()` is mocked to a fixed "today" (2026-07-02); each
        row exercises is_date_recent's day-difference-vs-threshold compare."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat

        assert is_date_recent(raw_date, threshold_days=threshold_days) is expected

    def test_with_hhmmss_format_parses_correctly(self):
        """HHMMSS format should parse correctly through normalize_datetime."""
        # This test verifies that normalize_datetime handles HHMMSS correctly,
        # not the date recency comparison (which requires mocking datetime.now)
        normalized = normalize_datetime("2026-07-02 120000")
        assert normalized == "2026-07-02T12:00:00"

    def test_invalid_datetime_returns_false(self):
        """Invalid datetime should return False gracefully."""
        result = is_date_recent("not-a-date", threshold_days=30)
        assert result is False


# ---------------------------------------------------------------------------
# Integration tests for datetime parsing scenarios
# ---------------------------------------------------------------------------

class TestDatetimeIntegration:
    """Integration tests for realistic datetime parsing scenarios."""

    def test_original_error_case(self):
        """The original error case from the bug report should work."""
        # User input from CLI
        start = "2026-06-24 000000"
        end = "2026-06-24 030000"
        
        start_norm = normalize_datetime(start)
        end_norm = normalize_datetime(end)
        
        # Should produce valid ISO format
        assert start_norm == "2026-06-24T00:00:00"
        assert end_norm == "2026-06-24T03:00:00"
        
        # Should be suitable for API URL construction
        url_start = start_norm + ".000Z"
        url_end = end_norm + ".000Z"
        
        assert url_start == "2026-06-24T00:00:00.000Z"
        assert url_end == "2026-06-24T03:00:00.000Z"

    def test_mixed_datetime_formats_in_recipe(self):
        """A recipe might have different datetime formats for start/end."""
        # Date-only format
        start = normalize_datetime("2026-01-01")
        # HHMMSS format
        end = normalize_datetime("2026-01-02 235959")
        
        assert start == "2026-01-01T00:00:00"
        assert end == "2026-01-02T23:59:59"

    def test_api_url_construction_flow(self):
        """Simulate the full flow from user input to API URL."""
        # User provides input with HHMMSS format
        user_start = "2026-06-24 000000"
        user_end = "2026-06-24 030000"

        # SAR downloader normalizes and appends .000Z
        norm_start = normalize_datetime(user_start)
        norm_end = normalize_datetime(user_end)

        api_start = norm_start + ".000Z"
        api_end = norm_end + ".000Z"

        # These should be valid for Copernicus API
        assert api_start == "2026-06-24T00:00:00.000Z"
        assert api_end == "2026-06-24T03:00:00.000Z"

        # Verify they parse as valid ISO datetime
        datetime.fromisoformat(api_start.rstrip("Z"))
        datetime.fromisoformat(api_end.rstrip("Z"))


# ---------------------------------------------------------------------------
# Tests for split_antimeridian_bbox()
# ---------------------------------------------------------------------------

class TestSplitAntimeridianBbox:
    def test_non_crossing_bbox_returned_unchanged(self):
        assert split_antimeridian_bbox(-20.0, 0.0) == [(-20.0, 0.0)]

    def test_equal_bounds_treated_as_non_crossing(self):
        assert split_antimeridian_bbox(10.0, 10.0) == [(10.0, 10.0)]

    def test_crossing_bbox_splits_into_two_windows(self):
        assert split_antimeridian_bbox(135.0, -120.0) == [(135.0, 180.0), (-180.0, -120.0)]

    def test_crossing_bbox_windows_are_each_non_crossing(self):
        windows = split_antimeridian_bbox(170.0, -170.0)
        for lo, hi in windows:
            assert lo <= hi


# ---------------------------------------------------------------------------
# Tests for prefer_ipv4_dns()
# ---------------------------------------------------------------------------

class TestPreferIpv4Dns:
    """Regression coverage for the IPv6-black-hole fix: some hosts (observed
    live for www.ncei.noaa.gov) have IPv6 addresses that silently never
    connect while their IPv4 addresses connect in ~0.1-0.2s.
    socket.create_connection() tries getaddrinfo() results in the order
    returned (IPv6-first by default on this system), wasting up to
    ``6 * timeout`` seconds per request. prefer_ipv4_dns() reorders results
    IPv4-first for the duration of a ``with`` block."""

    def test_reorders_ipv4_before_ipv6_preserving_relative_order(self, monkeypatch):
        fake_results = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::2", 443, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::3", 443, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.2", 443)),
        ]

        def fake_getaddrinfo(*args, **kwargs):
            return fake_results

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

        with prefer_ipv4_dns():
            result = socket.getaddrinfo("example.com", 443)

        families = [r[0] for r in result]
        assert families == [
            socket.AF_INET, socket.AF_INET,
            socket.AF_INET6, socket.AF_INET6, socket.AF_INET6,
        ]
        ipv4_addrs = [r[4][0] for r in result if r[0] == socket.AF_INET]
        assert ipv4_addrs == ["192.0.2.1", "192.0.2.2"]  # relative order preserved
        ipv6_addrs = [r[4][0] for r in result if r[0] == socket.AF_INET6]
        assert ipv6_addrs == ["2001:db8::1", "2001:db8::2", "2001:db8::3"]  # ditto

    def test_restores_original_getaddrinfo_after_normal_exit(self):
        original_getaddrinfo = socket.getaddrinfo
        with prefer_ipv4_dns():
            assert socket.getaddrinfo is not original_getaddrinfo
        assert socket.getaddrinfo is original_getaddrinfo

    def test_restores_original_getaddrinfo_after_exception(self):
        original_getaddrinfo = socket.getaddrinfo
        with pytest.raises(RuntimeError):
            with prefer_ipv4_dns():
                raise RuntimeError("boom")
        assert socket.getaddrinfo is original_getaddrinfo


# ---------------------------------------------------------------------------
# Tests for copernicus_marine_download_kwargs()
# ---------------------------------------------------------------------------

class TestCopernicusMarineDownloadKwargs:
    def test_default_skips_existing_files(self):
        assert copernicus_marine_download_kwargs(force_download=False) == {
            "skip_existing": True, "overwrite": False,
        }

    def test_force_download_overwrites(self):
        assert copernicus_marine_download_kwargs(force_download=True) == {
            "skip_existing": False, "overwrite": True,
        }


# ---------------------------------------------------------------------------
# Fake keyring backend — tests must never touch a real OS keyring
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_keyring(monkeypatch):
    """Replace keyring.get_password/set_password with an in-memory dict.

    Every authenticate_*/set_credential test that could reach the keyring
    layer uses this fixture so the suite passes identically whether or not
    a real OS keyring backend (GNOME Keyring/libsecret, etc.) is present --
    critical for headless CI runners.
    """
    store: dict[tuple[str, str], str] = {}

    def _get_password(service, key):
        return store.get((service, key))

    def _set_password(service, key, value):
        store[(service, key)] = value

    monkeypatch.setattr(keyring, "get_password", _get_password)
    monkeypatch.setattr(keyring, "set_password", _set_password)
    return store


@pytest.fixture
def broken_keyring(monkeypatch):
    """Simulate "no OS keyring backend available" (e.g. headless CI):
    keyring.get_password/set_password both raise NoKeyringError."""

    def _raise(*args, **kwargs):
        raise keyring.errors.NoKeyringError("no recommended backend available")

    monkeypatch.setattr(keyring, "get_password", _raise)
    monkeypatch.setattr(keyring, "set_password", _raise)


# ---------------------------------------------------------------------------
# Tests for authenticate_eumdac()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests for authenticate_eumdac() / authenticate_osi_saf_ftp() /
# authenticate_gportal() / authenticate_smos_ftp() — shared resolution order
# ---------------------------------------------------------------------------

def _call_eumdac(username=None, password=None, allow_prompt=True):
    """authenticate_eumdac() needs a fake `eumdac` module patched into
    sys.modules for the duration of the call; the returned "token" is just
    the (username, password) tuple, per _fake_eumdac_module's side_effect."""
    from sar_validation.downloaders.base import authenticate_eumdac

    fake_eumdac = MagicMock()
    fake_eumdac.AccessToken.side_effect = lambda creds: creds
    with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
        return authenticate_eumdac(username, password)


def _call_osi_saf_ftp(username=None, password=None, allow_prompt=True):
    from sar_validation.downloaders.base import authenticate_osi_saf_ftp

    return authenticate_osi_saf_ftp(username, password)


def _call_hsaf_ftp(username=None, password=None, allow_prompt=True):
    from sar_validation.downloaders.base import authenticate_hsaf_ftp

    return authenticate_hsaf_ftp(username, password)


def _call_space_track(username=None, password=None, allow_prompt=True):
    from sar_validation.downloaders.base import authenticate_space_track

    return authenticate_space_track(username, password)


def _call_gportal(username=None, password=None, allow_prompt=True):
    from sar_validation.downloaders.base import authenticate_gportal

    return authenticate_gportal(username, password, allow_prompt=allow_prompt)


def _call_smos_ftp(username=None, password=None, allow_prompt=True):
    from sar_validation.downloaders.base import authenticate_smos_ftp

    return authenticate_smos_ftp(username, password)


def _json_legacy_content(username, password):
    import json

    return json.dumps({"username": username, "password": password})


def _comma_separated_legacy_content(username, password):
    return f"{username},{password}"


# (name, call, env_user_var, env_pass_var, keyring_service, legacy_relpath,
#  legacy_content_fn, no_resolution_match, resolves_via_prompt)
CREDENTIAL_RESOLUTION_CASES = [
    pytest.param(
        "eumdac", _call_eumdac,
        "EUMDAC_USERNAME", "EUMDAC_PASSWORD", "sar-validation-eumdac",
        ".eumdac/credentials", _comma_separated_legacy_content,
        "EUMDAC credentials not found", False,
        id="eumdac",
    ),
    pytest.param(
        "osi_saf_ftp", _call_osi_saf_ftp,
        "OSI_SAF_FTP_USERNAME", "OSI_SAF_FTP_PASSWORD", "sar-validation-osi-saf",
        ".eumetsat_osi_saf_wind_credentials", _json_legacy_content,
        "OSI-SAF FTP credentials not found", False,
        id="osi_saf_ftp",
    ),
    pytest.param(
        "hsaf_ftp", _call_hsaf_ftp,
        "HSAF_FTP_USERNAME", "HSAF_FTP_PASSWORD", "sar-validation-hsaf",
        ".hsaf_ftp_credentials", _json_legacy_content,
        "H-SAF FTP credentials not found", False,
        id="hsaf_ftp",
    ),
    pytest.param(
        "space_track", _call_space_track,
        "SPACE_TRACK_USERNAME", "SPACE_TRACK_PASSWORD", "sar-validation-space-track",
        ".space_track_credentials", _json_legacy_content,
        "Space-Track credentials not found", False,
        id="space_track",
    ),
    pytest.param(
        "gportal", _call_gportal,
        "GPORTAL_USERNAME", "GPORTAL_PASSWORD", "sar-validation-gportal",
        ".jaxa_gportal_credentials", _json_legacy_content,
        "G-Portal credentials not found", True,
        id="gportal",
    ),
    pytest.param(
        "smos_ftp", _call_smos_ftp,
        "SMOS_FTP_USERNAME", "SMOS_FTP_PASSWORD", "sar-validation-smos",
        ".esa_smos_credentials", _json_legacy_content,
        "SMOS", False,
        id="smos_ftp",
    ),
]

# Subset that also has a distinct "empty (working) keyring, no legacy file"
# test alongside the "broken keyring backend" test below — eumdac only has
# the broken-keyring variant.
CREDENTIAL_RESOLUTION_CASES_WITH_EMPTY_KEYRING_VARIANT = [
    c for c in CREDENTIAL_RESOLUTION_CASES if c.id != "eumdac"
]

_CREDENTIAL_RESOLUTION_PARAMS = (
    "name,call,env_user_var,env_pass_var,keyring_service,legacy_relpath,"
    "legacy_content_fn,no_resolution_match,resolves_via_prompt"
)


class TestAuthenticateCredentialResolution:
    """Shared explicit-args / env / keyring / legacy-file / no-resolution
    behavior for the four authenticate_* helpers (eumdac, osi_saf_ftp,
    gportal, smos_ftp) that all share one resolution order. G-Portal's two
    behaviors unique to it (prompting instead of raising, and never
    persisting a prompted credential) are covered separately below."""

    @pytest.mark.parametrize(_CREDENTIAL_RESOLUTION_PARAMS, CREDENTIAL_RESOLUTION_CASES)
    def test_explicit_args_win_over_everything(
        self, name, call, env_user_var, env_pass_var, keyring_service,
        legacy_relpath, legacy_content_fn, no_resolution_match, resolves_via_prompt,
        monkeypatch, tmp_path, fake_keyring,
    ):
        monkeypatch.setenv(env_user_var, "env_user")
        monkeypatch.setenv(env_pass_var, "env_pass")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_keyring[(keyring_service, "username")] = "kr_user"
        fake_keyring[(keyring_service, "password")] = "kr_pass"

        result = call(username="explicit_user", password="explicit_pass")
        assert result == ("explicit_user", "explicit_pass")

    @pytest.mark.parametrize(_CREDENTIAL_RESOLUTION_PARAMS, CREDENTIAL_RESOLUTION_CASES)
    def test_env_vars_used_when_args_absent(
        self, name, call, env_user_var, env_pass_var, keyring_service,
        legacy_relpath, legacy_content_fn, no_resolution_match, resolves_via_prompt,
        monkeypatch, tmp_path, fake_keyring,
    ):
        monkeypatch.setenv(env_user_var, "env_user")
        monkeypatch.setenv(env_pass_var, "env_pass")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_keyring[(keyring_service, "username")] = "kr_user"
        fake_keyring[(keyring_service, "password")] = "kr_pass"

        result = call()
        assert result == ("env_user", "env_pass")

    @pytest.mark.parametrize(_CREDENTIAL_RESOLUTION_PARAMS, CREDENTIAL_RESOLUTION_CASES)
    def test_uses_keyring_when_no_args_or_env(
        self, name, call, env_user_var, env_pass_var, keyring_service,
        legacy_relpath, legacy_content_fn, no_resolution_match, resolves_via_prompt,
        monkeypatch, tmp_path, fake_keyring,
    ):
        monkeypatch.delenv(env_user_var, raising=False)
        monkeypatch.delenv(env_pass_var, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_keyring[(keyring_service, "username")] = "kr_user"
        fake_keyring[(keyring_service, "password")] = "kr_pass"

        result = call()
        assert result == ("kr_user", "kr_pass")

    @pytest.mark.parametrize(_CREDENTIAL_RESOLUTION_PARAMS, CREDENTIAL_RESOLUTION_CASES)
    def test_falls_back_to_legacy_file_and_migrates_to_keyring(
        self, name, call, env_user_var, env_pass_var, keyring_service,
        legacy_relpath, legacy_content_fn, no_resolution_match, resolves_via_prompt,
        monkeypatch, tmp_path, fake_keyring,
    ):
        monkeypatch.delenv(env_user_var, raising=False)
        monkeypatch.delenv(env_pass_var, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cred_file = tmp_path / legacy_relpath
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(legacy_content_fn("file_user", "file_pass"))

        result = call()
        assert result == ("file_user", "file_pass")
        # One-time migration: the legacy values are now stored in the keyring.
        assert fake_keyring[(keyring_service, "username")] == "file_user"
        assert fake_keyring[(keyring_service, "password")] == "file_pass"

    @pytest.mark.parametrize(_CREDENTIAL_RESOLUTION_PARAMS, CREDENTIAL_RESOLUTION_CASES)
    def test_no_keyring_backend_and_no_legacy_file(
        self, name, call, env_user_var, env_pass_var, keyring_service,
        legacy_relpath, legacy_content_fn, no_resolution_match, resolves_via_prompt,
        monkeypatch, tmp_path, broken_keyring,
    ):
        monkeypatch.delenv(env_user_var, raising=False)
        monkeypatch.delenv(env_pass_var, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        if resolves_via_prompt:
            monkeypatch.setattr("builtins.input", lambda prompt: "prompted_user")
            monkeypatch.setattr("getpass.getpass", lambda prompt: "prompted_pass")
            result = call()
            assert result == ("prompted_user", "prompted_pass")
        else:
            with pytest.raises(RuntimeError, match=no_resolution_match):
                call()

    @pytest.mark.parametrize(
        _CREDENTIAL_RESOLUTION_PARAMS, CREDENTIAL_RESOLUTION_CASES_WITH_EMPTY_KEYRING_VARIANT,
    )
    def test_empty_working_keyring_and_no_legacy_file(
        self, name, call, env_user_var, env_pass_var, keyring_service,
        legacy_relpath, legacy_content_fn, no_resolution_match, resolves_via_prompt,
        monkeypatch, tmp_path, fake_keyring,
    ):
        """Distinct from test_no_keyring_backend_and_no_legacy_file above:
        here the keyring backend works fine but has nothing stored yet,
        rather than being unavailable entirely -- both must reach the same
        "nothing resolved" behavior."""
        monkeypatch.delenv(env_user_var, raising=False)
        monkeypatch.delenv(env_pass_var, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        if resolves_via_prompt:
            monkeypatch.setattr("builtins.input", lambda prompt: "prompted_user")
            monkeypatch.setattr("getpass.getpass", lambda prompt: "prompted_pass")
            result = call()
            assert result == ("prompted_user", "prompted_pass")
        else:
            with pytest.raises(RuntimeError, match=no_resolution_match):
                call()


class TestAuthenticateGportalPromptOnlyBehavior:
    """Behavior unique to G-Portal among the four authenticate_* helpers
    above: it prompts instead of raising, and (unlike every migrate-to-
    keyring path above) a prompted credential must never be persisted."""

    def test_interactive_prompt_not_persisted_to_credentials_file(self, monkeypatch, tmp_path):
        """Deliberate deviation from every other authenticate_* helper:
        entered credentials must never be written to disk, per design."""
        from sar_validation.downloaders.base import authenticate_gportal

        monkeypatch.delenv("GPORTAL_USERNAME", raising=False)
        monkeypatch.delenv("GPORTAL_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("builtins.input", lambda prompt: "prompted_user")
        monkeypatch.setattr("getpass.getpass", lambda prompt: "prompted_pass")

        authenticate_gportal()
        assert not (tmp_path / ".jaxa_gportal_credentials").exists()

    def test_allow_prompt_false_raises_instead_of_prompting(self, monkeypatch, tmp_path, fake_keyring):
        """allow_prompt=False (used by the orchestrator's automatic G-Portal
        AMSR2 fallback) must raise RuntimeError instead of reaching the
        interactive input()/getpass.getpass() prompt when nothing resolves
        from explicit args/env vars/credentials file -- an unattended
        pipeline run must never block on a password prompt nobody expected
        to be asked for."""
        from sar_validation.downloaders.base import authenticate_gportal

        monkeypatch.delenv("GPORTAL_USERNAME", raising=False)
        monkeypatch.delenv("GPORTAL_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("input()/getpass.getpass() must not be called when allow_prompt=False")

        monkeypatch.setattr("builtins.input", _fail_if_called)
        monkeypatch.setattr("getpass.getpass", _fail_if_called)

        with pytest.raises(RuntimeError, match="G-Portal credentials not found"):
            authenticate_gportal(allow_prompt=False)


# ---------------------------------------------------------------------------
# Tests for authenticate_earthdata()
# ---------------------------------------------------------------------------

class TestAuthenticateEarthdata:
    def _fake_earthaccess_module(self):
        return MagicMock()

    def test_explicit_args_win_over_everything(self, monkeypatch, tmp_path, fake_keyring):
        from sar_validation.downloaders.base import authenticate_earthdata

        monkeypatch.setenv("EARTHDATA_USERNAME", "env_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "env_pass")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_keyring[("sar-validation-earthdata", "username")] = "kr_user"
        fake_keyring[("sar-validation-earthdata", "password")] = "kr_pass"
        fake_earthaccess = self._fake_earthaccess_module()

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            authenticate_earthdata("explicit_user", "explicit_pass")

        assert os.environ["EARTHDATA_USERNAME"] == "explicit_user"
        assert os.environ["EARTHDATA_PASSWORD"] == "explicit_pass"
        fake_earthaccess.login.assert_called_once_with(strategy="environment")

    def test_env_vars_used_when_args_absent(self, monkeypatch, tmp_path, fake_keyring):
        from sar_validation.downloaders.base import authenticate_earthdata

        monkeypatch.setenv("EARTHDATA_USERNAME", "env_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "env_pass")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_earthaccess = self._fake_earthaccess_module()

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            authenticate_earthdata()

        fake_earthaccess.login.assert_called_once_with(strategy="environment")

    def test_uses_keyring_when_no_args_or_env(self, monkeypatch, tmp_path, fake_keyring):
        from sar_validation.downloaders.base import authenticate_earthdata

        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_keyring[("sar-validation-earthdata", "username")] = "kr_user"
        fake_keyring[("sar-validation-earthdata", "password")] = "kr_pass"
        fake_earthaccess = self._fake_earthaccess_module()

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            authenticate_earthdata()

        assert os.environ["EARTHDATA_USERNAME"] == "kr_user"
        assert os.environ["EARTHDATA_PASSWORD"] == "kr_pass"
        fake_earthaccess.login.assert_called_once_with(strategy="environment")

    def test_falls_back_to_legacy_netrc_and_migrates_to_keyring(self, monkeypatch, tmp_path, fake_keyring):
        from sar_validation.downloaders.base import authenticate_earthdata

        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / ".netrc").write_text(
            "machine urs.earthdata.nasa.gov\nlogin file_user\npassword file_pass\n"
        )
        fake_earthaccess = self._fake_earthaccess_module()

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            authenticate_earthdata()

        assert os.environ["EARTHDATA_USERNAME"] == "file_user"
        assert os.environ["EARTHDATA_PASSWORD"] == "file_pass"
        assert fake_keyring[("sar-validation-earthdata", "username")] == "file_user"
        assert fake_keyring[("sar-validation-earthdata", "password")] == "file_pass"

    def test_nothing_resolves_falls_back_to_bare_earthaccess_login(self, monkeypatch, tmp_path, fake_keyring):
        """No explicit args, no env vars, no keyring, no ~/.netrc -- must
        not raise; falls back to earthaccess's own login() (netrc/
        interactive-prompt resolution), preserving today's behaviour."""
        from sar_validation.downloaders.base import authenticate_earthdata

        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fake_earthaccess = self._fake_earthaccess_module()

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            authenticate_earthdata()

        fake_earthaccess.login.assert_called_once_with()


# ---------------------------------------------------------------------------
# Tests for set_credential()
# ---------------------------------------------------------------------------

class TestSetCredential:
    def test_stores_username_and_password_in_keyring(self, fake_keyring):
        set_credential("eumdac", "alice", "secret")
        assert fake_keyring[("sar-validation-eumdac", "username")] == "alice"
        assert fake_keyring[("sar-validation-eumdac", "password")] == "secret"

    @pytest.mark.parametrize(
        "name,service",
        [
            ("eumdac", "sar-validation-eumdac"),
            ("osi_saf", "sar-validation-osi-saf"),
            ("gportal", "sar-validation-gportal"),
            ("smos", "sar-validation-smos"),
            ("earthdata", "sar-validation-earthdata"),
        ],
    )
    def test_uses_the_expected_service_name_per_credential_set(
        self, fake_keyring, name, service
    ):
        set_credential(name, "user", "pass")
        assert fake_keyring[(service, "username")] == "user"
        assert fake_keyring[(service, "password")] == "pass"

    def test_unknown_name_raises_value_error(self, fake_keyring):
        with pytest.raises(ValueError, match="Unknown credential"):
            set_credential("not_a_real_service", "user", "pass")

    def test_propagates_keyring_errors_to_the_caller(self, broken_keyring):
        """--set-credential is an explicit user action -- if the OS keyring
        backend is unavailable, the CLI needs a real exception to report,
        not a silent no-op."""
        with pytest.raises(keyring.errors.KeyringError):
            set_credential("eumdac", "user", "pass")


# ---------------------------------------------------------------------------
# SMOSDownloader
# ---------------------------------------------------------------------------

class TestSmosParseSensingWindow:
    def test_parses_real_naming_convention(self):
        """Filename comes from a real one downloaded file from smos-diss.eo.esa.int."""
        from sar_validation.downloaders.smos_downloader import _parse_sensing_window

        result = _parse_sensing_window(
            "W_XX-ESA,SMOS,NRTNN_C_LEMM_20260102131619_20260102103700_20260102123603_o_v300_l2sm.nc"
        )
        assert result == (
            datetime(2026, 1, 2, 10, 37, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 36, 3, tzinfo=timezone.utc),
        )

    def test_unparseable_filename_returns_none(self):
        from sar_validation.downloaders.smos_downloader import _parse_sensing_window

        assert _parse_sensing_window("SM_1.nc") is None


class TestSMOSDownloader:
    def test_dry_run_prints_params_without_network(self, tmp_path, capsys):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=True, username="u", password="p")
        out = dl.download(
            min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
            start="2026-07-01", end="2026-07-02",
        )
        assert out == []
        captured = capsys.readouterr().out
        assert "DRY RUN" in captured
        assert "smos-diss.eo.esa.int" in captured

    def test_download_logs_in_lists_and_fetches_products(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(
            output_dir=tmp_path, dry_run=False, username="u", password="p", orbit_prefilter=False,
        )
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {"filename": "SM_1.nc", "download_href": "/oads/access/login?r=x&d=SM_1.nc"},
        ])

        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(
            status_code=200, content=b"fake-netcdf-bytes",
        )
        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        dl._login.assert_called_once_with(fake_session, "u", "p")
        assert result == [tmp_path / "SM_1.nc"]
        assert (tmp_path / "SM_1.nc").read_bytes() == b"fake-netcdf-bytes"

    def test_already_downloaded_file_is_skipped(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        (tmp_path / "SM_1.nc").write_bytes(b"already here")
        dl = SMOSDownloader(
            output_dir=tmp_path, dry_run=False, username="u", password="p", orbit_prefilter=False,
        )
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {"filename": "SM_1.nc", "download_href": "/oads/access/login?r=x&d=SM_1.nc"},
        ])
        fake_session = MagicMock()

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        fake_session.get.assert_not_called()
        assert result == [tmp_path / "SM_1.nc"]

    def test_only_nc_and_tgz_filenames_are_downloaded(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(
            output_dir=tmp_path, dry_run=False, username="u", password="p", orbit_prefilter=False,
        )
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {"filename": "SM_1.nc", "download_href": "/oads/access/login?r=x&d=SM_1.nc"},
            {"filename": "readme.txt", "download_href": "/oads/access/login?r=x&d=readme.txt"},
        ])
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(status_code=200, content=b"data")

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        assert result == [tmp_path / "SM_1.nc"]


class TestSMOSDownloaderOrbitPrefilter:
    def test_default_orbit_prefilter_is_enabled(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, username="u", password="p")
        assert dl.orbit_prefilter is True

    def test_dropped_files_are_excluded_from_download(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {
                "filename": "W_XX-ESA,SMOS,NRTNN_C_LEMM_20260102131619_20260102103700_20260102123603_o_v300_l2sm.nc",
                "download_href": "/oads/access/login?r=x&d=SM_1.nc",
            },
        ])
        fake_session = MagicMock()

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox", return_value=False,
        ):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        assert result == []
        fake_session.get.assert_not_called()

    def test_orbit_prefilter_false_reproduces_todays_behavior(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(
            output_dir=tmp_path, dry_run=False, username="u", password="p", orbit_prefilter=False,
        )
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {
                "filename": "W_XX-ESA,SMOS,NRTNN_C_LEMM_20260102131619_20260102103700_20260102123603_o_v300_l2sm.nc",
                "download_href": "/oads/access/login?r=x&d=SM_1.nc",
            },
        ])
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(status_code=200, content=b"data")

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ), patch("sar_validation.core.orbit_coverage.orbit_overlaps_bbox") as mock_overlap:
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        assert not mock_overlap.called
        assert len(result) == 1

    def test_real_start_stop_used_when_filename_matches(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {
                "filename": "W_XX-ESA,SMOS,NRTNN_C_LEMM_20260102131619_20260102103700_20260102123603_o_v300_l2sm.nc",
                "download_href": "/oads/access/login?r=x&d=SM_1.nc",
            },
        ])
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(status_code=200, content=b"data")

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox", return_value=True,
        ) as mock_overlap:
            dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        assert mock_overlap.call_count == 1
        satellite, start, end = mock_overlap.call_args[0][0:3]
        assert satellite == "smos"
        assert start == datetime(2026, 1, 2, 10, 37, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 1, 2, 12, 36, 3, tzinfo=timezone.utc)

    def test_whole_day_window_used_when_filename_unparseable(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {"filename": "SM_1.nc", "download_href": "/oads/access/login?r=x&d=SM_1.nc"},
        ])
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(status_code=200, content=b"data")

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ), patch(
            "sar_validation.core.orbit_coverage.orbit_overlaps_bbox", return_value=True,
        ) as mock_overlap:
            dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        assert mock_overlap.call_count == 1
        satellite, start, end = mock_overlap.call_args[0][0:3]
        assert satellite == "smos"
        assert start == datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 7, 1, 23, 59, 59, tzinfo=timezone.utc)

    def test_tle_fetch_error_keeps_file_fail_open(self, tmp_path):
        """A mocked TleFetchError must keep the file (fail-open), matching
        H-SAF's equivalent test. This downloader already has a
        whole-day-fallback test (above) covering *filename* parse
        failure; this test covers the different, currently-untested
        *prediction* failure path. This exercises the real, unmocked
        orbit_overlaps_bbox by making the TLE fetch it depends on fail:
        patching get_tle (not orbit_overlaps_bbox itself) lets
        orbit_overlaps_bbox's own documented `except TleFetchError:
        return True` fail-open path run for real."""
        from sar_validation.core.orbit_coverage import TleFetchError
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        dl._login = MagicMock()
        dl._list_products_for_day = MagicMock(return_value=[
            {
                "filename": "W_XX-ESA,SMOS,NRTNN_C_LEMM_20260102131619_20260102103700_20260102123603_o_v300_l2sm.nc",
                "download_href": "/oads/access/login?r=x&d=SM_1.nc",
            },
        ])
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(status_code=200, content=b"data")

        with patch(
            "sar_validation.downloaders.smos_downloader.requests.Session",
            return_value=fake_session,
        ), patch(
            "sar_validation.core.orbit_coverage.get_tle",
            side_effect=TleFetchError("no TLE available"),
        ):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-01",
            )

        assert len(result) == 1


# ---------------------------------------------------------------------------
# SMOSDownloader — OADS product listing
# ---------------------------------------------------------------------------

class TestSMOSListProductsForDay:
    def test_parses_product_list_html(self, tmp_path):
        """Fixture mirrors the REAL *authenticated* portal response: the 
        download link is a direct URL (/oads/data/NRT_Open/<filename>), 
        not the login-gated redirect an unauthenticated fetch sees, and it 
        carries an extra target="_blank" attribute between href and the 
        closing '>' -- the original regex assumed '>' came immediately 
        after href's closing quote and silently matched 0 products against 
        this real page."""
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        html = """
        <div class="productContainer">
        <h5 id="SM_OPER_1.nc" class="productTitle">SM_OPER_1.nc</h5>
        <div class="productLinks">
        <a href="/oads/data/NRT_Open/SM_OPER_1.nc" target="_blank">Download Product</a>
        </div></div>
        <div class="productContainer">
        <h5 id="SM_OPER_2.nc" class="productTitle">SM_OPER_2.nc</h5>
        <div class="productLinks">
        <a href="/oads/data/NRT_Open/SM_OPER_2.nc" target="_blank">Download Product</a>
        </div></div>
        """
        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        fake_session = MagicMock()
        fake_session.post.return_value = MagicMock(status_code=200, text=html)

        from datetime import date
        products = dl._list_products_for_day(fake_session, date(2025, 7, 3))

        assert [p["filename"] for p in products] == [
            "SM_OPER_1.nc",
            "SM_OPER_2.nc",
        ]
        assert products[0]["download_href"] == "/oads/data/NRT_Open/SM_OPER_1.nc"

        fake_session.post.assert_called_once()
        call = fake_session.post.call_args
        assert call.args[0] == "https://smos-diss.eo.esa.int/oads/access/collection/NRT_Open/tree"
        assert call.kwargs["data"] == {
            "p0": "MIR_SMNRT2", "p1": "2025", "p2": "07", "p3": "03",
        }

    def test_empty_day_returns_empty_list(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        fake_session = MagicMock()
        fake_session.post.return_value = MagicMock(status_code=200, text="<div>No products</div>")

        from datetime import date
        products = dl._list_products_for_day(fake_session, date(2025, 7, 3))

        assert products == []

    def test_decodes_html_entities_in_href(self, tmp_path):
        """HTML-entity-encoded ampersands (&amp;) in href attributes are
        decoded to plain & -- exercised via the older, login-gated-redirect
        href shape (which does carry query-string ampersands), still a
        real shape this parser must handle for an unauthenticated or
        differently-configured fetch, alongside the direct-download shape
        the primary test above covers."""
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        html = """
        <div class="productContainer">
        <h5 id="SM_OPER_1.nc" class="productTitle">SM_OPER_1.nc</h5>
        <div class="productLinks">
        <a href="/oads/access/login?r=collection%2FNRT_Open%2Ftree&amp;d=SM_OPER_1.nc">Download Product</a>
        </div></div>
        """
        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        fake_session = MagicMock()
        fake_session.post.return_value = MagicMock(status_code=200, text=html)

        from datetime import date
        products = dl._list_products_for_day(fake_session, date(2025, 7, 3))

        assert len(products) == 1
        # The decoded href should have plain & (not &amp;)
        assert products[0]["download_href"] == "/oads/access/login?r=collection%2FNRT_Open%2Ftree&d=SM_OPER_1.nc"
        assert "&amp;" not in products[0]["download_href"]


# ---------------------------------------------------------------------------
# SMOSDownloader — OADS SAML2/WSO2 SSO login
# ---------------------------------------------------------------------------

class TestSMOSOadsLogin:
    def test_full_saml_round_trip(self, tmp_path):
        """The login flow: GET the login page (redirects to the IdP with a
        sessionDataKey), POST credentials to the IdP's samlsso endpoint,
        then POST the returned SAMLResponse form back to the service
        provider's ACS endpoint.

        Fixture HTML mirrors the real live page byte-for-byte in the two
        respects that broke the original (double-quote-only) regexes
        against the real server: the login form's action is a *relative*
        URL (``action="../samlsso"``), and the sessionDataKey hidden
        input's value is *single*-quoted (``value='...'``) while other
        attributes on the same page use double quotes.
        """
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        idp_login_page = """
        <form class="ui large form" action="../samlsso"
              method="post" id="loginForm">
        <input id="tocommonauth" name="tocommonauth" type="hidden" value="true">
        <input type="hidden" name="sessionDataKey"
            value='abc123' />
        </form>
        """
        # Real captured response (see a real user's ~/.esa_smos_credentials
        # run): the ACS form's own attributes are double-quoted, but its
        # SAMLResponse/RelayState <input> name= and value= attributes are
        # BOTH single-quoted -- a different, inconsistent style from the
        # login form above, which is exactly why a fix that only made
        # value= quote-agnostic (and left name= hardcoded to ") still
        # failed against this real page.
        saml_response_page = """
        <form id="samlsso-response-form" method="post" action="https://smos-diss.eo.esa.int/Shibboleth.sso/SAML2/POST">
        <input type='hidden' name='RelayState' value='https://smos-diss.eo.esa.int/oads/access/login'/>
        <input type='hidden' name='SAMLResponse' value='opaque-blob=='/>
        </form>
        """

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        fake_session = MagicMock()
        # url= is the post-redirect URL requests.Session().get() lands on --
        # the relative "../samlsso" action must resolve against THIS, not
        # against OADS_LOGIN_URL.
        fake_session.get.return_value = MagicMock(
            status_code=200, text=idp_login_page,
            url="https://eoiam-idp.eo.esa.int/authenticationendpoint/login.do?sessionDataKey=abc123",
        )
        fake_session.post.return_value = MagicMock(
            status_code=200, text=saml_response_page,
            url="https://eoiam-idp.eo.esa.int/samlsso",
        )

        dl._login(fake_session, "u", "p")

        # Step 1: GET the login page to obtain the sessionDataKey.
        fake_session.get.assert_any_call(
            "https://smos-diss.eo.esa.int/oads/access/login", timeout=60,
        )
        # Step 2: POST credentials to the IdP -- the relative "../samlsso"
        # action must have been resolved to an absolute URL before posting.
        idp_post_call = fake_session.post.call_args_list[0]
        assert idp_post_call.args[0] == "https://eoiam-idp.eo.esa.int/samlsso"
        assert idp_post_call.kwargs["data"]["username"] == "u"
        assert idp_post_call.kwargs["data"]["password"] == "p"
        assert idp_post_call.kwargs["data"]["sessionDataKey"] == "abc123"
        # Step 3: POST the SAMLResponse back to the ACS endpoint.
        acs_post_call = fake_session.post.call_args_list[1]
        assert acs_post_call.args[0] == "https://smos-diss.eo.esa.int/Shibboleth.sso/SAML2/POST"
        assert acs_post_call.kwargs["data"]["SAMLResponse"] == "opaque-blob=="

    def test_raises_when_login_form_not_found(self, tmp_path):
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(status_code=200, text="<html>unexpected</html>",
                                                     url="https://eoiam-idp.eo.esa.int/samlsso")

        with pytest.raises(RuntimeError, match="SMOS OADS login"):
            dl._login(fake_session, "u", "p")

    def test_raises_and_saves_debug_html_when_saml_response_missing(self, tmp_path):
        """When the IdP responds to the credential POST with something
        other than the expected auto-submit SAMLResponse form (e.g.
        rejected credentials, an MFA prompt, or a changed layout), the
        raw response body is saved for inspection and the error message
        says where -- this failure mode was previously unhandled beyond a
        generic message, and is the one a real (non-sandboxed) user run
        actually hit."""
        from sar_validation.downloaders.smos_downloader import SMOSDownloader

        idp_login_page = """
        <form class="ui large form" action="../samlsso"
              method="post" id="loginForm">
        <input type="hidden" name="sessionDataKey"
            value='abc123' />
        </form>
        """
        rejected_credentials_page = "<html><body>Invalid credentials for user someone</body></html>"

        dl = SMOSDownloader(output_dir=tmp_path, dry_run=False, username="u", password="p")
        fake_session = MagicMock()
        fake_session.get.return_value = MagicMock(
            status_code=200, text=idp_login_page,
            url="https://eoiam-idp.eo.esa.int/authenticationendpoint/login.do?sessionDataKey=abc123",
        )
        fake_session.post.return_value = MagicMock(
            status_code=200, text=rejected_credentials_page,
            url="https://eoiam-idp.eo.esa.int/samlsso",
        )

        with pytest.raises(RuntimeError, match="no SAMLResponse returned"):
            dl._login(fake_session, "u", "p")

        debug_path = tmp_path / "smos_saml_debug.html"
        assert debug_path.exists()
        assert debug_path.read_text() == rejected_credentials_page


# ---------------------------------------------------------------------------
# SARDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestSARDownloaderAntimeridian:
    def _record(self, id_):
        return {
            "Id": id_, "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_query_splits_crossing_bbox_into_two_windows(self, tmp_path):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_products.side_effect = [
            [self._record("a")], [self._record("b")],
        ]
        dl._client = fake_client

        df = dl.query(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )

        assert fake_client.query_products.call_count == 2
        first_kwargs = fake_client.query_products.call_args_list[0].kwargs
        second_kwargs = fake_client.query_products.call_args_list[1].kwargs
        assert (first_kwargs["min_lon"], first_kwargs["max_lon"]) == (135.0, 180.0)
        assert (second_kwargs["min_lon"], second_kwargs["max_lon"]) == (-180.0, -120.0)
        assert sorted(df["Id"]) == ["a", "b"]

    def test_query_dedupes_product_returned_by_both_windows(self, tmp_path):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        dup = self._record("dup")
        fake_client.query_products.side_effect = [[dup], [dup]]
        dl._client = fake_client

        df = dl.query(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert len(df) == 1

    def test_query_non_crossing_bbox_calls_once(self, tmp_path):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path)
        fake_client = MagicMock()
        fake_client.query_products.return_value = []
        dl._client = fake_client

        dl.query(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-01-01", end="2026-01-02",
        )
        assert fake_client.query_products.call_count == 1
        kwargs = fake_client.query_products.call_args.kwargs
        assert (kwargs["min_lon"], kwargs["max_lon"]) == (-20.0, 0.0)


# ---------------------------------------------------------------------------
# SARDownloader — per-product existence check
# ---------------------------------------------------------------------------

class TestSARDownloaderForceDownload:
    def _fake_record(self):
        return {
            "Id": "abc", "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_skips_product_whose_directory_already_exists(self, tmp_path, capsys):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=False)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._fake_record()]
        dl._client = fake_client
        product_dir = tmp_path / "S1A_IW_OCN__2SDV_20260702T000000"
        product_dir.mkdir()

        result = dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        fake_client.download_product.assert_not_called()
        assert "Already downloaded" in capsys.readouterr().out
        # An already-on-disk product must still be reported back -- not
        # just a freshly-downloaded one -- since orchestrator.py writes
        # this return value straight into download_metadata.json's
        # "sar"."files", which dry_collocation.py's real (non-dry)
        # collocation-gating check reads to find the real SAR footprints
        # to predict against.
        assert result == [product_dir]

    def test_force_download_redownloads_existing_product(self, tmp_path):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._fake_record()]
        dl._client = fake_client
        fake_client.download_product.return_value = (
            tmp_path / "S1A_IW_OCN__2SDV_20260702T000000.SAFE"
        )
        (tmp_path / "S1A_IW_OCN__2SDV_20260702T000000").mkdir()

        dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        fake_client.download_product.assert_called_once()


# ---------------------------------------------------------------------------
# SARDownloader — zip-branch extraction
# ---------------------------------------------------------------------------

class TestSARDownloaderZipExtraction:
    def test_extracted_product_directory_is_returned(self, tmp_path):
        """CDSE always delivers a Sentinel-1 SAFE product as a .zip whose
        sole top-level member is the product's own <name>.SAFE directory.
        The extracted product directory must be appended to the returned
        list (and hence written into download_metadata.json's
        "sar"."files") -- not silently dropped after extraction, the way
        TestASCATSoilMoistureDownloaderZipExtraction already covers for
        the (differently-shaped, flat-file) ASCAT SSM case. Silently
        dropping it here is what made the real (non-dry) collocation-
        gating path in orchestrator.py/dry_collocation.py always operate
        on zero real SAR footprints, since it reads this same file list."""
        import io
        import zipfile

        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        product_name = "S1C_IW_OCN__2SDV_20260712T185023_20260712T185048_008514_010DB0_276D.SAFE"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"{product_name}/manifest.safe", b"fake manifest bytes")
            zf.writestr(f"{product_name}/measurement/owi.nc", b"fake measurement bytes")
        zip_bytes = buf.getvalue()

        dl = SARDownloader(output_dir=tmp_path, dry_run=False)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [{
            "Id": "abc", "Name": product_name,
            "ContentDate_Start": "2026-07-12T18:50:23Z",
            "ContentDate_End": "2026-07-12T18:50:48Z",
            "ContentLength_GB": 1.0, "Online": True,
        }]
        dl._client = fake_client

        def _fake_download_product(product_id, output_dir, product_name_arg=""):
            zip_path = output_dir / f"{product_name}.zip"
            zip_path.write_bytes(zip_bytes)
            return zip_path

        fake_client.download_product.side_effect = _fake_download_product

        result = dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-12", end="2026-07-13",
        )

        expected_dir = tmp_path / product_name
        assert result == [expected_dir]
        assert expected_dir.is_dir()
        assert (expected_dir / "manifest.safe").exists()
        assert not (tmp_path / f"{product_name}.zip").exists()


# ---------------------------------------------------------------------------
# SARDownloader — found_count
# ---------------------------------------------------------------------------

class TestSARDownloaderFoundCount:
    def _record(self, id_="abc"):
        return {
            "Id": id_, "Name": "S1A_IW_OCN__2SDV_20260702T000000",
            "ContentDate_Start": "2026-07-02T00:00:00Z",
            "ContentDate_End": "2026-07-02T00:00:10Z",
            "ContentLength_GB": 1.0, "Online": True,
        }

    def test_found_count_set_from_query_even_in_dry_run(self, tmp_path):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=True)
        fake_client = MagicMock()
        fake_client.query_products.return_value = [self._record("a"), self._record("b")]
        dl._client = fake_client

        out = dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        assert out == []
        assert dl.found_count == 2

    def test_found_count_zero_when_no_products_match(self, tmp_path):
        from sar_validation.downloaders.sentinel1_l2_ocn_downloader import SARDownloader

        dl = SARDownloader(output_dir=tmp_path, dry_run=True)
        fake_client = MagicMock()
        fake_client.query_products.return_value = []
        dl._client = fake_client

        dl.download(
            min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
            start="2026-07-02", end="2026-07-03",
        )

        assert dl.found_count == 0


# ---------------------------------------------------------------------------
# AltimeterDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestAltimeterDownloaderAntimeridian:
    def _patch_subset(self):
        from pathlib import Path

        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        return fake_module

    def test_crossing_bbox_splits_into_two_windows_with_distinct_filenames(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
                frequencies=["1hz"], satellites=["al"],
            )

        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert (first_kwargs["minimum_longitude"], first_kwargs["maximum_longitude"]) == (135.0, 180.0)
        assert (second_kwargs["minimum_longitude"], second_kwargs["maximum_longitude"]) == (-180.0, -120.0)
        assert first_kwargs["output_filename"] != second_kwargs["output_filename"]
        assert len(paths) == 2

    def test_non_crossing_bbox_keeps_single_call_and_original_filename(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-06-01", end="2026-06-02",
                frequencies=["1hz"], satellites=["al"],
            )

        assert fake_module.subset.call_count == 1
        kwargs = fake_module.subset.call_args.kwargs
        assert kwargs["output_filename"] == "cmems_obs-wave_glo_phy-swh_nrt_al-l3_PT1S_2026-06-01_2026-06-02.nc"
        assert len(paths) == 1

    def test_satellite_whose_availability_ended_before_the_window_is_skipped(self, tmp_path, capsys):
        """h2c stopped producing data 2026-05-20 (see AVAILABILITY_END) --
        a window entirely after that must never call subset() for it."""
        from unittest.mock import patch

        from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader

        dl = AltimeterDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = self._patch_subset()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-01", end="2026-07-02",
                frequencies=["1hz"], satellites=["h2c"],
            )

        fake_module.subset.assert_not_called()
        assert paths == []
        out = capsys.readouterr().out
        assert "Skipping" in out and "availability ended" in out


# ---------------------------------------------------------------------------
# InSituDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestInSituDownloaderAntimeridian:
    def test_download_splits_crossing_bbox_into_two_windows(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        # force_download=True: dest_path.exists() pre-check would
        # otherwise skip subset() entirely for the pre-created files below
        # (they exist only to satisfy _download_window's post-call "already
        # at dest_path" branch, since the fake subset() doesn't write real
        # files). This test is about window splitting, not the pre-check.
        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None  # real subset writes to CWD; not needed here

        start_dt, end_dt = "2026-07-02T00:00:00", "2026-07-03T00:00:00"
        for lo, hi in [(135.0, 180.0), (-180.0, -120.0)]:
            fname = _build_csv_filename(lo, hi, -15.0, 30.0, start_dt, end_dt, -20.0, 20.0)
            # Pre-create the destination file so _download_window's
            # "already at dest_path" branch is taken instead of the
            # CWD-relative move (which the fake subset() doesn't produce).
            (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert (first_kwargs["minimum_longitude"], first_kwargs["maximum_longitude"]) == (135.0, 180.0)
        assert (second_kwargs["minimum_longitude"], second_kwargs["maximum_longitude"]) == (-180.0, -120.0)
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert paths[0].name != paths[1].name

    def test_non_crossing_bbox_calls_once_and_returns_single_path(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        # force_download=True: see comment in the sibling test above — the
        # pre-created file is a mock-download workaround, not the subject
        # under test here.
        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        assert fake_module.subset.call_count == 1
        assert len(paths) == 1
        assert paths[0].name == fname


class TestInSituDownloaderDatasetPartFallbackErrorMessage:
    """When a requested date predates the CMEMS in-situ dataset's coverage,
    both the auto-selected dataset_part and its fallback fail. Raising the
    fallback's bare exception is misleading: for an old date (e.g. 2019),
    the initial attempt is 'monthly' (whose own error correctly reports
    that part's ~2020-onwards coverage), then the retry against 'latest'
    fails too, but with 'latest's own ~30-day rolling-window bounds --
    those are the only bounds a caller ever sees if just e2 is raised,
    making the dataset look far more limited than it actually is. The
    combined message must surface both parts' own reported bounds."""

    def test_error_includes_both_dataset_parts_coverage_messages(self, tmp_path):
        from sar_validation.downloaders.insitu_downloader import InSituDownloader

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            part = kwargs["dataset_part"]
            if part == "monthly":
                raise ValueError(
                    "Some of your subset selection [2019-02-01T16:30:00, "
                    "2019-02-01T19:30:00] for the time dimension exceed the "
                    "dataset coordinates [2020-01-01T00:00:00, 2026-07-01T02:45:00]"
                )
            raise ValueError(
                "Some of your subset selection [2019-02-01T16:30:00, "
                "2019-02-01T19:30:00] for the time dimension exceed the "
                "dataset coordinates [2026-06-30T00:00:00, 2026-07-30T08:27:00]"
            )

        fake_module.subset.side_effect = fake_subset

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with pytest.raises(RuntimeError) as exc_info:
                dl.download(
                    min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                    start="2019-02-01T16:30:00", end="2019-02-01T19:30:00",
                )

        assert fake_module.subset.call_count == 2
        msg = str(exc_info.value)
        assert "monthly" in msg
        assert "latest" in msg
        assert "2020-01-01T00:00:00" in msg, (
            f"expected the monthly part's own reported lower bound in the "
            f"combined error, got: {msg}"
        )
        assert "2026-06-30T00:00:00" in msg, (
            f"expected the latest part's own reported lower bound in the "
            f"combined error, got: {msg}"
        )


class TestInSituDownloaderForceDownload:
    def test_skips_download_when_file_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            paths = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        fake_module.subset.assert_not_called()
        assert len(paths) == 1
        assert paths[0].name == fname

    def test_force_download_redownloads_existing_file(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.insitu_downloader import (
            InSituDownloader,
            _build_csv_filename,
        )

        dl = InSituDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        fake_module = MagicMock()
        fake_module.subset.side_effect = lambda **kwargs: None

        start_dt, end_dt = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
        fname = _build_csv_filename(-20.0, 0.0, 35.0, 60.0, start_dt, end_dt, -20.0, 20.0)
        (tmp_path / fname).write_text("platform_type\n")

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        fake_module.subset.assert_called_once()


# ---------------------------------------------------------------------------
# ScatterometerDownloader — antimeridian crossing
# ---------------------------------------------------------------------------

class TestScatterometerDownloaderAntimeridian:
    def test_dry_run_prints_both_windows(self, tmp_path, capsys):
        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert out == []
        captured = capsys.readouterr().out.replace(" ", "")
        assert "[135.0,180.0]" in captured
        assert "[-180.0,-120.0]" in captured

    def test_search_runs_once_per_window_and_dedupes_products(self, tmp_path, capsys):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        # "dup" is returned by both window searches and must be counted once.
        # None of these IDs contain "metopb"/"metopc", so the per-product
        # download loop skips them immediately — this test only exercises
        # the search+dedup logic, not the download loop.
        fake_collection.search.side_effect = [["dup", "east_only"], ["dup", "west_only"]]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            result = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert result == []
        assert fake_collection.search.call_count == 2
        first_kwargs = fake_collection.search.call_args_list[0].kwargs
        second_kwargs = fake_collection.search.call_args_list[1].kwargs
        assert first_kwargs["bbox"] == "135.0,-15.0,180.0,30.0"
        assert second_kwargs["bbox"] == "-180.0,-15.0,-120.0,30.0"
        assert "Found 3 ASCAT products." in capsys.readouterr().out

    def test_non_crossing_bbox_searches_once(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = []
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-01-01", end="2026-01-02",
            )

        assert fake_collection.search.call_count == 1
        assert fake_collection.search.call_args.kwargs["bbox"] == "-20.0,35.0,0.0,60.0"


# ---------------------------------------------------------------------------
# Scatterometer downloader — per-product existence check
# ---------------------------------------------------------------------------

class TestScatterometerDownloaderForceDownload:
    def test_force_download_redownloads_existing_product(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.scatterometer_downloader import ScatterometerDownloader

        dl = ScatterometerDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        dl._token = "fake-token"
        (tmp_path / "OASWC12_20260705_183300_71590_metopb.nc").write_bytes(b"")

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = ["71590_metopb"]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        fake_file = MagicMock()
        fake_file.name = "OASWC12_20260705_183300_71590_metopb.nc"
        fake_file.read.side_effect = [b"data", b""]
        fake_product = MagicMock()
        fake_product.open.return_value.__enter__.return_value = fake_file
        fake_product.open.return_value.__exit__.return_value = False
        fake_datastore.get_product.return_value = fake_product

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_called_once()


class TestScatterometerAndAscatForceDownloadSkipsExisting:
    """`test_skips_product_whose_output_file_already_exists` was identical
    logic (mock the eumdac collection.search() result, drop a pre-existing
    output file, assert get_product() is never called) duplicated once per
    downloader class; parametrized here over the class/product-id/filename
    triple that differed between them."""

    ASCAT_REAL_PRODUCT_ID = "ASCA_SMR_02_M02_20260705204500Z_20260705222658Z_N_O_20260705214249Z"

    @pytest.mark.parametrize(
        "downloader_module,downloader_cls_name,product_id,filename",
        [
            pytest.param(
                "sar_validation.downloaders.scatterometer_downloader",
                "ScatterometerDownloader",
                "71590_metopb",
                "OASWC12_20260705_183300_71590_metopb.nc",
                id="scatterometer",
            ),
            pytest.param(
                "sar_validation.downloaders.ascat_soil_moisture_downloader",
                "ASCATSoilMoistureDownloader",
                ASCAT_REAL_PRODUCT_ID,
                f"{ASCAT_REAL_PRODUCT_ID}.nc",
                id="ascat_soil_moisture",
            ),
        ],
    )
    def test_skips_product_whose_output_file_already_exists(
        self, downloader_module, downloader_cls_name, product_id, filename, tmp_path
    ):
        import importlib

        downloader_cls = getattr(importlib.import_module(downloader_module), downloader_cls_name)

        dl = downloader_cls(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"
        (tmp_path / filename).write_bytes(b"")

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = [product_id]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_not_called()


# ---------------------------------------------------------------------------
# ASCAT Soil Moisture downloader — antimeridian support
# ---------------------------------------------------------------------------

class TestASCATSoilMoistureDownloaderAntimeridian:
    def test_dry_run_prints_both_windows(self, tmp_path, capsys):
        from sar_validation.downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader

        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(
            min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
            start="2026-07-02", end="2026-07-03",
        )
        assert out == []
        captured = capsys.readouterr().out.replace(" ", "")
        assert "[135.0,180.0]" in captured
        assert "[-180.0,-120.0]" in captured

    def test_search_runs_once_per_window_and_dedupes_products(self, tmp_path, capsys):
        from unittest.mock import patch

        from sar_validation.downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader

        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.side_effect = [["dup", "east_only"], ["dup", "west_only"]]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            result = dl.download(
                min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert result == []
        assert fake_collection.search.call_count == 2
        assert "Found 3 ASCAT SSM products." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ASCAT Soil Moisture downloader — per-product existence check
# ---------------------------------------------------------------------------

class TestASCATSoilMoistureDownloaderForceDownload:
    # Real EUMETSAT SOMO12 product IDs use short satellite codes M01/M02/M03,
    # never the literal strings "metopb"/"metopc" (unlike the OSI-104 wind
    # collection). Fake IDs below mirror that real shape.
    REAL_PRODUCT_ID = "ASCA_SMR_02_M02_20260705204500Z_20260705222658Z_N_O_20260705214249Z"

    def test_downloads_product_with_real_metop_satellite_code(self, tmp_path):
        """Regression test: real SOMO12 product IDs contain M01/M02/M03, not
        the literal strings "metopb"/"metopc". The satellite-code filter must
        not silently skip every product on account of that."""
        from unittest.mock import patch

        from sar_validation.downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader

        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = [self.REAL_PRODUCT_ID]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        fake_file = MagicMock()
        fake_file.name = f"{self.REAL_PRODUCT_ID}.nc"
        fake_file.read.side_effect = [b"data", b""]
        fake_product = MagicMock()
        fake_product.open.return_value.__enter__.return_value = fake_file
        fake_product.open.return_value.__exit__.return_value = False
        fake_datastore.get_product.return_value = fake_product

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            downloaded = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_product.assert_called_once()
        assert downloaded == [tmp_path / f"{self.REAL_PRODUCT_ID}.nc"]

    def test_uses_somo12_collection_id(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.ascat_soil_moisture_downloader import (
            COLLECTION_ID,
            ASCATSoilMoistureDownloader,
        )

        assert COLLECTION_ID == "EO:EUM:DAT:METOP:SOMO12"

        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = []
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        fake_datastore.get_collection.assert_called_once_with(COLLECTION_ID)


# ---------------------------------------------------------------------------
# ASCAT Soil Moisture downloader — zip-branch extraction
# ---------------------------------------------------------------------------

class TestASCATSoilMoistureDownloaderZipExtraction:
    def test_extracted_files_are_returned_and_counted(self, tmp_path):
        """When a product arrives as a .zip, the extracted file(s) must be
        appended to the returned ``downloaded`` list (and hence counted in
        the printed "Downloaded N file(s)" total) -- not silently dropped
        after extraction."""
        import io
        import zipfile
        from unittest.mock import patch

        from sar_validation.downloaders.ascat_soil_moisture_downloader import ASCATSoilMoistureDownloader

        product_id = "ASCA_SMR_02_M02_20260705204500Z_20260705222658Z_N_O_20260705214249Z"
        inner_name = f"{product_id}.nat"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(inner_name, b"fake ascat product bytes")
        zip_bytes = buf.getvalue()

        dl = ASCATSoilMoistureDownloader(output_dir=tmp_path, dry_run=False)
        dl._token = "fake-token"

        fake_eumdac = MagicMock()
        fake_collection = MagicMock()
        fake_collection.search.return_value = [product_id]
        fake_datastore = MagicMock()
        fake_datastore.get_collection.return_value = fake_collection
        fake_eumdac.DataStore.return_value = fake_datastore

        fake_file = MagicMock()
        fake_file.name = f"{product_id}.zip"
        fake_file.read.side_effect = [zip_bytes, b""]
        fake_product = MagicMock()
        fake_product.open.return_value.__enter__.return_value = fake_file
        fake_product.open.return_value.__exit__.return_value = False
        fake_datastore.get_product.return_value = fake_product

        with patch.dict("sys.modules", {"eumdac": fake_eumdac}):
            downloaded = dl.download(
                min_lon=-20.0, max_lon=0.0, min_lat=35.0, max_lat=60.0,
                start="2026-07-02", end="2026-07-03",
            )

        assert downloaded == [tmp_path / inner_name]
        assert (tmp_path / inner_name).exists()


# ---------------------------------------------------------------------------
# Tests for in-situ source-type <-> Copernicus platform-code mapping
# ---------------------------------------------------------------------------

class TestInsituPlatformCodeMapping:
    def test_resolve_platform_codes_dedupes_shared_db(self):
        codes = _resolve_platform_codes(["buoy", "drifter"])
        assert codes == ["DB", "AD"]

    def test_resolve_platform_codes_unknown_source_type_raises(self):
        with pytest.raises(ValueError):
            _resolve_platform_codes(["not_a_real_type"])


# ---------------------------------------------------------------------------
# RadiometerDownloader (RSS radiometer over HTTPS)
# ---------------------------------------------------------------------------

from sar_validation.downloaders.radiometer_downloader import (
    SENSORS,
    SUPPORTED_SENSORS,
    RadiometerDownloader,
)


class TestRadiometerDownloader:
    def test_amsr2_is_a_supported_netcdf_sensor(self):
        assert "amsr2" in SUPPORTED_SENSORS
        assert SENSORS["amsr2"]["format"] == "netcdf"

    def test_bytemap_sensors_supported(self):
        # GMI/SSMIS/WindSat are RSS binary bytemaps and now downloadable.
        for s in ("gmi", "ssmis_f16", "ssmis_f17", "ssmis_f18", "windsat"):
            assert s in SENSORS
            assert SENSORS[s]["format"] == "bytemap"
            assert s in SUPPORTED_SENSORS
            assert SENSORS[s]["url_path"]        # has a configured download URL

    def test_only_windsat_has_direction(self):
        assert SENSORS["windsat"]["has_direction"] is True
        for s in ("amsr2", "gmi", "ssmis_f16"):
            assert SENSORS[s]["has_direction"] is False

    def test_dry_run_lists_urls_without_network(self, tmp_path, capsys):
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        paths = dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                            start="2024-06-01", end="2024-06-02")
        out = capsys.readouterr().out
        assert paths == []
        assert "DRY RUN" in out
        # Both days for the default (amsr2) sensor, with the correct URL shape.
        assert "RSS_AMSR2_ocean_L3_daily_2024-06-01_v08.2.nc" in out
        assert "RSS_AMSR2_ocean_L3_daily_2024-06-02_v08.2.nc" in out
        assert "data.remss.com/amsr2/ocean/L3" in out

    def test_dry_run_lists_bytemap_urls(self, tmp_path, capsys):
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                    start="2024-06-01", end="2024-06-01")
        out = capsys.readouterr().out
        # Monthly-subfolder .gz URLs with a YYYYMMDD stamp, per sensor.
        assert "gmi/bmaps_v08.2/y2024/m06/f35_20240601v8.2.gz" in out
        assert "ssmi/f16/bmaps_v07/y2024/m06/f16_20240601v7.gz" in out
        assert "windsat/bmaps_v07.0.1/y2024/m06/wsat_20240601v7.0.1.gz" in out

    def test_availability_window_skips_early_dates(self, tmp_path, capsys):
        # AMSR2 data starts 2012 — a 2010 request should be skipped, no download.
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                    start="2010-01-01", end="2010-01-02", sensors=["amsr2"])
        out = capsys.readouterr().out
        assert "Skipping amsr2" in out
        assert "availability" in out

    def test_unknown_sensor_warns(self, tmp_path, capsys):
        dl = RadiometerDownloader(output_dir=tmp_path, dry_run=True)
        dl.download(min_lon=-10, max_lon=5, min_lat=50, max_lat=62,
                    start="2024-06-01", end="2024-06-01", sensors=["not_a_sensor"])
        out = capsys.readouterr().out
        assert "unknown radiometer sensor" in out.lower()


# ---------------------------------------------------------------------------
# RSS binary bytemap reader (_rss_bytemap.read_rss_bytemap)
# ---------------------------------------------------------------------------

import gzip

import numpy as np

from sar_validation.downloaders._rss_bytemap import (
    BYTEMAP_LAYOUT,
    NLAT,
    NLON,
    NPASS,
    read_rss_bytemap,
)


def _write_bytemap(tmp_path, sensor, filename, cells):
    """Write a full-size RSS bytemap .gz (all-missing 255 except `cells`).

    cells: list of (pass, var_idx, lat_idx, lon_idx, byte_value).
    """
    nvar = len(BYTEMAP_LAYOUT[sensor]["vars"])
    arr = np.full((NPASS, nvar, NLAT, NLON), 255, np.uint8)
    for (p, v, la, lo, val) in cells:
        arr[p, v, la, lo] = val
    path = tmp_path / filename
    with gzip.open(path, "wb") as fh:
        fh.write(arr.tobytes())
    return path


class TestReadRssBytemap:
    def test_gmi_scale_offset_and_grid(self, tmp_path):
        # GMI var indices: 0=time (×0.1), 2=windLF (×0.2).
        p = _write_bytemap(tmp_path, "gmi", "f35_20240601v8.2.gz",
                           [(0, 2, 400, 600, 50), (0, 0, 400, 600, 100)])
        decoded, lon, lat = read_rss_bytemap(p, "gmi")
        assert decoded["windLF"].shape == (NPASS, NLAT, NLON)
        assert decoded["windLF"][0, 400, 600] == pytest.approx(10.0)   # 50×0.2
        assert decoded["time"][0, 400, 600] == pytest.approx(10.0)     # 100×0.1
        assert np.isnan(decoded["windLF"][1, 0, 0])                    # 255 → NaN
        assert lon[600] == pytest.approx(600 * 0.25 + 0.125)
        assert lat[400] == pytest.approx(400 * 0.25 - 89.875)

    def test_missing_code_threshold(self, tmp_path):
        # Byte 250 is valid; 251 is the first special/missing code.
        p = _write_bytemap(tmp_path, "gmi", "f35_20240101v8.2.gz",
                           [(0, 2, 0, 0, 250), (0, 2, 0, 1, 251)])
        decoded, _, _ = read_rss_bytemap(p, "gmi")
        assert decoded["windLF"][0, 0, 0] == pytest.approx(50.0)       # 250×0.2
        assert np.isnan(decoded["windLF"][0, 0, 1])                    # 251 masked

    def test_windsat_has_nine_vars_incl_wdir(self, tmp_path):
        p = _write_bytemap(tmp_path, "windsat", "wsat_20150601v7.0.1.gz",
                           [(0, 8, 300, 500, 40)])
        decoded, _, _ = read_rss_bytemap(p, "windsat")
        assert "wdir" in decoded and "w-lf" in decoded
        assert decoded["wdir"][0, 300, 500] == pytest.approx(60.0)     # 40×1.5

    def test_size_mismatch_raises(self, tmp_path):
        path = tmp_path / "f35_bad.gz"
        with gzip.open(path, "wb") as fh:
            fh.write(b"\x00" * 100)
        with pytest.raises(ValueError):
            read_rss_bytemap(path, "gmi")

    def test_unknown_sensor_raises(self, tmp_path):
        with pytest.raises(KeyError):
            read_rss_bytemap(tmp_path / "x.gz", "not_a_sensor")


from sar_validation.downloaders.noaa_hfradar_downloader import (
    ERDDAP_BASE,
    build_erddap_subset_url,
    clamp_to_region_bbox,
    select_backend,
    select_erddap_dataset,
)


class TestSelectErddapDataset:
    @pytest.mark.parametrize(
        "min_lon,max_lon,min_lat,max_lat,resolution_km,expected_dataset",
        [
            pytest.param(-125, -119, 33, 38, 6, "ucsdHfrW6", id="us_west_6km_default"),
            pytest.param(-125, -119, 33, 38, 2, "ucsdHfrW2", id="us_west_2km"),
            pytest.param(-80, -70, 35, 42, 6, "ucsdHfrE6", id="us_east_gulf_6km"),
        ],
    )
    def test_selects_expected_dataset(
        self, min_lon, max_lon, min_lat, max_lat, resolution_km, expected_dataset
    ):
        assert select_erddap_dataset(min_lon, max_lon, min_lat, max_lat, resolution_km) == expected_dataset

    def test_unsupported_region_raises(self):
        # German Bight isn't in NOAA_HFR_REGIONS at all (this suite's shared
        # table is US-only), so match_noaa_hfr_region's own "no region"
        # message surfaces here rather than an ERDDAP-specific one.
        with pytest.raises(ValueError, match="No NOAA HF-radar region"):
            select_erddap_dataset(2.0, 8.0, 53.0, 55.0, 6)

    def test_unsupported_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            # US-East/Gulf's shared-table entry has no 3 km dataset (only
            # 1/2/6 km).
            select_erddap_dataset(-80, -70, 35, 42, 3)


class TestClampToRegionBbox:
    def test_bbox_fully_inside_region_is_unchanged(self):
        assert clamp_to_region_bbox(-80, -70, 35, 42) == (-80.0, -70.0, 35.0, 42.0)

    def test_bbox_extending_south_of_region_is_clamped(self):
        # Reproduces the reported bug: a recipe bbox reaching down to 20.0N
        # (to also cover Puerto Rico) extends past US-East/Gulf's actual
        # southern grid edge (22.0N in _REGIONS), which ERDDAP rejects with
        # HTTP 404 rather than clipping server-side.
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(-80, -60, 20.0, 40.0)
        assert min_lat == 22.0
        assert (min_lon, max_lon, max_lat) == (-80.0, -60.0, 40.0)

    def test_clamped_bbox_stays_within_erddap_axis_bounds(self):
        # 22.0N (the _REGIONS config bound) must be >= the real ERDDAP grid's
        # minimum latitude (21.73596N per the dataset's .das), so clamping to
        # it never re-triggers the same out-of-bounds 404.
        _, _, min_lat, _ = clamp_to_region_bbox(-80, -60, 20.0, 40.0)
        assert min_lat >= 21.73596

    def test_west_coast_bbox_at_old_config_edge_is_clamped(self):
        # Reproduces the reported bug: recipes/currents_uswestcoast2.yaml's
        # bbox reaches min_lat=30.0, which used to equal the (too loose)
        # _REGIONS US_WEST config bound, so nothing got clamped and the
        # unclamped 30.0 reached ERDDAP below its real axis minimum
        # (30.25N per ucsdHfrW6's .das), triggering the same HTTP 404.
        # The recipe's max_lon=-115.0 is also past the real axis maximum
        # (-115.8056 per the .das), so it gets clamped too.
        min_lon, max_lon, min_lat, max_lat = clamp_to_region_bbox(
            -126.0, -115.0, 30.0, 48.0
        )
        assert min_lat == 30.25
        assert (min_lon, max_lon, max_lat) == (-126.0, -115.8056, 48.0)
        assert min_lat >= 30.25  # real ERDDAP grid's minimum latitude
        assert max_lon <= -115.8056  # real ERDDAP grid's maximum longitude


class TestBuildErddapSubsetUrl:
    def test_url_has_vars_bbox_and_time_selectors(self):
        url = build_erddap_subset_url(
            "ucsdHfrW6", -125, -119, 33, 38, "2024-05-01", "2024-05-01T06:00:00"
        )
        assert url.startswith(f"{ERDDAP_BASE}/ucsdHfrW6.nc?")
        assert "water_u[(2024-05-01T00:00:00Z):(2024-05-01T06:00:00Z)]" in url
        assert "water_v[(2024-05-01T00:00:00Z):(2024-05-01T06:00:00Z)]" in url
        assert "[(33.0):(38.0)]" in url   # latitude ascending
        assert "[(-125.0):(-119.0)]" in url  # longitude ascending


class TestSelectBackend:
    def test_recent_date_uses_erddap(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        assert select_backend(recent) == "erddap"

    def test_old_date_not_yet_supported(self):
        with pytest.raises(NotImplementedError, match="THREDDS archive backend"):
            select_backend("2015-01-01")


from unittest.mock import patch

from sar_validation.downloaders.noaa_hfradar_downloader import NOAAHFRadarDownloader

# NOTE: the task brief's verbatim test used a hardcoded "2024-05-01" date for
# `end`. select_backend() rejects any `end` older than the rolling ~90-day
# ERDDAP window relative to wall-clock "now", so a hardcoded past date goes
# stale and starts raising NotImplementedError once the suite is run more
# than ~90 days after the brief was written. Using a date a few days before
# "now" keeps these tests deterministically inside the window (mirroring the
# existing TestSelectBackend.test_recent_date_uses_erddar pattern above)
# without changing any of the brief's assertions.
_RECENT_START = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
_RECENT_END = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT06:00:00")


def _configure_fake_download_response(mock_urlopen, data: bytes = b"fake-netcdf-bytes"):
    """Configure a patched urllib.request.urlopen mock to behave as the
    context manager the real downloaders use:
    `with urllib.request.urlopen(url, timeout=...) as resp: dest.write_bytes(resp.read())`.
    """
    mock_urlopen.return_value.__enter__.return_value.read.return_value = data


class TestNOAAHFRadarDownload:
    def test_dry_run_returns_empty_list_and_no_fetch(self, tmp_path, capsys):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert out == []
        m.assert_not_called()
        assert "ucsdHfrW6.nc?" in capsys.readouterr().out

    def test_download_fetches_url_to_expected_path(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert len(out) == 1
        assert out[0].parent == tmp_path
        assert out[0].suffix == ".nc"
        m.assert_called_once()
        called_url = m.call_args[0][0]
        assert "ucsdHfrW6.nc?" in called_url
        assert out[0].read_bytes() == b"fake-netcdf-bytes"

    def test_download_uses_a_bounded_timeout(self, tmp_path):
        """Regression test: urlretrieve (the original implementation) has no
        timeout parameter at all, and a stalled connection would hang the
        download indefinitely. The fix moved to urlopen(..., timeout=...)
        specifically so a hung connection is bounded. Lowered from 60 to 15: 
        combined with prefer_ipv4_dns(), a genuinely broken IPv6
        path now fails fast per address instead of eating up to 6 * timeout
        seconds before ever reaching a working IPv4 address."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        m.assert_called_once()
        assert m.call_args.kwargs["timeout"] == 15

    def test_download_wraps_network_call_in_prefer_ipv4_dns(self, tmp_path):
        """Wiring test: the ERDDAP file-download call site must actually use
        prefer_ipv4_dns(), not just have it importable."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.prefer_ipv4_dns"
        ) as mock_prefer:
            mock_prefer.return_value.__exit__.return_value = False
            with patch(
                "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
            ) as m:
                _configure_fake_download_response(m)
                dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        mock_prefer.assert_called_once_with()
        mock_prefer.return_value.__enter__.assert_called_once()
        mock_prefer.return_value.__exit__.assert_called_once()

    def test_download_clamps_bbox_extending_past_region_edge(self, tmp_path):
        """A bbox reaching past a region's real grid edge must be clamped in
        the built URL, not passed straight through (root cause of the
        reported HTTP 404 'axis minimum' error)."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            dl.download(-80, -60, 20.0, 40.0, _RECENT_START, _RECENT_END)
        called_url = m.call_args[0][0]
        assert "[(22.0):(40.0)]" in called_url
        assert "[(20.0)" not in called_url

    def test_download_clamps_west_coast_recipe_bbox(self, tmp_path):
        """recipes/currents_uswestcoast2.yaml's exact bbox (min_lat=30.0)
        must be clamped to the real ERDDAP axis minimum (30.25N), not passed
        straight through and 404 at the server."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            dl.download(-126.0, -115.0, 30.0, 48.0, _RECENT_START, _RECENT_END)
        called_url = m.call_args[0][0]
        assert "[(30.25):(48.0)]" in called_url
        assert "[(30.0)" not in called_url
        assert "[(-126.0):(-115.8056)]" in called_url


_DAS_RANGE_START = (datetime.now(timezone.utc) - timedelta(days=30)).replace(microsecond=0)
_DAS_RANGE_END = (datetime.now(timezone.utc) - timedelta(days=1)).replace(microsecond=0)


def _fake_das_text(range_start: datetime, range_end: datetime) -> str:
    return f"""Attributes {{
 time {{
  String _CoordinateAxisType "Time";
  Float64 actual_range {range_start.timestamp()}, {range_end.timestamp()};
  String axis "T";
  String ioos_category "Time";
  String standard_name "time";
  String time_origin "01-JAN-1970 00:00:00";
  String units "seconds since 1970-01-01T00:00:00Z";
 }}
 latitude {{
  String _CoordinateAxisType "Lat";
  Float64 actual_range 30.25, 49.98;
 }}
 water_u {{
  Float64 colorBarMaximum 0.5;
  String ioos_category "Currents";
 }}
 NC_GLOBAL {{
  String title "HFRnet RTV";
 }}
}}
"""


_FAKE_DAS_TEXT = _fake_das_text(_DAS_RANGE_START, _DAS_RANGE_END)


class TestNoaaHfRadarParseDasTimeRange:
    def test_parses_actual_range_to_utc_datetimes(self):
        from sar_validation.downloaders.noaa_hfradar_downloader import _parse_das_time_range

        result = _parse_das_time_range(_FAKE_DAS_TEXT)
        assert result == (
            _DAS_RANGE_START.replace(tzinfo=None), _DAS_RANGE_END.replace(tzinfo=None),
        )

    def test_missing_time_block_returns_none(self):
        from sar_validation.downloaders.noaa_hfradar_downloader import _parse_das_time_range

        assert _parse_das_time_range("Attributes {\n latitude { }\n}\n") is None


class TestNOAAHFRadarCheckAvailabilityDry:
    def test_true_when_requested_window_overlaps_das_time_range(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m, data=_FAKE_DAS_TEXT.encode())
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is True
        called_url = m.call_args[0][0]
        assert called_url.endswith("ucsdHfrW6.das")

    def test_false_when_requested_window_outside_das_time_range(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        # A window entirely before the .das-reported coverage, but still
        # inside ERDDAP's own ~90-day rolling window (so select_backend
        # itself doesn't short-circuit this to False for the wrong reason).
        before_start = (datetime.now(timezone.utc) - timedelta(days=89)).strftime("%Y-%m-%d")
        before_end = (datetime.now(timezone.utc) - timedelta(days=88)).strftime("%Y-%m-%d")
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m, data=_FAKE_DAS_TEXT.encode())
            result = dl.check_availability_dry(-125, -119, 33, 38, before_start, before_end)

        assert result is False

    def test_true_when_das_has_no_parseable_time_range(self, tmp_path):
        """Fail-open: an unparseable .das response must not rule out
        availability."""
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m, data=b"Attributes { }\n")
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is True

    def test_false_when_end_date_outside_erddap_window(self, tmp_path):
        old = (datetime.now(timezone.utc) - timedelta(days=200)).strftime("%Y-%m-%d")
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            result = dl.check_availability_dry(-125, -119, 33, 38, old, old)

        assert result is False
        m.assert_not_called()

    def test_false_when_resolution_unavailable_for_region(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=0.5)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            # 0.5km isn't available for US-EastGulfCoast.
            result = dl.check_availability_dry(-80, -70, 35, 42, _RECENT_START, _RECENT_END)

        assert result is False
        m.assert_not_called()


class TestNOAAHFRadarDownload500m:
    def test_select_erddap_dataset_accepts_500m_for_us_west(self):
        from sar_validation.downloaders.noaa_hfradar_downloader import select_erddap_dataset

        assert select_erddap_dataset(-125, -119, 33, 38, 0.5) == "ucsdHfrW500"

    def test_500m_not_available_for_us_east_gulf(self):
        from sar_validation.downloaders.noaa_hfradar_downloader import select_erddap_dataset

        with pytest.raises(ValueError, match="500m"):
            select_erddap_dataset(-80, -70, 35, 42, 0.5)

    def test_download_500m_builds_500m_filename_not_500km(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=0.5)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert len(out) == 1
        assert "500m" in out[0].name
        assert "500km" not in out[0].name
        m.assert_called_once()
        called_url = m.call_args[0][0]
        assert "ucsdHfrW500.nc?" in called_url

    def test_existing_6km_filename_unaffected_by_resolution_token_change(self, tmp_path):
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)
        assert "6km" in out[0].name


class TestNoaaHfradarDownloaderUsesSharedRegionTable:
    def test_erddap_dataset_error_names_available_resolutions(self):
        from sar_validation.downloaders.noaa_hfradar_downloader import select_erddap_dataset

        with pytest.raises(ValueError, match=r"1\.0|2\.0|6\.0|0\.5|1km|2km|6km|500m"):
            select_erddap_dataset(-125, -119, 33, 38, 3)


class TestNOAAHFRadarDownloaderAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # 135E..120W doesn't overlap US_WEST or US_EAST_GULF on either side
        # of the split (NOAA's _match_region uses each window's *center*
        # point, and neither window's center falls inside either region).
        # Note: the unsplit pre-fix code also raises a ValueError matching
        # this message for a min_lon > max_lon input (its own center-point
        # math just lands on a different, still-uncovered point), so this
        # test alone doesn't distinguish pre-fix from post-fix — it guards
        # that the "truly nothing covers this" case keeps failing loudly
        # after the fix too. The next test is the one that actually fails
        # pre-fix.
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=True, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            with pytest.raises(ValueError, match="No NOAA HF-radar region"):
                dl.download(135.0, -120.0, -15.0, 30.0, _RECENT_START, _RECENT_END)
        m.assert_not_called()

    def test_crossing_bbox_downloads_the_side_whose_window_center_resolves(self, tmp_path):
        # NOAA's region match is center-point-based (not overlap-area, unlike
        # the Copernicus HFR regions), so only a window whose *own* center
        # (after splitting) lands inside a supported region resolves. Here
        # min_lon=179, max_lon=-66 splits into [179, 180] (center 179.5,
        # 36.5 — matches nothing) and [-180, -66] (center -123.0, 36.5 —
        # inside US_WEST's bbox). The raw (unsplit) request's own center,
        # (56.5, 36.5), matches nothing — that's what makes this fail today.
        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            out = dl.download(179.0, -66.0, 35.0, 38.0, _RECENT_START, _RECENT_END)
        assert len(out) == 1
        m.assert_called_once()
        called_url = m.call_args[0][0]
        assert "ucsdHfrW6.nc?" in called_url


class TestNOAAHFRadarDownloaderForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader,
            select_erddap_dataset,
        )

        dl = NOAAHFRadarDownloader(output_dir=tmp_path, dry_run=False, resolution_km=6)
        dataset_id = select_erddap_dataset(-125, -119, 33, 38, 6)
        out_path = tmp_path / f"{dataset_id}_6km_{_RECENT_START}.nc"
        out_path.write_bytes(b"")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            out = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        m.assert_not_called()
        assert out == [out_path]

    def test_force_download_refetches_existing_output(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.noaa_hfradar_downloader import (
            NOAAHFRadarDownloader,
            select_erddap_dataset,
        )

        dl = NOAAHFRadarDownloader(
            output_dir=tmp_path, dry_run=False, resolution_km=6, force_download=True,
        )
        dataset_id = select_erddap_dataset(-125, -119, 33, 38, 6)
        out_path = tmp_path / f"{dataset_id}_6km_{_RECENT_START}.nc"
        out_path.write_bytes(b"stale")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            _configure_fake_download_response(m)
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        m.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for DataOrchestrator "hf_radar_noaa" wiring
# ---------------------------------------------------------------------------
# DataOrchestrator can be built cheaply from a stub Recipe (no network, no
# real base-dir creation under dry_run), so a behavioural test is preferred
# over source-inspection alone.

class TestOrchestratorHFRadarNOAAWiring:
    def test_download_noaa_hfradar_dry_run_sets_metadata_and_makes_no_network_call(
        self, tmp_path
    ):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-noaa",
            variable="currents",
            output_dir=str(tmp_path),
            # US_WEST bbox: the only region select_erddap_dataset() accepts
            # for the default 6 km resolution.
            geographic_bounds=GeographicBounds(-125.0, -119.0, 33.0, 38.0),
            temporal_bounds=TemporalBounds(_RECENT_START, _RECENT_END),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar_noaa")

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as m:
            result = orchestrator._download_noaa_hfradar(source)

        assert result is True
        assert orchestrator.metadata["downloads"]["hf_radar_noaa"]["status"] == "dry_run"
        m.assert_not_called()

    def test_download_noaa_hfradar_honours_resolution_km_override(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-noaa-res",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(
            source_type="hf_radar_noaa",
            download_kwargs={"resolution_km": 1},
        )

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.NOAAHFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_noaa_hfradar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["resolution_km"] == 1


class TestOrchestratorHFRadarUSWiring:
    def test_download_hf_radar_us_dry_run_sets_metadata_and_makes_no_network_call(
        self, tmp_path
    ):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-us",
            variable="currents",
            output_dir=str(tmp_path),
            # US_WEST bbox, recent date -> resolves to NOAA internally.
            geographic_bounds=GeographicBounds(-125.0, -119.0, 33.0, 38.0),
            temporal_bounds=TemporalBounds(_RECENT_START, _RECENT_END),
        ))
        import urllib.error

        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar_us")

        # ERDDAP's own dry-run contract is unchanged: it only formats a
        # preview URL from parameters, no network call. THREDDS's dry-run
        # now queries each touched month's (lightweight) catalog.xml for a
        # real candidate count (see NOAATHREDDSHFRadarDownloader) -- the
        # waterfall still cascades into it even after ERDDAP's dry-run
        # "succeeds" (see the comment below), so its urlopen must be
        # mocked too; a 404 for every month simulates "not yet published",
        # which the THREDDS dry-run branch handles by reporting zero
        # matched granules rather than raising.
        def thredds_404(url, timeout=None):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with patch(
            "sar_validation.downloaders.noaa_hfradar_downloader.urllib.request.urlopen"
        ) as erddap_urlopen, patch(
            "sar_validation.downloaders.noaa_hfradar_thredds_downloader.urllib.request.urlopen",
            side_effect=thredds_404,
        ):
            result = orchestrator._download_hf_radar_us(source)

        assert result is True
        assert orchestrator.metadata["downloads"]["hf_radar_us"]["status"] == "dry_run"
        # Under dry_run, every backend's own dry-run contract is "print
        # intent, return []" (no exception, no files) -- so the waterfall
        # still cascades all the way through ERDDAP and THREDDS for
        # attempted_backends completeness, but resolved_backend reports the
        # FIRST backend that structurally applies (doesn't raise), not
        # whichever was tried last. For this US_WEST/recent-date bbox,
        # ERDDAP is the first (and only) backend that applies.
        assert orchestrator.metadata["downloads"]["hf_radar_us"]["backend"] == "erddap"
        erddap_urlopen.assert_not_called()

    def test_download_hf_radar_us_honours_resolution_km_override(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-us-res",
            variable="currents",
            output_dir=str(tmp_path),
            geographic_bounds=GeographicBounds(-125.0, -119.0, 33.0, 38.0),
            temporal_bounds=TemporalBounds(_RECENT_START, _RECENT_END),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(
            source_type="hf_radar_us",
            download_kwargs={"resolution_km": 1},
        )

        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_noaa:
            m_noaa.return_value.download.return_value = []
            orchestrator._download_hf_radar_us(source)

        assert m_noaa.call_args.kwargs["resolution_km"] == 1


# ---------------------------------------------------------------------------
# Tests for DataOrchestrator depth resolution (optional min_depth/max_depth)
# ---------------------------------------------------------------------------

class TestOrchestratorDepthResolution:
    def test_hf_radar_dispatch_uses_default_depth_when_unspecified(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-depth-default",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar")

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_hf_radar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0

    def test_hf_radar_dispatch_honours_explicit_depth(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-depth-explicit",
            variable="currents",
            output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar", min_depth=-2.0, max_depth=2.0)

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            orchestrator._download_hf_radar(source)

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -2.0
        assert kwargs["max_depth"] == 2.0

    def test_insitu_batch_uses_default_depth_when_all_unspecified(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-insitu-depth-default",
            variable="wind",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="buoy"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls:
            mock_cls.return_value.download.return_value = None
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            orchestrator.download_all()

        _, kwargs = mock_cls.call_args
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0

    def test_insitu_batch_widens_window_around_explicit_override(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-insitu-depth-mixed",
            variable="wind",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="mooring", min_depth=-5.0, max_depth=5.0),
                ValidationDataSource(source_type="buoy"),  # unspecified -> -20/20
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_cls, patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls:
            mock_cls.return_value.download.return_value = None
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            orchestrator.download_all()

        _, kwargs = mock_cls.call_args
        # most permissive window across resolved depths: min(-5,-20)=-20, max(5,20)=20
        assert kwargs["min_depth"] == -20.0
        assert kwargs["max_depth"] == 20.0


class TestDownloadHfRadarTracksFileCount:
    def test_metadata_records_file_count(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-hf-radar-file-count", variable="currents", output_dir=str(tmp_path),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        source = ValidationDataSource(source_type="hf_radar")

        with patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = [tmp_path / "hf_radar" / "a.nc"]
            orchestrator._download_hf_radar(source)

        assert orchestrator.metadata["downloads"]["hf_radar"]["file_count"] == 1


# ---------------------------------------------------------------------------
# Tests for _hf_radar_regions (shared Copernicus HF-radar region lookup)
# ---------------------------------------------------------------------------

from sar_validation.downloaders._hf_radar_regions import resolve_hfr_region


class TestHfRadarRegions:
    def test_us_east_gulf_bbox_resolves(self):
        assert resolve_hfr_region(-90.0, -60.0, 30.0, 40.0) == "US-EastGulfCoast"

    def test_us_west_coast_bbox_resolves(self):
        assert resolve_hfr_region(-125.0, -119.0, 33.0, 38.0) == "US-WestCoast"

    def test_no_overlap_raises_with_region_list(self):
        with pytest.raises(ValueError, match="US-EastGulfCoast"):
            resolve_hfr_region(100.0, 105.0, -10.0, -5.0)  # nowhere near any region

    def test_picks_largest_overlap_when_bbox_spans_two_regions(self):
        # DeltaEbro and ICATMAR genuinely overlap in the western
        # Mediterranean. A query bbox weighted toward each region's side of
        # the overlap should resolve to that region, exercising the
        # largest-overlap-area tie-break rather than "first match wins".
        assert resolve_hfr_region(0.0, 1.5, 39.6, 41.2) == "DeltaEbro"
        assert resolve_hfr_region(0.5, 4.0, 40.6, 42.9) == "ICATMAR"


# ---------------------------------------------------------------------------
# Tests for HFRadarDownloader querying the gridded radar-total dataset_parts
# ---------------------------------------------------------------------------

class TestHFRadarDownloaderGrid:
    def test_dry_run_prints_resolved_region_and_part(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-06-05", "2026-06-06")
        assert out == []
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "radar-total--US-EastGulfCoast" in captured

    def test_download_calls_subset_with_resolved_region_part(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            # Simulate copernicusmarine writing the requested file.
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert len(out) == 1
        assert out[0].exists()
        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "monthly-radar-total--US-EastGulfCoast"
        assert kwargs["minimum_longitude"] == -90.0
        assert kwargs["maximum_longitude"] == -60.0

    def test_recent_date_uses_latest_part_when_region_has_one(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        recent_end = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        # US-WestCoast has a `latest` feed (unlike US-EastGulfCoast).
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-125.0, -119.0, 33.0, 38.0, recent_start, recent_end)

        _, kwargs = fake_module.subset.call_args
        assert kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"

    def test_retries_with_monthly_part_when_latest_out_of_bounds(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        recent_end = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        call_count = {"n": 0}

        def fake_subset(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError(
                    "The requested time range appears to exceed the dataset "
                    "coordinates for this dataset_part."
                )
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        # US-WestCoast has a `latest` feed, so the first attempt uses it and
        # is expected to fail, triggering a retry with the `monthly` part.
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-125.0, -119.0, 33.0, 38.0, recent_start, recent_end)

        assert len(out) == 1
        assert out[0].exists()
        assert fake_module.subset.call_count == 2
        first_kwargs = fake_module.subset.call_args_list[0].kwargs
        second_kwargs = fake_module.subset.call_args_list[1].kwargs
        assert first_kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"
        assert second_kwargs["dataset_part"] == "monthly-radar-total--US-WestCoast"

    def test_raises_file_not_found_when_subset_writes_no_file(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()
        # subset() "succeeds" (no exception) but never writes the destination
        # file, simulating an empty/no-op response from copernicusmarine.
        fake_module.subset.side_effect = lambda **kwargs: None

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with pytest.raises(FileNotFoundError):
                dl.download(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")


class _FakeGridDataset:
    """Minimal stand-in for the xarray.Dataset copernicusmarine.open_dataset
    returns -- just enough surface (.sizes) for check_availability_dry to
    read, matching test_altimeter_downloader.py's _FakeDataset convention."""

    def __init__(self, time_size: int):
        self.sizes = {"time": time_size}


class TestHFRadarDownloaderCheckAvailabilityDry:
    def test_true_when_time_coordinate_non_empty(self, tmp_path):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path)
        fake_module = MagicMock()
        fake_module.open_dataset.return_value = _FakeGridDataset(3)

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            result = dl.check_availability_dry(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert result is True
        fake_module.subset.assert_not_called()
        _, kwargs = fake_module.open_dataset.call_args
        assert kwargs["dataset_part"] == "monthly-radar-total--US-EastGulfCoast"

    def test_false_when_time_coordinate_empty(self, tmp_path):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path)
        fake_module = MagicMock()
        fake_module.open_dataset.return_value = _FakeGridDataset(0)

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            result = dl.check_availability_dry(-90.0, -60.0, 30.0, 40.0, "2026-01-01", "2026-01-02")

        assert result is False

    def test_false_when_no_region_overlaps_bbox(self, tmp_path):
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path)
        fake_module = MagicMock()

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            result = dl.check_availability_dry(0.0, 5.0, 0.0, 5.0, "2026-01-01", "2026-01-02")

        assert result is False
        fake_module.open_dataset.assert_not_called()

    def test_recent_date_uses_latest_part_when_region_has_one(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        recent_end = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        recent_start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        dl = HFRadarDownloader(output_dir=tmp_path)
        fake_module = MagicMock()
        fake_module.open_dataset.return_value = _FakeGridDataset(1)

        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.check_availability_dry(-125.0, -119.0, 33.0, 38.0, recent_start, recent_end)

        _, kwargs = fake_module.open_dataset.call_args
        assert kwargs["dataset_part"] == "latest-radar-total--US-WestCoast"


class TestHFRadarDownloaderGridAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # lat 0-5 doesn't overlap any HFR_REGIONS entry on either side of
        # the split — the southernmost real region (US-Hawaii) starts at
        # 14.5N, so no window can resolve a region. Note: the *unsplit*
        # pre-fix code also raises a ValueError matching this same message
        # for a min_lon > max_lon input (its overlap-area formula degrades
        # to a spurious negative number for every region), so this test
        # alone doesn't distinguish pre-fix from post-fix — it guards that
        # the "truly nothing covers this" case keeps failing loudly after
        # the fix too. The next test is the one that actually fails pre-fix.
        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError, match="No Copernicus HF-radar region overlaps"):
            dl.download(135.0, -120.0, 0.0, 5.0, "2026-07-02", "2026-07-03")

    def test_crossing_bbox_downloads_the_side_that_resolves_to_a_region(self, tmp_path):
        from pathlib import Path
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_downloader import HFRadarDownloader

        # US-Alaska's bbox (-174.10..-128.66) overlaps the [-180, -120]
        # window but not the [135, 180] window, so only one window should
        # produce a download.
        dl = HFRadarDownloader(output_dir=tmp_path, dry_run=False)
        fake_module = MagicMock()

        def fake_subset(**kwargs):
            Path(kwargs["output_directory"], kwargs["output_filename"]).write_bytes(b"")

        fake_module.subset.side_effect = fake_subset
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(135.0, -120.0, 65.0, 75.0, "2026-01-01", "2026-01-02")

        assert len(out) == 1
        assert fake_module.subset.call_count == 1
        _, kwargs = fake_module.subset.call_args
        assert kwargs["minimum_longitude"] == -180.0
        assert kwargs["maximum_longitude"] == -120.0


# ---------------------------------------------------------------------------
# Tests for HFRadarHistoricalDownloader
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class FileGetResult:
    files: List[Any] = field(default_factory=list)


class TestHFRadarHistoricalDownloader:
    def test_dry_run_prints_resolved_region_and_filename(self, tmp_path, capsys):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(-90.0, -60.0, 30.0, 40.0, "2021-06-05", "2021-06-06")
        assert out == []
        captured = capsys.readouterr().out
        assert "US-EastGulfCoast" in captured
        assert "GL_TV_HF_HFR-US-EastGulfCoast_Total_2021.nc" in captured

    def test_unavailable_region_returns_empty(self, tmp_path, caplog):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        # GoS (Italy) has an NRT feed but no delayed-mode archive. This must
        # not raise — the orchestrator's NRT-fallback logic (see
        # orchestrator.py's _HISTORICAL_FIRST_PAIRS) depends on an empty
        # list, not an exception, to know it should try hf_radar instead.
        with caplog.at_level(logging.WARNING):
            out = dl.download(13.5, 15.5, 40.0, 41.0, "2021-01-01", "2021-01-02")

        assert out == []
        assert any("no delayed-mode HF-radar archive" in r.message for r in caplog.records)

    def test_multi_year_request_not_yet_supported(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(NotImplementedError, match="single calendar year"):
            dl.download(-90.0, -60.0, 30.0, 40.0, "2020-12-30", "2021-01-02")

    def test_year_outside_split_archive_range_returns_empty(self, tmp_path, caplog):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        # US-EastGulfCoast's historical archive is split into one file per
        # year, only for 2019-2024; a request for 2018 falls outside that
        # range. Must return [] (not raise) for the same reason as the
        # unavailable-region case above.
        with caplog.at_level(logging.WARNING):
            out = dl.download(-90.0, -60.0, 30.0, 40.0, "2018-01-01", "2018-01-02")

        assert out == []
        assert any(
            "No US-EastGulfCoast historical archive for year 2018" in r.message
            for r in caplog.records
        )

    def test_download_gets_file_then_subsets_locally(self, tmp_path):
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "GDOP": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            # FileGetResult is defined at module scope in this test file (see
            # the brief's note: it's a mock stand-in only, not a symbol the
            # implementation defines or imports).
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01T04:00:00")

        assert len(out) == 1
        assert out[0].exists()
        result = xr.open_dataset(out[0])
        assert "time" in result.dims and "latitude" in result.dims and "longitude" in result.dims
        assert "DEPTH" not in result.dims
        assert result.sizes["time"] == 5

    def test_archive_with_no_data_in_requested_window_returns_empty(self, tmp_path, caplog):
        """Reproduces the ARPAS report: the archive file exists and opens
        fine, but its real TIME coverage doesn't reach the requested window
        (e.g. the region's delayed-mode processing lags further behind than
        the fixed _MIN_AGE_DAYS guard assumes). Must return [] so the
        orchestrator can fall back to the NRT downloader, not raise."""
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        # Archive only covers early 2019; the request below (2021) falls
        # entirely outside this range, mirroring ARPAS's real archive
        # (covers 2022-11-11..2025-07-03) vs. a request past 2025-07-03.
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}), \
             caplog.at_level(logging.WARNING):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2021-06-05", "2021-06-06")

        assert out == []
        # self.output_dir.mkdir() already ran earlier in _download_region_window,
        # so the directory exists — it just must contain no .nc output.
        assert list((tmp_path / "out").glob("*.nc")) == []
        assert any("US-WestCoast" in r.message and "2021-06-05" in r.message for r in caplog.records)

    def test_out_of_order_timestamps_elsewhere_in_archive_do_not_break_slicing(self, tmp_path):
        """Reproduces the DeltaEbro report: a handful of out-of-order TIME
        values anywhere in the multi-year archive (e.g. from delayed QC
        reprocessing swapping two adjacent hourly readings) make pandas
        refuse *any* label-based time slice on that file -- even for a
        request nowhere near the bad points -- unless TIME is sorted
        first."""
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = list(pd.date_range("2019-01-01", periods=10, freq="1h"))
        # Swap two adjacent timestamps near the end out of order, far from
        # the requested window below -- mirrors DeltaEbro's real archive,
        # which has out-of-order timestamps in 2025 that broke a 2019
        # request.
        times[8], times[9] = times[9], times[8]
        times = pd.DatetimeIndex(times)
        shape = (10, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path / "out", dry_run=False)
        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01T00:00:00", "2019-01-01T03:00:00")

        assert len(out) == 1
        result = xr.open_dataset(out[0])
        assert result.sizes["time"] == 4


class TestHFRadarHistoricalDownloaderCheckAvailabilityDry:
    def test_true_for_recent_enough_region_year_with_known_archive(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        old_end = (datetime.now(timezone.utc) - timedelta(days=200)).strftime("%Y-%m-%d")
        old_start = (datetime.now(timezone.utc) - timedelta(days=201)).strftime("%Y-%m-%d")
        dl = HFRadarHistoricalDownloader(output_dir=tmp_path)
        assert dl.check_availability_dry(-121.0, -120.0, 33.0, 34.0, old_start, old_end) is True

    def test_false_when_end_is_too_recent(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path)
        # Well within the _MIN_AGE_DAYS=182 recency guard.
        assert dl.check_availability_dry(-121.0, -120.0, 33.0, 34.0, "2026-08-01", "2026-08-02") is False

    def test_false_when_no_region_overlaps_bbox(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path)
        assert dl.check_availability_dry(0.0, 5.0, 0.0, 5.0, "2021-01-01", "2021-01-02") is False

    def test_false_when_region_has_no_historical_archive_at_all(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path)
        # GoS (Italy) has an NRT feed but no delayed-mode archive.
        assert dl.check_availability_dry(13.5, 15.5, 40.0, 41.0, "2021-01-01", "2021-01-02") is False

    def test_false_when_split_by_year_region_year_out_of_range(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path)
        # US-EastGulfCoast's split-by-year archive only covers 2019-2024.
        assert dl.check_availability_dry(-90.0, -60.0, 30.0, 40.0, "2018-01-01", "2018-01-02") is False

    def test_multi_year_request_raises_not_implemented(self, tmp_path):
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path)
        with pytest.raises(NotImplementedError, match="single calendar year"):
            dl.check_availability_dry(-90.0, -60.0, 30.0, 40.0, "2020-12-30", "2021-01-02")


class TestHFRadarHistoricalDownloaderAntimeridian:
    def test_crossing_bbox_with_no_covering_region_on_either_side_raises(self, tmp_path):
        # lat 0-5 doesn't overlap any HFR_REGIONS entry on either side of
        # the split (the southernmost real region, US-Hawaii, starts at
        # 14.5N). Note: the unsplit pre-fix code also raises a ValueError
        # matching this message for a min_lon > max_lon input, so this test
        # alone doesn't distinguish pre-fix from post-fix — it guards that
        # the "truly nothing covers this" case keeps failing loudly after
        # the fix too. The next test is the one that actually fails pre-fix.
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        with pytest.raises(ValueError, match="No Copernicus HF-radar region overlaps"):
            dl.download(135.0, -120.0, 0.0, 5.0, "2021-07-02", "2021-07-03")

    def test_crossing_bbox_dry_run_resolves_the_side_that_has_a_region(self, tmp_path, capsys):
        # US-Alaska's bbox (-174.10..-128.66, 68.01..74.03) overlaps the
        # [-180, -120] window but not the [135, 180] window, so only that
        # window should resolve a region.
        from sar_validation.downloaders.hf_radar_historical_downloader import (
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=True)
        out = dl.download(135.0, -120.0, 69.0, 73.0, "2021-07-02", "2021-07-03")
        assert out == []
        assert "US-Alaska" in capsys.readouterr().out


class TestHFRadarHistoricalDownloaderForceDownload:
    def test_skips_when_output_already_exists(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            DATASET_ID,
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False)
        dest_path = tmp_path / f"{DATASET_ID}_US-WestCoast_2019-01-01.nc"
        dest_path.write_bytes(b"")

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            out = dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01")

        fake_module.get.assert_not_called()
        assert out == [dest_path]

    def test_force_download_refetches_existing_output(self, tmp_path):
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.downloaders.hf_radar_historical_downloader import (
            DATASET_ID,
            HFRadarHistoricalDownloader,
        )

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False, force_download=True)
        dest_path = tmp_path / f"{DATASET_ID}_US-WestCoast_2019-01-01.nc"
        dest_path.write_bytes(b"stale")

        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "GL_TV_HF_HFR-US-WestCoast_Total.nc"
        times = pd.date_range("2019-01-01", periods=5, freq="1h")
        shape = (5, 1, 2, 2)
        ds = xr.Dataset(
            {
                "EWCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
                "NSCT": (("TIME", "DEPTH", "LATITUDE", "LONGITUDE"), np.random.rand(*shape)),
            },
            coords={
                "TIME": times, "DEPTH": [0.0],
                "LATITUDE": [33.0, 34.0], "LONGITUDE": [-121.0, -120.0],
            },
        )
        ds.to_netcdf(raw_path)

        fake_module = MagicMock()

        def fake_get(**kwargs):
            return FileGetResult(files=[type("F", (), {"file_path": raw_path})()])

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            dl.download(-121.0, -120.0, 33.0, 34.0, "2019-01-01", "2019-01-01T04:00:00")

        fake_module.get.assert_called_once()

    def test_raw_archive_fetched_into_shared_cache_dir_not_per_run_output_dir(
        self, tmp_path, monkeypatch
    ):
        """The raw multi-year archive (100s of MB) must be fetched into a
        fixed, run-independent cache directory, not tmp_path/output_dir/
        _raw_archive — otherwise every dated run folder re-downloads the
        same file. monkeypatch the module's cache-dir constant so the test
        doesn't touch the real repo-relative data/_archive_cache/ path."""
        from unittest.mock import patch

        import sar_validation.downloaders.hf_radar_historical_downloader as hf_hist_mod

        shared_cache = tmp_path / "shared_cache"
        monkeypatch.setattr(hf_hist_mod, "_ARCHIVE_CACHE_DIR", shared_cache)

        per_run_output = tmp_path / "run1" / "hf_radar_historical"
        dl = hf_hist_mod.HFRadarHistoricalDownloader(output_dir=per_run_output, dry_run=False)

        fake_module = MagicMock()

        def fake_get(**kwargs):
            raise FileNotFoundError("no archive file matched (test stub, not exercised further)")

        fake_module.get.side_effect = fake_get
        with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
            with pytest.raises(FileNotFoundError):
                dl.download(-90.0, -60.0, 30.0, 40.0, "2021-06-05", "2021-06-06")

        get_kwargs = fake_module.get.call_args.kwargs
        assert get_kwargs["output_directory"] == str(shared_cache)
        assert shared_cache.exists()
        assert not (per_run_output / "_raw_archive").exists()


class TestOrchestratorHFRadarHistoricalWiring:
    def test_dispatch_source_registers_hf_radar_historical_handler(self):
        from unittest.mock import patch

        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="currents"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type="hf_radar_historical")

        with patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = None
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        mock_cls.assert_called_once()


class TestOrchestratorScatterometerFTPWiring:
    @pytest.mark.parametrize("source_type,satellite", [
        ("scatterometer_hy2b", "hy2b"),
        ("scatterometer_hy2c", "hy2c"),
        ("scatterometer_oceansat3", "oceansat3"),
    ])
    def test_dispatch_source_registers_handler_with_right_satellite(self, source_type, satellite):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="wind"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type=source_type)

        with patch(
            "sar_validation.downloaders.scatterometer_ftp_downloader.ScatterometerFTPDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        assert mock_cls.call_args.kwargs["satellite"] == satellite


class TestOrchestratorGPortalAmsrFallback:
    def test_gportal_not_tried_when_earthdata_returns_files(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="amsr_ssm")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = [tmp_path / "file1.h5"]
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        mock_gportal_cls.assert_not_called()

    def test_gportal_tried_when_earthdata_returns_zero_files(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="amsr_ssm")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = []
            mock_gportal_cls.return_value.download.return_value = [tmp_path / "gportal_file.h5"]
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        mock_gportal_cls.assert_called_once()
        assert orchestrator.metadata["downloads"]["amsr_ssm"]["files"] == [str(tmp_path / "gportal_file.h5")]
        assert orchestrator.metadata["downloads"]["amsr_ssm"]["gportal_fallback"] is True

    def test_gportal_fallback_constructed_with_allow_prompt_false(self, tmp_path):
        """The automatic fallback path must never block an unattended
        pipeline run on an interactive G-Portal password prompt -- it must
        always construct GPortalAMSR2Downloader with allow_prompt=False,
        regardless of what credentials happen to be configured."""
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="amsr_ssm")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = []
            mock_gportal_cls.return_value.download.return_value = []
            orchestrator._dispatch_source(source)

        mock_gportal_cls.assert_called_once()
        assert mock_gportal_cls.call_args.kwargs["allow_prompt"] is False

    def test_gportal_failure_logged_as_notice_not_error(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="amsr_ssm")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = []
            mock_gportal_cls.return_value.download.side_effect = RuntimeError("no credentials")
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        assert any("no credentials" in n for n in orchestrator.metadata["notices"])
        assert orchestrator.metadata["errors"] == []

    def test_no_failure_notice_when_files_already_exist_from_a_previous_run(self, tmp_path):
        """A fresh G-Portal reconnect can fail on a purely transient
        connection blip (see _connect_with_retry) even though real AMSR2
        files from an earlier successful run already sit in amsr_ssm/ --
        the pipeline still has good data to use, so this must not be
        reported as a failure in the report. Confirmed against a real
        recipes/soil_moisture_cds_nisar_test.yaml report that showed
        "G-Portal AMSR2 fallback failed" despite 4 real .h5 files already
        present in that exact run's amsr_ssm/ folder."""
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        orchestrator.base_dir = tmp_path
        source = ValidationDataSource(source_type="amsr_ssm")

        amsr_dir = tmp_path / "amsr_ssm"
        amsr_dir.mkdir()
        existing_file = amsr_dir / "GW1AM2_20260709_01D_EQMA_L3SGSMCHF3300300.h5"
        existing_file.write_bytes(b"fake")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = []
            mock_gportal_cls.return_value.download.side_effect = RuntimeError(
                "Error reading SSH protocol banner"
            )
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        assert not any("failed" in n for n in orchestrator.metadata["notices"])
        assert orchestrator.metadata["downloads"]["amsr_ssm"]["files"] == [str(existing_file)]
        assert orchestrator.metadata["downloads"]["amsr_ssm"]["status"] == "success"

    def test_no_coverage_cutoff_notice_when_gportal_fallback_finds_files(self, tmp_path):
        """The 'past this source's known coverage cutoff' notice must not
        fire just because NASA Earthdata alone returned 0 files -- only
        when AMSR2 truly has no data from *either* source. Confirmed
        against a real recipe run where this notice appeared even though
        G-Portal successfully downloaded real AMSR2 files."""
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="amsr_ssm")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = []
            mock_gportal_cls.return_value.download.return_value = [tmp_path / "gportal_file.h5"]
            orchestrator._dispatch_source(source)

        assert not any("coverage cutoff" in n for n in orchestrator.metadata["notices"])

    def test_coverage_cutoff_notice_when_both_sources_find_nothing(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="soil_moisture"))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="amsr_ssm")

        with patch(
            "sar_validation.downloaders.earthdata_soil_moisture_downloader.EarthdataSoilMoistureDownloader"
        ) as mock_earthdata_cls, patch(
            "sar_validation.downloaders.gportal_downloader.GPortalAMSR2Downloader"
        ) as mock_gportal_cls:
            mock_earthdata_cls.return_value.download.return_value = []
            mock_gportal_cls.return_value.download.return_value = []
            orchestrator._dispatch_source(source)

        assert any("coverage cutoff" in n for n in orchestrator.metadata["notices"])


class TestOrchestratorAscatCoverageCutoffNotice:
    """EUMETSAT's ASCAT NRT dissemination access this toolbox relies on
    stopped being populated for recent dates as of 2025-07-15 -- a request
    ending after that date returning 0 products is expected, not an error,
    and should say so (mirroring the equivalent AMSR2 coverage-cutoff
    notice) rather than silently logging "Found 0 ASCAT SSM products."
    with no explanation.

    Since the H-SAF NRT waterfall, EUMDAC is no longer
    queried for dates past its cutoff -- HSAFDownloader serves the rolling
    last-60-days window instead -- so these tests use recent dates and
    mock HSAFDownloader, not ASCATSoilMoistureDownloader."""

    def test_notice_when_zero_files_past_cutoff(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=5)).isoformat()
        end = today.isoformat()
        recipe = Recipe(RecipeConfig(
            name="test", variable="soil_moisture",
            temporal_bounds=TemporalBounds(start, end),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="ascat_ssm")

        with patch(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._dispatch_source(source)

        assert any(
            "ASCAT" in n and "coverage cutoff" in n for n in orchestrator.metadata["notices"]
        )

    def test_no_notice_when_files_found(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            Recipe,
            RecipeConfig,
            TemporalBounds,
            ValidationDataSource,
        )

        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=5)).isoformat()
        end = today.isoformat()
        recipe = Recipe(RecipeConfig(
            name="test", variable="soil_moisture",
            temporal_bounds=TemporalBounds(start, end),
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=False)
        source = ValidationDataSource(source_type="ascat_ssm")

        with patch(
            "sar_validation.downloaders.hsaf_downloader.HSAFDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = [tmp_path / "ascat_file.nc"]
            orchestrator._dispatch_source(source)

        assert not any("coverage cutoff" in n for n in orchestrator.metadata["notices"])


class TestOrchestratorCurrentsHistoricalWiring:
    @pytest.mark.parametrize("source_type,instrument", [
        ("adcp_historical", "adcp"),
        ("argo_historical", "argo"),
        ("drifter_historical", "drifter"),
        ("glider_historical", "glider"),
    ])
    def test_dispatch_source_registers_handler_with_right_instrument(self, source_type, instrument):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(name="test", variable="currents"))
        orchestrator = DataOrchestrator(recipe, dry_run=True)
        source = ValidationDataSource(source_type=source_type)

        with patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_cls:
            mock_cls.return_value.download.return_value = []
            ok = orchestrator._dispatch_source(source)

        assert ok is True
        assert mock_cls.call_args.kwargs["instrument"] == instrument


class TestOrchestratorHistoricalFirstDedup:
    def test_hf_radar_skipped_when_historical_covers_the_window(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-hfradar-skip",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="hf_radar"),
                ValidationDataSource(source_type="hf_radar_historical"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_nrt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_hist_cls.return_value.download.return_value = [tmp_path / "one.nc"]
            ok = orchestrator.download_all()

        assert ok is True
        mock_hist_cls.return_value.download.assert_called_once()
        mock_nrt_cls.return_value.download.assert_not_called()
        assert orchestrator.metadata["downloads"]["hf_radar"]["status"] == "skipped"
        assert orchestrator.metadata["downloads"]["hf_radar_historical"]["file_count"] == 1

    def test_hf_radar_dispatched_when_historical_returns_empty(self, tmp_path):
        """Also covers the ARPAS report: historical resolving a region but
        finding no data for this window (recency guard, archive-coverage
        gap, or unmapped region) must still let NRT fill in."""
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-hfradar-fallback",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="hf_radar"),
                ValidationDataSource(source_type="hf_radar_historical"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.hf_radar_historical_downloader.HFRadarHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_nrt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_hist_cls.return_value.download.return_value = []
            mock_nrt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        mock_nrt_cls.return_value.download.assert_called_once()
        assert orchestrator.metadata["downloads"]["hf_radar"]["status"] == "dry_run"
        assert orchestrator.metadata["downloads"]["hf_radar_historical"]["file_count"] == 0

    def test_hf_radar_alone_unaffected_by_dedup_logic(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-no-pair",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[ValidationDataSource(source_type="hf_radar")],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.hf_radar_downloader.HFRadarDownloader"
        ) as mock_nrt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_nrt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        mock_nrt_cls.return_value.download.assert_called_once()
        assert orchestrator.metadata["downloads"]["hf_radar"]["status"] == "dry_run"

    def test_drifter_excluded_from_nrt_batch_when_historical_covers_it(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-drifter-skip",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="drifter_historical"),
                ValidationDataSource(source_type="mooring"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_hist_cls.return_value.download.return_value = [tmp_path / "drifter.csv"]
            mock_insitu_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        assert orchestrator.metadata["downloads"]["insitu"]["source_types"] == ["mooring"]

    def test_drifter_kept_in_nrt_batch_when_historical_returns_empty(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-drifter-fallback",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="drifter_historical"),
                ValidationDataSource(source_type="mooring"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_hist_cls.return_value.download.return_value = []
            mock_insitu_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        source_types = orchestrator.metadata["downloads"]["insitu"]["source_types"]
        assert sorted(source_types) == ["drifter", "mooring"]

    def test_insitu_batch_fully_skipped_when_only_drifter_and_historical_covers_it(
        self, tmp_path
    ):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-insitu-full-skip",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="drifter_historical"),
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_hist_cls.return_value.download.return_value = [tmp_path / "drifter.csv"]
            ok = orchestrator.download_all()

        assert ok is True
        mock_insitu_cls.assert_not_called()
        assert "insitu" not in orchestrator.metadata["downloads"]

    def test_insitu_batch_depth_window_ignores_excluded_drifter(self, tmp_path):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig, ValidationDataSource

        recipe = Recipe(RecipeConfig(
            name="test-dedup-depth-window",
            variable="currents",
            output_dir=str(tmp_path),
            validation_sources=[
                ValidationDataSource(source_type="drifter", min_depth=-500.0, max_depth=500.0),
                ValidationDataSource(source_type="drifter_historical"),
                ValidationDataSource(source_type="mooring"),  # default -20/20
            ],
        ))
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_currents_historical_downloader."
            "InSituCurrentsHistoricalDownloader"
        ) as mock_hist_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_hist_cls.return_value.download.return_value = [tmp_path / "drifter.csv"]
            mock_insitu_cls.return_value.download.return_value = []
            orchestrator.download_all()

        # drifter's -500/500 depth override must not widen the NRT batch's
        # depth window, since drifter itself was excluded from that batch.
        _, insitu_ctor_kwargs = mock_insitu_cls.call_args
        assert insitu_ctor_kwargs["min_depth"] == -20.0
        assert insitu_ctor_kwargs["max_depth"] == 20.0


# ---------------------------------------------------------------------------
# End-to-end: orchestrator wiring for a Pacific-crossing recipe
# ---------------------------------------------------------------------------

class TestOrchestratorAntimeridianDryRun:
    def test_pacific_crossing_recipe_wires_through_without_error(self, tmp_path):
        from unittest.mock import patch

        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            SARDataSpec,
            TemporalBounds,
            ValidationDataSource,
        )

        cfg = RecipeConfig(
            name="pacific_dry_run_test",
            variable="waves",
            geographic_bounds=GeographicBounds(min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0),
            temporal_bounds=TemporalBounds(start="2026-07-02", end="2026-07-03"),
            sar_data=SARDataSpec(swath_mode=["WV", "SM"]),
            validation_sources=[
                ValidationDataSource(source_type="mooring"),
                ValidationDataSource(source_type="tidal_gauge"),
                ValidationDataSource(source_type="drifter"),
                ValidationDataSource(source_type="altimeter"),
            ],
            output_dir=str(tmp_path),
        )
        recipe = Recipe(cfg)
        orchestrator = DataOrchestrator(recipe, dry_run=True)

        with patch(
            "sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader"
        ) as mock_sar_cls, patch(
            "sar_validation.downloaders.insitu_downloader.InSituDownloader"
        ) as mock_insitu_cls, patch(
            "sar_validation.downloaders.altimeter_downloader.AltimeterDownloader"
        ) as mock_alt_cls:
            mock_sar_cls.return_value.download.return_value = []
            mock_sar_cls.return_value.found_count = 1
            mock_insitu_cls.return_value.download.return_value = []
            mock_alt_cls.return_value.download.return_value = []
            ok = orchestrator.download_all()

        assert ok is True
        _, sar_kwargs = mock_sar_cls.return_value.download.call_args
        assert (sar_kwargs["min_lon"], sar_kwargs["max_lon"]) == (135.0, -120.0)
        _, insitu_kwargs = mock_insitu_cls.return_value.download.call_args
        assert (insitu_kwargs["min_lon"], insitu_kwargs["max_lon"]) == (135.0, -120.0)
        _, alt_kwargs = mock_alt_cls.return_value.download.call_args
        assert (alt_kwargs["min_lon"], alt_kwargs["max_lon"]) == (135.0, -120.0)

    def test_waves_pacific_recipe_loads_with_crossing_convention(self):
        from sar_validation.core.recipe import Recipe

        recipe = Recipe.from_yaml("recipes/waves_pacific.yaml")
        bounds = recipe.config.geographic_bounds
        assert bounds.min_lon == 135.0
        assert bounds.max_lon == -120.0
        assert bounds.min_lon > bounds.max_lon  # crossing convention


# ---------------------------------------------------------------------------
# DataOrchestrator force_download wiring
# ---------------------------------------------------------------------------

class TestOrchestratorForceDownloadWiring:
    def _make_orchestrator(self, tmp_path, force_download):
        from sar_validation.core.orchestrator import DataOrchestrator
        from sar_validation.core.recipe import Recipe, RecipeConfig

        recipe = Recipe(RecipeConfig(
            name="test-force-download",
            variable="wind",
            output_dir=str(tmp_path),
        ))
        return DataOrchestrator(recipe, dry_run=True, force_download=force_download)

    def test_default_force_download_is_false(self, tmp_path):
        from unittest.mock import patch

        orchestrator = self._make_orchestrator(tmp_path, force_download=False)
        with patch("sar_validation.downloaders.sentinel1_l2_ocn_downloader.SARDownloader") as mock_cls:
            mock_cls.return_value.download.return_value = []
            orchestrator._download_sar()
        assert mock_cls.call_args.kwargs["force_download"] is False

# ---------------------------------------------------------------------------
# EarthdataSoilMoistureDownloader
# ---------------------------------------------------------------------------

class TestEarthdataSoilMoistureDownloader:
    def test_dry_run_lists_found_granules_and_does_not_download(self, tmp_path, monkeypatch, capsys):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset="NISAR_L3_PR_SME2_BETA", output_dir=tmp_path, dry_run=True,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        granule = MagicMock()
        granule.data_links.return_value = ["https://example.com/NISAR_L3_PR_SME2_20260301.h5"]
        granule.size.return_value = 42.5

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = [granule]

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            out = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-02",
            )

        assert out == []
        fake_earthaccess.login.assert_called_once()
        fake_earthaccess.search_data.assert_called_once_with(
            short_name="NISAR_L3_PR_SME2_BETA",
            version=None,
            bounding_box=(-10.0, 40.0, 10.0, 55.0),
            temporal=("2026-07-01T00:00:00", "2026-07-02T00:00:00"),
        )
        fake_earthaccess.download.assert_not_called()
        captured = capsys.readouterr().out
        assert "Found 1 NISAR_L3_PR_SME2_BETA granule(s)" in captured
        assert "NISAR_L3_PR_SME2_20260301.h5" in captured
        assert "42.5 MB" in captured
        assert "DRY RUN" in captured

    def test_dry_run_reports_zero_granules_found(self, tmp_path, monkeypatch, capsys):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset="NSIDC-0451", output_dir=tmp_path, dry_run=True,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = []

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            out = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-02",
            )

        assert out == []
        fake_earthaccess.download.assert_not_called()
        captured = capsys.readouterr().out
        assert "Found 0 NSIDC-0451 granule(s)" in captured

    def test_multiple_candidates_query_and_merge_all(self, tmp_path, monkeypatch, capsys):
        """A list of (short_name, version) candidates -- e.g. NISAR SME2's
        beta/provisional product-maturity transition, where the underlying
        collection changed partway through the mission with no temporal
        overlap -- must all be queried and their granules merged into one
        combined result, not just the first candidate."""
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset=[("NISAR_L3_SME2_BETA_V1", "1"), ("NISAR_L3_SME2_PROVISIONAL_V1", "1")],
            output_dir=tmp_path,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        fake_earthaccess = MagicMock()

        def fake_search_data(short_name, **kwargs):
            return {"NISAR_L3_SME2_BETA_V1": [], "NISAR_L3_SME2_PROVISIONAL_V1": ["granuleA", "granuleB"]}[short_name]

        fake_earthaccess.search_data.side_effect = fake_search_data
        fake_earthaccess.download.return_value = [
            str(tmp_path / "fileA.h5"), str(tmp_path / "fileB.h5"),
        ]

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            result = dl.download(
                min_lon=0.0, max_lon=30.0, min_lat=55.0, max_lat=70.0,
                start="2026-06-17", end="2026-06-18",
            )

        assert fake_earthaccess.search_data.call_count == 2
        fake_earthaccess.search_data.assert_any_call(
            short_name="NISAR_L3_SME2_BETA_V1", version="1",
            bounding_box=(0.0, 55.0, 30.0, 70.0),
            temporal=("2026-06-17T00:00:00", "2026-06-18T00:00:00"),
        )
        fake_earthaccess.search_data.assert_any_call(
            short_name="NISAR_L3_SME2_PROVISIONAL_V1", version="1",
            bounding_box=(0.0, 55.0, 30.0, 70.0),
            temporal=("2026-06-17T00:00:00", "2026-06-18T00:00:00"),
        )
        # earthaccess.download() must be called with the MERGED granule
        # list from both candidates, not just one.
        fake_earthaccess.download.assert_called_once_with(
            ["granuleA", "granuleB"], str(tmp_path),
        )
        assert result == [tmp_path / "fileA.h5", tmp_path / "fileB.h5"]
        captured = capsys.readouterr().out
        assert "Found 0 NISAR_L3_SME2_BETA_V1 granule(s)" in captured
        assert "Found 2 NISAR_L3_SME2_PROVISIONAL_V1 granule(s)" in captured

    def test_multiple_candidates_dry_run_lists_all_without_downloading(self, tmp_path, monkeypatch, capsys):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset=[("NISAR_L3_SME2_BETA_V1", "1"), ("NISAR_L3_SME2_PROVISIONAL_V1", "1")],
            output_dir=tmp_path, dry_run=True,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        fake_earthaccess = MagicMock()

        def fake_search_data(short_name, **kwargs):
            return {"NISAR_L3_SME2_BETA_V1": [], "NISAR_L3_SME2_PROVISIONAL_V1": ["granuleA"]}[short_name]

        fake_earthaccess.search_data.side_effect = fake_search_data

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            result = dl.download(
                min_lon=0.0, max_lon=30.0, min_lat=55.0, max_lat=70.0,
                start="2026-06-17", end="2026-06-18",
            )

        assert result == []
        fake_earthaccess.download.assert_not_called()
        captured = capsys.readouterr().out
        assert "Found 0 NISAR_L3_SME2_BETA_V1 granule(s)" in captured
        assert "Found 1 NISAR_L3_SME2_PROVISIONAL_V1 granule(s)" in captured
        assert "[DRY RUN] Would download 1 granule(s)" in captured

    def test_search_and_download_amsr(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(dataset="NSIDC-0451", output_dir=tmp_path)
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = ["granule1", "granule2"]
        fake_earthaccess.download.return_value = [
            str(tmp_path / "file1.h5"), str(tmp_path / "file2.h5"),
        ]

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-02",
            )

        fake_earthaccess.login.assert_called_once()
        fake_earthaccess.search_data.assert_called_once_with(
            short_name="NSIDC-0451",
            version=None,
            bounding_box=(-10.0, 40.0, 10.0, 55.0),
            temporal=("2026-07-01T00:00:00", "2026-07-02T00:00:00"),
        )
        assert result == [tmp_path / "file1.h5", tmp_path / "file2.h5"]

    def test_search_with_version_for_smap(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset="SPL2SMP_E", version="006", output_dir=tmp_path,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = []

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            result = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-02",
            )

        assert result == []
        fake_earthaccess.search_data.assert_called_once_with(
            short_name="SPL2SMP_E",
            version="006",
            bounding_box=(-10.0, 40.0, 10.0, 55.0),
            temporal=("2026-07-01T00:00:00", "2026-07-02T00:00:00"),
        )
        fake_earthaccess.download.assert_not_called()

    def test_found_count_set_from_search_even_in_dry_run(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset="NISAR_L3_PR_SME2_BETA", output_dir=tmp_path, dry_run=True,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        granule = MagicMock()
        granule.data_links.return_value = ["https://example.com/NISAR_L3_PR_SME2_20260301.h5"]
        granule.size.return_value = 42.5

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = [granule]

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            out = dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-02",
            )

        assert out == []
        assert dl.found_count == 1

    def test_found_count_zero_when_search_returns_nothing(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from sar_validation.downloaders.earthdata_soil_moisture_downloader import (
            EarthdataSoilMoistureDownloader,
        )

        dl = EarthdataSoilMoistureDownloader(
            dataset="NSIDC-0451", output_dir=tmp_path, dry_run=True,
        )
        monkeypatch.setenv("EARTHDATA_USERNAME", "test_user")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "test_pass")

        fake_earthaccess = MagicMock()
        fake_earthaccess.search_data.return_value = []

        with patch.dict("sys.modules", {"earthaccess": fake_earthaccess}):
            dl.download(
                min_lon=-10.0, max_lon=10.0, min_lat=40.0, max_lat=55.0,
                start="2026-07-01", end="2026-07-02",
            )

        assert dl.found_count == 0

# ---------------------------------------------------------------------------
# HFRadarUSDownloader
# ---------------------------------------------------------------------------

class TestHFRadarUSDownloaderWaterfall:
    def test_erddap_success_short_circuits_thredds_and_copernicus(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.download.return_value = [tmp_path / "a.nc"]
            result = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result == [tmp_path / "a.nc"]
        assert dl.resolved_backend == "erddap"
        assert dl.attempted_backends == ["erddap"]
        m_thredds.assert_not_called()
        m_cop.assert_not_called()
        m_cop_hist.assert_not_called()

    def test_erddap_not_implemented_falls_through_to_thredds(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        old = (datetime.now(timezone.utc) - timedelta(days=200)).strftime("%Y-%m-%d")
        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.download.side_effect = NotImplementedError("too old")
            m_thredds.return_value.download.return_value = [tmp_path / "b.nc"]
            result = dl.download(-125, -119, 33, 38, old, old)

        assert result == [tmp_path / "b.nc"]
        assert dl.resolved_backend == "thredds"
        assert dl.attempted_backends == ["erddap", "thredds"]
        m_cop.assert_not_called()
        m_cop_hist.assert_not_called()

    def test_erddap_and_thredds_empty_falls_through_to_copernicus(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        # dry_run=False here (unlike the rest of this class): this scenario
        # simulates a REAL run where ERDDAP/THREDDS genuinely found no data
        # and Copernicus genuinely did, not a dry-run preview -- under
        # dry_run=True, resolved_backend would report "erddap" (the first
        # backend that structurally applies) rather than reflecting which
        # backend actually produced files.
        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=False)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.download.return_value = []
            m_thredds.return_value.download.return_value = []
            m_cop_hist.return_value.download.return_value = []
            m_cop.return_value.download.return_value = [tmp_path / "c.nc"]
            result = dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result == [tmp_path / "c.nc"]
        assert dl.resolved_backend == "copernicus"
        assert dl.attempted_backends == ["erddap", "thredds", "copernicus"]

    def test_great_lakes_region_skips_erddap_entirely(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds:
            m_thredds.return_value.download.return_value = [tmp_path / "d.nc"]
            # US_GREAT_LAKES bbox center: lon ~-84.8, lat ~45.8
            result = dl.download(-85.3, -84.2, 45.6, 46.05, "2024-01-31", "2024-01-31")

        assert result == [tmp_path / "d.nc"]
        m_erddap.assert_not_called()
        assert dl.attempted_backends == ["thredds"]

    def test_dry_run_thredds_only_region_reports_thredds_not_copernicus(self, tmp_path):
        """US_GREAT_LAKES has no ERDDAP dataset at all, so under a dry run
        THREDDS is the first (and only NOAA) backend that structurally
        applies -- resolved_backend must report "thredds", not fall through
        to the terminal "copernicus" default the way it did before this
        fix."""
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_thredds.return_value.download.return_value = []  # dry-run contract: [] always
            m_cop_hist.return_value.download.return_value = []
            m_cop.return_value.download.return_value = []
            # US_GREAT_LAKES bbox center: lon ~-84.8, lat ~45.8
            dl.download(-85.3, -84.2, 45.6, 46.05, "2024-01-31", "2024-01-31")

        m_erddap.assert_not_called()
        assert dl.resolved_backend == "thredds"
        assert dl.attempted_backends == ["thredds", "copernicus"]

    def test_non_us_bbox_skips_erddap_and_thredds_entirely(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_cop_hist.return_value.download.return_value = []
            m_cop.return_value.download.return_value = []
            dl.download(2.0, 8.0, 53.0, 55.0, _RECENT_START, _RECENT_END)

        m_erddap.assert_not_called()
        m_thredds.assert_not_called()
        assert dl.attempted_backends == ["copernicus"]

    def test_unexpected_exception_from_erddap_is_not_caught(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap:
            m_erddap.return_value.download.side_effect = RuntimeError("network exploded")
            with pytest.raises(RuntimeError, match="network exploded"):
                dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

    def test_resolution_km_forwarded_to_erddap_and_thredds(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True, resolution_km=1)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds:
            m_erddap.return_value.download.return_value = []
            m_thredds.return_value.download.return_value = [tmp_path / "e.nc"]
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert m_erddap.call_args.kwargs["resolution_km"] == 1
        assert m_thredds.call_args.kwargs["resolution_km"] == 1

    def test_finest_resolves_per_region(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True, resolution_km="finest")
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap:
            m_erddap.return_value.download.return_value = [tmp_path / "f.nc"]
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)  # US_WEST -> finest 0.5

        assert m_erddap.call_args.kwargs["resolution_km"] == 0.5

    def test_no_override_uses_region_default(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap:
            m_erddap.return_value.download.return_value = [tmp_path / "g.nc"]
            dl.download(-159.0, -154.0, 19.0, 22.0, _RECENT_START, _RECENT_END)  # US_HAWAII -> 1

        assert m_erddap.call_args.kwargs["resolution_km"] == 1

    def test_warns_when_stale_other_backend_output_exists(self, tmp_path, caplog):
        import logging as _logging

        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        stale_dir = tmp_path / "hf_radar"
        stale_dir.mkdir(parents=True)
        (stale_dir / "stale.nc").write_bytes(b"")

        dl = HFRadarUSDownloader(output_dir=tmp_path, dry_run=True)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, caplog.at_level(_logging.WARNING):
            m_erddap.return_value.download.return_value = [tmp_path / "h.nc"]
            dl.download(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert any("already contains cached" in rec.message for rec in caplog.records)


class TestHFRadarUSDownloaderCheckAvailabilityDry:
    """check_availability_dry mirrors download()'s own waterfall try-order
    exactly, but delegates to the four wrapped downloaders' own
    check_availability_dry methods instead of re-deriving any
    region-resolution/dataset-selection logic."""

    def test_erddap_true_short_circuits_thredds_and_copernicus(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.check_availability_dry.return_value = True
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is True
        m_thredds.return_value.check_availability_dry.assert_not_called()
        m_cop.return_value.check_availability_dry.assert_not_called()
        m_cop_hist.return_value.check_availability_dry.assert_not_called()

    def test_erddap_false_falls_through_to_thredds(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.check_availability_dry.return_value = False
            m_thredds.return_value.check_availability_dry.return_value = True
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is True
        m_cop.return_value.check_availability_dry.assert_not_called()
        m_cop_hist.return_value.check_availability_dry.assert_not_called()

    def test_erddap_and_thredds_false_falls_through_to_copernicus_historical_then_nrt(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.check_availability_dry.return_value = False
            m_thredds.return_value.check_availability_dry.return_value = False
            m_cop_hist.return_value.check_availability_dry.return_value = False
            m_cop.return_value.check_availability_dry.return_value = True
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is True
        m_cop_hist.return_value.check_availability_dry.assert_called_once()
        m_cop.return_value.check_availability_dry.assert_called_once()

    def test_historical_true_short_circuits_nrt(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.check_availability_dry.return_value = False
            m_thredds.return_value.check_availability_dry.return_value = False
            m_cop_hist.return_value.check_availability_dry.return_value = True
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is True
        m_cop.return_value.check_availability_dry.assert_not_called()

    def test_non_us_bbox_skips_erddap_and_thredds_entirely(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_cop_hist.return_value.check_availability_dry.return_value = False
            m_cop.return_value.check_availability_dry.return_value = False
            dl.check_availability_dry(2.0, 8.0, 53.0, 55.0, _RECENT_START, _RECENT_END)

        m_erddap.return_value.check_availability_dry.assert_not_called()
        m_thredds.return_value.check_availability_dry.assert_not_called()

    def test_great_lakes_region_skips_erddap_entirely(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds:
            m_thredds.return_value.check_availability_dry.return_value = True
            # US_GREAT_LAKES bbox center: lon ~-84.8, lat ~45.8
            result = dl.check_availability_dry(-85.3, -84.2, 45.6, 46.05, "2024-01-31", "2024-01-31")

        assert result is True
        m_erddap.return_value.check_availability_dry.assert_not_called()

    def test_false_when_no_backend_finds_anything(self, tmp_path):
        from sar_validation.downloaders.hf_radar_us_downloader import HFRadarUSDownloader

        dl = HFRadarUSDownloader(output_dir=tmp_path)
        with patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAAHFRadarDownloader"
        ) as m_erddap, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.NOAATHREDDSHFRadarDownloader"
        ) as m_thredds, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarDownloader"
        ) as m_cop, patch(
            "sar_validation.downloaders.hf_radar_us_downloader.HFRadarHistoricalDownloader"
        ) as m_cop_hist:
            m_erddap.return_value.check_availability_dry.return_value = False
            m_thredds.return_value.check_availability_dry.return_value = False
            m_cop_hist.return_value.check_availability_dry.return_value = False
            m_cop.return_value.check_availability_dry.return_value = False
            result = dl.check_availability_dry(-125, -119, 33, 38, _RECENT_START, _RECENT_END)

        assert result is False
