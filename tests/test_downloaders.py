"""Tests for downloader utilities: datetime parsing and dataset_part selection."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from sar_validation.downloaders.base import normalize_datetime, is_date_recent
from sar_validation.downloaders.insitu_downloader import (
    SOURCE_TYPE_TO_PLATFORM,
    PLATFORM_CODE_TO_SOURCE_TYPE,
    _resolve_platform_codes,
)


# ---------------------------------------------------------------------------
# Tests for normalize_datetime()
# ---------------------------------------------------------------------------

class TestNormalizeDatetime:
    """Test datetime normalization with various input formats."""

    def test_date_only(self):
        """Date-only input should be normalized with T00:00:00."""
        result = normalize_datetime("2026-01-01")
        assert result == "2026-01-01T00:00:00"

    def test_date_with_iso_time(self):
        """ISO format with T separator should pass through (roughly)."""
        result = normalize_datetime("2026-01-01T12:34:56")
        assert result == "2026-01-01T12:34:56"

    def test_date_with_space_separator(self):
        """Space-separated datetime should convert space to T."""
        result = normalize_datetime("2026-01-01 12:34:56")
        assert result == "2026-01-01T12:34:56"

    def test_hhmmss_format_no_colons(self):
        """6-digit time without colons should be converted to HH:MM:SS."""
        result = normalize_datetime("2026-06-24 000000")
        assert result == "2026-06-24T00:00:00"

    def test_hhmmss_format_midnight(self):
        """HHMMSS midnight should convert correctly."""
        result = normalize_datetime("2026-06-24 000000")
        assert result == "2026-06-24T00:00:00"

    def test_hhmmss_format_midday(self):
        """HHMMSS midday should convert correctly."""
        result = normalize_datetime("2026-06-24 120000")
        assert result == "2026-06-24T12:00:00"

    def test_hhmmss_format_near_end_of_day(self):
        """HHMMSS near end of day should convert correctly."""
        result = normalize_datetime("2026-06-24 235959")
        assert result == "2026-06-24T23:59:59"

    def test_hhmmss_with_three_hour_offset(self):
        """HHMMSS from the original error case should work."""
        result = normalize_datetime("2026-06-24 030000")
        assert result == "2026-06-24T03:00:00"

    def test_trailing_z_removed(self):
        """Trailing Z should be stripped."""
        result = normalize_datetime("2026-01-01T12:34:56Z")
        assert result == "2026-01-01T12:34:56"

    def test_milliseconds_removed(self):
        """Milliseconds should be removed."""
        result = normalize_datetime("2026-01-01T12:34:56.123Z")
        assert result == "2026-01-01T12:34:56"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        result = normalize_datetime("  2026-01-01  ")
        assert result == "2026-01-01T00:00:00"

    def test_space_and_hhmmss(self):
        """Space-separated HHMMSS should convert both."""
        result = normalize_datetime("2026-01-01 120000")
        assert result == "2026-01-01T12:00:00"

    def test_iso_format_already_correct(self):
        """Already-correct ISO format should pass through."""
        result = normalize_datetime("2026-01-01T12:34:56")
        assert result == "2026-01-01T12:34:56"

    def test_iso_hhmmss_with_z(self):
        """ISO format HHMMSS with Z should be handled."""
        result = normalize_datetime("2026-01-01T120000Z")
        assert result == "2026-01-01T12:00:00"


# ---------------------------------------------------------------------------
# Tests for is_date_recent()
# ---------------------------------------------------------------------------

class TestIsDateRecent:
    """Test detection of recent dates."""

    @patch("sar_validation.downloaders.base.datetime")
    def test_today_is_recent(self, mock_datetime):
        """Today should be considered recent (0 days ago)."""
        today = datetime(2026, 7, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-07-02", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_yesterday_is_recent(self, mock_datetime):
        """Yesterday should be considered recent (1 day ago)."""
        today = datetime(2026, 7, 2)
        yesterday = datetime(2026, 7, 1)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-07-01", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_30_days_ago_is_recent(self, mock_datetime):
        """30 days ago should be at the boundary of recent."""
        today = datetime(2026, 7, 2)
        thirty_days_ago = datetime(2026, 6, 2)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-06-02", threshold_days=30)
        assert result is True

    @patch("sar_validation.downloaders.base.datetime")
    def test_31_days_ago_is_not_recent(self, mock_datetime):
        """31 days ago should exceed the 30-day threshold."""
        today = datetime(2026, 7, 2)
        thirty_one_days_ago = datetime(2026, 6, 1)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-06-01", threshold_days=30)
        assert result is False

    @patch("sar_validation.downloaders.base.datetime")
    def test_old_date_is_not_recent(self, mock_datetime):
        """Date from several months ago should not be recent."""
        today = datetime(2026, 7, 2)
        old_date = datetime(2026, 3, 15)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        result = is_date_recent("2026-03-15", threshold_days=30)
        assert result is False

    def test_with_hhmmss_format_parses_correctly(self):
        """HHMMSS format should parse correctly through normalize_datetime."""
        # This test verifies that normalize_datetime handles HHMMSS correctly,
        # not the date recency comparison (which requires mocking datetime.now)
        normalized = normalize_datetime("2026-07-02 120000")
        assert normalized == "2026-07-02T12:00:00"

    @patch("sar_validation.downloaders.base.datetime")
    def test_custom_threshold(self, mock_datetime):
        """Custom threshold should be respected."""
        today = datetime(2026, 7, 2)
        sixty_days_ago = datetime(2026, 5, 3)
        mock_datetime.now.return_value = today
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        # Should NOT be recent with 30-day threshold
        result = is_date_recent("2026-05-03", threshold_days=30)
        assert result is False
        
        # But SHOULD be recent with 60-day threshold
        result = is_date_recent("2026-05-03", threshold_days=60)
        assert result is True

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
# Tests for in-situ source-type <-> Copernicus platform-code mapping
# ---------------------------------------------------------------------------

class TestInsituPlatformCodeMapping:
    def test_drifter_covers_both_db_and_ad(self):
        assert SOURCE_TYPE_TO_PLATFORM["drifter"] == ["DB", "AD"]

    def test_buoy_is_db_only(self):
        assert SOURCE_TYPE_TO_PLATFORM["buoy"] == ["DB"]

    def test_resolve_platform_codes_dedupes_shared_db(self):
        codes = _resolve_platform_codes(["buoy", "drifter"])
        assert codes == ["DB", "AD"]

    def test_resolve_platform_codes_unknown_source_type_raises(self):
        with pytest.raises(ValueError):
            _resolve_platform_codes(["not_a_real_type"])

    def test_db_labels_as_buoy(self):
        assert PLATFORM_CODE_TO_SOURCE_TYPE["DB"] == "buoy"

    def test_ad_labels_as_drifter(self):
        assert PLATFORM_CODE_TO_SOURCE_TYPE["AD"] == "drifter"

    def test_mo_labels_as_mooring(self):
        assert PLATFORM_CODE_TO_SOURCE_TYPE["MO"] == "mooring"
