"""Tests for the recency guard in HFRadarHistoricalDownloader.download().

The delayed-mode Copernicus archive lags real-time by roughly 6 months, so
``download()`` short-circuits (logs a warning, returns ``[]``) whenever the
requested end date is younger than ``_MIN_AGE_DAYS`` (182) days old, before
attempting any region resolution or network access. These tests use end
dates computed relative to the real wall-clock time (no time-freezing
required) to keep the guard's behaviour deterministic without touching the
network.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sar_validation.downloaders.hf_radar_historical_downloader import (
    HFRadarHistoricalDownloader,
)

# Bbox for a region present in the delayed-mode archive (US-EastGulfCoast),
# reused from the existing HFRadarHistoricalDownloader tests in
# tests/test_downloaders.py.
_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT = -90.0, -60.0, 30.0, 40.0


class TestHFRadarHistoricalRecencyGuard:
    def test_recent_end_date_returns_empty_without_touching_network(self, tmp_path, caplog):
        """An end date well inside the ~6 month archive lag must short-circuit
        before any region resolution or download attempt is made, logging a
        warning explaining why nothing was fetched."""
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=31)).strftime("%Y-%m-%d")

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False)
        with patch.object(
            HFRadarHistoricalDownloader, "_download_region_window"
        ) as mock_download_window, caplog.at_level(logging.WARNING):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        assert out == []
        mock_download_window.assert_not_called()
        assert any(
            "less than" in r.message and "days old" in r.message for r in caplog.records
        )

    def test_old_enough_end_date_proceeds_past_the_guard(self, tmp_path, caplog):
        """An end date older than the archive lag must proceed into the
        window loop and attempt a (mocked) download, rather than
        short-circuiting like the recent-date case above."""
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=400)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=401)).strftime("%Y-%m-%d")

        dl = HFRadarHistoricalDownloader(output_dir=tmp_path, dry_run=False)
        fake_path = tmp_path / "fake.nc"
        with patch.object(
            HFRadarHistoricalDownloader,
            "_download_region_window",
            return_value=fake_path,
        ) as mock_download_window, caplog.at_level(logging.WARNING):
            out = dl.download(_MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT, start, end)

        mock_download_window.assert_called_once()
        assert out == [fake_path]
        assert not any(
            "less than" in r.message and "days old" in r.message for r in caplog.records
        )
