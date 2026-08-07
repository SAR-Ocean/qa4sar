"""Tests for ERA5Downloader."""

from __future__ import annotations

from datetime import date, datetime


class TestHoursNeededForDay:
    def test_narrow_window_within_one_day(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 12),
            datetime(2026, 7, 12, 18, 0, 0),
            datetime(2026, 7, 12, 23, 0, 0),
        )
        # [18-2, 23+2] = [16, 25] clipped to [0, 23]
        assert hours == list(range(16, 24))

    def test_window_crossing_midnight_first_day(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 12),
            datetime(2026, 7, 12, 22, 0, 0),
            datetime(2026, 7, 13, 2, 0, 0),
        )
        # [22-2, 26] clipped to day-1's own hours: [20, 23]
        assert hours == [20, 21, 22, 23]

    def test_window_crossing_midnight_second_day(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 13),
            datetime(2026, 7, 12, 22, 0, 0),
            datetime(2026, 7, 13, 2, 0, 0),
        )
        # [20, 4] clipped to day-2's own hours: [0, 4] -> [0, 0..4] but day only has 0-23,
        # buffered end 02:00+2h=04:00
        assert hours == [0, 1, 2, 3, 4]

    def test_wide_window_interior_day_gets_all_hours(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 13),
            datetime(2026, 7, 10, 0, 0, 0),
            datetime(2026, 7, 20, 0, 0, 0),
        )
        assert hours == list(range(24))

    def test_day_entirely_outside_window_returns_empty(self):
        from sar_validation.downloaders.era5_downloader import _hours_needed_for_day

        hours = _hours_needed_for_day(
            date(2026, 7, 20),
            datetime(2026, 7, 12, 0, 0, 0),
            datetime(2026, 7, 12, 23, 0, 0),
        )
        assert hours == []
