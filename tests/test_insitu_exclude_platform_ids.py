"""Tests for from_insitu_csv's exclude_platform_ids parameter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sar_validation.core.datatree_converter import DataTreeConverter


def _write_long_csv(tmp_path: Path, rows: list[dict], name: str = "insitu.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(platform_id, variable, value, *, time="2026-08-30T08:00:00", lon=13.87, lat=54.88):
    return {
        "platform_id": platform_id, "platform_type": "MO",
        "time": time, "longitude": lon, "latitude": lat,
        "variable": variable, "value": value,
    }


class TestExcludePlatformIds:
    def test_excluded_platform_is_dropped_entirely(self, tmp_path):
        path = _write_long_csv(tmp_path, [
            _row("6600021", "WSPD", 9.0),
            _row("6600021", "WDIR", 191.0),
            _row("1300002", "WSPD", 5.0),
            _row("1300002", "WDIR", 90.0),
        ])

        ds = DataTreeConverter.from_insitu_csv(
            path, source_type="buoy", exclude_platform_ids={"6600021"},
        )

        assert ds is not None
        assert list(ds["platform_id"].values) == ["1300002"]
        assert float(ds["WSPD"].values[0]) == pytest.approx(5.0)

    def test_no_exclusions_keeps_every_platform(self, tmp_path):
        path = _write_long_csv(tmp_path, [
            _row("6600021", "WSPD", 9.0),
            _row("1300002", "WSPD", 5.0),
        ])

        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")

        assert ds is not None
        assert sorted(ds["platform_id"].values.tolist()) == ["1300002", "6600021"]

    def test_exclusion_set_with_no_matches_is_a_no_op(self, tmp_path):
        path = _write_long_csv(tmp_path, [_row("6600021", "WSPD", 9.0)])

        ds = DataTreeConverter.from_insitu_csv(
            path, source_type="buoy", exclude_platform_ids={"9999999"},
        )

        assert ds is not None
        assert list(ds["platform_id"].values) == ["6600021"]

    def test_excluding_every_platform_returns_none(self, tmp_path):
        path = _write_long_csv(tmp_path, [_row("6600021", "WSPD", 9.0)])

        ds = DataTreeConverter.from_insitu_csv(
            path, source_type="buoy", exclude_platform_ids={"6600021"},
        )

        assert ds is None

    def test_blank_platform_id_row_does_not_corrupt_other_rows_ids(self, tmp_path):
        """A numeric-looking platform_id column containing even one blank
        value makes pandas infer float64 for the whole column (to
        accommodate NaN); a naive astype(str) on that column then yields
        "6600021.0" rather than "6600021" for every clean row, a value
        that can never match the plain-string platform ID decoded from
        GTS BUFR -- silently disabling exclude_platform_ids matching.
        Reading platform_id with an explicit string dtype keeps every
        clean row's ID exact and preserves the blank row's own value as a
        missing string rather than fabricating the literal text "nan"."""
        rows = [
            _row("6600021", "WSPD", 9.0),
            _row("", "WSPD", 3.0, lon=14.0, lat=55.0),
            _row("1300002", "WSPD", 5.0),
        ]
        path = _write_long_csv(tmp_path, rows)

        ds = DataTreeConverter.from_insitu_csv(
            path, source_type="buoy", exclude_platform_ids={"6600021"},
        )

        assert ds is not None
        remaining_ids = sorted(str(v) for v in ds["platform_id"].values)
        assert "6600021" not in remaining_ids
        assert "6600021.0" not in remaining_ids
        assert "1300002" in remaining_ids
