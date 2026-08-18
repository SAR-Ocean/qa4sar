"""Tests for dry_collocation.py's SarFootprint model and SAR-side discovery."""

from __future__ import annotations

from datetime import datetime

import pytest


class TestSarFootprint:
    def test_constructs_with_all_fields(self):
        from sar_validation.core.dry_collocation import SarFootprint

        fp = SarFootprint(
            kind="polygon",
            bbox=(-10.0, 10.0, 35.0, 55.0),
            polygon=[(35.0, -10.0), (35.0, 10.0), (55.0, 10.0), (55.0, -10.0)],
            points=None,
            sensing_start=datetime(2026, 8, 1, 0, 0, 0),
            sensing_end=datetime(2026, 8, 1, 0, 1, 30),
            source_file="S1A_IW_OCN__2SDV_20260801T000000.SAFE",
        )

        assert fp.kind == "polygon"
        assert fp.bbox == (-10.0, 10.0, 35.0, 55.0)
        assert fp.polygon is not None and len(fp.polygon) == 4
        assert fp.points is None

    def test_is_frozen(self):
        from sar_validation.core.dry_collocation import SarFootprint

        fp = SarFootprint(
            kind="wv_points", bbox=(0.0, 1.0, 0.0, 1.0), polygon=None,
            points=[(0.5, 0.5)], sensing_start=datetime(2026, 1, 1),
            sensing_end=datetime(2026, 1, 1), source_file="x.SAFE",
        )
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            fp.kind = "polygon"
