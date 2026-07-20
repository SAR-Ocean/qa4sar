"""
Tests for the EWCT/NSCT (eastward/northward current component) derivation in
``DataTreeConverter.from_insitu_csv``.

When a mooring/HF-radar in-situ CSV carries current speed (``HCSP``, m/s) and
direction (``HCDT``, degrees clockwise from North, "to" convention) but the
Cartesian components are absent or entirely unmeasured, the converter derives
them as ``EWCT = HCSP * sin(radians(HCDT))`` and
``NSCT = HCSP * cos(radians(HCDT))``. This derivation must never touch a
component column that already carries at least one valid value -- even a
column that is only *partially* filled is left completely alone, gaps and
all, since blending real observations with a derived value silently changes
the semantics of individual data points.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sar_validation.core.datatree_converter import DataTreeConverter


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "insitu.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


class TestEwctNsctDerivation:
    def test_derives_from_speed_and_direction_when_absent(self, tmp_path):
        # EWCT/NSCT columns are entirely missing; HCSP=1.0 m/s due East
        # (HCDT=90 deg) must derive EWCT ~= 1.0, NSCT ~= 0.0.
        df = pd.DataFrame({
            "longitude": [0.0],
            "latitude": [50.0],
            "time": ["2026-01-01T00:00:00"],
            "HCSP": [1.0],
            "HCDT": [90.0],
        })
        path = _write_csv(tmp_path, df)

        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")

        assert ds is not None
        assert "EWCT" in ds
        assert "NSCT" in ds
        assert ds["EWCT"].values[0] == pytest.approx(1.0, abs=1e-9)
        assert ds["NSCT"].values[0] == pytest.approx(0.0, abs=1e-9)

    def test_preserves_existing_valid_component_column(self, tmp_path):
        # EWCT already carries a real (non-NaN) value; HCSP/HCDT would derive
        # EWCT = 1.0 * sin(0) = 0.0 for that row, which must NOT overwrite
        # the genuine 0.5 measurement.
        df = pd.DataFrame({
            "longitude": [0.0],
            "latitude": [50.0],
            "time": ["2026-01-01T00:00:00"],
            "HCSP": [1.0],
            "HCDT": [0.0],
            "EWCT": [0.5],
        })
        path = _write_csv(tmp_path, df)

        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")

        assert ds is not None
        assert ds["EWCT"].values[0] == pytest.approx(0.5, abs=1e-9)

    def test_partially_valid_column_is_left_untouched_gaps_and_all(self, tmp_path):
        # Two time rows: EWCT is valid (0.5) on the first, NaN on the second.
        # HCSP/HCDT are valid on both rows. Because the EWCT column contains
        # at least one real value, the derivation must skip the whole
        # column -- including the NaN row, which must stay NaN rather than
        # being backfilled from HCSP/HCDT.
        df = pd.DataFrame({
            "longitude": [0.0, 0.0],
            "latitude": [50.0, 50.0],
            "time": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
            "HCSP": [1.0, 1.0],
            "HCDT": [90.0, 90.0],
            "EWCT": [0.5, np.nan],
        })
        path = _write_csv(tmp_path, df)

        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")

        assert ds is not None
        ewct = ds["EWCT"].values
        assert ewct[0] == pytest.approx(0.5, abs=1e-9)
        assert math.isnan(ewct[1])
