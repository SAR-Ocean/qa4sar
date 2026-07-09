"""Tests for the DataTreeConverter (step 2)."""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar_validation.core.datatree_converter import DataTreeConverter
from sar_validation.core.collocation import CollocatedPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_insitu_csv(tmp_path: Path, rows: int = 5) -> Path:
    """Write a minimal in-situ CSV to a temp file."""
    df = pd.DataFrame({
        "longitude":   np.linspace(-10.0, -5.0, rows),
        "latitude":    np.linspace(50.0,  55.0, rows),
        "time":        pd.date_range("2026-01-01", periods=rows, freq="h"),
        "WSPD":        np.random.default_rng(0).uniform(2, 15, rows),
        "WDIR":        np.random.default_rng(1).uniform(0, 360, rows),
        "platform_id": [f"mooring_{i}" for i in range(rows)],
        "platform_type": ["MO"] * rows,
    })
    path = tmp_path / "insitu_test.csv"
    df.to_csv(path, index=False)
    return path


def _make_collocations(n: int = 3) -> list[CollocatedPoint]:
    """Create a list of synthetic CollocatedPoint objects."""
    result = []
    for i in range(n):
        result.append(
            CollocatedPoint(
                sar_lon=float(i),
                sar_lat=50.0 + float(i),
                sar_time=datetime(2026, 1, 1, 12, i, 0),
                sar_data={"wind_speed": 8.0 + i, "wind_direction": 200.0 + i},
                val_lon=float(i) + 0.01,
                val_lat=50.01 + float(i),
                val_time=datetime(2026, 1, 1, 12, i + 1, 0),
                val_data={"WSPD": 7.5 + i, "WDIR": 195.0 + i},
                spatial_distance_km=1.5 * i,
                temporal_distance_minutes=float(i),
                val_source="mooring",
                val_id=f"MO_{i:03d}",
            )
        )
    return result


# ---------------------------------------------------------------------------
# from_sar_l2_ocn
# ---------------------------------------------------------------------------

class TestFromSarL2Ocn:
    def test_accepts_dataset(self):
        ds_in = xr.Dataset({"wind_speed": ("x", [5.0, 6.0, 7.0])})
        ds = DataTreeConverter.from_sar_l2_ocn(ds_in)
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["data_type"] == "sar_l2_ocn"

    def test_accepts_dict(self):
        data = {"wind_speed": np.array([5.0, 6.0]), "lat": np.array([50.0, 51.0])}
        ds = DataTreeConverter.from_sar_l2_ocn(data)
        assert "wind_speed" in ds

    def test_preserves_existing_attrs(self):
        ds_in = xr.Dataset({"v": ("x", [1.0])}, attrs={"source": "custom"})
        ds = DataTreeConverter.from_sar_l2_ocn(ds_in)
        assert ds.attrs["source"] == "custom"
        assert ds.attrs["data_type"] == "sar_l2_ocn"


# ---------------------------------------------------------------------------
# from_insitu_csv
# ---------------------------------------------------------------------------

class TestFromInsituCsv:
    def test_basic(self, tmp_path):
        path = _make_insitu_csv(tmp_path)
        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")
        assert ds is not None
        assert "WSPD" in ds
        assert "lon" in ds.coords
        assert "lat" in ds.coords
        assert "time" in ds.coords
        assert "platform_id" in ds.coords

    def test_point_dimension(self, tmp_path):
        path = _make_insitu_csv(tmp_path, rows=10)
        ds = DataTreeConverter.from_insitu_csv(path)
        assert ds.dims["point"] == 10

    def test_returns_none_for_missing_file(self, tmp_path):
        ds = DataTreeConverter.from_insitu_csv(tmp_path / "nonexistent.csv")
        assert ds is None

    def test_attributes(self, tmp_path):
        path = _make_insitu_csv(tmp_path)
        ds = DataTreeConverter.from_insitu_csv(path, source_type="buoy")
        assert ds.attrs["platform_type"] == "buoy"
        assert ds.attrs["source"] == "Copernicus Marine"

    def test_missing_required_column(self, tmp_path):
        df = pd.DataFrame({"longitude": [0.0], "WSPD": [5.0]})  # no lat, no time
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="lat"):
            DataTreeConverter.from_insitu_csv(path)

    def test_mixed_platform_types_labeled_per_point(self, tmp_path):
        # A single Copernicus in-situ CSV mixes platform types (long format,
        # one row per variable measurement) — must be labeled per point, not
        # collapsed into one blanket source_type.
        rows = []
        for pid, code in (("MO001", "MO"), ("MO002", "MO"), ("DB001", "DB")):
            rows.append({"variable": "WSPD", "platform_id": pid, "platform_type": code,
                         "time": "2026-01-01T00:00:00", "longitude": 0.0, "latitude": 50.0,
                         "value": 7.5})
        df = pd.DataFrame(rows)
        path = tmp_path / "mixed.csv"
        df.to_csv(path, index=False)

        ds = DataTreeConverter.from_insitu_csv(path, source_type="insitu")
        assert ds is not None
        assert "platform_type" in ds.coords
        labels = dict(zip(ds["platform_id"].values, ds["platform_type"].values))
        assert labels["MO001"] == "mooring"
        assert labels["MO002"] == "mooring"
        assert labels["DB001"] == "buoy"

    def test_platform_type_falls_back_to_source_type_without_column(self, tmp_path):
        path = _make_insitu_csv(tmp_path, rows=3)
        # Overwrite: this fixture already has a platform_type column, so
        # build one without it to exercise the fallback path directly.
        df = pd.read_csv(path).drop(columns=["platform_type"])
        df.to_csv(path, index=False)

        ds = DataTreeConverter.from_insitu_csv(path, source_type="tidal_gauge")
        assert set(ds["platform_type"].values) == {"tidal_gauge"}


# ---------------------------------------------------------------------------
# from_scatterometer_nc
# ---------------------------------------------------------------------------

def _make_scatterometer_nc(tmp_path: Path, rows: int = 4, cells: int = 3,
                            wind_dir: np.ndarray = None) -> Path:
    """Write a minimal OSI-SAF/ASCAT-shaped NetCDF (NUMROWS x NUMCELLS)."""
    rng = np.random.default_rng(2)
    if wind_dir is None:
        wind_dir = rng.uniform(0, 360, (rows, cells))
    ds = xr.Dataset(
        {
            "wind_speed": (("NUMROWS", "NUMCELLS"), rng.uniform(2, 15, (rows, cells))),
            "wind_dir":   (("NUMROWS", "NUMCELLS"), wind_dir),
            "model_speed": (("NUMROWS", "NUMCELLS"), rng.uniform(2, 15, (rows, cells))),
        },
        coords={
            "lat": (("NUMROWS", "NUMCELLS"), np.linspace(50.0, 55.0, rows * cells).reshape(rows, cells)),
            "lon": (("NUMROWS", "NUMCELLS"), np.linspace(-10.0, -5.0, rows * cells).reshape(rows, cells)),
        },
        attrs={"time_coverage_start": "2026-07-05T18:33:00Z"},
    )
    path = tmp_path / "OASWC12_20260705_183300_71590_M01.nc"
    ds.to_netcdf(path)
    return path


class TestFromScatterometerNc:
    def test_renames_wind_vars_to_canonical_codes(self, tmp_path):
        path = _make_scatterometer_nc(tmp_path)
        ds = DataTreeConverter.from_scatterometer_nc(path)
        assert ds is not None
        assert "WSPD" in ds
        assert "WDIR" in ds
        assert "wind_speed" not in ds
        assert "wind_dir" not in ds

    def test_leaves_other_variables_untouched(self, tmp_path):
        path = _make_scatterometer_nc(tmp_path)
        ds = DataTreeConverter.from_scatterometer_nc(path)
        assert "model_speed" in ds

    def test_platform_type_attr(self, tmp_path):
        path = _make_scatterometer_nc(tmp_path)
        ds = DataTreeConverter.from_scatterometer_nc(path)
        assert ds.attrs["platform_type"] == "scatterometer"
        assert ds.attrs["data_type"] == "scatterometer"

    def test_returns_none_for_missing_file(self, tmp_path):
        ds = DataTreeConverter.from_scatterometer_nc(tmp_path / "nonexistent.nc")
        assert ds is None

    def test_direction_converted_from_oceanographic_to_meteorological(self, tmp_path):
        # ASCAT reports the direction the wind blows TOWARDS (oceanographic).
        # from_scatterometer_nc must rotate 180° to the meteorological
        # "blows FROM" convention used by owiWindDirection / in-situ WDIR.
        raw_dir = np.array([[0.0, 90.0, 350.0], [180.0, 270.0, 10.0]])
        expected = np.array([[180.0, 270.0, 170.0], [0.0, 90.0, 190.0]])
        path = _make_scatterometer_nc(tmp_path, rows=2, cells=3, wind_dir=raw_dir)
        ds = DataTreeConverter.from_scatterometer_nc(path)
        np.testing.assert_allclose(sorted(ds["WDIR"].values), sorted(expected.ravel()))


# ---------------------------------------------------------------------------
# from_collocations
# ---------------------------------------------------------------------------

class TestFromCollocations:
    def test_basic(self):
        colls = _make_collocations(3)
        ds = DataTreeConverter.from_collocations(colls)
        assert ds is not None
        assert ds.dims["collocation"] == 3
        assert "sar_wind_speed" in ds
        assert "val_WSPD" in ds
        assert "spatial_distance_km" in ds
        assert "temporal_distance_minutes" in ds

    def test_empty_returns_none(self):
        ds = DataTreeConverter.from_collocations([])
        assert ds is None

    def test_coordinate_times(self):
        colls = _make_collocations(2)
        ds = DataTreeConverter.from_collocations(colls)
        assert "time" in ds.coords
        assert "val_time" in ds.coords
        assert "val_id" in ds.coords

    def test_nan_filling(self):
        """Variables absent in some collocations should be NaN there."""
        c1 = _make_collocations(1)[0]
        c2 = _make_collocations(1)[0]
        c2.sar_data = {"wind_speed": 9.0}   # no wind_direction
        ds = DataTreeConverter.from_collocations([c1, c2])
        assert "sar_wind_direction" in ds
        assert np.isnan(ds["sar_wind_direction"].values[1])


# ---------------------------------------------------------------------------
# to_datatree
# ---------------------------------------------------------------------------

class TestToDataTree:
    def test_basic(self, tmp_path):
        ds_sar = xr.Dataset({"ws": ("x", [5.0])})
        ds_in  = xr.Dataset({"WSPD": ("point", [4.5])})
        tree = DataTreeConverter.to_datatree({"sar": ds_sar, "insitu": ds_in})
        assert isinstance(tree, xr.DataTree)
        assert "sar"    in tree.children
        assert "insitu" in tree.children

    def test_none_datasets_skipped(self):
        ds = xr.Dataset({"v": ("x", [1.0])})
        tree = DataTreeConverter.to_datatree({"ok": ds, "missing": None})
        assert "ok" in tree.children
        assert "missing" not in tree.children


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

class TestToDataFrame:
    def test_columns(self):
        colls = _make_collocations(3)
        df = DataTreeConverter.to_dataframe(colls)
        assert len(df) == 3
        assert "sar_lon" in df.columns
        assert "sar_wind_speed" in df.columns
        assert "val_WSPD" in df.columns
        assert "spatial_distance_km" in df.columns

    def test_empty_input(self):
        df = DataTreeConverter.to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
