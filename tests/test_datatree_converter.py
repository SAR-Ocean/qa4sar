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

from sar_validation.core.datatree_converter import DataTreeConverter, _subset_point_ds
from sar_validation.core.collocation import CollocatedPoint
from sar_validation.core.recipe import (
    GeographicBounds, Recipe, RecipeConfig, TemporalBounds,
)


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

    def test_cf_metadata_from_table(self, tmp_path):
        path = _make_insitu_csv(tmp_path)
        ds = DataTreeConverter.from_insitu_csv(path, source_type="mooring")
        assert ds["WSPD"].attrs["standard_name"] == "wind_speed"
        assert ds["WDIR"].attrs["standard_name"] == "wind_from_direction"
        assert ds["lat"].attrs["units"] == "degrees_north"
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "INSITU_GLO_PHYBGCWAV" in ds.attrs["references"]

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
            "wind_speed": (("NUMROWS", "NUMCELLS"), rng.uniform(2, 15, (rows, cells)),
                           {"standard_name": "wind_speed",
                            "long_name": "wind speed at 10 m",
                            "units": "m s-1",
                            "valid_min": 0, "valid_max": 5000}),
            "wind_dir":   (("NUMROWS", "NUMCELLS"), wind_dir,
                           {"standard_name": "wind_to_direction",
                            "long_name": "wind direction at 10 m",
                            "units": "degree"}),
            "model_speed": (("NUMROWS", "NUMCELLS"), rng.uniform(2, 15, (rows, cells))),
        },
        coords={
            "lat": (("NUMROWS", "NUMCELLS"), np.linspace(50.0, 55.0, rows * cells).reshape(rows, cells)),
            "lon": (("NUMROWS", "NUMCELLS"), np.linspace(-10.0, -5.0, rows * cells).reshape(rows, cells)),
        },
        attrs={"time_coverage_start": "2026-07-05T18:33:00Z",
               "title": "MetOp-B ASCAT Level 2 Coastal Ocean Surface Wind Vector Product",
               "institution": "EUMETSAT/OSI SAF/KNMI"},
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

    def test_cf_metadata(self, tmp_path):
        path = _make_scatterometer_nc(tmp_path)
        ds = DataTreeConverter.from_scatterometer_nc(path)
        # WSPD: descriptive attrs copied from the raw file, packing attrs dropped
        assert ds["WSPD"].attrs["standard_name"] == "wind_speed"
        assert ds["WSPD"].attrs["units"] == "m s-1"
        assert "valid_min" not in ds["WSPD"].attrs
        # WDIR: standard_name corrected for the 180° rotation
        assert ds["WDIR"].attrs["standard_name"] == "wind_from_direction"
        assert "comment" in ds["WDIR"].attrs
        # Coordinates + globals
        assert ds["lon"].attrs["units"] == "degrees_east"
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "osi-104-b" in ds.attrs["references"]
        assert ds.attrs["institution"] == "EUMETSAT/OSI SAF/KNMI"
        assert "history" in ds.attrs


# ---------------------------------------------------------------------------
# from_altimeter
# ---------------------------------------------------------------------------

def _make_altimeter_nc(tmp_path: Path, n: int = 5) -> Path:
    """Minimal CMEMS L3 along-track altimeter file (1 Hz layout)."""
    rng = np.random.default_rng(3)
    ds = xr.Dataset(
        {
            "VAVH": ("time", rng.uniform(0.5, 4.0, n),
                     {"standard_name": "sea_surface_wave_significant_height",
                      "units": "m"}),
            "WIND_SPEED": ("time", rng.uniform(2.0, 15.0, n),
                           {"standard_name": "wind_speed", "units": "m s-1"}),
        },
        coords={
            "time":      pd.date_range("2026-07-08T18:00", periods=n, freq="s"),
            "latitude":  ("time", np.linspace(50.0, 51.0, n)),
            "longitude": ("time", np.linspace(352.0, 353.0, n)),  # 0-360 convention
        },
        attrs={"platform": "Jason-3", "doi": "https://doi.org/10.48670/moi-00179"},
    )
    path = tmp_path / "cmems_obs-wave_glo_phy-swh_nrt_j3-l3_PT1S.nc"
    ds.to_netcdf(path)
    return path


class TestFromAltimeter:
    def test_wind_speed_renamed_to_wspd(self, tmp_path):
        path = _make_altimeter_nc(tmp_path)
        ds = DataTreeConverter.from_altimeter(path)
        assert ds is not None
        assert "WSPD" in ds
        assert "WIND_SPEED" not in ds
        assert "VAVH" in ds

    def test_cf_metadata(self, tmp_path):
        path = _make_altimeter_nc(tmp_path)
        ds = DataTreeConverter.from_altimeter(path)
        # Raw attrs follow the variable through the rename
        assert ds["WSPD"].attrs["standard_name"] == "wind_speed"
        assert ds["WSPD"].attrs["units"] == "m s-1"
        assert ds["VAVH"].attrs["standard_name"] == "sea_surface_wave_significant_height"
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "WAVE_GLO_PHY_SWH_L3_NRT_014_001" in ds.attrs["references"]
        assert "doi.org" in ds.attrs["references"]

    def test_longitude_normalized(self, tmp_path):
        path = _make_altimeter_nc(tmp_path)
        ds = DataTreeConverter.from_altimeter(path)
        assert float(ds["lon"].min()) >= -180.0
        assert float(ds["lon"].max()) <= 180.0
        np.testing.assert_allclose(ds["lon"].values.min(), -8.0)


# ---------------------------------------------------------------------------
# annotate_collocation_ds
# ---------------------------------------------------------------------------

class TestAnnotateCollocationDs:
    def test_attrs_transferred_from_datatree(self):
        from sar_validation.core._cf_metadata import annotate_collocation_ds

        sar_ds = xr.Dataset({
            "owiWindSpeed": (("y", "x"), np.ones((2, 2)),
                             {"standard_name": "wind_speed", "units": "m/s"}),
        })
        val_ds = xr.Dataset({
            "WSPD": ("point", [7.0],
                     {"standard_name": "wind_speed", "units": "m s-1"}),
        })
        tree = DataTreeConverter.to_datatree({"sar/scene1": sar_ds, "validation/src": val_ds})

        coll_ds = DataTreeConverter.from_collocations(_make_collocations(2))
        # give the synthetic collocations a SAR column matching the tree node
        coll_ds["sar_owiWindSpeed"] = coll_ds["sar_wind_speed"]

        annotate_collocation_ds(coll_ds, tree)

        assert coll_ds["sar_owiWindSpeed"].attrs["standard_name"] == "wind_speed"
        assert coll_ds["val_WSPD"].attrs["units"] == "m s-1"
        assert coll_ds["spatial_distance_km"].attrs["units"] == "km"
        assert coll_ds["sar_lon"].attrs["units"] == "degrees_east"
        assert coll_ds.attrs["Conventions"] == "CF-1.8"


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
# Recipe-domain filtering (_subset_point_ds / convert_downloaded_data)
# ---------------------------------------------------------------------------

def _make_point_ds(lons, lats, times) -> xr.Dataset:
    n = len(lons)
    return xr.Dataset(
        {"WSPD": ("point", np.full(n, 8.0))},
        coords={
            "lon":  ("point", np.asarray(lons, dtype=float)),
            "lat":  ("point", np.asarray(lats, dtype=float)),
            "time": ("point", pd.to_datetime(times).values),
        },
    )


_SUBSET_KW = dict(
    min_lon=-10.0, max_lon=0.0, min_lat=50.0, max_lat=60.0,
    t_start="2026-01-01", t_end="2026-01-02",
    buffer_km=25.0, time_tolerance_minutes=180,
)


class TestSubsetPointDs:
    def test_drops_out_of_bbox_points(self):
        ds = _make_point_ds(
            lons=[-5.0, 20.0], lats=[55.0, 55.0],
            times=["2026-01-01T12:00"] * 2,
        )
        out = _subset_point_ds(ds, **_SUBSET_KW)
        assert out.sizes["point"] == 1
        assert float(out["lon"].values[0]) == -5.0

    def test_keeps_points_within_spatial_buffer(self):
        # 25 km buffer ≈ 0.45°; a point 0.2° outside the strict bbox stays.
        ds = _make_point_ds(
            lons=[0.2, 1.0], lats=[55.0, 55.0],
            times=["2026-01-01T12:00"] * 2,
        )
        out = _subset_point_ds(ds, **_SUBSET_KW)
        assert out.sizes["point"] == 1
        assert float(out["lon"].values[0]) == 0.2

    def test_drops_out_of_window_points_keeps_tolerance(self):
        # Window ends 2026-01-02T00:00; +180 min tolerance keeps 02:00,
        # drops 05:00.
        ds = _make_point_ds(
            lons=[-5.0] * 3, lats=[55.0] * 3,
            times=["2026-01-01T12:00", "2026-01-02T02:00", "2026-01-02T05:00"],
        )
        out = _subset_point_ds(ds, **_SUBSET_KW)
        assert out.sizes["point"] == 2

    def test_keeps_nat_times(self):
        ds = _make_point_ds(
            lons=[-5.0, -5.0], lats=[55.0, 55.0],
            times=[pd.Timestamp("2026-01-01T12:00"), pd.NaT],
        )
        out = _subset_point_ds(ds, **_SUBSET_KW)
        assert out.sizes["point"] == 2

    def test_returns_none_when_nothing_survives(self):
        ds = _make_point_ds(
            lons=[100.0], lats=[-30.0], times=["2026-01-01T12:00"],
        )
        assert _subset_point_ds(ds, **_SUBSET_KW) is None

    def test_non_point_dataset_passes_through(self):
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        out = _subset_point_ds(ds, **_SUBSET_KW)
        assert out is ds


def _make_scatterometer_nc_at(tmp_path: Path, lons, lats,
                              time_str="2026-07-05T18:33:00Z") -> Path:
    """Scatterometer-shaped file with explicit per-point coordinates."""
    n = len(lons)
    ds = xr.Dataset(
        {"wind_speed": ("NUMCELLS", np.full(n, 9.0)),
         "wind_dir":   ("NUMCELLS", np.full(n, 90.0))},
        coords={
            "lat": ("NUMCELLS", np.asarray(lats, dtype=float)),
            "lon": ("NUMCELLS", np.asarray(lons, dtype=float)),
        },
        attrs={"time_coverage_start": time_str},
    )
    path = tmp_path / "OASWC12_20260705_183300_71590_M01.nc"
    ds.to_netcdf(path)
    return path


def _make_recipe() -> Recipe:
    return Recipe(RecipeConfig(
        name="filter-test",
        variable="wind",
        geographic_bounds=GeographicBounds(-10.0, 0.0, 50.0, 60.0),
        temporal_bounds=TemporalBounds("2026-07-05", "2026-07-06"),
    ))


class TestConvertDownloadedDataFiltering:
    def _base_dir(self, tmp_path: Path) -> Path:
        base = tmp_path / "run"
        (base / "osi_saf_winds").mkdir(parents=True)
        return base

    def test_recipe_filters_validation_nodes(self, tmp_path):
        base = self._base_dir(tmp_path)
        # 2 points inside the recipe bbox, 3 on the other side of the world
        _make_scatterometer_nc_at(
            base / "osi_saf_winds",
            lons=[-5.0, -6.0, 120.0, 121.0, 122.0],
            lats=[55.0, 56.0, -10.0, -11.0, -12.0],
        )
        tree = DataTreeConverter.convert_downloaded_data(base, recipe=_make_recipe())
        assert tree is not None
        node = tree["validation/osi_saf_winds"].children
        (scat_node,) = node.values()
        assert scat_node.to_dataset().sizes["point"] == 2

    def test_node_dropped_when_no_points_survive(self, tmp_path):
        base = self._base_dir(tmp_path)
        _make_scatterometer_nc_at(
            base / "osi_saf_winds", lons=[120.0, 121.0], lats=[-10.0, -11.0],
        )
        tree = DataTreeConverter.convert_downloaded_data(base, recipe=_make_recipe())
        assert tree is None  # only node in the run was dropped entirely

    def test_no_recipe_keeps_everything(self, tmp_path):
        base = self._base_dir(tmp_path)
        _make_scatterometer_nc_at(
            base / "osi_saf_winds",
            lons=[-5.0, 120.0], lats=[55.0, -10.0],
        )
        tree = DataTreeConverter.convert_downloaded_data(base)
        (scat_node,) = tree["validation/osi_saf_winds"].children.values()
        assert scat_node.to_dataset().sizes["point"] == 2

    def test_datatree_written_with_compression(self, tmp_path):
        import netCDF4

        base = self._base_dir(tmp_path)
        _make_scatterometer_nc_at(
            base / "osi_saf_winds", lons=[-5.0, -6.0], lats=[55.0, 56.0],
        )
        DataTreeConverter.convert_downloaded_data(base, recipe=_make_recipe())
        with netCDF4.Dataset(base / "datatree.nc") as nc:
            group = nc["validation/osi_saf_winds"]
            (subgroup,) = group.groups.values()
            filters = subgroup.variables["WSPD"].filters()
            assert filters["zlib"]


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
