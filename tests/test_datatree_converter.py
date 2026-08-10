"""Tests for the DataTreeConverter (step 2)."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar_validation.core.collocation import CollocatedPoint
from sar_validation.core.datatree_converter import DataTreeConverter, _parse_ssm_timestamp, _subset_point_ds
from sar_validation.core.recipe import (
    GeographicBounds,
    Recipe,
    RecipeConfig,
    TemporalBounds,
    ValidationDataSource,
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


def _make_ocn_safe(
    tmp_path: Path,
    safe_name: str,
    *,
    rvl_swaths: int | None = None,
    wv: bool = False,
    with_owi: bool = True,
    ny: int = 5,
    nx: int = 4,
    seed: int = 0,
    land_rows: int = 0,
    owi_land_rows: int = 0,
) -> Path:
    """
    Build a *.SAFE dir containing one '-ocn-' measurement NetCDF.

    rvl_swaths=None -> no rvl* variables written.
    rvl_swaths=S (wv=False) -> 3-D rvl (rvlAzSize, rvlRaSize, rvlSwath=S).
    wv=True -> 2-D 13x13 rvl (rvlAzSize, rvlRaSize), as in WV imagettes.
    land_rows=N -> the first N rows of the rvlAzSize axis are written with
        rvlLandFlag=1 (land) across every column/swath; the rest are 0.
        land_rows=0 (default) omits rvlLandFlag entirely, simulating a
        product that doesn't carry it.
    owi_land_rows=N -> the first N rows of the owiAzSize axis are written
        with owiMask land-bit set (row 0 uses combo value 5 = land +
        no_data, mirroring a real product's bitmask combinations; the rest
        of the land rows use plain 1); the remaining rows are 0 (valid).
        owi_land_rows=0 (default) omits owiMask entirely, simulating a
        product that doesn't carry it.
    """
    rng = np.random.default_rng(seed)
    safe = tmp_path / safe_name
    meas = safe / "measurement"
    meas.mkdir(parents=True)

    data: dict = {}
    if with_owi:
        odims = ("owiAzSize", "owiRaSize")
        data["owiWindSpeed"] = (odims, rng.uniform(2, 15, (ny, nx)).astype("float32"))
        data["owiWindDirection"] = (odims, rng.uniform(0, 360, (ny, nx)).astype("float32"))
        data["owiLon"] = (odims, rng.uniform(-20.0, -19.0, (ny, nx)).astype("float32"))
        data["owiLat"] = (odims, rng.uniform(50.0, 51.0, (ny, nx)).astype("float32"))
        if owi_land_rows > 0:
            owi_mask = np.zeros((ny, nx), dtype="int8")
            owi_mask[:owi_land_rows, :] = 1
            if owi_land_rows > 1:
                owi_mask[0, :] = 5  # land + no_data combo, first row
            data["owiMask"] = (odims, owi_mask)

    if rvl_swaths is not None:
        if wv:
            shape, rdims = (13, 13), ("rvlAzSize", "rvlRaSize")
        else:
            shape, rdims = (ny, nx, rvl_swaths), ("rvlAzSize", "rvlRaSize", "rvlSwath")
        data["rvlRadVel"] = (rdims, rng.uniform(-3, 3, shape).astype("float32"))
        data["rvlLon"] = (rdims, rng.uniform(-20.0, -19.0, shape).astype("float32"))
        data["rvlLat"] = (rdims, rng.uniform(50.0, 51.0, shape).astype("float32"))
        data["rvlHeading"] = (rdims, rng.uniform(0, 360, shape).astype("float32"))
        data["rvlIncidenceAngle"] = (rdims, rng.uniform(20, 45, shape).astype("float32"))
        data["rvlRadVelStd"] = (rdims, rng.uniform(0.0, 0.5, shape).astype("float32"))
        if land_rows > 0:
            land_flag = np.zeros(shape, dtype="float32")
            land_flag[:land_rows, ...] = 1.0
            data["rvlLandFlag"] = (rdims, land_flag)

    ds = xr.Dataset(data, attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"})
    mode = "wv1" if wv else "ew"
    fname = f"s1a-{mode}-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
    ds.to_netcdf(meas / fname)
    return safe


def _make_wv_rvl_safe(
    tmp_path: Path,
    *,
    land_rows_per_file: list[int],
    seed: int = 0,
) -> Path:
    """
    Build a WV *.SAFE dir with one 13x13-imagette RVL measurement file per
    entry in land_rows_per_file. Entry i controls how many of that file's
    13 rvlAzSize rows are land-flagged (0 = no rvlLandFlag var at all for
    that file).
    """
    rng = np.random.default_rng(seed)
    safe = tmp_path / "S1A_WV_OCN.SAFE"
    meas = safe / "measurement"
    meas.mkdir(parents=True)
    shape, rdims = (13, 13), ("rvlAzSize", "rvlRaSize")

    for i, land_rows in enumerate(land_rows_per_file):
        data = {
            "rvlRadVel": (rdims, rng.uniform(-3, 3, shape).astype("float32")),
            "rvlLon": (rdims, rng.uniform(-20.0, -19.0, shape).astype("float32")),
            "rvlLat": (rdims, rng.uniform(50.0, 51.0, shape).astype("float32")),
            "rvlHeading": (rdims, rng.uniform(0, 360, shape).astype("float32")),
            "rvlIncidenceAngle": (rdims, rng.uniform(20, 45, shape).astype("float32")),
            "rvlRadVelStd": (rdims, rng.uniform(0.0, 0.5, shape).astype("float32")),
        }
        if land_rows > 0:
            land_flag = np.zeros(shape, dtype="float32")
            land_flag[:land_rows, :] = 1.0
            data["rvlLandFlag"] = (rdims, land_flag)
        ds_raw = xr.Dataset(data, attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"})
        fname = f"s1a-wv1-ocn-vv-20260620t19152{i}-20260620t19162{i}-065057-08333{i}-001.nc"
        ds_raw.to_netcdf(meas / fname)

    return safe


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


@pytest.mark.parametrize(
    "method_name,filename,extra_args",
    [
        pytest.param("from_insitu_csv", "nonexistent.csv", (), id="from_insitu_csv"),
        pytest.param("from_scatterometer_nc", "nonexistent.nc", (), id="from_scatterometer_nc"),
        pytest.param("from_sar_l3_ssm_geotiff", "missing.tif", (), id="from_sar_l3_ssm_geotiff"),
        pytest.param("from_nisar_sme2", "does_not_exist.h5", (), id="from_nisar_sme2"),
        pytest.param("from_radiometer_nc", "nonexistent.nc", (), id="from_radiometer_nc"),
        pytest.param("from_radiometer_bytemap", "nope.gz", (), id="from_radiometer_bytemap"),
        pytest.param("from_hf_radar_grid", "nope.nc", (), id="from_hf_radar_grid"),
        pytest.param("from_ascat_ssm", "does_not_exist.nc", (), id="from_ascat_ssm"),
        pytest.param("from_amsr_ssm", "does_not_exist.h5", (), id="from_amsr_ssm"),
        pytest.param("from_smap_ssm", "does_not_exist.h5", (), id="from_smap_ssm"),
        pytest.param("from_smos_ssm", "does_not_exist.nc", (), id="from_smos_ssm"),
        # from_c3s_ssm takes a required second positional arg (product_type)
        # that none of the other 11 from_*() converters take -- folded in
        # here (rather than left as TestFromC3sSsm's own one-off test) via
        # extra_args so this cluster still covers all 12 converters.
        pytest.param("from_c3s_ssm", "does_not_exist.nc", ("active",), id="from_c3s_ssm"),
    ],
)
def test_converter_returns_none_for_missing_file(method_name, filename, extra_args, tmp_path):
    """Every DataTreeConverter.from_*() static method opens with the same
    `if not path.exists(): return None` guard -- one parametrized test
    covers all 12 without duplicating the one-line body 12 times."""
    method = getattr(DataTreeConverter, method_name)
    assert method(tmp_path / filename, *extra_args) is None


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


def _make_ssm_geotiff(tmp_path: Path, nrows: int = 3, ncols: int = 4) -> Path:
    """Write a small synthetic CLMS SSM GeoTIFF (uint8, EPSG:4326), with the
    same embedded GDAL tags (nodata, scale_factor, add_offset, flag_values)
    confirmed against a real downloaded product — see design spec §9 /
    from_sar_l3_ssm_geotiff's docstring."""
    import rasterio
    from rasterio.transform import from_origin

    data = np.array(
        [[100, 120, 255, 60],
         [20, 251, 40, 80],
         [140, 160, 180, 255]],
        dtype=np.uint8,
    )[:nrows, :ncols]
    transform = from_origin(-10.0, 55.0, 1.0, 1.0)  # west, north, xsize, ysize

    path = tmp_path / "c_gls_SSM1km-SSM_202601010000_CEURO_S1CSAR_V1.1.1.tiff"
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs="EPSG:4326", transform=transform, nodata=255,
    ) as dst:
        dst.write(data, 1)
        dst.scales = (0.5,)
        dst.offsets = (0.0,)
        dst.update_tags(1, flag_values="{241, 242, 251, 252, 253}")
    return path


class TestFromSarL3SsmGeotiff:
    def test_decodes_grid_and_masks_no_data(self, tmp_path):
        path = _make_ssm_geotiff(tmp_path)
        ds = DataTreeConverter.from_sar_l3_ssm_geotiff(path)

        assert ds is not None
        assert ds.attrs["data_type"] == "sar_l3_ssm"
        assert "sarSSM" in ds
        assert ds["sarSSM"].dims == ("y", "x")
        assert ds["lon"].dims == ("y", "x")
        assert ds["lat"].dims == ("y", "x")
        # DN=255 (no-data, missing_value tag) must be masked to NaN.
        assert np.isnan(ds["sarSSM"].values[0, 2])
        # DN=251 (a named QC flag_values code, WaterMask) must also be
        # masked to NaN, distinct from the no-data handling above.
        assert np.isnan(ds["sarSSM"].values[1, 1])
        # DN=100 -> 100*0.5 + 0.0 = 50% saturation (real scale_factor/add_offset).
        assert ds["sarSSM"].values[0, 0] == pytest.approx(50.0)

    def test_time_parsed_from_filename(self, tmp_path):
        path = _make_ssm_geotiff(tmp_path)
        ds = DataTreeConverter.from_sar_l3_ssm_geotiff(path)
        assert pd.Timestamp(ds["time"].values) == pd.Timestamp("2026-01-01T00:00:00")

    def test_parse_ssm_timestamp_8_digit_date_only(self):
        """Test that an 8-digit YYYYMMDD token defaults to midnight."""
        filename = "c_gls_SSM1km_20260115_CEURO_S1CSAR_V1.1.1.tif"
        result = _parse_ssm_timestamp(filename)
        assert pd.Timestamp(result) == pd.Timestamp("2026-01-15T00:00:00")

    def test_parse_ssm_timestamp_no_date_token_raises(self):
        """Test that a filename with no date token raises ValueError."""
        filename = "no_date_here.tif"
        with pytest.raises(ValueError, match="Could not find a date token"):
            _parse_ssm_timestamp(filename)


class TestFromNisarSme2:
    """Tests for DataTreeConverter.from_nisar_sme2 -- fixture layout
    confirmed 2026-07-31 against a real downloaded granule
    (NISAR_L3_PR_SME2_003_005_A_014_..._001.h5): soilMoisture/longitude/
    latitude live directly under science/LSAR/SME2/grids (not a
    frequencyA subgroup); longitude/latitude are 1-D EASE-grid axes (not
    a 2-D meshgrid); the fill value is soilMoisture's own _FillValue
    dataset attribute (not a group-level attribute); the acquisition time
    is a scalar string dataset at science/LSAR/identification/
    zeroDopplerStartTime (not a root file attribute)."""

    def _write_fake_granule(self, path, *, fill_value=-9999.0):
        import h5py
        import numpy as np

        ny, nx = 4, 5
        lon_1d = np.linspace(10.0, 10.4, nx)
        lat_1d = np.linspace(45.3, 45.0, ny)  # descending, like real data
        sm = np.linspace(0.05, 0.35, ny * nx, dtype="float32").reshape(ny, nx)
        sm[0, 0] = fill_value  # one masked cell

        with h5py.File(path, "w") as f:
            grp = f.create_group("science/LSAR/SME2/grids")
            sm_dset = grp.create_dataset("soilMoisture", data=sm)
            sm_dset.attrs["_FillValue"] = np.float32(fill_value)
            grp.create_dataset("longitude", data=lon_1d.astype("float32"))
            grp.create_dataset("latitude", data=lat_1d.astype("float32"))
            ident = f.create_group("science/LSAR/identification")
            ident.create_dataset("zeroDopplerStartTime", data=b"2026-06-20T01:30:00")

    def test_reads_grid_and_masks_fill_value(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "NISAR_L3_PR_SME2_001_A_20260620T013000.h5"
        self._write_fake_granule(h5_path)

        ds = DataTreeConverter.from_nisar_sme2(h5_path)

        assert ds is not None
        assert "sarSSM" in ds
        assert ds["sarSSM"].dims == ("y", "x")
        assert ds["sarSSM"].attrs["units"] == "m3 m-3"
        assert bool(np.isnan(ds["sarSSM"].values[0, 0]))  # fill value masked
        assert float(ds["sarSSM"].values[1, 1]) == pytest.approx(
            np.linspace(0.05, 0.35, 20, dtype="float32").reshape(4, 5)[1, 1]
        )

    def test_time_parsed_from_zero_doppler_start_time(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "granule.h5"
        self._write_fake_granule(h5_path)

        ds = DataTreeConverter.from_nisar_sme2(h5_path)

        assert ds is not None
        assert pd.Timestamp(ds["time"].values) == pd.Timestamp("2026-06-20T01:30:00")

    def test_missing_group_returns_none(self, tmp_path):
        import h5py

        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "wrong_shape.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_group("some/other/path")

        ds = DataTreeConverter.from_nisar_sme2(h5_path)
        assert ds is None


def _make_radarsat2_wind_nc(
    tmp_path: Path, name: str = "radarsat2_wind.nc", include_quality_flags: bool = True,
) -> Path:
    """Build a synthetic RADARSAT-2 wind granule.

    With include_quality_flags=True (new filename era), the grid
    reproduces a real, live-confirmed (2026-08-05) finding: mask/icemask
    alone are NOT a substitute for pixel_level_quality_flags. Cell (0,1)
    is water/water per mask/icemask (the same condition the old-era
    fallback below relies on) yet is flagged 4 ("valid wind in buffer
    region" -- NOT the strict flag 5) and must still be dropped. An
    earlier design draft assumed mask==-1 & icemask==1 implied flag==5;
    this was checked directly against a real downloaded granule and
    found false (the mask/icemask condition alone kept 115,267 pixels
    vs. flag==5's 63,810) -- see design-choices.md Sec 10.

    With include_quality_flags=False (old filename era, where this
    variable does not exist), the converter's mask/icemask fallback path
    is exercised instead, and every water/water cell is kept.

      (0,0) flag 5, water/water           -> kept    (speed 5.0)
      (0,1) flag 4, water/water (buffer)  -> dropped  when flags present
                                              (speed 12.0 -- proves the
                                              flag, not mask/icemask,
                                              decides)
      (0,2) flag 0, water/water, fill-like -> dropped when flags present
                                              (speed 0.0)
      (1,0) flag 4, land/land              -> dropped either way
                                              (speed 3.0)
      (1,1) flag 1, shore/water            -> dropped either way
                                              (speed 20.0)
      (1,2) flag 5, water/water            -> kept    (speed 7.5)
    """
    sar_wind = np.array([[5.0, 12.0, 0.0], [3.0, 20.0, 7.5]], dtype="float32")
    mask = np.array([[-1, -1, -1], [1, 0, -1]], dtype="int16")
    icemask = np.array([[1, 1, 1], [2, 1, 1]], dtype="int16")
    lon = np.array([[170.0, 171.0, 172.0], [170.0, 171.0, 172.0]], dtype="float32")
    lat = np.array([[64.0, 64.0, 64.0], [63.0, 63.0, 63.0]], dtype="float32")

    data_vars = {
        "sar_wind": (("y", "x"), sar_wind),
        "mask": (("y", "x"), mask),
        "icemask": (("y", "x"), icemask),
    }
    if include_quality_flags:
        data_vars["pixel_level_quality_flags"] = (
            ("y", "x"), np.array([[5, 4, 0], [4, 1, 5]], dtype="int16"),
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "longitude": (("y", "x"), lon),
            "latitude": (("y", "x"), lat),
        },
    )
    ds["sar_wind"].attrs["_FillValue"] = np.float32(-999.0)
    ds["mask"].attrs["flag_values"] = [-1, 0, 1]
    ds["mask"].attrs["flag_meanings"] = "water shore land"
    ds["icemask"].attrs["_FillValue"] = np.int16(0)
    ds["icemask"].attrs["flag_meanings"] = "no_data water land sea_ice snow"
    ds.attrs["time_coverage_start"] = "2026-06-04T05:52:51Z"
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


class TestFromRadarsat2Wind:
    def test_only_owi_wind_speed_produced_no_direction(self, tmp_path):
        path = _make_radarsat2_wind_nc(tmp_path)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        assert ds is not None
        assert "owiWindSpeed" in ds
        assert "owiWindDirection" not in ds

    def test_quality_flag_5_kept_others_dropped(self, tmp_path):
        path = _make_radarsat2_wind_nc(tmp_path, include_quality_flags=True)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        speed = ds["owiWindSpeed"].values
        assert speed[0, 0] == pytest.approx(5.0)   # flag 5 -- kept
        assert speed[1, 2] == pytest.approx(7.5)   # flag 5 -- kept
        assert np.isnan(speed[0, 1])                # flag 4
        assert np.isnan(speed[0, 2])                # flag 0
        assert np.isnan(speed[1, 0])                # flag 4
        assert np.isnan(speed[1, 1])                # flag 1

    def test_mask_icemask_water_alone_is_not_sufficient_when_flag_present(self, tmp_path):
        """Regression guard for the mistake an earlier design draft made:
        cell (0,1) is water/water per mask/icemask (the same condition
        the old-era fallback relies on) but its quality flag is 4, not
        5 -- it must be dropped, proving the converter checks the flag
        directly rather than falling back to mask/icemask when the flag
        is present."""
        path = _make_radarsat2_wind_nc(tmp_path, include_quality_flags=True)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        assert np.isnan(ds["owiWindSpeed"].values[0, 1])

    def test_old_era_falls_back_to_mask_icemask(self, tmp_path):
        path = _make_radarsat2_wind_nc(tmp_path, include_quality_flags=False)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        speed = ds["owiWindSpeed"].values
        # Without pixel_level_quality_flags, every water/water cell (per
        # mask/icemask) is kept -- including (0,1) and (0,2), unlike the
        # flag-based new-era test above.
        assert speed[0, 0] == pytest.approx(5.0)
        assert speed[0, 1] == pytest.approx(12.0)
        assert speed[0, 2] == pytest.approx(0.0)
        assert speed[1, 2] == pytest.approx(7.5)
        assert np.isnan(speed[1, 0])   # land
        assert np.isnan(speed[1, 1])   # shore (mask=0, not -1)

    def test_grid_shape_and_coords(self, tmp_path):
        path = _make_radarsat2_wind_nc(tmp_path)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        assert ds["owiWindSpeed"].dims == ("y", "x")
        assert ds["lon"].shape == (2, 3)
        assert ds["lat"].shape == (2, 3)

    def test_time_parsed_from_time_coverage_start(self, tmp_path):
        path = _make_radarsat2_wind_nc(tmp_path)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        assert pd.Timestamp(ds["time"].values) == pd.Timestamp("2026-06-04T05:52:51")

    def test_cf_metadata_and_attrs(self, tmp_path):
        path = _make_radarsat2_wind_nc(tmp_path)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        assert ds["owiWindSpeed"].attrs["units"] == "m s-1"
        assert ds.attrs["data_type"] == "sar_l2_ocn"
        assert ds.attrs["source"] == "RADARSAT-2"

    def test_missing_file_returns_none(self, tmp_path):
        ds = DataTreeConverter.from_radarsat2_wind(tmp_path / "does_not_exist.nc")
        assert ds is None

    def test_missing_sar_wind_variable_returns_none(self, tmp_path):
        ds_raw = xr.Dataset({"other_var": (("y", "x"), np.zeros((2, 2)))})
        path = tmp_path / "no_sar_wind.nc"
        ds_raw.to_netcdf(path)
        ds = DataTreeConverter.from_radarsat2_wind(path)
        assert ds is None


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
        }, attrs={"platform_type": "mooring"})
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
        assert ds.sizes["collocation"] == 3
        assert "sar_wind_speed" in ds
        assert "val_WSPD" in ds
        assert "spatial_distance_km" in ds
        assert "temporal_distance_minutes" in ds

    def test_empty_returns_none(self):
        ds = DataTreeConverter.from_collocations([])
        assert ds is None

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


_CROSSING_SUBSET_KW = dict(
    min_lon=135.0, max_lon=-120.0, min_lat=-15.0, max_lat=30.0,
    t_start="2026-07-02", t_end="2026-07-03",
    buffer_km=25.0, time_tolerance_minutes=180,
)


class TestSubsetPointDsAntimeridian:
    def test_keeps_points_on_both_sides_of_the_dateline(self):
        ds = _make_point_ds(
            lons=[170.0, -170.0], lats=[0.0, 0.0],
            times=["2026-07-02T12:00"] * 2,
        )
        out = _subset_point_ds(ds, **_CROSSING_SUBSET_KW)
        assert out.sizes["point"] == 2

    def test_drops_points_in_the_excluded_middle(self):
        ds = _make_point_ds(
            lons=[0.0, 45.0], lats=[0.0, 0.0],
            times=["2026-07-02T12:00"] * 2,
        )
        assert _subset_point_ds(ds, **_CROSSING_SUBSET_KW) is None

    def test_keeps_latitude_filtering_alongside_the_lon_union(self):
        ds = _make_point_ds(
            lons=[170.0, 170.0], lats=[0.0, 80.0],
            times=["2026-07-02T12:00"] * 2,
        )
        out = _subset_point_ds(ds, **_CROSSING_SUBSET_KW)
        assert out.sizes["point"] == 1
        assert float(out["lat"].values[0]) == 0.0


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

    def test_s1_l3_ssm_geotiffs_land_under_sar_node(self, tmp_path):
        # SoilMoistureDownloader unzips each product into its own subfolder
        # (one "-SSM_" GeoTIFF alongside a sibling "-NOISE_" file) — mirror
        # that nested layout here rather than a flat S1_L3_SSM/*.tif.
        ssm_dir = tmp_path / "S1_L3_SSM"
        product_dir = ssm_dir / "c_gls_SSM1km_202601010000_CEURO_S1CSAR_V1.1.1_cog"
        product_dir.mkdir(parents=True)
        _make_ssm_geotiff(product_dir)

        tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        assert tree is not None
        assert "sar" in tree.children
        sar_names = list(tree["sar"].children)
        assert any("c_gls_SSM1km" in name for name in sar_names)

    def test_s1_l3_ssm_ignores_noise_layer_geotiff(self, tmp_path):
        """The sibling -NOISE_ uncertainty-layer GeoTIFF (extracted
        alongside -SSM_ in the same product folder) must not be discovered
        as if it were its own SAR scene."""
        ssm_dir = tmp_path / "S1_L3_SSM"
        product_dir = ssm_dir / "c_gls_SSM1km_202601010000_CEURO_S1CSAR_V1.1.1_cog"
        product_dir.mkdir(parents=True)
        ssm_path = _make_ssm_geotiff(product_dir)
        noise_path = product_dir / ssm_path.name.replace("-SSM_", "-NOISE_")
        noise_path.write_bytes(ssm_path.read_bytes())

        tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        assert tree is not None
        sar_names = list(tree["sar"].children)
        assert len(sar_names) == 1
        assert "NOISE" not in sar_names[0]


class TestConvertDownloadedDataOnlyUsesRecipeSarSource:
    """convert_downloaded_data must convert ONLY the recipe's chosen SAR
    source's subdirectory, even when a stale sibling SAR-shaped folder
    from a previous run (using a different source) is still on disk --
    see design doc §9 and design-choices.md §8.11."""

    def test_stale_l3_ssm_folder_ignored_when_recipe_wants_l2_ocn(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            SARDataSpec,
            TemporalBounds,
        )

        # A stale S1_L3_SSM folder left over from an earlier soil_moisture
        # run against the same output_dir -- must be ignored entirely.
        ssm_dir = tmp_path / "S1_L3_SSM"
        ssm_dir.mkdir()
        (ssm_dir / "not_a_real_geotiff.tif").write_bytes(b"not a real geotiff")

        recipe = Recipe(RecipeConfig(
            name="t", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, 10.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            sar_data=SARDataSpec(source="sentinel1_l2_ocn"),
        ))

        tree = DataTreeConverter.convert_downloaded_data(
            tmp_path, product_type="wind", recipe=recipe,
        )
        # No S1_L2_OCN folder exists either -- but the point of this test
        # is that from_sar_l3_ssm_geotiff must never even be attempted
        # against the bogus S1_L3_SSM content, which would raise/log a
        # parse error if it were (wrongly) picked up.
        if tree is not None:
            assert "sar" not in tree or len(list(tree["sar"].children)) == 0

    def test_only_resolved_source_subdir_is_scanned(self, tmp_path, monkeypatch):
        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            SARDataSpec,
            TemporalBounds,
        )

        # Two sibling folders: only S1_L3_SSM should be scanned when the
        # recipe's source is sentinel1_clms_ssm. The filename must contain
        # the "-SSM_" marker (as real SoilMoistureDownloader output does)
        # since the sentinel1_clms_ssm branch filters on it to skip the
        # sibling "-NOISE_" uncertainty-layer GeoTIFF -- a plain "real.tif"
        # would be excluded by that filter regardless of this fix.
        ssm_dir = tmp_path / "S1_L3_SSM"
        ssm_dir.mkdir()
        (ssm_dir / "real-SSM_product.tif").write_bytes(b"x")
        ocn_dir = tmp_path / "S1_L2_OCN"
        (ocn_dir / "S1A_IW_OCN.SAFE").mkdir(parents=True)

        recipe = Recipe(RecipeConfig(
            name="t", variable="soil_moisture",
            geographic_bounds=GeographicBounds(-10.0, 10.0, 40.0, 55.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            sar_data=SARDataSpec(source="sentinel1_clms_ssm"),
        ))

        calls = []

        def fake_from_sar_l3_ssm_geotiff(path):
            calls.append(path)
            return None

        monkeypatch.setattr(
            DataTreeConverter, "from_sar_l3_ssm_geotiff",
            staticmethod(fake_from_sar_l3_ssm_geotiff),
        )

        DataTreeConverter.convert_downloaded_data(tmp_path, product_type="wind", recipe=recipe)

        assert len(calls) == 1
        assert calls[0].name == "real-SSM_product.tif"


class TestConvertDownloadedDataIsmn:
    def test_ismn_csv_lands_under_validation_ismn(self, tmp_path):
        ismn_dir = tmp_path / "ismn"
        ismn_dir.mkdir()
        df = pd.DataFrame({
            "platform_id":   ["station_a"] * 3,
            "platform_type": ["ismn"] * 3,
            "time":          pd.date_range("2026-01-01", periods=3, freq="D"),
            "lon":           [10.0] * 3,
            "lat":           [45.0] * 3,
            "depth":         [0.0] * 3,
            "variable":      ["SOIL_MOISTURE"] * 3,
            "value":         [0.20, 0.21, 0.19],
        })
        df.to_csv(ismn_dir / "ismn_station_a_sensor1.csv", index=False)

        tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        assert tree is not None
        assert "validation" in tree.children
        assert "ismn" in tree["validation"].children
        node_names = list(tree["validation"]["ismn"].children)
        assert any("ismn_station_a_sensor1" in name for name in node_names)

        ds = tree["validation"]["ismn"]["ismn_station_a_sensor1"].to_dataset()
        assert ds.attrs["data_type"] == "insitu_observations"
        assert "SOIL_MOISTURE" in ds


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

class TestToDataFrame:
    def test_empty_input(self):
        df = DataTreeConverter.to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# from_radiometer_nc
# ---------------------------------------------------------------------------

def _make_radiometer_nc(tmp_path: Path, npass: int = 2, nlat: int = 4,
                        nlon: int = 6, nan_frac: float = 0.0,
                        sensor: str = "AMSR2",
                        with_direction: bool = False) -> Path:
    """Write a minimal RSS AMSR2-shaped daily gridded NetCDF (pass x lat x lon)."""
    rng = np.random.default_rng(3)
    lat = np.linspace(-80.0, 80.0, nlat).astype("float32")
    lon = np.linspace(0.125, 359.875, nlon).astype("float32")   # 0..360, like RSS
    shape = (npass, nlat, nlon)

    wspd = rng.uniform(2, 15, shape).astype("float32")
    if nan_frac > 0:
        mask = rng.random(shape) < nan_frac
        wspd[mask] = np.nan

    # Per-cell time on 2024-06-01, spread across the day.
    base = np.datetime64("2024-06-01T00:00:00", "ns")
    secs = rng.integers(0, 86400, shape).astype("timedelta64[s]").astype("timedelta64[ns]")
    time = base + secs

    data = {
        "wind_speed_LF": (("pass", "lat", "lon"), wspd,
                          {"standard_name": "wind_speed",
                           "long_name": "AMSR2 Low Frequency (LF) wind speed",
                           "units": "m s-1", "valid_min": 0, "valid_max": 70}),
        "SST": (("pass", "lat", "lon"), rng.uniform(0, 25, shape).astype("float32")),
        "time": (("pass", "lat", "lon"), time,
                 {"long_name": "fractional hours of day since midnight UTC"}),
    }
    if with_direction:
        data["wind_direction"] = (("pass", "lat", "lon"),
                                  rng.uniform(0, 360, shape).astype("float32"),
                                  {"units": "degree"})

    ds = xr.Dataset(
        data,
        coords={
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "pass": ("pass", np.arange(1, npass + 1, dtype="int32")),
        },
        attrs={"sensor": sensor, "platform": "GCOM-W1",
               "title": "RSS AMSR2 V8.2 Air-Sea ECV",
               "institution": "Remote Sensing Systems"},
    )
    path = tmp_path / f"RSS_{sensor}_ocean_L3_daily_2024-06-01_v08.2.nc"
    ds.to_netcdf(path)
    return path


class TestFromRadiometerNc:
    def test_flattens_to_points_and_renames_wspd(self, tmp_path):
        path = _make_radiometer_nc(tmp_path, npass=2, nlat=4, nlon=6)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert ds is not None
        assert "point" in ds.dims
        assert ds.sizes["point"] == 2 * 4 * 6          # no NaNs → every cell kept
        assert "WSPD" in ds
        assert "wind_speed_LF" not in ds
        assert set(ds.coords) >= {"lon", "lat", "time"}

    def test_longitude_normalized_to_pm180(self, tmp_path):
        path = _make_radiometer_nc(tmp_path)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert float(ds["lon"].min()) >= -180.0
        assert float(ds["lon"].max()) <= 180.0

    def test_drops_nan_wind_cells(self, tmp_path):
        path = _make_radiometer_nc(tmp_path, npass=2, nlat=5, nlon=5, nan_frac=0.5)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert ds is not None
        assert ds.sizes["point"] < 2 * 5 * 5           # some cells dropped
        assert np.isfinite(ds["WSPD"].values).all()    # no NaNs survive

    def test_per_cell_time_preserved(self, tmp_path):
        path = _make_radiometer_nc(tmp_path)
        ds = DataTreeConverter.from_radiometer_nc(path)
        # Times span the observation day and are not all identical.
        assert str(ds["time"].min().values).startswith("2024-06-01")
        assert ds["time"].to_index().nunique() > 1

    def test_optional_wind_direction(self, tmp_path):
        path = _make_radiometer_nc(tmp_path, with_direction=True)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert "WDIR" in ds
        assert "wind_direction" not in ds

    def test_no_direction_when_absent(self, tmp_path):
        path = _make_radiometer_nc(tmp_path, with_direction=False)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert "WDIR" not in ds

    def test_cf_metadata(self, tmp_path):
        path = _make_radiometer_nc(tmp_path)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert ds["WSPD"].attrs["units"] == "m s-1"
        assert "valid_min" not in ds["WSPD"].attrs       # packing attrs dropped
        assert "remss.com" in ds.attrs.get("references", "")

    def test_all_nan_returns_none(self, tmp_path):
        path = _make_radiometer_nc(tmp_path, nan_frac=1.0)
        ds = DataTreeConverter.from_radiometer_nc(path)
        assert ds is None


# ---------------------------------------------------------------------------
# from_radiometer_bytemap (RSS binary bytemaps: GMI / SSMIS / WindSat)
# ---------------------------------------------------------------------------

import gzip as _gzip


def _make_bytemap_gz(tmp_path: Path, sensor: str, filename: str, cells) -> Path:
    """Write a full-size RSS bytemap .gz (all-missing 255 except `cells`).

    cells: list of (pass, var_idx, lat_idx, lon_idx, byte_value).
    """
    from sar_validation.downloaders._rss_bytemap import BYTEMAP_LAYOUT, NLAT, NLON, NPASS
    nvar = len(BYTEMAP_LAYOUT[sensor]["vars"])
    arr = np.full((NPASS, nvar, NLAT, NLON), 255, np.uint8)
    for (p, v, la, lo, val) in cells:
        arr[p, v, la, lo] = val
    path = tmp_path / filename
    with _gzip.open(path, "wb") as fh:
        fh.write(arr.tobytes())
    return path


class TestFromRadiometerBytemap:
    def test_gmi_to_points(self, tmp_path):
        # GMI: var 0=time(×0.1 h), var 2=windLF(×0.2). One valid cell.
        p = _make_bytemap_gz(tmp_path, "gmi", "f35_20240601v8.2.gz",
                             [(0, 2, 400, 600, 50), (0, 0, 400, 600, 100)])
        ds = DataTreeConverter.from_radiometer_bytemap(p)
        assert ds is not None
        assert ds.sizes["point"] == 1
        assert float(ds["WSPD"].values[0]) == pytest.approx(10.0)   # 50×0.2
        assert "WDIR" not in ds
        assert ds.attrs["data_type"] == "radiometer"
        assert ds.attrs["sensor"] == "gmi"
        # time = 2024-06-01 + 10.0 h
        assert str(ds["time"].values[0]).startswith("2024-06-01T10:00")
        # lon normalized to -180..180
        assert -180.0 <= float(ds["lon"].values[0]) <= 180.0

    def test_windsat_direction_rotated_to_meteorological(self, tmp_path):
        # WindSat: var 0=mingmt(×6 min), 2=w-lf(×0.2), 8=wdir(×1.5 oceanographic).
        # wdir byte 40 → 60° oceanographic → 240° meteorological (rotate 180°).
        p = _make_bytemap_gz(tmp_path, "windsat", "wsat_20150601v7.0.1.gz",
                             [(0, 2, 300, 500, 50), (0, 0, 300, 500, 120), (0, 8, 300, 500, 40)])
        ds = DataTreeConverter.from_radiometer_bytemap(p)
        assert "WDIR" in ds
        assert float(ds["WDIR"].values[0]) == pytest.approx((60.0 + 180.0) % 360.0)  # 240
        assert ds["WDIR"].attrs["standard_name"] == "wind_from_direction"
        assert ds.attrs["sensor"] == "windsat"
        # mingmt 120×6 = 720 min = 12:00
        assert str(ds["time"].values[0]).startswith("2015-06-01T12:00")

    def test_sensor_inferred_from_filename(self, tmp_path):
        # SSMIS: var 0=time, var 1=wspd_mf(×0.2). No explicit sensor arg.
        p = _make_bytemap_gz(tmp_path, "ssmis_f17", "f17_20240601v7.gz",
                             [(0, 1, 10, 10, 45), (0, 0, 10, 10, 60)])
        ds = DataTreeConverter.from_radiometer_bytemap(p)   # sensor=None → inferred
        assert ds.attrs["sensor"] == "ssmis_f17"
        assert float(ds["WSPD"].values[0]) == pytest.approx(9.0)    # 45×0.2

    def test_special_codes_dropped(self, tmp_path):
        # A cell with valid time but masked wind (251) yields no point.
        p = _make_bytemap_gz(tmp_path, "gmi", "f35_20240601v8.2.gz",
                             [(0, 0, 5, 5, 100), (0, 2, 5, 5, 251)])
        ds = DataTreeConverter.from_radiometer_bytemap(p)
        assert ds is None            # the only touched cell has masked wind

    def test_unresolvable_sensor_returns_none(self, tmp_path):
        p = tmp_path / "unknownprefix_20240601.gz"
        with _gzip.open(p, "wb") as fh:
            fh.write(b"\x00" * 10)
        assert DataTreeConverter.from_radiometer_bytemap(p) is None


# ---------------------------------------------------------------------------
# _extract_rvl_grid_data
# ---------------------------------------------------------------------------

class TestExtractRvlGridData:
    def test_multiswath_reshaped_to_grid_keeps_all_swaths(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=5, ny=5, nx=4)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert "rvlRadVel" in ds
        assert ds["rvlRadVel"].dims == ("y", "x")
        # All 5 sub-swaths retained: x == rvlRaSize * n_swaths, not just rvlRaSize.
        assert ds.sizes["y"] == 5
        assert ds.sizes["x"] == 4 * 5
        # No data lost: every input cell survives (fixture has no NaNs).
        assert int(np.isfinite(ds["rvlRadVel"].values).sum()) == 5 * 4 * 5

    def test_multiswath_merge_is_swath_contiguous(self, tmp_path):
        # Sub-swath k occupies its own longitude block [k, k+0.03], ascending
        # in range within the swath — mirroring real IW/EW products, where
        # sub-swath index order matches ground-range order. The merged grid
        # must lay swaths side by side along x, not interleave their columns
        # (the interleave smears pcolormesh quads across the whole swath).
        ny, nx, ns = 2, 4, 3
        rdims = ("rvlAzSize", "rvlRaSize", "rvlSwath")
        lon = np.zeros((ny, nx, ns), dtype="float32")
        radvel = np.zeros((ny, nx, ns), dtype="float32")
        for k in range(ns):
            lon[:, :, k] = k + 0.01 * np.arange(nx, dtype="float32")
            radvel[:, :, k] = k
        lat = np.full((ny, nx, ns), 50.0, dtype="float32")

        safe = tmp_path / "S1A_EW_OCN.SAFE"
        meas = safe / "measurement"
        meas.mkdir(parents=True)
        ds_raw = xr.Dataset(
            {
                "rvlRadVel": (rdims, radvel),
                "rvlLon": (rdims, lon),
                "rvlLat": (rdims, lat),
            },
            attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"},
        )
        ds_raw.to_netcdf(
            meas / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
        )

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        merged_lon = ds["lon"].values
        merged_radvel = ds["rvlRadVel"].values
        assert merged_lon.shape == (ny, nx * ns)

        # Longitudes increase monotonically across each row — the interleaved
        # ordering zig-zags (0.00, 1.00, 2.00, 0.01, ...) and fails this.
        assert (np.diff(merged_lon, axis=1) > 0).all()

        # Values travel with their coordinates: columns [k*nx, (k+1)*nx) are
        # exactly sub-swath k.
        for k in range(ns):
            assert (merged_radvel[:, k * nx:(k + 1) * nx] == k).all()

    def test_single_swath_2d_passes_through(self, tmp_path):
        # WV-style 13x13 2-D rvl, read through the grid (non-flatten) branch.
        safe = _make_ocn_safe(tmp_path, "S1A_SM_OCN.SAFE", rvl_swaths=1, wv=True)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert ds["rvlRadVel"].dims == ("y", "x")
        assert ds.sizes == {"y": 13, "x": 13}

    def test_returns_none_when_no_rvl(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is None

    @staticmethod
    def _grid_masking_scene(tmp_path):
        safe = _make_ocn_safe(
            tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=3, ny=5, nx=4, land_rows=2,
        )
        raw = xr.open_dataset(
            safe / "measurement" / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
        )
        raw_radvel = raw["rvlRadVel"].values.reshape(5, -1)  # (az=5, ra*swath=12)
        expected_land_mean = float(np.nanmean(raw_radvel[:2, :]))
        raw.close()
        return safe, 2, 4, 3, expected_land_mean  # land_rows, nx, n_swaths, mean

    @staticmethod
    def _points_masking_scene(tmp_path):
        safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[3])
        raw = xr.open_dataset(sorted((safe / "measurement").glob("*.nc"))[0])
        raw_radvel = raw["rvlRadVel"].values  # (13, 13)
        expected_land_mean = float(np.nanmean(raw_radvel[:3, :]))
        raw.close()
        return safe, 3, 13, 1, expected_land_mean

    @pytest.mark.parametrize(
        "flatten_to_points,build_scene",
        [
            pytest.param(False, _grid_masking_scene, id="grid"),
            pytest.param(True, _points_masking_scene, id="points"),
        ],
    )
    def test_land_flag_masks_radvel_and_std(self, tmp_path, flatten_to_points, build_scene):
        safe, land_rows, nx, n_swaths, expected_land_mean = build_scene(tmp_path)

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=flatten_to_points
        )
        assert ds is not None

        if flatten_to_points:
            n_points_total = nx * nx  # 13 * 13, points scene is square
            land_n = land_rows * nx
            assert np.isnan(ds["rvlRadVel"].values[:land_n]).all()
            assert np.isfinite(ds["rvlRadVel"].values[land_n:]).all()
            assert np.isnan(ds["rvlRadVelStd"].values[:land_n]).all()
            assert np.isfinite(ds["rvlHeading"].values).all()
            assert ds.attrs["rvl_land_pixel_count"] == land_n
            assert ds.attrs["rvl_land_pixel_fraction"] == pytest.approx(land_n / n_points_total)
        else:
            assert np.isnan(ds["rvlRadVel"].values[:land_rows, :]).all()
            assert np.isfinite(ds["rvlRadVel"].values[land_rows:, :]).all()
            assert np.isnan(ds["rvlRadVelStd"].values[:land_rows, :]).all()
            assert np.isfinite(ds["rvlRadVelStd"].values[land_rows:, :]).all()
            assert np.isfinite(ds["rvlHeading"].values).all()
            assert np.isfinite(ds["rvlIncidenceAngle"].values).all()
            land_n = land_rows * nx * n_swaths
            assert ds.attrs["rvl_land_pixel_count"] == land_n
            assert ds.attrs["rvl_land_pixel_fraction"] == pytest.approx(land_n / (5 * nx * n_swaths))

        assert ds.attrs["rvl_land_mean_radvel"] == pytest.approx(expected_land_mean, abs=1e-5)

    @pytest.mark.parametrize(
        "flatten_to_points",
        [
            pytest.param(False, id="grid"),
            pytest.param(True, id="points"),
        ],
    )
    def test_zero_land_pixels_no_masking(self, tmp_path, flatten_to_points):
        if flatten_to_points:
            safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[0])
        else:
            # land_rows=0 (default) -> no rvlLandFlag written at all.
            safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=3, ny=5, nx=4)

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=flatten_to_points
        )
        assert ds is not None
        assert np.isfinite(ds["rvlRadVel"].values).all()
        assert ds.attrs["rvl_land_pixel_count"] == 0
        assert math.isnan(ds.attrs["rvl_land_mean_radvel"])

    def test_single_swath_land_flag_grid(self, tmp_path):
        # SM/WV-style single-swath 2-D grid (13x13), 3 land rows.
        safe = _make_ocn_safe(tmp_path, "S1A_SM_OCN.SAFE", rvl_swaths=1, wv=True, land_rows=3)
        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=False
        )
        assert ds is not None
        assert ds.sizes == {"y": 13, "x": 13}
        assert np.isnan(ds["rvlRadVel"].values[:3, :]).all()
        assert np.isfinite(ds["rvlRadVel"].values[3:, :]).all()
        assert ds.attrs["rvl_land_pixel_count"] == 3 * 13

    def test_land_flag_accumulates_across_files(self, tmp_path):
        safe = _make_wv_rvl_safe(tmp_path, land_rows_per_file=[3, 5])
        files = sorted((safe / "measurement").glob("*.nc"))
        assert len(files) == 2
        raw0, raw1 = xr.open_dataset(files[0]), xr.open_dataset(files[1])
        land_sum = (
            float(np.nansum(raw0["rvlRadVel"].values[:3, :]))
            + float(np.nansum(raw1["rvlRadVel"].values[:5, :]))
        )
        raw0.close()
        raw1.close()
        expected_count = 3 * 13 + 5 * 13
        expected_mean = land_sum / expected_count

        ds = DataTreeConverter._extract_rvl_grid_data(
            safe / "measurement", safe, flatten_to_points=True
        )
        assert ds is not None
        assert ds.sizes["point"] == 2 * 13 * 13
        assert ds.attrs["rvl_land_pixel_count"] == expected_count
        assert ds.attrs["rvl_land_mean_radvel"] == pytest.approx(expected_mean, abs=1e-5)

    def test_land_mean_radvel_excludes_nan_land_cells(self, tmp_path):
        # A land-flagged cell can also have a NaN rvlRadVel (e.g. an
        # edge-of-swath cell with no valid measurement). rvl_land_mean_radvel
        # must match np.nanmean semantics: such a cell is dropped from BOTH
        # the numerator and the denominator, in both the grid path and the
        # points path — they must agree on the same input shape.
        rdims = ("rvlAzSize", "rvlRaSize")

        def _write_fixture(meas_dir: Path, fname: str, shape, land_rows: int):
            # Reset the rng before each call so the grid and points fixtures
            # draw the identical random sequence — otherwise the two
            # fixtures would have different rvlRadVel arrays and the
            # cross-branch agreement assertion below would be meaningless.
            rng = np.random.default_rng(0)
            meas_dir.mkdir(parents=True)
            rvl_radvel = rng.uniform(-3, 3, shape).astype("float32")
            land_flag = np.zeros(shape, dtype="float32")
            land_flag[:land_rows, :] = 1.0
            # Inject a NaN into a land-flagged cell.
            rvl_radvel[0, 0] = np.nan
            data = {
                "rvlRadVel": (rdims, rvl_radvel),
                "rvlLon": (rdims, rng.uniform(-20.0, -19.0, shape).astype("float32")),
                "rvlLat": (rdims, rng.uniform(50.0, 51.0, shape).astype("float32")),
                "rvlHeading": (rdims, rng.uniform(0, 360, shape).astype("float32")),
                "rvlIncidenceAngle": (rdims, rng.uniform(20, 45, shape).astype("float32")),
                "rvlRadVelStd": (rdims, rng.uniform(0.0, 0.5, shape).astype("float32")),
                "rvlLandFlag": (rdims, land_flag),
            }
            ds_raw = xr.Dataset(data, attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"})
            ds_raw.to_netcdf(meas_dir / fname)
            return rvl_radvel

        fname = "s1a-wv1-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"

        # --- grid path (IW/EW/SM) ---
        grid_safe = tmp_path / "grid" / "S1A_SM_OCN.SAFE"
        grid_radvel = _write_fixture(grid_safe / "measurement", fname, (13, 13), land_rows=3)
        expected_grid_mean = float(np.nanmean(grid_radvel[:3, :]))

        grid_ds = DataTreeConverter._extract_rvl_grid_data(
            grid_safe / "measurement", grid_safe, flatten_to_points=False
        )
        assert grid_ds is not None
        assert grid_ds.attrs["rvl_land_mean_radvel"] == pytest.approx(
            expected_grid_mean, abs=1e-5
        )

        # --- points path (WV) ---
        points_safe = tmp_path / "points" / "S1A_WV_OCN.SAFE"
        points_radvel = _write_fixture(points_safe / "measurement", fname, (13, 13), land_rows=3)
        expected_points_mean = float(np.nanmean(points_radvel[:3, :]))

        points_ds = DataTreeConverter._extract_rvl_grid_data(
            points_safe / "measurement", points_safe, flatten_to_points=True
        )
        assert points_ds is not None
        assert points_ds.attrs["rvl_land_mean_radvel"] == pytest.approx(
            expected_points_mean, abs=1e-5
        )

        # Both paths compute the same conceptual metric over the same input
        # shape and must agree.
        assert grid_ds.attrs["rvl_land_mean_radvel"] == pytest.approx(
            points_ds.attrs["rvl_land_mean_radvel"], abs=1e-5
        )


# ---------------------------------------------------------------------------
# _from_sar_l2_ocn_iw_safe (currents, no OWI fallback)
# ---------------------------------------------------------------------------

class TestIwSafeCurrentsNoOwiFallback:
    def test_currents_with_rvl_returns_rvl_grid(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=3, with_owi=True)
        ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="currents")
        assert ds is not None
        assert "rvlRadVel" in ds.data_vars
        assert "owiWindSpeed" not in ds.data_vars
        assert ds.attrs["swath_mode"] == "IW/EW/SM"

    def test_currents_without_rvl_returns_none_and_warns(self, tmp_path, caplog):
        # OWI present but no rvl* variables — must NOT fall back to wind.
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, with_owi=True)
        with caplog.at_level("WARNING"):
            ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="currents")
        assert ds is None
        assert any("no RVL" in r.message for r in caplog.records)

    def test_wind_still_extracts_owi(self, tmp_path):
        # Regression: wind behavior unchanged.
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, with_owi=True)
        ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="wind")
        assert ds is not None
        assert "owiWindSpeed" in ds.data_vars

    def test_wind_extracts_owi_with_2d_nrcs(self, tmp_path):
        # A product with a 2-D owiNrcs (no polarisation axis) must not crash the
        # whole wind extraction. Previously the inline `[:, :, 0]` slice raised
        # IndexError, which the broad except swallowed → entire grid lost.
        ny, nx = 5, 4
        rng = np.random.default_rng(3)
        safe = tmp_path / "S1A_EW_OCN.SAFE"
        meas = safe / "measurement"
        meas.mkdir(parents=True)
        odims = ("owiAzSize", "owiRaSize")
        ds_raw = xr.Dataset(
            {
                "owiWindSpeed": (odims, rng.uniform(2, 15, (ny, nx)).astype("float32")),
                "owiWindDirection": (odims, rng.uniform(0, 360, (ny, nx)).astype("float32")),
                "owiLon": (odims, rng.uniform(-20.0, -19.0, (ny, nx)).astype("float32")),
                "owiLat": (odims, rng.uniform(50.0, 51.0, (ny, nx)).astype("float32")),
                "owiNrcs": (odims, rng.uniform(-25, -5, (ny, nx)).astype("float32")),
            },
            attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"},
        )
        ds_raw.to_netcdf(meas / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc")
        ds = DataTreeConverter._from_sar_l2_ocn_iw_safe(safe, product_type="wind")
        assert ds is not None
        assert "owiWindSpeed" in ds.data_vars
        assert ds["owiNrcs"].dims == ("y", "x")
        assert ds["owiNrcs"].shape == (ny, nx)


# ---------------------------------------------------------------------------
# _extract_owi_grid_data (land masking via owiMask)
# ---------------------------------------------------------------------------

class TestOwiMaskLandFiltering:
    def test_land_bit_masks_windspeed_and_direction(self, tmp_path):
        safe = _make_ocn_safe(
            tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, ny=5, nx=4, owi_land_rows=2,
        )
        raw = xr.open_dataset(
            safe / "measurement"
            / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
        )
        raw_speed = raw["owiWindSpeed"].values.copy()
        raw_dir = raw["owiWindDirection"].values.copy()
        raw_mask = raw["owiMask"].values.copy()
        raw.close()

        ds = DataTreeConverter._extract_owi_grid_data(safe / "measurement", safe)
        assert ds is not None

        land_rows = 2
        # Land-flagged rows (first 2, values 5 and 1) -> NaN in both
        # owiWindSpeed and owiWindDirection.
        assert np.isnan(ds["owiWindSpeed"].values[:land_rows, :]).all()
        assert np.isnan(ds["owiWindDirection"].values[:land_rows, :]).all()
        # Valid rows (owiMask == 0) are unchanged from the raw synthetic
        # values.
        np.testing.assert_array_equal(
            ds["owiWindSpeed"].values[land_rows:, :], raw_speed[land_rows:, :]
        )
        np.testing.assert_array_equal(
            ds["owiWindDirection"].values[land_rows:, :], raw_dir[land_rows:, :]
        )
        # owiMask itself passes through unmodified for downstream inspection.
        np.testing.assert_array_equal(ds["owiMask"].values, raw_mask)

        land_n = land_rows * 4  # nx
        assert ds.attrs["owi_land_pixel_count"] == land_n
        assert ds.attrs["owi_land_pixel_fraction"] == pytest.approx(land_n / (5 * 4))

    def test_zero_land_pixels_no_masking(self, tmp_path):
        # owi_land_rows=0 (default) -> no owiMask written at all -> nothing
        # masked, matching the RVL land_rows=0 convention.
        safe = _make_ocn_safe(tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, ny=5, nx=4)
        ds = DataTreeConverter._extract_owi_grid_data(safe / "measurement", safe)
        assert ds is not None
        assert np.isfinite(ds["owiWindSpeed"].values).all()
        assert np.isfinite(ds["owiWindDirection"].values).all()
        assert ds.attrs["owi_land_pixel_count"] == 0

    def test_land_masking_is_logged(self, tmp_path, caplog):
        safe = _make_ocn_safe(
            tmp_path, "S1A_EW_OCN.SAFE", rvl_swaths=None, ny=5, nx=4, owi_land_rows=2,
        )
        with caplog.at_level("WARNING"):
            ds = DataTreeConverter._extract_owi_grid_data(safe / "measurement", safe)
        assert ds is not None
        assert any(
            "land-flagged" in r.message and "owiMask" in r.message
            for r in caplog.records
        )

    def test_ice_and_rfi_bits_not_filtered(self, tmp_path):
        # Scope check: only the land bit (1) is filtered. Ice (2) and rfi
        # (8) pixels must be left untouched (no follow-up implemented yet).
        ny, nx = 3, 2
        rng = np.random.default_rng(7)
        safe = tmp_path / "S1A_EW_OCN.SAFE"
        meas = safe / "measurement"
        meas.mkdir(parents=True)
        odims = ("owiAzSize", "owiRaSize")
        owi_mask = np.array([[2, 2], [8, 8], [0, 0]], dtype="int8")
        ds_raw = xr.Dataset(
            {
                "owiWindSpeed": (odims, rng.uniform(2, 15, (ny, nx)).astype("float32")),
                "owiWindDirection": (odims, rng.uniform(0, 360, (ny, nx)).astype("float32")),
                "owiLon": (odims, rng.uniform(-20.0, -19.0, (ny, nx)).astype("float32")),
                "owiLat": (odims, rng.uniform(50.0, 51.0, (ny, nx)).astype("float32")),
                "owiMask": (odims, owi_mask),
            },
            attrs={"firstMeasurementTime": "2026-06-20T19:15:21Z"},
        )
        ds_raw.to_netcdf(meas / "s1a-ew-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc")
        ds = DataTreeConverter._extract_owi_grid_data(meas, safe)
        assert ds is not None
        assert np.isfinite(ds["owiWindSpeed"].values).all()
        assert np.isfinite(ds["owiWindDirection"].values).all()
        assert ds.attrs["owi_land_pixel_count"] == 0


# ---------------------------------------------------------------------------
# from_sar_l2_ocn_safe (WV product type routing)
# ---------------------------------------------------------------------------

class TestWvSafeProductTypeRouting:
    def test_wv_currents_returns_rvl_points(self, tmp_path):
        safe = _make_ocn_safe(tmp_path, "S1A_WV_OCN.SAFE", rvl_swaths=1, wv=True, with_owi=False)
        ds = DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type="currents")
        assert ds is not None
        assert "point" in ds.dims
        assert "rvlRadVel" in ds.data_vars
        assert ds.attrs.get("swath_mode") == "WV"

    def _make_wv_waves_safe(self, tmp_path, *, osw_hs, osw_total_hs=None):
        """
        Build a WV SAFE whose imagette measurement file carries partitioned
        ``oswHs`` and optionally an integrated ``oswTotalHs``.

        ``osw_hs`` is the per-partition Hs array (dims oswAzSize, oswRaSize,
        oswPartitions = 1, 1, N); ``-1`` entries mimic the product fill code.
        """
        safe = tmp_path / "S1A_WV_OCN.SAFE"
        meas = safe / "measurement"
        meas.mkdir(parents=True)
        osw_hs = np.asarray(osw_hs, dtype="float32").reshape(1, 1, -1)
        data = {
            "oswHs": (("oswAzSize", "oswRaSize", "oswPartitions"), osw_hs),
            "oswLon": (("oswAzSize", "oswRaSize"), np.array([[-19.5]], "float32")),
            "oswLat": (("oswAzSize", "oswRaSize"), np.array([[50.5]], "float32")),
        }
        if osw_total_hs is not None:
            data["oswTotalHs"] = (
                ("oswAzSize", "oswRaSize"),
                np.array([[osw_total_hs]], "float32"),
            )
        ds_raw = xr.Dataset(data)
        ds_raw.to_netcdf(
            meas / "s1a-wv1-ocn-vv-20260620t191521-20260620t191626-065057-083333-001.nc"
        )
        return safe

    def test_wv_waves_uses_osw_total_hs(self, tmp_path):
        # WV waves must validate the product's integrated total significant
        # wave height (oswTotalHs), NOT an individual oswHs partition.
        safe = self._make_wv_waves_safe(
            tmp_path,
            osw_hs=[0.65, 0.78, 0.54, -1.0, -1.0],
            osw_total_hs=2.85,
        )
        out = DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type="waves")
        assert out is not None
        assert "oswTotalHs" in out.data_vars
        # The stored value is the integrated total, not partition 0 (0.65) or
        # the partition mean.
        assert float(out["oswTotalHs"].values[0]) == pytest.approx(2.85, abs=1e-4)

    def test_wv_waves_falls_back_to_partition_mean(self, tmp_path):
        # No oswTotalHs (legacy product) → fall back to the mean of the valid
        # partitions (ignoring the -1 fill code), not partition 0.
        safe = self._make_wv_waves_safe(
            tmp_path,
            osw_hs=[0.60, 0.80, 1.00, -1.0, -1.0],
            osw_total_hs=None,
        )
        out = DataTreeConverter.from_sar_l2_ocn_safe(safe, product_type="waves")
        assert out is not None
        assert "oswTotalHs" in out.data_vars
        assert float(out["oswTotalHs"].values[0]) == pytest.approx(0.80, abs=1e-4)


# ---------------------------------------------------------------------------
# from_hf_radar_grid
# ---------------------------------------------------------------------------

def _make_hfr_grid_nc(tmp_path, n_time=2, n_lat=3, n_lon=4):
    """Write a minimal NOAA HFRnet-shaped gridded RTV NetCDF (time, lat, lon)."""
    rng = np.random.default_rng(7)
    times = pd.date_range("2024-05-01T00:00:00", periods=n_time, freq="1h").values
    lats = np.linspace(33.0, 38.0, n_lat)
    lons = np.linspace(-125.0, -119.0, n_lon)
    shape = (n_time, n_lat, n_lon)
    ds = xr.Dataset(
        {
            "water_u": (("time", "lat", "lon"), rng.uniform(-0.6, 0.6, shape),
                        {"standard_name": "surface_eastward_sea_water_velocity",
                         "units": "m s-1"}),
            "water_v": (("time", "lat", "lon"), rng.uniform(-0.6, 0.6, shape),
                        {"standard_name": "surface_northward_sea_water_velocity",
                         "units": "m s-1"}),
            "DOPx": (("time", "lat", "lon"), rng.uniform(0, 2, shape)),
            "DOPy": (("time", "lat", "lon"), rng.uniform(0, 2, shape)),
            "number_of_radials": (("time", "lat", "lon"),
                                  rng.integers(1, 8, shape).astype(float)),
            "number_of_sites": (("time", "lat", "lon"),
                                rng.integers(1, 4, shape).astype(float)),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"title": "NOAA HFRnet RTV", "institution": "UCSD/NOAA"},
    )
    path = tmp_path / "ucsdHfrW6_6km_2024-05-01.nc"
    ds.to_netcdf(path)
    return path


class TestFromHfRadarGrid:
    def test_renames_uv_to_ewct_nsct(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert ds is not None
        assert "EWCT" in ds and "NSCT" in ds
        assert "water_u" not in ds and "water_v" not in ds

    def test_retains_ancillary_uncertainty_fields(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert "hfr_gdop" in ds        # derived from DOPx/DOPy
        assert "hfr_n_radials" in ds
        assert "hfr_n_sites" in ds


class TestFromHfRadarGridResolutionDerivation:
    def test_derives_approximately_6km_spacing(self, tmp_path):
        # lats span 33.0..38.0 over 3 points -> 2.5 deg spacing -> ~278km;
        # this fixture isn't a real 6km product, so just assert the
        # derivation runs and lands in a sane physical range, not an exact
        # 6.0 -- the precise-spacing case is covered by the next test.
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert ds is not None
        assert "hfr_resolution_km" in ds.attrs
        assert ds.attrs["hfr_resolution_km"] > 0

    def test_derives_known_1km_spacing_precisely(self, tmp_path):
        import numpy as np
        import xarray as xr

        # ~1km spacing: 0.009 deg lat (~1.0km), lon widened by /cos(lat) at
        # this latitude is handled by the derivation itself.
        n_lat, n_lon = 5, 5
        lat0 = 36.0
        lats = lat0 + np.arange(n_lat) * (1.0 / 111.0)
        lons = -122.0 + np.arange(n_lon) * (1.0 / (111.0 * np.cos(np.radians(lat0))))
        shape = (1, n_lat, n_lon)
        raw = xr.Dataset(
            {
                "water_u": (("time", "lat", "lon"), np.zeros(shape),
                            {"standard_name": "surface_eastward_sea_water_velocity"}),
                "water_v": (("time", "lat", "lon"), np.zeros(shape),
                            {"standard_name": "surface_northward_sea_water_velocity"}),
            },
            coords={"time": [np.datetime64("2024-05-01")], "lat": lats, "lon": lons},
        )
        path = tmp_path / "synthetic_1km.nc"
        raw.to_netcdf(path)

        ds = DataTreeConverter.from_hf_radar_grid(path)
        assert ds is not None
        assert ds.attrs["hfr_resolution_km"] == pytest.approx(1.0, rel=0.05)

    def test_copernicus_shaped_input_also_gets_a_value(self, tmp_path):
        import numpy as np
        import xarray as xr

        n_lat, n_lon = 5, 5
        lats = 40.0 + np.arange(n_lat) * 0.05  # Copernicus HF-radar-total's own native spacing
        lons = -1.0 + np.arange(n_lon) * 0.05
        shape = (1, n_lat, n_lon)
        raw = xr.Dataset(
            {
                "EWCT": (("time", "lat", "lon"), np.zeros(shape)),
                "NSCT": (("time", "lat", "lon"), np.zeros(shape)),
            },
            coords={"time": [np.datetime64("2024-05-01")], "lat": lats, "lon": lons},
        )
        path = tmp_path / "synthetic_copernicus.nc"
        raw.to_netcdf(path)

        ds = DataTreeConverter.from_hf_radar_grid(path, u_var="EWCT", v_var="NSCT")
        assert ds is not None
        assert ds.attrs["hfr_resolution_km"] > 0
        assert ds.attrs["hfr_resolution_km"] != 6.0  # not a hardcoded constant

    def test_single_lat_or_lon_omits_attribute_rather_than_erroring(self, tmp_path):
        import numpy as np
        import xarray as xr

        raw = xr.Dataset(
            {
                "water_u": (("time", "lat", "lon"), np.zeros((1, 1, 3)),
                            {"standard_name": "surface_eastward_sea_water_velocity"}),
                "water_v": (("time", "lat", "lon"), np.zeros((1, 1, 3)),
                            {"standard_name": "surface_northward_sea_water_velocity"}),
            },
            coords={"time": [np.datetime64("2024-05-01")], "lat": [40.0], "lon": [-1.0, -0.9, -0.8]},
        )
        path = tmp_path / "single_lat.nc"
        raw.to_netcdf(path)

        ds = DataTreeConverter.from_hf_radar_grid(path)
        assert ds is not None
        assert "hfr_resolution_km" not in ds.attrs


def _make_copernicus_hfr_grid_nc(tmp_path, n_time=2, n_lat=3, n_lon=4):
    """Write a minimal Copernicus radar-total-shaped gridded NetCDF."""
    rng = np.random.default_rng(11)
    times = pd.date_range("2026-06-05T00:00:00", periods=n_time, freq="1h").values
    lats = np.linspace(30.0, 40.0, n_lat)
    lons = np.linspace(-90.0, -60.0, n_lon)
    shape = (n_time, n_lat, n_lon)
    ds = xr.Dataset(
        {
            "EWCT": (("time", "latitude", "longitude"), rng.uniform(-0.6, 0.6, shape),
                     {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"}),
            "NSCT": (("time", "latitude", "longitude"), rng.uniform(-0.6, 0.6, shape),
                     {"standard_name": "northward_sea_water_velocity", "units": "m s-1"}),
            "GDOP": (("time", "latitude", "longitude"), rng.uniform(0, 2, shape)),
            "EWCS": (("time", "latitude", "longitude"), rng.uniform(0, 0.1, shape)),
            "NSCS": (("time", "latitude", "longitude"), rng.uniform(0, 0.1, shape)),
            "QCflag": (("time", "latitude", "longitude"), rng.integers(0, 2, shape).astype(float)),
            "CSPD_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "DDNS_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "GDOP_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "VART_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
            "POSITION_QC": (("time", "latitude", "longitude"), rng.integers(0, 5, shape).astype(float)),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={"title": "Copernicus HFR radar-total", "institution": "HFR-EU"},
    )
    path = tmp_path / "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr_radar-total_US-EastGulfCoast_2026-06-05.nc"
    ds.to_netcdf(path)
    return path


class TestFromHfRadarGridCopernicus:
    def test_reads_ewct_nsct_directly(self, tmp_path):
        ds = DataTreeConverter.from_hf_radar_grid(
            _make_copernicus_hfr_grid_nc(tmp_path), u_var="EWCT", v_var="NSCT",
            source_label="Copernicus Marine HFR radar-total",
        )
        assert ds is not None
        assert "EWCT" in ds and "NSCT" in ds

    def test_noaa_default_args_unaffected(self, tmp_path):
        # Default u_var/v_var must still resolve NOAA's water_u/water_v.
        ds = DataTreeConverter.from_hf_radar_grid(_make_hfr_grid_nc(tmp_path))
        assert ds is not None
        assert "EWCT" in ds and ds.attrs["source"] == "NOAA HFRnet RTV"


class TestBuildDatatreeHfrNoaa:
    def test_hfr_noaa_folder_becomes_validation_node(self, tmp_path):
        base = tmp_path / "run"
        (base / "hfr_noaa").mkdir(parents=True)
        _make_hfr_grid_nc(base / "hfr_noaa")  # writes ucsdHfrW6_6km_2024-05-01.nc
        tree = DataTreeConverter.convert_downloaded_data(base, product_type="currents")
        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any("hfr_noaa" in p for p in node_paths)


class TestBuildDatatreeHfRadarCopernicus:
    def test_hf_radar_folder_becomes_validation_node(self, tmp_path):
        base = tmp_path / "run"
        (base / "hf_radar").mkdir(parents=True)
        _make_copernicus_hfr_grid_nc(base / "hf_radar")
        tree = DataTreeConverter.convert_downloaded_data(base, product_type="currents")
        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any("hf_radar" in p and "hf_radar_noaa" not in p for p in node_paths)


class TestBuildDatatreeHfRadarHistorical:
    def test_hf_radar_historical_folder_becomes_validation_node(self, tmp_path):
        base = tmp_path / "run"
        (base / "hf_radar_historical").mkdir(parents=True)
        _make_copernicus_hfr_grid_nc(base / "hf_radar_historical")
        tree = DataTreeConverter.convert_downloaded_data(base, product_type="currents")
        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any(
            "hf_radar_historical" in p and "hf_radar_noaa" not in p
            for p in node_paths
        )


def test_build_ssm_point_dataset_sets_attrs_and_data():
    ds = DataTreeConverter._build_ssm_point_dataset(
        np.array([0.1, 0.2]), np.array([10.0, 11.0]), np.array([50.0, 51.0]),
        np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
        data_type="scatterometer_ssm",
        var_attrs={"SOIL_MOISTURE": {"units": "%"}},
        platform_type="ascat_ssm",
        source="Test Source",
        sensing_depth_cm="0-5",
        band="C",
        filename="test.nc",
    )
    assert list(ds["SOIL_MOISTURE"].values) == [0.1, 0.2]
    assert ds.attrs["data_type"] == "scatterometer_ssm"
    assert ds.attrs["platform_type"] == "ascat_ssm"
    assert "sensor" not in ds.attrs
    assert ds.attrs["filename"] == "test.nc"


def _assert_ssm_point_dataset(ds, *, data_type, sensing_depth_cm, band, n_points, units, sensor=None):
    """Shared assertion tail for from_*_ssm converters: PR #26's
    `_build_ssm_point_dataset` now stamps the same attrs from every
    format-specific caller (ASCAT/AMSR/SMAP/SMOS), so each format test's
    5-6 line attr-assembly check was pure duplication of this helper's
    logic layered on top of format-specific parsing checks. `units` has
    no default since ASCAT uses "%" while the three radiometer-family
    converters use "m3 m-3" -- every caller must pass it explicitly."""
    assert ds is not None
    assert "SOIL_MOISTURE" in ds
    assert ds["SOIL_MOISTURE"].attrs["units"] == units
    assert ds.attrs["data_type"] == data_type
    assert ds.attrs["sensing_depth_cm"] == sensing_depth_cm
    assert ds.attrs["band"] == band
    assert ds.sizes["point"] == n_points
    if sensor is not None:
        assert ds.attrs["sensor"] == sensor


class TestFromASCATSsm:
    def test_converts_synthetic_file_via_ascat_package(self, tmp_path, monkeypatch):
        import numpy as np
        import pandas as pd
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        fake_nc = tmp_path / "fake_ascat_ssm.nc"
        fake_nc.write_bytes(b"")  # AscatL2File is mocked below, content unused

        synthetic = xr.Dataset(
            {
                "sm": ("obs", np.array([25.0, 50.0, np.nan, 75.0])),
            },
            coords={
                "lon": ("obs", np.array([-10.0, -5.0, 0.0, 5.0])),
                "lat": ("obs", np.array([40.0, 45.0, 50.0, 55.0])),
                "time": ("obs", pd.to_datetime(
                    ["2026-01-01T00:00:00"] * 4
                ).values),
            },
        )

        fake_reader = MagicMock()
        fake_reader.read.return_value = (synthetic, {})
        fake_ascat_module = MagicMock()
        fake_ascat_module.AscatL2File.return_value = fake_reader

        with patch.dict("sys.modules", {"ascat.eumetsat.level2": fake_ascat_module}):
            ds = DataTreeConverter.from_ascat_ssm(fake_nc)

        _assert_ssm_point_dataset(
            ds, data_type="scatterometer_ssm", sensing_depth_cm="0-5", band="C",
            n_points=3, units="%",
        )
        # The NaN cell (index 2) must be dropped.
        assert set(ds["SOIL_MOISTURE"].values) == {25.0, 50.0, 75.0}

    def test_returns_none_when_lon_missing_from_reader_output(self, tmp_path):
        """A real product missing lon/lat/time must not raise a KeyError
        that would propagate out of convert_downloaded_data and abort the
        whole conversion batch -- from_ascat_ssm must guard against it and
        return None instead, same as from_scatterometer_nc does."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        fake_nc = tmp_path / "fake_ascat_ssm_no_lon.nc"
        fake_nc.write_bytes(b"")  # AscatL2File is mocked below, content unused

        # Has "sm" but no "lon" coord/var at all.
        synthetic = xr.Dataset(
            {"sm": ("obs", np.array([25.0, 50.0]))},
        )

        fake_reader = MagicMock()
        fake_reader.read.return_value = (synthetic, {})
        fake_ascat_module = MagicMock()
        fake_ascat_module.AscatL2File.return_value = fake_reader

        with patch.dict("sys.modules", {"ascat.eumetsat.level2": fake_ascat_module}):
            ds = DataTreeConverter.from_ascat_ssm(fake_nc)

        assert ds is None


class TestFromAmsrSsm:
    def test_converts_synthetic_h5_file(self, tmp_path):
        import h5py
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "fake_nsidc0451.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("vsm", data=np.array([0.10, 0.25, -9999.0, 0.40], dtype=np.float32))
            f.create_dataset("longitude", data=np.array([-10.0, -5.0, 0.0, 5.0], dtype=np.float32))
            f.create_dataset("latitude", data=np.array([40.0, 45.0, 50.0, 55.0], dtype=np.float32))
            f.attrs["time_coverage_start"] = "2026-07-01T00:00:00"

        ds = DataTreeConverter.from_amsr_ssm(h5_path)

        _assert_ssm_point_dataset(
            ds, data_type="radiometer_ssm", sensing_depth_cm="0-1", band="X/Ka",
            n_points=3, sensor="amsr", units="m3 m-3",
        )

    def test_returns_none_for_corrupted_file(self, tmp_path):
        """A corrupted/truncated/non-HDF5 file at the AMSR SSM path must be
        caught by from_amsr_ssm's own guard and return None -- not raise --
        since convert_downloaded_data's per-file loop has no surrounding
        try/except of its own and an unhandled exception here would abort
        the entire conversion batch, not just this one file. This covers
        the format-detection ``h5py.File`` open added for the AU_Land
        branch, which must stay inside the existing try/except."""
        from sar_validation.core.datatree_converter import DataTreeConverter

        bad_path = tmp_path / "corrupted.h5"
        bad_path.write_bytes(b"this is not a valid hdf5 file")

        result = DataTreeConverter.from_amsr_ssm(bad_path)

        assert result is None

    def test_returns_none_when_vsm_missing_from_file(self, tmp_path):
        """A real product missing the vsm/longitude/latitude field(s) must
        not raise a KeyError that would propagate out of
        convert_downloaded_data and abort the whole conversion batch --
        from_amsr_ssm must guard against it and return None instead, same
        as from_ascat_ssm/from_scatterometer_nc do."""
        import h5py
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "fake_nsidc0451_no_vsm.h5"
        with h5py.File(h5_path, "w") as f:
            # Has longitude/latitude but no vsm at all.
            f.create_dataset("longitude", data=np.array([-10.0, -5.0], dtype=np.float32))
            f.create_dataset("latitude", data=np.array([40.0, 45.0], dtype=np.float32))

        ds = DataTreeConverter.from_amsr_ssm(h5_path)

        assert ds is None


class TestFromAmsrSsmAuLandFormat:
    """The real AU_Land granule structure (confirmed 2026-08-07 via a live
    NASA Earthdata download of AMSR_U2_L2_Land_B02_202312312326_D.he5):
    an HDF-EOS5 POINTS layout, not SWATHS -- a single compound dataset
    with named fields, including two independent soil-moisture retrievals
    (NPD, SCA) with no stated "primary" one. See
    _from_amsr_ssm_au_land_points's docstring and design-choices.md for
    why NPD is used."""

    _DTYPE = np.dtype([
        ("Time", "f8"), ("Latitude", "f4"), ("Longitude", "f4"),
        ("SoilMoistureNPD", "f4"), ("RetrievalQualityFlagNPD", "i4"),
        ("SoilMoistureSCA", "f4"), ("RetrievalQualityFlagSCA", "i4"),
    ])

    def _write_granule(self, path, rows):
        import h5py

        arr = np.array(rows, dtype=self._DTYPE)
        with h5py.File(path, "w") as f:
            group = f.create_group("HDFEOS/POINTS/AMSR-2 Level 2 Land Data/Data")
            group.create_dataset("Combined NPD and SCA Output Fields", data=arr)

    def test_reads_au_land_points_granule(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "AMSR2_AU_Land_sample.he5"
        # TAI93 seconds for 2024-01-01T00:17:02 -> matches the real
        # sample granule's own filename-embedded timestamp.
        t0 = 978221822.0
        self._write_granule(path, [
            (t0, 50.0, -9.0, 0.05, 0, -9999.0, -9999),
            (t0 + 1, 50.5, -8.5, 0.12, 0, 0.10, 0),
            (t0 + 2, 51.0, -8.0, -9999.0, -9999, 0.20, 0),  # fill NPD, dropped
            (t0 + 3, 51.5, -7.5, 0.30, 1, 0.28, 1),
        ])

        ds = DataTreeConverter.from_amsr_ssm(path)

        assert ds is not None
        assert ds.sizes["point"] == 3  # the fill-value NPD row is dropped
        assert float(ds["SOIL_MOISTURE"].values[0]) == pytest.approx(0.05)
        assert ds.attrs["platform_type"] == "amsr_ssm"
        assert ds.attrs["data_type"] == "radiometer_ssm"
        assert ds.attrs["sensor"] == "amsr"

    def test_time_uses_tai93_epoch_not_unix_epoch(self, tmp_path):
        """seconds-since-1993 (TAI93), not seconds-since-1970 (Unix) --
        confirmed live: the real sample granule's Time field numerically
        matches its own filename-embedded acquisition timestamp only
        under the 1993 epoch."""
        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "AMSR2_AU_Land_time.he5"
        t0 = 978221822.0  # -> 2024-01-01T00:17:02 under TAI93
        self._write_granule(path, [(t0, 50.0, -9.0, 0.05, 0, 0.05, 0)])

        ds = DataTreeConverter.from_amsr_ssm(path)

        assert ds is not None
        assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("2024-01-01T00:17:02")

    def test_sca_field_is_not_used(self, tmp_path):
        """Rows where NPD is filled but SCA has a real value must still be
        dropped -- NPD is the chosen algorithm (see design-choices.md),
        SCA is never a fallback."""
        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "AMSR2_AU_Land_sca_only.he5"
        t0 = 978221822.0
        self._write_granule(path, [(t0, 50.0, -9.0, -9999.0, -9999, 0.25, 0)])

        ds = DataTreeConverter.from_amsr_ssm(path)

        assert ds is None


class TestFromAmsrSsmGPortalL3GridFormat:
    """The format actually delivered by GPortalAMSR2Downloader (confirmed
    against a real downloaded granule -- see from_amsr_ssm's docstring):
    root-level "Geophysical Data"/"Time Information" datasets on a fixed
    0.1-degree global grid, not the hypothetical NSIDC-0451 vsm/longitude/
    latitude layout nor the AU_Land HDF-EOS5 swath layout."""

    def _write_granule(self, path, sm_grid, time_grid, sm_scale=0.1, time_scale=1.0):
        import h5py

        with h5py.File(path, "w") as f:
            sm_ds = f.create_dataset(
                "Geophysical Data", data=sm_grid[:, :, None].astype("int16"),
            )
            sm_ds.attrs["SCALE FACTOR"] = [sm_scale]
            sm_ds.attrs["UNIT"] = b"%"
            time_ds = f.create_dataset("Time Information", data=time_grid.astype("int16"))
            time_ds.attrs["SCALE FACTOR"] = [time_scale]
            time_ds.attrs["UNIT"] = b"min"

    def test_reads_gportal_l3_grid_granule(self, tmp_path):
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "GW1AM2_20250701_01D_EQMA_L3SGSMCHF3300300.h5"
        # A tiny 2x2 grid: one valid cell (raw sm=300 -> 30% -> 0.30 m3 m-3,
        # raw time=120min -> 02:00), one sm-invalid (-32768 sentinel), one
        # time-invalid (-32767 sentinel, must be dropped despite a valid
        # sm reading), one fully valid second cell.
        sm_grid = np.array([[300, -32768], [400, -32767]], dtype=np.int16)
        time_grid = np.array([[120, 0], [-32767, 300]], dtype=np.int16)
        self._write_granule(path, sm_grid, time_grid)

        ds = DataTreeConverter.from_amsr_ssm(path)

        assert ds is not None
        assert ds.sizes["point"] == 1
        assert float(ds["SOIL_MOISTURE"].values[0]) == pytest.approx(0.30)
        assert ds["SOIL_MOISTURE"].attrs["units"] == "m3 m-3"
        assert ds.attrs["data_type"] == "radiometer_ssm"
        assert ds.attrs["platform_type"] == "amsr_ssm"
        assert ds.attrs["sensor"] == "amsr"
        assert str(ds["time"].values[0])[:16] == "2025-07-01T02:00"

    def test_all_cells_invalid_returns_none(self, tmp_path):
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "GW1AM2_20250701_01D_EQMA_L3SGSMCHF3300300.h5"
        sm_grid = np.array([[-32768, -32767]], dtype=np.int16)
        time_grid = np.array([[0, 0]], dtype=np.int16)
        self._write_granule(path, sm_grid, time_grid)

        assert DataTreeConverter.from_amsr_ssm(path) is None

    def test_unparseable_filename_date_returns_none(self, tmp_path):
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "no_date_here.h5"
        sm_grid = np.array([[300]], dtype=np.int16)
        time_grid = np.array([[120]], dtype=np.int16)
        self._write_granule(path, sm_grid, time_grid)

        assert DataTreeConverter.from_amsr_ssm(path) is None

    def test_gportal_stamps_native_grid_deg_attr(self, tmp_path):
        """G-Portal's L3SGSMC format is a fixed 0.1x0.1-degree global EQR
        grid (not km-based/equal-area) -- collocation's spatial snap must
        use this exact native step directly rather than converting a
        generic aggregation_window_km through /111.0, which would coarsen
        it into merging up to 9 native cells together. See
        docs/design-choices.md §8.7 addendum."""
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "GW1AM2_20250701_01D_EQMA_L3SGSMCHF3300300.h5"
        sm_grid = np.array([[300, 400]], dtype=np.int16)
        time_grid = np.array([[120, 130]], dtype=np.int16)
        self._write_granule(path, sm_grid, time_grid)

        ds = DataTreeConverter.from_amsr_ssm(path)

        assert ds is not None
        assert ds.attrs["native_grid_deg"] == pytest.approx(0.1)


class TestConvertDownloadedDataAmsrHe5Discovery:
    """AU_Land_NRT_R02/AU_Land is an HDF-EOS5 product, which conventionally
    ships with a ``.he5`` extension (unlike NSIDC-0451's plain ``.h5``).
    convert_downloaded_data's amsr_ssm file-discovery loop must find these
    too, not just ``*.h5``, or real AU_Land downloads would silently never
    be converted."""

    def test_he5_file_in_amsr_ssm_subdir_is_discovered_and_converted(self, tmp_path):
        import h5py
        import numpy as np

        amsr_dir = tmp_path / "amsr_ssm"
        amsr_dir.mkdir()
        path = amsr_dir / "AMSR2_AU_Land_sample.he5"
        # Real AU_Land layout (see TestFromAmsrSsmAuLandFormat / §8.12 of
        # design-choices.md): HDF-EOS5 POINTS, one compound dataset.
        dtype = np.dtype([
            ("Time", "f8"), ("Latitude", "f4"), ("Longitude", "f4"),
            ("SoilMoistureNPD", "f4"), ("RetrievalQualityFlagNPD", "i4"),
        ])
        arr = np.array(
            [(978221822.0, 50.0, -9.0, 0.05, 0), (978221823.0, 50.5, -8.5, 0.12, 0)],
            dtype=dtype,
        )
        with h5py.File(path, "w") as f:
            group = f.create_group("HDFEOS/POINTS/AMSR-2 Level 2 Land Data/Data")
            group.create_dataset("Combined NPD and SCA Output Fields", data=arr)

        tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any(p.endswith(f"validation/amsr_ssm/{path.stem}") for p in node_paths)


class TestFromSmapSsm:
    def test_converts_synthetic_h5_file(self, tmp_path):
        import h5py
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "fake_spl2smp_e.h5"
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("Soil_Moisture_Retrieval_Data")
            grp.create_dataset("soil_moisture", data=np.array([0.10, 0.25, -9999.0, 0.40], dtype=np.float32))
            grp.create_dataset("longitude", data=np.array([-10.0, -5.0, 0.0, 5.0], dtype=np.float32))
            grp.create_dataset("latitude", data=np.array([40.0, 45.0, 50.0, 55.0], dtype=np.float32))
            grp.create_dataset(
                "tb_time_utc",
                data=np.array([b"2026-07-01T01:00:00.000Z"] * 4, dtype="S24"),
            )

        ds = DataTreeConverter.from_smap_ssm(h5_path)

        _assert_ssm_point_dataset(
            ds, data_type="radiometer_ssm", sensing_depth_cm="0-5", band="L",
            n_points=3, sensor="smap", units="m3 m-3",
        )

    def test_drops_cells_with_malformed_fill_pattern_time_string(self, tmp_path):
        """Real SPL2SMP_E data (confirmed against a downloaded granule) uses
        a literal "***" placeholder for tb_time_utc's fractional-seconds
        digits at some cells where soil_moisture is otherwise valid -- a
        strict pd.to_datetime on these raises ValueError and crashes the
        whole conversion. Cells with this pattern must be dropped (as if
        invalid), not raise."""
        import h5py
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "fake_spl2smp_e_bad_time.h5"
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("Soil_Moisture_Retrieval_Data")
            # 3 cells: two with valid soil_moisture, one -9999.0 fill.
            # Of the two valid-soil_moisture cells, one has a malformed
            # "***" time string despite its soil_moisture being fine.
            grp.create_dataset("soil_moisture", data=np.array([0.10, 0.25, -9999.0], dtype=np.float32))
            grp.create_dataset("longitude", data=np.array([-10.0, -5.0, 0.0], dtype=np.float32))
            grp.create_dataset("latitude", data=np.array([40.0, 45.0, 50.0], dtype=np.float32))
            grp.create_dataset(
                "tb_time_utc",
                data=np.array(
                    [b"2025-07-03T17:19:25.***Z", b"2025-07-03T17:20:01.506Z", b"2025-07-03T17:20:02.000Z"],
                    dtype="S24",
                ),
            )

        ds = DataTreeConverter.from_smap_ssm(h5_path)

        assert ds is not None
        # Only the second cell (valid soil_moisture AND valid time) survives.
        assert ds.sizes["point"] == 1
        assert float(ds["SOIL_MOISTURE"].values[0]) == pytest.approx(0.25)

    def test_returns_none_when_soil_moisture_missing_from_file(self, tmp_path):
        """A real product missing the soil_moisture/longitude/latitude/
        tb_time_utc field(s) must not raise a KeyError that would propagate
        out of convert_downloaded_data and abort the whole conversion batch
        -- from_smap_ssm must guard against it and return None instead, same
        as from_amsr_ssm does."""
        import h5py
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        h5_path = tmp_path / "fake_spl2smp_e_no_sm.h5"
        with h5py.File(h5_path, "w") as f:
            # Has longitude/latitude but no soil_moisture/tb_time_utc at all.
            grp = f.create_group("Soil_Moisture_Retrieval_Data")
            grp.create_dataset("longitude", data=np.array([-10.0, -5.0], dtype=np.float32))
            grp.create_dataset("latitude", data=np.array([40.0, 45.0], dtype=np.float32))

        ds = DataTreeConverter.from_smap_ssm(h5_path)

        assert ds is None


class TestFromSmosSsm:
    def test_converts_synthetic_netcdf_file(self, tmp_path):
        """Fixture uses the field names/time convention CONFIRMED against a
        real downloaded SMOS product (see from_smos_ssm's docstring):
        lowercase soil_moisture/longitude/latitude, and per-point time
        split across days_since_01-01-2000 (int, days since 2000-01-01)
        and seconds_since_midnight (int, seconds within that day) rather
        than a single time field -- e.g. days=9314, seconds=521 is
        2025-07-02T00:08:41, matching a real product's own filename-
        encoded acquisition window almost exactly."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "SM_OPER_MIR_SMUDP2_fake.nc"
        raw = xr.Dataset(
            {"soil_moisture": ("obs", np.array([0.10, 0.25, -999.0, 0.40]))},
            coords={
                "longitude": ("obs", np.array([-10.0, -5.0, 0.0, 5.0])),
                "latitude": ("obs", np.array([40.0, 45.0, 50.0, 55.0])),
                "days_since_01-01-2000": ("obs", np.array([9314, 9314, 9314, 9314], dtype="int32")),
                "seconds_since_midnight": ("obs", np.array([521, 520, 519, 518], dtype="int32")),
            },
        )
        raw.to_netcdf(nc_path)

        ds = DataTreeConverter.from_smos_ssm(nc_path)

        _assert_ssm_point_dataset(
            ds, data_type="radiometer_ssm", sensing_depth_cm="0-5", band="L",
            n_points=3, sensor="smos", units="m3 m-3",
        )
        # The -999.0 point is dropped -- 4 input points, 3 valid.
        assert ds["time"].values[0] == np.datetime64("2025-07-02T00:08:41")

    def test_returns_none_when_soil_moisture_missing_from_file(self, tmp_path):
        """A real product missing the soil_moisture/longitude/latitude/
        days_since_01-01-2000/seconds_since_midnight field(s) must not
        raise a KeyError that would propagate out of
        convert_downloaded_data and abort the whole conversion batch --
        from_smos_ssm must guard against it and return None instead, same
        as from_amsr_ssm/from_smap_ssm do."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "SM_OPER_MIR_SMUDP2_no_sm.nc"
        # Has longitude/latitude but no soil_moisture/time fields at all.
        raw = xr.Dataset(
            coords={
                "longitude": ("obs", np.array([-10.0, -5.0])),
                "latitude": ("obs", np.array([40.0, 45.0])),
            },
        )
        raw.to_netcdf(nc_path)

        ds = DataTreeConverter.from_smos_ssm(nc_path)

        assert ds is None


class TestConvertDownloadedDataSmosTgzWarning:
    """A real SMOS product might ship as a .tgz (see from_smos_ssm's
    docstring) which convert_downloaded_data does not extract/convert --
    if that happens, it must at least surface a warning instead of silently
    reporting zero SMOS collocations with no explanation anywhere."""

    def test_warns_when_only_tgz_files_present(self, tmp_path, caplog):
        smos_dir = tmp_path / "smos_ssm"
        smos_dir.mkdir()
        (smos_dir / "SM_OPER_MIR_SMUDP2_fake.tgz").write_bytes(b"fake archive")

        with caplog.at_level("WARNING"):
            tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        assert tree is None
        assert any(
            "smos_ssm" in record.message and ".nc" in record.message
            for record in caplog.records
        )


class TestConvertDownloadedDataAscatSidecarFiles:
    """A real EUMDAC ASCAT SSM order (confirmed against a real download)
    delivers sidecar metadata files (EOPMetadata.xml, manifest.xml) sitting
    flat alongside the .nat data products in the same directory --
    convert_downloaded_data must skip these, not attempt (and noisily fail)
    to convert them via the ascat package."""

    def test_sidecar_xml_files_are_skipped_without_warning(self, tmp_path, caplog, monkeypatch):
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd
        import xarray as xr

        ascat_dir = tmp_path / "ascat_ssm"
        ascat_dir.mkdir()
        (ascat_dir / "EOPMetadata.xml").write_text("<xml/>")
        (ascat_dir / "manifest.xml").write_text("<xml/>")
        nat_path = ascat_dir / "ASCA_SMR_02_M01_20250703T000000Z.nat"
        nat_path.write_bytes(b"fake nat content")

        synthetic = xr.Dataset(
            {"sm": ("obs", np.array([25.0]))},
            coords={
                "lon": ("obs", np.array([0.0])),
                "lat": ("obs", np.array([45.0])),
                "time": ("obs", pd.to_datetime(["2025-07-03T00:00:00"]).values),
            },
        )
        fake_reader = MagicMock()
        fake_reader.read.return_value = (synthetic, {})
        fake_ascat_module = MagicMock()
        fake_ascat_module.AscatL2File.return_value = fake_reader

        with patch.dict("sys.modules", {"ascat.eumetsat.level2": fake_ascat_module}):
            with caplog.at_level("WARNING"):
                tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        # Only the real .nat file should ever reach AscatL2File -- the two
        # XML sidecars must never trigger an attempted read (and therefore
        # never log a "format unknown" warning).
        assert fake_ascat_module.AscatL2File.call_count == 1
        assert str(nat_path) in str(fake_ascat_module.AscatL2File.call_args)
        assert not any("format unknown" in record.message for record in caplog.records)
        assert tree is not None


class TestFromHfRadarGridQCFlagFilter:
    def test_drops_qcflag_bad_cells(self, tmp_path):
        times = pd.date_range("2026-06-01", periods=1, freq="1h")
        lats = np.array([10.0, 11.0, 12.0, 13.0])
        lons = np.array([20.0, 21.0])
        ewct = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        nsct = np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]])
        # Flattened order is (time, lat, lon) row-major: [1,4,4,1,1,1,4,4]
        qcflag = np.array([[[1.0, 4.0], [4.0, 1.0], [1.0, 1.0], [4.0, 4.0]]])
        ds = xr.Dataset(
            {
                "EWCT": (("time", "latitude", "longitude"), ewct),
                "NSCT": (("time", "latitude", "longitude"), nsct),
                "QCflag": (("time", "latitude", "longitude"), qcflag),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        path = tmp_path / "cop_qc_test.nc"
        ds.to_netcdf(path)

        result = DataTreeConverter.from_hf_radar_grid(path, u_var="EWCT", v_var="NSCT")

        assert result is not None
        # 8 cells total; QCflag==4 marks 4 of them "bad" (indices 1,2,6,7 of
        # the flattened [1,4,4,1,1,1,4,4] QCflag array) -- those must be
        # dropped, leaving the 4 cells at indices 0,3,4,5 (EWCT 1,4,5,6).
        assert result.sizes["point"] == 4
        assert sorted(np.round(result["EWCT"].values, 1).tolist()) == [1.0, 4.0, 5.0, 6.0]
        assert all(q != 4 for q in result["hfr_qc"].values)

    def test_noaa_style_file_without_qcflag_is_unaffected(self, tmp_path):
        times = pd.date_range("2026-06-01", periods=1, freq="1h")
        lats = np.array([10.0, 11.0])
        lons = np.array([20.0, 21.0])
        water_u = np.array([[[1.0, 2.0], [3.0, np.nan]]])
        water_v = np.array([[[0.1, 0.2], [0.3, np.nan]]])
        ds = xr.Dataset(
            {
                "water_u": (("time", "latitude", "longitude"), water_u),
                "water_v": (("time", "latitude", "longitude"), water_v),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        path = tmp_path / "noaa_style_test.nc"
        ds.to_netcdf(path)

        result = DataTreeConverter.from_hf_radar_grid(path)

        assert result is not None
        # Only the NaN cell is dropped; no QCflag variable exists to filter on.
        assert result.sizes["point"] == 3


class TestFromC3sSsm:
    """Tests for DataTreeConverter.from_c3s_ssm."""

    def test_returns_none_for_unknown_product_type(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        path = tmp_path / "fake.nc"
        path.write_bytes(b"")
        result = DataTreeConverter.from_c3s_ssm(path, "invalid_type")
        assert result is None

    def test_converts_active_product(self, tmp_path):
        """Active product → units='%', data_type='cds_ssm', source contains ACTIVE."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        lat = np.array([50.0, 50.25])
        lon = np.array([10.0, 10.25])
        sm_vals = np.array([[35.0, np.nan], [42.0, 10.0]], dtype=float)

        time_dim = np.array(["2026-01-01"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {"sm": (("time", "lat", "lon"), sm_vals[np.newaxis, :, :])},
            coords={"time": time_dim, "lat": lat, "lon": lon},
        )
        nc_path = tmp_path / "c3s_ssm_active_20260101.nc"
        ds.to_netcdf(nc_path)

        result = DataTreeConverter.from_c3s_ssm(nc_path, "active")

        assert result is not None
        assert "SOIL_MOISTURE" in result
        assert result["SOIL_MOISTURE"].attrs["units"] == "%"
        assert result.attrs["data_type"] == "cds_ssm"
        assert result.attrs["platform_type"] == "cds_ssm"
        assert "ACTIVE" in result.attrs["source"]
        # NaN filtered: 3 valid out of 4
        assert result.sizes["point"] == 3

    def test_converts_passive_product(self, tmp_path):
        """Passive product → units='m3 m-3', source contains PASSIVE."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        lat = np.array([50.0])
        lon = np.array([10.0, 10.25])
        sm_vals = np.array([[0.3, 0.25]], dtype=float)

        time_dim = np.array(["2026-01-01"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {"sm": (("time", "lat", "lon"), sm_vals[np.newaxis, :, :])},
            coords={"time": time_dim, "lat": lat, "lon": lon},
        )
        nc_path = tmp_path / "c3s_ssm_passive_20260101.nc"
        ds.to_netcdf(nc_path)

        result = DataTreeConverter.from_c3s_ssm(nc_path, "passive")

        assert result is not None
        assert result["SOIL_MOISTURE"].attrs["units"] == "m3 m-3"
        assert "PASSIVE" in result.attrs["source"]

    def test_fill_value_masked_to_nan(self, tmp_path):
        """Cells equal to _FillValue must be masked (dropped as NaN)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        lat = np.array([50.0])
        lon = np.array([10.0, 10.25, 10.5])
        FILL = -9999.0
        sm_vals = np.array([[30.0, FILL, 45.0]], dtype=float)

        time_dim = np.array(["2026-01-01"], dtype="datetime64[ns]")
        da = xr.DataArray(
            sm_vals[np.newaxis, :, :],
            dims=("time", "lat", "lon"),
            coords={"time": time_dim, "lat": lat, "lon": lon},
            attrs={"_FillValue": FILL},
        )
        ds = xr.Dataset({"sm": da})
        nc_path = tmp_path / "c3s_ssm_active_20260101.nc"
        ds.to_netcdf(nc_path)

        result = DataTreeConverter.from_c3s_ssm(nc_path, "active")

        assert result is not None
        assert result.sizes["point"] == 2

    def test_returns_none_when_sm_variable_missing(self, tmp_path):
        """A file without 'sm' variable must return None without raising."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        ds = xr.Dataset(
            {"other_var": (("lat", "lon"), np.ones((2, 2)))},
            coords={"lat": [50.0, 50.25], "lon": [10.0, 10.25]},
        )
        nc_path = tmp_path / "c3s_ssm_active_20260101.nc"
        ds.to_netcdf(nc_path)

        result = DataTreeConverter.from_c3s_ssm(nc_path, "active")
        assert result is None


def _make_c3s_ssm_nc_at(tmp_path: Path, subdir_name: str = "cds_ssm") -> Path:
    """A minimal C3S CDS SSM NetCDF, same shape convert_downloaded_data's
    cds_ssm discovery loop must find (see from_c3s_ssm's docstring)."""
    cds_dir = tmp_path / subdir_name
    cds_dir.mkdir(parents=True, exist_ok=True)
    lat = np.array([50.0, 50.25])
    lon = np.array([10.0, 10.25])
    sm_vals = np.array([[35.0, 40.0], [42.0, 38.0]], dtype=float)
    time_dim = np.array(["2026-01-01"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"sm": (("time", "lat", "lon"), sm_vals[np.newaxis, :, :])},
        coords={"time": time_dim, "lat": lat, "lon": lon},
    )
    path = cds_dir / "c3s_ssm_20260101.nc"
    ds.to_netcdf(path)
    return path


class TestConvertDownloadedDataCdsSsm:
    """convert_downloaded_data must discover and convert cds_ssm/*.nc files
    into validation/cds_ssm/<stem> nodes, same as its sibling *_ssm sources
    -- otherwise CDSSoilMoistureDownloader's output is never reachable by
    collocation/statistics/the PDF report despite from_c3s_ssm itself
    working correctly in isolation."""

    def test_cds_ssm_file_is_discovered_and_converted(self, tmp_path):
        nc_path = _make_c3s_ssm_nc_at(tmp_path)

        tree = DataTreeConverter.convert_downloaded_data(tmp_path)

        assert tree is not None
        node_paths = [node.path for node in tree.subtree]
        assert any(p.endswith(f"validation/cds_ssm/{nc_path.stem}") for p in node_paths)

    def test_product_type_from_recipe_download_kwargs_is_honored(self, tmp_path):
        """When a recipe's cds_ssm source specifies product_type='passive',
        the converted node must carry passive (m3 m-3) units, not the
        'active' default."""
        _make_c3s_ssm_nc_at(tmp_path)
        recipe = Recipe(RecipeConfig(
            name="cds-ssm-passive-test",
            variable="soil_moisture",
            geographic_bounds=GeographicBounds(0.0, 20.0, 40.0, 60.0),
            temporal_bounds=TemporalBounds("2026-01-01", "2026-01-02"),
            validation_sources=[
                ValidationDataSource(
                    source_type="cds_ssm",
                    download_kwargs={"product_type": "passive"},
                ),
            ],
        ))

        tree = DataTreeConverter.convert_downloaded_data(tmp_path, recipe=recipe)

        assert tree is not None
        (node,) = tree["validation/cds_ssm"].children.values()
        ds = node.to_dataset()
        assert ds["SOIL_MOISTURE"].attrs["units"] == "m3 m-3"


class TestFromEra5:
    def _write_era5_nc(self, path, var_names, n_lat=4, n_lon=4, n_time=3, start_hour="2026-07-12T00:00:00"):
        import numpy as np
        import xarray as xr

        lat = np.linspace(40.0, 41.5, n_lat)
        lon = np.linspace(-10.0, -8.5, n_lon)
        time = xr.date_range(start_hour, periods=n_time, freq="1h")
        arrays = {
            v: np.random.rand(n_time, n_lat, n_lon).astype("float32") for v in var_names
        }
        data_vars = {v: (("time", "latitude", "longitude"), arr) for v, arr in arrays.items()}
        ds = xr.Dataset(data_vars, coords={"time": time, "latitude": lat, "longitude": lon})
        ds.to_netcdf(path)
        return arrays

    def test_wind_returns_gridded_dataset_with_correct_data_type(self, tmp_path):
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_wind_20260712.nc"
        arrays = self._write_era5_nc(nc_path, ["u10", "v10"])

        ds = DataTreeConverter.from_era5(nc_path, "wind")
        assert ds is not None
        assert ds.attrs["data_type"] == "era5_wind"
        assert set(ds.dims) == {"time", "lat", "lon"}
        # u10/v10 are kept as raw components -- NOT renamed/derived into
        # WSPD/WDIR here. WDIR is a circular quantity and this Dataset is
        # exactly what gets bilinearly/hyperbolically interpolated at
        # collocation time; deriving WDIR before that interpolation would
        # break across the 0/360 seam (see C1 fix,
        # model_collocation._derive_wind_wspd_wdir, which now does this
        # derivation AFTER interpolation instead).
        assert "u10" in ds.data_vars and "v10" in ds.data_vars
        assert "WSPD" not in ds.data_vars and "WDIR" not in ds.data_vars
        np.testing.assert_allclose(ds["u10"].values, arrays["u10"], rtol=1e-5)
        np.testing.assert_allclose(ds["v10"].values, arrays["v10"], rtol=1e-5)
        # CF attrs from _ERA5_VARS's wind entry must be preserved.
        assert ds["u10"].attrs["standard_name"] == "eastward_wind"
        assert ds["v10"].attrs["standard_name"] == "northward_wind"

    def test_wind_lsm_present_becomes_lat_lon_coord_not_data_var(self, tmp_path):
        """land_sea_mask ("lsm" on the wire) is a per-cell land-mask
        LOOKUP, not a per-hour model quantity to interpolate/report at
        collocation points -- it must not show up as a spurious extra
        val_<name> statistics column downstream. Keeping it as a
        coordinate (not a data_var) achieves that for free: every
        downstream consumer that iterates `era5_ds.data_vars`
        (_model_values_at_points, _collocate_cell_averaging_grid) already
        skips coordinates automatically."""
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_wind_20260712.nc"
        arrays = self._write_era5_nc(nc_path, ["u10", "v10", "lsm"], n_time=3)

        ds = DataTreeConverter.from_era5(nc_path, "wind")
        assert ds is not None
        assert "lsm" not in ds.data_vars
        assert "lsm" in ds.coords
        # Time-invariant in reality (CDS just echoes it per-hour) -- kept
        # collapsed to (lat, lon) only, matching the raw fixture's
        # first-hour slice.
        assert set(ds["lsm"].dims) == {"lat", "lon"}
        np.testing.assert_allclose(ds["lsm"].values, arrays["lsm"][0])

    def test_wind_without_lsm_still_works(self, tmp_path):
        """Backward compatibility: a wind file with no lsm variable (e.g.
        a fixture or an older cached download) must not crash, and the
        Dataset simply has no lsm coord."""
        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_wind_20260712.nc"
        self._write_era5_nc(nc_path, ["u10", "v10"])

        ds = DataTreeConverter.from_era5(nc_path, "wind")
        assert ds is not None
        assert "lsm" not in ds.coords
        assert "lsm" not in ds.data_vars

    def test_waves_lsm_not_extracted_even_if_present(self, tmp_path):
        """lsm extraction is scoped to wind only -- waves never requests
        it (see era5_downloader.py), but even if a stray lsm variable
        showed up in a waves file, from_era5 must not extract it (only the
        wind branch does)."""
        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_waves_20260712.nc"
        self._write_era5_nc(nc_path, ["swh", "lsm"])

        ds = DataTreeConverter.from_era5(nc_path, "waves")
        assert ds is not None
        assert "lsm" not in ds.coords
        assert "lsm" not in ds.data_vars

    def test_wind_components_hand_checkable_case_passthrough(self, tmp_path):
        """u10=0, v10=-1 is wind blowing FROM the north (northerly wind) --
        from_era5 must pass these raw components through unchanged (no
        WSPD/WDIR derivation at conversion time any more). The
        corresponding WDIR=0/360 hand-check now lives in
        test_model_collocation.py against `_derive_wind_wspd_wdir`, which
        runs the derivation AFTER interpolation instead (see C1 fix)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter

        lat = np.array([40.0])
        lon = np.array([-10.0])
        time = xr.date_range("2026-07-12T00:00:00", periods=1, freq="1h")
        ds_in = xr.Dataset(
            {
                "u10": (("time", "latitude", "longitude"), np.zeros((1, 1, 1), dtype="float32")),
                "v10": (("time", "latitude", "longitude"), -np.ones((1, 1, 1), dtype="float32")),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )
        nc_path = tmp_path / "era5_wind_20260712.nc"
        ds_in.to_netcdf(nc_path)

        ds = DataTreeConverter.from_era5(nc_path, "wind")
        assert ds is not None
        assert float(ds["u10"].values.ravel()[0]) == pytest.approx(0.0)
        assert float(ds["v10"].values.ravel()[0]) == pytest.approx(-1.0)

    def test_waves_returns_vhm0_variable(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_waves_20260712.nc"
        self._write_era5_nc(nc_path, ["swh"])

        ds = DataTreeConverter.from_era5(nc_path, "waves")
        assert ds is not None
        assert ds.attrs["data_type"] == "era5_waves"
        assert "VHM0" in ds.data_vars
        assert "swh" not in ds.data_vars

    def test_soil_moisture_returns_soil_moisture_variable(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_soil_moisture_20260712.nc"
        self._write_era5_nc(nc_path, ["swvl1"])

        ds = DataTreeConverter.from_era5(nc_path, "soil_moisture")
        assert ds is not None
        assert ds.attrs["data_type"] == "era5_soil_moisture"
        assert "SOIL_MOISTURE" in ds.data_vars
        assert "swvl1" not in ds.data_vars

    def test_multiple_daily_files_concatenated_along_time(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        day1 = tmp_path / "era5_wind_20260712.nc"
        day2 = tmp_path / "era5_wind_20260713.nc"
        self._write_era5_nc(day1, ["u10", "v10"], n_time=24, start_hour="2026-07-12T00:00:00")
        self._write_era5_nc(day2, ["u10", "v10"], n_time=24, start_hour="2026-07-13T00:00:00")

        ds = DataTreeConverter.from_era5([day1, day2], "wind")
        assert ds is not None
        assert ds.sizes["time"] == 48

    def test_unknown_variable_returns_none(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_currents_20260712.nc"
        self._write_era5_nc(nc_path, ["u10"])

        assert DataTreeConverter.from_era5(nc_path, "currents") is None

    def test_missing_file_returns_none(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        assert DataTreeConverter.from_era5(tmp_path / "nope.nc", "wind") is None

    def test_missing_expected_variable_returns_none(self, tmp_path):
        from sar_validation.core.datatree_converter import DataTreeConverter

        nc_path = tmp_path / "era5_wind_20260712.nc"
        self._write_era5_nc(nc_path, ["v10"])  # missing u10

        assert DataTreeConverter.from_era5(nc_path, "wind") is None


class TestConvertDownloadedDataEra5:
    def test_era5_dir_discovered_and_combined_into_one_node(self, tmp_path):
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )

        era5_dir = tmp_path / "era5"
        era5_dir.mkdir()
        lat = np.linspace(40.0, 41.5, 4)
        lon = np.linspace(-10.0, -8.5, 4)
        for day, stem in [("2026-07-12T00:00:00", "20260712"), ("2026-07-13T00:00:00", "20260713")]:
            time = xr.date_range(day, periods=24, freq="1h")
            ds = xr.Dataset(
                {
                    "u10": (("time", "latitude", "longitude"), np.random.rand(24, 4, 4).astype("float32")),
                    "v10": (("time", "latitude", "longitude"), np.random.rand(24, 4, 4).astype("float32")),
                },
                coords={"time": time, "latitude": lat, "longitude": lon},
            )
            ds.to_netcdf(era5_dir / f"era5_wind_{stem}.nc")

        recipe = Recipe(RecipeConfig(
            name="t", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, -8.5, 40.0, 41.5),
            temporal_bounds=TemporalBounds("2026-07-12", "2026-07-13"),
        ))

        tree = DataTreeConverter.convert_downloaded_data(tmp_path, recipe=recipe)
        assert tree is not None
        era5_ds = tree["validation/era5/era5"].to_dataset()
        assert era5_ds.sizes["time"] == 48
        assert era5_ds.attrs["data_type"] == "era5_wind"

    def test_stale_files_from_other_variable_are_ignored(self, tmp_path):
        """Reruns of a recipe with a different `variable` in the same output
        dir must not have their leftover era5_<other>_*.nc files globbed in
        alongside the current variable's files (see HF-radar stale-cache
        incident for the same bug class)."""
        import numpy as np
        import xarray as xr

        from sar_validation.core.datatree_converter import DataTreeConverter
        from sar_validation.core.recipe import (
            GeographicBounds,
            Recipe,
            RecipeConfig,
            TemporalBounds,
        )

        era5_dir = tmp_path / "era5"
        era5_dir.mkdir()
        lat = np.linspace(40.0, 41.5, 4)
        lon = np.linspace(-10.0, -8.5, 4)

        # Current variable: wind, single day, 24 hourly steps.
        wind_time = xr.date_range("2026-07-12T00:00:00", periods=24, freq="1h")
        wind_ds = xr.Dataset(
            {
                "u10": (("time", "latitude", "longitude"), np.random.rand(24, 4, 4).astype("float32")),
                "v10": (("time", "latitude", "longitude"), np.random.rand(24, 4, 4).astype("float32")),
            },
            coords={"time": wind_time, "latitude": lat, "longitude": lon},
        )
        wind_ds.to_netcdf(era5_dir / "era5_wind_20260712.nc")

        # Stale leftover from a previous run of a different recipe/variable
        # (waves) that was never cleared out of the shared output dir.
        waves_time = xr.date_range("2026-07-11T00:00:00", periods=24, freq="1h")
        waves_ds = xr.Dataset(
            {"swh": (("time", "latitude", "longitude"), np.random.rand(24, 4, 4).astype("float32"))},
            coords={"time": waves_time, "latitude": lat, "longitude": lon},
        )
        waves_ds.to_netcdf(era5_dir / "era5_waves_20260711.nc")

        recipe = Recipe(RecipeConfig(
            name="t", variable="wind",
            geographic_bounds=GeographicBounds(-10.0, -8.5, 40.0, 41.5),
            temporal_bounds=TemporalBounds("2026-07-12", "2026-07-12"),
        ))

        tree = DataTreeConverter.convert_downloaded_data(tmp_path, recipe=recipe)
        assert tree is not None
        era5_ds = tree["validation/era5/era5"].to_dataset()

        # Only the wind file's 24 timesteps should appear -- the stale waves
        # file must be ignored, not concatenated in.
        assert era5_ds.sizes["time"] == 24
        assert era5_ds.attrs["data_type"] == "era5_wind"
        assert "swh" not in era5_ds.data_vars
        assert "VHM0" not in era5_ds.data_vars
        assert "u10" in era5_ds.data_vars and not era5_ds["u10"].isnull().any()


class TestFromEra5Antimeridian:
    def _write_window_nc(self, path, lon_values, n_lat=3, n_time=3, start_hour="2026-07-12T00:00:00"):
        import numpy as np
        import xarray as xr

        lat = np.linspace(40.0, 42.0, n_lat)
        lon = np.array(lon_values)
        time = xr.date_range(start_hour, periods=n_time, freq="1h")
        u10 = np.random.rand(n_time, n_lat, len(lon)).astype("float32")
        v10 = np.random.rand(n_time, n_lat, len(lon)).astype("float32")
        ds = xr.Dataset(
            {
                "u10": (("time", "latitude", "longitude"), u10),
                "v10": (("time", "latitude", "longitude"), v10),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )
        ds.to_netcdf(path)

    def test_group_by_day_separates_window_pairs_from_single_files(self, tmp_path):
        from sar_validation.core.datatree_converter import _group_era5_paths_by_day

        paths = [
            tmp_path / "era5_wind_20260712_w0.nc",
            tmp_path / "era5_wind_20260712_w1.nc",
            tmp_path / "era5_wind_20260713.nc",
        ]
        groups = _group_era5_paths_by_day(paths)
        assert groups == {
            "era5_wind_20260712": [paths[0], paths[1]],
            "era5_wind_20260713": [paths[2]],
        }

    def test_stitch_helper_returns_none_if_window_missing(self, tmp_path):
        from sar_validation.core.datatree_converter import _stitch_antimeridian_window_files

        w0 = tmp_path / "era5_wind_20260712_w0.nc"
        self._write_window_nc(w0, [175.0, 180.0])

        assert _stitch_antimeridian_window_files([w0]) is None

    def test_stitches_two_window_files_into_one_contiguous_lon_axis(self, tmp_path):
        import numpy as np

        from sar_validation.core.datatree_converter import DataTreeConverter

        east = tmp_path / "era5_wind_20260712_w0.nc"
        west = tmp_path / "era5_wind_20260712_w1.nc"
        self._write_window_nc(east, [175.0, 177.5, 180.0])
        self._write_window_nc(west, [-180.0, -177.5, -175.0])

        ds = DataTreeConverter.from_era5([east, west], "wind")
        assert ds is not None
        lon = sorted(ds["lon"].values.tolist())
        # West window shifted by +360: [-180,-177.5,-175] -> [180,182.5,185]
        assert np.allclose(lon, [175.0, 177.5, 180.0, 182.5, 185.0])
        assert all(b > a for a, b in zip(lon, lon[1:]))  # strictly increasing after stitch
