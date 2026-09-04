"""Tests for insitu_index_fallback (Copernicus Marine in-situ TAC index
file parsing, used when ARCO subsetting is unavailable for a dataset
part)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from sar_validation.downloaders.insitu_index_fallback import (
    IndexRow,
    download_index_files,
    download_via_index,
    iter_index_rows,
    parse_platform_file,
    rows_matching_query,
)

_FIXTURE_INDEX = (  # noqa: E501
    "# Title : in-situ files catalog\n"
    "# Description : catalog of available in-situ files compliant with Marine Data Store\n"
    "# Project : Copernicus Marine In Situ TAC\n"
    "# Format version : 3.0\n"
    "# Date of update : 2026-08-30T10:55:41Z\n"
    "# product_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,geospatial_lon_max,time_coverage_start,time_coverage_end,institution,date_update,data_mode,parameters\n"  # noqa: E501
    "COP-AR-01,history/MO/AR_TS_MO_A-Sulafjorden.nc,62.4247,62.4283,6.0422,6.049,2022-03-01T00:00:00Z,2024-04-02T07:59:00Z,The Norwegian Public Roads Administration,2025-05-07T14:08:19Z,R,DEPH HCDT HCSP WSPD WDIR\n"  # noqa: E501
    "COP-GL-02,history/DC/GL_TS_DC_1301742.nc,-28.917,-11.368,-179.986,179.997,2023-05-15T17:00:00Z,2026-01-30T23:00:00Z,CLS,2026-05-05T21:19:02Z,D,EWCT NSCT EWCT_WS NSCT_WS\n"  # noqa: E501
    "COP-GLOBAL-01,history/BO/GL_PR_BO_58GS.nc,63.3516,64.84318,1.53416,4.04008,2021-06-23T02:38:02Z,2021-06-28T08:12:57Z,Institute of Marine Research,2025-05-05T02:40:04Z,M,BATH PRES NTRI NTRA\n"  # noqa: E501
    "COP-TEST-01,test/TS_ANTIMERIDIAN_EAST.nc,30.0,35.0,174.5,178.5,2023-06-01T00:00:00Z,2023-06-30T23:59:00Z,Test Org,2026-01-01T00:00:00Z,R,TEMP SALT\n"  # noqa: E501
)


def test_iter_index_rows_parses_data_lines_skipping_comment_header(tmp_path):
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = list(iter_index_rows(index_path))

    assert len(rows) == 4
    first = rows[0]
    assert first == IndexRow(
        file_name="history/MO/AR_TS_MO_A-Sulafjorden.nc",
        lat_min=62.4247, lat_max=62.4283, lon_min=6.0422, lon_max=6.049,
        time_start=datetime(2022, 3, 1, 0, 0, 0),
        time_end=datetime(2024, 4, 2, 7, 59, 0),
        institution="The Norwegian Public Roads Administration",
        parameters={"DEPH", "HCDT", "HCSP", "WSPD", "WDIR"},
    )


def test_iter_index_rows_on_missing_data_returns_empty(tmp_path):
    index_path = tmp_path / "index_empty.txt"
    index_path.write_text("# Title : in-situ files catalog\n# just comments\n")

    assert list(iter_index_rows(index_path)) == []


def test_rows_matching_query_filters_by_variable_time_and_bbox(tmp_path):
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = rows_matching_query(
        index_path,
        min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
        start=datetime(2023, 1, 1), end=datetime(2023, 12, 31),
        wanted_variables={"WSPD", "WDIR"},
    )

    assert [r.file_name for r in rows] == ["history/MO/AR_TS_MO_A-Sulafjorden.nc"]


def test_rows_matching_query_treats_wrapped_bbox_row_as_matching_any_window(tmp_path):
    """The drifter row's own bbox is [-179.986, 179.997] -- it crossed the
    antimeridian, not the whole globe -- so it must match a query window
    anywhere, the same way a genuinely wide platform track would."""
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = rows_matching_query(
        index_path,
        min_lon=170.0, max_lon=-170.0, min_lat=-30.0, max_lat=-10.0,
        start=datetime(2023, 1, 1), end=datetime(2026, 12, 31),
        wanted_variables={"EWCT", "NSCT"},
    )

    assert [r.file_name for r in rows] == ["history/DC/GL_TS_DC_1301742.nc"]


def test_rows_matching_query_excludes_rows_outside_time_window(tmp_path):
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = rows_matching_query(
        index_path,
        min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
        start=datetime(2010, 1, 1), end=datetime(2010, 12, 31),
        wanted_variables={"WSPD", "WDIR"},
    )

    assert rows == []


def test_rows_matching_query_with_nonwrapping_row_inside_wrapped_query_window(tmp_path):
    """A row whose own bbox does not wrap (lon_max - lon_min <= 180), positioned
    near the antimeridian, should match when queried with a wrapping bbox that
    spans the dateline. The _row_matches_bbox function should split the wrapping
    query into two windows and check overlap against both."""
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = rows_matching_query(
        index_path,
        min_lon=170.0, max_lon=-170.0, min_lat=25.0, max_lat=40.0,
        start=datetime(2023, 6, 1), end=datetime(2023, 6, 30),
        wanted_variables={"TEMP", "SALT"},
    )

    assert [r.file_name for r in rows] == ["test/TS_ANTIMERIDIAN_EAST.nc"]


def test_rows_matching_query_with_nonwrapping_row_outside_wrapped_query_windows(tmp_path):
    """A row whose own bbox does not wrap, positioned far from the antimeridian
    (lon 1.5-4.0), should not match when queried with a wrapping bbox that spans
    the dateline (170 to -170). The row falls outside both split windows."""
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = rows_matching_query(
        index_path,
        min_lon=170.0, max_lon=-170.0, min_lat=60.0, max_lat=65.0,
        start=datetime(2021, 6, 23), end=datetime(2021, 6, 28),
        wanted_variables={"BATH", "PRES"},
    )

    assert rows == []


def test_download_index_files_writes_file_list_and_calls_get(tmp_path):
    rows = [
        IndexRow(
            file_name="history/MO/AR_TS_MO_A-Sulafjorden.nc",
            lat_min=62.4247, lat_max=62.4283, lon_min=6.0422, lon_max=6.049,
            time_start=datetime(2022, 3, 1),
            time_end=datetime(2024, 4, 2),
            institution="x", parameters={"HCDT", "HCSP"},
        ),
    ]
    downloaded_path = tmp_path / "AR_TS_MO_A-Sulafjorden.nc"

    fake_module = MagicMock()
    fake_file = MagicMock(file_status="DOWNLOADED", file_path=downloaded_path)
    fake_module.get.return_value = MagicMock(files=[fake_file])

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        paths = download_index_files(
            "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", "history", rows, tmp_path,
        )

    assert paths == [downloaded_path]
    call_kwargs = fake_module.get.call_args.kwargs
    assert call_kwargs["dataset_id"] == "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"
    assert call_kwargs["dataset_part"] == "history"
    assert call_kwargs["no_directories"] is True
    # Verify a _file_list_*.txt file was passed to copernicusmarine.get(),
    # containing the correct row names (the file is cleaned up after the call).
    file_list_arg = call_kwargs["file_list"]
    assert file_list_arg.startswith(str(tmp_path))
    assert "_file_list_" in file_list_arg and file_list_arg.endswith(".txt")


def test_download_index_files_returns_empty_for_no_rows(tmp_path):
    assert download_index_files("dataset", "history", [], tmp_path) == []


def test_download_index_files_uses_a_unique_file_list_name(tmp_path):
    """A fixed _file_list.txt name would risk collisions once work_dir is a
    directory shared across recipe runs (Task 7's shared cache) -- confirm
    the file list gets a unique name and is cleaned up after the call."""
    rows = [
        IndexRow(
            file_name="history/MO/AR_TS_MO_A-Sulafjorden.nc",
            lat_min=62.4247, lat_max=62.4283, lon_min=6.0422, lon_max=6.049,
            time_start=__import__("datetime").datetime(2022, 3, 1),
            time_end=__import__("datetime").datetime(2024, 4, 2),
            institution="x", parameters={"HCDT", "HCSP"},
        ),
    ]
    downloaded_path = tmp_path / "AR_TS_MO_A-Sulafjorden.nc"
    fake_module = MagicMock()
    fake_file = MagicMock(file_status="DOWNLOADED", file_path=downloaded_path)
    fake_module.get.return_value = MagicMock(files=[fake_file])

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        download_index_files(
            "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", "history", rows, tmp_path,
        )

    # No leftover _file_list*.txt scratch file in a directory meant to
    # persist only the downloaded platform files themselves.
    assert list(tmp_path.glob("_file_list*.txt")) == []


def _write_mooring_fixture(path):
    """Mirrors AR_TS_MO_A-Sulafjorden.nc's real layout: scalar LONGITUDE/
    LATITUDE plus per-observation PRECISE_LONGITUDE/PRECISE_LATITUDE, and a
    (TIME, DEPTH) DEPH data variable."""
    time = pd.date_range("2023-01-01", periods=3, freq="h")
    ds = xr.Dataset(
        data_vars={
            "HCDT": (("TIME", "DEPTH"), [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]),
            "HCSP": (("TIME", "DEPTH"), [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        },
        coords={
            "TIME": time,
            "PRECISE_LONGITUDE": ("TIME", [6.04, 6.05, 6.06]),
            "PRECISE_LATITUDE": ("TIME", [62.42, 62.43, 62.44]),
            "DEPH": (("TIME", "DEPTH"), [[5.0, 25.0], [5.0, 25.0], [5.0, 25.0]]),
            "LONGITUDE": 6.045,
            "LATITUDE": 62.43,
        },
        attrs={"platform_code": "A-Sulafjorden"},
    )
    ds.to_netcdf(path)


def _write_argo_fixture(path):
    """Mirrors GL_TS_PF_13857.nc's real layout: only per-observation
    LONGITUDE/LATITUDE, and a (TIME, DEPTH) PRES data variable instead of
    DEPH."""
    time = pd.date_range("2023-06-01", periods=2, freq="D")
    ds = xr.Dataset(
        data_vars={
            "EWCT": (("TIME", "DEPTH"), [[0.05], [0.07]]),
            "NSCT": (("TIME", "DEPTH"), [[-0.02], [-0.01]]),
        },
        coords={
            "TIME": time,
            "LONGITUDE": ("TIME", [1.0, 1.2]),
            "LATITUDE": ("TIME", [3.0, 3.1]),
            "PRES": (("TIME", "DEPTH"), [[10.0], [10.0]]),
        },
        attrs={"platform_code": "13857"},
    )
    ds.to_netcdf(path)


def _write_surface_only_fixture(path):
    """Surface-only platform with no depth variable: only per-observation
    LONGITUDE/LATITUDE and TIME-dimensioned data variables (no DEPTH
    dimension, no DEPH, no PRES)."""
    time = pd.date_range("2023-03-01", periods=2, freq="D")
    ds = xr.Dataset(
        data_vars={
            "TEMP": ("TIME", [15.5, 16.2]),
            "SALT": ("TIME", [35.1, 35.2]),
        },
        coords={
            "TIME": time,
            "LONGITUDE": ("TIME", [10.0, 10.1]),
            "LATITUDE": ("TIME", [45.0, 45.1]),
        },
        attrs={"platform_code": "SURFACE_001", "institution": "Test Lab"},
    )
    ds.to_netcdf(path)


def test_parse_platform_file_mooring_layout_uses_precise_coords_and_deph(tmp_path):
    nc_path = tmp_path / "mooring.nc"
    _write_mooring_fixture(nc_path)

    df = parse_platform_file(
        nc_path, wanted_variables={"HCDT", "HCSP"},
        min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-01-02"),
        min_depth=-20.0, max_depth=20.0,
        platform_type_code="MO",
    )

    assert set(df.columns) == {
        "variable", "platform_id", "platform_type", "time",
        "longitude", "latitude", "depth", "value", "institution",
    }
    # DEPH=25.0 falls outside [-20, 20] and must be dropped; DEPH=5.0 kept,
    # for both HCDT and HCSP, across all 3 timestamps.
    assert len(df) == 6
    assert set(df["variable"]) == {"HCDT", "HCSP"}
    assert (df["depth"] == 5.0).all()
    assert (df["platform_id"] == "A-Sulafjorden").all()
    assert (df["platform_type"] == "MO").all()
    assert np.isclose(df.loc[df["longitude"].round(2) == 6.04, "longitude"].iloc[0], 6.04)


def test_parse_platform_file_argo_layout_uses_plain_coords_and_pres(tmp_path):
    nc_path = tmp_path / "argo.nc"
    _write_argo_fixture(nc_path)

    df = parse_platform_file(
        nc_path, wanted_variables={"EWCT", "NSCT"},
        min_lon=0.0, max_lon=2.0, min_lat=0.0, max_lat=5.0,
        start=pd.Timestamp("2023-06-01"), end=pd.Timestamp("2023-06-02"),
        min_depth=0.0, max_depth=20.0,
        platform_type_code="PF",
    )

    assert len(df) == 4  # 2 variables x 2 timestamps
    assert set(df["variable"]) == {"EWCT", "NSCT"}
    assert (df["platform_id"] == "13857").all()
    assert (df["platform_type"] == "PF").all()


def test_parse_platform_file_no_requested_variable_present_returns_empty(tmp_path):
    nc_path = tmp_path / "mooring.nc"
    _write_mooring_fixture(nc_path)

    df = parse_platform_file(
        nc_path, wanted_variables={"WSPD", "WDIR"},
        min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-01-02"),
        min_depth=-20.0, max_depth=20.0,
        platform_type_code="MO",
    )

    assert df.empty


def test_parse_platform_file_surface_only_defaults_depth_to_zero(tmp_path):
    """Platform with no depth variable (no DEPH, PRES, or DEPTH dimension)
    must default all rows to depth=0.0."""
    nc_path = tmp_path / "surface.nc"
    _write_surface_only_fixture(nc_path)

    df = parse_platform_file(
        nc_path, wanted_variables={"TEMP", "SALT"},
        min_lon=0.0, max_lon=20.0, min_lat=40.0, max_lat=50.0,
        start=pd.Timestamp("2023-03-01"), end=pd.Timestamp("2023-03-03"),
        min_depth=-5.0, max_depth=5.0,
        platform_type_code="SU",
    )

    assert len(df) == 4  # 2 variables x 2 timestamps
    assert set(df["variable"]) == {"TEMP", "SALT"}
    assert (df["depth"] == 0.0).all()
    assert (df["platform_id"] == "SURFACE_001").all()
    assert (df["platform_type"] == "SU").all()
    assert (df["institution"] == "Test Lab").all()


def test_parse_platform_file_surface_only_depth_filter_excludes_zero(tmp_path):
    """Depth filter that excludes 0.0 (e.g., min_depth=10, max_depth=20)
    must drop all rows from a surface-only platform."""
    nc_path = tmp_path / "surface.nc"
    _write_surface_only_fixture(nc_path)

    df = parse_platform_file(
        nc_path, wanted_variables={"TEMP", "SALT"},
        min_lon=0.0, max_lon=20.0, min_lat=40.0, max_lat=50.0,
        start=pd.Timestamp("2023-03-01"), end=pd.Timestamp("2023-03-03"),
        min_depth=10.0, max_depth=20.0,
        platform_type_code="SU",
    )

    assert df.empty


def test_download_via_index_returns_none_when_no_rows_match(tmp_path):
    index_path = tmp_path / "cache" / "index_history.txt"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(_FIXTURE_INDEX)

    fake_module = MagicMock()
    fake_module.get.return_value = MagicMock(
        files=[MagicMock(filename="index_history.txt", file_path=index_path)],
    )

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = download_via_index(
            dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr",
            dataset_part="history",
            min_lon=-170.0, max_lon=-160.0, min_lat=80.0, max_lat=85.0,
            start_dt="2023-01-01T00:00:00", end_dt="2023-12-31T00:00:00",
            min_depth=-20.0, max_depth=20.0,
            wanted_variables={"WSPD", "WDIR"},
            dest_path=tmp_path / "out.csv",
            work_dir=tmp_path / "cache",
        )

    assert result is None
    assert not (tmp_path / "out.csv").exists()


def test_download_via_index_writes_combined_csv_from_matched_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_path = cache_dir / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)
    mooring_nc = cache_dir / "AR_TS_MO_A-Sulafjorden.nc"
    _write_mooring_fixture(mooring_nc)

    fake_module = MagicMock()

    def fake_get(**kwargs):
        if kwargs.get("index_parts"):
            return MagicMock(files=[MagicMock(filename="index_history.txt", file_path=index_path)])
        return MagicMock(files=[MagicMock(file_path=mooring_nc)])

    fake_module.get.side_effect = fake_get

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = download_via_index(
            dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr",
            dataset_part="history",
            min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
            start_dt="2023-01-01T00:00:00", end_dt="2023-01-02T00:00:00",
            min_depth=-20.0, max_depth=20.0,
            wanted_variables={"HCDT", "HCSP", "WSPD", "WDIR"},
            dest_path=tmp_path / "out.csv",
            work_dir=cache_dir,
        )

    assert result == tmp_path / "out.csv"
    df = pd.read_csv(tmp_path / "out.csv")
    assert set(df["variable"]) == {"HCDT", "HCSP"}
