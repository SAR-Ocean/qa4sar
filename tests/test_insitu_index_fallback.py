"""Tests for insitu_index_fallback (Copernicus Marine in-situ TAC index
file parsing, used when ARCO subsetting is unavailable for a dataset
part)."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from sar_validation.downloaders.insitu_index_fallback import (
    _INDEX_MAX_AGE,
    IndexRow,
    download_index_files,
    download_via_index,
    fetch_index_file,
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


def test_fetch_index_file_reuses_a_fresh_cached_copy_without_a_network_call(tmp_path):
    cached_path = tmp_path / "index_history.txt"
    cached_path.write_text(_FIXTURE_INDEX)

    fake_module = MagicMock()

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = fetch_index_file("dataset", "history", tmp_path)

    assert result == cached_path
    fake_module.get.assert_not_called()


def test_fetch_index_file_refetches_a_stale_cached_copy(tmp_path):
    cached_path = tmp_path / "index_history.txt"
    cached_path.write_text(_FIXTURE_INDEX)
    stale_time = time.time() - (_INDEX_MAX_AGE.total_seconds() + 3600)
    os.utime(cached_path, (stale_time, stale_time))

    fake_module = MagicMock()
    fake_module.get.return_value = MagicMock(
        files=[MagicMock(filename="index_history.txt", file_path=cached_path)],
    )

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = fetch_index_file("dataset", "history", tmp_path)

    assert result == cached_path
    fake_module.get.assert_called_once()
    assert fake_module.get.call_args.kwargs["overwrite"] is True
    # no_directories=True is what makes the staleness check above able to
    # find the cached copy at work_dir/index_history.txt at all --
    # copernicusmarine's default layout nests index files under a
    # product-id/version subdirectory it does not otherwise expose to the
    # caller, so a flat cached_path lookup without this would never hit.
    assert fake_module.get.call_args.kwargs["no_directories"] is True


def test_fetch_index_file_force_download_refetches_even_a_fresh_copy(tmp_path):
    cached_path = tmp_path / "index_history.txt"
    cached_path.write_text(_FIXTURE_INDEX)

    fake_module = MagicMock()
    fake_module.get.return_value = MagicMock(
        files=[MagicMock(filename="index_history.txt", file_path=cached_path)],
    )

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        fetch_index_file("dataset", "history", tmp_path, force_download=True)

    fake_module.get.assert_called_once()


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
    near the antimeridian, matches a query with a wrapping bbox that spans
    the dateline, by overlapping one of the two windows the wrapping query
    splits into."""
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
    captured_file_list_contents = {}

    def fake_get(**kwargs):
        # The real file list is unlinked in a `finally` right after this
        # call returns, so its contents must be captured here rather than
        # read back afterward.
        captured_file_list_contents["text"] = Path(kwargs["file_list"]).read_text()
        return MagicMock(files=[fake_file], files_not_found=None, total_size=1.23)

    fake_module.get.side_effect = fake_get

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        paths = download_index_files(
            "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", "history", rows, tmp_path,
        )

    assert paths == [downloaded_path]
    call_kwargs = fake_module.get.call_args.kwargs
    assert call_kwargs["dataset_id"] == "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"
    assert call_kwargs["dataset_part"] == "history"
    assert call_kwargs["no_directories"] is True
    # A _file_list_*.txt file was passed to copernicusmarine.get(),
    # containing the correct row names, and cleaned up after the call.
    file_list_arg = call_kwargs["file_list"]
    assert file_list_arg.startswith(str(tmp_path))
    assert "_file_list_" in file_list_arg and file_list_arg.endswith(".txt")
    assert captured_file_list_contents["text"] == "history/MO/AR_TS_MO_A-Sulafjorden.nc\n"
    assert not Path(file_list_arg).exists()


def test_download_index_files_returns_empty_for_no_rows(tmp_path):
    assert download_index_files("dataset", "history", [], tmp_path) == []


def test_download_index_files_uses_a_unique_file_list_name(tmp_path):
    """A fixed _file_list.txt name would risk collisions when work_dir is a
    directory shared across concurrent recipe runs -- confirm the file
    list gets a unique name and is cleaned up after the call."""
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
    fake_module.get.return_value = MagicMock(files=[fake_file], files_not_found=None, total_size=1.23)

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
        "longitude", "latitude", "depth", "value", "value_qc", "institution",
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


def _write_mooring_fixture_with_qc(path):
    """Mirrors AR_TS_MO_A-Sulafjorden.nc's real layout, plus a "<VAR>_QC"
    companion for HCDT only -- HCSP is left without one, as would happen
    for a parameter whose QC code was never populated."""
    time = pd.date_range("2023-01-01", periods=2, freq="h")
    ds = xr.Dataset(
        data_vars={
            "HCDT": (("TIME", "DEPTH"), [[10.0], [12.0]]),
            "HCDT_QC": (("TIME", "DEPTH"), [[1], [4]]),
            "HCSP": (("TIME", "DEPTH"), [[0.1], [0.3]]),
        },
        coords={
            "TIME": time,
            "PRECISE_LONGITUDE": ("TIME", [6.04, 6.05]),
            "PRECISE_LATITUDE": ("TIME", [62.42, 62.43]),
            "DEPH": (("TIME", "DEPTH"), [[5.0], [5.0]]),
            "LONGITUDE": 6.045,
            "LATITUDE": 62.43,
        },
        attrs={"platform_code": "A-Sulafjorden"},
    )
    ds.to_netcdf(path)


def test_parse_platform_file_carries_per_variable_qc_code(tmp_path):
    """HCDT's own QC code must travel with its value row-for-row; HCSP has
    no QC companion in the file, so its value_qc must be NaN rather than
    silently reusing HCDT's or being dropped from the schema -- the same
    "no QC code, no trustworthy value" convention from_insitu_csv already
    applies to the ARCO/subset() path."""
    nc_path = tmp_path / "mooring_qc.nc"
    _write_mooring_fixture_with_qc(nc_path)

    df = parse_platform_file(
        nc_path, wanted_variables={"HCDT", "HCSP"},
        min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-01-02"),
        min_depth=-20.0, max_depth=20.0,
        platform_type_code="MO",
    )

    hcdt = df[df["variable"] == "HCDT"].sort_values("time")
    assert list(hcdt["value_qc"]) == [1, 4]
    hcsp = df[df["variable"] == "HCSP"]
    assert hcsp["value_qc"].isna().all()


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


from sar_validation.downloaders.insitu_index_fallback import (
    _observations_overlap_window,
    _row_is_effectively_stationary,
)


def test_observations_overlap_window_true_when_any_point_matches():
    times = pd.to_datetime(["2023-01-01", "2023-06-15", "2024-01-01"])
    lons = np.array([6.04, 6.05, 170.0])
    lats = np.array([62.42, 62.43, -20.0])

    assert _observations_overlap_window(
        times.values, lons, lats,
        min_lon=6.0, max_lon=6.1, min_lat=62.4, max_lat=62.5,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-12-31"),
    ) is True


def test_observations_overlap_window_false_when_no_point_matches():
    times = pd.to_datetime(["2001-01-01", "2025-09-01"])
    lons = np.array([-53.9, 79.1])
    lats = np.array([1e-5, 55.25])

    assert _observations_overlap_window(
        times.values, lons, lats,
        min_lon=6.0, max_lon=6.1, min_lat=62.4, max_lat=62.5,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-12-31"),
    ) is False


def test_observations_overlap_window_handles_wrapped_query_bbox():
    """A point at 175 deg E must match a query bbox that wraps the
    antimeridian (min_lon=170, max_lon=-170), the same wrap convention
    used everywhere else in this module."""
    times = pd.to_datetime(["2023-06-01"])
    lons = np.array([175.0])
    lats = np.array([-25.0])

    assert _observations_overlap_window(
        times.values, lons, lats,
        min_lon=170.0, max_lon=-170.0, min_lat=-30.0, max_lat=-10.0,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-12-31"),
    ) is True


def test_observations_overlap_window_ignores_nan_coordinates():
    """Real in-situ files carry occasional NaN lon/lat/time at bad
    observations -- these must not crash the check or count as a match."""
    times = pd.to_datetime(["2023-06-01", "NaT"])
    lons = np.array([np.nan, 6.05])
    lats = np.array([62.42, np.nan])

    assert _observations_overlap_window(
        times.values, lons, lats,
        min_lon=6.0, max_lon=6.1, min_lat=62.4, max_lat=62.5,
        start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-12-31"),
    ) is False


def test_row_is_effectively_stationary_true_for_a_tight_mooring_bbox():
    """A real anchored mooring's own reported index bbox (e.g.
    AR_TS_MO_A-Sulafjorden.nc: lat 62.4247-62.4283, lon 6.0422-6.049) spans
    well under a tenth of a degree -- GPS jitter, not real movement."""
    row = IndexRow(
        file_name="history/MO/AR_TS_MO_A-Sulafjorden.nc",
        lat_min=62.4247, lat_max=62.4283, lon_min=6.0422, lon_max=6.049,
        time_start=datetime(2022, 3, 1),
        time_end=datetime(2024, 4, 2),
        institution="x", parameters={"HCDT"},
    )
    assert _row_is_effectively_stationary(row) is True


def test_row_is_effectively_stationary_false_for_a_wide_moving_track():
    """A real research-vessel platform's reported bbox (e.g.
    GL_PR_AD_FNCM.nc: lon -53.9 to 79.1, lat 1e-5 to 55.25 over a 24-year
    deployment) spans most of an ocean basin -- genuinely moving, or wide
    fixed-network coverage (e.g. HF-radar) -- either way, worth checking."""
    row = IndexRow(
        file_name="history/AD/GL_PR_AD_FNCM.nc",
        lat_min=1e-5, lat_max=55.2521, lon_min=-53.90302, lon_max=79.13086,
        time_start=datetime(2001, 1, 4),
        time_end=datetime(2025, 9, 14),
        institution="x", parameters={"EWCT", "NSCT"},
    )
    assert _row_is_effectively_stationary(row) is False


def test_row_is_effectively_stationary_respects_a_custom_threshold():
    row = IndexRow(
        file_name="history/MO/example.nc",
        lat_min=62.0, lat_max=62.2, lon_min=6.0, lon_max=6.2,
        time_start=datetime(2022, 1, 1),
        time_end=datetime(2022, 1, 2),
        institution="x", parameters={"HCDT"},
    )
    assert _row_is_effectively_stationary(row, threshold_deg=0.1) is False
    assert _row_is_effectively_stationary(row, threshold_deg=0.5) is True


def test_row_overlaps_window_uses_local_cached_file_when_present(tmp_path):
    """A row whose file is already in work_dir must be checked locally --
    no copernicusmarine call at all."""
    from sar_validation.downloaders.insitu_index_fallback import row_overlaps_window

    row = IndexRow(
        file_name="history/MO/AR_TS_MO_A-Sulafjorden.nc",
        lat_min=62.4, lat_max=62.5, lon_min=6.0, lon_max=6.1,
        time_start=datetime(2023, 1, 1),
        time_end=datetime(2023, 12, 31),
        institution="x", parameters={"HCDT"},
    )
    nc_path = tmp_path / "AR_TS_MO_A-Sulafjorden.nc"
    _write_mooring_fixture(nc_path)

    fake_module = MagicMock()  # must not be called
    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = row_overlaps_window(
            row, "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", "history", tmp_path,
            min_lon=6.0, max_lon=6.1, min_lat=62.4, max_lat=62.5,
            start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-01-02"),
        )

    assert result is True
    fake_module.get.assert_not_called()


def test_row_overlaps_window_fails_open_on_any_error(tmp_path):
    """A row whose remote check cannot be completed (here: dry_run itself
    raises) must be treated as a match rather than silently dropped."""
    from sar_validation.downloaders.insitu_index_fallback import row_overlaps_window

    row = IndexRow(
        file_name="history/MO/does_not_exist_locally.nc",
        lat_min=1.0, lat_max=2.0, lon_min=1.0, lon_max=2.0,
        time_start=datetime(2023, 1, 1),
        time_end=datetime(2023, 12, 31),
        institution="x", parameters={"HCDT"},
    )

    fake_module = MagicMock()
    fake_module.get.side_effect = RuntimeError("network error")
    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = row_overlaps_window(
            row, "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", "history", tmp_path,
            min_lon=1.0, max_lon=2.0, min_lat=1.0, max_lat=2.0,
            start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-01-02"),
        )

    assert result is True


def test_row_overlaps_window_skips_check_entirely_for_a_stationary_row(tmp_path):
    """A row with a tight (mooring-sized) reported bbox must skip the
    check entirely -- no local file access, no copernicusmarine call --
    trusting rows_matching_query's own bbox/time match directly."""
    from sar_validation.downloaders.insitu_index_fallback import row_overlaps_window

    row = IndexRow(
        file_name="history/MO/does_not_exist_anywhere.nc",
        lat_min=62.4247, lat_max=62.4283, lon_min=6.0422, lon_max=6.049,
        time_start=datetime(2022, 3, 1),
        time_end=datetime(2024, 4, 2),
        institution="x", parameters={"HCDT"},
    )

    fake_module = MagicMock()  # must not be called
    with patch.dict("sys.modules", {"copernicusmarine": fake_module}):
        result = row_overlaps_window(
            row, "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", "history", tmp_path,
            min_lon=6.0, max_lon=6.1, min_lat=62.4, max_lat=62.5,
            start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2023-01-02"),
        )

    assert result is True
    fake_module.get.assert_not_called()


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
        return MagicMock(files=[MagicMock(file_path=mooring_nc)], files_not_found=None, total_size=1.23)

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


def test_download_via_index_skips_rows_that_do_not_overlap_the_window(tmp_path):
    from sar_validation.downloaders.insitu_index_fallback import download_via_index

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_path = cache_dir / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    fake_module = MagicMock()
    fake_module.get.return_value = MagicMock(
        files=[MagicMock(filename="index_history.txt", file_path=index_path)],
    )

    with patch.dict("sys.modules", {"copernicusmarine": fake_module}), patch(
        "sar_validation.downloaders.insitu_index_fallback.row_overlaps_window",
        return_value=False,
    ) as mock_overlap, patch(
        "sar_validation.downloaders.insitu_index_fallback.download_index_files",
    ) as mock_download:
        result = download_via_index(
            dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr",
            dataset_part="history",
            min_lon=0.0, max_lon=10.0, min_lat=60.0, max_lat=65.0,
            start_dt="2023-01-01T00:00:00", end_dt="2023-01-02T00:00:00",
            min_depth=-20.0, max_depth=20.0,
            wanted_variables={"HCDT", "HCSP"},
            dest_path=tmp_path / "out.csv",
            work_dir=cache_dir,
        )

    assert result is None
    mock_overlap.assert_called_once()
    mock_download.assert_not_called()


_TWO_PLATFORM_TYPES_INDEX = (
    "# Title : in-situ files catalog\n"
    "# product_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,geospatial_lon_max,time_coverage_start,time_coverage_end,institution,date_update,data_mode,parameters\n"  # noqa: E501
    "COP-MO-01,history/MO/MOORING_1.nc,60.0,61.0,5.0,6.0,2023-01-01T00:00:00Z,2023-12-31T00:00:00Z,Org,2023-01-01T00:00:00Z,R,WSPD\n"  # noqa: E501
    "COP-HF-01,history/HF/RADAR_1.nc,60.0,61.0,5.0,6.0,2023-01-01T00:00:00Z,2023-12-31T00:00:00Z,Org,2023-01-01T00:00:00Z,R,WSPD\n"  # noqa: E501
)


def test_download_via_index_filters_rows_by_platform_code_before_downloading(tmp_path):
    """platform_codes must narrow which rows even reach
    download_index_files -- unlike a post-hoc CSV filter, this is what
    actually avoids fetching an irrelevant platform's whole-archive file."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index_history.txt").write_text(_TWO_PLATFORM_TYPES_INDEX)

    with patch.dict("sys.modules", {"copernicusmarine": MagicMock()}), patch(
        "sar_validation.downloaders.insitu_index_fallback.row_overlaps_window",
        return_value=True,
    ), patch(
        "sar_validation.downloaders.insitu_index_fallback.download_index_files",
        return_value=[],
    ) as mock_download:
        download_via_index(
            dataset_id="dataset",
            dataset_part="history",
            min_lon=0.0, max_lon=10.0, min_lat=55.0, max_lat=65.0,
            start_dt="2023-06-01T00:00:00", end_dt="2023-06-02T00:00:00",
            min_depth=-20.0, max_depth=20.0,
            wanted_variables={"WSPD"},
            dest_path=tmp_path / "out.csv",
            work_dir=cache_dir,
            platform_codes={"MO"},
        )

    downloaded_rows = mock_download.call_args.args[2]
    assert [r.file_name for r in downloaded_rows] == ["history/MO/MOORING_1.nc"]


def test_download_via_index_logs_a_warning_on_duplicate_file_stems(tmp_path, caplog):
    """Two index rows resolving to the same on-disk file stem (from
    different platform-type directories) must not silently collide --
    row_by_stem keeps one, but the collision itself is now observable."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_text = (
        "# Title : in-situ files catalog\n"
        "# product_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,geospatial_lon_max,time_coverage_start,time_coverage_end,institution,date_update,data_mode,parameters\n"  # noqa: E501
        "COP-1,history/MO/DUP.nc,60.0,61.0,5.0,6.0,2023-01-01T00:00:00Z,2023-12-31T00:00:00Z,Org,2023-01-01T00:00:00Z,R,WSPD\n"  # noqa: E501
        "COP-2,history/AD/DUP.nc,60.0,61.0,5.0,6.0,2023-01-01T00:00:00Z,2023-12-31T00:00:00Z,Org,2023-01-01T00:00:00Z,R,WSPD\n"  # noqa: E501
    )
    (cache_dir / "index_history.txt").write_text(index_text)
    dummy_nc = cache_dir / "DUP.nc"
    dummy_nc.touch()

    with patch.dict("sys.modules", {"copernicusmarine": MagicMock()}), patch(
        "sar_validation.downloaders.insitu_index_fallback.row_overlaps_window",
        return_value=True,
    ), patch(
        "sar_validation.downloaders.insitu_index_fallback.download_index_files",
        return_value=[dummy_nc],
    ), patch(
        "sar_validation.downloaders.insitu_index_fallback.parse_platform_file",
        return_value=pd.DataFrame(),
    ), caplog.at_level("WARNING"):
        download_via_index(
            dataset_id="dataset",
            dataset_part="history",
            min_lon=0.0, max_lon=10.0, min_lat=55.0, max_lat=65.0,
            start_dt="2023-06-01T00:00:00", end_dt="2023-06-02T00:00:00",
            min_depth=-20.0, max_depth=20.0,
            wanted_variables={"WSPD"},
            dest_path=tmp_path / "out.csv",
            work_dir=cache_dir,
        )

    assert "Duplicate platform file stem" in caplog.text
