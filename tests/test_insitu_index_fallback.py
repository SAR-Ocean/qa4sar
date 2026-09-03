"""Tests for insitu_index_fallback (Copernicus Marine in-situ TAC index
file parsing, used when ARCO subsetting is unavailable for a dataset
part)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from sar_validation.downloaders.insitu_index_fallback import (
    IndexRow,
    download_index_files,
    iter_index_rows,
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
    file_list_path = tmp_path / "_file_list.txt"
    assert file_list_path.read_text().strip() == "history/MO/AR_TS_MO_A-Sulafjorden.nc"


def test_download_index_files_returns_empty_for_no_rows(tmp_path):
    assert download_index_files("dataset", "history", [], tmp_path) == []
