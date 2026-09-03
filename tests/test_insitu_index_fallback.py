"""Tests for insitu_index_fallback (Copernicus Marine in-situ TAC index
file parsing, used when ARCO subsetting is unavailable for a dataset
part)."""

from __future__ import annotations

from datetime import datetime

from sar_validation.downloaders.insitu_index_fallback import (
    IndexRow,
    iter_index_rows,
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
)


def test_iter_index_rows_parses_data_lines_skipping_comment_header(tmp_path):
    index_path = tmp_path / "index_history.txt"
    index_path.write_text(_FIXTURE_INDEX)

    rows = list(iter_index_rows(index_path))

    assert len(rows) == 3
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
