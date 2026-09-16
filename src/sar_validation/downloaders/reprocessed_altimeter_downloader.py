"""
Download reprocessed (multi-year) along-track altimeter significant wave
height from Copernicus Marine.

Data source: WAVE_GLO_PHY_SWH_L3_MY_014_005
    One combined NetCDF file per day, covering every altimeter mission
    active that day, from 1991-08-03 to 2023-12-31. From 2024-01-01
    onward, use the near-real-time product instead (see
    altimeter_downloader.py).

Library usage::

    from sar_validation.downloaders.reprocessed_altimeter_downloader import ReprocessedAltimeterDownloader
    dl = ReprocessedAltimeterDownloader(output_dir=Path("data/run1/altimeter_reprocessed"))
    paths = dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                         start="2023-12-30", end="2023-12-31")
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .base import copernicus_marine_download_kwargs, normalize_datetime

__all__ = ["ReprocessedAltimeterDownloader", "MISSIONS", "COVERAGE_END", "DATASET_ID"]

DATASET_ID = "cci_obs-wave_glo_phy-swh_my_l3_PT1S-i"

#: Last day this product has data for. A request extending past this
#: date is clipped -- 2024-01-01 onward is covered by the near-real-time
#: product instead.
COVERAGE_END = "2023-12-31"

#: Mission activity windows. Keys are this product's own mission names,
#: exactly as they appear in a real downloaded file's "satellite"
#: variable flag_meanings (confirmed live, not guessed from the QUID's
#: own mission-name spellings, which differ for three of these:
#: "topex-poseidon" not "topex", and "sentinel-3_a"/"sentinel-3_b"/
#: "sentinel-6_a" with an underscore before the trailing letter). Dates
#: are from the WAVE_GLO_PHY_SWH_L3_MY_014_005 quality information
#: document (Table 2) and product user manual (Table 3). "orbit_key" is
#: the matching entry in orbit_coverage.SATELLITE_ORBIT_SPECS, used for
#: the pre-download orbit check -- deliberately a separate key space
#: from the mission name, since orbit_coverage.py's existing convention
#: for these satellites has no underscore.
MISSIONS: dict = {
    "ers-1":          {"orbit_key": "ers-1",       "start": "1991-08-03", "end": "1996-06-02"},
    "topex-poseidon": {"orbit_key": "topex",       "start": "1992-10-13", "end": "2005-10-04"},
    "ers-2":          {"orbit_key": "ers-2",       "start": "1995-05-14", "end": "2003-07-02"},
    # Jason-1's end date is 2013-06-21 per the product's own "List of
    # altimeters input data" table (its most specific source for exactly
    # this fact) -- the product's separate temporal-availability summary
    # table states 2012-03-03 instead, an inconsistency between the two
    # official documents themselves, not a choice made here.
    "jason-1":        {"orbit_key": "jason-1",     "start": "2002-01-15", "end": "2013-06-21"},
    "envisat":        {"orbit_key": "envisat",     "start": "2002-05-14", "end": "2012-04-08"},
    "jason-2":        {"orbit_key": "jason-2",     "start": "2008-07-04", "end": "2019-10-01"},
    "cryosat-2":      {"orbit_key": "cryosat-2",   "start": "2010-07-16", "end": "2023-12-31"},
    "saral":          {"orbit_key": "saral",       "start": "2013-03-14", "end": "2023-12-31"},
    "jason-3":        {"orbit_key": "jason-3",     "start": "2016-02-17", "end": "2023-12-31"},
    "sentinel-3_a":   {"orbit_key": "sentinel-3a", "start": "2016-07-01", "end": "2023-12-31"},
    "sentinel-3_b":   {"orbit_key": "sentinel-3b", "start": "2018-05-08", "end": "2023-12-31"},
    "sentinel-6_a":   {"orbit_key": "sentinel-6a", "start": "2020-12-17", "end": "2023-12-31"},
}

#: Same margin the altimeter dry-collocation predicate uses -- sized for
#: an along-track altimeter's narrow (~8km) footprint rather than a wide
#: swath instrument.
_ORBIT_MARGIN_KM = 12.0


def _missions_active_on(day: datetime) -> "list[str]":
    day_str = day.strftime("%Y-%m-%d")
    return [code for code, spec in MISSIONS.items() if spec["start"] <= day_str <= spec["end"]]


def _day_range(start_dt: datetime, end_dt: datetime) -> "list[datetime]":
    """Every UTC calendar day touched by [start_dt, end_dt], inclusive."""
    first = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    days = []
    day = first
    while day <= last:
        days.append(day)
        day += timedelta(days=1)
    return days


class ReprocessedAltimeterDownloader:
    """
    Download the reprocessed (multi-year) along-track altimeter
    significant wave height product from Copernicus Marine.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded NetCDF files.
    dry_run : bool
        If True, print what would be downloaded without downloading.
    force_download : bool
        If True, re-download a day's file even if it already exists.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False, force_download: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download

    def download(
        self,
        min_lon: float, max_lon: float, min_lat: float, max_lat: float,
        start: str, end: str,
    ) -> "list[Path]":
        """
        Download one combined NetCDF file per day in [start, end] that
        has at least one mission crossing the requested area, clipped
        to this product's 1991-08-03 to 2023-12-31 coverage.

        Returns
        -------
        list[Path]
            Paths to the downloaded NetCDF files.
        """
        from ..core.orbit_coverage import orbit_overlaps_bbox

        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for reprocessed altimeter downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        start_dt = datetime.fromisoformat(normalize_datetime(start))
        end_dt = datetime.fromisoformat(normalize_datetime(end))
        coverage_end_dt = datetime.fromisoformat(f"{COVERAGE_END}T23:59:59")
        end_dt = min(end_dt, coverage_end_dt)
        if end_dt < start_dt:
            print("Reprocessed altimeter: requested window starts after 2023-12-31 -- skipped.")
            return []

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: "list[Path]" = []

        for day in _day_range(start_dt, end_dt):
            day_start = day
            day_end = day.replace(hour=23, minute=59, second=59, microsecond=0)

            active = _missions_active_on(day)
            if not active:
                continue

            any_crossing = any(
                orbit_overlaps_bbox(
                    MISSIONS[code]["orbit_key"], day_start, day_end,
                    min_lon, max_lon, min_lat, max_lat,
                    margin_km=_ORBIT_MARGIN_KM,
                )
                for code in active
            )
            if not any_crossing:
                continue

            day_str = day.strftime("%Y%m%d")

            if self.dry_run:
                print(
                    f"[DRY RUN] Would download reprocessed altimeter data for "
                    f"{day.strftime('%Y-%m-%d')} (filter=*{day_str}*) to:\n  {self.output_dir}"
                )
                continue

            print(f"Downloading reprocessed altimeter data for {day.strftime('%Y-%m-%d')} …")
            try:
                result = copernicusmarine.get(
                    dataset_id=DATASET_ID,
                    filter=f"*{day_str}*",
                    output_directory=self.output_dir,
                    no_directories=True,
                    disable_progress_bar=True,
                    **copernicus_marine_download_kwargs(self.force_download),
                )
                if not result.files:
                    print("  No data for this day -- skipped.")
                    continue
                for f in result.files:
                    file_path = Path(f.file_path)
                    downloaded.append(file_path)
                    print(f"  Saved to {file_path}")
            except Exception as exc:
                print(f"  Skipping {day.strftime('%Y-%m-%d')}: {exc}")

        print(f"Downloaded {len(downloaded)} reprocessed altimeter file(s).")
        return downloaded
