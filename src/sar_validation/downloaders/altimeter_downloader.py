"""
Download along-track altimeter significant wave height (and 1 Hz wind speed)
from Copernicus Marine.

Data source: WAVE_GLO_PHY_SWH_L3_NRT_014_001
    One NetCDF dataset per satellite mission, at two possible frequencies:
        1 Hz  (~7km along-track resolution):   VAVH, VAVH_UNFILTERED, WIND_SPEED
        5 Hz  (~1.4km along-track resolution): VAVH, VAVH_UNFILTERED, VAVH_UNCERTAINTY
    5 Hz is only produced for 6 of the missions (2026-03-09 onwards)
    1 Hz is available for all missions (2024-01-01 onwards).

Library usage::

    from sar_validation.downloaders.altimeter_downloader import AltimeterDownloader
    dl = AltimeterDownloader(output_dir=Path("data/run1/altimeter"))
    paths = dl.download(min_lon=-20, max_lon=0, min_lat=35, max_lat=60,
                         start="2026-06-01", end="2026-06-02")

CLI usage::

    python -m sar_validation.downloaders.altimeter_downloader \\
        --min-lon -20 --max-lon 0 --min-lat 35 --max-lat 60 \\
        --start 2026-06-01 --end 2026-06-02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from .base import (
    build_output_dir,
    copernicus_marine_download_kwargs,
    normalize_datetime,
    split_antimeridian_bbox,
)

__all__ = ["AltimeterDownloader"]

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

DATASET_ID_TEMPLATE = {
    "1hz": "cmems_obs-wave_glo_phy-swh_nrt_{sat}-l3_PT1S",
    "5hz": "cmems_obs-wave_glo_phy-swh_nrt_{sat}-l3-1km_PT0.2S-i",
}

# Satellite code -> human-readable mission name, per the WAVE_GLO_PHY_SWH_L3_NRT_014_001
# Product User Manual (CMEMS-WAV-PUM-014-001, issue 1.2).
SATELLITES_1HZ = {
    "al":   "Saral/AltiKa",
    "c2":   "CryoSat-2",
    "cfo":  "CFOSAT",
    "h2b":  "HaiYang-2B",
    "h2c":  "HaiYang-2C",   # frozen since 2026-05-20; historical data still queryable
    "j3":   "Jason-3",
    "s3a":  "Sentinel-3A",
    "s3b":  "Sentinel-3B",
    "s6a":  "Sentinel-6A",
    "swon": "SWOT nadir",
}

# 5 Hz is only produced for 6 of the 1 Hz missions.
SATELLITES_5HZ = {
    code: name for code, name in SATELLITES_1HZ.items()
    if code in {"al", "c2", "j3", "s3a", "s3b", "s6a"}
}

VARIABLES = {
    "1hz": ["VAVH", "VAVH_UNFILTERED", "WIND_SPEED"],
    "5hz": ["VAVH", "VAVH_UNFILTERED", "VAVH_UNCERTAINTY"],
}

AVAILABILITY_START = {
    "1hz": "2024-01-01T00:00:00",
    "5hz": "2026-03-09T00:00:00",
}

VALID_FREQUENCIES = ("1hz", "5hz")


def _satellite_map(frequency: str) -> dict:
    return SATELLITES_1HZ if frequency == "1hz" else SATELLITES_5HZ


class AltimeterDownloader:
    """
    Download along-track altimeter data from Copernicus Marine.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded NetCDF files.
    dry_run : bool
        If True, print what would be downloaded without actually downloading.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        force_download: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.force_download = force_download

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        frequencies: Iterable[str] = VALID_FREQUENCIES,
        satellites: Optional[list[str]] = None,
    ) -> list[Path]:
        """
        Download altimeter products for every requested satellite/frequency
        combination that intersects the given time window.

        Parameters
        ----------
        frequencies : iterable of str
            Which frequency bands to fetch: "1hz", "5hz", or both (default).
        satellites : list[str], optional
            Restrict to these satellite codes (see SATELLITES_1HZ for valid
            codes). None (default) means all satellites available for each
            requested frequency.

        Returns
        -------
        list[Path]
            Paths to the downloaded NetCDF files (one per satellite/frequency
            combination that returned data).
        """
        try:
            import copernicusmarine
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for altimeter downloads.\n"
                "Install it with:  pip install copernicusmarine"
            ) from exc

        start_dt = normalize_datetime(start)
        end_dt = normalize_datetime(end)

        frequencies = [f.lower() for f in frequencies]
        invalid = [f for f in frequencies if f not in VALID_FREQUENCIES]
        if invalid:
            raise ValueError(
                f"Invalid frequency/frequencies {invalid}. "
                f"Valid values: {', '.join(VALID_FREQUENCIES)}"
            )

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []

        for freq in frequencies:
            avail_start = AVAILABILITY_START[freq]
            if end_dt < avail_start:
                print(
                    f"  Skipping {freq.upper()} altimeter data: requested window "
                    f"ends {end_dt}, before {freq.upper()} availability starts "
                    f"({avail_start})."
                )
                continue
            eff_start_dt = max(start_dt, avail_start)

            sat_map = _satellite_map(freq)
            if satellites:
                requested = {s.lower() for s in satellites}
                unknown = requested - set(sat_map)
                if unknown:
                    print(
                        f"  WARNING: satellite code(s) {sorted(unknown)} not valid "
                        f"for {freq.upper()}; skipping them."
                    )
                sat_codes = [s for s in sat_map if s in requested]
            else:
                sat_codes = list(sat_map)

            variables = VARIABLES[freq]
            template = DATASET_ID_TEMPLATE[freq]

            for sat_code in sat_codes:
                dataset_id = template.format(sat=sat_code)
                start_d = eff_start_dt.split("T")[0]
                end_d = end_dt.split("T")[0]

                windows = split_antimeridian_bbox(min_lon, max_lon)
                for i, (win_min_lon, win_max_lon) in enumerate(windows):
                    suffix = f"_w{i}" if len(windows) > 1 else ""
                    filename = f"{dataset_id}_{start_d}_{end_d}{suffix}.nc"
                    dest_path = self.output_dir / filename

                    if self.dry_run:
                        print(
                            f"[DRY RUN] Would download {freq.upper()} altimeter data "
                            f"({sat_map[sat_code]}, dataset_id={dataset_id}) to:\n"
                            f"  {dest_path}"
                        )
                        continue

                    print(f"Downloading {freq.upper()} altimeter data ({sat_map[sat_code]}) …")
                    print(f"  Dataset: {dataset_id}")
                    print(f"  Region:  lon [{win_min_lon}, {win_max_lon}] lat [{min_lat}, {max_lat}]")
                    print(f"  Time:    {eff_start_dt} → {end_dt}")

                    try:
                        copernicusmarine.subset(
                            dataset_id=dataset_id,
                            variables=variables,
                            minimum_longitude=win_min_lon,
                            maximum_longitude=win_max_lon,
                            minimum_latitude=min_lat,
                            maximum_latitude=max_lat,
                            start_datetime=eff_start_dt,
                            end_datetime=end_dt,
                            minimum_depth=0,
                            maximum_depth=0,
                            output_directory=self.output_dir,
                            output_filename=filename,
                            **copernicus_marine_download_kwargs(self.force_download),
                        )
                        if dest_path.is_dir():
                            # copernicusmarine cannot always merge the request
                            # into a single file (e.g. multiple platforms in
                            # one dataset_id) — it then writes a directory
                            # named after output_filename containing one .nc
                            # per platform instead.
                            new_files = sorted(dest_path.rglob("*.nc"))
                            downloaded.extend(new_files)
                            for f in new_files:
                                print(f"  Saved to {f}")
                            if not new_files:
                                print("  No data in this region/time window — skipped.")
                        elif dest_path.exists():
                            downloaded.append(dest_path)
                            print(f"  Saved to {dest_path}")
                        else:
                            # copernicusmarine.subset() writes nothing (and
                            # raises no error) when the satellite's ground
                            # track doesn't cross this region/time window —
                            # expected for most satellites on a small bbox.
                            print("  No data in this region/time window — skipped.")
                    except Exception as exc:
                        print(f"  Skipping {dataset_id}: {exc}")

        print(f"Downloaded {len(downloaded)} altimeter file(s).")
        return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download along-track altimeter data from Copernicus Marine.",
    )
    p.add_argument("--params-file", metavar="FILE")
    p.add_argument("--min-lon", type=float)
    p.add_argument("--max-lon", type=float)
    p.add_argument("--min-lat", type=float)
    p.add_argument("--max-lat", type=float)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument(
        "--frequencies",
        default="1hz,5hz",
        help="Comma-separated: 1hz,5hz",
    )
    p.add_argument(
        "--satellites",
        default=None,
        help="Comma-separated satellite codes (default: all). "
             "See SATELLITES_1HZ for valid codes.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        min_lon = params["minimum_longitude"]
        max_lon = params["maximum_longitude"]
        min_lat = params["minimum_latitude"]
        max_lat = params["maximum_latitude"]
        start = params["start_datetime"]
        end = params["end_datetime"]
    else:
        for attr in ("min_lon", "max_lon", "min_lat", "max_lat", "start", "end"):
            if getattr(args, attr) is None:
                print(f"Error: --{attr.replace('_','-')} is required (or use --params-file)")
                sys.exit(1)
        min_lon, max_lon = args.min_lon, args.max_lon
        min_lat, max_lat = args.min_lat, args.max_lat
        start, end = args.start, args.end

    output_dir = Path(args.output_dir) if args.output_dir else (
        build_output_dir(start, end, min_lon, max_lon, min_lat, max_lat) / "altimeter"
    )

    frequencies = [f.strip() for f in args.frequencies.split(",")]
    satellites = (
        [s.strip() for s in args.satellites.split(",")]
        if args.satellites else None
    )

    dl = AltimeterDownloader(
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    dl.download(
        min_lon=min_lon, max_lon=max_lon,
        min_lat=min_lat, max_lat=max_lat,
        start=start, end=end,
        frequencies=frequencies,
        satellites=satellites,
    )


if __name__ == "__main__":
    main()
