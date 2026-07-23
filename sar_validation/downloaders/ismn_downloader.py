"""
Select ISMN (International Soil Moisture Network) in-situ soil-moisture
stations from a manually-downloaded local archive.

ISMN has no download API — data must be obtained via the registration-gated
web portal (https://ismn.earth/en/dataviewer/), which itself
supports filtering by geographic rectangle, date range, sensor depth, and
variable before download. This module is a **local-archive selector**, not
a network downloader: given a path to that manually-downloaded archive (zip
or extracted folder), it filters to stations/sensors inside the recipe's
bbox/depth window and writes one CSV per surviving sensor, in the same
long-format schema the Copernicus Marine in-situ CSVs use, so
``DataTreeConverter.from_insitu_csv`` needs zero changes.

No ``archive_path`` needs to be configured for the common case: if it's
absent (or stale), ``download()`` auto-detects the most-recently-modified
``*.zip`` sitting directly in ``output_dir`` — just drop the downloaded
zip into this run's own ISMN folder and re-run. An explicit
``archive_path`` still takes priority when given, for reusing one archive
across multiple recipes.

Library usage::

    from sar_validation.downloaders.ismn_downloader import ISMNDownloader
    dl = ISMNDownloader(output_dir=Path("data/run1/ismn"))
    dl.download(
        min_lon=-10, max_lon=20, min_lat=40, max_lat=55,
        start="2026-01-01", end="2026-01-02",
        min_depth=0.0, max_depth=0.05,
        archive_path="/path/to/ismn_export.zip",  # optional -- see above
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ISMNDownloader"]


def _auto_detect_archive(output_dir: Path) -> Optional[Path]:
    """
    Return the most-recently-modified ``*.zip`` sitting directly in
    *output_dir*, or ``None`` if it doesn't exist or contains none.

    Lets a user just drop the manually-downloaded ISMN zip into this run's
    own ISMN output folder instead of having to edit the recipe's
    ``download_kwargs`` — the common case of "one archive for one run".
    """
    if not output_dir.exists():
        return None
    zips = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def _print_portal_instructions(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float,
    start: str, end: str, min_depth: float, max_depth: float,
    output_dir: Path,
) -> None:
    """Print copy-pasteable filter values for the ISMN web portal."""
    print(
        "ISMN has no download API — data must be requested manually.\n"
        "1. Go to https://ismn.earth/en/dataviewer/ and log in\n"
        "   (registration is free).\n"
        "2. Use these filter values:\n"
        f"     Bounding box:  lon [{min_lon}, {max_lon}], lat [{min_lat}, {max_lat}]\n"
        f"     Date range:    {start} to {end}\n"
        "     Variable:      soil moisture\n"
        f"     Sensor depth:  {min_depth} to {max_depth} m (positive, below ground)\n"
        "3. Use these download options:\n"
        "     Format:        'Variables stored in separate files (CEOP\n"
        "                    formatted) (zipped)' — see\n"
        "                    https://ismn.earth/en/data/ceop-standard/.\n"
        "                    (ismn_downloader auto-detects this format;\n"
        "                    Header+Values also works if CEOP is\n"
        "                    unavailable for a given network.)\n"
        "     Quality flags: 'Good' only — see\n"
        "                    https://ismn.earth/en/data/ismn-quality-flags/.\n"
        "                    Flagged-bad observations would otherwise\n"
        "                    corrupt the validation statistics.\n"
        "     Gap filling:   leave disabled (do not fill gaps with NaN) —\n"
        "                    this toolbox matches on actual observation\n"
        "                    timestamps, so synthetic NaN-filled slots\n"
        "                    add nothing but archive size.\n"
        "4. Download the resulting zip and drop it directly into:\n"
        f"     {output_dir}\n"
        "   No recipe edits needed.\n"
        "5. Re-run this recipe — the archive is picked up automatically\n"
        "   once it's there.\n"
        "   To reuse one archive across multiple recipes instead, set its\n"
        "   path explicitly via 'ismn_archive_path' in this recipe's\n"
        "   download_kwargs for the 'ismn' validation source."
    )


def _sensor_depth_from(meta: pd.Series, default: float) -> float:
    """
    Extract a sensor's ``depth_from`` from its real per-sensor metadata
    ``pandas.Series`` (see ``ismn.meta.MetaData.to_pd``), preferring the
    "instrument" entry (matching ``ismn.filehandlers.IsmnFile.check_metadata``'s
    own precedence) and falling back to "variable", since ``ismn`` assigns the
    same ``Depth`` to both. Either key may be entirely absent (dropped by
    ``to_pd(dropna=True)``) if the sensor has no depth, e.g. for
    non-depth-resolved variables.
    """
    for key in (("instrument", "depth_from"), ("variable", "depth_from")):
        if key in meta.index:
            return float(meta.loc[key])
    return default


class ISMNDownloader:
    """
    Local-archive selector for ISMN soil-moisture stations.

    Parameters
    ----------
    output_dir : Path
        Directory to write one CSV per surviving sensor.
    dry_run : bool
        If True, print what would be selected without writing any files.
    """

    def __init__(self, output_dir: Path, dry_run: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

    def download(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        start: str,
        end: str,
        min_depth: float = 0.0,
        max_depth: float = 0.05,
        archive_path: Optional[str] = None,
    ) -> list[Path]:
        """
        Filter a local ISMN archive to the recipe's window and write one CSV
        per surviving sensor.

        Returns
        -------
        list[Path]
            Paths to the written CSV files (empty if no archive is
            configured, no stations/sensors survive filtering, or
            ``dry_run`` is True).
        """
        resolved_archive_path = (
            Path(archive_path) if archive_path and Path(archive_path).exists()
            else _auto_detect_archive(self.output_dir)
        )
        if resolved_archive_path is None:
            _print_portal_instructions(
                min_lon, max_lon, min_lat, max_lat, start, end, min_depth, max_depth,
                self.output_dir,
            )
            return []

        from ismn.interface import ISMN_Interface

        try:
            reader = ISMN_Interface(resolved_archive_path, parallel=False)
        except ValueError as exc:
            if "No objects to concatenate" in str(exc):
                # ismn's own file reader needs at least two lines per .stm
                # file to bootstrap metadata (it reads the first, second,
                # and last line) -- a portal request for a single calendar
                # day produces exactly one reading per station file, so
                # every file fails to parse and this is what ismn raises
                # once none of them survive. Confirmed against a real
                # single-day ISMN export.
                raise ValueError(
                    "ismn could not parse any station file in this archive "
                    "-- every file appears to contain only a single "
                    "reading. This usually means the portal request's date "
                    "range covered a single calendar day; ismn's file "
                    "reader needs at least two readings per station file. "
                    "Re-download from https://ismn.earth/en/dataviewer/ "
                    "with a wider date range (e.g. a full week) and "
                    "replace this archive."
                ) from exc
            raise

        # ``ISMN_Interface.metadata`` is already a plain pandas.DataFrame (one
        # row per sensor/filehandler, indexed by the same integer id used by
        # ``get_dataset_ids``/``read``), with ("<name>", "val")-style
        # MultiIndex columns — there is no ``.to_pd()`` method on it. There is
        # no dedicated bbox-filter method in the ismn package (as of the
        # version pinned in pyproject.toml), so bbox filtering is done with
        # plain pandas indexing here; variable/depth filtering is delegated
        # to ``get_dataset_ids``, and the two id sets are intersected.
        meta_df = reader.metadata

        in_bbox = (
            (meta_df[("longitude", "val")] >= min_lon)
            & (meta_df[("longitude", "val")] <= max_lon)
            & (meta_df[("latitude", "val")] >= min_lat)
            & (meta_df[("latitude", "val")] <= max_lat)
        )
        bbox_ids = set(meta_df.index[in_bbox].tolist())

        if self.dry_run:
            print(
                f"[DRY RUN] {len(bbox_ids)} sensor(s) in bbox; "
                "would filter by depth/variable and write CSVs."
            )
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        sensor_ids = reader.get_dataset_ids(
            variable="soil_moisture",
            min_depth=min_depth,
            max_depth=max_depth,
        )
        candidate_ids = [sensor_id for sensor_id in sensor_ids if sensor_id in bbox_ids]

        for sensor_id in candidate_ids:
            ts, meta = reader.read(sensor_id, return_meta=True)
            if ts is None or ts.empty:
                continue
            ts = ts.loc[start:end]
            if ts.empty:
                continue

            # ``meta`` is a pandas.Series with a (variable, key)-style
            # MultiIndex (built by ``MetaData.to_pd()`` in ismn/meta.py), not
            # a flat dict — index it with the real (name, "val") tuples.
            station_name = str(meta.loc[("station", "val")])
            lon = float(meta.loc[("longitude", "val")])
            lat = float(meta.loc[("latitude", "val")])
            depth = _sensor_depth_from(meta, default=min_depth)

            df_out = pd.DataFrame({
                "platform_id":   station_name,
                "platform_type": "ismn",
                "time":          ts.index,
                "lon":           lon,
                "lat":           lat,
                "depth":         depth,
                "variable":      "SOIL_MOISTURE",
                "value":         ts["soil_moisture"].values,
            })
            out_path = self.output_dir / f"ismn_{station_name}_{sensor_id}.csv"
            df_out.to_csv(out_path, index=False)
            written.append(out_path)
            print(f"  Wrote {out_path}")

        if not written:
            logger.warning("ISMNDownloader: no stations/sensors survived bbox+depth+window filtering.")
        return written
