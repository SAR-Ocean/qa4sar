"""
Download radiosonde wind profile data.

TODO: This downloader is not yet implemented.

Candidate data sources:
- REMSS (Remote Sensing Systems) FTP:
    ftp://ftp.remss.com/
    Provides wind data from radiometers (WindSat, AMSR2, etc.).
- IGRA2 (Integrated Global Radiosonde Archive) via NOAA:
    https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive
    Access via HTTPS download or THREDDS OPeNDAP.
- EUMETSAT (Metop AMSU-A, MHS):
    Via EUMDAC (same credentials as scatterometer).
- HY-2B / HY-2C scatterometer:
    See scatterometer_downloader.py TODO section.

When implemented, this downloader will be added to the recipe system
by including 'radiosonde' in the validation_sources list of a recipe.
"""

from __future__ import annotations


class RadiosondeDownloader:
    """Placeholder — radiosonde download not yet implemented."""

    def __init__(self, output_dir, dry_run: bool = False, **kwargs):
        self.output_dir = output_dir
        self.dry_run = dry_run

    def download(self, *args, **kwargs):
        raise NotImplementedError(
            "Radiosonde download is not yet implemented.\n"
            "See the module docstring for planned data sources."
        )
