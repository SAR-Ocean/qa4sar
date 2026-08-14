"""
SAR source registry.

The single place mapping a recipe's ``sar_data.source`` key to its
downloader, output subdirectory, converter, and per-source recipe-template
defaults. See
docs/superpowers/specs/2026-07-30-nisar-soil-moisture-and-sar-source-selection-design.md

Available SAR sources (satellite family -> ``--sar-source`` name -> which
recipe variables it supports), kept here for readability -- see
``AVAILABLE_SATELLITES``/``SAR_SOURCES`` below for the code-derived,
always-in-sync version of this same list:

  - sentinel1: wind, waves, currents (Sentinel-1 L2 OCN) and
    soil_moisture (Sentinel-1 CLMS SSM, 1km, Europe)
  - nisar:     soil_moisture (NISAR SME2, beta)
  - radarsat2: wind (NOAA NCEI SAR-derived ocean surface wind speed)

A future satellite is added by (1) registering a new ``SARSourceSpec`` in
``SAR_SOURCES`` below with its own ``satellite`` name, and (2) updating the
list above -- ``AVAILABLE_SATELLITES`` and ``resolve_sar_source`` need no
changes, since both derive from the registry automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Tuple

if TYPE_CHECKING:
    import xarray as xr

    from .recipe import SARDataSpec

__all__ = ["SARSourceSpec", "SAR_SOURCES", "AVAILABLE_SATELLITES", "resolve_sar_source"]


@dataclass(frozen=True)
class SARSourceSpec:
    """One SAR-side product a recipe can select via ``sar_data.source``."""

    key: str
    #: Satellite family this product belongs to (e.g. "sentinel1",
    #: "nisar") -- the name --sar-source accepts on the CLI. Distinct from
    #: ``key`` because one satellite can have more than one product
    #: (Sentinel-1 has L2 OCN for wind/waves/currents and CLMS SSM for
    #: soil_moisture); ``resolve_sar_source`` picks the right ``key`` for
    #: a given (satellite, variable) pair.
    satellite: str
    #: recipe "variable" categories that accept this source
    variables: FrozenSet[str]
    #: base_dir / this
    output_subdir: str
    #: glob pattern (relative to output_subdir) matching each downloaded
    #: product, e.g. "*.SAFE", "*.tif*", "*.h5"
    file_glob: str
    #: (output_dir, dry_run, force_download) -> downloader instance with a
    #: .download(min_lon, max_lon, min_lat, max_lat, start, end, **kwargs)
    #: method. Does its own lazy import.
    build_downloader: Callable[[Path, bool, bool], Any]
    #: Maps the recipe's sar_data fields to whatever extra kwargs THIS
    #: source's .download() call accepts -- existing downloaders'
    #: .download() signatures aren't uniform (SARDownloader takes
    #: modes=/limit=; others don't and would raise TypeError if given them).
    extra_download_kwargs: Callable[["SARDataSpec"], Dict[str, Any]]
    #: (path, product_type) -> Dataset or None. product_type (OWI/OSW/RVL)
    #: only matters for sentinel1_l2_ocn; every other source's convert
    #: callback ignores it. Does its own lazy import.
    convert: Callable[[Path, str], Optional["xr.Dataset"]]
    default_min_depth: Optional[float] = None
    default_max_depth: Optional[float] = None
    default_time_tolerance_minutes: Optional[int] = None
    default_aggregation_window_km: Optional[float] = None
    default_spatial_tolerance_km: Optional[float] = None
    #: Whether plot_collocation_diagnostics should render one PNG per SAR
    #: scene instead of one combined overview map, for a soil_moisture
    #: recipe with multiple scenes. True for sentinel1_clms_ssm, whose
    #: scenes are daily, mutually-overlapping, continent-wide mosaics --
    #: overlaying more than one on a single map makes individual days
    #: visually indistinguishable. False (the default) suits sources like
    #: nisar_sme2, whose small, non-overlapping per-orbit granules can
    #: coexist on one map without that problem.
    diagnostics_split_by_scene: bool = False
    #: Whether plot_geographic's per-scene panels should clamp their
    #: extent to the recipe's full requested bbox instead of the scene's
    #: own actual data. True for sentinel1_clms_ssm, whose raw grid
    #: literally covers all of mainland Europe (mostly NaN outside that
    #: day's real swath) regardless of what was requested -- without
    #: clamping, autoscaling to "valid data" would show the whole
    #: continent instead of the requested region. False (the default)
    #: suits sources like nisar_sme2, whose own native grid is already
    #: tight around real data -- clamping to a much larger recipe bbox
    #: there just shrinks the actual scene into a small corner of an
    #: otherwise-empty panel.
    geographic_plot_clamp_to_bounds: bool = False


# ---------------------------------------------------------------------------
# sentinel1_l2_ocn -- wind / waves / currents
# ---------------------------------------------------------------------------

def _build_sentinel1_l2_ocn_downloader(output_dir: Path, dry_run: bool, force_download: bool) -> Any:
    from ..downloaders.sentinel1_l2_ocn_downloader import SARDownloader
    return SARDownloader(output_dir=output_dir, dry_run=dry_run, force_download=force_download)


def _sentinel1_l2_ocn_kwargs(sd: "SARDataSpec") -> Dict[str, Any]:
    return {"modes": sd.swath_mode or None, "limit": sd.max_downloads, **sd.download_kwargs}


def _convert_sentinel1_l2_ocn(path: Path, product_type: str) -> Optional["xr.Dataset"]:
    from .datatree_converter import DataTreeConverter
    return DataTreeConverter.from_sar_l2_ocn_safe(path, product_type=product_type)


# ---------------------------------------------------------------------------
# sentinel1_clms_ssm -- soil_moisture
# ---------------------------------------------------------------------------

def _build_sentinel1_clms_ssm_downloader(output_dir: Path, dry_run: bool, force_download: bool) -> Any:
    from ..downloaders.sentinel1_soil_moisture_downloader import SoilMoistureDownloader
    return SoilMoistureDownloader(output_dir=output_dir, dry_run=dry_run, force_download=force_download)


def _sentinel1_clms_ssm_kwargs(sd: "SARDataSpec") -> Dict[str, Any]:
    return dict(sd.download_kwargs)


def _convert_sentinel1_clms_ssm(path: Path, product_type: str) -> Optional["xr.Dataset"]:
    from .datatree_converter import DataTreeConverter
    return DataTreeConverter.from_sar_l3_ssm_geotiff(path)


# ---------------------------------------------------------------------------
# nisar_sme2 -- soil_moisture (beta)
# ---------------------------------------------------------------------------

#: NISAR SME2 (beta) CMR short_name/version -- confirmed 2026-07-31 against
#: NASA's live CMR catalog (collection concept_id C2850265000-ASF, "NISAR
#: Beta Soil Moisture (Version 1)"; granule ids start with
#: "NISAR_L3_PR_SME2_", matching the .h5 file_glob below). A separate,
#: more-mature "NISAR_L3_SME2_PROVISIONAL_V1" collection also exists
#: (C2854344945-ASF) with more granules over a later time range -- kept on
#: BETA here since that's the maturity level this source's docs/attrs
#: ("NISAR SME2 (beta)") were written for; switching to PROVISIONAL is a
#: deliberate product-choice decision, not a typo fix, so it's left for a
#: separate change if wanted.
NISAR_SME2_SHORT_NAME = "NISAR_L3_SME2_BETA_V1"
NISAR_SME2_VERSION: Optional[str] = "1"

#: NISAR SME2's underlying CMR collection changed mid-mission with no
#: temporal overlap between the two -- confirmed 2026-07-31 directly
#: against NASA's live CMR catalog (cross-checked against a real
#: user-reported gap, itself independently cross-checked against ASF
#: Vertex): NISAR_L3_SME2_BETA_V1's real granules run 2025-10-01 through
#: 2026-01-20, then NOTHING (not even in Vertex) until
#: NISAR_L3_SME2_PROVISIONAL_V1's real granules pick up on 2026-06-17.
#: Both candidates are queried and merged (see EarthdataSoilMoistureDownloader's
#: multi-candidate support) rather than picking one via a hardcoded date
#: cutoff, since CMR itself is the source of truth for which collection
#: actually has data in a given window -- a hardcoded cutoff would just be
#: another guess of exactly the kind that turned out wrong for AU_Land_NRT_R02.
NISAR_SME2_CANDIDATES: List[Tuple[str, Optional[str]]] = [
    (NISAR_SME2_SHORT_NAME, NISAR_SME2_VERSION),
    ("NISAR_L3_SME2_PROVISIONAL_V1", "1"),
]


def _build_nisar_sme2_downloader(output_dir: Path, dry_run: bool, force_download: bool) -> Any:
    # force_download intentionally unused -- EarthdataSoilMoistureDownloader
    # has no such parameter yet (see design doc §6/§12); earthaccess.download()
    # already skips files it finds present under the same name.
    from ..downloaders.earthdata_soil_moisture_downloader import EarthdataSoilMoistureDownloader
    return EarthdataSoilMoistureDownloader(
        dataset=NISAR_SME2_CANDIDATES,
        output_dir=output_dir, dry_run=dry_run,
    )


def _nisar_sme2_kwargs(sd: "SARDataSpec") -> Dict[str, Any]:
    return dict(sd.download_kwargs)


def _convert_nisar_sme2(path: Path, product_type: str) -> Optional["xr.Dataset"]:
    # Lazy import keeps this module decoupled from datatree_converter.py at
    # load time (it's only needed when a NISAR SME2 conversion is actually
    # requested).
    from .datatree_converter import DataTreeConverter
    return DataTreeConverter.from_nisar_sme2(path)


# ---------------------------------------------------------------------------
# radarsat2 -- wind (speed only; see design-choices.md Sec 10)
# ---------------------------------------------------------------------------

def _build_radarsat2_downloader(output_dir: Path, dry_run: bool, force_download: bool) -> Any:
    from ..downloaders.radarsat2_wind_downloader import RADARSAT2WindDownloader
    return RADARSAT2WindDownloader(output_dir=output_dir, dry_run=dry_run, force_download=force_download)


def _radarsat2_kwargs(sd: "SARDataSpec") -> Dict[str, Any]:
    return dict(sd.download_kwargs)


def _convert_radarsat2_wind(path: Path, product_type: str) -> Optional["xr.Dataset"]:
    from .datatree_converter import DataTreeConverter
    return DataTreeConverter.from_radarsat2_wind(path, product_type)


SAR_SOURCES: Dict[str, SARSourceSpec] = {
    "sentinel1_l2_ocn": SARSourceSpec(
        key="sentinel1_l2_ocn",
        satellite="sentinel1",
        variables=frozenset({"wind", "waves", "currents"}),
        output_subdir="S1_L2_OCN",
        file_glob="*.SAFE",
        build_downloader=_build_sentinel1_l2_ocn_downloader,
        extra_download_kwargs=_sentinel1_l2_ocn_kwargs,
        convert=_convert_sentinel1_l2_ocn,
    ),
    "sentinel1_clms_ssm": SARSourceSpec(
        key="sentinel1_clms_ssm",
        satellite="sentinel1",
        variables=frozenset({"soil_moisture"}),
        output_subdir="S1_L3_SSM",
        file_glob="*.tif*",
        build_downloader=_build_sentinel1_clms_ssm_downloader,
        extra_download_kwargs=_sentinel1_clms_ssm_kwargs,
        convert=_convert_sentinel1_clms_ssm,
        default_min_depth=0.0,
        default_max_depth=0.05,
        default_time_tolerance_minutes=720,
        default_aggregation_window_km=1.0,
        default_spatial_tolerance_km=2.0,
        diagnostics_split_by_scene=True,
        geographic_plot_clamp_to_bounds=True,
    ),
    "nisar_sme2": SARSourceSpec(
        key="nisar_sme2",
        satellite="nisar",
        variables=frozenset({"soil_moisture"}),
        output_subdir="NISAR_L3_SME2",
        file_glob="*.h5",
        build_downloader=_build_nisar_sme2_downloader,
        extra_download_kwargs=_nisar_sme2_kwargs,
        convert=_convert_nisar_sme2,
        default_min_depth=0.0,
        default_max_depth=0.05,
        default_time_tolerance_minutes=360,
        default_aggregation_window_km=0.2,
        default_spatial_tolerance_km=2.0,
    ),
    "radarsat2": SARSourceSpec(
        key="radarsat2",
        satellite="radarsat2",
        variables=frozenset({"wind"}),
        output_subdir="RADARSAT2_WIND",
        file_glob="*.nc",
        build_downloader=_build_radarsat2_downloader,
        extra_download_kwargs=_radarsat2_kwargs,
        convert=_convert_radarsat2_wind,
    ),
}

#: Satellite family names accepted by --sar-source, derived from the
#: registry so this can never drift out of sync as new sources are added.
AVAILABLE_SATELLITES: List[str] = sorted({spec.satellite for spec in SAR_SOURCES.values()})


def resolve_sar_source(name: str, variable: str) -> str:
    """
    Resolve a ``--sar-source`` CLI value to an internal ``SAR_SOURCES`` key.

    Accepts either a satellite family name (e.g. ``"sentinel1"``,
    ``"nisar"`` -- see ``AVAILABLE_SATELLITES``) or an exact internal
    registry key (e.g. ``"sentinel1_clms_ssm"``), kept working for
    backward compatibility with recipe files that already have a specific
    key stored in ``sar_data.source``. A satellite name resolves to
    whichever registered product actually supports *variable* -- e.g.
    ``"sentinel1"`` resolves to the L2 OCN product for wind/waves/currents
    and to the CLMS SSM product for soil_moisture.

    Raises
    ------
    ValueError
        If *name* is not a known satellite or internal key, or if the
        matched source (satellite or exact key) has no product for
        *variable*.
    """
    if name in SAR_SOURCES:
        spec = SAR_SOURCES[name]
        if variable not in spec.variables:
            raise ValueError(
                f"SAR source {name!r} is only valid for: "
                f"{', '.join(sorted(spec.variables))} (got variable={variable!r})"
            )
        return name

    matches = [
        key for key, spec in SAR_SOURCES.items()
        if spec.satellite == name and variable in spec.variables
    ]
    if matches:
        assert len(matches) == 1, f"ambiguous SAR source {name!r} for variable {variable!r}: {matches}"
        return matches[0]

    if name in AVAILABLE_SATELLITES:
        raise ValueError(f"SAR source {name!r} has no product for variable {variable!r}.")
    raise ValueError(
        f"Unknown SAR source {name!r}. Available satellites: "
        f"{', '.join(AVAILABLE_SATELLITES)}."
    )
