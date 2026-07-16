"""
Shared Copernicus Marine HF-radar-total region table.

Bounding boxes and the "has_latest" flag were read from
``copernicusmarine.describe(dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr")``
on 2026-07-15: each ``<latest|monthly>-radar-total--<Region>`` dataset_part's
declared variable bbox is the region's real grid extent. ``monthly-radar-total``
covers all 25 regions; ``latest-radar-total`` only exists for the 18 with a
near-real-time feed.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["HFR_REGIONS", "resolve_hfr_region"]

_NO_LATEST = {
    "ARPAS", "COSYNA", "Finnmark", "US-Alaska",
    "US-EastGulfCoast", "US-Hawaii", "WHub",
}

_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "ARPAS": (8.164931297302246, 8.878068923950195, 40.756874084472656, 41.243125915527344),
    "CALYPSO": (13.69489860534668, 15.365100860595703, 35.742252349853516, 36.87774658203125),
    "COSYNA": (5.916669845581055, 8.970867156982422, 53.41814422607422, 55.20000076293945),
    "DeltaEbro": (0.06353014707565308, 2.078089952468872, 39.59859848022461, 41.219600677490234),
    "EUSKOOS": (-3.1965925693511963, -1.2034072875976562, 43.31999969482422, 44.58000183105469),
    "Finnmark": (18.042926788330078, 35.45707321166992, 69.00939178466797, 73.49060821533203),
    "Galicia": (-11.327714920043945, -7.9691948890686035, 40.35469055175781, 44.67578887939453),
    "Gibraltar": (-5.8481950759887695, -4.994204998016357, 35.805084228515625, 36.1926155090332),
    "GoS": (13.072955131530762, 15.999811172485352, 39.672447204589844, 41.38346862792969),
    "Granitola": (12.176433563232422, 13.496066093444824, 37.00240707397461, 37.59709167480469),
    "ICATMAR": (1.0088672637939453, 4.291132926940918, 40.50752258300781, 42.99247741699219),
    "Ibiza": (0.5038551688194275, 1.4006848335266113, 38.3229866027832, 39.1067008972168),
    "Lisboa": (-10.595585823059082, -8.704414367675781, 37.90182113647461, 38.8981819152832),
    "NAdr": (13.375, 13.780560493469238, 45.526851654052734, 45.783329010009766),
    "PLOCAN": (-15.420296669006348, -14.89370346069336, 27.85053062438965, 28.34457015991211),
    "Skagerrak": (7.502049922943115, 11.997950553894043, 57.011016845703125, 59.488983154296875),
    "South": (-9.593195915222168, -6.206803798675537, 36.005245208740234, 37.19475555419922),
    "TirLig": (7.50698709487915, 10.493012428283691, 43.25399398803711, 44.49600601196289),
    "US-Alaska": (-174.09815979003906, -128.659912109375, 68.0091781616211, 74.0321044921875),
    "US-EastGulfCoast": (-97.88055419921875, -57.2345085144043, 21.75611114501953, 46.47426986694336),
    "US-Hawaii": (-163.11717224121094, -152.01162719726562, 16.224061965942383, 24.895137786865234),
    "US-PuertoRicoVirginIslands": (-70.49944305419922, -61.024757385253906, 14.508465766906738, 21.989192962646484),
    "US-WestCoast": (-130.33265686035156, -115.83291625976562, 30.25959587097168, 49.982444763183594),
    "Vestlandet": (1.0030561685562134, 8.496943473815918, 56.50277328491211, 63.99722671508789),
    "WHub": (-6.13332986831665, -5.0837883949279785, 50.21573257446289, 51.01667022705078),
}

HFR_REGIONS: Dict[str, Dict[str, object]] = {
    name: {"bbox": bbox, "has_latest": name not in _NO_LATEST}
    for name, bbox in _BBOXES.items()
}


def resolve_hfr_region(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> str:
    """Return the HFR_REGIONS name whose bbox overlaps the request the most.

    Raises ``ValueError`` if no known region overlaps the request bbox at all.
    """
    best_name = None
    best_area = 0.0
    for name, cfg in HFR_REGIONS.items():
        r_min_lon, r_max_lon, r_min_lat, r_max_lat = cfg["bbox"]
        overlap_lon = min(max_lon, r_max_lon) - max(min_lon, r_min_lon)
        overlap_lat = min(max_lat, r_max_lat) - max(min_lat, r_min_lat)
        if overlap_lon > 0 and overlap_lat > 0:
            area = overlap_lon * overlap_lat
            if area > best_area:
                best_area = area
                best_name = name
    if best_name is None:
        raise ValueError(
            f"No Copernicus HF-radar region overlaps bbox lon[{min_lon},{max_lon}] "
            f"lat[{min_lat},{max_lat}]. Known regions: {sorted(HFR_REGIONS)}"
        )
    return best_name
