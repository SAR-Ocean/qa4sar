"""
SAR patch extractor — step 4a of the validation pipeline.

For each collocated pair produced by step 3, extracts a configurable
N × N pixel neighbourhood from the original SAR swath and stores the
patches in ``collocation_patches.nc``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "extract_patches",
    "add_patches_to_dataset",
    "run_patch_extraction",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_patch_size(patch_size: int) -> int:
    """Return a valid (positive, odd) patch size, warning if adjusted."""
    if patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size}.")
    if patch_size % 2 == 0:
        adjusted = patch_size + 1
        warnings.warn(
            f"patch_size={patch_size} is even; rounding up to {adjusted} (must be odd).",
            UserWarning,
            stacklevel=3,
        )
        return adjusted
    return patch_size


def _extract_one(
    array: np.ndarray,   # shape (y, x)
    y_idx: int,
    x_idx: int,
    radius: int,
) -> np.ndarray:
    """
    Extract a (2*radius+1) × (2*radius+1) patch centred on (y_idx, x_idx).

    Pixels that fall outside the array bounds are filled with NaN.
    """
    ny, nx = array.shape
    side = 2 * radius + 1
    patch = np.full((side, side), np.nan, dtype=float)

    y0 = y_idx - radius
    x0 = x_idx - radius

    # Source slice in the full array
    src_y0 = max(0, y0)
    src_y1 = min(ny, y0 + side)
    src_x0 = max(0, x0)
    src_x1 = min(nx, x0 + side)

    # Corresponding slice in the output patch
    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    patch[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1].astype(float)
    return patch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_patches(
    collocation_ds: xr.Dataset,
    datatree: "xr.DataTree",
    patch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Extract SAR patches for every collocated pair.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Dataset produced by step 3 (``collocation_results.nc``).  Must
        contain ``sar_y_idx``, ``sar_x_idx``, and ``sar_scene_name``.
    datatree : xr.DataTree
        DataTree produced by step 2 (``datatree.nc``).  SAR scenes are
        expected at ``/sar/<scene_name>``.
    patch_size : int
        Side length of the patch in pixels (must be a positive odd integer;
        even values are silently rounded up).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``"sar_patch_<var>"`` → array of shape
        ``(n_collocations, patch_size, patch_size)``.
    """
    patch_size = _validate_patch_size(patch_size)
    radius = patch_size // 2

    n = collocation_ds.sizes["collocation"]
    scene_names  = collocation_ds["sar_scene_name"].values
    y_indices    = collocation_ds["sar_y_idx"].values.astype(int)
    x_indices    = collocation_ds["sar_x_idx"].values.astype(int)

    # Pre-load all needed SAR scenes into memory (one Dataset per scene)
    sar_node = datatree.get("sar")
    if sar_node is None:
        raise ValueError("DataTree has no '/sar' group — cannot extract patches.")

    scene_cache: Dict[str, xr.Dataset] = {}
    for scene_name in np.unique(scene_names):
        if scene_name not in sar_node.children:
            logger.warning("SAR scene '%s' not found in DataTree — patches will be NaN.", scene_name)
            continue
        scene_cache[scene_name] = sar_node[scene_name].to_dataset()

    # Identify SAR variables shared across all scenes
    sar_vars: List[str] = []
    for ds in scene_cache.values():
        sar_vars = [v for v in ds.data_vars if ds[v].dims == ("y", "x")]
        break  # all scenes have the same variables

    if not sar_vars:
        raise ValueError("No (y, x) SAR variables found — cannot extract patches.")

    patches: Dict[str, np.ndarray] = {
        f"sar_patch_{v}": np.full((n, patch_size, patch_size), np.nan)
        for v in sar_vars
    }

    for i in range(n):
        scene_name = str(scene_names[i])
        y_idx = int(y_indices[i])
        x_idx = int(x_indices[i])

        if scene_name not in scene_cache:
            continue  # scene missing — patch stays NaN

        ds = scene_cache[scene_name]
        for v in sar_vars:
            arr = ds[v].values  # (y, x)
            patches[f"sar_patch_{v}"][i] = _extract_one(arr, y_idx, x_idx, radius)

    return patches


def add_patches_to_dataset(
    collocation_ds: xr.Dataset,
    patches: Dict[str, np.ndarray],
    patch_size: int,
) -> xr.Dataset:
    """
    Attach patch arrays to *collocation_ds* as new data variables.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Base collocation dataset.
    patches : dict
        Output of :func:`extract_patches` — mapping ``"sar_patch_<var>"``
        → array ``(n_collocations, patch_size, patch_size)``.
    patch_size : int
        Validated patch side length (must be odd).

    Returns
    -------
    xr.Dataset
        Augmented dataset with new variables ``sar_patch_<var>`` on dims
        ``(collocation, patch_y, patch_x)`` and relative pixel-offset
        coordinates ``patch_y`` and ``patch_x`` (range ``−r … +r``).
    """
    radius = patch_size // 2
    offsets = np.arange(-radius, radius + 1)

    new_vars = {}
    for key, arr in patches.items():
        new_vars[key] = xr.DataArray(
            arr,
            dims=["collocation", "patch_y", "patch_x"],
            coords={
                "patch_y": ("patch_y", offsets, {"long_name": "row offset from matched pixel"}),
                "patch_x": ("patch_x", offsets, {"long_name": "column offset from matched pixel"}),
            },
            attrs={"patch_size": patch_size},
        )

    return collocation_ds.assign(new_vars)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_patch_extraction(
    collocation_ds: xr.Dataset,
    datatree: "xr.DataTree",
    patch_size: int,
    base_dir: Union[str, Path],
) -> Optional[xr.Dataset]:
    """
    Extract patches and save the augmented dataset to
    ``<base_dir>/collocation_patches.nc``.

    Parameters
    ----------
    collocation_ds : xr.Dataset
        Step-3 collocations (``collocation_results.nc``).
    datatree : xr.DataTree
        Step-2 DataTree (``datatree.nc``).
    patch_size : int
        Patch side length in pixels.
    base_dir : str or Path
        Output directory.

    Returns
    -------
    xr.Dataset or None
        Augmented dataset, or None on failure.
    """
    base_dir = Path(base_dir)
    patch_size = _validate_patch_size(patch_size)

    logger.info("Extracting %d×%d pixel SAR patches …", patch_size, patch_size)

    try:
        patches = extract_patches(collocation_ds, datatree, patch_size)
    except (ValueError, KeyError) as exc:
        logger.error("Patch extraction failed: %s", exc)
        return None

    augmented_ds = add_patches_to_dataset(collocation_ds, patches, patch_size)

    out_path = base_dir / "collocation_patches.nc"
    augmented_ds.to_netcdf(out_path)
    logger.info("Patch dataset saved to %s", out_path)
    return augmented_ds
