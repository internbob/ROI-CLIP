import os
import re
from typing import Dict, Optional, Sequence

import numpy as np


CACHE_VERSION = 3


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def preprocess_cache_path(cache_root: str, sample_id: str) -> str:
    return os.path.join(str(cache_root), f"{_safe_sample_id(sample_id)}.npz")


def load_preprocess_cache(
    cache_root: str,
    sample_id: str,
    num_vertices: int,
    num_rois: int,
    roi_shape: Sequence[int],
    target_mri_shape: Optional[Sequence[int]] = None,
    target_mri_affine: Optional[np.ndarray] = None,
    allow_unaligned_control: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    if not cache_root:
        return None
    path = preprocess_cache_path(cache_root, sample_id)
    if not os.path.exists(path):
        return None

    target_shape = tuple(int(v) for v in roi_shape)
    with np.load(path, allow_pickle=False) as payload:
        version = int(np.asarray(payload["cache_version"]).item())
        if version != CACHE_VERSION:
            return None
        surface_data = payload["surface_data"].astype(np.float32, copy=False)
        surface_roi_masks = payload["surface_roi_masks"].astype(np.float32, copy=False)
        local_label_map = payload["local_label_map"].astype(np.int16, copy=False)
        roi_label_map = payload["roi_label_map"].astype(np.int16, copy=False)
        cached_mri_shape = payload["target_mri_shape"].astype(np.int32, copy=False)
        cached_mri_affine = payload["target_mri_affine"].astype(np.float32, copy=False)
        atlas_resampled_to_mri = bool(np.asarray(payload["atlas_resampled_to_mri"]).item())

    if surface_data.shape != (int(num_vertices), 3):
        return None
    if surface_roi_masks.shape != (int(num_rois), int(num_vertices)):
        return None
    if local_label_map.shape != target_shape or roi_label_map.shape != target_shape:
        return None
    if cached_mri_shape.shape != (3,) or cached_mri_affine.shape != (4, 4):
        return None
    if not atlas_resampled_to_mri and not allow_unaligned_control:
        return None
    if target_mri_shape is not None and tuple(cached_mri_shape.tolist()) != tuple(int(v) for v in target_mri_shape):
        return None
    if target_mri_affine is not None and not np.allclose(
        cached_mri_affine,
        np.asarray(target_mri_affine, dtype=np.float32),
        rtol=0.0,
        atol=1e-4,
    ):
        return None

    return {
        "surface_data": surface_data,
        "surface_roi_masks": surface_roi_masks,
        "local_label_map": local_label_map,
        "roi_label_map": roi_label_map,
        "global_brain_mask": (roi_label_map >= 0).astype(np.float32),
        "target_mri_shape": cached_mri_shape,
        "target_mri_affine": cached_mri_affine,
        "atlas_resampled_to_mri": np.asarray(atlas_resampled_to_mri, dtype=np.bool_),
    }


def save_preprocess_cache(
    cache_root: str,
    sample_id: str,
    surface_data: np.ndarray,
    surface_roi_masks: np.ndarray,
    local_label_map: np.ndarray,
    roi_label_map: np.ndarray,
    roi_shape: Sequence[int],
    target_mri_shape: Optional[Sequence[int]] = None,
    target_mri_affine: Optional[np.ndarray] = None,
    atlas_resampled_to_mri: bool = True,
    overwrite: bool = False,
) -> str:
    os.makedirs(cache_root, exist_ok=True)
    path = preprocess_cache_path(cache_root, sample_id)
    if os.path.exists(path) and not overwrite:
        return path

    temp_path = f"{path}.tmp.{os.getpid()}"
    target_shape = tuple(int(v) for v in roi_shape)
    if target_mri_shape is None or target_mri_affine is None:
        raise ValueError("target_mri_shape and target_mri_affine are required for cache provenance")
    mri_shape = np.asarray(tuple(int(v) for v in target_mri_shape), dtype=np.int32)
    mri_affine = np.asarray(target_mri_affine, dtype=np.float32)
    if mri_shape.shape != (3,) or mri_affine.shape != (4, 4) or not np.isfinite(mri_affine).all():
        raise ValueError("Expected target_mri_shape=(D,H,W) and a finite 4x4 target_mri_affine")
    local_label_map = resize_label_map_nearest(local_label_map, target_shape)
    roi_label_map = resize_label_map_nearest(roi_label_map, target_shape)
    try:
        with open(temp_path, "wb") as f:
            np.savez(
                f,
                cache_version=np.asarray(CACHE_VERSION, dtype=np.int16),
                surface_data=np.asarray(surface_data, dtype=np.float32),
                surface_roi_masks=np.asarray(surface_roi_masks, dtype=np.uint8),
                local_label_map=np.asarray(local_label_map, dtype=np.int16),
                roi_label_map=np.asarray(roi_label_map, dtype=np.int16),
                target_mri_shape=mri_shape,
                target_mri_affine=mri_affine,
                atlas_resampled_to_mri=np.asarray(bool(atlas_resampled_to_mri), dtype=np.uint8),
            )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return path


def resize_label_map_nearest(label_map: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize an integer atlas map with the same nearest-neighbor rule used in training."""
    label_map = np.asarray(label_map, dtype=np.int16)
    target_shape = tuple(int(v) for v in target_shape)
    if label_map.shape == target_shape:
        return label_map
    if label_map.ndim != 3 or len(target_shape) != 3:
        raise ValueError(f"Expected 3D label map and target shape, got {label_map.shape} -> {target_shape}")

    indices = [
        np.floor(np.arange(dst, dtype=np.float64) * (src / float(dst))).astype(np.int64)
        for src, dst in zip(label_map.shape, target_shape)
    ]
    return label_map[np.ix_(*indices)].astype(np.int16, copy=False)
