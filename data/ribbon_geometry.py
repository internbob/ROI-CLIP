import os
import re
from typing import Dict, Sequence

import nibabel as nib
import numpy as np

from .surface_utils import get_hemi_resample_index


RIBBON_CACHE_VERSION = 1


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def ribbon_bundle_path(cache_root: str, sample_id: str) -> str:
    return os.path.join(str(cache_root), f"{_safe_sample_id(sample_id)}.npz")


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [np.asarray(points, dtype=np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]


def surface_ras_to_target_voxel(
    surface_points: np.ndarray,
    freesurfer_reference,
    target_affine: np.ndarray,
) -> np.ndarray:
    """Map FreeSurfer surface-RAS coordinates onto the cropped network MRI grid."""
    target_affine = np.asarray(target_affine, dtype=np.float64)
    if target_affine.shape != (4, 4) or not np.isfinite(target_affine).all():
        raise ValueError("target_affine must be a finite 4x4 matrix")
    vox2ras_tkr = np.asarray(freesurfer_reference.header.get_vox2ras_tkr(), dtype=np.float64)
    transform = (
        np.linalg.inv(target_affine)
        @ np.asarray(freesurfer_reference.affine, dtype=np.float64)
        @ np.linalg.inv(vox2ras_tkr)
    )
    return _apply_affine(transform, surface_points).astype(np.float32)


def voxel_to_grid_sample_coordinates(voxel_points: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Convert (i,j,k) voxel coordinates to grid_sample (x,y,z), align_corners=False."""
    shape = np.asarray(tuple(int(value) for value in target_shape), dtype=np.float32)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(f"Expected a positive 3D target shape, got {tuple(shape.tolist())}")
    points = np.asarray(voxel_points, dtype=np.float32)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected voxel_points[...,3], got {points.shape}")
    normalized_ijk = 2.0 * (points + 0.5) / shape - 1.0
    return normalized_ijk[..., [2, 1, 0]].astype(np.float32, copy=False)


def _vertex_roi_index(surface_roi_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(surface_roi_masks)
    if masks.ndim != 2:
        raise ValueError(f"surface_roi_masks must have shape [K,N], got {masks.shape}")
    membership = masks > 0
    index = membership.argmax(axis=0).astype(np.int64)
    index[~membership.any(axis=0)] = -1
    if np.any(membership.sum(axis=0) > 1):
        raise ValueError("A surface vertex belongs to more than one ROI")
    return index


def build_ribbon_bundle(
    fs_subject_dir: str,
    target_shape: Sequence[int],
    target_affine: np.ndarray,
    num_vertices: int,
    surface_roi_masks: np.ndarray,
    depths: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Build real white-to-pial sampling grids using the model's surface resampling."""
    target_shape = tuple(int(value) for value in target_shape)
    depths = np.asarray(tuple(float(value) for value in depths), dtype=np.float32)
    if depths.ndim != 1 or depths.size == 0 or np.any(depths < 0.0) or np.any(depths > 1.0):
        raise ValueError("depths must be a non-empty 1D sequence in [0,1]")

    surf_dir = os.path.join(fs_subject_dir, "surf")
    reference_path = os.path.join(fs_subject_dir, "mri", "orig.mgz")
    if not os.path.exists(reference_path):
        reference_path = os.path.join(fs_subject_dir, "mri", "T1.mgz")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Missing FreeSurfer geometry reference under {fs_subject_dir}/mri")
    reference = nib.load(reference_path)

    per_hemi = int(num_vertices) // 2
    if per_hemi <= 0 or per_hemi * 2 != int(num_vertices):
        raise ValueError(f"num_vertices must be a positive even number, got {num_vertices}")

    white_voxels = []
    pial_voxels = []
    for hemi in ("lh", "rh"):
        white_path = os.path.join(surf_dir, f"{hemi}.white")
        pial_path = os.path.join(surf_dir, f"{hemi}.pial")
        if not os.path.exists(white_path) or not os.path.exists(pial_path):
            raise FileNotFoundError(f"Missing real white/pial surfaces for {hemi}: {surf_dir}")
        white, _ = nib.freesurfer.read_geometry(white_path)
        pial, _ = nib.freesurfer.read_geometry(pial_path)
        if white.shape != pial.shape:
            raise ValueError(f"white/pial vertex mismatch for {hemi}: {white.shape} vs {pial.shape}")
        indices = get_hemi_resample_index(surf_dir, hemi, per_hemi)
        white_voxels.append(surface_ras_to_target_voxel(white[indices], reference, target_affine))
        pial_voxels.append(surface_ras_to_target_voxel(pial[indices], reference, target_affine))

    white_voxel = np.concatenate(white_voxels, axis=0).astype(np.float32)
    pial_voxel = np.concatenate(pial_voxels, axis=0).astype(np.float32)
    ribbon_voxel = (
        (1.0 - depths[None, :, None]) * white_voxel[:, None, :]
        + depths[None, :, None] * pial_voxel[:, None, :]
    ).astype(np.float32)

    shape = np.asarray(target_shape, dtype=np.float32)
    ribbon_valid_mask = np.logical_and(ribbon_voxel >= 0.0, ribbon_voxel <= shape - 1.0).all(axis=-1)
    vertex_roi_index = _vertex_roi_index(surface_roi_masks)
    if vertex_roi_index.shape != (int(num_vertices),):
        raise ValueError(
            f"ROI assignment has shape {vertex_roi_index.shape}, expected ({int(num_vertices)},)"
        )
    vertex_valid_mask = ribbon_valid_mask.any(axis=1) & (vertex_roi_index >= 0)
    ribbon_grid = voxel_to_grid_sample_coordinates(ribbon_voxel, target_shape)

    return {
        "ribbon_grid": ribbon_grid,
        "ribbon_valid_mask": ribbon_valid_mask.astype(np.bool_),
        "vertex_valid_mask": vertex_valid_mask.astype(np.bool_),
        "vertex_roi_index": vertex_roi_index,
        "white_voxel": white_voxel,
        "pial_voxel": pial_voxel,
        "depths": depths,
        "target_mri_shape": np.asarray(target_shape, dtype=np.int32),
        "target_mri_affine": np.asarray(target_affine, dtype=np.float32),
    }


def save_ribbon_bundle(
    cache_root: str,
    sample_id: str,
    bundle: Dict[str, np.ndarray],
    overwrite: bool = False,
) -> str:
    os.makedirs(cache_root, exist_ok=True)
    path = ribbon_bundle_path(cache_root, sample_id)
    if os.path.exists(path) and not overwrite:
        return path
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "wb") as handle:
            np.savez_compressed(
                handle,
                ribbon_cache_version=np.asarray(RIBBON_CACHE_VERSION, dtype=np.int16),
                **bundle,
            )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return path


def load_ribbon_bundle(
    cache_root: str,
    sample_id: str,
    num_vertices: int,
    num_depths: int,
    target_shape: Sequence[int],
    target_affine: np.ndarray,
    expected_depths: Sequence[float] = None,
) -> Dict[str, np.ndarray]:
    path = ribbon_bundle_path(cache_root, sample_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ribbon mode requires a real precomputed white/pial bundle, missing: {path}"
        )
    with np.load(path, allow_pickle=False) as payload:
        version = int(np.asarray(payload["ribbon_cache_version"]).item())
        if version != RIBBON_CACHE_VERSION:
            raise ValueError(f"Unsupported ribbon cache version {version} in {path}")
        result = {key: payload[key] for key in payload.files if key != "ribbon_cache_version"}

    stored_grid = result["ribbon_grid"]
    if stored_grid.ndim != 3 or stored_grid.shape[0] != int(num_vertices) or stored_grid.shape[-1] != 3:
        raise ValueError(f"Invalid ribbon_grid in {path}: {stored_grid.shape}")
    if result["ribbon_valid_mask"].shape != stored_grid.shape[:2]:
        raise ValueError(f"Invalid ribbon_valid_mask in {path}: {result['ribbon_valid_mask'].shape}")
    if result["vertex_valid_mask"].shape != (int(num_vertices),):
        raise ValueError(f"Invalid vertex_valid_mask in {path}: {result['vertex_valid_mask'].shape}")
    if result["vertex_roi_index"].shape != (int(num_vertices),):
        raise ValueError(f"Invalid vertex_roi_index in {path}: {result['vertex_roi_index'].shape}")
    if tuple(result["target_mri_shape"].tolist()) != tuple(int(value) for value in target_shape):
        raise ValueError(f"Ribbon target shape mismatch in {path}")
    if not np.allclose(
        result["target_mri_affine"], np.asarray(target_affine, dtype=np.float32), rtol=0.0, atol=1e-4
    ):
        raise ValueError(f"Ribbon target affine mismatch in {path}")
    if not np.isfinite(result["ribbon_grid"]).all():
        raise ValueError(f"Non-finite ribbon_grid in {path}")

    requested_depths = (
        np.asarray(tuple(expected_depths), dtype=np.float32)
        if expected_depths is not None
        else result["depths"].astype(np.float32, copy=False)
    )
    if requested_depths.shape != (int(num_depths),) or np.any(requested_depths < 0.0) or np.any(requested_depths > 1.0):
        raise ValueError(f"Invalid requested ribbon depths in {path}: {requested_depths}")
    if np.allclose(result["depths"], requested_depths, rtol=0.0, atol=1e-6):
        ribbon_grid = result["ribbon_grid"].astype(np.float32, copy=False)
        ribbon_valid_mask = result["ribbon_valid_mask"].astype(np.bool_, copy=False)
    else:
        # Geometry caches store the real white and pial coordinates, so new
        # intracortical depth schedules can be evaluated without duplicating
        # the subject cache or approximating in feature space.
        if "white_voxel" not in result or "pial_voxel" not in result:
            raise ValueError(f"Ribbon cache cannot resample depths because white/pial coordinates are missing: {path}")
        white = result["white_voxel"].astype(np.float32, copy=False)
        pial = result["pial_voxel"].astype(np.float32, copy=False)
        ribbon_voxel = (
            (1.0 - requested_depths[None, :, None]) * white[:, None, :]
            + requested_depths[None, :, None] * pial[:, None, :]
        ).astype(np.float32)
        shape = np.asarray(tuple(int(value) for value in target_shape), dtype=np.float32)
        ribbon_valid_mask = np.logical_and(ribbon_voxel >= 0.0, ribbon_voxel <= shape - 1.0).all(axis=-1)
        ribbon_grid = voxel_to_grid_sample_coordinates(ribbon_voxel, target_shape)

    vertex_roi_index = result["vertex_roi_index"].astype(np.int64, copy=False)
    vertex_valid_mask = ribbon_valid_mask.any(axis=1) & (vertex_roi_index >= 0)

    return {
        "ribbon_grid": ribbon_grid,
        "ribbon_valid_mask": ribbon_valid_mask.astype(np.bool_, copy=False),
        "vertex_valid_mask": vertex_valid_mask.astype(np.bool_, copy=False),
        "vertex_roi_index": vertex_roi_index,
    }
