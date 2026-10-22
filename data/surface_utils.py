import os
from functools import lru_cache
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

from .ico_utils import build_resample_index, get_ico_topology, infer_ico_level


REQUIRED_SURFACE_FILES = [
    "lh.sphere.reg",
    "rh.sphere.reg",
    "lh.thickness",
    "rh.thickness",
    "lh.curv",
    "rh.curv",
]


def normalize_feature(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return values - mean
    return (values - mean) / std


@lru_cache(maxsize=64)
def get_hemi_resample_index(surf_dir: str, hemi: str, target_points: int) -> np.ndarray:
    coords, _ = nib.freesurfer.read_geometry(os.path.join(surf_dir, f"{hemi}.sphere.reg"))
    level = infer_ico_level(int(target_points))
    target_coords, _ = get_ico_topology(level)
    return build_resample_index(coords, target_coords)


def load_hemi_features(surf_dir: str, hemi: str, target_points: int, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    coords, _ = nib.freesurfer.read_geometry(os.path.join(surf_dir, f"{hemi}.sphere.reg"))
    thickness = nib.freesurfer.read_morph_data(os.path.join(surf_dir, f"{hemi}.thickness"))
    sulc_path = os.path.join(surf_dir, f"{hemi}.sulc")
    curv = nib.freesurfer.read_morph_data(os.path.join(surf_dir, f"{hemi}.curv"))
    sulc = nib.freesurfer.read_morph_data(sulc_path) if os.path.exists(sulc_path) else curv

    indices = get_hemi_resample_index(surf_dir, hemi, target_points)
    level = infer_ico_level(int(target_points))
    ico_coords, _ = get_ico_topology(level)
    coords = ico_coords.astype(np.float32)
    thickness = thickness[indices].astype(np.float32)
    sulc = sulc[indices].astype(np.float32)
    curv = curv[indices].astype(np.float32)

    if normalize:
        thickness = normalize_feature(thickness)
        sulc = normalize_feature(sulc)
        curv = normalize_feature(curv)
        coords = coords / (coords.std(axis=0, keepdims=True) + 1e-6)

    morph = np.stack([thickness, sulc, curv], axis=-1)
    return coords, morph


def load_surface_data(surf_dir: str, num_vertices: int, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    per_hemi = max(1, int(num_vertices) // 2)
    lh_coords, lh_morph = load_hemi_features(surf_dir, "lh", per_hemi, normalize=normalize)
    rh_coords, rh_morph = load_hemi_features(surf_dir, "rh", per_hemi, normalize=normalize)
    coords = np.concatenate([lh_coords, rh_coords], axis=0)
    morph = np.concatenate([lh_morph, rh_morph], axis=0)
    return coords.astype(np.float32), morph.astype(np.float32)


def infer_surface_roi_masks(surface_coords: np.ndarray, num_rois: int) -> np.ndarray:
    coords = surface_coords.astype(np.float32, copy=False)
    coords = coords / (np.linalg.norm(coords, axis=1, keepdims=True) + 1e-6)

    azimuth = np.arctan2(coords[:, 1], coords[:, 0])
    elevation = np.arctan2(coords[:, 2], np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) + 1e-6)
    hemisphere = (coords[:, 0] >= 0).astype(np.int64)

    hemi_rois = max(1, int(np.ceil(num_rois / 2.0)))
    n_az = max(1, int(np.ceil(np.sqrt(hemi_rois))))
    n_el = max(1, int(np.ceil(hemi_rois / n_az)))

    az_edges = np.linspace(-np.pi, np.pi, n_az + 1)
    el_edges = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_el + 1)

    masks = []
    for hemi in [0, 1]:
        hemi_mask = hemisphere == hemi
        for az_idx in range(n_az):
            for el_idx in range(n_el):
                mask = (
                    hemi_mask
                    & (azimuth >= az_edges[az_idx])
                    & (azimuth < az_edges[az_idx + 1] + 1e-6)
                    & (elevation >= el_edges[el_idx])
                    & (elevation < el_edges[el_idx + 1] + 1e-6)
                )
                if mask.any():
                    masks.append(mask.astype(np.float32))
                if len(masks) >= num_rois:
                    break
            if len(masks) >= num_rois:
                break
        if len(masks) >= num_rois:
            break

    while len(masks) < num_rois:
        masks.append(np.ones(coords.shape[0], dtype=np.float32))
    return np.stack(masks[:num_rois], axis=0)


def load_surface_roi_masks(mask_path: str, num_vertices: int, num_rois: int) -> Optional[np.ndarray]:
    if not mask_path or not os.path.exists(mask_path):
        return None

    if mask_path.endswith(".npz"):
        payload = np.load(mask_path)
        masks = payload["surface_roi_masks"]
    elif mask_path.endswith(".npy"):
        masks = np.load(mask_path)
    elif mask_path.endswith(".pt"):
        import torch

        payload = torch.load(mask_path, map_location="cpu")
        masks = payload["surface_roi_masks"].cpu().numpy()
    else:
        return None

    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim != 2:
        return None
    if masks.shape[1] != num_vertices:
        source_points = masks.shape[1] // 2
        target_points = num_vertices // 2
        if source_points <= 0 or target_points <= 0:
            return None
        source_level = infer_ico_level(int(source_points))
        target_level = infer_ico_level(int(target_points))
        source_coords, _ = get_ico_topology(source_level)
        target_coords, _ = get_ico_topology(target_level)
        indices = build_resample_index(source_coords, target_coords)
        lh_masks = masks[:, :source_points][:, indices]
        rh_masks = masks[:, source_points:][:, indices]
        masks = np.concatenate([lh_masks, rh_masks], axis=1)
    if masks.shape[0] < num_rois:
        pad = np.repeat(masks[-1:, :], num_rois - masks.shape[0], axis=0)
        masks = np.concatenate([masks, pad], axis=0)
    return masks[:num_rois]
