import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


PSEUDO_ATLAS_NAMES = {"pseudo", "pseudo_kmeans", "pseudo-atlas", "learned_pseudo"}


def is_pseudo_atlas_name(name: str) -> bool:
    return str(name or "").strip().lower() in PSEUDO_ATLAS_NAMES


def _safe_fold_name(split_csv: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(split_csv)))
    return parent or "fold"


def pseudo_atlas_path(root: str, split_csv: str, num_vertices: int, num_rois: int, seed: int) -> str:
    fold_name = _safe_fold_name(split_csv)
    filename = f"pseudo_atlas_{fold_name}_v{int(num_vertices)}_k{int(num_rois)}_seed{int(seed)}.npz"
    return os.path.join(str(root), filename)


def surface_labels_to_masks(labels: np.ndarray, num_rois: int, num_vertices: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    masks = np.zeros((int(num_rois), int(num_vertices)), dtype=np.float32)
    valid = (labels >= 0) & (labels < int(num_rois))
    masks[labels[valid], np.nonzero(valid)[0]] = 1.0
    empty = masks.sum(axis=1) <= 0
    if empty.any():
        fallback = np.ones((int(num_vertices),), dtype=np.float32)
        masks[empty] = fallback
    return masks


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def _kmeans(features: np.ndarray, k: int, seed: int, max_iter: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    features = np.asarray(features, dtype=np.float32)
    n = int(features.shape[0])
    k = max(1, min(int(k), n))
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = features[first]
    min_dist = np.sum((features - centers[0]) ** 2, axis=1)
    for center_idx in range(1, k):
        probs = min_dist / max(float(min_dist.sum()), 1e-12)
        chosen = int(rng.choice(n, p=probs))
        centers[center_idx] = features[chosen]
        dist = np.sum((features - centers[center_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)

    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(int(max_iter)):
        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        new_labels = distances.argmin(axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for center_idx in range(k):
            members = features[labels == center_idx]
            if members.size == 0:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                centers[center_idx] = features[farthest]
            else:
                centers[center_idx] = members.mean(axis=0)
    return labels, centers


def fit_pseudo_atlas(
    samples: Sequence[Dict[str, object]],
    load_surface_data_fn,
    num_vertices: int,
    num_rois: int,
    normalize_surface: bool,
    seed: int,
    max_subjects: int = 256,
    geometry_weight: float = 1.0,
    morph_weight: float = 1.0,
) -> Dict[str, np.ndarray]:
    if not samples:
        raise RuntimeError("Cannot fit pseudo-atlas without training samples")

    rng = np.random.default_rng(int(seed))
    selected = list(samples)
    if int(max_subjects) > 0 and len(selected) > int(max_subjects):
        indices = np.sort(rng.choice(len(selected), size=int(max_subjects), replace=False))
        selected = [selected[int(idx)] for idx in indices]

    morph_sum = None
    coords_ref = None
    count = 0
    for sample in selected:
        coords, morph = load_surface_data_fn(
            str(sample["surf_dir"]),
            num_vertices=int(num_vertices),
            normalize=bool(normalize_surface),
        )
        if coords_ref is None:
            coords_ref = coords.astype(np.float32, copy=True)
            morph_sum = np.zeros_like(morph, dtype=np.float64)
        morph_sum += morph.astype(np.float64)
        count += 1

    if coords_ref is None or morph_sum is None or count <= 0:
        raise RuntimeError("No valid surface data found for pseudo-atlas fitting")

    mean_morph = (morph_sum / float(count)).astype(np.float32)
    per_hemi = max(1, int(num_vertices) // 2)
    roi_left = int(num_rois) // 2
    roi_right = int(num_rois) - roi_left
    labels = np.full((int(num_vertices),), fill_value=-1, dtype=np.int16)
    centroids = np.zeros((int(num_rois), 3), dtype=np.float32)

    for hemi_idx, (start, stop, k_hemi, offset) in enumerate(
        [(0, per_hemi, roi_left, 0), (per_hemi, per_hemi * 2, roi_right, roi_left)]
    ):
        coords = coords_ref[start:stop]
        morph = mean_morph[start:stop]
        features = np.concatenate(
            [
                _normalize_columns(coords) * float(geometry_weight),
                _normalize_columns(morph) * float(morph_weight),
            ],
            axis=1,
        )
        hemi_labels, _ = _kmeans(features, k=k_hemi, seed=int(seed) + 1009 * hemi_idx)
        labels[start:stop] = (hemi_labels + offset).astype(np.int16)
        for local_idx in range(k_hemi):
            members = coords[hemi_labels == local_idx]
            if members.size == 0:
                members = coords
            center = members.mean(axis=0)
            center = center / (np.linalg.norm(center) + 1e-6)
            centroids[offset + local_idx] = center.astype(np.float32)

    region_names = np.asarray([f"pseudo_roi_{idx:02d}" for idx in range(int(num_rois))])
    return {
        "surface_labels": labels,
        "surface_centroids": centroids,
        "region_names": region_names,
        "num_subjects_fit": np.asarray(count, dtype=np.int32),
        "seed": np.asarray(int(seed), dtype=np.int32),
    }


def load_or_fit_pseudo_atlas(
    root: str,
    split_csv: str,
    samples: Sequence[Dict[str, object]],
    load_surface_data_fn,
    num_vertices: int,
    num_rois: int,
    normalize_surface: bool,
    seed: int,
    max_subjects: int = 256,
    overwrite: bool = False,
) -> Dict[str, np.ndarray]:
    if not root:
        raise ValueError("data.atlas.pseudo_root must be set for pseudo-atlas mode")
    os.makedirs(str(root), exist_ok=True)
    path = pseudo_atlas_path(root, split_csv, num_vertices, num_rois, seed)
    if os.path.exists(path) and not overwrite:
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    definition = fit_pseudo_atlas(
        samples=samples,
        load_surface_data_fn=load_surface_data_fn,
        num_vertices=num_vertices,
        num_rois=num_rois,
        normalize_surface=normalize_surface,
        seed=seed,
        max_subjects=max_subjects,
    )
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "wb") as f:
            np.savez(f, **definition)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return definition


def pseudo_volume_label_maps(
    volume_3d: np.ndarray,
    surface_centroids: np.ndarray,
    num_rois: int,
    threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume_3d = np.asarray(volume_3d, dtype=np.float32)
    brain_mask = (volume_3d > float(threshold))
    roi_label_map = np.full(volume_3d.shape, fill_value=-1, dtype=np.int16)
    if not brain_mask.any():
        global_mask = np.zeros(volume_3d.shape, dtype=np.float32)
        return roi_label_map.copy(), roi_label_map, global_mask

    coords = np.argwhere(brain_mask).astype(np.float32)
    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    norm_coords = 2.0 * (coords - mins) / np.maximum(maxs - mins, 1.0) - 1.0
    norm_coords = norm_coords / (np.linalg.norm(norm_coords, axis=1, keepdims=True) + 1e-6)

    centroids = np.asarray(surface_centroids, dtype=np.float32)
    half = max(1, int(num_rois) // 2)
    x_mid = 0.5 * float(mins[0, 0] + maxs[0, 0])
    left_sel = coords[:, 0] < x_mid
    right_sel = ~left_sel
    labels = np.zeros((coords.shape[0],), dtype=np.int16)
    for selector, start, stop in [(left_sel, 0, half), (right_sel, half, int(num_rois))]:
        if not selector.any():
            continue
        local_centroids = centroids[start:stop]
        if local_centroids.size == 0:
            local_centroids = centroids
            start = 0
        distances = np.sum((norm_coords[selector, None, :] - local_centroids[None, :, :]) ** 2, axis=-1)
        labels[selector] = (distances.argmin(axis=1) + int(start)).astype(np.int16)
    idx = coords.astype(np.int64)
    roi_label_map[idx[:, 0], idx[:, 1], idx[:, 2]] = labels
    local_label_map = roi_label_map.copy()
    return local_label_map, roi_label_map, brain_mask.astype(np.float32)
