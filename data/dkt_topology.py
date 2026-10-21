"""Build a DKT parcel graph from a triangular cortical template mesh.

The Phase 2.1 spatial tolerance metric used spherical parcel centroids.  This
module deliberately uses only mesh edges whose endpoints have different DKT
labels, so the resulting graph describes shared cortical boundaries instead.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .atlas_utils import ATLAS_IGNORE_NAMES


def _decode_names(names: Iterable[object]) -> Tuple[str, ...]:
    return tuple(name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names)


def _unit_sphere(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Sphere contains a zero-length vertex")
    return vertices / norms


def project_annotation_to_template(
    source_sphere_path: str,
    source_annot_path: str,
    target_sphere_path: str,
    target_cortex_label_path: str,
    ignored_names: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Nearest-neighbour annotation transfer in registered spherical space.

    Only valid cortical DKT parcels are used as sources.  Every target cortex
    vertex is consequently assigned one of the retained parcels, while
    non-cortex vertices remain ``-1``.
    """

    for path in (source_sphere_path, source_annot_path, target_sphere_path, target_cortex_label_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    ignored = set(ignored_names or ATLAS_IGNORE_NAMES["DKTatlas"])
    source_vertices, _ = nib.freesurfer.read_geometry(source_sphere_path)
    target_vertices, target_faces = nib.freesurfer.read_geometry(target_sphere_path)
    source_labels, _, raw_names = nib.freesurfer.read_annot(source_annot_path)
    names = _decode_names(raw_names)
    retained_label_ids = tuple(idx for idx, name in enumerate(names) if name not in ignored)
    retained_names = tuple(names[idx] for idx in retained_label_ids)
    if len(set(retained_names)) != len(retained_names):
        raise ValueError("Retained annotation names are not unique")

    label_id_to_roi = {label_id: roi_idx for roi_idx, label_id in enumerate(retained_label_ids)}
    source_valid = np.isin(source_labels, np.asarray(retained_label_ids, dtype=np.int64))
    if not np.any(source_valid):
        raise ValueError("Source annotation has no retained cortical DKT vertices")

    source_roi = np.full(source_labels.shape, -1, dtype=np.int16)
    for label_id, roi_idx in label_id_to_roi.items():
        source_roi[source_labels == label_id] = roi_idx

    cortex_vertices = nib.freesurfer.read_label(target_cortex_label_path).astype(np.int64, copy=False)
    if np.any(cortex_vertices < 0) or np.any(cortex_vertices >= target_vertices.shape[0]):
        raise ValueError("Target cortex label contains an invalid vertex index")

    source_sphere = _unit_sphere(source_vertices)
    target_sphere = _unit_sphere(target_vertices)
    tree = cKDTree(source_sphere[source_valid])
    distances, nearest = tree.query(target_sphere[cortex_vertices], k=1)
    source_valid_indices = np.flatnonzero(source_valid)

    projected = np.full(target_vertices.shape[0], -1, dtype=np.int16)
    projected[cortex_vertices] = source_roi[source_valid_indices[nearest]]
    if np.any(projected[cortex_vertices] < 0):
        raise AssertionError("Projection left an unlabeled target cortex vertex")

    return {
        "labels": projected,
        "faces": np.asarray(target_faces, dtype=np.int64),
        "region_names": retained_names,
        "cortex_vertices": cortex_vertices,
        "projection_distance_mean": float(np.mean(distances)),
        "projection_distance_max": float(np.max(distances)),
    }


def mesh_boundary_graph(
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    num_rois: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary parcel adjacency and shared-boundary mesh-edge counts."""

    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(vertex_labels, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces [F,3], got {faces.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected one label per vertex, got {labels.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= labels.size):
        raise ValueError("Mesh face contains an invalid vertex index")

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    left = labels[edges[:, 0]]
    right = labels[edges[:, 1]]
    valid = (left >= 0) & (right >= 0) & (left < num_rois) & (right < num_rois) & (left != right)
    left = left[valid]
    right = right[valid]

    boundary = np.zeros((num_rois, num_rois), dtype=np.int64)
    np.add.at(boundary, (left, right), 1)
    np.add.at(boundary, (right, left), 1)
    adjacency = (boundary > 0).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)
    np.fill_diagonal(boundary, 0)
    return adjacency, boundary


def row_normalize_boundary(boundary: np.ndarray) -> np.ndarray:
    boundary = np.asarray(boundary, dtype=np.float64)
    row_sum = boundary.sum(axis=1, keepdims=True)
    return np.divide(boundary, row_sum, out=np.zeros_like(boundary), where=row_sum > 0)


def build_fsaverage_dkt_topology(
    fsaverage_dir: str,
    source_subject_dir: str,
) -> Dict[str, object]:
    """Build 62-parcel DKT topology on fsaverage from a registered source annot."""

    hemi_payloads: Dict[str, Dict[str, object]] = {}
    for hemi in ("lh", "rh"):
        hemi_payloads[hemi] = project_annotation_to_template(
            source_sphere_path=os.path.join(source_subject_dir, "surf", f"{hemi}.sphere.reg"),
            source_annot_path=os.path.join(source_subject_dir, "label", f"{hemi}.aparc.DKTatlas.annot"),
            target_sphere_path=os.path.join(fsaverage_dir, "surf", f"{hemi}.sphere"),
            target_cortex_label_path=os.path.join(fsaverage_dir, "label", f"{hemi}.cortex.label"),
        )

    left_names = tuple(hemi_payloads["lh"]["region_names"])
    right_names = tuple(hemi_payloads["rh"]["region_names"])
    if set(left_names) != set(right_names):
        raise ValueError("Left/right DKT annotations do not contain the same retained labels")
    right_name_to_local = {name: idx for idx, name in enumerate(right_names)}
    per_hemi = len(left_names)
    num_rois = per_hemi * 2
    if num_rois != 62:
        raise ValueError(f"Expected 62 retained DKT parcels, got {num_rois}")

    adjacency = np.zeros((num_rois, num_rois), dtype=np.uint8)
    boundary = np.zeros((num_rois, num_rois), dtype=np.int64)
    hemisphere_stats: Dict[str, Dict[str, object]] = {}
    for hemi, offset in (("lh", 0), ("rh", per_hemi)):
        payload = hemi_payloads[hemi]
        local_adjacency, local_boundary = mesh_boundary_graph(
            payload["faces"], payload["labels"], per_hemi
        )
        adjacency[offset : offset + per_hemi, offset : offset + per_hemi] = local_adjacency
        boundary[offset : offset + per_hemi, offset : offset + per_hemi] = local_boundary
        labels = np.asarray(payload["labels"])
        cortex = np.asarray(payload["cortex_vertices"])
        hemisphere_stats[hemi] = {
            "vertices": int(labels.size),
            "cortex_vertices": int(cortex.size),
            "faces": int(np.asarray(payload["faces"]).shape[0]),
            "boundary_edges": int(local_boundary.sum() // 2),
            "projection_distance_mean": float(payload["projection_distance_mean"]),
            "projection_distance_max": float(payload["projection_distance_max"]),
            "invalid_cortex_labels": int(np.count_nonzero((labels[cortex] < 0) | (labels[cortex] >= per_hemi))),
        }

    homology = np.zeros_like(adjacency)
    homolog_index = np.full(num_rois, -1, dtype=np.int16)
    for left_idx, name in enumerate(left_names):
        right_idx = per_hemi + right_name_to_local[name]
        homology[left_idx, right_idx] = 1
        homology[right_idx, left_idx] = 1
        homolog_index[left_idx] = right_idx
        homolog_index[right_idx] = left_idx

    region_names = tuple(f"lh_{name}" for name in left_names) + tuple(
        f"rh_{name}" for name in right_names
    )
    if not np.array_equal(adjacency, adjacency.T) or not np.array_equal(boundary, boundary.T):
        raise AssertionError("Raw mesh adjacency/boundary matrix is not symmetric")
    if np.any(np.diag(adjacency)) or np.any(np.diag(boundary)) or np.any(np.diag(homology)):
        raise AssertionError("Topology matrices must have zero diagonal")
    if not np.array_equal(homology, homology.T) or not np.all(homology.sum(axis=1) == 1):
        raise AssertionError("Each DKT parcel must have exactly one bilateral homologue")

    degrees = adjacency.sum(axis=1).astype(np.int16)
    return {
        "region_names": region_names,
        "adjacency": adjacency,
        "boundary_edge_count": boundary,
        "boundary_strength": row_normalize_boundary(boundary).astype(np.float32),
        "homology": homology,
        "homolog_index": homolog_index,
        "degree": degrees,
        "hemisphere_stats": hemisphere_stats,
        "validation": {
            "raw_adjacency_symmetric": True,
            "raw_boundary_symmetric": True,
            "zero_diagonal": True,
            "exact_one_to_one_homology": True,
            "invalid_dkt_labels": int(sum(v["invalid_cortex_labels"] for v in hemisphere_stats.values())),
        },
    }


def reorder_topology(topology: Mapping[str, object], target_region_names: Sequence[str]) -> Dict[str, object]:
    """Reorder topology arrays to the exact model ROI order."""

    source_names = tuple(str(name) for name in topology["region_names"])
    target_names = tuple(str(name) for name in target_region_names)
    if set(source_names) != set(target_names) or len(source_names) != len(target_names):
        missing = sorted(set(target_names) - set(source_names))
        extra = sorted(set(source_names) - set(target_names))
        raise ValueError(f"Topology/model ROI mismatch: missing={missing}, extra={extra}")
    order = np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)
    result = dict(topology)
    for key in ("adjacency", "boundary_edge_count", "boundary_strength", "homology"):
        matrix = np.asarray(topology[key])
        result[key] = matrix[np.ix_(order, order)]
    result["degree"] = np.asarray(result["adjacency"]).sum(axis=1).astype(np.int16)
    result["homolog_index"] = np.asarray(result["homology"]).argmax(axis=1).astype(np.int16)
    result["region_names"] = target_names
    return result
