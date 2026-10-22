import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from .surface_utils import get_hemi_resample_index


ATLAS_NAME_ALIASES = {
    "dkt": "DKTatlas",
    "dktatlas": "DKTatlas",
    "aparc.dktatlas": "DKTatlas",
    "aparc": "aparc",
    "desikan": "aparc",
    "desikan-killiany": "aparc",
    "a2009s": "a2009s",
    "destrieux": "a2009s",
    "aal": "aal116",
    "aal116": "aal116",
    "aal_spm12": "aal116",
    "aal-spm12": "aal116",
}

ATLAS_IGNORE_NAMES = {
    "DKTatlas": {"unknown", "bankssts", "corpuscallosum", "frontalpole", "temporalpole"},
    "aparc": {"unknown", "corpuscallosum"},
    "a2009s": {"unknown", "corpuscallosum"},
}

ATLAS_BASENAME = {
    "DKTatlas": "aparc.DKTatlas",
    "aparc": "aparc",
    "a2009s": "aparc.a2009s",
}

TEMPLATE_ATLASES = {"aal116"}
DEFAULT_AAL_ATLAS_ROOT = "/home_data/home/zhouwl2023/Data/atlases/aal_SPM12"


def canonical_atlas_name(name: str) -> str:
    raw = str(name or "DKTatlas").strip()
    lowered = raw.lower()
    return ATLAS_NAME_ALIASES.get(lowered, raw)


def is_template_atlas(name: str) -> bool:
    return canonical_atlas_name(name) in TEMPLATE_ATLASES


def atlas_surface_filename(hemi: str, atlas_name: str) -> str:
    atlas_name = canonical_atlas_name(atlas_name)
    return f"{hemi}.{ATLAS_BASENAME[atlas_name]}.annot"


def atlas_volume_filename(atlas_name: str) -> str:
    atlas_name = canonical_atlas_name(atlas_name)
    return f"{ATLAS_BASENAME[atlas_name]}+aseg.mgz"


def _decode_names(names: List[bytes]) -> List[str]:
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]


def _aal_atlas_root() -> str:
    return os.environ.get("AAL_ATLAS_ROOT", DEFAULT_AAL_ATLAS_ROOT)


def _aal_image_path() -> str:
    return os.path.join(_aal_atlas_root(), "ROI_MNI_V4.nii")


def _aal_label_path() -> str:
    return os.path.join(_aal_atlas_root(), "ROI_MNI_V4.xml")


@lru_cache(maxsize=4)
def get_template_atlas_definition(atlas_name: str) -> Dict[str, object]:
    atlas_name = canonical_atlas_name(atlas_name)
    if atlas_name != "aal116":
        raise KeyError(f"Unsupported template atlas: {atlas_name}")

    image_path = _aal_image_path()
    label_path = _aal_label_path()
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    root = ET.parse(label_path).getroot()
    entries = []
    for label_node in root.findall(".//label"):
        index_node = label_node.find("index")
        name_node = label_node.find("name")
        if index_node is None or name_node is None:
            continue
        label_value = int(index_node.text)
        region_name = str(name_node.text)
        entries.append({"name": region_name, "volume_label": label_value})
    entries.sort(key=lambda item: int(item["volume_label"]))

    return {
        "atlas_name": atlas_name,
        "image_path": image_path,
        "label_path": label_path,
        "region_entries": tuple(entries),
        "region_names": tuple(str(entry["name"]) for entry in entries),
        "num_rois": int(len(entries)),
    }


@lru_cache(maxsize=16)
def get_atlas_definition(label_dir: str, atlas_name: str) -> Dict[str, object]:
    atlas_name = canonical_atlas_name(atlas_name)
    annot_path = os.path.join(label_dir, atlas_surface_filename("lh", atlas_name))
    if not os.path.exists(annot_path):
        raise FileNotFoundError(annot_path)

    _, _, raw_names = nib.freesurfer.read_annot(annot_path)
    names = _decode_names(raw_names)
    ignored = ATLAS_IGNORE_NAMES.get(atlas_name, {"unknown", "corpuscallosum"})

    label_indices = [idx for idx, region_name in enumerate(names) if region_name not in ignored]
    region_entries = []
    region_names = []
    for hemi, hemi_offset in [("lh", 1000), ("rh", 2000)]:
        for label_idx in label_indices:
            region_name = names[label_idx]
            region_names.append(f"{hemi}_{region_name}")
            region_entries.append(
                {
                    "hemi": hemi,
                    "name": region_name,
                    "surface_label_index": int(label_idx),
                    "volume_label": int(hemi_offset + label_idx),
                }
            )

    return {
        "atlas_name": atlas_name,
        "label_indices": tuple(int(idx) for idx in label_indices),
        "base_region_names": tuple(names[idx] for idx in label_indices),
        "region_entries": tuple(region_entries),
        "region_names": tuple(region_names),
        "num_rois": int(len(region_entries)),
    }


def load_surface_roi_masks_from_atlas(fs_subject_dir: str, num_vertices: int, atlas_name: str) -> Tuple[np.ndarray, List[str]]:
    label_dir = os.path.join(fs_subject_dir, "label")
    surf_dir = os.path.join(fs_subject_dir, "surf")
    definition = get_atlas_definition(label_dir, atlas_name)
    label_indices = list(definition["label_indices"])
    per_hemi = max(1, int(num_vertices) // 2)
    total_vertices = per_hemi * 2

    masks = []
    region_names = []
    for hemi in ["lh", "rh"]:
        annot_path = os.path.join(label_dir, atlas_surface_filename(hemi, atlas_name))
        labels, _, _ = nib.freesurfer.read_annot(annot_path)
        vertex_indices = get_hemi_resample_index(surf_dir, hemi, per_hemi)
        labels = labels[vertex_indices]
        for label_idx, region_name in zip(label_indices, definition["base_region_names"]):
            hemi_mask = np.zeros(total_vertices, dtype=np.float32)
            if hemi == "lh":
                hemi_mask[:per_hemi] = (labels == label_idx).astype(np.float32)
            else:
                hemi_mask[per_hemi:] = (labels == label_idx).astype(np.float32)
            masks.append(hemi_mask)
            region_names.append(f"{hemi}_{region_name}")

    return np.stack(masks, axis=0), region_names


def resample_label_image_to_grid(
    image: nib.spatialimages.SpatialImage,
    target_shape: Sequence[int],
    target_affine: np.ndarray,
) -> np.ndarray:
    """Resample a discrete label image onto an MRI grid using nearest-neighbor interpolation."""
    shape = tuple(int(v) for v in target_shape)
    affine = np.asarray(target_affine, dtype=np.float64)
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError(f"Expected a positive 3D target shape, got {shape}")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("target_affine must be a finite 4x4 matrix")
    resampled = resample_from_to(image, (shape, affine), order=0)
    return np.rint(resampled.get_fdata()).astype(np.int32, copy=False)


def load_volume_roi_label_maps_from_atlas(
    fs_subject_dir: str,
    target_shape: Sequence[int],
    target_affine: np.ndarray,
    atlas_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    atlas_name = canonical_atlas_name(atlas_name)
    label_dir = os.path.join(fs_subject_dir, "label")
    mri_dir = os.path.join(fs_subject_dir, "mri")
    definition = get_atlas_definition(label_dir, atlas_name)

    aparc_path = os.path.join(mri_dir, atlas_volume_filename(atlas_name))
    if not os.path.exists(aparc_path):
        raise FileNotFoundError(aparc_path)

    shape = tuple(int(v) for v in target_shape)
    aparc = resample_label_image_to_grid(nib.load(aparc_path), shape, target_affine)

    lh_ribbon = None
    rh_ribbon = None
    lh_ribbon_path = os.path.join(mri_dir, "lh.ribbon.mgz")
    rh_ribbon_path = os.path.join(mri_dir, "rh.ribbon.mgz")
    if os.path.exists(lh_ribbon_path):
        lh_ribbon = resample_label_image_to_grid(nib.load(lh_ribbon_path), shape, target_affine)
    if os.path.exists(rh_ribbon_path):
        rh_ribbon = resample_label_image_to_grid(nib.load(rh_ribbon_path), shape, target_affine)

    entries = list(definition["region_entries"])
    max_volume_label = max(int(entry["volume_label"]) for entry in entries)
    roi_lookup = np.full(max_volume_label + 1, fill_value=-1, dtype=np.int16)
    hemi_lookup = np.full(max_volume_label + 1, fill_value=-1, dtype=np.int8)
    region_names = []
    for roi_idx, entry in enumerate(entries):
        volume_label = int(entry["volume_label"])
        roi_lookup[volume_label] = int(roi_idx)
        hemi_lookup[volume_label] = 0 if entry["hemi"] == "lh" else 1
        region_names.append(f"{entry['hemi']}_{entry['name']}")

    # Map every atlas voxel once instead of scanning the full volume for each ROI.
    roi_label_map = np.full(shape, fill_value=-1, dtype=np.int16)
    hemi_label_map = np.full(shape, fill_value=-1, dtype=np.int8)
    valid_labels = (aparc >= 0) & (aparc < roi_lookup.size)
    roi_label_map[valid_labels] = roi_lookup[aparc[valid_labels]]
    hemi_label_map[valid_labels] = hemi_lookup[aparc[valid_labels]]

    local_label_map = np.full(shape, fill_value=-1, dtype=np.int16)
    lh_local = (roi_label_map >= 0) & (hemi_label_map == 0)
    rh_local = (roi_label_map >= 0) & (hemi_label_map == 1)
    if lh_ribbon is not None:
        lh_local &= lh_ribbon > 0
    if rh_ribbon is not None:
        rh_local &= rh_ribbon > 0
    local_label_map[lh_local] = roi_label_map[lh_local]
    local_label_map[rh_local] = roi_label_map[rh_local]

    global_brain_mask = (roi_label_map >= 0).astype(np.float32)
    return local_label_map, roi_label_map, global_brain_mask, region_names


def load_volume_roi_label_maps_from_template_atlas(
    target_shape: Sequence[int],
    target_affine: np.ndarray,
    atlas_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    definition = get_template_atlas_definition(atlas_name)
    entries = list(definition["region_entries"])
    region_names = list(definition["region_names"])
    image = nib.load(str(definition["image_path"]))
    resampled = resample_from_to(
        image,
        (tuple(int(v) for v in target_shape), np.asarray(target_affine, dtype=np.float64)),
        order=0,
    )
    atlas = np.rint(resampled.get_fdata()).astype(np.int32, copy=False)

    max_label = max(int(entry["volume_label"]) for entry in entries)
    roi_lookup = np.full(max_label + 1, fill_value=-1, dtype=np.int16)
    for roi_idx, entry in enumerate(entries):
        roi_lookup[int(entry["volume_label"])] = int(roi_idx)

    roi_label_map = np.full(tuple(int(v) for v in target_shape), fill_value=-1, dtype=np.int16)
    valid_labels = (atlas >= 0) & (atlas < roi_lookup.size)
    roi_label_map[valid_labels] = roi_lookup[atlas[valid_labels]]
    local_label_map = roi_label_map.copy()
    global_brain_mask = (roi_label_map >= 0).astype(np.float32)
    return local_label_map, roi_label_map, global_brain_mask, region_names
