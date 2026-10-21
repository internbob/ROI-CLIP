import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .atlas_utils import (
    atlas_surface_filename,
    atlas_volume_filename,
    canonical_atlas_name,
    get_atlas_definition,
    get_template_atlas_definition,
    is_template_atlas,
    load_surface_roi_masks_from_atlas,
    load_volume_roi_label_maps_from_atlas,
    load_volume_roi_label_maps_from_template_atlas,
)
from .collate import roiclip_collate_fn
from .mask_utils import infer_local_and_roi_masks
from .preprocess_cache import load_preprocess_cache, save_preprocess_cache
from .ribbon_geometry import load_ribbon_bundle
from .pseudo_atlas import (
    is_pseudo_atlas_name,
    load_or_fit_pseudo_atlas,
    pseudo_volume_label_maps,
    surface_labels_to_masks,
)
from .surface_utils import REQUIRED_SURFACE_FILES, infer_surface_roi_masks, load_surface_data, load_surface_roi_masks
from .transforms import (
    adjust_affine_for_voxel_offset,
    center_crop_or_pad_3d,
    center_crop_or_pad_voxel_offset,
    normalize_volume_minmax,
    random_augment_volume,
)


DEFAULT_CLINICAL_FEATURES = (
    "age",
    "education",
    "apoe",
    "animal",
    "gds",
    "lm_imm",
    "lm_del",
    "faq_BILLS",
    "faq_TAXES",
    "faq_SHOPPING",
    "faq_GAMES",
    "faq_STOVE",
    "faq_MEALPREP",
    "faq_EVENTS",
    "faq_PAYATTN",
    "faq_REMDATES",
    "faq_TRAVEL",
    "npiq_DEL",
    "npiq_HALL",
    "npiq_AGIT",
    "npiq_DEPD",
    "npiq_ANX",
    "npiq_ELAT",
    "npiq_APA",
    "npiq_DISN",
    "npiq_IRR",
    "npiq_MOT",
    "npiq_NITE",
    "npiq_APP",
)


def _load_label_map(label_csv: Optional[str]) -> Dict[str, int]:
    if not label_csv:
        return {}
    label_map = {}
    with open(label_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["subject"].strip()] = int(row["label"])
    return label_map


def _split_filename(filename: str) -> Tuple[str, str]:
    parts = filename.split("_", 1)
    subject = parts[0]
    session = parts[1] if len(parts) == 2 else ""
    return subject, session


def build_mri_path(row: Dict[str, str]) -> str:
    filename = row["filename"]
    subject, session = _split_filename(filename)
    if session:
        return os.path.join(row["path"], subject, session, "t1", filename + ".nii.gz")
    return os.path.join(row["path"], subject, "t1", filename + ".nii.gz")


def find_local_mri_path(surface_root: str, subject: str) -> Optional[str]:
    candidates = [
        os.path.join(surface_root, subject, "input", "image.nii.gz"),
        os.path.join(surface_root, subject, "fake_bids", subject, "anat", f"{subject}_T1w.nii.gz"),
        os.path.join(surface_root, subject, "workbench", subject, "volumes", f"{subject}.image.nii.gz"),
        os.path.join(surface_root, subject, "freesurfer", subject, "mri", "T1.mgz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class CortexWeaveROIClipDataset(Dataset):
    REQUIRED_FILES = REQUIRED_SURFACE_FILES

    def __init__(
        self,
        surface_root: str,
        split_csv: str,
        task: str = "AD",
        label_csv: str = "",
        num_vertices: int = 2048,
        num_rois: int = 16,
        mri_shape: Sequence[int] = (160, 192, 160),
        normalize_surface: bool = True,
        train_mode: bool = False,
        require_teacher_inputs: bool = True,
        filter_teacher_available: bool = False,
        allowed_subject_prefixes: Optional[Sequence[str]] = None,
        roi_bundle_name: str = "roi_bundle.npz",
        preprocess_cache_root: str = "",
        roi_cache_shape: Optional[Sequence[int]] = None,
        allow_unaligned_control: bool = False,
        atlas_enabled: bool = False,
        atlas_name: str = "DKTatlas",
        pseudo_atlas_root: str = "",
        pseudo_atlas_seed: int = 3407,
        pseudo_atlas_max_subjects: int = 256,
        pseudo_atlas_definition: Optional[Dict[str, np.ndarray]] = None,
        mri_aug_prob: float = 0.0,
        mri_aug_max_voxel_shift: int = 0,
        mri_aug_intensity_scale: float = 0.0,
        mri_aug_intensity_shift: float = 0.0,
        mri_aug_gamma: float = 0.0,
        mri_aug_noise_std: float = 0.0,
        clinical_features: Optional[Sequence[str]] = None,
        clinical_stats: Optional[Dict[str, np.ndarray]] = None,
        ribbon_enabled: bool = False,
        ribbon_cache_root: str = "",
        ribbon_num_depths: int = 5,
        ribbon_depths: Optional[Sequence[float]] = None,
    ):
        self.surface_root = surface_root
        self.split_csv = split_csv
        self.task = task
        self.num_vertices = int(num_vertices)
        self.num_rois = int(num_rois)
        self.mri_shape = tuple(int(v) for v in mri_shape)
        self.normalize_surface = bool(normalize_surface)
        self.train_mode = bool(train_mode)
        self.require_teacher_inputs = bool(require_teacher_inputs)
        self.filter_teacher_available = bool(filter_teacher_available)
        self.allowed_subject_prefixes = tuple(str(v) for v in (allowed_subject_prefixes or []))
        self.roi_bundle_name = roi_bundle_name
        self.preprocess_cache_root = str(preprocess_cache_root or "")
        self.allow_unaligned_control = bool(allow_unaligned_control)
        self.roi_cache_shape = tuple(
            int(v) for v in (roi_cache_shape or tuple(max(1, size // 8) for size in self.mri_shape))
        )
        self.atlas_enabled = bool(atlas_enabled)
        self.atlas_name = canonical_atlas_name(atlas_name)
        self.pseudo_atlas = is_pseudo_atlas_name(self.atlas_name)
        self.pseudo_atlas_root = str(pseudo_atlas_root or "")
        self.pseudo_atlas_seed = int(pseudo_atlas_seed)
        self.pseudo_atlas_max_subjects = int(pseudo_atlas_max_subjects)
        self.pseudo_atlas_definition = pseudo_atlas_definition
        self.template_atlas = is_template_atlas(self.atlas_name)
        self.mri_aug_prob = float(mri_aug_prob)
        self.mri_aug_max_voxel_shift = int(mri_aug_max_voxel_shift)
        self.mri_aug_intensity_scale = float(mri_aug_intensity_scale)
        self.mri_aug_intensity_shift = float(mri_aug_intensity_shift)
        self.mri_aug_gamma = float(mri_aug_gamma)
        self.mri_aug_noise_std = float(mri_aug_noise_std)
        self.clinical_features = tuple(str(value) for value in (clinical_features or []))
        self.clinical_input_dim = 2 * len(self.clinical_features)
        self.ribbon_enabled = bool(ribbon_enabled)
        self.ribbon_cache_root = str(ribbon_cache_root or "")
        self.ribbon_num_depths = int(ribbon_num_depths)
        self.ribbon_depths = tuple(
            float(value)
            for value in (
                ribbon_depths
                if ribbon_depths is not None
                else np.linspace(0.0, 1.0, self.ribbon_num_depths).tolist()
            )
        )
        if self.ribbon_enabled and not self.require_teacher_inputs:
            raise ValueError("ribbon_enabled requires teacher inputs")
        if self.ribbon_enabled and not self.ribbon_cache_root:
            raise ValueError("data.ribbon_cache_root is required when volume_pooling.mode=ribbon")
        if self.ribbon_num_depths <= 0:
            raise ValueError("ribbon_num_depths must be positive")
        if len(self.ribbon_depths) != self.ribbon_num_depths:
            raise ValueError("ribbon_depths length must equal ribbon_num_depths")
        self.label_map = _load_label_map(label_csv or "")
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No valid samples found: split_csv={split_csv}")
        if clinical_stats is None:
            clinical_stats = self.fit_clinical_stats()
        self.clinical_mean = np.asarray(clinical_stats["mean"], dtype=np.float32)
        self.clinical_std = np.asarray(clinical_stats["std"], dtype=np.float32)
        self.atlas_region_names = []
        if self.atlas_enabled:
            if self.pseudo_atlas:
                if self.pseudo_atlas_definition is None:
                    self.pseudo_atlas_definition = load_or_fit_pseudo_atlas(
                        root=self.pseudo_atlas_root,
                        split_csv=self.split_csv,
                        samples=self.samples,
                        load_surface_data_fn=load_surface_data,
                        num_vertices=self.num_vertices,
                        num_rois=self.num_rois,
                        normalize_surface=self.normalize_surface,
                        seed=self.pseudo_atlas_seed,
                        max_subjects=self.pseudo_atlas_max_subjects,
                    )
                self.num_rois = int(self.pseudo_atlas_definition["surface_centroids"].shape[0])
                self.atlas_region_names = [
                    str(value) for value in list(self.pseudo_atlas_definition.get("region_names", []))
                ]
                if not self.atlas_region_names:
                    self.atlas_region_names = [f"pseudo_roi_{idx:02d}" for idx in range(self.num_rois)]
            elif self.template_atlas:
                definition = get_template_atlas_definition(self.atlas_name)
                self.num_rois = int(definition["num_rois"])
                self.atlas_region_names = list(definition["region_names"])
            else:
                fs_subject_dir = self._fs_subject_dir(
                    str(self.samples[0]["subject_root"]),
                    str(self.samples[0].get("freesurfer_subject", "")),
                )
                definition = get_atlas_definition(os.path.join(fs_subject_dir, "label"), self.atlas_name)
                self.num_rois = int(definition["num_rois"])
                self.atlas_region_names = list(definition["region_names"])

    def _resolve_label(self, row: Dict[str, str], subject: str) -> int:
        raw = row.get(self.task, "")
        if raw is not None and str(raw).strip() != "":
            return int(float(raw))
        return self.label_map.get(subject, -1)

    def _clinical_values(self, row: Dict[str, str]) -> np.ndarray:
        values = []
        for name in self.clinical_features:
            raw = str(row.get(name, "")).strip()
            try:
                values.append(float(raw) if raw else np.nan)
            except ValueError:
                values.append(np.nan)
        return np.asarray(values, dtype=np.float32)

    def fit_clinical_stats(self) -> Dict[str, np.ndarray]:
        if not self.clinical_features:
            empty = np.zeros((0,), dtype=np.float32)
            return {"mean": empty, "std": empty}
        values = np.stack([np.asarray(sample["clinical_values"], dtype=np.float32) for sample in self.samples])
        observed = np.isfinite(values)
        counts = observed.sum(axis=0)
        safe_counts = np.maximum(counts, 1)
        filled = np.where(observed, values, 0.0)
        mean = filled.sum(axis=0) / safe_counts
        variance = np.where(observed, (values - mean) ** 2, 0.0).sum(axis=0) / safe_counts
        std = np.sqrt(variance)
        mean[counts == 0] = 0.0
        std[(counts == 0) | (std < 1e-6)] = 1.0
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    @property
    def clinical_stats(self) -> Dict[str, np.ndarray]:
        return {"mean": self.clinical_mean.copy(), "std": self.clinical_std.copy()}

    def _encode_clinical(self, sample: Dict[str, object]) -> np.ndarray:
        if not self.clinical_features:
            return np.zeros((0,), dtype=np.float32)
        values = np.asarray(sample["clinical_values"], dtype=np.float32)
        observed = np.isfinite(values)
        imputed = np.where(observed, values, self.clinical_mean)
        normalized = (imputed - self.clinical_mean) / self.clinical_std
        missing = (~observed).astype(np.float32)
        return np.concatenate([normalized.astype(np.float32), missing], axis=0)

    def _find_roi_bundle_path(self, row: Dict[str, str], subject: str) -> str:
        explicit = row.get("roi_bundle_path", "").strip()
        if explicit and os.path.exists(explicit):
            return explicit

        candidates = [
            os.path.join(self.surface_root, subject, "roiclip", self.roi_bundle_name),
            os.path.join(self.surface_root, subject, "roi_clip", self.roi_bundle_name),
            os.path.join(self.surface_root, subject, "masks", self.roi_bundle_name),
            os.path.join(self.surface_root, subject, self.roi_bundle_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    def _fs_subject_dir(self, subject_root: str, freesurfer_subject: str = "") -> str:
        subject = freesurfer_subject or os.path.basename(subject_root)
        return os.path.join(subject_root, "freesurfer", subject)

    def _has_required_atlas_files(self, subject_root: str, freesurfer_subject: str = "") -> bool:
        if self.pseudo_atlas:
            return True
        if self.template_atlas:
            try:
                get_template_atlas_definition(self.atlas_name)
            except FileNotFoundError:
                return False
            return True
        fs_subject_dir = self._fs_subject_dir(subject_root, freesurfer_subject)
        label_dir = os.path.join(fs_subject_dir, "label")
        mri_dir = os.path.join(fs_subject_dir, "mri")
        required_paths = [
            os.path.join(label_dir, atlas_surface_filename("lh", self.atlas_name)),
            os.path.join(label_dir, atlas_surface_filename("rh", self.atlas_name)),
            os.path.join(mri_dir, atlas_volume_filename(self.atlas_name)),
        ]
        return all(os.path.exists(path) for path in required_paths)

    def _build_samples(self) -> List[Dict[str, object]]:
        if not os.path.exists(self.split_csv):
            raise FileNotFoundError(self.split_csv)
        samples = []
        with open(self.split_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                if self.allowed_subject_prefixes and not filename.startswith(self.allowed_subject_prefixes):
                    continue
                subject, _ = _split_filename(filename)
                surface_subject = row.get("surface_subject", "").strip() or subject
                freesurfer_subject = row.get("freesurfer_subject", "").strip() or os.path.basename(surface_subject)
                subject_root = os.path.join(self.surface_root, surface_subject)
                surf_dir = os.path.join(
                    subject_root,
                    "freesurfer",
                    freesurfer_subject,
                    "surf",
                )
                has_surface = os.path.isdir(surf_dir) and all(
                    os.path.exists(os.path.join(surf_dir, name)) for name in self.REQUIRED_FILES
                )

                require_teacher_available = self.require_teacher_inputs or self.filter_teacher_available
                if require_teacher_available and not has_surface:
                    continue

                if require_teacher_available and self.atlas_enabled and not self._has_required_atlas_files(
                    subject_root, freesurfer_subject
                ):
                    continue

                mri_path = build_mri_path(row) if row.get("path", "").strip() else ""
                if not mri_path or not os.path.exists(mri_path):
                    mri_path = find_local_mri_path(self.surface_root, surface_subject) or ""
                if not mri_path or not os.path.exists(mri_path):
                    continue

                samples.append(
                    {
                        "subject_id": filename,
                        "filename": filename,
                        "surface_subject": surface_subject,
                        "freesurfer_subject": freesurfer_subject,
                        "subject_root": subject_root,
                        "surf_dir": surf_dir,
                        "has_surface": has_surface,
                        "mri_path": mri_path,
                        "roi_bundle_path": self._find_roi_bundle_path(row, surface_subject),
                        "label": self._resolve_label(row, subject),
                        "clinical_values": self._clinical_values(row),
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, volume_path: str):
        image = nib.load(volume_path)
        volume = image.get_fdata().astype(np.float32)
        volume, voxel_offset = center_crop_or_pad_3d(volume, self.mri_shape)
        affine = adjust_affine_for_voxel_offset(np.asarray(image.affine, dtype=np.float32), voxel_offset)
        volume = normalize_volume_minmax(volume)
        volume = volume[None, ...]
        return volume, affine

    def _load_volume_geometry(self, volume_path: str) -> np.ndarray:
        image = nib.load(volume_path)
        source_shape = tuple(int(v) for v in image.shape[:3])
        voxel_offset = center_crop_or_pad_voxel_offset(source_shape, self.mri_shape)
        return adjust_affine_for_voxel_offset(np.asarray(image.affine, dtype=np.float32), voxel_offset)

    def _load_cached_inputs(self, sample, target_affine=None):
        return load_preprocess_cache(
            cache_root=self.preprocess_cache_root,
            sample_id=str(sample["subject_id"]),
            num_vertices=self.num_vertices,
            num_rois=self.num_rois,
            roi_shape=self.roi_cache_shape,
            target_mri_shape=self.mri_shape,
            target_mri_affine=target_affine,
            allow_unaligned_control=self.allow_unaligned_control,
        )

    def _load_ribbon_inputs(self, sample, target_affine):
        return load_ribbon_bundle(
            cache_root=self.ribbon_cache_root,
            sample_id=str(sample["subject_id"]),
            num_vertices=self.num_vertices,
            num_depths=self.ribbon_num_depths,
            target_shape=self.mri_shape,
            target_affine=target_affine,
            expected_depths=self.ribbon_depths,
        )

    def write_preprocess_cache(self, sample, overwrite: bool = False) -> str:
        if not self.preprocess_cache_root:
            raise ValueError("preprocess_cache_root must be set before writing caches")
        volume_3d = None
        affine = None
        if not self.atlas_enabled or self.template_atlas or self.pseudo_atlas:
            image, affine = self._load_volume(str(sample["mri_path"]))
            volume_3d = image[0]
        else:
            # Real FreeSurfer atlases only need the cropped MRI grid geometry.
            # Avoid loading the full MRI array during parallel cache generation.
            affine = self._load_volume_geometry(str(sample["mri_path"]))
        surface_data, surface_roi_masks, local_masks, roi_masks, _ = self._load_teacher_inputs(
            sample, volume_3d, affine
        )
        return save_preprocess_cache(
            cache_root=self.preprocess_cache_root,
            sample_id=str(sample["subject_id"]),
            surface_data=surface_data,
            surface_roi_masks=surface_roi_masks,
            local_label_map=local_masks,
            roi_label_map=roi_masks,
            roi_shape=self.roi_cache_shape,
            target_mri_shape=self.mri_shape,
            target_mri_affine=affine,
            overwrite=overwrite,
        )

    def _match_mask_shape(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=np.float32)
        if masks.shape[-3:] == self.mri_shape:
            return masks
        aligned = []
        for roi_idx in range(masks.shape[0]):
            cropped, _ = center_crop_or_pad_3d(masks[roi_idx], self.mri_shape)
            aligned.append(cropped.astype(np.float32))
        return np.stack(aligned, axis=0)

    def _load_teacher_inputs(self, sample, volume_3d, affine=None):
        fs_subject_dir = self._fs_subject_dir(
            str(sample["subject_root"]), str(sample.get("freesurfer_subject", ""))
        )
        if sample["has_surface"]:
            surface_coords, surface_data = load_surface_data(
                str(sample["surf_dir"]),
                num_vertices=self.num_vertices,
                normalize=self.normalize_surface,
            )
        else:
            surface_coords = np.zeros((self.num_vertices, 3), dtype=np.float32)
            surface_data = np.zeros((self.num_vertices, 3), dtype=np.float32)

        if self.atlas_enabled:
            if self.pseudo_atlas:
                definition = self.pseudo_atlas_definition
                if definition is None:
                    raise RuntimeError("Pseudo-atlas definition has not been fitted or loaded")
                surface_roi_masks = surface_labels_to_masks(
                    np.asarray(definition["surface_labels"], dtype=np.int16),
                    num_rois=self.num_rois,
                    num_vertices=self.num_vertices,
                )
                local_label_map, roi_label_map, global_brain_mask = pseudo_volume_label_maps(
                    volume_3d=volume_3d,
                    surface_centroids=np.asarray(definition["surface_centroids"], dtype=np.float32),
                    num_rois=self.num_rois,
                )
            elif self.template_atlas:
                surface_roi_masks = infer_surface_roi_masks(surface_coords, self.num_rois)
                if affine is None:
                    raise ValueError("Template atlas requires target MRI affine")
                local_label_map, roi_label_map, global_brain_mask, _ = load_volume_roi_label_maps_from_template_atlas(
                    target_shape=self.mri_shape,
                    target_affine=affine,
                    atlas_name=self.atlas_name,
                )
            else:
                if affine is None:
                    raise ValueError("FreeSurfer atlas requires target MRI affine")
                surface_roi_masks, _ = load_surface_roi_masks_from_atlas(
                    fs_subject_dir=fs_subject_dir,
                    num_vertices=self.num_vertices,
                    atlas_name=self.atlas_name,
                )
                local_label_map, roi_label_map, global_brain_mask, _ = load_volume_roi_label_maps_from_atlas(
                    fs_subject_dir=fs_subject_dir,
                    target_shape=self.mri_shape,
                    target_affine=affine,
                    atlas_name=self.atlas_name,
                )
            return surface_data, surface_roi_masks, local_label_map, roi_label_map, global_brain_mask

        mask_bundle_path = str(sample["roi_bundle_path"])
        surface_roi_masks = load_surface_roi_masks(mask_bundle_path, self.num_vertices, self.num_rois)
        local_masks = None
        roi_masks = None
        global_brain_mask = None

        if mask_bundle_path and os.path.exists(mask_bundle_path):
            payload = np.load(mask_bundle_path)
            local_masks = payload["local_ribbon_masks"] if "local_ribbon_masks" in payload else None
            roi_masks = payload["roi_volume_masks"] if "roi_volume_masks" in payload else None
            global_brain_mask = payload["global_brain_mask"] if "global_brain_mask" in payload else None

        if surface_roi_masks is None:
            surface_roi_masks = infer_surface_roi_masks(surface_coords, self.num_rois)

        if local_masks is None or roi_masks is None:
            local_masks, roi_masks, global_brain_mask = infer_local_and_roi_masks(volume_3d, self.num_rois)

        if global_brain_mask is None:
            global_brain_mask = (volume_3d > 0.05).astype(np.float32)

        local_masks = self._match_mask_shape(local_masks)
        roi_masks = self._match_mask_shape(roi_masks)
        global_brain_mask, _ = center_crop_or_pad_3d(np.asarray(global_brain_mask, dtype=np.float32), self.mri_shape)
        return surface_data, surface_roi_masks, local_masks, roi_masks, global_brain_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # MRI is intentionally read from the original /public_bme source. The
        # cache contains only compact surface and downsampled atlas inputs.
        image, affine = self._load_volume(str(sample["mri_path"]))
        cached = self._load_cached_inputs(sample, target_affine=affine) if self.require_teacher_inputs else None
        image = random_augment_volume(
            image,
            prob=self.mri_aug_prob if self.train_mode else 0.0,
            max_voxel_shift=self.mri_aug_max_voxel_shift,
            intensity_scale=self.mri_aug_intensity_scale,
            intensity_shift=self.mri_aug_intensity_shift,
            gamma=self.mri_aug_gamma,
            noise_std=self.mri_aug_noise_std,
        )

        if self.require_teacher_inputs:
            if cached is None:
                surface_data, surface_roi_masks, local_masks, roi_masks, global_brain_mask = self._load_teacher_inputs(
                    sample, image[0], affine
                )
            else:
                surface_data = cached["surface_data"]
                surface_roi_masks = cached["surface_roi_masks"]
                local_masks = cached["local_label_map"]
                roi_masks = cached["roi_label_map"]
                global_brain_mask = cached["global_brain_mask"]
            has_teacher_inputs = np.array(1.0, dtype=np.float32)
        else:
            # Keep collate-compatible placeholders without allocating unused ROI tensors.
            surface_data = np.zeros((1, 3), dtype=np.float32)
            surface_roi_masks = np.zeros((1, 1), dtype=np.float32)
            local_masks = np.zeros((1, 1, 1), dtype=np.int16 if self.atlas_enabled else np.float32)
            roi_masks = np.zeros((1, 1, 1), dtype=np.int16 if self.atlas_enabled else np.float32)
            global_brain_mask = np.zeros((1, 1, 1), dtype=np.float32)
            has_teacher_inputs = np.array(0.0, dtype=np.float32)

        if self.ribbon_enabled:
            ribbon = self._load_ribbon_inputs(sample, target_affine=affine)
            ribbon_grid = ribbon["ribbon_grid"]
            ribbon_valid_mask = ribbon["ribbon_valid_mask"]
            vertex_valid_mask = ribbon["vertex_valid_mask"]
            vertex_roi_index = ribbon["vertex_roi_index"]
        else:
            # Fixed-size placeholders keep the existing collate contract simple.
            ribbon_grid = np.zeros((1, 1, 3), dtype=np.float32)
            ribbon_valid_mask = np.zeros((1, 1), dtype=np.bool_)
            vertex_valid_mask = np.zeros((1,), dtype=np.bool_)
            vertex_roi_index = np.full((1,), -1, dtype=np.int64)

        return {
            "subject_id": str(sample["subject_id"]),
            "filename": str(sample["filename"]),
            "mri_path": str(sample["mri_path"]),
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "clinical_data": torch.from_numpy(self._encode_clinical(sample)).float(),
            "surface_data": torch.from_numpy(surface_data).float(),
            "surface_roi_masks": torch.from_numpy(surface_roi_masks).float(),
            "local_ribbon_masks": torch.from_numpy(local_masks).short() if self.atlas_enabled else torch.from_numpy(local_masks).float(),
            "roi_volume_masks": torch.from_numpy(roi_masks).short() if self.atlas_enabled else torch.from_numpy(roi_masks).float(),
            "global_brain_mask": torch.from_numpy(global_brain_mask).float(),
            "affine": torch.from_numpy(affine).float(),
            "has_teacher_inputs": torch.tensor(has_teacher_inputs, dtype=torch.float32),
            "ribbon_grid": torch.from_numpy(ribbon_grid).float(),
            "ribbon_valid_mask": torch.from_numpy(ribbon_valid_mask).bool(),
            "vertex_valid_mask": torch.from_numpy(vertex_valid_mask).bool(),
            "vertex_roi_index": torch.from_numpy(vertex_roi_index).long(),
        }


def build_dataloaders(
    cfg,
    fold: int,
    require_teacher_inputs: bool = True,
    filter_teacher_available: bool = False,
):
    split_root = os.path.join(str(cfg.data.split_root), f"cross{fold}")
    clinical_cfg = getattr(cfg.data, "clinical", {})
    clinical_features = (
        list(getattr(clinical_cfg, "features", DEFAULT_CLINICAL_FEATURES))
        if bool(getattr(clinical_cfg, "enabled", False))
        else []
    )
    volume_pooling_cfg = getattr(cfg.model, "volume_pooling", {})
    volume_pooling_mode = str(getattr(volume_pooling_cfg, "mode", "legacy_mask")).lower()
    cross_geometry_cfg = getattr(cfg.model, "cross_geometry", {})
    cross_geometry_enabled = bool(getattr(cross_geometry_cfg, "enabled", False))
    ribbon_cfg = getattr(cfg.model, "ribbon", {})
    ribbon_depths = list(getattr(ribbon_cfg, "depths", [0.0, 0.25, 0.5, 0.75, 1.0]))
    configured_num_depths = int(getattr(ribbon_cfg, "num_depths", len(ribbon_depths)))
    if configured_num_depths != len(ribbon_depths):
        raise ValueError(
            f"model.ribbon.num_depths={configured_num_depths} does not match "
            f"len(model.ribbon.depths)={len(ribbon_depths)}"
        )
    ribbon_enabled = bool(
        require_teacher_inputs
        and (
            volume_pooling_mode in {"ribbon", "hybrid_ribbon"}
            or cross_geometry_enabled
        )
    )
    if ribbon_enabled and int(cfg.data.mri_aug.max_voxel_shift) != 0:
        raise ValueError(
            "Ribbon grids are fixed to the MRI geometry; set data.mri_aug.max_voxel_shift=0 "
            "or implement the same spatial transform for ribbon_grid."
        )
    common_kwargs = dict(
        surface_root=str(cfg.data.surface_root),
        task=str(cfg.data.task),
        label_csv=str(getattr(cfg.data, "label_csv", "")),
        num_vertices=int(cfg.data.num_vertices),
        num_rois=int(cfg.model.num_rois),
        mri_shape=tuple(int(v) for v in cfg.data.mri_shape),
        normalize_surface=bool(cfg.data.normalize_surface),
        require_teacher_inputs=require_teacher_inputs,
        filter_teacher_available=filter_teacher_available,
        allowed_subject_prefixes=list(getattr(cfg.data, "allowed_subject_prefixes", [])),
        roi_bundle_name=str(cfg.data.roi_bundle_name),
        preprocess_cache_root=str(getattr(cfg.data, "preprocess_cache_root", "")),
        roi_cache_shape=tuple(int(v) for v in getattr(cfg.data, "roi_cache_shape", (20, 24, 20))),
        allow_unaligned_control=bool(getattr(cfg.data, "allow_unaligned_control", False)),
        atlas_enabled=bool(getattr(getattr(cfg.data, "atlas", {}), "enabled", False)),
        atlas_name=str(getattr(getattr(cfg.data, "atlas", {}), "name", "DKTatlas")),
        pseudo_atlas_root=str(getattr(getattr(cfg.data, "atlas", {}), "pseudo_root", "")),
        pseudo_atlas_seed=int(getattr(getattr(cfg.data, "atlas", {}), "pseudo_seed", cfg.training.seed)) + int(fold),
        pseudo_atlas_max_subjects=int(getattr(getattr(cfg.data, "atlas", {}), "pseudo_max_subjects", 256)),
        mri_aug_prob=float(cfg.data.mri_aug.prob),
        mri_aug_max_voxel_shift=int(cfg.data.mri_aug.max_voxel_shift),
        mri_aug_intensity_scale=float(cfg.data.mri_aug.intensity_scale),
        mri_aug_intensity_shift=float(cfg.data.mri_aug.intensity_shift),
        mri_aug_gamma=float(cfg.data.mri_aug.gamma),
        mri_aug_noise_std=float(cfg.data.mri_aug.noise_std),
        clinical_features=clinical_features,
        ribbon_enabled=ribbon_enabled,
        ribbon_cache_root=str(getattr(cfg.data, "ribbon_cache_root", "")),
        ribbon_num_depths=len(ribbon_depths),
        ribbon_depths=ribbon_depths,
    )
    ds_train = CortexWeaveROIClipDataset(
        split_csv=os.path.join(split_root, "train.csv"),
        train_mode=True,
        **common_kwargs,
    )
    ds_val = CortexWeaveROIClipDataset(
        split_csv=os.path.join(split_root, "valid.csv"),
        train_mode=False,
        clinical_stats=ds_train.clinical_stats,
        pseudo_atlas_definition=ds_train.pseudo_atlas_definition,
        **common_kwargs,
    )
    evaluate_test = bool(getattr(getattr(cfg, "runtime", {}), "evaluate_test", True))
    ds_test = None
    if evaluate_test:
        ds_test = CortexWeaveROIClipDataset(
            split_csv=os.path.join(split_root, "test.csv"),
            train_mode=False,
            clinical_stats=ds_train.clinical_stats,
            pseudo_atlas_definition=ds_train.pseudo_atlas_definition,
            **common_kwargs,
        )
    cfg.model.num_rois = int(ds_train.num_rois)
    cfg.model.clinical_input_dim = int(ds_train.clinical_input_dim)

    num_workers = int(cfg.training.num_workers)
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": roiclip_collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        ds_train,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            ds_test,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if ds_test is not None
        else None
    )
    return train_loader, val_loader, test_loader
