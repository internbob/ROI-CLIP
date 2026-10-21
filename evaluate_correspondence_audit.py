#!/usr/bin/env python3
"""Phase 2.1.5 validation-only Surface<->Ribbon correspondence audit.

This program loads frozen Phase 2.1 checkpoints.  It never creates an
optimizer, never calls backward, and never constructs a held-out test dataset.
The optional gradient diagnostic uses ``torch.autograd.grad`` only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader

from config import Config
from data.collate import roiclip_collate_fn
from data.dataset import CortexWeaveROIClipDataset
from data.dkt_topology import build_fsaverage_dkt_topology, reorder_topology
from data.ico_utils import get_ico_topology, infer_ico_level
from losses import classification_loss, cross_geometry_correspondence_loss
from train_teacher import _load_reference_surface_roi_masks, _roi_hemi_and_base, build_model
from utils import move_to_device


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = (
    ROOT / "outputs/vnext_phase2_1_20261002/sweep_lambda001_fold0/fold_0/best_teacher.pt",
    ROOT / "outputs/vnext_phase2_1_20261002/cv3_lambda001_fold1/fold_1/best_teacher.pt",
    ROOT / "outputs/vnext_phase2_1_20261002/cv3_lambda001_fold2/fold_2/best_teacher.pt",
)
DEFAULT_OUTPUT = ROOT / "outputs/vnext_phase2_1_5_20261003"
DEFAULT_FS_HOME = Path("/home_data/home/zhouwl2023/softwares/freesurfer")


def as_config(value):
    if isinstance(value, dict):
        return Config({key: as_config(child) for key, child in value.items()})
    if isinstance(value, list):
        return [as_config(child) for child in value]
    return value


def jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def model_inputs(batch):
    return {
        "image": batch["image"],
        "surface_data": batch["surface_data"],
        "surface_roi_masks": batch["surface_roi_masks"],
        "local_ribbon_masks": batch["local_ribbon_masks"],
        "roi_volume_masks": batch["roi_volume_masks"],
        "labels": batch["label"],
        "clinical_data": batch["clinical_data"],
        "ribbon_grid": batch["ribbon_grid"],
        "ribbon_valid_mask": batch["ribbon_valid_mask"],
        "vertex_valid_mask": batch["vertex_valid_mask"],
        "vertex_roi_index": batch["vertex_roi_index"],
    }


def build_audit_datasets(cfg, fold: int, state: Mapping[str, object]):
    """Construct train-reference and validation datasets, intentionally no test."""

    split_root = Path(str(cfg.data.split_root)) / f"cross{fold}"
    train_csv = split_root / "train.csv"
    valid_csv = split_root / "valid.csv"
    for path in (train_csv, valid_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    clinical_features = list(state.get("clinical_features", []))
    ribbon_cfg = cfg.model.ribbon
    ribbon_depths = list(ribbon_cfg.depths)
    kwargs = dict(
        surface_root=str(cfg.data.surface_root),
        task=str(cfg.data.task),
        label_csv=str(getattr(cfg.data, "label_csv", "")),
        num_vertices=int(cfg.data.num_vertices),
        num_rois=int(cfg.model.num_rois),
        mri_shape=tuple(int(value) for value in cfg.data.mri_shape),
        normalize_surface=bool(cfg.data.normalize_surface),
        require_teacher_inputs=True,
        filter_teacher_available=True,
        allowed_subject_prefixes=list(getattr(cfg.data, "allowed_subject_prefixes", [])),
        roi_bundle_name=str(cfg.data.roi_bundle_name),
        preprocess_cache_root=str(cfg.data.preprocess_cache_root),
        roi_cache_shape=tuple(int(value) for value in cfg.data.roi_cache_shape),
        allow_unaligned_control=bool(getattr(cfg.data, "allow_unaligned_control", False)),
        atlas_enabled=bool(cfg.data.atlas.enabled),
        atlas_name=str(cfg.data.atlas.name),
        mri_aug_prob=0.0,
        mri_aug_max_voxel_shift=0,
        mri_aug_intensity_scale=0.0,
        mri_aug_intensity_shift=0.0,
        mri_aug_gamma=0.0,
        mri_aug_noise_std=0.0,
        clinical_features=clinical_features,
        clinical_stats=state["clinical_stats"],
        ribbon_enabled=True,
        ribbon_cache_root=str(cfg.data.ribbon_cache_root),
        ribbon_num_depths=len(ribbon_depths),
        ribbon_depths=ribbon_depths,
    )
    train_reference = CortexWeaveROIClipDataset(
        split_csv=str(train_csv), train_mode=False, **kwargs
    )
    validation = CortexWeaveROIClipDataset(
        split_csv=str(valid_csv), train_mode=False, **kwargs
    )
    return train_reference, validation


def collect_tokens(model, loader, device, max_batches: int = 0):
    surfaces, ribbons, subjects = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            subjects.extend(cpu_batch["subject_id"])
            batch = move_to_device(cpu_batch, device)
            outputs = model(**model_inputs(batch))
            surfaces.append(outputs["cross_geometry_surface"].detach().cpu())
            ribbons.append(outputs["cross_geometry_ribbon"].detach().cpu())
    surface = F.normalize(torch.cat(surfaces, dim=0).float(), dim=-1)
    ribbon = F.normalize(torch.cat(ribbons, dim=0).float(), dim=-1)
    if surface.shape != ribbon.shape or surface.ndim != 3:
        raise AssertionError(f"Unexpected token shapes: {surface.shape}, {ribbon.shape}")
    return surface, ribbon, subjects


def subject_similarity_analysis(surface: torch.Tensor, ribbon: torch.Tensor):
    surface = surface.double()
    ribbon = ribbon.double()
    n_subjects, num_rois, _ = surface.shape
    if n_subjects < 2 or num_rois < 2:
        raise ValueError("Subject-specific analysis requires >=2 subjects and >=2 ROIs")
    same_subject_same_roi_sum = torch.einsum("bkc,bkc->", surface, ribbon)
    same_roi_all_sum = torch.einsum("kc,kc->", surface.sum(0), ribbon.sum(0))
    same_subject_all_sum = torch.einsum("bc,bc->", surface.sum(1), ribbon.sum(1))
    total_sum = torch.dot(surface.sum((0, 1)), ribbon.sum((0, 1)))
    different_subject_same_roi_sum = same_roi_all_sum - same_subject_same_roi_sum
    same_subject_different_roi_sum = same_subject_all_sum - same_subject_same_roi_sum
    different_subject_different_roi_sum = (
        total_sum - same_subject_all_sum - different_subject_same_roi_sum
    )
    values = {
        "same_subject_same_roi": float(same_subject_same_roi_sum / (n_subjects * num_rois)),
        "different_subject_same_roi": float(
            different_subject_same_roi_sum / (n_subjects * (n_subjects - 1) * num_rois)
        ),
        "same_subject_different_roi": float(
            same_subject_different_roi_sum / (n_subjects * num_rois * (num_rois - 1))
        ),
        "different_subject_different_roi": float(
            different_subject_different_roi_sum
            / (n_subjects * (n_subjects - 1) * num_rois * (num_rois - 1))
        ),
    }
    values["g_subject"] = values["same_subject_same_roi"] - values["different_subject_same_roi"]
    values["roi_identity_component"] = (
        values["different_subject_same_roi"] - values["different_subject_different_roi"]
    )
    values["correspondence_excess"] = (
        values["same_subject_same_roi"] - values["different_subject_different_roi"]
    )
    denominator = values["correspondence_excess"]
    values["roi_identity_share"] = (
        values["roi_identity_component"] / denominator if abs(denominator) > 1e-12 else float("nan")
    )
    values["subject_specific_share"] = (
        values["g_subject"] / denominator if abs(denominator) > 1e-12 else float("nan")
    )
    return values


def _directional_retrieval(similarity: torch.Tensor):
    n_subjects, num_rois, _ = similarity.shape
    ranking = similarity.argsort(dim=-1, descending=True)
    target = torch.arange(num_rois, device=similarity.device).view(1, num_rois, 1)
    rank = ranking.eq(target).float().argmax(dim=-1) + 1
    return {
        "acc1": float((rank == 1).float().mean()),
        "acc5": float((rank <= min(5, num_rois)).float().mean()),
        "mrr": float(rank.float().reciprocal().mean()),
        "top1": ranking[..., 0].detach().cpu().numpy(),
        "n_queries": n_subjects * num_rois,
    }


def retrieval_analysis(surface: torch.Tensor, ribbon: torch.Tensor, device: torch.device):
    surface = surface.to(device)
    ribbon = ribbon.to(device)
    similarity = torch.matmul(surface, ribbon.transpose(-1, -2))
    sr = _directional_retrieval(similarity)
    rs = _directional_retrieval(similarity.transpose(-1, -2))
    num_rois = similarity.shape[1]
    diagonal = similarity.diagonal(dim1=-2, dim2=-1).mean()
    eye = torch.eye(num_rois, device=device, dtype=torch.bool).unsqueeze(0)
    off_diagonal = similarity.masked_select(~eye).mean()
    result = {
        key: 0.5 * (float(sr[key]) + float(rs[key]))
        for key in ("acc1", "acc5", "mrr")
    }
    result.update(
        {
            "diagonal_similarity": float(diagonal),
            "off_diagonal_similarity": float(off_diagonal),
            "diagonal_gap": float(diagonal - off_diagonal),
            "surface_to_ribbon_top1": sr["top1"],
            "ribbon_to_surface_top1": rs["top1"],
        }
    )
    return result, similarity.detach()


def deterministic_derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("A subject derangement requires at least two subjects")
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(10000):
        permutation = rng.permutation(size)
        if np.all(permutation != identity):
            return permutation
    # Guaranteed deterministic fallback.
    shift = int(rng.integers(1, size))
    return np.roll(identity, shift)


def shuffled_retrieval(
    surface: torch.Tensor,
    ribbon: torch.Tensor,
    device: torch.device,
    num_permutations: int,
    seed: int,
):
    per_permutation = []
    for permutation_idx in range(num_permutations):
        permutation = deterministic_derangement(surface.shape[0], seed + permutation_idx)
        metrics, _ = retrieval_analysis(surface, ribbon[permutation], device)
        per_permutation.append({key: metrics[key] for key in ("acc1", "acc5", "mrr", "diagonal_gap")})
    return {
        key: float(np.mean([entry[key] for entry in per_permutation]))
        for key in per_permutation[0]
    } | {
        f"{key}_permutation_sd": float(np.std([entry[key] for entry in per_permutation], ddof=1))
        for key in per_permutation[0]
    }


def spatial_top3_neighbors(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce centroid geometry but enforce top-3 same-hemi neighbours."""

    num_rois = int(dataset.num_rois)
    per_hemi_vertices = int(dataset.num_vertices) // 2
    ico_level = infer_ico_level(per_hemi_vertices)
    hemi_coords, _ = get_ico_topology(ico_level)
    coords = np.concatenate([hemi_coords, hemi_coords], axis=0).astype(np.float64)
    coords /= np.linalg.norm(coords, axis=1, keepdims=True) + 1e-12
    masks = np.asarray(_load_reference_surface_roi_masks(dataset), dtype=np.float64)
    centroids = []
    for roi_idx in range(num_rois):
        weights = masks[roi_idx]
        center = (weights[:, None] * coords).sum(axis=0) / max(float(weights.sum()), 1e-12)
        center /= np.linalg.norm(center) + 1e-12
        centroids.append(center)
    centroids = np.stack(centroids)
    parts = [
        _roi_hemi_and_base(name, idx, num_rois)
        for idx, name in enumerate(dataset.atlas_region_names)
    ]
    hemis = np.asarray([part[0] for part in parts])
    distance = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    neighbors = np.zeros((num_rois, num_rois), dtype=np.uint8)
    for roi_idx in range(num_rois):
        candidates = np.flatnonzero((hemis == hemis[roi_idx]) & (np.arange(num_rois) != roi_idx))
        selected = candidates[np.argsort(distance[roi_idx, candidates])[:3]]
        neighbors[roi_idx, selected] = 1
    return neighbors, centroids


def compare_spatial_mesh(
    fold: int,
    region_names: Sequence[str],
    spatial: np.ndarray,
    anatomical: np.ndarray,
):
    rows = []
    for roi_idx, name in enumerate(region_names):
        spatial_set = set(np.flatnonzero(spatial[roi_idx]).tolist())
        mesh_set = set(np.flatnonzero(anatomical[roi_idx]).tolist())
        intersection = spatial_set & mesh_set
        union = spatial_set | mesh_set
        rows.append(
            {
                "fold": fold,
                "roi_index": roi_idx,
                "roi_name": name,
                "spatial_top3": ";".join(str(value) for value in sorted(spatial_set)),
                "mesh_neighbors": ";".join(str(value) for value in sorted(mesh_set)),
                "overlap_count": len(intersection),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "spatial_only": ";".join(str(value) for value in sorted(spatial_set - mesh_set)),
                "mesh_only": ";".join(str(value) for value in sorted(mesh_set - spatial_set)),
            }
        )
    return rows


MISMATCH_CATEGORIES = (
    "true_mesh_neighbor",
    "bilateral_homolog",
    "spatially_close_not_mesh_adjacent",
    "remote_roi",
)


def classify_mismatches(
    top1: np.ndarray,
    anatomical: np.ndarray,
    homology: np.ndarray,
    spatial: np.ndarray,
):
    counts = Counter()
    num_rois = top1.shape[1]
    for predicted_by_subject in top1:
        for target in range(num_rois):
            predicted = int(predicted_by_subject[target])
            if predicted == target:
                continue
            if anatomical[target, predicted]:
                counts["true_mesh_neighbor"] += 1
            elif homology[target, predicted]:
                counts["bilateral_homolog"] += 1
            elif spatial[target, predicted]:
                counts["spatially_close_not_mesh_adjacent"] += 1
            else:
                counts["remote_roi"] += 1
    total = sum(counts.values())
    return {
        "count": {name: int(counts[name]) for name in MISMATCH_CATEGORIES},
        "fraction": {name: float(counts[name] / total) if total else 0.0 for name in MISMATCH_CATEGORIES},
        "total_mismatches": int(total),
    }


def merge_mismatch_directions(sr, rs):
    counts = {name: sr["count"][name] + rs["count"][name] for name in MISMATCH_CATEGORIES}
    total = sum(counts.values())
    return {
        "count": counts,
        "fraction": {name: counts[name] / total if total else 0.0 for name in MISMATCH_CATEGORIES},
        "total_mismatches": total,
    }


def graph_metrics(similarity: torch.Tensor, anatomical: np.ndarray, homology: np.ndarray, temperature: float):
    anatomical = np.asarray(anatomical, dtype=np.uint8)
    graph_with_homology = np.maximum(anatomical, homology).astype(np.uint8)
    distance = shortest_path(graph_with_homology, directed=False, unweighted=True)
    if not np.isfinite(distance).all():
        raise ValueError("DKT mesh+homology graph is disconnected")
    mesh_distance = shortest_path(anatomical, directed=False, unweighted=True)
    distance_t = torch.as_tensor(distance, dtype=similarity.dtype, device=similarity.device)
    num_rois = similarity.shape[1]
    target = torch.arange(num_rois, device=similarity.device).view(1, num_rois)

    def direction(sim):
        probability = torch.softmax(sim / max(float(temperature), 1e-6), dim=-1)
        prediction = sim.argmax(dim=-1)
        tde = distance_t[target.expand_as(prediction), prediction].float().mean()
        etd = (probability * distance_t.unsqueeze(0)).sum(dim=-1).mean()
        pred_np = prediction.detach().cpu().numpy()
        mesh_top1 = mesh_distance[np.arange(num_rois)[None, :], pred_np]
        finite = np.isfinite(mesh_top1)
        within_hemi_tde = float(mesh_top1[finite].mean()) if finite.any() else float("nan")
        return {
            "tde_mesh_plus_homology": float(tde),
            "etd_mesh_plus_homology": float(etd),
            "mesh_only_top1_unreachable_fraction": float((~finite).mean()),
            "mesh_only_within_hemi_tde": within_hemi_tde,
        }

    sr = direction(similarity)
    rs = direction(similarity.transpose(-1, -2))
    return {
        "surface_to_ribbon": sr,
        "ribbon_to_surface": rs,
        "bidirectional": {key: 0.5 * (sr[key] + rs[key]) for key in sr},
    }


def gradient_conflict_diagnostic(model, loader, device, class_weights, temperature, max_batches):
    grouped_parameters: Dict[str, List[torch.nn.Parameter]] = {}
    for name, parameter in model.volume_encoder.named_parameters():
        stage = name.split(".", 1)[0]
        grouped_parameters.setdefault(stage, []).append(parameter)
    ordered_parameters = [parameter for values in grouped_parameters.values() for parameter in values]
    offsets = {}
    cursor = 0
    for stage, values in grouped_parameters.items():
        offsets[stage] = (cursor, cursor + len(values))
        cursor += len(values)
    observations = {stage: [] for stage in grouped_parameters}
    norm_observations = {stage: {"classification": [], "cross_geometry": []} for stage in grouped_parameters}
    model.eval()
    for batch_idx, cpu_batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(**model_inputs(batch))
        loss_cls = classification_loss(outputs["logits"], batch["label"], class_weights=class_weights)
        loss_cg = cross_geometry_correspondence_loss(
            outputs["cross_geometry_surface"], outputs["cross_geometry_ribbon"], temperature=temperature
        )
        grad_cls = torch.autograd.grad(
            loss_cls, ordered_parameters, retain_graph=True, allow_unused=True
        )
        grad_cg = torch.autograd.grad(
            loss_cg, ordered_parameters, retain_graph=False, allow_unused=True
        )
        for stage, (start, end) in offsets.items():
            dot = loss_cls.new_zeros(())
            norm_cls_sq = loss_cls.new_zeros(())
            norm_cg_sq = loss_cls.new_zeros(())
            for cls_grad, cg_grad in zip(grad_cls[start:end], grad_cg[start:end]):
                if cls_grad is not None:
                    norm_cls_sq = norm_cls_sq + cls_grad.detach().float().square().sum()
                if cg_grad is not None:
                    norm_cg_sq = norm_cg_sq + cg_grad.detach().float().square().sum()
                if cls_grad is not None and cg_grad is not None:
                    dot = dot + (cls_grad.detach().float() * cg_grad.detach().float()).sum()
            norm_cls = float(torch.sqrt(norm_cls_sq))
            norm_cg = float(torch.sqrt(norm_cg_sq))
            norm_observations[stage]["classification"].append(norm_cls)
            norm_observations[stage]["cross_geometry"].append(norm_cg)
            if norm_cls > 0.0 and norm_cg > 0.0:
                observations[stage].append(float(dot / (torch.sqrt(norm_cls_sq) * torch.sqrt(norm_cg_sq))))

    result = {}
    for stage in grouped_parameters:
        cosines = observations[stage]
        cg_norms = norm_observations[stage]["cross_geometry"]
        result[stage] = {
            "status": "measured" if cosines else "not_reached_by_cross_geometry_loss",
            "cosine_mean": float(np.mean(cosines)) if cosines else None,
            "cosine_sd": float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0 if cosines else None,
            "cosine_min": float(np.min(cosines)) if cosines else None,
            "cosine_max": float(np.max(cosines)) if cosines else None,
            "classification_grad_norm_mean": float(np.mean(norm_observations[stage]["classification"])),
            "cross_geometry_grad_norm_mean": float(np.mean(cg_norms)),
            "diagnostic_batches": len(norm_observations[stage]["classification"]),
        }
    return result


def mean_sd(folds: Sequence[Mapping[str, float]], key: str):
    values = np.asarray([float(fold[key]) for fold in folds], dtype=np.float64)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def format_mean_sd(summary, digits=5):
    return f"{summary['mean']:.{digits}f} ± {summary['sd']:.{digits}f}"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(audit: Mapping[str, object]) -> str:
    folds = audit["folds"]
    subject_keys = (
        "same_subject_same_roi",
        "different_subject_same_roi",
        "same_subject_different_roi",
        "different_subject_different_roi",
        "g_subject",
    )
    subject_labels = {
        "same_subject_same_roi": "same subject, same ROI",
        "different_subject_same_roi": "different subject, same ROI",
        "same_subject_different_roi": "same subject, different ROI",
        "different_subject_different_roi": "different subject, different ROI",
        "g_subject": "G_subject",
    }
    lines = [
        "# CortexBridge Phase 2.1.5 — Correspondence Audit",
        "",
        "Date: 2026-10-03",
        "",
        "## Scope and safeguards",
        "",
        "This is a frozen-checkpoint, validation-only audit. No model was trained, no optimizer was created, "
        "`backward()` was not called, topology-soft targets were not implemented, and no held-out test split was read. "
        "The optional gradient analysis used `torch.autograd.grad` on diagnostic validation batches without updating parameters.",
        "",
        "## 1. Subject-specific 2×2 similarity analysis",
        "",
        "All entries are cosine similarities after the learned projectors and L2 normalization. The aggregate is the "
        "mean ± sample SD over the three development folds.",
        "",
        "| Pair type | Fold 0 | Fold 1 | Fold 2 | Three-fold mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in subject_keys:
        values = [fold["subject_similarity"][key] for fold in folds]
        summary = mean_sd([fold["subject_similarity"] for fold in folds], key)
        lines.append(
            f"| {subject_labels[key]} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {format_mean_sd(summary)} |"
        )
    all_positive = all(fold["subject_similarity"]["g_subject"] > 0 for fold in folds)
    lines += [
        "",
        r"$G_{subject}=sim(S[b,i],R[b,i])-sim(S[b,i],R[b',i])$.",
        "",
        f"`G_subject` is {'positive on all folds' if all_positive else 'not positive on every fold'}, so the result "
        f"{'supports' if all_positive else 'does not consistently support'} a subject-specific increment beyond atlas ROI identity.",
        "",
        "For an interpretable effect decomposition, the ROI-identity component is "
        "`different-subject/same-ROI − different-subject/different-ROI`; the subject component is `G_subject`. "
        "Their sum is the total same-subject/same-ROI excess over the different-subject/different-ROI baseline.",
        "",
        "| Fold | ROI-identity component | Subject component | ROI-identity share | Subject share |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold in folds:
        values = fold["subject_similarity"]
        lines.append(
            f"| {fold['fold']} | {values['roi_identity_component']:.5f} | {values['g_subject']:.5f} | "
            f"{values['roi_identity_share']:.2%} | {values['subject_specific_share']:.2%} |"
        )
    roi_share = mean_sd([fold["subject_similarity"] for fold in folds], "roi_identity_share")
    subject_share = mean_sd([fold["subject_similarity"] for fold in folds], "subject_specific_share")
    lines += [
        "",
        f"Across folds, the effect-share means are {roi_share['mean']:.2%} for ROI identity and "
        f"{subject_share['mean']:.2%} for the subject-specific increment. These are descriptive contrast shares, "
        "not causal variance explained.",
        "",
        "## 2. Normal versus subject-shuffled retrieval",
        "",
        f"Ribbon tokens were shuffled with {audit['num_shuffle_permutations']} deterministic subject derangements per fold; "
        "every shuffled pair therefore uses a different subject. Shuffled values are means across permutations.",
        "",
        "| Fold | Condition | Acc@1 | Acc@5 | MRR | Diagonal gap |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for fold in folds:
        for label, key in (("normal", "normal_retrieval"), ("subject-shuffled", "shuffled_retrieval")):
            values = fold[key]
            lines.append(
                f"| {fold['fold']} | {label} | {values['acc1']:.5f} | {values['acc5']:.5f} | "
                f"{values['mrr']:.5f} | {values['diagonal_gap']:.5f} |"
            )
    normal_acc = mean_sd([fold["normal_retrieval"] for fold in folds], "acc1")
    shuffle_acc = mean_sd([fold["shuffled_retrieval"] for fold in folds], "acc1")
    lines += [
        "",
        f"Across folds, normal Acc@1 is {format_mean_sd(normal_acc)} and subject-shuffled Acc@1 is "
        f"{format_mean_sd(shuffle_acc)}. The shuffle test preserves ROI identity while removing matched-subject information.",
        "",
        "## 3. True DKT mesh topology",
        "",
        "The graph was constructed from fsaverage triangular mesh edges. Because `mris_ca_label` could not run on the "
        "compute-node system libraries, FreeSurfer's bundled `bert` DKTatlas40 annotation was transferred to fsaverage "
        "by nearest-neighbour matching in registered `sphere.reg`/fsaverage spherical coordinates. Only the 31 retained "
        "project DKT parcels per hemisphere were valid projection sources; non-cortex vertices were excluded.",
        "",
    ]
    topo = audit["topology"]
    lines += [
        f"- ROI count: `{topo['num_rois']}` (31 left + 31 right)",
        f"- Undirected anatomical edges: `{topo['undirected_edges']}`",
        f"- Degree min / median / mean / max: `{topo['degree_min']}` / `{topo['degree_median']:.1f}` / "
        f"`{topo['degree_mean']:.2f}` / `{topo['degree_max']}`",
        f"- Shared boundary mesh edges: `{topo['shared_boundary_edges']}`",
        f"- Symmetric raw adjacency and boundary counts: `{topo['raw_symmetric']}`",
        f"- Zero diagonal: `{topo['zero_diagonal']}`",
        f"- Invalid cortical DKT labels: `{topo['invalid_dkt_labels']}`",
        f"- Exact one-to-one left/right homologues: `{topo['exact_homology']}`",
        "",
        "The exported `true_dkt_topology.npz` contains `A_anat`, `B_boundary`, row-normalized `A_boundary`, "
        "and the separate `A_hom` matrix in exact model ROI order.",
        "",
        "## 4. Spatial top-3 versus mesh neighbours",
        "",
        "| Fold | Mean overlap count (of 3) | Mean Jaccard | ROIs with zero overlap |",
        "|---:|---:|---:|---:|",
    ]
    for fold in folds:
        comparison = fold["spatial_mesh_comparison"]
        lines.append(
            f"| {fold['fold']} | {comparison['mean_overlap']:.3f} | {comparison['mean_jaccard']:.3f} | "
            f"{comparison['zero_overlap_rois']} / {topo['num_rois']} |"
        )
    lines += [
        "",
        "Per-ROI mismatches are in `roi_topology_comparison.csv`. The Phase 2.1 metric is therefore renamed "
        "**spatially tolerant retrieval accuracy**; it is not treated as anatomical topology accuracy.",
        "",
        "## 5. Top-1 mismatch decomposition",
        "",
        "Categories are applied sequentially: mesh neighbour → bilateral homologue → spatial top-3 but not mesh-adjacent → remote.",
        "",
        "| Fold | Mesh neighbour | Homologue | Spatial-only | Remote | Mismatches |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in folds:
        mismatch = fold["mismatch_bidirectional"]
        frac = mismatch["fraction"]
        lines.append(
            f"| {fold['fold']} | {frac['true_mesh_neighbor']:.3%} | {frac['bilateral_homolog']:.3%} | "
            f"{frac['spatially_close_not_mesh_adjacent']:.3%} | {frac['remote_roi']:.3%} | "
            f"{mismatch['total_mismatches']} |"
        )
    aggregate = audit["aggregate_mismatch"]
    frac = aggregate["fraction"]
    lines.append(
        f"| All | {frac['true_mesh_neighbor']:.3%} | {frac['bilateral_homolog']:.3%} | "
        f"{frac['spatially_close_not_mesh_adjacent']:.3%} | {frac['remote_roi']:.3%} | "
        f"{aggregate['total_mismatches']} |"
    )
    lines += [
        "",
        "## 6. True graph-distance metrics",
        "",
        "The primary finite graph is `A_anat ∪ A_hom`: mesh edges are anatomical within a hemisphere and the separately "
        "validated homology edges connect hemispheres. Pure `A_anat` is disconnected across hemispheres, so its "
        "cross-hemisphere top-1 unreachable fraction and within-hemisphere TDE are reported as diagnostics instead of "
        "silently replacing infinity with an arbitrary number.",
        "",
        "| Fold | Direction | TDE | ETD | Mesh-only unreachable | Mesh-only within-hemi TDE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for fold in folds:
        for direction, label in (("surface_to_ribbon", "surface→ribbon"), ("ribbon_to_surface", "ribbon→surface"), ("bidirectional", "bidirectional")):
            graph = fold["graph_metrics"][direction]
            lines.append(
                f"| {fold['fold']} | {label} | {graph['tde_mesh_plus_homology']:.5f} | "
                f"{graph['etd_mesh_plus_homology']:.5f} | {graph['mesh_only_top1_unreachable_fraction']:.3%} | "
                f"{graph['mesh_only_within_hemi_tde']:.5f} |"
            )
    lines += [
        "",
        "## 7. Gradient-conflict diagnostic",
        "",
        "No optimizer step was performed. Cosines compare CE classification gradients with unweighted cross-geometry "
        "gradients on the shared MRI encoder.",
        "",
        "| Fold | Stage | Status | Gradient cosine mean ± SD | Mean ‖g_cls‖ | Mean ‖g_cg‖ |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for fold in folds:
        for stage, values in fold["gradient_conflict"].items():
            cosine = (
                f"{values['cosine_mean']:.5f} ± {values['cosine_sd']:.5f}"
                if values["cosine_mean"] is not None
                else "n/a"
            )
            lines.append(
                f"| {fold['fold']} | {stage} | {values['status']} | {cosine} | "
                f"{values['classification_grad_norm_mean']:.5g} | {values['cross_geometry_grad_norm_mean']:.5g} |"
            )

    random_neighbor_fraction = topo["degree_mean"] / (topo["num_rois"] - 1)
    anatomy_enriched = frac["true_mesh_neighbor"] > random_neighbor_fraction
    roi_identity = normal_acc["mean"] > 1.0 / topo["num_rois"]
    subject_specific = all_positive and normal_acc["mean"] > shuffle_acc["mean"]
    subject_relative_drop = (
        (normal_acc["mean"] - shuffle_acc["mean"]) / normal_acc["mean"]
        if normal_acc["mean"] > 0.0
        else 0.0
    )
    lines += [
        "",
        "The high-resolution ribbon anchor branches from the stem output. Consequently, `L_CG` reaches the MRI stem "
        "but not later SFCN blocks; later-stage `not_reached` entries are architectural, not zero-valued conflict estimates.",
        "",
        "## 8. Conclusion",
        "",
        f"- ROI identity correspondence: **{'supported' if roi_identity else 'not supported'}**. Normal Acc@1 is "
        f"{normal_acc['mean']:.4f} versus the 62-way random value {1.0 / topo['num_rois']:.4f}.",
        f"- Subject-specific correspondence: **{'supported, but weak relative to ROI identity' if subject_specific else 'not established'}**. "
        f"`G_subject` is {'positive on all folds' if all_positive else 'not consistently positive'}, and subject shuffling "
        f"changes mean Acc@1 from {normal_acc['mean']:.4f} to {shuffle_acc['mean']:.4f} "
        f"(relative drop {subject_relative_drop:.2%}). The effect decomposition should be used to judge its magnitude.",
        f"- Anatomical neighbourhood evidence: **{'present' if anatomy_enriched else 'not clearly enriched'}**. "
        f"Mesh-neighbour errors account for {frac['true_mesh_neighbor']:.2%} of mismatches versus an approximate "
        f"uniform wrong-ROI expectation of {random_neighbor_fraction:.2%} from mean graph degree.",
        "- Phase 2.2 was not implemented. These findings only determine whether a future topology-soft experiment has "
        "real mesh-based anatomical motivation; they do not show that topology-soft supervision will improve classification.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `audit_metrics.json`: all fold-level and aggregate numerical results",
        "- `true_dkt_topology.npz`: mesh adjacency, boundary counts/strength, homology, names, and degrees",
        "- `roi_topology_comparison.csv`: per-fold, per-ROI spatial/mesh neighbourhood mismatches",
        "- `mismatch_decomposition.csv`: direction-specific and bidirectional mismatch counts/fractions",
        "",
        "`held_out_test_accessed: false`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=3, default=[str(path) for path in DEFAULT_CHECKPOINTS])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shuffle-permutations", type=int, default=20)
    parser.add_argument("--gradient-batches", type=int, default=4)
    parser.add_argument("--max-validation-batches", type=int, default=0, help="Audit smoke-test limit; 0 means all")
    parser.add_argument("--fsaverage-dir", default=str(DEFAULT_FS_HOME / "subjects/fsaverage"))
    parser.add_argument("--dkt-source-subject-dir", default=str(DEFAULT_FS_HOME / "subjects/bert"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    # Build topology first and reorder it later using the first dataset's exact ROI names.
    topology = build_fsaverage_dkt_topology(args.fsaverage_dir, args.dkt_source_subject_dir)
    fold_results = []
    topology_rows = []
    mismatch_rows = []
    ordered_topology = None

    for checkpoint_path in args.checkpoints:
        print(f"[audit] loading {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fold = int(state["fold"])
        cfg = as_config(state["config"])
        if not bool(cfg.model.cross_geometry.enabled):
            raise ValueError(f"Checkpoint fold {fold} has no cross-geometry branch")
        if bool(getattr(cfg.runtime, "evaluate_test", False)):
            raise ValueError("Audit requires a development-only checkpoint configuration")

        train_reference, validation = build_audit_datasets(cfg, fold, state)
        if ordered_topology is None:
            ordered_topology = reorder_topology(topology, validation.atlas_region_names)
        elif tuple(ordered_topology["region_names"]) != tuple(validation.atlas_region_names):
            raise ValueError("ROI order differs between folds")

        loader = DataLoader(
            validation,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(args.num_workers),
            collate_fn=roiclip_collate_fn,
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.num_workers) > 0,
        )
        model = build_model(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        surface, ribbon, subject_ids = collect_tokens(
            model, loader, device, max_batches=int(args.max_validation_batches)
        )
        normal, similarity = retrieval_analysis(surface, ribbon, device)
        shuffled = shuffled_retrieval(
            surface,
            ribbon,
            device,
            num_permutations=int(args.shuffle_permutations),
            seed=210500 + fold * 1000,
        )
        subject_similarity = subject_similarity_analysis(surface, ribbon)
        spatial, _ = spatial_top3_neighbors(train_reference)
        comparison_rows = compare_spatial_mesh(
            fold,
            ordered_topology["region_names"],
            spatial,
            ordered_topology["adjacency"],
        )
        topology_rows.extend(comparison_rows)
        spatial_summary = {
            "mean_overlap": float(np.mean([row["overlap_count"] for row in comparison_rows])),
            "mean_jaccard": float(np.mean([row["jaccard"] for row in comparison_rows])),
            "zero_overlap_rois": int(sum(row["overlap_count"] == 0 for row in comparison_rows)),
        }
        mismatch_sr = classify_mismatches(
            normal["surface_to_ribbon_top1"],
            ordered_topology["adjacency"], ordered_topology["homology"], spatial,
        )
        mismatch_rs = classify_mismatches(
            normal["ribbon_to_surface_top1"],
            ordered_topology["adjacency"], ordered_topology["homology"], spatial,
        )
        mismatch_bi = merge_mismatch_directions(mismatch_sr, mismatch_rs)
        for direction, mismatch in (
            ("surface_to_ribbon", mismatch_sr),
            ("ribbon_to_surface", mismatch_rs),
            ("bidirectional", mismatch_bi),
        ):
            row = {"fold": fold, "direction": direction, "total_mismatches": mismatch["total_mismatches"]}
            for category in MISMATCH_CATEGORIES:
                row[f"{category}_count"] = mismatch["count"][category]
                row[f"{category}_fraction"] = mismatch["fraction"][category]
            mismatch_rows.append(row)

        graph = graph_metrics(
            similarity,
            ordered_topology["adjacency"],
            ordered_topology["homology"],
            float(cfg.model.cross_geometry.temperature),
        )
        class_weights = state.get("class_weights")
        gradients = gradient_conflict_diagnostic(
            model,
            loader,
            device,
            class_weights=class_weights,
            temperature=float(cfg.model.cross_geometry.temperature),
            max_batches=int(args.gradient_batches),
        ) if int(args.gradient_batches) > 0 else {}

        normal_clean = {key: value for key, value in normal.items() if not key.endswith("_top1")}
        fold_result = {
            "fold": fold,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": int(state["epoch"]),
            "validation_subjects": len(subject_ids),
            "subject_similarity": subject_similarity,
            "normal_retrieval": normal_clean,
            "shuffled_retrieval": shuffled,
            "spatial_mesh_comparison": spatial_summary,
            "mismatch_surface_to_ribbon": mismatch_sr,
            "mismatch_ribbon_to_surface": mismatch_rs,
            "mismatch_bidirectional": mismatch_bi,
            "graph_metrics": graph,
            "gradient_conflict": gradients,
        }
        fold_results.append(fold_result)
        print(
            f"[audit] fold={fold} n={len(subject_ids)} acc1={normal_clean['acc1']:.4f} "
            f"shuffle_acc1={shuffled['acc1']:.4f} G_subject={subject_similarity['g_subject']:.4f}",
            flush=True,
        )
        del model, loader, surface, ribbon, similarity
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_results.sort(key=lambda result: result["fold"])
    if [result["fold"] for result in fold_results] != [0, 1, 2]:
        raise ValueError("Full audit requires development folds 0, 1, and 2")
    degree = np.asarray(ordered_topology["degree"])
    topology_summary = {
        "num_rois": int(degree.size),
        "undirected_edges": int(np.asarray(ordered_topology["adjacency"]).sum() // 2),
        "shared_boundary_edges": int(np.asarray(ordered_topology["boundary_edge_count"]).sum() // 2),
        "degree_min": int(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_mean": float(degree.mean()),
        "degree_max": int(degree.max()),
        "degree_per_roi": degree.tolist(),
        "raw_symmetric": bool(
            np.array_equal(ordered_topology["adjacency"], np.asarray(ordered_topology["adjacency"]).T)
            and np.array_equal(ordered_topology["boundary_edge_count"], np.asarray(ordered_topology["boundary_edge_count"]).T)
        ),
        "zero_diagonal": bool(
            not np.any(np.diag(ordered_topology["adjacency"]))
            and not np.any(np.diag(ordered_topology["boundary_edge_count"]))
        ),
        "invalid_dkt_labels": int(ordered_topology["validation"]["invalid_dkt_labels"]),
        "exact_homology": bool(ordered_topology["validation"]["exact_one_to_one_homology"]),
        "hemisphere_stats": ordered_topology["hemisphere_stats"],
        "projection_source": str(Path(args.dkt_source_subject_dir).resolve()),
        "template": str(Path(args.fsaverage_dir).resolve()),
    }
    aggregate_counts = {
        category: sum(fold["mismatch_bidirectional"]["count"][category] for fold in fold_results)
        for category in MISMATCH_CATEGORIES
    }
    aggregate_total = sum(aggregate_counts.values())
    audit = {
        "phase": "2.1.5",
        "audit_only": True,
        "new_model_training": False,
        "optimizer_created": False,
        "backward_called": False,
        "topology_soft_implemented": False,
        "held_out_test_accessed": False,
        "num_shuffle_permutations": int(args.shuffle_permutations),
        "gradient_diagnostic_batches_per_fold": int(args.gradient_batches),
        "topology": topology_summary,
        "folds": fold_results,
        "aggregate_mismatch": {
            "count": aggregate_counts,
            "fraction": {
                category: aggregate_counts[category] / aggregate_total if aggregate_total else 0.0
                for category in MISMATCH_CATEGORIES
            },
            "total_mismatches": aggregate_total,
        },
    }

    np.savez_compressed(
        output_dir / "true_dkt_topology.npz",
        A_anat=np.asarray(ordered_topology["adjacency"], dtype=np.uint8),
        B_boundary=np.asarray(ordered_topology["boundary_edge_count"], dtype=np.int64),
        A_boundary=np.asarray(ordered_topology["boundary_strength"], dtype=np.float32),
        A_hom=np.asarray(ordered_topology["homology"], dtype=np.uint8),
        homolog_index=np.asarray(ordered_topology["homolog_index"], dtype=np.int16),
        degree=np.asarray(ordered_topology["degree"], dtype=np.int16),
        region_names=np.asarray(ordered_topology["region_names"]),
    )
    write_csv(output_dir / "roi_topology_comparison.csv", topology_rows)
    write_csv(output_dir / "mismatch_decomposition.csv", mismatch_rows)
    with (output_dir / "audit_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(audit), handle, indent=2, ensure_ascii=False, allow_nan=False)
    report = render_report(audit)
    (output_dir / "PHASE2_1_5_CORRESPONDENCE_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"[audit] wrote {output_dir / 'PHASE2_1_5_CORRESPONDENCE_AUDIT.md'}", flush=True)


if __name__ == "__main__":
    main()
