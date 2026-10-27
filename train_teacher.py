import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import build_dataloaders
from data.ico_utils import get_ico_topology, infer_ico_level
from losses import (
    anatomical_transport_roi_clip_loss,
    classification_loss,
    cross_geometry_correspondence_loss,
    cross_geometry_retrieval_metrics,
    distance_weighted_roi_clip_loss,
    prototype_loss,
    relation_consistency_loss,
    roi_clip_loss,
)
from metrics import AverageMeter, compute_classification_metrics, format_classification_metrics
from models.teacher import (
    CortexWeaveROIClipTeacher,
    MRIClinicalCLIPBaseline,
    SurfaceOnlyTransformerBaseline,
    SurfaceVolumeConcatBaseline,
)
from options import Option
from utils import (
    CheckpointIO,
    Logger,
    count_parameters,
    format_metric_summary,
    format_duration,
    move_to_device,
    now,
    save_component_weights,
    seed_everything,
    summarize_metric_results,
)


def infer_num_classes(task):
    return 3 if str(task).upper() == "COG" else 2


def compute_class_weights(dataset, num_classes, mode="sqrt_balanced"):
    labels = [int(sample["label"]) for sample in dataset.samples if int(sample["label"]) >= 0]
    if mode == "none" or not labels:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts)
    valid = counts > 0
    weights[valid] = float(len(labels)) / (float(valid.sum()) * counts[valid])
    if mode == "sqrt_balanced":
        weights[valid] = np.sqrt(weights[valid])
    if valid.any():
        weights[valid] = weights[valid] / weights[valid].mean()
    return weights.tolist()


def auxiliary_loss_scale_for_epoch(epoch, ce_warmup_epochs):
    return 0.0 if int(epoch) < max(0, int(ce_warmup_epochs)) else 1.0


def _roi_hemi_and_base(region_name, index, num_rois):
    name = str(region_name or "").lower()
    for prefix, hemi in [("lh_", "lh"), ("left_", "lh"), ("l_", "lh"), ("rh_", "rh"), ("right_", "rh"), ("r_", "rh")]:
        if name.startswith(prefix):
            return hemi, name[len(prefix):]
    return ("lh" if int(index) < int(num_rois) // 2 else "rh"), name or f"roi_{int(index)}"


def _load_reference_surface_roi_masks(dataset):
    if getattr(dataset, "preprocess_cache_root", "") and getattr(dataset, "samples", None):
        try:
            cached = dataset._load_cached_inputs(dataset.samples[0])
            if cached is not None:
                return cached["surface_roi_masks"]
        except Exception:
            pass
    sample = dataset[0]
    return sample["surface_roi_masks"].detach().cpu().numpy()


def build_roi_clip_affinity(
    dataset,
    cfg,
    logger=None,
    force=False,
    topk_override=None,
    log_label="Teacher ROI-CLIP alignment",
):
    mode = str(getattr(cfg.training, "roi_clip_alignment_mode", "hard")).lower()
    if mode in {"hard", "identity", "onehot"} and not force:
        return None

    num_rois = int(cfg.model.num_rois)
    num_vertices = int(cfg.data.num_vertices)
    per_hemi = max(1, num_vertices // 2)
    ico_level = infer_ico_level(per_hemi)
    hemi_coords, _ = get_ico_topology(ico_level)
    hemi_coords = hemi_coords.astype(np.float32)
    coords = np.concatenate([hemi_coords, hemi_coords], axis=0)
    coords = coords / (np.linalg.norm(coords, axis=1, keepdims=True) + 1e-6)

    roi_masks = np.asarray(_load_reference_surface_roi_masks(dataset), dtype=np.float32)
    if roi_masks.shape[0] != num_rois or roi_masks.shape[1] != coords.shape[0]:
        raise ValueError(
            f"Cannot build ROI-CLIP affinity: masks={roi_masks.shape}, expected=({num_rois}, {coords.shape[0]})"
        )

    centroids = []
    for roi_idx in range(num_rois):
        weights = roi_masks[roi_idx]
        denom = float(weights.sum())
        center = coords.mean(axis=0) if denom <= 1e-6 else (weights[:, None] * coords).sum(axis=0) / denom
        center = center / (np.linalg.norm(center) + 1e-6)
        centroids.append(center.astype(np.float32))
    centroids = np.stack(centroids, axis=0)

    region_names = list(getattr(dataset, "atlas_region_names", []))
    if len(region_names) != num_rois:
        region_names = [f"roi_{idx:02d}" for idx in range(num_rois)]
    parts = [_roi_hemi_and_base(name, idx, num_rois) for idx, name in enumerate(region_names)]
    hemis = np.asarray([part[0] for part in parts])
    bases = np.asarray([part[1] for part in parts])
    same_hemi = hemis[:, None] == hemis[None, :]
    homology = (hemis[:, None] != hemis[None, :]) & (bases[:, None] == bases[None, :])

    distance = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    offdiag = ~np.eye(num_rois, dtype=bool)
    within_distances = distance[same_hemi & offdiag]
    sigma = float(getattr(cfg.training, "roi_clip_distance_sigma", 0.0))
    if sigma <= 0.0:
        sigma = float(np.median(within_distances[within_distances > 1e-6])) if np.any(within_distances > 1e-6) else 1.0
    sigma = max(sigma, 1e-6)

    affinity = np.exp(-(distance ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    affinity[~same_hemi] = 0.0
    np.fill_diagonal(affinity, 0.0)

    homology_weight = max(0.0, float(getattr(cfg.training, "roi_clip_homology_weight", 1.0)))
    affinity = affinity + homology_weight * homology.astype(np.float32)
    np.fill_diagonal(affinity, 0.0)

    topk = (
        int(topk_override)
        if topk_override is not None
        else int(getattr(cfg.training, "roi_clip_affinity_topk", 0))
    )
    if topk > 0:
        sparse = np.zeros_like(affinity, dtype=np.float32)
        for roi_idx in range(num_rois):
            candidates = np.argsort(affinity[roi_idx])[::-1]
            selected = [idx for idx in candidates if affinity[roi_idx, idx] > 0.0][:topk]
            if selected:
                sparse[roi_idx, selected] = affinity[roi_idx, selected]
        affinity = np.maximum(sparse, sparse.T)

    if logger is not None:
        logger.print_message(
            f"{log_label} "
            f"mode={mode} sigma={sigma:.4f} homology_weight={homology_weight:.3f} "
            f"topk={topk} nonzero_edges={int(np.count_nonzero(affinity))}"
        )
    return torch.from_numpy(affinity.astype(np.float32))


def load_roi_graph_topology(cfg, force=False):
    graph_cfg = getattr(cfg.model, "roi_graph", {})
    if not bool(getattr(graph_cfg, "enabled", False)) and not bool(force):
        return {}
    topology_path = os.path.abspath(os.path.expanduser(str(getattr(graph_cfg, "topology_path", ""))))
    if not topology_path or not os.path.isfile(topology_path):
        raise FileNotFoundError(f"ROI graph topology archive not found: {topology_path}")
    required = {"A_anat", "A_boundary", "A_hom"}
    with np.load(topology_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"ROI graph topology archive is missing: {missing}")
        anatomy = np.asarray(archive["A_anat"], dtype=np.float32)
        boundary = np.asarray(archive["A_boundary"], dtype=np.float32)
        homology = np.asarray(archive["A_hom"], dtype=np.float32)
    num_rois = int(cfg.model.num_rois)
    expected = (num_rois, num_rois)
    for name, matrix in [("A_anat", anatomy), ("A_boundary", boundary), ("A_hom", homology)]:
        if matrix.shape != expected:
            raise ValueError(f"{name} has shape {matrix.shape}; expected {expected}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        if not np.allclose(np.diag(matrix), 0.0):
            raise ValueError(f"{name} must have a zero diagonal")
    if not np.array_equal(anatomy, anatomy.T):
        raise ValueError("A_anat must be symmetric")
    if not np.array_equal(homology, homology.T):
        raise ValueError("A_hom must be symmetric")

    bias_boundary = boundary.copy()
    bias_homology = homology.copy()
    if bool(getattr(graph_cfg, "shuffle_topology_bias", False)):
        rng = np.random.default_rng(int(getattr(graph_cfg, "topology_shuffle_seed", 3407)))
        permutation = rng.permutation(num_rois)
        bias_boundary = bias_boundary[np.ix_(permutation, permutation)]
        bias_homology = bias_homology[np.ix_(permutation, permutation)]
    return {
        "anatomy_adjacency": torch.from_numpy(anatomy),
        "boundary_strength": torch.from_numpy(bias_boundary),
        "evaluation_boundary_strength": torch.from_numpy(boundary),
        "homology": torch.from_numpy(bias_homology),
        "evaluation_homology": torch.from_numpy(homology),
        "path": topology_path,
    }


def build_model(cfg):
    num_classes = int(getattr(cfg.model, "num_classes", infer_num_classes(cfg.data.task)))
    use_clinical = bool(getattr(cfg.model, "use_clinical", False))
    method = str(getattr(cfg.model, "method", "")).lower()
    volume_pooling_cfg = getattr(cfg.model, "volume_pooling", {})
    ribbon_cfg = getattr(cfg.model, "ribbon", {})
    cross_geometry_cfg = getattr(cfg.model, "cross_geometry", {})
    roi_graph_cfg = getattr(cfg.model, "roi_graph", {})
    roi_graph_topology = load_roi_graph_topology(cfg)
    if "surface-only-transformer" in method or "surface_only_transformer" in method:
        return SurfaceOnlyTransformerBaseline(
            surface_in_channels=int(cfg.model.surface_in_channels),
            feature_dim=int(cfg.model.feature_dim),
            proj_dim=int(cfg.model.proj_dim),
            num_rois=int(cfg.model.num_rois),
            num_classes=num_classes,
            num_vertices=int(cfg.data.num_vertices),
            surface_hidden_dim=int(cfg.model.surface_hidden_dim),
            dropout=float(cfg.model.dropout),
            surface_refiner=str(getattr(cfg.model, "surface_refiner", "none")),
            surface_refiner_hidden_dim=int(getattr(cfg.model, "surface_refiner_hidden_dim", 0)),
            surface_refiner_dropout=float(getattr(cfg.model, "surface_refiner_dropout", 0.0)),
            surface_refiner_residual_init=float(getattr(cfg.model, "surface_refiner_residual_init", 0.1)),
            num_heads=int(getattr(cfg.model, "clinical_num_heads", 4)),
            num_layers=int(getattr(cfg.model, "clinical_num_layers", 2)),
        )
    if "mri-clinical-clip" in method or "mri_clinical_clip" in method:
        return MRIClinicalCLIPBaseline(
            feature_dim=int(cfg.model.feature_dim),
            proj_dim=int(cfg.model.proj_dim),
            num_classes=num_classes,
            base_channels=int(cfg.model.base_channels),
            dropout=float(cfg.model.dropout),
            volume_norm=str(cfg.model.volume_norm),
            clinical_input_dim=int(getattr(cfg.model, "clinical_input_dim", 0)),
            clinical_hidden_dim=int(getattr(cfg.model, "clinical_hidden_dim", 256)),
            clinical_num_heads=int(getattr(cfg.model, "clinical_num_heads", 4)),
            clinical_num_layers=int(getattr(cfg.model, "clinical_num_layers", 2)),
        )
    if "surface-volume-concat" in method or "surface_volume_concat" in method:
        return SurfaceVolumeConcatBaseline(
            surface_in_channels=int(cfg.model.surface_in_channels),
            feature_dim=int(cfg.model.feature_dim),
            num_classes=num_classes,
            num_vertices=int(cfg.data.num_vertices),
            base_channels=int(cfg.model.base_channels),
            surface_hidden_dim=int(cfg.model.surface_hidden_dim),
            dropout=float(cfg.model.dropout),
            volume_norm=str(cfg.model.volume_norm),
            use_clinical=use_clinical,
            clinical_input_dim=int(getattr(cfg.model, "clinical_input_dim", 0)),
            clinical_hidden_dim=int(getattr(cfg.model, "clinical_hidden_dim", 256)),
            surface_refiner=str(getattr(cfg.model, "surface_refiner", "none")),
            surface_refiner_hidden_dim=int(getattr(cfg.model, "surface_refiner_hidden_dim", 0)),
            surface_refiner_dropout=float(getattr(cfg.model, "surface_refiner_dropout", 0.0)),
            surface_refiner_residual_init=float(getattr(cfg.model, "surface_refiner_residual_init", 0.1)),
            num_rois=int(cfg.model.num_rois),
        )
    return CortexWeaveROIClipTeacher(
        surface_in_channels=int(cfg.model.surface_in_channels),
        feature_dim=int(cfg.model.feature_dim),
        proj_dim=int(cfg.model.proj_dim),
        num_rois=int(cfg.model.num_rois),
        num_classes=num_classes,
        num_vertices=int(cfg.data.num_vertices),
        base_channels=int(cfg.model.base_channels),
        surface_hidden_dim=int(cfg.model.surface_hidden_dim),
        dropout=float(cfg.model.dropout),
        prototype_temperature=float(cfg.temperature.prototype),
        roi_residual_init=float(cfg.model.roi_residual_init),
        use_roi_branch=bool(cfg.model.use_roi_branch),
        volume_norm=str(cfg.model.volume_norm),
        use_clinical=use_clinical,
        clinical_input_dim=int(getattr(cfg.model, "clinical_input_dim", 0)),
        clinical_fusion_mode=str(getattr(cfg.model, "clinical_fusion_mode", "gated")),
        clinical_hidden_dim=int(getattr(cfg.model, "clinical_hidden_dim", 256)),
        clinical_dropout=float(getattr(cfg.model, "clinical_dropout", 0.1)),
        clinical_num_heads=int(getattr(cfg.model, "clinical_num_heads", 4)),
        clinical_num_layers=int(getattr(cfg.model, "clinical_num_layers", 2)),
        surface_refiner=str(getattr(cfg.model, "surface_refiner", "none")),
        surface_refiner_hidden_dim=int(getattr(cfg.model, "surface_refiner_hidden_dim", 0)),
        surface_refiner_dropout=float(getattr(cfg.model, "surface_refiner_dropout", 0.0)),
        surface_refiner_residual_init=float(getattr(cfg.model, "surface_refiner_residual_init", 0.1)),
        volume_pooling_mode=str(getattr(volume_pooling_cfg, "mode", "legacy_mask")),
        ribbon_depth_attention=bool(getattr(ribbon_cfg, "depth_attention", True)),
        ribbon_feature_stage=str(getattr(ribbon_cfg, "feature_stage", "roi")),
        ribbon_gate_init=float(getattr(ribbon_cfg, "gate_init", 0.1)),
        cross_geometry_enabled=bool(getattr(cross_geometry_cfg, "enabled", False)),
        cross_geometry_dim=int(getattr(cross_geometry_cfg, "projection_dim", cfg.model.proj_dim)),
        cross_geometry_feature_stage=str(getattr(cross_geometry_cfg, "feature_stage", "high_res")),
        roi_graph_enabled=bool(getattr(roi_graph_cfg, "enabled", False)),
        roi_graph_num_layers=int(getattr(roi_graph_cfg, "num_layers", 1)),
        roi_graph_num_heads=int(getattr(roi_graph_cfg, "num_heads", 4)),
        roi_graph_dropout=float(getattr(roi_graph_cfg, "dropout", 0.1)),
        roi_graph_residual_gate_init=float(getattr(roi_graph_cfg, "residual_gate_init", 0.1)),
        roi_graph_use_boundary_strength=bool(
            getattr(roi_graph_cfg, "use_boundary_strength", False)
        ),
        roi_graph_use_homology=bool(getattr(roi_graph_cfg, "use_homology", False)),
        roi_graph_anatomy_adjacency=roi_graph_topology.get("anatomy_adjacency"),
        roi_graph_boundary_strength=roi_graph_topology.get("boundary_strength"),
        roi_graph_evaluation_boundary_strength=roi_graph_topology.get(
            "evaluation_boundary_strength"
        ),
        roi_graph_homology=roi_graph_topology.get("homology"),
        roi_graph_evaluation_homology=roi_graph_topology.get("evaluation_homology"),
    )


def model_requires_teacher_inputs(cfg):
    method = str(getattr(cfg.model, "method", "")).lower()
    if "surface-only-transformer" in method or "surface_only_transformer" in method:
        return True
    if "mri-clinical-clip" in method or "mri_clinical_clip" in method:
        return False
    return bool(cfg.model.use_roi_branch)


def selection_score(metrics, name):
    if name == "acc":
        return float(metrics["acc"])
    if name == "f1":
        return float(metrics.get("f1", metrics["f1_macro"]))
    return float(metrics["mcc"])


def teacher_roi_clip_loss(z_surface, z_volume, cfg, roi_affinity=None):
    mode = str(getattr(cfg.training, "roi_clip_alignment_mode", "hard")).lower()
    if mode in {"topology_soft", "anatomical_transport", "soft"}:
        return anatomical_transport_roi_clip_loss(
            z_surface,
            z_volume,
            roi_affinity=roi_affinity,
            temperature=cfg.temperature.clip,
            epsilon=float(getattr(cfg.training, "roi_clip_soft_epsilon", 0.1)),
        )
    if mode in {"distance_weighted", "weighted_negatives", "dnce"}:
        return distance_weighted_roi_clip_loss(
            z_surface,
            z_volume,
            roi_affinity=roi_affinity,
            temperature=cfg.temperature.clip,
            min_negative_weight=float(getattr(cfg.training, "roi_clip_min_negative_weight", 0.05)),
        )
    return roi_clip_loss(z_surface, z_volume, temperature=cfg.temperature.clip)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg,
    class_weights,
    train_mode,
    auxiliary_loss_scale=1.0,
    roi_affinity=None,
    cross_geometry_affinity=None,
):
    model.train(train_mode)
    use_roi_branch = bool(cfg.model.use_roi_branch)
    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    clip_meter = AverageMeter()
    proto_meter = AverageMeter()
    rel_meter = AverageMeter()
    ribbon_valid_meter = AverageMeter()
    ribbon_depth_entropy_meter = AverageMeter()
    ribbon_roi_valid_meter = AverageMeter()
    hybrid_gate_meter = AverageMeter()
    legacy_token_norm_meter = AverageMeter()
    ribbon_token_norm_meter = AverageMeter()
    cross_geometry_loss_meter = AverageMeter()
    roi_graph_metric_meters = {
        name: AverageMeter()
        for name in [
            "graph_anatomical_mass",
            "graph_homology_mass",
            "graph_identity_mass",
            "graph_remote_mass",
            "graph_attention_entropy",
            "graph_attention_uniform_kl",
            "graph_attention_std",
            "graph_boundary_correlation",
            "graph_uniform_anatomical_mass",
            "graph_uniform_homology_mass",
            "graph_residual_gate",
            "graph_output_delta_l2",
        ]
    }
    cross_geometry_metric_meters = {
        name: AverageMeter()
        for name in [
            "roi_retrieval_acc1",
            "roi_retrieval_acc5",
            "roi_retrieval_mrr",
            "same_roi_similarity",
            "off_diagonal_similarity",
            "diagonal_similarity_gap",
            "topology_retrieval_acc",
        ]
    }
    all_labels = []
    all_probs = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if use_roi_branch:
                batch = move_to_device(batch, device)
            else:
                batch["image"] = batch["image"].to(device)
                batch["label"] = batch["label"].to(device)
                if batch.get("clinical_data") is not None:
                    batch["clinical_data"] = batch["clinical_data"].to(device)
            model_inputs = dict(
                image=batch["image"],
                surface_data=batch["surface_data"] if use_roi_branch else None,
                surface_roi_masks=batch["surface_roi_masks"] if use_roi_branch else None,
                local_ribbon_masks=batch["local_ribbon_masks"] if use_roi_branch else None,
                roi_volume_masks=batch["roi_volume_masks"] if use_roi_branch else None,
                labels=batch["label"],
                clinical_data=batch["clinical_data"],
            )
            cross_geometry_cfg = getattr(cfg.model, "cross_geometry", {})
            cross_geometry_enabled = bool(getattr(cross_geometry_cfg, "enabled", False))
            if (
                str(getattr(getattr(cfg.model, "volume_pooling", {}), "mode", "legacy_mask")).lower()
                in {"ribbon", "hybrid_ribbon"}
                or cross_geometry_enabled
            ):
                model_inputs.update(
                    ribbon_grid=batch["ribbon_grid"],
                    ribbon_valid_mask=batch["ribbon_valid_mask"],
                    vertex_valid_mask=batch["vertex_valid_mask"],
                    vertex_roi_index=batch["vertex_roi_index"],
                )
            outputs = model(**model_inputs)

            loss_ce = classification_loss(outputs["logits"], batch["label"], class_weights=class_weights)
            loss_clip = loss_ce.new_zeros(())
            loss_proto = loss_ce.new_zeros(())
            loss_rel = loss_ce.new_zeros(())
            loss_cross_geometry = loss_ce.new_zeros(())
            if use_roi_branch and auxiliary_loss_scale > 0.0 and "z_surface" in outputs and "z_volume" in outputs:
                loss_clip = teacher_roi_clip_loss(
                    outputs["z_surface"],
                    outputs["z_volume"],
                    cfg=cfg,
                    roi_affinity=roi_affinity,
                )
                loss_proto = (
                    prototype_loss(outputs["surface_prototype_logits"], batch["label"])
                    + prototype_loss(outputs["volume_prototype_logits"], batch["label"])
                    + prototype_loss(outputs["prototype_logits"], batch["label"])
                ) / 3.0
                loss_rel = relation_consistency_loss(outputs["surface_relation"], outputs["volume_relation"])
            elif auxiliary_loss_scale > 0.0 and "z_mri" in outputs and "z_clinical" in outputs:
                loss_clip = roi_clip_loss(outputs["z_mri"], outputs["z_clinical"], temperature=cfg.temperature.clip)
            if cross_geometry_enabled:
                loss_cross_geometry = cross_geometry_correspondence_loss(
                    outputs["cross_geometry_surface"],
                    outputs["cross_geometry_ribbon"],
                    temperature=float(getattr(cross_geometry_cfg, "temperature", 0.1)),
                )
            total_loss = (
                float(cfg.loss_weights.ce) * loss_ce
                + auxiliary_loss_scale * float(cfg.loss_weights.roi_clip) * loss_clip
                + auxiliary_loss_scale * float(cfg.loss_weights.prototype) * loss_proto
                + auxiliary_loss_scale * float(cfg.loss_weights.relation) * loss_rel
                + float(getattr(cross_geometry_cfg, "lambda_cg", 0.0)) * loss_cross_geometry
            )

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            batch_size = int(batch["label"].size(0))
            total_meter.update(total_loss.item(), batch_size)
            ce_meter.update(loss_ce.item(), batch_size)
            clip_meter.update(loss_clip.item(), batch_size)
            proto_meter.update(loss_proto.item(), batch_size)
            rel_meter.update(loss_rel.item(), batch_size)
            cross_geometry_loss_meter.update(loss_cross_geometry.item(), batch_size)
            if cross_geometry_enabled:
                cross_geometry_metrics = cross_geometry_retrieval_metrics(
                    outputs["cross_geometry_surface"],
                    outputs["cross_geometry_ribbon"],
                    topology_affinity=cross_geometry_affinity,
                )
                for name, value in cross_geometry_metrics.items():
                    cross_geometry_metric_meters[name].update(float(value.item()), batch_size)
            if outputs.get("depth_attention") is not None:
                valid = outputs["ribbon_vertex_valid_mask"]
                attention = outputs["depth_attention"].clamp_min(1e-8)
                entropy = -(attention * attention.log()).sum(dim=-1)
                valid_count = valid.sum().clamp_min(1)
                ribbon_valid_meter.update(valid.float().mean().item(), batch_size)
                ribbon_depth_entropy_meter.update(
                    ((entropy * valid.float()).sum() / valid_count).item(), batch_size
                )
                ribbon_roi_valid_meter.update(outputs["ribbon_roi_valid_ratio"].mean().item(), batch_size)
                ribbon_token_norm_meter.update(
                    outputs["mri_ribbon_tokens"].norm(dim=-1).mean().item(), batch_size
                )
            if outputs.get("legacy_volume_tokens") is not None:
                legacy_token_norm_meter.update(
                    outputs["legacy_volume_tokens"].norm(dim=-1).mean().item(), batch_size
                )
            if outputs.get("hybrid_ribbon_gate") is not None:
                hybrid_gate_meter.update(outputs["hybrid_ribbon_gate"].mean().item(), batch_size)
            graph_output_map = {
                "graph_anatomical_mass": "roi_graph_anatomical_mass",
                "graph_homology_mass": "roi_graph_homology_mass",
                "graph_identity_mass": "roi_graph_identity_mass",
                "graph_remote_mass": "roi_graph_remote_mass",
                "graph_attention_entropy": "roi_graph_attention_entropy",
                "graph_attention_uniform_kl": "roi_graph_attention_uniform_kl",
                "graph_attention_std": "roi_graph_attention_std",
                "graph_boundary_correlation": "roi_graph_boundary_correlation",
                "graph_uniform_anatomical_mass": "roi_graph_uniform_anatomical_mass",
                "graph_uniform_homology_mass": "roi_graph_uniform_homology_mass",
                "graph_residual_gate": "roi_graph_residual_gate",
                "graph_output_delta_l2": "roi_graph_output_delta_l2",
            }
            for metric_name, output_name in graph_output_map.items():
                if outputs.get(output_name) is not None:
                    roi_graph_metric_meters[metric_name].update(
                        float(outputs[output_name].detach().item()), batch_size
                    )
            all_labels.extend(batch["label"].detach().cpu().tolist())
            all_probs.extend(outputs["prob"].detach().cpu().tolist())

            if cfg.runtime.dry_run:
                break

    metrics = compute_classification_metrics(all_labels, all_probs)
    metrics.update(
        {name: meter.avg for name, meter in cross_geometry_metric_meters.items()}
    )
    metrics.update({name: meter.avg for name, meter in roi_graph_metric_meters.items()})
    losses = {
        "loss": total_meter.avg,
        "ce": ce_meter.avg,
        "clip": clip_meter.avg,
        "proto": proto_meter.avg,
        "rel": rel_meter.avg,
        "ribbon_valid_fraction": ribbon_valid_meter.avg,
        "ribbon_depth_entropy": ribbon_depth_entropy_meter.avg,
        "ribbon_roi_valid_ratio": ribbon_roi_valid_meter.avg,
        "hybrid_ribbon_gate": hybrid_gate_meter.avg,
        "legacy_token_l2": legacy_token_norm_meter.avg,
        "ribbon_token_l2": ribbon_token_norm_meter.avg,
        "cross_geometry": cross_geometry_loss_meter.avg,
    }
    return metrics, losses


def evaluate_teacher(
    model,
    loader,
    device,
    cfg,
    roi_affinity=None,
    cross_geometry_affinity=None,
):
    metrics, losses = run_epoch(
        model,
        loader,
        optimizer=None,
        device=device,
        cfg=cfg,
        class_weights=None,
        train_mode=False,
        roi_affinity=roi_affinity,
        cross_geometry_affinity=cross_geometry_affinity,
    )
    return metrics, losses


def collect_validation_predictions(model, loader, device, cfg):
    """Collect development predictions without touching the held-out test split."""
    model.eval()
    use_roi_branch = bool(cfg.model.use_roi_branch)
    records = []
    with torch.no_grad():
        for batch in loader:
            if use_roi_branch:
                batch = move_to_device(batch, device)
            else:
                batch["image"] = batch["image"].to(device)
                batch["label"] = batch["label"].to(device)
                if batch.get("clinical_data") is not None:
                    batch["clinical_data"] = batch["clinical_data"].to(device)
            model_inputs = dict(
                image=batch["image"],
                surface_data=batch["surface_data"] if use_roi_branch else None,
                surface_roi_masks=batch["surface_roi_masks"] if use_roi_branch else None,
                local_ribbon_masks=batch["local_ribbon_masks"] if use_roi_branch else None,
                roi_volume_masks=batch["roi_volume_masks"] if use_roi_branch else None,
                labels=batch["label"],
                clinical_data=batch["clinical_data"],
            )
            cross_geometry_cfg = getattr(cfg.model, "cross_geometry", {})
            cross_geometry_enabled = bool(getattr(cross_geometry_cfg, "enabled", False))
            if (
                str(getattr(getattr(cfg.model, "volume_pooling", {}), "mode", "legacy_mask")).lower()
                in {"ribbon", "hybrid_ribbon"}
                or cross_geometry_enabled
            ):
                model_inputs.update(
                    ribbon_grid=batch["ribbon_grid"],
                    ribbon_valid_mask=batch["ribbon_valid_mask"],
                    vertex_valid_mask=batch["vertex_valid_mask"],
                    vertex_roi_index=batch["vertex_roi_index"],
                )
            outputs = model(**model_inputs)
            probabilities = outputs["prob"].detach().cpu()
            labels = batch["label"].detach().cpu()
            filenames = list(batch["filename"])
            subject_ids = list(batch["subject_id"])
            for index in range(probabilities.size(0)):
                filename = str(filenames[index])
                records.append(
                    {
                        "filename": filename,
                        "subject": filename.split("_", 1)[0],
                        "subject_id": str(subject_ids[index]),
                        "label": int(labels[index].item()),
                        "prediction": int(probabilities[index].argmax().item()),
                        "probabilities": probabilities[index].tolist(),
                    }
                )
            if cfg.runtime.dry_run:
                break
    return records


def main():
    cfg = Option(
        stage="teacher",
        default_paths=[
            os.path.join(ROOT, "configs", "data.yaml"),
            os.path.join(ROOT, "configs", "teacher.yaml"),
        ],
    ).parse()
    seed_everything(int(cfg.training.seed))
    device = torch.device(cfg.runtime.device if cfg.runtime.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg.model.num_classes = int(getattr(cfg.model, "num_classes", infer_num_classes(cfg.data.task)))
    expr_dir = cfg.expr_dir
    root_logger = Logger(expr_dir)
    root_logger.print_message(f"Teacher experiment dir: {expr_dir}")

    folds = [int(cfg.runtime.fold)] if getattr(cfg.runtime, "fold", None) is not None else list(range(int(cfg.training.num_folds)))
    results = []

    for fold in folds:
        fold_start_time = now()
        fold_dir = os.path.join(expr_dir, f"fold_{fold}")
        logger = Logger(fold_dir)
        logger.print_message(f"Starting fold {fold}")
        use_roi_branch = model_requires_teacher_inputs(cfg)
        train_loader, val_loader, test_loader = build_dataloaders(
            cfg,
            fold=fold,
            require_teacher_inputs=use_roi_branch,
            filter_teacher_available=True,
        )
        logger.print_message(f"ROI branch enabled={use_roi_branch}")
        logger.print_message(
            "Volume pooling "
            f"mode={str(getattr(getattr(cfg.model, 'volume_pooling', {}), 'mode', 'legacy_mask'))} "
            f"depths={list(getattr(getattr(cfg.model, 'ribbon', {}), 'depths', []))}"
        )
        cross_geometry_cfg = getattr(cfg.model, "cross_geometry", {})
        cross_geometry_enabled = bool(getattr(cross_geometry_cfg, "enabled", False))
        logger.print_message(
            "Cross-geometry "
            f"enabled={cross_geometry_enabled} "
            f"feature_stage={str(getattr(cross_geometry_cfg, 'feature_stage', 'high_res'))} "
            f"lambda_cg={float(getattr(cross_geometry_cfg, 'lambda_cg', 0.0)):.4f} "
            f"temperature={float(getattr(cross_geometry_cfg, 'temperature', 0.1)):.4f} "
            "target=hard_same_roi"
        )
        roi_graph_cfg = getattr(cfg.model, "roi_graph", {})
        logger.print_message(
            "ROI graph "
            f"enabled={bool(getattr(roi_graph_cfg, 'enabled', False))} "
            f"layers={int(getattr(roi_graph_cfg, 'num_layers', 1))} "
            f"heads={int(getattr(roi_graph_cfg, 'num_heads', 4))} "
            f"boundary_bias={bool(getattr(roi_graph_cfg, 'use_boundary_strength', False))} "
            f"homology_bias={bool(getattr(roi_graph_cfg, 'use_homology', False))} "
            f"shuffled_bias={bool(getattr(roi_graph_cfg, 'shuffle_topology_bias', False))}"
        )
        logger.print_message(
            f"Clinical fusion enabled={bool(getattr(cfg.model, 'use_clinical', False))} "
            f"mode={str(getattr(cfg.model, 'clinical_fusion_mode', 'gated'))} "
            f"features={len(train_loader.dataset.clinical_features)} "
            f"input_dim={int(cfg.model.clinical_input_dim)}"
        )
        logger.print_message(
            f"Atlas mode={bool(getattr(getattr(cfg.data, 'atlas', {}), 'enabled', False))} "
            f"atlas_name={str(getattr(getattr(cfg.data, 'atlas', {}), 'name', 'none'))} "
            f"resolved_num_rois={int(cfg.model.num_rois)}"
        )
        logger.print_message(
            f"Surface refiner={str(getattr(cfg.model, 'surface_refiner', 'none'))} "
            f"hidden={int(getattr(cfg.model, 'surface_refiner_hidden_dim', 0))} "
            f"dropout={float(getattr(cfg.model, 'surface_refiner_dropout', 0.0))} "
            f"residual_init={float(getattr(cfg.model, 'surface_refiner_residual_init', 0.1))}"
        )
        logger.print_message(
            f"Fold {fold} sizes: train={len(train_loader.dataset)} ({len(train_loader)} steps) "
            f"val={len(val_loader.dataset)} ({len(val_loader)} steps) "
            + (
                f"test={len(test_loader.dataset)} ({len(test_loader)} steps)"
                if test_loader is not None
                else "test=NOT_LOADED (development-only)"
            )
        )
        roi_affinity = build_roi_clip_affinity(train_loader.dataset, cfg, logger=logger)
        if roi_affinity is not None:
            roi_affinity = roi_affinity.to(device)
        else:
            logger.print_message("Teacher ROI-CLIP alignment mode=hard (identity targets).")
        cross_geometry_affinity = None
        if cross_geometry_enabled:
            cross_geometry_affinity = build_roi_clip_affinity(
                train_loader.dataset,
                cfg,
                logger=logger,
                force=True,
                topk_override=int(getattr(cross_geometry_cfg, "topology_metric_topk", 3)),
                log_label="Cross-geometry topology metric",
            ).to(device)
        class_weights = compute_class_weights(
            train_loader.dataset,
            num_classes=int(cfg.model.num_classes),
            mode=str(cfg.training.class_weight_mode),
        )
        logger.print_message(f"Class weights: {class_weights}")

        model = build_model(cfg).to(device)
        logger.print_message(f"Teacher parameters: {count_parameters(model)}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
        ckpt = CheckpointIO(os.path.join(fold_dir, "best_teacher.pt"))

        best_score = float("-inf")
        stale = 0
        ce_warmup_epochs = max(0, int(cfg.training.ce_warmup_epochs))
        logger.print_message(f"Teacher CE-only warmup epochs: {ce_warmup_epochs}")
        fold_train_seconds = 0.0
        fold_val_seconds = 0.0
        for epoch in range(int(cfg.training.epochs)):
            auxiliary_loss_scale = auxiliary_loss_scale_for_epoch(epoch, ce_warmup_epochs)
            epoch_start_time = now()
            train_start_time = now()
            train_metrics, train_losses = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                class_weights,
                train_mode=True,
                auxiliary_loss_scale=auxiliary_loss_scale,
                roi_affinity=roi_affinity,
                cross_geometry_affinity=cross_geometry_affinity,
            )
            train_elapsed = now() - train_start_time
            val_start_time = now()
            val_metrics, val_losses = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                cfg=cfg,
                class_weights=class_weights,
                train_mode=False,
                auxiliary_loss_scale=auxiliary_loss_scale,
                roi_affinity=roi_affinity,
                cross_geometry_affinity=cross_geometry_affinity,
            )
            val_elapsed = now() - val_start_time
            epoch_elapsed = now() - epoch_start_time
            fold_train_seconds += train_elapsed
            fold_val_seconds += val_elapsed

            logger.print_message(
                f"Epoch {epoch + 1}/{cfg.training.epochs} "
                f"train_loss={train_losses['loss']:.4f} val_loss={val_losses['loss']:.4f} "
                f"train_ce={train_losses['ce']:.4f} train_clip={train_losses['clip']:.4f} "
                f"train_cg={train_losses['cross_geometry']:.4f} "
                f"aux_scale={auxiliary_loss_scale:.1f} "
                f"ribbon_valid={val_losses['ribbon_valid_fraction']:.4f} "
                f"ribbon_depth_entropy={val_losses['ribbon_depth_entropy']:.4f} "
                f"ribbon_roi_valid={val_losses['ribbon_roi_valid_ratio']:.4f} "
                f"hybrid_gate={val_losses['hybrid_ribbon_gate']:.4f} "
                f"legacy_l2={val_losses['legacy_token_l2']:.3f} "
                f"ribbon_l2={val_losses['ribbon_token_l2']:.3f} "
                f"cg_acc1={val_metrics['roi_retrieval_acc1']:.4f} "
                f"cg_acc5={val_metrics['roi_retrieval_acc5']:.4f} "
                f"cg_mrr={val_metrics['roi_retrieval_mrr']:.4f} "
                f"cg_diag={val_metrics['same_roi_similarity']:.4f} "
                f"cg_offdiag={val_metrics['off_diagonal_similarity']:.4f} "
                f"cg_gap={val_metrics['diagonal_similarity_gap']:.4f} "
                f"cg_topology_acc={val_metrics['topology_retrieval_acc']:.4f} "
                f"graph_anat={val_metrics['graph_anatomical_mass']:.4f} "
                f"graph_hom={val_metrics['graph_homology_mass']:.4f} "
                f"graph_identity={val_metrics['graph_identity_mass']:.4f} "
                f"graph_remote={val_metrics['graph_remote_mass']:.4f} "
                f"graph_entropy={val_metrics['graph_attention_entropy']:.4f} "
                f"graph_uniform_kl={val_metrics['graph_attention_uniform_kl']:.4f} "
                f"graph_std={val_metrics['graph_attention_std']:.4f} "
                f"graph_boundary_corr={val_metrics['graph_boundary_correlation']:.4f} "
                f"graph_gate={val_metrics['graph_residual_gate']:.4f} "
                f"graph_delta={val_metrics['graph_output_delta_l2']:.4f} "
                f"roi_weight={float(torch.sigmoid(model.roi_logit_scale).detach()) if hasattr(model, 'roi_logit_scale') else 0.0:.4f} "
                f"{format_classification_metrics(val_metrics, prefix='val_')} "
                f"train_time={format_duration(train_elapsed)} val_time={format_duration(val_elapsed)} "
                f"epoch_time={format_duration(epoch_elapsed)}"
            )
            score = selection_score(val_metrics, str(cfg.training.selection_metric))
            if score > best_score:
                best_score = score
                stale = 0
                payload = {
                    "model": model.state_dict(),
                    "config": dict(cfg),
                    "fold": fold,
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "val_losses": val_losses,
                    "class_weights": class_weights,
                    "clinical_features": list(train_loader.dataset.clinical_features),
                    "clinical_stats": train_loader.dataset.clinical_stats,
                }
                ckpt.save(payload)
                save_component_weights(model, fold_dir)
                logger.print_message("Saved new best teacher checkpoint.")
            else:
                if epoch + 1 <= ce_warmup_epochs:
                    stale = 0
                else:
                    stale += 1
                    if stale >= int(cfg.training.early_stop_patience):
                        logger.print_message(f"Early stopping at epoch {epoch + 1}.")
                        break

            if cfg.runtime.dry_run:
                break

        state = ckpt.load(map_location=device)
        model.load_state_dict(state["model"])
        validation_predictions = collect_validation_predictions(model, val_loader, device, cfg)
        logger.save_json(
            {
                "split": "validation",
                "fold": fold,
                "held_out_test_evaluated": False,
                "records": validation_predictions,
            },
            "validation_predictions.json",
        )
        if not bool(getattr(cfg.runtime, "evaluate_test", True)):
            fold_elapsed = now() - fold_start_time
            gpu_peak_memory_mb = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if device.type == "cuda"
                else 0.0
            )
            logger.print_message(
                f"Fold {fold} development-only complete: best_epoch={int(state['epoch'])} "
                f"{format_classification_metrics(state['val_metrics'], prefix='best_val_')} "
                f"fold_total_time={format_duration(fold_elapsed)} "
                f"gpu_peak_memory_mb={gpu_peak_memory_mb:.1f}; held-out test not evaluated"
            )
            logger.save_json(
                {
                    "best_epoch": int(state["epoch"]),
                    "best_val_metrics": state["val_metrics"],
                    "best_val_losses": state["val_losses"],
                    "gpu_peak_memory_mb": gpu_peak_memory_mb,
                    "fold_total_seconds": fold_elapsed,
                    "held_out_test_evaluated": False,
                },
                "development_metrics.json",
            )
            continue
        test_start_time = now()
        if test_loader is None:
            raise RuntimeError("Test evaluation requested but the held-out test loader was not built")
        test_metrics, test_losses = evaluate_teacher(
            model,
            test_loader,
            device,
            cfg,
            roi_affinity=roi_affinity,
            cross_geometry_affinity=cross_geometry_affinity,
        )
        test_elapsed = now() - test_start_time
        fold_elapsed = now() - fold_start_time
        gpu_peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        logger.print_message(
            f"Fold {fold} test: loss={test_losses['loss']:.4f} "
            f"{format_classification_metrics(test_metrics)} "
            f"train_total_time={format_duration(fold_train_seconds)} "
            f"val_total_time={format_duration(fold_val_seconds)} "
            f"test_time={format_duration(test_elapsed)} fold_total_time={format_duration(fold_elapsed)} "
            f"gpu_peak_memory_mb={gpu_peak_memory_mb:.1f}"
        )
        logger.save_json(
            {
                "metrics": test_metrics,
                "losses": test_losses,
                "timing": {
                    "train_total_seconds": fold_train_seconds,
                    "val_total_seconds": fold_val_seconds,
                    "test_seconds": test_elapsed,
                    "fold_total_seconds": fold_elapsed,
                },
                "gpu_peak_memory_mb": gpu_peak_memory_mb,
                "best_epoch": int(state["epoch"]),
                "best_val_metrics": state["val_metrics"],
                "best_val_losses": state["val_losses"],
            },
            "test_metrics.json",
        )
        results.append(test_metrics)

    if results:
        summary = summarize_metric_results(results)
        root_logger.save_json({"fold_metrics": results, "summary": summary}, "crossval_summary.json")
        root_logger.print_message(f"Final mean+/-std {format_metric_summary(summary)}")


if __name__ == "__main__":
    main()
