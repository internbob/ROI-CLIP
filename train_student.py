import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import build_dataloaders
from losses import (
    attention_distill_loss,
    classification_loss,
    kd_logit_loss,
    multiscale_relation_distill_loss,
    prototype_distill_loss,
    relation_distill_loss,
    token_distill_loss,
)
from data.ico_utils import get_ico_topology, infer_ico_level
from metrics import AverageMeter, compute_classification_metrics, format_classification_metrics
from models.student import MRIOnlyStudent
from models.teacher import CortexWeaveROIClipTeacher
from options import Option
from utils import (
    CheckpointIO,
    Logger,
    count_parameters,
    format_duration,
    format_metric_summary,
    move_to_device,
    now,
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


def build_teacher(cfg):
    use_clinical = bool(getattr(cfg.model, "use_clinical", False))
    return CortexWeaveROIClipTeacher(
        surface_in_channels=int(getattr(cfg.model, "surface_in_channels", 3)),
        feature_dim=int(cfg.model.feature_dim),
        proj_dim=int(cfg.model.proj_dim),
        num_rois=int(cfg.model.num_rois),
        num_classes=int(cfg.model.num_classes),
        num_vertices=int(cfg.data.num_vertices),
        base_channels=int(cfg.model.base_channels),
        surface_hidden_dim=int(getattr(cfg.model, "surface_hidden_dim", 128)),
        dropout=float(cfg.model.dropout),
        prototype_temperature=float(cfg.temperature.prototype),
        roi_residual_init=float(cfg.model.roi_residual_init),
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
    )


def build_student(cfg):
    use_clinical = bool(getattr(cfg.model, "use_clinical", False))
    return MRIOnlyStudent(
        feature_dim=int(cfg.model.feature_dim),
        proj_dim=int(cfg.model.proj_dim),
        num_rois=int(cfg.model.num_rois),
        num_classes=int(cfg.model.num_classes),
        base_channels=int(cfg.model.base_channels),
        dropout=float(cfg.model.dropout),
        prototype_temperature=float(cfg.temperature.prototype),
        roi_residual_init=float(cfg.model.roi_residual_init),
        volume_norm=str(cfg.model.volume_norm),
        use_clinical=use_clinical,
        clinical_input_dim=int(getattr(cfg.model, "clinical_input_dim", 0)),
        clinical_fusion_mode=str(getattr(cfg.model, "clinical_fusion_mode", "gated")),
        clinical_hidden_dim=int(getattr(cfg.model, "clinical_hidden_dim", 256)),
        clinical_dropout=float(getattr(cfg.model, "clinical_dropout", 0.1)),
        clinical_num_heads=int(getattr(cfg.model, "clinical_num_heads", 4)),
        clinical_num_layers=int(getattr(cfg.model, "clinical_num_layers", 2)),
    )


def initialize_student_from_teacher(student, teacher):
    student.volume_encoder.load_state_dict(teacher.volume_encoder.state_dict())
    student.global_head.load_state_dict(teacher.global_head.state_dict())
    if student.clinical_fusion is not None and teacher.clinical_fusion is not None:
        student.clinical_fusion.load_state_dict(teacher.clinical_fusion.state_dict())
    if student.global_to_cross is not None and teacher.global_to_cross is not None:
        student.global_to_cross.load_state_dict(teacher.global_to_cross.state_dict())
        student.cross_global_head.load_state_dict(teacher.cross_global_head.state_dict())
    if student.clinical_cross_fusion is not None and teacher.clinical_cross_fusion is not None:
        student.clinical_cross_fusion.load_state_dict(teacher.clinical_cross_fusion.state_dict())
    if student.clinical_atlas_prompt is not None and teacher.clinical_atlas_prompt is not None:
        student.clinical_atlas_prompt.load_state_dict(teacher.clinical_atlas_prompt.state_dict())
    student.projector.load_state_dict(teacher.roi_clip.volume_proj.state_dict())
    student.prototype_bank.prototypes.data.copy_(teacher.prototype_bank.prototypes.data)


def inherited_student_modules(student):
    modules = [
        student.volume_encoder,
        student.global_head,
        student.projector,
        student.prototype_bank,
    ]
    if student.clinical_fusion is not None:
        modules.append(student.clinical_fusion)
    if student.global_to_cross is not None:
        modules.extend([student.global_to_cross, student.cross_global_head])
    if student.clinical_cross_fusion is not None:
        modules.append(student.clinical_cross_fusion)
    if student.clinical_atlas_prompt is not None:
        modules.append(student.clinical_atlas_prompt)
    return tuple(modules)


def set_inherited_trainable(student, trainable):
    for module in inherited_student_modules(student):
        for param in module.parameters():
            param.requires_grad = bool(trainable)


def build_student_optimizer(student, cfg):
    encoder_params = list(student.volume_encoder.parameters())
    inherited_params = (
        list(student.global_head.parameters())
        + list(student.projector.parameters())
        + list(student.prototype_bank.parameters())
    )
    if student.clinical_fusion is not None:
        inherited_params += list(student.clinical_fusion.parameters())
    if student.global_to_cross is not None:
        inherited_params += list(student.global_to_cross.parameters())
        inherited_params += list(student.cross_global_head.parameters())
    if student.clinical_cross_fusion is not None:
        inherited_params += list(student.clinical_cross_fusion.parameters())
    if student.clinical_atlas_prompt is not None:
        inherited_params += list(student.clinical_atlas_prompt.parameters())
    inherited_ids = {id(param) for param in encoder_params + inherited_params}
    new_head_params = [param for param in student.parameters() if id(param) not in inherited_ids]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(cfg.training.encoder_lr), "name": "volume_encoder"},
            {"params": inherited_params, "lr": float(cfg.training.inherited_lr), "name": "inherited_heads"},
            {"params": new_head_params, "lr": float(cfg.training.lr), "name": "student_heads"},
        ],
        weight_decay=float(cfg.training.weight_decay),
    )


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


def build_relation_distill_weights(dataset, cfg, logger=None):
    mode = str(getattr(cfg.training, "relation_distill_mode", "global")).lower()
    if mode != "local_remote":
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
            f"Cannot build relation weights: masks={roi_masks.shape}, expected=({num_rois}, {coords.shape[0]})"
        )

    centroids = []
    for roi_idx in range(num_rois):
        weights = roi_masks[roi_idx]
        denom = float(weights.sum())
        if denom <= 1e-6:
            center = coords.mean(axis=0)
        else:
            center = (weights[:, None] * coords).sum(axis=0) / denom
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
    sigma = float(getattr(cfg.training, "relation_distance_sigma", 0.0))
    if sigma <= 0.0:
        sigma = float(np.median(within_distances[within_distances > 1e-6])) if np.any(within_distances > 1e-6) else 1.0
    sigma = max(sigma, 1e-6)

    affinity = np.exp(-(distance ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    affinity[~same_hemi] = 0.0
    np.fill_diagonal(affinity, 0.0)

    homology_weight = max(0.0, float(getattr(cfg.training, "relation_homology_weight", 1.0)))
    local_score = affinity + homology_weight * homology.astype(np.float32)
    np.fill_diagonal(local_score, 0.0)

    topk = max(1, int(getattr(cfg.training, "relation_local_topk", 6)))
    local_weight = np.zeros_like(local_score, dtype=np.float32)
    for roi_idx in range(num_rois):
        candidates = np.argsort(local_score[roi_idx])[::-1]
        selected = [idx for idx in candidates if local_score[roi_idx, idx] > 0.0][:topk]
        if selected:
            local_weight[roi_idx, selected] = local_score[roi_idx, selected]
    local_weight = np.maximum(local_weight, local_weight.T)

    remote_weight = (1.0 - affinity).astype(np.float32) * offdiag.astype(np.float32)
    remote_weight[local_weight > 0.0] = 0.0
    if float(remote_weight.sum()) <= 1e-6:
        remote_weight = offdiag.astype(np.float32)
        remote_weight[local_weight > 0.0] = 0.0

    if logger is not None:
        local_edges = int(np.count_nonzero(local_weight))
        remote_edges = int(np.count_nonzero(remote_weight))
        logger.print_message(
            "Relation distillation mode=local_remote "
            f"local_topk={topk} sigma={sigma:.4f} homology_weight={homology_weight:.3f} "
            f"local_edges={local_edges} remote_edges={remote_edges}"
        )

    return {
        "local": torch.from_numpy(local_weight.astype(np.float32)),
        "remote": torch.from_numpy(remote_weight.astype(np.float32)),
    }


def selection_score(metrics, name):
    if name == "acc":
        return float(metrics["acc"])
    if name == "f1":
        return float(metrics.get("f1", metrics["f1_macro"]))
    return float(metrics["mcc"])


def locate_teacher_checkpoint(cfg, fold):
    candidates = []
    if str(cfg.training.teacher_checkpoint):
        candidates.append(str(cfg.training.teacher_checkpoint))
    if str(getattr(cfg.runtime, "checkpoint", "")):
        candidates.append(str(cfg.runtime.checkpoint))
    for candidate in candidates:
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, f"fold_{fold}", "best_teacher.pt")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Teacher checkpoint not found. Set training.teacher_checkpoint or pass --checkpoint.")


def run_epoch(student, teacher, loader, optimizer, device, cfg, class_weights, train_mode, warmup=False, relation_weights=None):
    student.train(train_mode)
    if train_mode and warmup:
        # Frozen BatchNorm statistics must stay fixed during the head-only warmup.
        for module in inherited_student_modules(student):
            module.eval()
    teacher.eval()
    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    logit_meter = AverageMeter()
    token_meter = AverageMeter()
    attn_meter = AverageMeter()
    rel_meter = AverageMeter()
    proto_meter = AverageMeter()
    all_labels = []
    all_probs = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            with torch.no_grad():
                teacher_outputs = teacher(
                    image=batch["image"],
                    surface_data=batch["surface_data"],
                    surface_roi_masks=batch["surface_roi_masks"],
                    local_ribbon_masks=batch["local_ribbon_masks"],
                    roi_volume_masks=batch["roi_volume_masks"],
                    labels=batch["label"],
                    clinical_data=batch["clinical_data"],
                )
                if bool(getattr(cfg.training, "control_shuffle_roi", False)):
                    teacher_outputs = shuffle_teacher_roi_outputs(teacher_outputs)
            student_outputs = student(batch["image"], clinical_data=batch["clinical_data"])

            loss_ce = classification_loss(student_outputs["logits"], batch["label"], class_weights=class_weights)
            loss_logit = kd_logit_loss(teacher_outputs["logits"], student_outputs["logits"], temperature=cfg.temperature.kd)
            loss_token = token_distill_loss(teacher_outputs["z_volume"], student_outputs["z_student"])
            if teacher_outputs.get("cross_image_tokens") is not None:
                loss_token = (
                    loss_token
                    + token_distill_loss(
                        teacher_outputs["cross_image_tokens"][:, 1:],
                        student_outputs["cross_image_tokens"][:, 1:],
                    )
                    + token_distill_loss(
                        teacher_outputs["clinical_tokens"],
                        student_outputs["clinical_tokens"],
                    )
                ) / 3.0
            loss_attn = student_outputs["logits"].new_zeros(())
            if float(cfg.loss_weights.kd_attention) > 0.0:
                loss_attn = attention_distill_loss(teacher_outputs["mask_attention"], student_outputs["attention_maps"])
            if relation_weights is not None:
                loss_rel = multiscale_relation_distill_loss(
                    teacher_outputs["volume_relation"],
                    student_outputs["relation"],
                    local_weight=relation_weights["local"],
                    remote_weight=relation_weights["remote"],
                    local_scale=float(getattr(cfg.training, "relation_local_weight", 0.7)),
                    remote_scale=float(getattr(cfg.training, "relation_remote_weight", 0.3)),
                )
            else:
                loss_rel = relation_distill_loss(teacher_outputs["volume_relation"], student_outputs["relation"])
            loss_proto = prototype_distill_loss(
                teacher_outputs["volume_prototype_logits"],
                student_outputs["prototype_logits"],
                temperature=1.0,
            )
            total_loss = (
                float(cfg.loss_weights.ce) * loss_ce
                + float(cfg.loss_weights.kd_logit) * loss_logit
                + float(cfg.loss_weights.kd_token) * loss_token
                + float(cfg.loss_weights.kd_attention) * loss_attn
                + float(cfg.loss_weights.kd_relation) * loss_rel
                + float(cfg.loss_weights.kd_prototype) * loss_proto
            )

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            batch_size = int(batch["label"].size(0))
            total_meter.update(total_loss.item(), batch_size)
            ce_meter.update(loss_ce.item(), batch_size)
            logit_meter.update(loss_logit.item(), batch_size)
            token_meter.update(loss_token.item(), batch_size)
            attn_meter.update(loss_attn.item(), batch_size)
            rel_meter.update(loss_rel.item(), batch_size)
            proto_meter.update(loss_proto.item(), batch_size)
            all_labels.extend(batch["label"].detach().cpu().tolist())
            all_probs.extend(student_outputs["prob"].detach().cpu().tolist())

            if cfg.runtime.dry_run:
                break

    metrics = compute_classification_metrics(all_labels, all_probs)
    losses = {
        "loss": total_meter.avg,
        "ce": ce_meter.avg,
        "kd_logit": logit_meter.avg,
        "kd_token": token_meter.avg,
        "kd_attention": attn_meter.avg,
        "kd_relation": rel_meter.avg,
        "kd_prototype": proto_meter.avg,
    }
    return metrics, losses


def shuffle_teacher_roi_outputs(outputs):
    """Break anatomical ROI correspondence while preserving sample-level logits."""
    shuffled = dict(outputs)
    logits = outputs.get("logits")
    device = logits.device if logits is not None else next(
        value.device for value in outputs.values() if torch.is_tensor(value)
    )
    num_rois = None
    for key in ("z_volume", "volume_prototype_logits", "volume_relation", "mask_attention"):
        value = outputs.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            num_rois = int(value.shape[1])
            break
    if not num_rois:
        return shuffled
    perm = torch.randperm(num_rois, device=device)
    roi_tensor_keys = [
        "surface_tokens",
        "volume_tokens",
        "local_tokens",
        "roi_tokens",
        "z_surface",
        "z_volume",
        "z_teacher",
        "surface_prototype_logits",
        "volume_prototype_logits",
        "prototype_logits",
        "mask_attention",
        "surface_roi_weights",
    ]
    for key in roi_tensor_keys:
        value = outputs.get(key)
        if torch.is_tensor(value) and value.dim() >= 2 and int(value.shape[1]) == num_rois:
            shuffled[key] = value[:, perm].contiguous()
    relation_keys = ["surface_relation", "volume_relation", "relation", "similarity_matrix"]
    for key in relation_keys:
        value = outputs.get(key)
        if torch.is_tensor(value) and value.dim() >= 3 and int(value.shape[1]) == num_rois and int(value.shape[2]) == num_rois:
            shuffled[key] = value[:, perm][:, :, perm].contiguous()
    cross_tokens = outputs.get("cross_image_tokens")
    if torch.is_tensor(cross_tokens) and cross_tokens.dim() >= 3 and int(cross_tokens.shape[1]) == num_rois + 1:
        shuffled["cross_image_tokens"] = torch.cat(
            [cross_tokens[:, :1], cross_tokens[:, 1:][:, perm]],
            dim=1,
        ).contiguous()
    return shuffled


def export_predictions(student, loader, device, csv_path, dry_run=False):
    student.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = student(batch["image"], clinical_data=batch["clinical_data"])
            probs = outputs["prob"].detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = batch["label"].detach().cpu().numpy()
            for idx, filename in enumerate(batch["filename"]):
                row = {"filename": filename, "label": int(labels[idx]), "pred": int(preds[idx])}
                for cls_idx in range(probs.shape[1]):
                    row[f"score_{cls_idx}"] = float(probs[idx, cls_idx])
                rows.append(row)
            if dry_run:
                break
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = Option(
        stage="student",
        default_paths=[
            os.path.join(ROOT, "configs", "data.yaml"),
            os.path.join(ROOT, "configs", "student.yaml"),
        ],
    ).parse()
    seed_everything(int(cfg.training.seed))
    device = torch.device(cfg.runtime.device if cfg.runtime.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg.model.num_classes = int(getattr(cfg.model, "num_classes", infer_num_classes(cfg.data.task)))
    expr_dir = cfg.expr_dir
    root_logger = Logger(expr_dir)
    root_logger.print_message(f"Student experiment dir: {expr_dir}")

    folds = [int(cfg.runtime.fold)] if getattr(cfg.runtime, "fold", None) is not None else list(range(int(cfg.training.num_folds)))
    results = []

    for fold in folds:
        fold_start_time = now()
        fold_dir = os.path.join(expr_dir, f"fold_{fold}")
        logger = Logger(fold_dir)
        train_loader, val_loader, test_loader = build_dataloaders(cfg, fold=fold, require_teacher_inputs=True)
        logger.print_message(
            f"Atlas mode={bool(getattr(getattr(cfg.data, 'atlas', {}), 'enabled', False))} "
            f"atlas_name={str(getattr(getattr(cfg.data, 'atlas', {}), 'name', 'none'))} "
            f"resolved_num_rois={int(cfg.model.num_rois)}"
        )
        logger.print_message(
            f"Teacher surface refiner={str(getattr(cfg.model, 'surface_refiner', 'none'))} "
            f"hidden={int(getattr(cfg.model, 'surface_refiner_hidden_dim', 0))} "
            f"dropout={float(getattr(cfg.model, 'surface_refiner_dropout', 0.0))} "
            f"residual_init={float(getattr(cfg.model, 'surface_refiner_residual_init', 0.1))}"
        )
        logger.print_message(
            f"Fold {fold} sizes: train={len(train_loader.dataset)} ({len(train_loader)} steps) "
            f"val={len(val_loader.dataset)} ({len(val_loader)} steps) "
            f"test={len(test_loader.dataset)} ({len(test_loader)} steps)"
        )
        logger.print_message(
            f"Clinical fusion enabled={bool(getattr(cfg.model, 'use_clinical', False))} "
            f"mode={str(getattr(cfg.model, 'clinical_fusion_mode', 'gated'))} "
            f"features={len(train_loader.dataset.clinical_features)} "
            f"input_dim={int(cfg.model.clinical_input_dim)}"
        )
        random_teacher_control = bool(getattr(cfg.training, "control_random_teacher", False))
        shuffle_roi_control = bool(getattr(cfg.training, "control_shuffle_roi", False))
        teacher_ckpt_path = "" if random_teacher_control else locate_teacher_checkpoint(cfg, fold)
        logger.print_message(f"Teacher checkpoint: {teacher_ckpt_path or '[random untrained teacher control]'}")
        logger.print_message(
            f"Control flags: random_teacher={random_teacher_control} shuffle_roi={shuffle_roi_control}"
        )
        teacher = build_teacher(cfg).to(device)
        if not random_teacher_control:
            teacher_state = torch.load(teacher_ckpt_path, map_location=device)
            teacher.load_state_dict(teacher_state["model"])
        for param in teacher.parameters():
            param.requires_grad = False

        student = build_student(cfg).to(device)
        initialize_student_from_teacher(student, teacher)
        logger.print_message(
            f"Initialized student volume encoder, {str(getattr(cfg.model, 'clinical_fusion_mode', 'gated'))} "
            "clinical fusion, global heads, volume projector, and prototypes from "
            f"{'random teacher control' if random_teacher_control else 'teacher'}."
        )
        logger.print_message(f"Student parameters: {count_parameters(student)}")

        warmup_epochs = max(0, min(int(cfg.training.warmup_epochs), int(cfg.training.epochs)))
        set_inherited_trainable(student, trainable=warmup_epochs == 0)
        optimizer = build_student_optimizer(student, cfg)
        logger.print_message(
            f"Student schedule: warmup_epochs={warmup_epochs} encoder_lr={float(cfg.training.encoder_lr):.2e} "
            f"inherited_lr={float(cfg.training.inherited_lr):.2e} head_lr={float(cfg.training.lr):.2e}"
        )
        ckpt = CheckpointIO(os.path.join(fold_dir, "best_student.pt"))
        class_weights = compute_class_weights(
            train_loader.dataset,
            num_classes=int(cfg.model.num_classes),
            mode=str(cfg.training.class_weight_mode),
        )
        logger.print_message(f"Class weights: {class_weights}")
        relation_weights = build_relation_distill_weights(train_loader.dataset, cfg, logger=logger)
        if relation_weights is None:
            logger.print_message("Relation distillation mode=global (original full-matrix MSE).")

        best_score = float("-inf")
        stale = 0
        fold_train_seconds = 0.0
        fold_val_seconds = 0.0
        for epoch in range(int(cfg.training.epochs)):
            warmup = epoch < warmup_epochs
            if epoch == warmup_epochs and warmup_epochs > 0:
                set_inherited_trainable(student, trainable=True)
                stale = 0
                logger.print_message("Warmup complete; unfroze inherited MRI modules for low-LR fine-tuning.")
            epoch_start_time = now()
            train_start_time = now()
            train_metrics, train_losses = run_epoch(
                student,
                teacher,
                train_loader,
                optimizer,
                device,
                cfg,
                class_weights,
                train_mode=True,
                warmup=warmup,
                relation_weights=relation_weights,
            )
            train_elapsed = now() - train_start_time
            val_start_time = now()
            val_metrics, val_losses = run_epoch(
                student,
                teacher,
                val_loader,
                optimizer=None,
                device=device,
                cfg=cfg,
                class_weights=class_weights,
                train_mode=False,
                relation_weights=relation_weights,
            )
            val_elapsed = now() - val_start_time
            epoch_elapsed = now() - epoch_start_time
            fold_train_seconds += train_elapsed
            fold_val_seconds += val_elapsed
            logger.print_message(
                f"Epoch {epoch + 1}/{cfg.training.epochs} "
                f"stage={'warmup' if warmup else 'finetune'} "
                f"train_loss={train_losses['loss']:.4f} val_loss={val_losses['loss']:.4f} "
                f"train_ce={train_losses['ce']:.4f} train_kd_logit={train_losses['kd_logit']:.4f} "
                f"train_kd_token={train_losses['kd_token']:.4f} train_kd_proto={train_losses['kd_prototype']:.4f} "
                f"roi_weight={float(torch.sigmoid(student.roi_logit_scale).detach()):.4f} "
                f"{format_classification_metrics(val_metrics, prefix='val_')} "
                f"train_time={format_duration(train_elapsed)} val_time={format_duration(val_elapsed)} "
                f"epoch_time={format_duration(epoch_elapsed)}"
            )
            score = selection_score(val_metrics, str(cfg.training.selection_metric))
            if score > best_score:
                best_score = score
                stale = 0
                ckpt.save(
                    {
                        "model": student.state_dict(),
                        "teacher_checkpoint": teacher_ckpt_path,
                        "fold": fold,
                        "epoch": epoch + 1,
                        "val_metrics": val_metrics,
                        "val_losses": val_losses,
                        "config": dict(cfg),
                        "clinical_features": list(train_loader.dataset.clinical_features),
                        "clinical_stats": train_loader.dataset.clinical_stats,
                        "relation_distill_mode": str(getattr(cfg.training, "relation_distill_mode", "global")),
                    }
                )
                logger.print_message("Saved new best student checkpoint.")
            else:
                stale += 1
                if not warmup and stale >= int(cfg.training.early_stop_patience):
                    logger.print_message(f"Early stopping at epoch {epoch + 1}.")
                    break

            if cfg.runtime.dry_run:
                break

        state = ckpt.load(map_location=device)
        student.load_state_dict(state["model"])
        test_start_time = now()
        test_metrics, test_losses = run_epoch(
            student,
            teacher,
            test_loader,
            optimizer=None,
            device=device,
            cfg=cfg,
            class_weights=class_weights,
            train_mode=False,
            relation_weights=relation_weights,
        )
        test_elapsed = now() - test_start_time
        fold_elapsed = now() - fold_start_time
        logger.print_message(
            f"Fold {fold} test: loss={test_losses['loss']:.4f} "
            f"{format_classification_metrics(test_metrics)} "
            f"train_total_time={format_duration(fold_train_seconds)} "
            f"val_total_time={format_duration(fold_val_seconds)} "
            f"test_time={format_duration(test_elapsed)} fold_total_time={format_duration(fold_elapsed)}"
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
            },
            "test_metrics.json",
        )
        export_predictions(
            student,
            test_loader,
            device,
            os.path.join(fold_dir, "test_predictions.csv"),
            dry_run=bool(cfg.runtime.dry_run),
        )
        results.append(test_metrics)

    if results:
        summary = summarize_metric_results(results)
        root_logger.save_json({"fold_metrics": results, "summary": summary}, "crossval_summary.json")
        root_logger.print_message(f"Final mean+/-std {format_metric_summary(summary)}")


if __name__ == "__main__":
    main()
