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
    prototype_distill_loss,
    relation_distill_loss,
    token_distill_loss,
)
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


def run_epoch(student, teacher, loader, optimizer, device, cfg, class_weights, train_mode, warmup=False):
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
        teacher_ckpt_path = locate_teacher_checkpoint(cfg, fold)
        logger.print_message(f"Teacher checkpoint: {teacher_ckpt_path}")

        teacher = build_teacher(cfg).to(device)
        teacher_state = torch.load(teacher_ckpt_path, map_location=device)
        teacher.load_state_dict(teacher_state["model"])
        for param in teacher.parameters():
            param.requires_grad = False

        student = build_student(cfg).to(device)
        initialize_student_from_teacher(student, teacher)
        logger.print_message(
            f"Initialized student volume encoder, {str(getattr(cfg.model, 'clinical_fusion_mode', 'gated'))} "
            "clinical fusion, global heads, volume projector, and prototypes from teacher."
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
            )
            train_elapsed = now() - train_start_time
            val_start_time = now()
            val_metrics, val_losses = run_epoch(student, teacher, val_loader, optimizer=None, device=device, cfg=cfg, class_weights=class_weights, train_mode=False)
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
        test_metrics, test_losses = run_epoch(student, teacher, test_loader, optimizer=None, device=device, cfg=cfg, class_weights=class_weights, train_mode=False)
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
