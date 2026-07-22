import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import build_dataloaders
from losses import classification_loss, prototype_loss, relation_consistency_loss, roi_clip_loss
from metrics import AverageMeter, compute_classification_metrics, format_classification_metrics
from models.teacher import CortexWeaveROIClipTeacher, MRIClinicalCLIPBaseline, SurfaceOnlyTransformerBaseline
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


def build_model(cfg):
    num_classes = int(getattr(cfg.model, "num_classes", infer_num_classes(cfg.data.task)))
    use_clinical = bool(getattr(cfg.model, "use_clinical", False))
    method = str(getattr(cfg.model, "method", "")).lower()
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


def run_epoch(model, loader, optimizer, device, cfg, class_weights, train_mode, auxiliary_loss_scale=1.0):
    model.train(train_mode)
    use_roi_branch = bool(cfg.model.use_roi_branch)
    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    clip_meter = AverageMeter()
    proto_meter = AverageMeter()
    rel_meter = AverageMeter()
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
            outputs = model(
                image=batch["image"],
                surface_data=batch["surface_data"] if use_roi_branch else None,
                surface_roi_masks=batch["surface_roi_masks"] if use_roi_branch else None,
                local_ribbon_masks=batch["local_ribbon_masks"] if use_roi_branch else None,
                roi_volume_masks=batch["roi_volume_masks"] if use_roi_branch else None,
                labels=batch["label"],
                clinical_data=batch["clinical_data"],
            )

            loss_ce = classification_loss(outputs["logits"], batch["label"], class_weights=class_weights)
            loss_clip = loss_ce.new_zeros(())
            loss_proto = loss_ce.new_zeros(())
            loss_rel = loss_ce.new_zeros(())
            if use_roi_branch and auxiliary_loss_scale > 0.0 and "z_surface" in outputs and "z_volume" in outputs:
                loss_clip = roi_clip_loss(outputs["z_surface"], outputs["z_volume"], temperature=cfg.temperature.clip)
                loss_proto = (
                    prototype_loss(outputs["surface_prototype_logits"], batch["label"])
                    + prototype_loss(outputs["volume_prototype_logits"], batch["label"])
                    + prototype_loss(outputs["prototype_logits"], batch["label"])
                ) / 3.0
                loss_rel = relation_consistency_loss(outputs["surface_relation"], outputs["volume_relation"])
            elif auxiliary_loss_scale > 0.0 and "z_mri" in outputs and "z_clinical" in outputs:
                loss_clip = roi_clip_loss(outputs["z_mri"], outputs["z_clinical"], temperature=cfg.temperature.clip)
            total_loss = (
                float(cfg.loss_weights.ce) * loss_ce
                + auxiliary_loss_scale * float(cfg.loss_weights.roi_clip) * loss_clip
                + auxiliary_loss_scale * float(cfg.loss_weights.prototype) * loss_proto
                + auxiliary_loss_scale * float(cfg.loss_weights.relation) * loss_rel
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
            all_labels.extend(batch["label"].detach().cpu().tolist())
            all_probs.extend(outputs["prob"].detach().cpu().tolist())

            if cfg.runtime.dry_run:
                break

    metrics = compute_classification_metrics(all_labels, all_probs)
    losses = {
        "loss": total_meter.avg,
        "ce": ce_meter.avg,
        "clip": clip_meter.avg,
        "proto": proto_meter.avg,
        "rel": rel_meter.avg,
    }
    return metrics, losses


def evaluate_teacher(model, loader, device, cfg):
    metrics, losses = run_epoch(model, loader, optimizer=None, device=device, cfg=cfg, class_weights=None, train_mode=False)
    return metrics, losses


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
            f"test={len(test_loader.dataset)} ({len(test_loader)} steps)"
        )
        class_weights = compute_class_weights(
            train_loader.dataset,
            num_classes=int(cfg.model.num_classes),
            mode=str(cfg.training.class_weight_mode),
        )
        logger.print_message(f"Class weights: {class_weights}")

        model = build_model(cfg).to(device)
        logger.print_message(f"Teacher parameters: {count_parameters(model)}")
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
            )
            val_elapsed = now() - val_start_time
            epoch_elapsed = now() - epoch_start_time
            fold_train_seconds += train_elapsed
            fold_val_seconds += val_elapsed

            logger.print_message(
                f"Epoch {epoch + 1}/{cfg.training.epochs} "
                f"train_loss={train_losses['loss']:.4f} val_loss={val_losses['loss']:.4f} "
                f"train_ce={train_losses['ce']:.4f} train_clip={train_losses['clip']:.4f} "
                f"aux_scale={auxiliary_loss_scale:.1f} "
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
        test_start_time = now()
        test_metrics, test_losses = evaluate_teacher(model, test_loader, device, cfg)
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
        results.append(test_metrics)

    if results:
        summary = summarize_metric_results(results)
        root_logger.save_json({"fold_metrics": results, "summary": summary}, "crossval_summary.json")
        root_logger.print_message(f"Final mean+/-std {format_metric_summary(summary)}")


if __name__ == "__main__":
    main()
