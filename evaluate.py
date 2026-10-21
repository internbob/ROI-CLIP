import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_config
from data import build_dataloaders
from metrics import compute_classification_metrics, format_classification_metrics
from models.student import MRIOnlyStudent
from models.teacher import CortexWeaveROIClipTeacher
from utils import Logger, move_to_device


def infer_num_classes(task):
    return 3 if str(task).upper() == "COG" else 2


def build_teacher(cfg):
    volume_pooling_cfg = getattr(cfg.model, "volume_pooling", {})
    ribbon_cfg = getattr(cfg.model, "ribbon", {})
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
        roi_residual_init=float(getattr(cfg.model, "roi_residual_init", 0.1)),
        use_roi_branch=bool(getattr(cfg.model, "use_roi_branch", True)),
        volume_norm=str(getattr(cfg.model, "volume_norm", "batch")),
        use_clinical=bool(getattr(cfg.model, "use_clinical", False)),
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
    )


def build_student(cfg):
    return MRIOnlyStudent(
        feature_dim=int(cfg.model.feature_dim),
        proj_dim=int(cfg.model.proj_dim),
        num_rois=int(cfg.model.num_rois),
        num_classes=int(cfg.model.num_classes),
        base_channels=int(cfg.model.base_channels),
        dropout=float(cfg.model.dropout),
        prototype_temperature=float(cfg.temperature.prototype),
        roi_residual_init=float(getattr(cfg.model, "roi_residual_init", 0.1)),
        volume_norm=str(getattr(cfg.model, "volume_norm", "batch")),
        use_clinical=bool(getattr(cfg.model, "student_use_clinical", False)),
        clinical_input_dim=int(getattr(cfg.model, "clinical_input_dim", 0)),
        clinical_fusion_mode=str(getattr(cfg.model, "clinical_fusion_mode", "gated")),
        clinical_hidden_dim=int(getattr(cfg.model, "clinical_hidden_dim", 256)),
        clinical_dropout=float(getattr(cfg.model, "clinical_dropout", 0.1)),
        clinical_num_heads=int(getattr(cfg.model, "clinical_num_heads", 4)),
        clinical_num_layers=int(getattr(cfg.model, "clinical_num_layers", 2)),
    )


def main():
    cfg = parse_config(
        [os.path.join(ROOT, "configs", "data.yaml"), os.path.join(ROOT, "configs", "student.yaml")],
        "Evaluate CortexWeave-ROIClip teacher or student",
    )
    cfg.model.num_classes = int(getattr(cfg.model, "num_classes", infer_num_classes(cfg.data.task)))
    device = torch.device(cfg.runtime.device if cfg.runtime.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = str(getattr(cfg.runtime, "checkpoint", ""))
    if not checkpoint:
        raise ValueError("Pass --checkpoint to evaluate.py.")

    model_type = "student" if "student" in os.path.basename(checkpoint).lower() else "teacher"
    require_teacher_inputs = model_type == "teacher"
    _, _, test_loader = build_dataloaders(cfg, fold=int(getattr(cfg.runtime, "fold", 0) or 0), require_teacher_inputs=require_teacher_inputs)

    if model_type == "teacher":
        model = build_teacher(cfg).to(device)
    else:
        model = build_student(cfg).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    save_dir = os.path.dirname(checkpoint)
    logger = Logger(save_dir, log_name="eval.log")
    rows = []
    probs_all = []
    labels_all = []
    payload_dir = os.path.join(save_dir, "eval_payloads")
    os.makedirs(payload_dir, exist_ok=True)
    saved_payloads = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = move_to_device(batch, device)
            if model_type == "teacher":
                model_inputs = dict(
                    image=batch["image"],
                    surface_data=batch["surface_data"],
                    surface_roi_masks=batch["surface_roi_masks"],
                    local_ribbon_masks=batch["local_ribbon_masks"],
                    roi_volume_masks=batch["roi_volume_masks"],
                    labels=batch["label"],
                    clinical_data=batch["clinical_data"],
                )
                if str(getattr(getattr(cfg.model, "volume_pooling", {}), "mode", "legacy_mask")).lower() == "ribbon":
                    model_inputs.update(
                        ribbon_grid=batch["ribbon_grid"],
                        ribbon_valid_mask=batch["ribbon_valid_mask"],
                        vertex_valid_mask=batch["vertex_valid_mask"],
                        vertex_roi_index=batch["vertex_roi_index"],
                    )
                outputs = model(**model_inputs)
            else:
                if bool(getattr(cfg.model, "student_use_clinical", False)):
                    outputs = model(batch["image"], clinical_data=batch["clinical_data"])
                else:
                    outputs = model(batch["image"])
            probs = outputs["prob"].detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = batch["label"].detach().cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.tolist())
            for idx, filename in enumerate(batch["filename"]):
                row = {"filename": filename, "label": int(labels[idx]), "pred": int(preds[idx])}
                for cls_idx in range(probs.shape[1]):
                    row[f"score_{cls_idx}"] = float(probs[idx, cls_idx])
                rows.append(row)
                if saved_payloads < 8:
                    payload = {
                        "filename": filename,
                        "label": int(labels[idx]),
                        "pred": int(preds[idx]),
                        "prob": torch.as_tensor(probs[idx]),
                        "attention_maps": (
                            outputs["attention_maps"][idx].detach().cpu()
                            if "attention_maps" in outputs
                            else outputs["mask_attention"][idx].detach().cpu()
                        ),
                    }
                    torch.save(payload, os.path.join(payload_dir, f"{filename.replace('/', '_')}.pt"))
                    saved_payloads += 1

    metrics = compute_classification_metrics(labels_all, probs_all)
    logger.print_message(
        f"Evaluation ({model_type}) {format_classification_metrics(metrics)}"
    )

    csv_path = os.path.join(save_dir, f"eval_{model_type}_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
