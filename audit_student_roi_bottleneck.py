"""Inference-only audit of the global, ROI, and graph Student branches.

This script only evaluates development-validation subjects.  It deliberately
constructs no held-out test dataset and performs no optimization.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import load_configs
from data import build_dataloaders
from metrics import compute_classification_metrics
from train_student import build_student
from utils import move_to_device, seed_everything


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "vnext_roi_bottleneck_ablation_20261021"
CHECKPOINT_ROOTS = {
    "s3": ROOT / "outputs" / "vnext_graph_student_20261019" / "s3_logit_kd_corrected",
    "s4": ROOT / "outputs" / "vnext_graph_student_20261019" / "s4_roi_kd_corrected",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def make_config(output_dir, num_workers):
    cfg = load_configs(
        [
            ROOT / "configs" / "data.yaml",
            ROOT / "configs" / "student.yaml",
            ROOT / "configs" / "runtime_vnext_graph_student_locked_s4.yaml",
        ]
    )
    cfg.runtime.evaluate_test = False
    cfg.runtime.output_dir = str(output_dir)
    cfg.runtime.dry_run = False
    cfg.training.num_workers = int(num_workers)
    cfg.model.num_classes = 3
    return cfg


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return state


def off_diagonal_cosine(tokens):
    tokens = F.normalize(tokens.float(), dim=-1)
    similarity = tokens @ tokens.transpose(-1, -2)
    count = similarity.shape[-1]
    mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    return similarity[:, mask].mean().item()


def token_variance(tokens):
    return tokens.float().var(dim=1, unbiased=False).mean().item()


def evaluate_s4(model, loader, device):
    labels = []
    filenames = []
    probabilities = {name: [] for name in ("d0_fused", "d1_global", "d2_roi", "d3_pregraph")}
    diagnostics = {
        "pregraph_offdiag_cosine": [],
        "postgraph_offdiag_cosine": [],
        "pregraph_token_variance": [],
        "postgraph_token_variance": [],
        "graph_delta_relative_norm": [],
        "global_logit_norm": [],
        "roi_logit_norm": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch["image"])
            pregraph_roi_logits = model.classifier(outputs["z_student_base"])
            logits = {
                "d0_fused": outputs["logits"],
                "d1_global": outputs["global_logits"],
                "d2_roi": outputs["roi_logits"],
                "d3_pregraph": outputs["global_logits"] + outputs["roi_weight"] * pregraph_roi_logits,
            }
            for name, value in logits.items():
                probabilities[name].extend(torch.softmax(value, dim=-1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
            filenames.extend(batch["filename"])

            pre = outputs["z_student_base"]
            post = outputs["z_student_graph"]
            diagnostics["pregraph_offdiag_cosine"].append(off_diagonal_cosine(pre))
            diagnostics["postgraph_offdiag_cosine"].append(off_diagonal_cosine(post))
            diagnostics["pregraph_token_variance"].append(token_variance(pre))
            diagnostics["postgraph_token_variance"].append(token_variance(post))
            diagnostics["graph_delta_relative_norm"].append(
                ((post - pre).norm(dim=-1) / pre.norm(dim=-1).clamp_min(1e-8)).mean().item()
            )
            diagnostics["global_logit_norm"].append(outputs["global_logits"].norm(dim=-1).mean().item())
            diagnostics["roi_logit_norm"].append(outputs["roi_logits"].norm(dim=-1).mean().item())

    metrics = {
        name: compute_classification_metrics(labels, np.asarray(probs, dtype=np.float32))
        for name, probs in probabilities.items()
    }
    return {
        "labels": labels,
        "filenames": filenames,
        "probabilities": probabilities,
        "metrics": metrics,
        "diagnostics": {name: float(np.mean(values)) for name, values in diagnostics.items()},
    }


def evaluate_global(model, loader, device):
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch["image"])
            labels.extend(batch["label"].cpu().tolist())
            probabilities.extend(torch.softmax(outputs["global_logits"], dim=-1).cpu().tolist())
    return compute_classification_metrics(labels, np.asarray(probabilities, dtype=np.float32))


def validate_against_saved(fold, result):
    path = CHECKPOINT_ROOTS["s4"] / f"fold_{fold}" / "validation_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    saved_names = [row["filename"] for row in rows]
    saved_labels = [int(row["label"]) for row in rows]
    saved_probs = np.asarray(
        [[float(row[f"score_{index}"]) for index in range(3)] for row in rows], dtype=np.float32
    )
    current = np.asarray(result["probabilities"]["d0_fused"], dtype=np.float32)
    if saved_names != result["filenames"] or saved_labels != result["labels"]:
        raise RuntimeError(f"Fold {fold}: validation subject order does not match the saved run")
    max_delta = float(np.max(np.abs(saved_probs - current)))
    if max_delta > 1e-5:
        raise RuntimeError(f"Fold {fold}: current fused probabilities differ by {max_delta:.3g}")
    return max_delta


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args.output_dir, args.num_workers)
    seed_everything(int(cfg.training.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fold_records = []

    for fold in range(3):
        _, val_loader, test_loader = build_dataloaders(
            cfg,
            fold=fold,
            require_teacher_inputs=False,
            filter_teacher_available=True,
        )
        if test_loader is not None:
            raise RuntimeError("Held-out test loader must not be constructed during this audit")

        s4_model = build_student(cfg).to(device)
        s4_state = load_checkpoint(
            s4_model, CHECKPOINT_ROOTS["s4"] / f"fold_{fold}" / "best_student.pt", device
        )
        s4_result = evaluate_s4(s4_model, val_loader, device)
        reproduction_delta = validate_against_saved(fold, s4_result)

        s3_model = build_student(cfg).to(device)
        s3_state = load_checkpoint(
            s3_model, CHECKPOINT_ROOTS["s3"] / f"fold_{fold}" / "best_student.pt", device
        )
        s3_global_metrics = evaluate_global(s3_model, val_loader, device)

        roi_weight = float(torch.sigmoid(s4_model.roi_logit_scale).cpu())
        graph_gate = float(
            torch.sigmoid(s4_model.roi_graph.layers[0].residual_gate_logit).cpu()
        )
        record = {
            "fold": fold,
            "held_out_test_evaluated": False,
            "s4_best_epoch": int(s4_state["epoch"]),
            "s3_best_epoch": int(s3_state["epoch"]),
            "saved_prediction_max_abs_delta": reproduction_delta,
            "roi_logit_weight": roi_weight,
            "graph_residual_gate": graph_gate,
            "metrics": s4_result["metrics"],
            "s3_global_metrics": s3_global_metrics,
            "diagnostics": s4_result["diagnostics"],
        }
        fold_records.append(record)
        print(
            f"fold={fold} fused_mcc={record['metrics']['d0_fused']['mcc']:.5f} "
            f"global_mcc={record['metrics']['d1_global']['mcc']:.5f} "
            f"pregraph_mcc={record['metrics']['d3_pregraph']['mcc']:.5f} "
            f"s3_global_mcc={record['s3_global_metrics']['mcc']:.5f}"
        )

    variants = {
        "D0 current fused": lambda row: row["metrics"]["d0_fused"],
        "D1 S4 global-only": lambda row: row["metrics"]["d1_global"],
        "D2 S4 ROI-only": lambda row: row["metrics"]["d2_roi"],
        "D3 global+pregraph ROI": lambda row: row["metrics"]["d3_pregraph"],
        "D4 S3 global-only": lambda row: row["s3_global_metrics"],
    }
    summary = {}
    for name, accessor in variants.items():
        summary[name] = {}
        for metric in ("mcc", "acc", "auc", "f1_macro", "sen", "spe", "balanced_acc"):
            summary[name][metric] = dict(
                zip(("mean", "sd"), mean_sd([accessor(row)[metric] for row in fold_records]))
            )

    output = {
        "protocol": {
            "training_performed": False,
            "held_out_test_loaded": False,
            "folds": [0, 1, 2],
            "s3_checkpoint_root": str(CHECKPOINT_ROOTS["s3"]),
            "s4_checkpoint_root": str(CHECKPOINT_ROOTS["s4"]),
        },
        "folds": fold_records,
        "summary": summary,
    }
    output_path = args.output_dir / "branch_audit.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
