#!/usr/bin/env python3
"""Development-only E7/E8 missingness audit for the unified E4 Student."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from config import load_configs
from data import build_dataloaders
from metrics import compute_classification_metrics
from train_student import build_student
from utils import move_to_device, seed_everything


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = [
    ROOT / "configs/data.yaml",
    ROOT / "configs/student.yaml",
    ROOT / "configs/runtime_unified_clinical_common.yaml",
    ROOT / "configs/runtime_unified_student_e4_full_kd.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--rates", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--checkpoint-template",
        default=str(
            ROOT
            / "outputs/vnext_unified_clinical_20261023/e4_student/fold_{fold}/best_student.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/vnext_unified_clinical_20261023/missingness_audit",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def mask_observed_features(clinical_data, rate, seed):
    values, missing = clinical_data.chunk(2, dim=-1)
    if rate <= 0.0:
        return clinical_data
    if rate >= 1.0:
        return torch.cat([torch.zeros_like(values), torch.ones_like(missing)], dim=-1)
    generator = torch.Generator(device=clinical_data.device)
    generator.manual_seed(int(seed))
    sampled = torch.rand(values.shape, device=values.device, generator=generator) < float(rate)
    newly_missing = sampled & (missing < 0.5)
    masked_values = values.masked_fill(newly_missing, 0.0)
    masked_flags = torch.maximum(missing, newly_missing.to(missing.dtype))
    return torch.cat([masked_values, masked_flags], dim=-1)


def metric_row(fold, rate, labels, probabilities):
    metrics = compute_classification_metrics(labels, probabilities)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    mci_recall = float(matrix[1, 1] / max(1, matrix[1].sum()))
    return {
        "fold": int(fold),
        "synthetic_missing_rate": float(rate),
        "n": int(len(labels)),
        "mcc": float(metrics["mcc"]),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "f1_macro": float(metrics["f1_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "specificity_macro": float(metrics["spe"]),
        "mci_recall": mci_recall,
    }


def summarize(rows):
    result = []
    for rate in sorted({float(row["synthetic_missing_rate"]) for row in rows}):
        selected = [row for row in rows if float(row["synthetic_missing_rate"]) == rate]
        record = {"synthetic_missing_rate": rate, "folds": len(selected)}
        for key in (
            "mcc",
            "auc",
            "acc",
            "f1_macro",
            "recall_macro",
            "specificity_macro",
            "mci_recall",
        ):
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(record)
    return result


def stratum_name(observed_fraction):
    if observed_fraction >= 1.0 - 1e-8:
        return "complete"
    if observed_fraction >= 0.75:
        return "75_to_lt100pct"
    if observed_fraction >= 0.50:
        return "50_to_lt75pct"
    if observed_fraction > 0.0:
        return "gt0_to_lt50pct"
    return "all_missing"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value):
    return "NA" if value is None else f"{float(value):.5f}"


def real_stratum_metric_row(scope, name, labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    metrics = compute_classification_metrics(labels, probabilities)
    classes_present = int(len(np.unique(labels)))
    return {
        "scope": scope,
        "stratum": name,
        "n": int(len(labels)),
        "classes_present": classes_present,
        "reportable": bool(len(labels) >= 30 and classes_present == 3),
        "mcc": float(metrics["mcc"]),
        "auc": float(metrics["auc"]) if classes_present == 3 else None,
        "acc": float(metrics["acc"]),
        "f1_macro": float(metrics["f1_macro"]),
    }


def report(summary_rows, stratum_rows, pooled_stratum_rows):
    lines = [
        "# CortexBridge Unified Clinical Missingness Audit",
        "",
        "Development folds only; held-out test accessed: `false`.",
        "",
        "## E7 synthetic masking",
        "",
        "| Missing | MCC | AUC | ACC | Macro-F1 | MCI recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {100*row['synthetic_missing_rate']:.0f}% | "
            f"{row['mcc_mean']:.5f} ± {row['mcc_sd']:.5f} | "
            f"{row['auc_mean']:.5f} ± {row['auc_sd']:.5f} | "
            f"{row['acc_mean']:.5f} ± {row['acc_sd']:.5f} | "
            f"{row['f1_macro_mean']:.5f} ± {row['f1_macro_sd']:.5f} | "
            f"{row['mci_recall_mean']:.5f} ± {row['mci_recall_sd']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## E8 real missingness strata",
            "",
            "| Scope | Observed fraction stratum | N | Classes | Reportable | MCC | AUC | ACC | Macro-F1 |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in [*pooled_stratum_rows, *stratum_rows]:
        lines.append(
            f"| {row['scope']} | {row['stratum']} | {row['n']} | "
            f"{row['classes_present']} | {str(row['reportable']).lower()} | "
            f"{format_metric(row['mcc'])} | {format_metric(row['auc'])} | "
            f"{format_metric(row['acc'])} | {format_metric(row['f1_macro'])} |"
        )
    lines.extend(
        [
            "",
            "Synthetic masks are applied only to originally observed variables. Existing missing values remain missing. A real-missingness stratum is marked reportable only when pooled/fold N >= 30 and all three classes are present; AUC is `NA` when fewer than three classes are present. Tiny strata are retained for provenance but must not be interpreted. At 100% synthetic missingness the model predicts MCI for every subject, so MCI recall=1.0 reflects single-class collapse, not useful sensitivity. No model is retrained and no test split is loaded.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    rates = [float(value) for value in args.rates.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rate_rows = []
    stratum_rows = []
    all_predictions = []
    for fold in folds:
        seed_everything(args.seed + fold)
        cfg = load_configs([str(path) for path in DEFAULT_CONFIGS])
        cfg.runtime.evaluate_test = False
        cfg.runtime.dry_run = False
        cfg.model.num_classes = 3
        _, val_loader, test_loader = build_dataloaders(
            cfg,
            fold=fold,
            require_teacher_inputs=False,
            filter_teacher_available=True,
        )
        if test_loader is not None:
            raise RuntimeError("held-out test loader must remain disabled")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        student = build_student(cfg).to(device)
        checkpoint_path = Path(args.checkpoint_template.format(fold=fold))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["model"])
        student.eval()
        by_rate = {rate: {"labels": [], "probabilities": []} for rate in rates}
        observed_fractions = []
        filenames = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = move_to_device(batch, device)
                _, original_missing = batch["clinical_data"].chunk(2, dim=-1)
                observed = 1.0 - original_missing.mean(dim=-1)
                observed_fractions.extend(observed.detach().cpu().tolist())
                filenames.extend([str(value) for value in batch["filename"]])
                labels = batch["label"].detach().cpu().tolist()
                for rate in rates:
                    clinical = mask_observed_features(
                        batch["clinical_data"],
                        rate,
                        seed=args.seed + fold * 100000 + batch_idx * 101 + int(rate * 1000),
                    )
                    probabilities = student(batch["image"], clinical_data=clinical)["prob"]
                    by_rate[rate]["labels"].extend(labels)
                    by_rate[rate]["probabilities"].extend(probabilities.detach().cpu().tolist())
        for rate in rates:
            labels = np.asarray(by_rate[rate]["labels"], dtype=np.int64)
            probabilities = np.asarray(by_rate[rate]["probabilities"], dtype=np.float64)
            rate_rows.append(metric_row(fold, rate, labels, probabilities))
            for index, filename in enumerate(filenames):
                all_predictions.append(
                    {
                        "fold": fold,
                        "filename": filename,
                        "label": int(labels[index]),
                        "synthetic_missing_rate": rate,
                        "observed_fraction_original": float(observed_fractions[index]),
                        "prob_0": float(probabilities[index, 0]),
                        "prob_1": float(probabilities[index, 1]),
                        "prob_2": float(probabilities[index, 2]),
                    }
                )
        base_labels = np.asarray(by_rate[0.0]["labels"], dtype=np.int64)
        base_probs = np.asarray(by_rate[0.0]["probabilities"], dtype=np.float64)
        names = np.asarray([stratum_name(value) for value in observed_fractions], dtype=object)
        for name in sorted(set(names.tolist())):
            take = np.flatnonzero(names == name)
            stratum_rows.append(
                real_stratum_metric_row(
                    f"fold_{fold}", name, base_labels[take], base_probs[take]
                )
            )
    summary_rows = summarize(rate_rows)
    base_predictions = [
        row for row in all_predictions if float(row["synthetic_missing_rate"]) == 0.0
    ]
    pooled_stratum_rows = []
    pooled_names = sorted(
        {stratum_name(float(row["observed_fraction_original"])) for row in base_predictions}
    )
    for name in pooled_names:
        selected = [
            row
            for row in base_predictions
            if stratum_name(float(row["observed_fraction_original"])) == name
        ]
        pooled_stratum_rows.append(
            real_stratum_metric_row(
                "pooled",
                name,
                [row["label"] for row in selected],
                [[row["prob_0"], row["prob_1"], row["prob_2"]] for row in selected],
            )
        )
    write_csv(args.output_dir / "missingness_metrics_by_fold.csv", rate_rows)
    write_csv(args.output_dir / "missingness_metrics_summary.csv", summary_rows)
    write_csv(args.output_dir / "real_missingness_strata.csv", stratum_rows)
    write_csv(args.output_dir / "real_missingness_strata_pooled.csv", pooled_stratum_rows)
    write_csv(args.output_dir / "missingness_predictions.csv", all_predictions)
    payload = {
        "folds": folds,
        "rates": rates,
        "held_out_test_accessed": False,
        "retrained": False,
        "summary": summary_rows,
        "real_missingness_strata": stratum_rows,
        "real_missingness_strata_pooled": pooled_stratum_rows,
    }
    (args.output_dir / "missingness_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "MISSINGNESS_AUDIT.md").write_text(
        report(summary_rows, stratum_rows, pooled_stratum_rows), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "held_out_test_accessed": False}))


if __name__ == "__main__":
    main()
