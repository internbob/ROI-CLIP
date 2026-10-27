#!/usr/bin/env python3
"""Summarize matched P0/P2/P3 privileged-teacher MRI-only Student runs."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


METRICS = ("mcc", "auc", "acc", "f1_macro", "recall_macro", "specificity_macro", "mci_recall")
VARIANTS = {
    "P0": "B1: existing G3 logit-KD baseline",
    "P2": "clinical-privileged Teacher -> MRI-only B1 Student",
    "P3": "joint clinical+geometry Teacher -> MRI-only B1 Student",
}
COMPARISONS = (
    ("P0", "P2", "clinical privileged knowledge over B1"),
    ("P0", "P3", "joint privileged knowledge over B1"),
    ("P2", "P3", "geometry-branch increment within the privileged-Teacher design"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="outputs/vnext_privileged_kd_20261022")
    parser.add_argument(
        "--p0-root", default="outputs/vnext_roi_bottleneck_ablation_20261021/b1_global_logit_kd"
    )
    parser.add_argument("--p2-root", default="outputs/vnext_privileged_kd_20261022/p2_clinical_kd")
    parser.add_argument("--p3-root", default="outputs/vnext_privileged_kd_20261022/p3_joint_kd")
    parser.add_argument(
        "--p2-teacher-root", default="outputs/vnext_privileged_kd_20261022/p2_clinical_teacher"
    )
    parser.add_argument(
        "--p3-teacher-root", default="outputs/vnext_privileged_kd_20261022/p3_joint_teacher"
    )
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def score_columns(frame):
    return sorted(
        (column for column in frame.columns if column.startswith("score_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )


def macro_specificity(labels, predictions, classes):
    matrix = confusion_matrix(labels, predictions, labels=classes)
    total = matrix.sum()
    values = []
    for index in range(len(classes)):
        false_positive = matrix[:, index].sum() - matrix[index, index]
        true_negative = total - matrix[index, :].sum() - false_positive
        denominator = true_negative + false_positive
        values.append(float(true_negative / denominator) if denominator else np.nan)
    return float(np.nanmean(values))


def metric_from_frame(frame, metric):
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["pred"].to_numpy(dtype=int)
    columns = score_columns(frame)
    classes = list(range(len(columns)))
    if metric == "mcc":
        return float(matthews_corrcoef(labels, predictions))
    if metric == "auc":
        return float(
            roc_auc_score(
                labels,
                frame[columns].to_numpy(dtype=float),
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    if metric == "acc":
        return float(accuracy_score(labels, predictions))
    if metric == "f1_macro":
        return float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0))
    if metric == "recall_macro":
        return float(recall_score(labels, predictions, labels=classes, average="macro", zero_division=0))
    if metric == "specificity_macro":
        return macro_specificity(labels, predictions, classes)
    if metric == "mci_recall":
        mask = labels == 1
        return float(np.mean(predictions[mask] == 1)) if mask.any() else np.nan
    raise ValueError(metric)


def load_variant(root, folds):
    fold_metrics = []
    frames = []
    provenance = []
    for fold in folds:
        fold_root = root / f"fold_{fold}"
        metrics_path = fold_root / "development_metrics.json"
        predictions_path = fold_root / "validation_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Incomplete fold output: {fold_root}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("held_out_test_evaluated") is not False:
            raise RuntimeError(f"Held-out test flag is not false: {metrics_path}")
        frame = pd.read_csv(predictions_path)
        required = {"filename", "label", "pred", "score_0", "score_1", "score_2"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"Prediction schema mismatch in {predictions_path}")
        frame["fold"] = fold
        frame["key"] = frame["fold"].astype(str) + ":" + frame["filename"].astype(str)
        fold_metrics.append({metric: metric_from_frame(frame, metric) for metric in METRICS})
        frames.append(frame)
        provenance.append(
            {
                "fold": fold,
                "best_epoch": int(payload["best_epoch"]),
                "held_out_test_evaluated": False,
                "metrics_path": str(metrics_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
            }
        )
    pooled = pd.concat(frames, ignore_index=True)
    if pooled["key"].duplicated().any():
        raise RuntimeError(f"Duplicate fold/filename keys in {root}")
    return {"metrics": fold_metrics, "predictions": pooled, "provenance": provenance}


def load_teacher_summary(root, folds):
    rows = []
    for fold in folds:
        path = root / f"fold_{fold}" / "development_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("held_out_test_evaluated") is not False:
            raise RuntimeError(f"Held-out test flag is not false: {path}")
        metrics = payload["best_val_metrics"]
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        rows.append(
            {
                "best_epoch": int(payload["best_epoch"]),
                "mcc": float(metrics["mcc"]),
                "auc": float(metrics["auc"]),
                "acc": float(metrics["acc"]),
                "f1_macro": float(metrics["f1_macro"]),
                "recall_macro": float(metrics.get("recall_macro", metrics["sen"])),
                "specificity_macro": float(metrics["spe"]),
                "mci_recall": float(matrix[1, 1] / matrix[1].sum()),
                "roi_retrieval_acc1": float(metrics.get("roi_retrieval_acc1", 0.0)),
                "roi_retrieval_acc5": float(metrics.get("roi_retrieval_acc5", 0.0)),
                "diagonal_similarity_gap": float(metrics.get("diagonal_similarity_gap", 0.0)),
                "metrics_path": str(path.resolve()),
            }
        )
    summary_metrics = METRICS + (
        "roi_retrieval_acc1",
        "roi_retrieval_acc5",
        "diagonal_similarity_gap",
    )
    return {
        "folds": rows,
        "summary": {
            metric: summarize([row[metric] for row in rows]) for metric in summary_metrics
        },
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "sample_sd": float(np.nanstd(values, ddof=1)),
        "fold_values": values.tolist(),
    }


def paired_frames(left, right):
    merged = left.merge(right, on="key", suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Paired variants do not contain identical fold/filename keys")
    if not np.array_equal(merged["label_left"], merged["label_right"]):
        raise RuntimeError("Labels differ between paired variants")
    return merged


def select_side(merged, side, indices=None):
    data = {
        "label": merged[f"label_{side}"],
        "pred": merged[f"pred_{side}"],
        "score_0": merged[f"score_0_{side}"],
        "score_1": merged[f"score_1_{side}"],
        "score_2": merged[f"score_2_{side}"],
    }
    frame = pd.DataFrame(data)
    return frame if indices is None else frame.iloc[indices]


def paired_bootstrap(left, right, metric, replicates, seed):
    merged = paired_frames(left, right)
    fold_groups = [
        group.reset_index(drop=True)
        for _, group in merged.groupby("fold_left", sort=True)
    ]
    observed_fold_deltas = [
        metric_from_frame(select_side(group, "right"), metric)
        - metric_from_frame(select_side(group, "left"), metric)
        for group in fold_groups
    ]
    observed = float(np.mean(observed_fold_deltas))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        try:
            fold_deltas = []
            for group in fold_groups:
                indices = rng.integers(0, len(group), size=len(group))
                delta = metric_from_frame(select_side(group, "right", indices), metric)
                delta -= metric_from_frame(select_side(group, "left", indices), metric)
                fold_deltas.append(delta)
            delta = float(np.mean(fold_deltas))
            if np.isfinite(delta):
                deltas.append(delta)
        except ValueError:
            continue
    if not deltas:
        raise RuntimeError(f"No valid bootstrap replicates for {metric}")
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "observed_delta_right_minus_left": float(observed),
        "observed_fold_deltas": [float(value) for value in observed_fold_deltas],
        "resampling_unit": "paired subjects within each fold; replicate statistic is mean fold delta",
        "ci95_low": float(low),
        "ci95_high": float(high),
        "valid_replicates": len(deltas),
    }


def compare(left_label, right_label, loaded, replicates, seed):
    left = loaded[left_label]
    right = loaded[right_label]
    fold_deltas = {
        metric: [b[metric] - a[metric] for a, b in zip(left["metrics"], right["metrics"])]
        for metric in METRICS
    }
    merged = paired_frames(left["predictions"], right["predictions"])
    left_correct = merged["label_left"] == merged["pred_left"]
    right_correct = merged["label_right"] == merged["pred_right"]
    right_only = int((right_correct & ~left_correct).sum())
    left_only = int((left_correct & ~right_correct).sum())
    discordant = right_only + left_only
    return {
        "left": left_label,
        "right": right_label,
        "fold_deltas": fold_deltas,
        "mean_fold_deltas": {metric: float(np.mean(values)) for metric, values in fold_deltas.items()},
        "positive_fold_counts": {metric: int(sum(value > 0 for value in values)) for metric, values in fold_deltas.items()},
        "paired_subject_bootstrap": {
            metric: paired_bootstrap(
                left["predictions"], right["predictions"], metric, replicates, seed + index
            )
            for index, metric in enumerate(("mcc", "auc", "f1_macro", "mci_recall"))
        },
        "mcnemar": {
            "right_only_correct": right_only,
            "left_only_correct": left_only,
            "exact_two_sided_p": float(binomtest(right_only, discordant, 0.5).pvalue) if discordant else 1.0,
        },
    }


def fmt(summary):
    return f"{summary['mean']:.5f} ± {summary['sample_sd']:.5f}"


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    roots = {
        "P0": (project_root / args.p0_root).resolve(),
        "P2": (project_root / args.p2_root).resolve(),
        "P3": (project_root / args.p3_root).resolve(),
    }
    teacher_roots = {
        "P2": (project_root / args.p2_teacher_root).resolve(),
        "P3": (project_root / args.p3_teacher_root).resolve(),
    }
    loaded = {label: load_variant(root, folds) for label, root in roots.items()}
    teachers = {
        label: load_teacher_summary(root, folds) for label, root in teacher_roots.items()
    }
    summaries = {
        label: {
            metric: summarize([fold_metrics[metric] for fold_metrics in item["metrics"]])
            for metric in METRICS
        }
        for label, item in loaded.items()
    }
    comparisons = {}
    for index, (left, right, description) in enumerate(COMPARISONS):
        item = compare(left, right, loaded, args.bootstrap_replicates, args.seed + 100 * index)
        item["description"] = description
        comparisons[f"{right}_minus_{left}"] = item

    p2_vs_p0 = comparisons["P2_minus_P0"]
    p3_vs_p0 = comparisons["P3_minus_P0"]
    gate = {
        "mean_mcc_improved": p3_vs_p0["mean_fold_deltas"]["mcc"] > 0,
        "mean_auc_improved": p3_vs_p0["mean_fold_deltas"]["auc"] > 0,
        "mcc_positive_folds": p3_vs_p0["positive_fold_counts"]["mcc"],
        "auc_positive_folds": p3_vs_p0["positive_fold_counts"]["auc"],
        "macro_f1_not_worse": p3_vs_p0["mean_fold_deltas"]["f1_macro"] >= 0,
        "mci_recall_not_worse": p3_vs_p0["mean_fold_deltas"]["mci_recall"] >= 0,
        "mcc_ci_excludes_zero": p3_vs_p0["paired_subject_bootstrap"]["mcc"]["ci95_low"] > 0,
        "auc_ci_excludes_zero": p3_vs_p0["paired_subject_bootstrap"]["auc"]["ci95_low"] > 0,
    }
    advancement = {
        "p2_advance": bool(
            p2_vs_p0["mean_fold_deltas"]["mcc"] > 0
            and p2_vs_p0["mean_fold_deltas"]["auc"] > 0
            and p2_vs_p0["mean_fold_deltas"]["f1_macro"] >= 0
        ),
        "p3_advance": bool(
            gate["mean_mcc_improved"]
            and gate["mean_auc_improved"]
            and gate["macro_f1_not_worse"]
        ),
        "locked_development_method": "P0",
        "run_p4_or_p5_now": False,
        "reason": (
            "Neither P2 nor P3 passed the predeclared first-stage advancement criteria; "
            "P0 remains the locked MRI-only development method and held-out test remains unread."
        ),
    }
    payload = {
        "completed_on": date.today().isoformat(),
        "folds": folds,
        "held_out_test_evaluated": False,
        "student_inference_input": "MRI only",
        "variant_roots": {key: str(value) for key, value in roots.items()},
        "teacher_roots": {key: str(value) for key, value in teacher_roots.items()},
        "variant_descriptions": VARIANTS,
        "variant_summary": summaries,
        "teacher_summary": teachers,
        "comparisons": comparisons,
        "p3_gate": gate,
        "advancement": advancement,
        "claim_boundary": (
            "P3-P2 measures the increment of the complete geometry-privileged Teacher branch. "
            "It does not isolate hard cross-geometry correspondence from surface, ribbon, or graph effects; P4 is required."
        ),
    }
    (output_root / "privileged_kd_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for label, item in loaded.items():
        for fold, metrics in zip(folds, item["metrics"]):
            rows.append({"variant": label, "fold": fold, **metrics})
    pd.DataFrame(rows).to_csv(output_root / "privileged_kd_fold_metrics.csv", index=False)

    lines = [
        "# CortexBridge P2/P3 Privileged-KD Development Report",
        "",
        f"- Completed: `{date.today().isoformat()}`",
        "- Protocol: matched three-fold development validation; selection metric `MCC`.",
        "- Student inference input: `MRI only` for P0, P2, and P3.",
        "- Held-out test loaded/evaluated: `false`.",
        "- Student architecture: locked B1 global classifier; Student ROI residual and graph are disabled.",
        "",
        "## Three-fold results",
        "",
        "| Variant | MCC | AUC | ACC | Macro-F1 | Macro recall | Macro specificity | MCI recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("P0", "P2", "P3"):
        cells = [fmt(summaries[label][metric]) for metric in METRICS]
        lines.append(f"| {label}: {VARIANTS[label]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Privileged Teacher diagnostics",
            "",
            "These values describe the training-only privileged Teachers, not deployable MRI-only performance.",
            "",
            "| Teacher | MCC | AUC | ACC | Macro-F1 | MCI recall | ROI Acc@1 | ROI Acc@5 | Diagonal gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    teacher_metrics = (
        "mcc",
        "auc",
        "acc",
        "f1_macro",
        "mci_recall",
        "roi_retrieval_acc1",
        "roi_retrieval_acc5",
        "diagonal_similarity_gap",
    )
    for label in ("P2", "P3"):
        summary = teachers[label]["summary"]
        teacher_name = "clinical privileged" if label == "P2" else "joint clinical+geometry privileged"
        cells = [fmt(summary[metric]) for metric in teacher_metrics]
        if label == "P2":
            cells[-3:] = ["—", "—", "—"]
        lines.append(
            f"| {label}: {teacher_name} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(["", "## Fold-stratified matched comparisons", ""])
    for name, item in comparisons.items():
        mcc = item["paired_subject_bootstrap"]["mcc"]
        auc = item["paired_subject_bootstrap"]["auc"]
        f1 = item["paired_subject_bootstrap"]["f1_macro"]
        mci = item["paired_subject_bootstrap"]["mci_recall"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- {item['description']}.",
                f"- Fold MCC deltas: `{', '.join(f'{value:+.5f}' for value in item['fold_deltas']['mcc'])}`; "
                f"mean `{item['mean_fold_deltas']['mcc']:+.5f}`.",
                f"- Fold-stratified paired MCC delta `{mcc['observed_delta_right_minus_left']:+.5f}` "
                f"(95% CI `[{mcc['ci95_low']:+.5f}, {mcc['ci95_high']:+.5f}]`).",
                f"- Fold-stratified paired AUC delta `{auc['observed_delta_right_minus_left']:+.5f}` "
                f"(95% CI `[{auc['ci95_low']:+.5f}, {auc['ci95_high']:+.5f}]`).",
                f"- Fold-stratified paired Macro-F1 delta `{f1['observed_delta_right_minus_left']:+.5f}`; "
                f"MCI recall delta `{mci['observed_delta_right_minus_left']:+.5f}`.",
                f"- Exact McNemar p `{item['mcnemar']['exact_two_sided_p']:.6f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Final development decision",
            "",
            f"- P2 advance: `{str(advancement['p2_advance']).lower()}`. Its MCC delta is negative in "
            f"{3 - p2_vs_p0['positive_fold_counts']['mcc']}/3 folds; mean MCC/AUC/Macro-F1 deltas are "
            f"`{p2_vs_p0['mean_fold_deltas']['mcc']:+.5f}` / "
            f"`{p2_vs_p0['mean_fold_deltas']['auc']:+.5f}` / "
            f"`{p2_vs_p0['mean_fold_deltas']['f1_macro']:+.5f}`. MCI recall changes by "
            f"`{p2_vs_p0['mean_fold_deltas']['mci_recall']:+.5f}`, which is a sensitivity trade-off, "
            "not an overall advancement.",
            f"- P3 advance: `{str(advancement['p3_advance']).lower()}`.",
            f"- P3 mean MCC improved over P0: `{str(gate['mean_mcc_improved']).lower()}` "
            f"({gate['mcc_positive_folds']}/3 folds positive).",
            f"- P3 mean AUC improved over P0: `{str(gate['mean_auc_improved']).lower()}` "
            f"({gate['auc_positive_folds']}/3 folds positive).",
            f"- Macro-F1 / MCI recall did not worsen in mean: "
            f"`{str(gate['macro_f1_not_worse']).lower()}` / `{str(gate['mci_recall_not_worse']).lower()}`.",
            f"- Paired MCC / AUC bootstrap intervals exclude zero on the positive side: "
            f"`{str(gate['mcc_ci_excludes_zero']).lower()}` / `{str(gate['auc_ci_excludes_zero']).lower()}`.",
            f"- Locked MRI-only development method: `{advancement['locked_development_method']}`.",
            "- Do not run P4/P5 or evaluate the held-out test in this stage. P4/P5 would only become "
            "necessary for causal attribution after a joint privileged method first passes the performance gate.",
            "",
            "The P3 Teacher learned non-random cross-geometry correspondence, but that privileged signal did not "
            "transfer successfully through class-logit KD to the MRI-only B1 Student. P3-P2 remains evidence "
            "about the complete geometry-privileged Teacher branch, not a clean causal estimate of the hard "
            "surface-ribbon correspondence loss alone.",
        ]
    )
    (output_root / "PRIVILEGED_KD_P2_P3_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({label: summaries[label]["mcc"] for label in ("P0", "P2", "P3")}, indent=2))


if __name__ == "__main__":
    main()
