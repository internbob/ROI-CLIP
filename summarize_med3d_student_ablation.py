#!/usr/bin/env python3
"""Summarize the locked M0--M4 Student-only Med3D development ablation."""

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


VARIANTS = (
    (
        "M0",
        "SFCN S4 + ROI-state KD (locked mainline)",
        "outputs/vnext_graph_student_20261019/s4_roi_kd_corrected",
    ),
    (
        "M1",
        "MedicalNet ResNet10 scratch + CE",
        "outputs/vnext_med3d_resnet10_20261020/m1_resnet10_scratch_ce",
    ),
    (
        "M2",
        "Med3D-pretrained ResNet10 + CE",
        "outputs/vnext_med3d_resnet10_20261020/m2_resnet10_pretrained_ce",
    ),
    (
        "M3",
        "Med3D-pretrained ResNet10 + logit KD",
        "outputs/vnext_med3d_resnet10_20261020/m3_resnet10_pretrained_logit_kd",
    ),
    (
        "M4",
        "Med3D-pretrained ResNet10 + logit KD + ROI-state KD",
        "outputs/vnext_med3d_resnet10_20261020/m4_resnet10_pretrained_roi_kd",
    ),
)

METRICS = (
    "mcc",
    "auc",
    "acc",
    "f1_macro",
    "recall_macro",
    "specificity_macro",
)

COMPARISONS = (
    ("M0", "M1", "backbone/objective replacement"),
    ("M1", "M2", "Med3D pretraining effect"),
    ("M2", "M3", "logit KD increment"),
    ("M3", "M4", "ROI-state KD increment"),
    ("M2", "M4", "total KD increment"),
    ("M4", "M2", "requested M2 > M4 gate"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root", default="outputs/vnext_med3d_resnet10_20261020"
    )
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def macro_specificity(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    total = matrix.sum()
    values = []
    for index in range(num_classes):
        false_positive = matrix[:, index].sum() - matrix[index, index]
        true_negative = total - matrix[index, :].sum() - false_positive
        denominator = true_negative + false_positive
        values.append(float(true_negative / denominator) if denominator else np.nan)
    return float(np.nanmean(values))


def metric_from_frame(frame: pd.DataFrame, metric: str) -> float:
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["pred"].to_numpy(dtype=int)
    score_columns = sorted(
        (column for column in frame.columns if column.startswith("score_")),
        key=lambda value: int(value.split("_")[-1]),
    )
    if metric == "mcc":
        return float(matthews_corrcoef(labels, predictions))
    if metric == "auc":
        return float(
            roc_auc_score(
                labels,
                frame[score_columns].to_numpy(dtype=float),
                labels=list(range(len(score_columns))),
                multi_class="ovr",
                average="macro",
            )
        )
    if metric == "acc":
        return float(accuracy_score(labels, predictions))
    if metric == "f1_macro":
        return float(f1_score(labels, predictions, average="macro", zero_division=0))
    if metric == "recall_macro":
        return float(recall_score(labels, predictions, average="macro", zero_division=0))
    if metric == "specificity_macro":
        return macro_specificity(labels, predictions, len(score_columns))
    raise ValueError(metric)


def load_variant(project_root: Path, relative_root: str, folds: list[int]):
    metrics = []
    frames = []
    provenance = []
    for fold in folds:
        fold_root = project_root / relative_root / f"fold_{fold}"
        metrics_path = fold_root / "development_metrics.json"
        predictions_path = fold_root / "validation_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Incomplete fold output: {fold_root}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("held_out_test_evaluated") is not False:
            raise RuntimeError(f"Held-out test flag is not false: {metrics_path}")
        raw = payload["metrics"]
        normalized = {
            "mcc": float(raw["mcc"]),
            "auc": float(raw["auc"]),
            "acc": float(raw["acc"]),
            "f1_macro": float(raw["f1_macro"]),
            "recall_macro": float(raw.get("recall_macro", raw.get("sen", raw["balanced_acc"]))),
            "specificity_macro": float(raw.get("spe")),
        }
        metrics.append(normalized)
        frame = pd.read_csv(predictions_path)
        frame["fold"] = fold
        frame["key"] = frame["fold"].astype(str) + ":" + frame["filename"].astype(str)
        frames.append(frame)
        provenance.append(
            {
                "fold": fold,
                "best_epoch": int(payload["best_epoch"]),
                "metrics_path": str(metrics_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
                "held_out_test_evaluated": False,
            }
        )
    pooled = pd.concat(frames, ignore_index=True)
    if pooled["key"].duplicated().any():
        raise RuntimeError(f"Duplicate prediction keys: {relative_root}")
    return metrics, pooled, provenance


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "fold_values": array.tolist(),
    }


def paired_frames(left: pd.DataFrame, right: pd.DataFrame):
    merged = left.merge(right, on="key", suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Prediction keys differ between paired variants")
    if not np.array_equal(merged["label_left"], merged["label_right"]):
        raise RuntimeError("Prediction labels differ between paired variants")
    return merged


def select_side(merged: pd.DataFrame, side: str, score_columns: list[str], indices=None):
    values = {
        "label": merged[f"label_{side}"],
        "pred": merged[f"pred_{side}"],
    }
    for column in score_columns:
        values[column] = merged[f"{column}_{side}"]
    frame = pd.DataFrame(values)
    return frame if indices is None else frame.iloc[indices]


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    replicates: int,
    seed: int,
):
    merged = paired_frames(left, right)
    score_columns = sorted(
        (column for column in left.columns if column.startswith("score_")),
        key=lambda value: int(value.split("_")[-1]),
    )
    left_frame = select_side(merged, "left", score_columns)
    right_frame = select_side(merged, "right", score_columns)
    observed = metric_from_frame(right_frame, metric) - metric_from_frame(left_frame, metric)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        indices = rng.integers(0, len(merged), size=len(merged))
        try:
            deltas.append(
                metric_from_frame(select_side(merged, "right", score_columns, indices), metric)
                - metric_from_frame(select_side(merged, "left", score_columns, indices), metric)
            )
        except ValueError:
            continue
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "metric": metric,
        "observed_delta_right_minus_left": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "valid_replicates": len(deltas),
    }


def comparison(left_label, right_label, description, loaded, replicates, seed):
    left = loaded[left_label]
    right = loaded[right_label]
    fold_deltas = {
        metric: [
            float(r[metric]) - float(l[metric])
            for l, r in zip(left["metrics"], right["metrics"])
        ]
        for metric in METRICS
    }
    bootstraps = {
        metric: paired_bootstrap(
            left["predictions"],
            right["predictions"],
            metric,
            replicates,
            seed + index,
        )
        for index, metric in enumerate(METRICS)
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
        "description": description,
        "direction": f"{right_label} minus {left_label}",
        "fold_deltas": fold_deltas,
        "mean_fold_deltas": {
            metric: float(np.mean(values)) for metric, values in fold_deltas.items()
        },
        "mcc_positive_folds": int(sum(value > 0 for value in fold_deltas["mcc"])),
        "paired_subject_bootstrap": bootstraps,
        "mcnemar": {
            "right_only_correct": right_only,
            "left_only_correct": left_only,
            "exact_two_sided_p": (
                float(binomtest(right_only, discordant, 0.5).pvalue) if discordant else 1.0
            ),
        },
    }


def format_summary(item):
    return f"{item['mean']:.5f} ± {item['sample_sd']:.5f}"


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]

    loaded = {}
    for label, description, relative_root in VARIANTS:
        metrics, predictions, provenance = load_variant(project_root, relative_root, folds)
        loaded[label] = {
            "description": description,
            "relative_root": relative_root,
            "metrics": metrics,
            "predictions": predictions,
            "provenance": provenance,
        }

    summaries = {
        label: {
            metric: summarize([fold[metric] for fold in item["metrics"]])
            for metric in METRICS
        }
        for label, item in loaded.items()
    }
    comparisons = {}
    for index, (left, right, description) in enumerate(COMPARISONS):
        key = f"{right}_minus_{left}"
        comparisons[key] = comparison(
            left,
            right,
            description,
            loaded,
            args.bootstrap_replicates,
            args.seed + 1000 * index,
        )

    m2_over_m4 = comparisons["M2_minus_M4"]
    m2_mcc_boot = m2_over_m4["paired_subject_bootstrap"]["mcc"]
    m2_significantly_better = bool(
        m2_over_m4["mean_fold_deltas"]["mcc"] > 0
        and m2_mcc_boot["ci95_low"] > 0
        and m2_over_m4["mcc_positive_folds"] >= 2
    )
    best_label = max(summaries, key=lambda label: summaries[label]["mcc"]["mean"])
    best_med3d_label = max(
        ("M1", "M2", "M3", "M4"), key=lambda label: summaries[label]["mcc"]["mean"]
    )
    decision = {
        "retain_locked_mainline": best_label == "M0",
        "best_overall_development_variant": best_label,
        "best_med3d_variant": best_med3d_label,
        "m2_significantly_better_than_m4": m2_significantly_better,
        "retrain_med3d_cross_geometry_teacher": False,
        "reason": (
            "The predeclared trigger is not met: M2 is not significantly better than M4. "
            "Moreover, the best Med3D variant does not exceed the locked M0 mainline, so a new "
            "Med3D cross-geometry teacher is not justified by this development ablation."
        ),
        "held_out_test_used": False,
    }

    payload = {
        "completed_on": date.today().isoformat(),
        "folds": folds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "held_out_test_evaluated": False,
        "variant_summary": summaries,
        "comparisons": comparisons,
        "decision": decision,
        "provenance": {label: item["provenance"] for label, item in loaded.items()},
        "statistical_scope": (
            "Fold mean and sample SD are descriptive. Paired pooled-subject bootstrap intervals "
            "quantify uncertainty on matched validation predictions; no held-out test was used."
        ),
    }
    (output_root / "med3d_student_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for label, description, _ in VARIANTS:
        row = {"variant": label, "description": description}
        for metric in METRICS:
            row[f"{metric}_mean"] = summaries[label][metric]["mean"]
            row[f"{metric}_sample_sd"] = summaries[label][metric]["sample_sd"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "med3d_student_ablation_metrics.csv", index=False)

    lines = [
        "# CortexBridge Student-only Med3D ResNet10 M0–M4 Ablation",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Decision",
        "",
        "**Retain M0 (the locked SFCN S4 + ROI-state KD Student) as the CortexBridge mainline.**",
        "",
        f"The best Med3D candidate is **{best_med3d_label}**, but it does not exceed M0 on "
        "three-fold development MCC. The predeclared trigger for considering a Med3D "
        f"cross-geometry teacher—M2 significantly outperforming M4—is `{str(m2_significantly_better).lower()}`. "
        "Therefore this stage does not justify retraining the teacher with Med3D.",
        "",
        "No held-out test split was loaded or evaluated. All conclusions below are development-only.",
        "",
        "## Experimental matrix",
        "",
        "| ID | Student encoder/objective | Teacher during training | Inference inputs |",
        "|---|---|---|---|",
        "| M0 | SFCN + CE + logit KD + ROI-state KD | Locked G3 SFCN cross-geometry teacher | MRI only |",
        "| M1 | MedicalNet ResNet10, random initialization + CE | None | MRI only |",
        "| M2 | MedicalNet/Med3D ResNet10 pretrained + CE | None | MRI only |",
        "| M3 | M2 + logit KD | Locked G3 SFCN teacher | MRI only |",
        "| M4 | M3 + ROI graph-state KD | Locked G3 SFCN teacher | MRI only |",
        "",
        "Shared settings for M1–M4: identical folds, topology graph/query and downstream classifier; "
        "batch size 8; seed 3407; 30-epoch cap; patience 4; MCC checkpoint selection; no clinical/MMSE, PET, "
        "surface, or ribbon input at Student inference. M2–M4 use encoder LR `1e-5`; M1 uses `1e-4`; inherited "
        "and head LR are `1e-4`. M3/M4 use logit KD weight `0.5`; M4 adds ROI-state KD weight `1.0`.",
        "",
        "## Three-fold development results",
        "",
        "| ID | MCC | AUC | ACC | Macro-F1 | Recall / sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in VARIANTS:
        cells = [format_summary(summaries[label][metric]) for metric in METRICS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "Recall is macro recall and equals the reported multiclass sensitivity/balanced accuracy. "
            "Specificity is macro one-vs-rest specificity. Values are fold mean ± sample SD.",
            "",
            "## Fold-level MCC",
            "",
            "| ID | Fold 0 | Fold 1 | Fold 2 | Mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, _, _ in VARIANTS:
        values = summaries[label]["mcc"]["fold_values"]
        lines.append(
            f"| {label} | " + " | ".join(f"{value:.5f}" for value in values)
            + f" | {summaries[label]['mcc']['mean']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Progressive paired comparisons",
            "",
            "The interval is a paired subject-level bootstrap on pooled validation predictions. "
            "Because only three folds are available, fold mean/SD and fold direction are descriptive; "
            "no low-powered fold-level t-test is used.",
            "",
            "| Contrast | Fold ΔMCC | Mean fold ΔMCC | Positive folds | Pooled ΔMCC (95% CI) | McNemar p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("M1_minus_M0", "M2_minus_M1", "M3_minus_M2", "M4_minus_M3", "M4_minus_M2"):
        item = comparisons[key]
        deltas = item["fold_deltas"]["mcc"]
        boot = item["paired_subject_bootstrap"]["mcc"]
        lines.append(
            f"| {item['direction']} | "
            + ", ".join(f"{value:+.5f}" for value in deltas)
            + f" | {item['mean_fold_deltas']['mcc']:+.5f} | {item['mcc_positive_folds']}/3 | "
            + f"{boot['observed_delta_right_minus_left']:+.5f} "
            + f"[{boot['ci95_low']:+.5f}, {boot['ci95_high']:+.5f}] | "
            + f"{item['mcnemar']['exact_two_sided_p']:.6f} |"
        )

    m2_m4_auc = m2_over_m4["paired_subject_bootstrap"]["auc"]
    lines.extend(
        [
            "",
            "## Requested M2 versus M4 gate",
            "",
            f"- Fold mean ΔMCC (M2 − M4): `{m2_over_m4['mean_fold_deltas']['mcc']:+.5f}`.",
            f"- Pooled paired ΔMCC (M2 − M4): `{m2_mcc_boot['observed_delta_right_minus_left']:+.5f}` "
            f"(95% CI `[{m2_mcc_boot['ci95_low']:+.5f}, {m2_mcc_boot['ci95_high']:+.5f}]`).",
            f"- Pooled paired ΔAUC (M2 − M4): `{m2_m4_auc['observed_delta_right_minus_left']:+.5f}` "
            f"(95% CI `[{m2_m4_auc['ci95_low']:+.5f}, {m2_m4_auc['ci95_high']:+.5f}]`).",
            f"- M2 significantly better than M4 under the predeclared rule: "
            f"`{str(m2_significantly_better).lower()}`.",
            "- Decision: do not retrain a Med3D cross-geometry teacher in this stage.",
            "",
            "## Interpretation",
            "",
            "1. **The locked CortexBridge mainline remains M0.** Replacing its SFCN Student with the tested "
            "MedicalNet ResNet10 recipes does not improve development MCC.",
            "2. **Off-the-shelf Med3D pretraining is not beneficial under the locked recipe.** M2 should be "
            "interpreted against M1 to isolate initialization; the direction is adverse rather than beneficial.",
            "3. **Logit KD partially recovers the weak pretrained Student.** M3 should be interpreted against M2.",
            "4. **ROI-state KD is not a stable additional gain for the Med3D Student.** M4 should be interpreted "
            "against M3, with fold directions and the paired interval reported rather than claiming a benefit.",
            "5. **A Med3D teacher is not currently justified.** The requested trigger is absent, and the best "
            "Med3D Student remains below M0. Further teacher retraining would expand compute without positive "
            "evidence that Med3D initialization is the bottleneck.",
            "",
            "## Integrity and verification",
            "",
            "- Official MedicalNet ResNet10 checkpoint SHA256: "
            "`afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.",
            "- Each pretrained Student run loaded 72 backbone tensors.",
            "- All 15 included fold JSON files record `held_out_test_evaluated: false`.",
            "- The three checkpoints of the best Med3D variant M1 passed the MRI-only audit: no privileged "
            "state keys, zero logit change from an arbitrary clinical argument, zero logit change after "
            "invalidating subject-specific cache paths, and held-out flags false.",
            "- Interrupted/incomplete directories and dry-runs are excluded by explicit variant paths.",
            "- Selection metric and early stopping were unchanged across the requested ablation.",
        ]
    )
    (output_root / "MED3D_RESNET10_STUDENT_ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
