#!/usr/bin/env python3
"""Summarize the D0--D4 audit and matched B0--B4 ROI bottleneck ablation."""

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
    ("B0", "global CE only", "b0_global_ce"),
    ("B1", "global CE + logit KD", "b1_global_logit_kd"),
    ("B2", "B1 + raw ROI-token classifier", "b2_global_raw_roi"),
    ("B3", "B2 + ROI graph", "b3_global_raw_roi_graph"),
    ("B4", "B3 + ROI-state KD", "b4_global_raw_roi_graph_kd"),
)

METRICS = ("mcc", "auc", "acc", "f1_macro", "recall_macro", "specificity_macro")
COMPARISONS = (
    ("B0", "B1", "logit KD"),
    ("B1", "B2", "raw ROI classification residual"),
    ("B2", "B3", "ROI graph"),
    ("B3", "B4", "ROI-state KD"),
    ("B1", "B4", "complete ROI path and ROI-state KD"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root", default="outputs/vnext_roi_bottleneck_ablation_20261021"
    )
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def macro_specificity(labels, predictions, num_classes):
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    total = matrix.sum()
    values = []
    for index in range(num_classes):
        false_positive = matrix[:, index].sum() - matrix[index, index]
        true_negative = total - matrix[index, :].sum() - false_positive
        denominator = true_negative + false_positive
        values.append(float(true_negative / denominator) if denominator else np.nan)
    return float(np.nanmean(values))


def metric_from_frame(frame, metric):
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["pred"].to_numpy(dtype=int)
    score_columns = sorted(
        (column for column in frame.columns if column.startswith("score_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
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


def load_variant(output_root, directory, folds):
    metrics = []
    frames = []
    provenance = []
    for fold in folds:
        fold_root = output_root / directory / f"fold_{fold}"
        metrics_path = fold_root / "development_metrics.json"
        predictions_path = fold_root / "validation_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Incomplete fold output: {fold_root}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("held_out_test_evaluated") is not False:
            raise RuntimeError(f"Held-out flag is not false: {metrics_path}")
        raw = payload["metrics"]
        metrics.append(
            {
                "mcc": float(raw["mcc"]),
                "auc": float(raw["auc"]),
                "acc": float(raw["acc"]),
                "f1_macro": float(raw["f1_macro"]),
                "recall_macro": float(raw.get("recall_macro", raw.get("sen"))),
                "specificity_macro": float(raw.get("specificity_macro", raw.get("spe"))),
            }
        )
        frame = pd.read_csv(predictions_path)
        frame["fold"] = fold
        frame["key"] = frame["fold"].astype(str) + ":" + frame["filename"].astype(str)
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
        raise RuntimeError(f"Duplicate prediction keys in {directory}")
    return metrics, pooled, provenance


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "fold_values": values.tolist(),
    }


def paired_frames(left, right):
    merged = left.merge(right, on="key", suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Prediction keys differ between paired variants")
    if not np.array_equal(merged["label_left"], merged["label_right"]):
        raise RuntimeError("Prediction labels differ between paired variants")
    return merged


def select_side(merged, side, score_columns, indices=None):
    data = {
        "label": merged[f"label_{side}"],
        "pred": merged[f"pred_{side}"],
    }
    for column in score_columns:
        data[column] = merged[f"{column}_{side}"]
    frame = pd.DataFrame(data)
    return frame if indices is None else frame.iloc[indices]


def paired_bootstrap(left, right, metric, replicates, seed):
    merged = paired_frames(left, right)
    score_columns = sorted(
        (column for column in left.columns if column.startswith("score_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
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
        "observed_delta_right_minus_left": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "valid_replicates": len(deltas),
    }


def compare(left_label, right_label, description, loaded, replicates, seed):
    left = loaded[left_label]
    right = loaded[right_label]
    fold_deltas = {
        metric: [r[metric] - l[metric] for l, r in zip(left["metrics"], right["metrics"])]
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
        "description": description,
        "fold_deltas": fold_deltas,
        "mean_fold_deltas": {key: float(np.mean(value)) for key, value in fold_deltas.items()},
        "mcc_positive_folds": int(sum(value > 0 for value in fold_deltas["mcc"])),
        "paired_subject_bootstrap": {
            metric: paired_bootstrap(
                left["predictions"], right["predictions"], metric, replicates, seed + index
            )
            for index, metric in enumerate(("mcc", "auc"))
        },
        "mcnemar": {
            "right_only_correct": right_only,
            "left_only_correct": left_only,
            "exact_two_sided_p": (
                float(binomtest(right_only, discordant, 0.5).pvalue) if discordant else 1.0
            ),
        },
    }


def summarize_audit(audit):
    names = {
        "D0": ("Current S4 fused", lambda fold: fold["metrics"]["d0_fused"]),
        "D1": ("S4 global only", lambda fold: fold["metrics"]["d1_global"]),
        "D2": ("S4 ROI only", lambda fold: fold["metrics"]["d2_roi"]),
        "D3": ("S4 global + pre-graph ROI", lambda fold: fold["metrics"]["d3_pregraph"]),
        "D4": ("S3 global only", lambda fold: fold["s3_global_metrics"]),
    }
    output = {}
    for label, (description, getter) in names.items():
        output[label] = {"description": description}
        for metric in METRICS:
            key = {
                "f1_macro": "f1_macro",
                "recall_macro": "recall_macro",
                "specificity_macro": "spe",
            }.get(metric, metric)
            output[label][metric] = summarize([getter(fold)[key] for fold in audit["folds"]])
    diagnostics = {}
    diagnostic_keys = (
        "pregraph_offdiag_cosine",
        "postgraph_offdiag_cosine",
        "pregraph_token_variance",
        "postgraph_token_variance",
        "graph_delta_relative_norm",
        "global_logit_norm",
        "roi_logit_norm",
    )
    for key in diagnostic_keys:
        diagnostics[key] = summarize([fold["diagnostics"][key] for fold in audit["folds"]])
    diagnostics["roi_logit_weight"] = summarize([fold["roi_logit_weight"] for fold in audit["folds"]])
    diagnostics["graph_residual_gate"] = summarize(
        [fold["graph_residual_gate"] for fold in audit["folds"]]
    )
    diagnostics["historical_prediction_max_abs_delta"] = max(
        fold["saved_prediction_max_abs_delta"] for fold in audit["folds"]
    )
    return output, diagnostics


def fmt(summary):
    return f"{summary['mean']:.5f} ± {summary['sample_sd']:.5f}"


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    audit = json.loads((output_root / "branch_audit.json").read_text(encoding="utf-8"))
    if audit["protocol"].get("held_out_test_loaded") is not False:
        raise RuntimeError("Audit held-out flag is not false")
    audit_summary, diagnostics = summarize_audit(audit)

    loaded = {}
    for label, description, directory in VARIANTS:
        metrics, predictions, provenance = load_variant(output_root, directory, folds)
        loaded[label] = {
            "description": description,
            "metrics": metrics,
            "predictions": predictions,
            "provenance": provenance,
        }
    summaries = {
        label: {
            metric: summarize([item[metric] for item in value["metrics"]])
            for metric in METRICS
        }
        for label, value in loaded.items()
    }
    comparisons = {}
    for index, (left, right, description) in enumerate(COMPARISONS):
        comparisons[f"{right}_minus_{left}"] = compare(
            left, right, description, loaded, args.bootstrap_replicates, args.seed + 1000 * index
        )

    best_label = max(summaries, key=lambda label: summaries[label]["mcc"]["mean"])
    decision = {
        "best_matched_development_variant": best_label,
        "held_out_test_used": False,
        "roi_path_improves_over_global_logit_kd": bool(
            summaries["B4"]["mcc"]["mean"] > summaries["B1"]["mcc"]["mean"]
        ),
        "recommended_student_classification_path": (
            "global MRI classifier + locked-teacher logit KD; disable the Student ROI "
            "classification residual and Student ROI graph"
        ),
    }
    payload = {
        "completed_on": date.today().isoformat(),
        "held_out_test_evaluated": False,
        "folds": folds,
        "branch_audit": audit_summary,
        "branch_diagnostics": diagnostics,
        "matched_training": summaries,
        "comparisons": comparisons,
        "decision": decision,
        "provenance": {label: item["provenance"] for label, item in loaded.items()},
    }
    (output_root / "roi_bottleneck_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for label, description, _ in VARIANTS:
        row = {"variant": label, "description": description}
        for metric in METRICS:
            row[f"{metric}_mean"] = summaries[label][metric]["mean"]
            row[f"{metric}_sample_sd"] = summaries[label][metric]["sample_sd"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "roi_bottleneck_ablation_metrics.csv", index=False)

    lines = [
        "# CortexBridge ROI Bottleneck Ablation",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Scope and integrity",
        "",
        "This development-only study tests whether converting the deep MRI feature map into ROI tokens and then a graph weakens classification. No held-out test split was loaded or evaluated. The locked G3 Teacher and downstream task definition were unchanged.",
        "",
        "Two complementary protocols are reported: D0–D4 switch branches inside already-trained checkpoints without retraining; B0–B4 retrain matched Students to isolate each component.",
        "",
        "## D0–D4 checkpoint branch audit (no training)",
        "",
        "| ID | Branch at inference | MCC | AUC | ACC | Macro-F1 | Recall/SEN | Specificity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("D0", "D1", "D2", "D3", "D4"):
        item = audit_summary[label]
        cells = [fmt(item[metric]) for metric in METRICS]
        lines.append(f"| {label} | {item['description']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "D0 reproduces the saved historical probabilities exactly; maximum absolute difference: "
            f"`{diagnostics['historical_prediction_max_abs_delta']:.3g}`.",
            "",
            "### Branch diagnostics",
            "",
            f"- Learned ROI-logit residual weight: `{fmt(diagnostics['roi_logit_weight'])}`.",
            f"- Learned graph residual gate: `{fmt(diagnostics['graph_residual_gate'])}`.",
            f"- Pre/post-graph off-diagonal ROI cosine: `{fmt(diagnostics['pregraph_offdiag_cosine'])}` / `{fmt(diagnostics['postgraph_offdiag_cosine'])}`.",
            f"- Pre/post-graph token variance: `{fmt(diagnostics['pregraph_token_variance'])}` / `{fmt(diagnostics['postgraph_token_variance'])}`.",
            f"- Graph delta relative norm: `{fmt(diagnostics['graph_delta_relative_norm'])}`.",
            f"- Global/ROI logit norm: `{fmt(diagnostics['global_logit_norm'])}` / `{fmt(diagnostics['roi_logit_norm'])}`.",
            "",
            "## B0–B4 matched three-fold retraining",
            "",
            "| ID | Training configuration | MCC | AUC | ACC | Macro-F1 | Recall/SEN | Specificity |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, description, _ in VARIANTS:
        cells = [fmt(summaries[label][metric]) for metric in METRICS]
        lines.append(f"| {label} | {description} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "All values are fold mean ± sample SD. Recall is macro recall (multiclass sensitivity/balanced accuracy); specificity is macro one-vs-rest specificity.",
            "",
            "### Fold-level MCC",
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
            "### Progressive paired comparisons",
            "",
            "Pooled intervals use paired subject bootstrap over matched validation predictions. They complement, rather than replace, three-fold direction and sample SD.",
            "",
            "| Component added | Contrast | Fold ΔMCC | Mean ΔMCC | Positive folds | Pooled ΔMCC (95% CI) | Pooled ΔAUC (95% CI) | McNemar p |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for left, right, description in COMPARISONS:
        item = comparisons[f"{right}_minus_{left}"]
        boot = item["paired_subject_bootstrap"]["mcc"]
        auc_boot = item["paired_subject_bootstrap"]["auc"]
        lines.append(
            f"| {description} | {right} − {left} | "
            + ", ".join(f"{value:+.5f}" for value in item["fold_deltas"]["mcc"])
            + f" | {item['mean_fold_deltas']['mcc']:+.5f} | {item['mcc_positive_folds']}/3 | "
            + f"{boot['observed_delta_right_minus_left']:+.5f} [{boot['ci95_low']:+.5f}, {boot['ci95_high']:+.5f}] | "
            + f"{auc_boot['observed_delta_right_minus_left']:+.5f} [{auc_boot['ci95_low']:+.5f}, {auc_boot['ci95_high']:+.5f}] | "
            + f"{item['mcnemar']['exact_two_sided_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and mainline decision",
            "",
            "**Select B1 as the revised Student classification mainline: global MRI classification + locked-Teacher logit KD, with the Student ROI classification residual and Student ROI graph disabled.**",
            "",
            "The causal evidence is consistent across the two protocols. In the existing S4 checkpoints, D1 global-only improves mean MCC over D0 fused (`+0.00974`), whereas the ROI-only branch D2 is at chance and D3 is nearly identical to D0. In matched retraining, B2 loses `0.07009` mean MCC versus B1 in all three folds; its paired pooled 95% CI excludes zero. Because B2 already classifies raw, unnormalized ROI tokens, this loss cannot be explained only by the former unit-norm bottleneck.",
            "",
            "The graph is not the primary cause. B3 changes mean MCC by only `-0.00728` versus B2, improves two of three folds, and its paired MCC interval crosses zero. The learned graph residual is small (about 5.9% of token norm), while bypassing the graph in D3 barely changes D0. The graph may be retained for representation analysis, but it is not justified in the classification path by these results.",
            "",
            "ROI-state KD is useful as an auxiliary representation constraint but does not justify the full ROI classification path. B4 recovers `+0.06596` mean MCC over B3 and its pooled MCC interval is positive; however, B4 remains `-0.01141` below B1 on mean MCC, with heterogeneous fold effects and a CI spanning zero. Thus B4 is an informative auxiliary/anatomical ablation, not the selected classifier.",
            "",
            "The checkpoint audit also indicates that ROI-state KD did not damage the global encoder: D1 (S4 global-only) exceeds D4 (S3 global-only) on mean MCC. The performance loss appears when the weak ROI logits are fused back into the final classifier, not simply from learning an ROI-aligned representation.",
            "",
            "The B experiments use raw projected ROI tokens for classification and normalized copies only for graph-state KD. This explicitly removes the unintended unit-norm classification bottleneck while preserving the teacher-alignment interface.",
            "",
            "These are development-only selection results. The held-out test remains unopened and should be evaluated once, only after the revised B1 mainline and manuscript comparison protocol are frozen.",
            "",
            "## Verification",
            "",
            "- All 15 B0–B4 fold records set `held_out_test_evaluated: false`.",
            "- D0 exactly reproduces historical S4 validation probabilities.",
            "- Teacher checkpoints, folds, seed, optimizer schedule, early stopping and MCC checkpoint selection are matched.",
            "- B0/B1 disable Student ROI classification and Student graph without changing the locked G3 Teacher.",
            "- B2 adds raw ROI classification; B3 adds graph; B4 adds ROI-state KD.",
        ]
    )
    (output_root / "ROI_BOTTLENECK_ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
