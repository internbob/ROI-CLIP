#!/usr/bin/env python3
"""Summarize the locked clinical Logit + ROI-state KD development experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from summarize_unified_clinical import (
    load_predictions,
    mean_sd,
    metric_values,
    paired_bootstrap,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/vnext_clinical_two_level_kd_20261027"
ARMS = {
    "E3": (
        "Earlier MRI+clinical Student + MRI+clinical Teacher Logit KD",
        ROOT / "outputs/vnext_unified_clinical_20261023/e3_student",
    ),
    "E4": (
        "Earlier full privileged Teacher Logit KD",
        ROOT / "outputs/vnext_unified_clinical_20261023/e4_student",
    ),
    "D0": ("Matched graph Student, Logit KD only", OUTPUT / "d0_logit_only"),
    "D1": ("Final clinical Logit + ROI-state KD", OUTPUT / "lambda_sweep_roi100"),
    "D2": ("D1 with cross-geometry-disabled Teacher", OUTPUT / "d2_cg_off"),
    "D3": ("D1 with Teacher ROI correspondence shuffled", OUTPUT / "d3_roi_shuffle"),
}
METRICS = ("mcc", "auc", "acc", "f1_macro", "recall_macro", "spe", "mci_recall")


def load_fold(arm: str, fold: int):
    directory = ARMS[arm][1] / f"fold_{fold}"
    metadata = json.loads((directory / "development_metrics.json").read_text(encoding="utf-8"))
    if metadata.get("held_out_test_evaluated") is not False:
        raise RuntimeError(f"held-out test flag is not false: {directory}")
    ids, labels, probabilities = load_predictions(directory / "validation_predictions.csv")
    return ids, labels, probabilities


def main():
    selection = json.loads((OUTPUT / "lambda_selection.json").read_text(encoding="utf-8"))
    fold_rows = []
    pooled = []
    for fold in range(3):
        reference_ids = reference_labels = None
        loaded = {}
        for arm in ARMS:
            ids, labels, probabilities = load_fold(arm, fold)
            if reference_ids is None:
                reference_ids, reference_labels = ids, labels
            elif ids != reference_ids or not np.array_equal(labels, reference_labels):
                raise RuntimeError(f"unpaired validation samples for {arm}, fold {fold}")
            loaded[arm] = probabilities
            metrics = metric_values(labels, probabilities)
            fold_rows.append(
                {"arm": arm, "fold": fold, **{key: float(metrics[key]) for key in METRICS}}
            )
        for index, filename in enumerate(reference_ids):
            row = {"fold": fold, "filename": filename, "label": int(reference_labels[index])}
            for arm, probabilities in loaded.items():
                row[arm] = probabilities[index].tolist()
            pooled.append(row)

    summary_rows = []
    for arm in ARMS:
        records = [row for row in fold_rows if row["arm"] == arm]
        summary = {"arm": arm, "folds": 3}
        for key in METRICS:
            mean, sd = mean_sd([row[key] for row in records])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd
        summary_rows.append(summary)

    comparisons = []
    for index, (candidate, baseline) in enumerate(
        [("D1", "D0"), ("D1", "E3"), ("D1", "E4"), ("D1", "D2"), ("D1", "D3")]
    ):
        comparisons.append(
            paired_bootstrap(pooled, candidate, baseline, 5000, 3407 + index)
        )

    forbidden = sorted(
        str(path.relative_to(OUTPUT))
        for pattern in ("test_metrics.json", "test_predictions.csv")
        for path in OUTPUT.rglob(pattern)
    )
    if forbidden:
        raise RuntimeError(f"held-out test artifacts found: {forbidden}")

    analysis_dir = OUTPUT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "metrics_by_fold.csv", fold_rows)
    write_csv(analysis_dir / "metrics_summary.csv", summary_rows)
    write_csv(analysis_dir / "paired_comparisons.csv", comparisons)
    payload = {
        "held_out_test_accessed": False,
        "folds": [0, 1, 2],
        "lambda_selection": selection,
        "summary": summary_rows,
        "paired_comparisons": comparisons,
    }
    (analysis_dir / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summaries = {row["arm"]: row for row in summary_rows}
    contrasts = {(row["candidate"], row["baseline"]): row for row in comparisons}
    lines = [
        "# Clinical Two-Level Distillation — Development Experiment",
        "",
        "## Decision",
        "",
        "- **The requested Teacher+Student clinical Logit+ROI-state experiment is complete, but D1 is not selected as the performance-primary method.**",
        f"- Against the matched Logit-only arm, D1−D0 pooled delta MCC is {contrasts[('D1', 'D0')]['mcc_delta']:+.5f} "
        f"(95% CI [{contrasts[('D1', 'D0')]['mcc_ci_low']:+.5f}, {contrasts[('D1', 'D0')]['mcc_ci_high']:+.5f}]); the interval includes zero.",
        f"- D1 does not exceed the earlier E3 performance configuration: delta MCC {contrasts[('D1', 'E3')]['mcc_delta']:+.5f}, "
        f"delta AUC {contrasts[('D1', 'E3')]['auc_delta']:+.5f} "
        f"(95% CI [{contrasts[('D1', 'E3')]['auc_ci_low']:+.5f}, {contrasts[('D1', 'E3')]['auc_ci_high']:+.5f}]).",
        f"- Correct ROI correspondence is not supported as the source of improvement: D1−D3 delta MCC is {contrasts[('D1', 'D3')]['mcc_delta']:+.5f}; the shuffled-ROI control has a slightly higher mean MCC.",
        "- E3 should remain the classification-performance configuration. D1/D3 belong in the ablation section as evidence that the current ROI-state KD objective behaves mainly as a weak regularizer, not as validated anatomical knowledge transfer.",
        "",
        "## Protocol",
        "",
        "- Task: ADNI NC/MCI/DE three-class classification.",
        "- Both Teacher and Student use the same 29 clinical variables; MMSE is excluded.",
        "- Teacher: E4 MRI + surface + ribbon correspondence + anatomical ROI graph + clinical fusion.",
        "- Student: MRI + clinical global classifier; ROI classification residual remains disabled.",
        "- Two-level supervision: temperature-scaled class-logit KD plus cosine ROI-state KD.",
        "- Model selection and all comparisons use subject-level development folds 0–2 only.",
        "- Held-out test accessed: `false`.",
        "",
        "## Fold-0 ROI-state weight selection",
        "",
        "| lambda_roi | MCC | AUC | ACC | Macro-F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in selection["candidates"]:
        lines.append(
            f"| {row['lambda_roi']:.2f} | {row['mcc']:.5f} | {row['auc']:.5f} | "
            f"{row['acc']:.5f} | {row['f1_macro']:.5f} |"
        )
    lines.extend(
        [
            "",
            f"Selected `lambda_roi={selection['selected_lambda']:.2f}` by the locked development rule: "
            "highest fold-0 MCC, then AUC, then the smaller weight.",
            "",
            "## Three-fold results",
            "",
            "| Arm | Configuration | MCC | AUC | ACC | Macro-F1 | Recall | SPE | MCI recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, (description, _) in ARMS.items():
        row = summaries[arm]
        lines.append(
            f"| {arm} | {description} | {row['mcc_mean']:.5f} ± {row['mcc_sd']:.5f} | "
            f"{row['auc_mean']:.5f} ± {row['auc_sd']:.5f} | "
            f"{row['acc_mean']:.5f} ± {row['acc_sd']:.5f} | "
            f"{row['f1_macro_mean']:.5f} ± {row['f1_macro_sd']:.5f} | "
            f"{row['recall_macro_mean']:.5f} ± {row['recall_macro_sd']:.5f} | "
            f"{row['spe_mean']:.5f} ± {row['spe_sd']:.5f} | "
            f"{row['mci_recall_mean']:.5f} ± {row['mci_recall_sd']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Paired subject-bootstrap contrasts",
            "",
            "| Contrast | Delta MCC [95% CI] | Delta AUC [95% CI] | Delta ACC [95% CI] | McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['candidate']} − {row['baseline']} | "
            f"{row['mcc_delta']:+.5f} [{row['mcc_ci_low']:+.5f}, {row['mcc_ci_high']:+.5f}] | "
            f"{row['auc_delta']:+.5f} [{row['auc_ci_low']:+.5f}, {row['auc_ci_high']:+.5f}] | "
            f"{row['acc_delta']:+.5f} [{row['acc_ci_low']:+.5f}, {row['acc_ci_high']:+.5f}] | "
            f"{row['mcnemar_p']:.6g} |"
        )
    primary = contrasts[("D1", "D0")]
    shuffle = contrasts[("D1", "D3")]
    cg = contrasts[("D1", "D2")]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- The matched incremental effect of adding ROI-state KD is D1−D0 delta MCC "
            f"{primary['mcc_delta']:+.5f} (95% CI [{primary['mcc_ci_low']:+.5f}, {primary['mcc_ci_high']:+.5f}]).",
            f"- Correct ROI identity contributes only if D1 exceeds D3; observed delta MCC is "
            f"{shuffle['mcc_delta']:+.5f} (95% CI [{shuffle['mcc_ci_low']:+.5f}, {shuffle['mcc_ci_high']:+.5f}]).",
            f"- The cross-geometry-trained Teacher contribution is D1−D2 delta MCC "
            f"{cg['mcc_delta']:+.5f} (95% CI [{cg['mcc_ci_low']:+.5f}, {cg['mcc_ci_high']:+.5f}]).",
            "- These development results do not authorize a held-out-test claim.",
            "",
        ]
    )
    (OUTPUT / "CLINICAL_TWO_LEVEL_KD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "held_out_test_accessed": False}))


if __name__ == "__main__":
    main()
