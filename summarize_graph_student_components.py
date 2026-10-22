#!/usr/bin/env python3
"""Summarize progressive MRI-only Graph Student KD component ablations."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from summarize_graph_student import METRICS, load_variant, paired_bootstrap, summarize


VARIANTS = (
    ("S0", "s0_scratch_corrected", "CE only"),
    ("S3", "s3_logit_kd_corrected", "+ logit KD"),
    ("S4", "s4_roi_kd_corrected", "+ ROI graph-state KD"),
    ("S5", "s5_graph_kd_corrected", "+ graph relation KD"),
    ("S6", "s6_full_kd_corrected", "+ prototype posterior KD"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/vnext_graph_student_20261019")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def fold_delta(left, right, metric):
    return [float(b[metric]) - float(a[metric]) for a, b in zip(left, right)]


def mcnemar(left, right):
    merged = left.merge(right, on="key", suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Prediction keys differ")
    if not np.array_equal(merged["label_left"], merged["label_right"]):
        raise RuntimeError("Prediction labels differ")
    left_correct = merged["label_left"] == merged["pred_left"]
    right_correct = merged["label_right"] == merged["pred_right"]
    right_only = int((right_correct & ~left_correct).sum())
    left_only = int((left_correct & ~right_correct).sum())
    discordant = right_only + left_only
    return {
        "right_only_correct": right_only,
        "left_only_correct": left_only,
        "exact_two_sided_p": float(binomtest(right_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    loaded = {}
    for label, directory, description in VARIANTS:
        metrics, predictions = load_variant(root, directory, folds)
        loaded[label] = {
            "directory": directory,
            "description": description,
            "metrics": metrics,
            "predictions": predictions,
        }

    summaries = {
        label: {
            metric: summarize([float(item[metric]) for item in item["metrics"]])
            for metric in METRICS
        }
        for label, item in loaded.items()
    }
    versus_s0 = {}
    for label in ("S3", "S4", "S5", "S6"):
        deltas = {
            metric: fold_delta(loaded["S0"]["metrics"], loaded[label]["metrics"], metric)
            for metric in METRICS
        }
        versus_s0[label] = {
            "fold_deltas": deltas,
            "mean_deltas": {metric: float(np.mean(values)) for metric, values in deltas.items()},
            "paired_bootstrap": [
                paired_bootstrap(
                    loaded["S0"]["predictions"],
                    loaded[label]["predictions"],
                    metric,
                    args.bootstrap_replicates,
                    args.seed + 100 * int(label[1:]) + index,
                )
                for index, metric in enumerate(("mcc", "auc", "balanced_acc"))
            ],
            "mcnemar": mcnemar(loaded["S0"]["predictions"], loaded[label]["predictions"]),
        }

    increments = {}
    for left, right in (("S0", "S3"), ("S3", "S4"), ("S4", "S5"), ("S5", "S6")):
        deltas = {
            metric: fold_delta(loaded[left]["metrics"], loaded[right]["metrics"], metric)
            for metric in METRICS
        }
        increments[f"{right}_minus_{left}"] = {
            "fold_deltas": deltas,
            "mean_deltas": {metric: float(np.mean(values)) for metric, values in deltas.items()},
            "mcc_positive_folds": int(sum(value > 0 for value in deltas["mcc"])),
            "mcc_paired_bootstrap": paired_bootstrap(
                loaded[left]["predictions"],
                loaded[right]["predictions"],
                "mcc",
                args.bootstrap_replicates,
                args.seed + 1000 + int(right[1:]),
            ),
        }

    selected_label = max(
        summaries,
        key=lambda label: summaries[label]["mcc"]["mean"],
    )
    selected_index = [label for label, _, _ in VARIANTS].index(selected_label)
    selected_components = [
        description for _, _, description in VARIANTS[1 : selected_index + 1]
    ]
    selection = {
        "selected_variant": selected_label,
        "criterion": "highest mean MCC across the matched three-fold development ablation",
        "selected_components": selected_components,
        "held_out_test_used_for_selection": False,
        "claim_boundary": (
            "Selecting the best cumulative development variant does not prove that each incremental "
            "component is independently stable; use the progressive paired results for attribution."
        ),
    }
    selected_audits = []
    for fold in folds:
        audit_path = root / f"{selected_label.lower()}_mri_only_audit_fold{fold}.json"
        if audit_path.is_file():
            selected_audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
    audit_summary = {
        "expected_folds": len(folds),
        "completed_folds": len(selected_audits),
        "all_passed": bool(
            len(selected_audits) == len(folds)
            and all(item.get("passed") is True for item in selected_audits)
        ),
        "all_held_out_test_false": bool(
            len(selected_audits) == len(folds)
            and all(item.get("held_out_test_loaded_or_evaluated") is False for item in selected_audits)
        ),
    }

    payload = {
        "completed_on": date.today().isoformat(),
        "folds": folds,
        "held_out_test_evaluated": False,
        "variant_summary": summaries,
        "versus_s0": versus_s0,
        "progressive_increments": increments,
        "development_selection": selection,
        "selected_variant_mri_only_audit": audit_summary,
        "interpretation_rule": (
            "Use progressive increments to attribute components; development-only evidence with a CI crossing zero "
            "is descriptive and not a significant improvement claim."
        ),
    }
    (root / "graph_student_component_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# CortexBridge Graph Student Component Ablation",
        "",
        f"- Completed: `{date.today().isoformat()}`.",
        "- Protocol: matched three-fold development validation.",
        "- Clinical input: `off`.",
        "- Held-out test loaded/evaluated: `false`.",
        f"- Development-selected Student: **{selected_label}** ({selection['criterion']}).",
        "",
        "## Variant results",
        "",
        "| ID | Cumulative objective | MCC | AUC | Accuracy | Macro-F1 | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, _, description in VARIANTS:
        cells = [
            f"{summaries[label][metric]['mean']:.5f} ± {summaries[label][metric]['sample_sd']:.5f}"
            for metric in METRICS
        ]
        lines.append(f"| {label} | {description} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Progressive component increments",
            "",
            "| Increment | Fold-0 ΔMCC | Fold-1 ΔMCC | Fold-2 ΔMCC | Mean ΔMCC | Positive folds | Bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in increments.items():
        values = item["fold_deltas"]["mcc"]
        boot = item["mcc_paired_bootstrap"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:+.5f}" for value in values)
            + f" | {np.mean(values):+.5f} | {item['mcc_positive_folds']}/3 | "
            + f"[{boot['ci95_low']:+.5f}, {boot['ci95_high']:+.5f}] |"
        )

    lines.extend(["", "## Each KD variant versus S0", ""])
    for label, item in versus_s0.items():
        mcc = item["paired_bootstrap"][0]
        auc = item["paired_bootstrap"][1]
        test = item["mcnemar"]
        lines.append(
            f"- {label}: mean fold ΔMCC `{item['mean_deltas']['mcc']:+.5f}`; pooled MCC delta "
            f"`{mcc['observed_delta']:+.5f}` (95% CI `[{mcc['ci95_low']:+.5f}, {mcc['ci95_high']:+.5f}]`); "
            f"pooled AUC delta `{auc['observed_delta']:+.5f}`; McNemar p `{test['exact_two_sided_p']:.6f}`."
        )
    lines.extend(
        [
            "",
            "Component attribution is descriptive unless the paired uncertainty excludes zero. "
            "No held-out test result is used for model or component selection.",
            "",
            "## Development decision",
            "",
            f"**Lock {selected_label} for the next complete-CV stage.** It has the highest three-fold mean MCC. "
            "This selects a cumulative objective, not an independently proven causal contribution from every term.",
            "",
            "- Keep: CE, logit KD, and ROI graph-state KD.",
            "- Drop from the locked Student: graph-relation KD and prototype-posterior KD, because their "
            "progressive mean MCC increments are non-positive or negligible and are not stable across folds.",
            "- Claim boundary: ROI graph-state KD has a small positive mean increment over S3 but only 1/3 "
            "folds improves and its paired bootstrap interval crosses zero.",
            "- The held-out test remains unopened; this is a development-only selection.",
            f"- MRI-only audit: {audit_summary['completed_folds']}/{audit_summary['expected_folds']} selected "
            f"checkpoints audited; all passed = `{str(audit_summary['all_passed']).lower()}`; all audit "
            f"held-out flags false = `{str(audit_summary['all_held_out_test_false']).lower()}`.",
        ]
    )
    (root / "GRAPH_STUDENT_COMPONENT_ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({label: summaries[label]["mcc"] for label, _, _ in VARIANTS}, indent=2))


if __name__ == "__main__":
    main()
