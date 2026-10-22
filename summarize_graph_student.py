#!/usr/bin/env python3
"""Summarize matched MRI-only Graph Student scratch and KD experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score


METRICS = ("mcc", "auc", "acc", "f1_macro", "balanced_acc")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/vnext_graph_student_20261019")
    parser.add_argument("--baseline", default="s0_scratch")
    parser.add_argument("--candidate", default="s6_full_kd")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_variant(root: Path, variant: str, folds: list[int]):
    fold_metrics = []
    frames = []
    for fold in folds:
        fold_root = root / variant / f"fold_{fold}"
        with (fold_root / "development_metrics.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if bool(payload.get("held_out_test_evaluated", True)):
            raise RuntimeError(f"Held-out test flag is not false: {fold_root}")
        fold_metrics.append(payload["metrics"])
        frame = pd.read_csv(fold_root / "validation_predictions.csv")
        frame["fold"] = fold
        frame["key"] = frame["fold"].astype(str) + ":" + frame["filename"].astype(str)
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    if pooled["key"].duplicated().any():
        raise RuntimeError(f"Duplicate prediction keys for {variant}")
    return fold_metrics, pooled


def metric_from_frame(frame: pd.DataFrame, metric: str) -> float:
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["pred"].to_numpy(dtype=int)
    if metric == "mcc":
        return float(matthews_corrcoef(labels, predictions))
    if metric == "balanced_acc":
        return float(balanced_accuracy_score(labels, predictions))
    if metric == "auc":
        score_columns = sorted(
            (column for column in frame.columns if column.startswith("score_")),
            key=lambda value: int(value.split("_")[-1]),
        )
        return float(
            roc_auc_score(
                labels,
                frame[score_columns].to_numpy(dtype=float),
                labels=list(range(len(score_columns))),
                multi_class="ovr",
                average="macro",
            )
        )
    raise ValueError(metric)


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    replicates: int,
    seed: int,
):
    merged = baseline.merge(candidate, on="key", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise RuntimeError("Baseline and candidate prediction keys differ")
    if not np.array_equal(merged["label_baseline"], merged["label_candidate"]):
        raise RuntimeError("Baseline and candidate labels differ")

    def select(suffix: str, indices=None):
        columns = {
            "label": merged[f"label_{suffix}"],
            "pred": merged[f"pred_{suffix}"],
        }
        for column in baseline.columns:
            if column.startswith("score_"):
                columns[column] = merged[f"{column}_{suffix}"]
        frame = pd.DataFrame(columns)
        return frame if indices is None else frame.iloc[indices]

    observed = metric_from_frame(select("candidate"), metric) - metric_from_frame(select("baseline"), metric)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        indices = rng.integers(0, len(merged), size=len(merged))
        try:
            delta = metric_from_frame(select("candidate", indices), metric) - metric_from_frame(
                select("baseline", indices), metric
            )
        except ValueError:
            continue
        deltas.append(delta)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "metric": metric,
        "observed_delta": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "valid_replicates": len(deltas),
    }


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1))}


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    base_metrics, base_predictions = load_variant(root, args.baseline, folds)
    cand_metrics, cand_predictions = load_variant(root, args.candidate, folds)

    variants = {}
    fold_deltas = {}
    for variant, fold_values in ((args.baseline, base_metrics), (args.candidate, cand_metrics)):
        variants[variant] = {
            metric: summarize([float(item[metric]) for item in fold_values]) for metric in METRICS
        }
    for metric in METRICS:
        fold_deltas[metric] = [
            float(candidate[metric]) - float(baseline[metric])
            for baseline, candidate in zip(base_metrics, cand_metrics)
        ]

    bootstraps = [
        paired_bootstrap(
            base_predictions,
            cand_predictions,
            metric,
            args.bootstrap_replicates,
            args.seed + index,
        )
        for index, metric in enumerate(("mcc", "auc", "balanced_acc"))
    ]
    merged = base_predictions.merge(
        cand_predictions, on="key", suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    base_correct = merged["label_baseline"] == merged["pred_baseline"]
    cand_correct = merged["label_candidate"] == merged["pred_candidate"]
    candidate_only = int((cand_correct & ~base_correct).sum())
    baseline_only = int((base_correct & ~cand_correct).sum())
    discordant = candidate_only + baseline_only
    mcnemar_p = float(binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0

    mean_deltas = {metric: float(np.mean(values)) for metric, values in fold_deltas.items()}
    stable_benefit = bool(
        mean_deltas["mcc"] > 0.01
        and sum(value > 0.0 for value in fold_deltas["mcc"]) >= 2
        and all(mean_deltas[metric] >= -0.005 for metric in ("auc", "acc", "f1_macro", "balanced_acc"))
    )
    payload = {
        "folds": folds,
        "held_out_test_evaluated": False,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "variant_summary": variants,
        "fold_deltas_candidate_minus_baseline": fold_deltas,
        "mean_deltas_candidate_minus_baseline": mean_deltas,
        "paired_subject_bootstrap": bootstraps,
        "mcnemar": {
            "candidate_only_correct": candidate_only,
            "baseline_only_correct": baseline_only,
            "exact_two_sided_p": mcnemar_p,
        },
        "gate": {
            "stable_distillation_benefit": stable_benefit,
            "criterion": "mean MCC delta > 0.01, MCC improves in >=2/3 folds, and no other mean primary-metric delta < -0.005",
            "component_ablation_authorized": stable_benefit,
        },
    }
    (root / "graph_student_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# CortexBridge MRI-only Graph Student",
        "",
        f"- Baseline: `{args.baseline}`.",
        f"- Full graph-state distillation: `{args.candidate}`.",
        "- Clinical input: `off`.",
        "- Subject-specific surface/PET/cognition input at Student inference: `none`.",
        "- Held-out test loaded/evaluated: `false`.",
        "",
        "## Three-fold development results",
        "",
        "| Variant | MCC | AUC | Accuracy | Macro-F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in (args.baseline, args.candidate):
        summary = variants[variant]
        cells = [f"{summary[name]['mean']:.5f} ± {summary[name]['sample_sd']:.5f}" for name in METRICS]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Candidate − baseline fold deltas",
            "",
            "| Metric | Fold 0 | Fold 1 | Fold 2 | Mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, values in fold_deltas.items():
        lines.append(
            f"| {metric} | " + " | ".join(f"{value:+.5f}" for value in values) + f" | {np.mean(values):+.5f} |"
        )
    lines.extend(["", "## Paired pooled-subject inference", ""])
    for item in bootstraps:
        lines.append(
            f"- {item['metric']}: delta `{item['observed_delta']:+.5f}`, paired bootstrap 95% CI "
            f"`[{item['ci95_low']:+.5f}, {item['ci95_high']:+.5f}]`."
        )
    lines.extend(
        [
            f"- McNemar: candidate-only `{candidate_only}`, baseline-only `{baseline_only}`, exact p `{mcnemar_p:.6f}`.",
            "",
            "## Gate",
            "",
            f"- Predeclared development gate passed: `{str(stable_benefit).lower()}`.",
            f"- Component ablations authorized: `{str(stable_benefit).lower()}`.",
            "",
            "The gate is intentionally based on matched architecture, matched subjects, and development validation only. "
            "Passing it authorizes component ablations; it is not itself a claim of statistically stable or externally "
            "generalizable benefit. A failed gate is reported as a negative result and does not justify expansion.",
        ]
    )
    (root / "GRAPH_STUDENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))


if __name__ == "__main__":
    main()
