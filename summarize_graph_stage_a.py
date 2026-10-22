#!/usr/bin/env python3
"""Summarize development-only CortexBridge graph ablations.

This script only reads ``development_metrics.json`` and validation prediction
artifacts. It never opens a held-out test CSV or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from metrics import compute_classification_metrics


VARIANTS = ("g0", "g1", "g2", "g3")
CLASSIFICATION_METRICS = ("mcc", "auc", "acc", "f1_macro", "balanced_acc")
GRAPH_METRICS = (
    "graph_anatomical_mass",
    "graph_homology_mass",
    "graph_identity_mass",
    "graph_remote_mass",
    "graph_attention_entropy",
    "graph_attention_uniform_kl",
    "graph_attention_std",
    "graph_boundary_correlation",
    "graph_uniform_anatomical_mass",
    "graph_uniform_homology_mass",
    "graph_residual_gate",
    "graph_output_delta_l2",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/vnext_graph_stageA_20261019")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(root, folds):
    metrics = {}
    predictions = {}
    missing = []
    for variant in VARIANTS:
        predictions[variant] = []
        for fold in folds:
            fold_root = root / f"{variant}_fold{fold}" / f"fold_{fold}"
            metrics_path = fold_root / "development_metrics.json"
            predictions_path = fold_root / "validation_predictions.json"
            if not metrics_path.is_file() or not predictions_path.is_file():
                missing.extend(str(path) for path in (metrics_path, predictions_path) if not path.is_file())
                continue
            payload = read_json(metrics_path)
            if bool(payload.get("held_out_test_evaluated", True)):
                raise RuntimeError(f"Held-out test flag is not false: {metrics_path}")
            metrics[(variant, fold)] = payload
            prediction_payload = read_json(predictions_path)
            if prediction_payload.get("split") != "validation":
                raise RuntimeError(f"Non-validation predictions found: {predictions_path}")
            predictions[variant].extend(prediction_payload["records"])
    if missing:
        raise FileNotFoundError("Stage A is incomplete:\n" + "\n".join(missing))
    return metrics, predictions


def align_predictions(predictions):
    reference = {record["filename"]: record for record in predictions["g0"]}
    aligned = {"g0": reference}
    for variant in VARIANTS[1:]:
        current = {record["filename"]: record for record in predictions[variant]}
        if set(current) != set(reference):
            raise RuntimeError(f"Validation prediction keys differ for {variant}")
        for key in reference:
            if int(current[key]["label"]) != int(reference[key]["label"]):
                raise RuntimeError(f"Validation label mismatch for {variant}: {key}")
        aligned[variant] = current
    return aligned


def evaluate_records(records, keys):
    labels = np.asarray([int(records[key]["label"]) for key in keys], dtype=np.int64)
    probabilities = np.asarray([records[key]["probabilities"] for key in keys], dtype=np.float64)
    return compute_classification_metrics(labels, probabilities)


def paired_subject_bootstrap(aligned, candidate, metric, replicates, seed):
    reference = aligned["g0"]
    subject_keys = {}
    for key, record in reference.items():
        subject_keys.setdefault(str(record["subject"]), []).append(key)
    subjects = np.asarray(sorted(subject_keys), dtype=object)
    all_keys = sorted(reference)
    point = float(evaluate_records(aligned[candidate], all_keys)[metric]) - float(
        evaluate_records(reference, all_keys)[metric]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sampled_keys = [key for subject in sampled_subjects for key in subject_keys[str(subject)]]
        samples[index] = float(evaluate_records(aligned[candidate], sampled_keys)[metric]) - float(
            evaluate_records(reference, sampled_keys)[metric]
        )
    return {
        "variant": candidate,
        "metric": metric,
        "difference": point,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "subjects": int(len(subjects)),
        "replicates": int(replicates),
    }


def mcnemar(aligned, candidate):
    keys = sorted(aligned["g0"])
    baseline_correct = np.asarray(
        [int(aligned["g0"][key]["prediction"]) == int(aligned["g0"][key]["label"]) for key in keys]
    )
    candidate_correct = np.asarray(
        [int(aligned[candidate][key]["prediction"]) == int(aligned[candidate][key]["label"]) for key in keys]
    )
    baseline_only = int(np.sum(baseline_correct & ~candidate_correct))
    candidate_only = int(np.sum(~baseline_correct & candidate_correct))
    discordant = baseline_only + candidate_only
    p_value = float(binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "variant": candidate,
        "g0_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "p_value": p_value,
    }


def holm_adjust(items, key="p_value"):
    order = sorted(range(len(items)), key=lambda index: items[index][key])
    adjusted = [1.0] * len(items)
    running = 0.0
    count = len(items)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(items[index][key]))
        running = max(running, value)
        adjusted[index] = running
    for item, value in zip(items, adjusted):
        item["holm_p_value"] = value
    return items


def summarize(metrics, folds):
    rows = []
    for variant in VARIANTS:
        for fold in folds:
            payload = metrics[(variant, fold)]
            row = {"variant": variant, "fold": fold, "best_epoch": payload["best_epoch"]}
            row.update(payload["best_val_metrics"])
            row["gpu_peak_memory_mb"] = payload["gpu_peak_memory_mb"]
            row["fold_total_seconds"] = payload["fold_total_seconds"]
            rows.append(row)
    summaries = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        summaries[variant] = {}
        for metric in CLASSIFICATION_METRICS + GRAPH_METRICS:
            values = np.asarray([float(row.get(metric, 0.0)) for row in variant_rows], dtype=np.float64)
            summaries[variant][metric] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return rows, summaries


def assess_gate(summaries):
    baseline = np.asarray(summaries["g0"]["mcc"]["values"])
    assessments = {}
    for variant in VARIANTS[1:]:
        values = np.asarray(summaries[variant]["mcc"]["values"])
        deltas = values - baseline
        diagnostics = summaries[variant]
        assessments[variant] = {
            "mcc_deltas": deltas.tolist(),
            "mean_mcc_delta": float(deltas.mean()),
            "improved_folds": int(np.sum(deltas > 0.0)),
            "no_stable_decline": bool(deltas.mean() >= -0.005 and np.sum(deltas >= 0.0) >= 1),
            "non_uniform": bool(diagnostics["graph_attention_uniform_kl"]["mean"] > 1e-4),
            "not_identity_only": bool(diagnostics["graph_identity_mass"]["mean"] < 0.5),
            "not_topology_copy": bool(
                abs(diagnostics["graph_boundary_correlation"]["mean"]) < 0.98
                and diagnostics["graph_remote_mass"]["mean"] > 0.05
            ),
            "nonzero_classifier_input_contribution": bool(
                diagnostics["graph_output_delta_l2"]["mean"] > 1e-4
            ),
        }
    topology_candidates = []
    for variant in ("g2", "g3"):
        item = assessments[variant]
        if (
            item["no_stable_decline"]
            and item["improved_folds"] >= 2
            and item["mean_mcc_delta"] > 0.0
            and item["non_uniform"]
            and item["not_identity_only"]
            and item["not_topology_copy"]
            and item["nonzero_classifier_input_contribution"]
        ):
            topology_candidates.append(variant)
    selected = max(topology_candidates, key=lambda name: assessments[name]["mean_mcc_delta"]) if topology_candidates else None
    return {
        "variants": assessments,
        "graph_teacher_gate_passed": selected is not None,
        "selected_topology_variant": selected,
        "next_required_control": "G2-R/G3-R topology randomization" if selected else "none; stop graph escalation",
        "student_stage_authorized": selected is not None,
    }


def fmt(value):
    return f"{float(value):.5f}"


def write_report(root, folds, summaries, comparisons, mcnemar_results, gate):
    lines = [
        "# CortexBridge vNext — Disease-Aware Graph Stage A",
        "",
        "## Protocol",
        "",
        f"- Development folds: `{folds}`.",
        "- Clinical input: `off` for every arm, based on the frozen leakage audit.",
        "- Held-out test loaded/evaluated: `false`.",
        "- G0: hard cross-geometry baseline; G1: unbiased graph; G2: boundary bias; G3: boundary + homology bias.",
        "",
        "## Three-fold classification",
        "",
        "| Variant | MCC | AUC | ACC | Macro-F1 | Balanced ACC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        summary = summaries[variant]
        lines.append(
            f"| {variant.upper()} | {fmt(summary['mcc']['mean'])} ± {fmt(summary['mcc']['sample_sd'])} | "
            f"{fmt(summary['auc']['mean'])} ± {fmt(summary['auc']['sample_sd'])} | "
            f"{fmt(summary['acc']['mean'])} ± {fmt(summary['acc']['sample_sd'])} | "
            f"{fmt(summary['f1_macro']['mean'])} ± {fmt(summary['f1_macro']['sample_sd'])} | "
            f"{fmt(summary['balanced_acc']['mean'])} ± {fmt(summary['balanced_acc']['sample_sd'])} |"
        )
    lines.extend(
        [
            "",
            "## Graph diagnostics",
            "",
            "| Variant | Anatomy mass | Homology mass | Remote mass | Uniform KL | Boundary corr. | Gate | Output delta L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS[1:]:
        summary = summaries[variant]
        lines.append(
            f"| {variant.upper()} | {fmt(summary['graph_anatomical_mass']['mean'])} | "
            f"{fmt(summary['graph_homology_mass']['mean'])} | {fmt(summary['graph_remote_mass']['mean'])} | "
            f"{fmt(summary['graph_attention_uniform_kl']['mean'])} | "
            f"{fmt(summary['graph_boundary_correlation']['mean'])} | "
            f"{fmt(summary['graph_residual_gate']['mean'])} | "
            f"{fmt(summary['graph_output_delta_l2']['mean'])} |"
        )
    lines.extend(["", "## Paired validation comparisons", ""])
    lines.append("| Variant − G0 | Metric | Difference | 95% subject-bootstrap CI |")
    lines.append("|---|---|---:|---:|")
    for item in comparisons:
        lines.append(
            f"| {item['variant'].upper()} − G0 | {item['metric']} | {fmt(item['difference'])} | "
            f"[{fmt(item['ci_low'])}, {fmt(item['ci_high'])}] |"
        )
    lines.extend(["", "Exact McNemar tests use paired validation predictions and Holm correction:", ""])
    for item in mcnemar_results:
        lines.append(
            f"- {item['variant'].upper()} vs G0: G0-only `{item['g0_only_correct']}`, "
            f"candidate-only `{item['candidate_only_correct']}`, exact p `{item['p_value']:.6g}`, "
            f"Holm p `{item['holm_p_value']:.6g}`."
        )
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            f"- Graph Teacher gate passed: `{str(gate['graph_teacher_gate_passed']).lower()}`.",
            f"- Selected topology variant: `{gate['selected_topology_variant']}`.",
            f"- Next required control: `{gate['next_required_control']}`.",
            f"- Student stage authorized: `{str(gate['student_stage_authorized']).lower()}`.",
            "",
            "This report does not implement or start Student KD. A topology-randomization control remains required before treating an anatomical bias as causal evidence.",
        ]
    )
    (root / "GRAPH_STAGE_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.input_dir).resolve()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    metrics, predictions = load_results(root, folds)
    aligned = align_predictions(predictions)
    rows, summaries = summarize(metrics, folds)

    comparisons = []
    for variant_index, variant in enumerate(VARIANTS[1:]):
        for metric_index, metric in enumerate(("mcc", "auc", "balanced_acc")):
            comparisons.append(
                paired_subject_bootstrap(
                    aligned,
                    candidate=variant,
                    metric=metric,
                    replicates=int(args.bootstrap_replicates),
                    seed=int(args.seed) + variant_index * 10 + metric_index,
                )
            )
    mcnemar_results = holm_adjust([mcnemar(aligned, variant) for variant in VARIANTS[1:]])
    fold_tests = []
    baseline = np.asarray(summaries["g0"]["mcc"]["values"])
    for variant in VARIANTS[1:]:
        candidate = np.asarray(summaries[variant]["mcc"]["values"])
        try:
            p_value = float(wilcoxon(candidate, baseline, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        fold_tests.append({"variant": variant, "test": "paired_wilcoxon_mcc", "p_value": p_value})
    gate = assess_gate(summaries)

    root.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (root / "graph_fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "folds": folds,
        "held_out_test_evaluated": False,
        "summaries": summaries,
        "paired_bootstrap": comparisons,
        "mcnemar": mcnemar_results,
        "fold_tests": fold_tests,
        "gate": gate,
    }
    with (root / "graph_stage_a_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_report(root, folds, summaries, comparisons, mcnemar_results, gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
