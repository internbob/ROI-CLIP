#!/usr/bin/env python3
"""Analyze the development-only S4 clinical-fusion ablation.

The registry is deliberately explicit.  Dry-runs and held-out test artifacts
are never discovered implicitly.  All metrics are recomputed from saved
validation probabilities, and all paired comparisons require identical IDs
and labels within each fold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from recompute_ablation_classification_metrics import (
    CLASS_NAMES,
    METRIC_KEYS,
    calculate,
    mean_sd,
    read_flat_options,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "vnext_graph_student_clinical_20261020" / "analysis"
AUDIT_DIR = ROOT / "outputs" / "vnext_clinical_leakage_audit_s4c_20261020"

RUNS = (
    {
        "variant": "C0",
        "description": "S4 MRI-only locked baseline",
        "run": "outputs/vnext_graph_student_20261019/s4_roi_kd_corrected",
        "config": "configs/runtime_vnext_graph_student_locked_s4.yaml",
        "clinical_arm": None,
    },
    {
        "variant": "C1",
        "description": "S4 + safe background clinical input",
        "run": "outputs/vnext_graph_student_clinical_20261020/c1_safe_gated",
        "config": "configs/runtime_vnext_graph_student_s4c_safe_gated.yaml",
        "clinical_arm": "safe_background",
    },
    {
        "variant": "C2",
        "description": "S4 + safe background + neuropsychiatric input",
        "run": "outputs/vnext_graph_student_clinical_20261020/c2_safe_neuro_gated",
        "config": "configs/runtime_vnext_graph_student_s4c_safe_neuro_gated.yaml",
        "clinical_arm": "safe_neuropsychiatric",
    },
    {
        "variant": "C3",
        "description": "S4 + cognitive/FAQ input (high-risk upper bound)",
        "run": "outputs/vnext_graph_student_clinical_20261020/c3_cognition_gated",
        "config": "configs/runtime_vnext_graph_student_s4c_cognition_gated.yaml",
        "clinical_arm": "high_risk_cognition",
    },
    {
        "variant": "C4",
        "description": "S4 + all 29 non-MMSE clinical variables (high-risk upper bound)",
        "run": "outputs/vnext_graph_student_clinical_20261020/c4_full29_no_mmse_gated",
        "config": "configs/runtime_vnext_graph_student_s4c_full29_no_mmse_gated.yaml",
        "clinical_arm": "current_clinical_29",
    },
)


def subject_id(filename: str) -> str:
    return filename.split("_", 1)[0]


def load_run(run: dict, fold: int) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    fold_dir = ROOT / run["run"] / f"fold_{fold}"
    pred_path = fold_dir / "validation_predictions.csv"
    metric_path = fold_dir / "development_metrics.json"
    if not pred_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError(f"missing formal fold artifact: {pred_path} or {metric_path}")
    with pred_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row["filename"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row[f"score_{index}"]) for index in range(len(CLASS_NAMES))] for row in rows],
        dtype=np.float64,
    )
    saved = np.asarray([int(row["pred"]) for row in rows], dtype=np.int64)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {pred_path}")
    if probs.shape != (len(labels), len(CLASS_NAMES)) or not np.isfinite(probs).all():
        raise ValueError(f"invalid probability array in {pred_path}")
    if np.any(probs < 0) or not np.allclose(probs.sum(1), 1.0, atol=2e-5):
        raise ValueError(f"invalid probability simplex in {pred_path}")
    if not np.array_equal(saved, probs.argmax(1)):
        raise ValueError(f"prediction/argmax mismatch in {pred_path}")
    meta = json.loads(metric_path.read_text())
    if meta.get("held_out_test_evaluated") is not False:
        raise ValueError(f"held-out test flag is not false in {metric_path}")
    return ids, labels, probs, meta


def load_clinical_audit() -> dict[str, dict[int, tuple[list[str], np.ndarray, np.ndarray]]]:
    path = AUDIT_DIR / "oof_predictions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[int, tuple[list[str], np.ndarray, np.ndarray]]] = {}
    for arm in sorted({run["clinical_arm"] for run in RUNS if run["clinical_arm"]}):
        by_fold = {}
        for fold in range(3):
            selected = [row for row in rows if int(row["fold"]) == fold]
            ids = [str(row["filename"]) for row in selected]
            labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
            probs = np.asarray([json.loads(row[f"prob_{arm}"]) for row in selected], dtype=np.float64)
            by_fold[fold] = (ids, labels, probs)
        result[arm] = by_fold
    return result


def paired_bootstrap(
    ids: list[str],
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict:
    subjects = np.asarray([subject_id(value) for value in ids], dtype=object)
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=object)
    indices = {value: np.flatnonzero(subjects == value) for value in unique}
    rng = np.random.default_rng(seed)
    distributions = {"mcc": [], "auc": [], "acc": []}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[value] for value in sampled])
        if len(np.unique(labels[take])) < len(CLASS_NAMES):
            continue
        cand_metrics, _ = calculate(labels[take], candidate[take])
        base_metrics, _ = calculate(labels[take], baseline[take])
        for key in distributions:
            distributions[key].append(cand_metrics[key] - base_metrics[key])
    cand_metrics, _ = calculate(labels, candidate)
    base_metrics, _ = calculate(labels, baseline)
    result = {}
    for key, values in distributions.items():
        values_array = np.asarray(values, dtype=np.float64)
        result[f"pooled_{key}_delta"] = cand_metrics[key] - base_metrics[key]
        result[f"bootstrap_{key}_ci_low"] = float(np.quantile(values_array, 0.025))
        result[f"bootstrap_{key}_ci_high"] = float(np.quantile(values_array, 0.975))
    cand_correct = candidate.argmax(1) == labels
    base_correct = baseline.argmax(1) == labels
    candidate_only = int(np.sum(cand_correct & ~base_correct))
    baseline_only = int(np.sum(~cand_correct & base_correct))
    result.update(
        {
            "n": int(len(labels)),
            "unique_subjects": int(len(unique)),
            "bootstrap_replicates_valid": int(len(distributions["mcc"])),
            "candidate_only_correct": candidate_only,
            "baseline_only_correct": baseline_only,
            "mcnemar_exact_p": float(
                binomtest(candidate_only, candidate_only + baseline_only, 0.5).pvalue
                if candidate_only + baseline_only
                else 1.0
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_run: dict[str, dict[int, tuple[list[str], np.ndarray, np.ndarray]]] = {}
    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    class_summary_rows: list[dict] = []
    manifest_rows: list[dict] = []
    reference: dict[int, tuple[list[str], list[int]]] = {}

    for run in RUNS:
        variant = run["variant"]
        by_run[variant] = {}
        fold_metrics = []
        for fold in range(3):
            ids, labels, probs, _ = load_run(run, fold)
            signature = (ids, labels.tolist())
            if fold not in reference:
                reference[fold] = signature
            elif reference[fold] != signature:
                raise ValueError(f"split mismatch for {variant} fold {fold}")
            by_run[variant][fold] = (ids, labels, probs)
            metrics, per_class = calculate(labels, probs)
            fold_metrics.append(metrics)
            fold_rows.append({"variant": variant, "fold": fold, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
            for item in per_class:
                class_rows.append({"variant": variant, "fold": fold, **item})
        summary = {"variant": variant, "description": run["description"], "folds": 3}
        for key in METRIC_KEYS:
            mean, sd = mean_sd([float(row[key]) for row in fold_metrics])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd
        summary_rows.append(summary)

        for class_index, class_name in enumerate(CLASS_NAMES):
            rows = [row for row in class_rows if row["variant"] == variant and row["class_index"] == class_index]
            item = {
                "variant": variant,
                "class_index": class_index,
                "class_name": class_name,
                "support_total": sum(int(row["support"]) for row in rows),
            }
            for key in ("precision", "sensitivity", "recall", "specificity", "f1", "auc_ovr"):
                mean, sd = mean_sd([float(row[key]) for row in rows])
                item[f"{key}_mean"] = mean
                item[f"{key}_sd"] = sd
            class_summary_rows.append(item)

        opt_path = ROOT / run["run"] / "opt.txt"
        options = read_flat_options(opt_path)
        clinical_features = options.get("data.clinical.features", "[]")
        if "mmse" in clinical_features.lower():
            raise ValueError(f"MMSE found in parsed clinical features for {run['config']}")
        manifest_rows.append(
            {
                "variant": variant,
                "description": run["description"],
                "runtime_config": run["config"],
                "clinical_enabled": options.get("data.clinical.enabled"),
                "teacher_use_clinical": options.get("model.use_clinical"),
                "student_use_clinical": options.get("model.student_use_clinical", "False"),
                "clinical_fusion_mode": options.get("model.clinical_fusion_mode", "none"),
                "clinical_features": clinical_features,
                "evaluate_test": options.get("runtime.evaluate_test"),
                "selection_metric": options.get("training.selection_metric"),
            }
        )

    paired_rows = []
    base_fold = by_run["C0"]
    for run_index, run in enumerate(RUNS[1:], start=1):
        variant = run["variant"]
        fold_deltas = {key: [] for key in METRIC_KEYS}
        all_ids, all_labels, all_cand, all_base = [], [], [], []
        for fold in range(3):
            ids, labels, candidate = by_run[variant][fold]
            base_ids, base_labels, baseline = base_fold[fold]
            if ids != base_ids or not np.array_equal(labels, base_labels):
                raise ValueError(f"pairing mismatch: {variant} fold {fold}")
            candidate_metrics, _ = calculate(labels, candidate)
            baseline_metrics, _ = calculate(labels, baseline)
            for key in METRIC_KEYS:
                fold_deltas[key].append(candidate_metrics[key] - baseline_metrics[key])
            all_ids.extend(ids)
            all_labels.append(labels)
            all_cand.append(candidate)
            all_base.append(baseline)
        row = {"candidate": variant, "baseline": "C0"}
        for key, values in fold_deltas.items():
            mean, sd = mean_sd(values)
            row[f"fold_{key}_delta_mean"] = mean
            row[f"fold_{key}_delta_sd"] = sd
        row.update(
            paired_bootstrap(
                all_ids,
                np.concatenate(all_labels),
                np.concatenate(all_cand),
                np.concatenate(all_base),
                seed=3407 + run_index,
                replicates=args.bootstrap_replicates,
            )
        )
        paired_rows.append(row)

    audit = load_clinical_audit()
    clinical_rows = []
    fusion_rows = []
    for run_index, run in enumerate(RUNS[1:], start=1):
        variant, arm = run["variant"], run["clinical_arm"]
        arm_metrics = []
        all_ids, all_labels, all_fusion, all_clinical = [], [], [], []
        for fold in range(3):
            ids, labels, fusion_probs = by_run[variant][fold]
            audit_ids, audit_labels, clinical_probs = audit[arm][fold]
            lookup = {value: index for index, value in enumerate(audit_ids)}
            # The clinical-only audit starts from label CSV rows, whereas the
            # neural model additionally requires an available imaging/cache
            # sample.  Require every model ID to exist in the audit and then
            # restrict the clinical-only comparator to that exact model set.
            if not set(ids).issubset(set(audit_ids)):
                raise ValueError(f"clinical pairing is missing model IDs: {variant}/{arm} fold {fold}")
            order = np.asarray([lookup[value] for value in ids], dtype=np.int64)
            if not np.array_equal(labels, audit_labels[order]):
                raise ValueError(f"clinical pairing label mismatch: {variant}/{arm} fold {fold}")
            clinical_probs = clinical_probs[order]
            metrics, _ = calculate(labels, clinical_probs)
            arm_metrics.append(metrics)
            clinical_rows.append(
                {
                    "arm": arm,
                    "paired_variant": variant,
                    "fold": fold,
                    "audit_rows_excluded_without_model_sample": len(audit_ids) - len(ids),
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                }
            )
            all_ids.extend(ids)
            all_labels.append(labels)
            all_fusion.append(fusion_probs)
            all_clinical.append(clinical_probs)
        row = {"candidate": variant, "baseline": f"clinical-only:{arm}"}
        for key in METRIC_KEYS:
            values = []
            for fold in range(3):
                fusion_metrics, _ = calculate(by_run[variant][fold][1], by_run[variant][fold][2])
                values.append(fusion_metrics[key] - arm_metrics[fold][key])
            mean, sd = mean_sd(values)
            row[f"fold_{key}_delta_mean"] = mean
            row[f"fold_{key}_delta_sd"] = sd
        row.update(
            paired_bootstrap(
                all_ids,
                np.concatenate(all_labels),
                np.concatenate(all_fusion),
                np.concatenate(all_clinical),
                seed=4407 + run_index,
                replicates=args.bootstrap_replicates,
            )
        )
        fusion_rows.append(row)

    write_csv(output_dir / "s4_clinical_metrics_by_fold.csv", fold_rows)
    write_csv(output_dir / "s4_clinical_metrics_summary.csv", summary_rows)
    write_csv(output_dir / "s4_clinical_per_class_by_fold.csv", class_rows)
    write_csv(output_dir / "s4_clinical_per_class_summary.csv", class_summary_rows)
    write_csv(output_dir / "s4_clinical_config_manifest.csv", manifest_rows)
    write_csv(output_dir / "paired_vs_s4_mri_only.csv", paired_rows)
    write_csv(output_dir / "matched_clinical_only_by_fold.csv", clinical_rows)
    write_csv(output_dir / "fusion_vs_matched_clinical_only.csv", fusion_rows)
    payload = {
        "scope": "development folds 0,1,2 only",
        "held_out_test_accessed": False,
        "mmse_included": False,
        "bootstrap_unit": "subject",
        "bootstrap_replicates": args.bootstrap_replicates,
        "summary": summary_rows,
        "paired_vs_s4": paired_rows,
        "fusion_vs_clinical_only": fusion_rows,
    }
    (output_dir / "s4_clinical_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "held_out_test_accessed": False, "mmse_included": False}, indent=2))


if __name__ == "__main__":
    main()
