#!/usr/bin/env python3
"""Summarize the locked E0-E6 development-only comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

from config import load_configs
from data import build_dataloaders
from metrics import compute_classification_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/vnext_unified_clinical_20261023"
CONFIGS = [
    ROOT / "configs/data.yaml",
    ROOT / "configs/student.yaml",
    ROOT / "configs/runtime_unified_clinical_common.yaml",
    ROOT / "configs/runtime_unified_student_e2_mri_clinical_ce.yaml",
]
ARMS = {
    "E1": ("MRI-only CE", OUTPUT_ROOT / "e1_student"),
    "E2": ("MRI+clinical CE", OUTPUT_ROOT / "e2_student"),
    "E3": ("MRI+clinical ordinary Teacher logit KD", OUTPUT_ROOT / "e3_student"),
    "E4": ("Full surface-privileged CortexBridge", OUTPUT_ROOT / "e4_student"),
    "E5": ("E4 Student clinical subject-shuffle", OUTPUT_ROOT / "e5_student_shuffle"),
    "E6": ("E4 with Teacher cross-geometry disabled", OUTPUT_ROOT / "e6_student_cg_off"),
}
METRICS = ("mcc", "auc", "acc", "f1_macro", "recall_macro", "spe", "mci_recall")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_predictions(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row["filename"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[f"score_{index}"]) for index in range(3)] for row in rows],
        dtype=np.float64,
    )
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate validation IDs in {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError(f"invalid probabilities in {path}")
    return ids, labels, probabilities


def metric_values(labels, probabilities):
    result = compute_classification_metrics(labels, probabilities)
    matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
    result["mci_recall"] = float(matrix[1, 1] / max(1, matrix[1].sum()))
    return result


def clinical_only_fold(fold, reference_ids, reference_labels, seed):
    cfg = load_configs([str(path) for path in CONFIGS])
    cfg.runtime.evaluate_test = False
    cfg.model.num_classes = 3
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        fold=fold,
        require_teacher_inputs=False,
        filter_teacher_available=True,
    )
    if test_loader is not None:
        raise RuntimeError("held-out test loader must remain disabled")
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    x_train = np.stack(
        [train_dataset._encode_clinical(sample) for sample in train_dataset.samples]
    ).astype(np.float64)
    y_train = np.asarray([int(sample["label"]) for sample in train_dataset.samples], dtype=np.int64)
    val_lookup = {str(sample["filename"]): sample for sample in val_dataset.samples}
    if set(reference_ids) != set(val_lookup):
        raise ValueError(f"E0/E1 validation ID mismatch in fold {fold}")
    ordered = [val_lookup[value] for value in reference_ids]
    labels = np.asarray([int(sample["label"]) for sample in ordered], dtype=np.int64)
    if not np.array_equal(labels, reference_labels):
        raise ValueError(f"E0/E1 label mismatch in fold {fold}")
    x_valid = np.stack([val_dataset._encode_clinical(sample) for sample in ordered]).astype(
        np.float64
    )
    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        random_state=int(seed) + fold,
    )
    classifier.fit(x_train, y_train)
    raw = classifier.predict_proba(x_valid)
    probabilities = np.zeros((len(ordered), 3), dtype=np.float64)
    for source_index, class_index in enumerate(classifier.classes_):
        probabilities[:, int(class_index)] = raw[:, source_index]
    return probabilities


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def paired_bootstrap(records, candidate, baseline, replicates, seed):
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    candidate_prob = np.asarray([record[candidate] for record in records], dtype=np.float64)
    baseline_prob = np.asarray([record[baseline] for record in records], dtype=np.float64)
    subjects = np.asarray([str(record["filename"]).split("_", 1)[0] for record in records])
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=object)
    indices = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    candidate_metrics = metric_values(labels, candidate_prob)
    baseline_metrics = metric_values(labels, baseline_prob)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("mcc", "auc", "acc")}
    for _ in range(int(replicates)):
        selected_subjects = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[subject] for subject in selected_subjects])
        if len(np.unique(labels[take])) < 3:
            continue
        cand = metric_values(labels[take], candidate_prob[take])
        base = metric_values(labels[take], baseline_prob[take])
        for key in samples:
            samples[key].append(float(cand[key]) - float(base[key]))
    result = {"candidate": candidate, "baseline": baseline, "n": int(len(labels))}
    for key in samples:
        distribution = np.asarray(samples[key], dtype=np.float64)
        result[f"{key}_delta"] = float(candidate_metrics[key]) - float(baseline_metrics[key])
        result[f"{key}_ci_low"] = float(np.quantile(distribution, 0.025))
        result[f"{key}_ci_high"] = float(np.quantile(distribution, 0.975))
    candidate_correct = candidate_prob.argmax(1) == labels
    baseline_correct = baseline_prob.argmax(1) == labels
    candidate_only = int(np.sum(candidate_correct & ~baseline_correct))
    baseline_only = int(np.sum(~candidate_correct & baseline_correct))
    result["mcnemar_p"] = float(
        binomtest(candidate_only, candidate_only + baseline_only, 0.5).pvalue
        if candidate_only + baseline_only
        else 1.0
    )
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(summary_rows, comparisons):
    descriptions = {"E0": "Clinical-only logistic regression", **{k: v[0] for k, v in ARMS.items()}}
    summary = {row["arm"]: row for row in summary_rows}
    contrast = {(row["candidate"], row["baseline"]): row for row in comparisons}
    e4_e2 = contrast[("E4", "E2")]
    e4_e3 = contrast[("E4", "E3")]
    e4_e6 = contrast[("E4", "E6")]
    e4_e5 = contrast[("E4", "E5")]
    lines = [
        "# CortexBridge Unified MRI–Clinical Development Comparison",
        "",
        "- Task: ADNI NC/MCI/DE three-class classification.",
        "- Folds: subject-level development folds 0–2.",
        "- Clinical set: 29 variables, MMSE excluded.",
        "- Held-out test accessed: `false`.",
        "- Student ROI/graph classification residual: disabled.",
        "",
        "## Decision",
        "",
        f"- **Primary performance configuration: E3.** It has the highest neural-model development MCC ({summary['E3']['mcc_mean']:.5f}) and is not improved by the full E4 privileged Teacher (E4−E3 ΔMCC {e4_e3['mcc_delta']:+.5f}, 95% CI [{e4_e3['mcc_ci_low']:+.5f}, {e4_e3['mcc_ci_high']:+.5f}]).",
        f"- **Mechanism-oriented CortexBridge configuration: E4.** E4 reaches MCC {summary['E4']['mcc_mean']:.5f} and AUC {summary['E4']['auc_mean']:.5f}, but its increment over direct MRI+clinical CE is small and uncertain (E4−E2 ΔMCC {e4_e2['mcc_delta']:+.5f}, 95% CI [{e4_e2['mcc_ci_low']:+.5f}, {e4_e2['mcc_ci_high']:+.5f}]).",
        f"- **No classification-superiority claim for cross-geometry.** Isolating cross-geometry gives E4−E6 ΔMCC {e4_e6['mcc_delta']:+.5f}, 95% CI [{e4_e6['mcc_ci_low']:+.5f}, {e4_e6['mcc_ci_high']:+.5f}]. Cross-geometry remains supported as correspondence/representation supervision, not as a proven classifier improvement.",
        f"- **Correct clinical pairing is essential.** E4−E5 ΔMCC is {e4_e5['mcc_delta']:+.5f}, 95% CI [{e4_e5['mcc_ci_low']:+.5f}, {e4_e5['mcc_ci_high']:+.5f}], with McNemar p={e4_e5['mcnemar_p']:.3g}.",
        "- These are development-only conclusions. No held-out test should be opened until the method, clinical-variable policy, and missing-data handling are frozen.",
        "",
        "## E0–E6 results",
        "",
        "| Arm | Description | MCC | AUC | ACC | Macro-F1 | Recall | SPE | MCI recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {descriptions[row['arm']]} | "
            f"{row['mcc_mean']:.5f} ± {row['mcc_sd']:.5f} | "
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
            "## Predeclared paired contrasts",
            "",
            "| Contrast | ΔMCC [95% CI] | ΔAUC [95% CI] | ΔACC [95% CI] | McNemar p |",
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
    lines.extend(
        [
            "",
            "Interpretation: E4−E2 is the total privileged-Teacher increment; E4−E3 is the increment over ordinary MRI–clinical KD; E4−E6 isolates hard cross-geometry training within the same surface/graph Teacher; E4−E5 tests correct Student clinical pairing; E4−E0 tests imaging plus privileged supervision beyond clinical scales alone.",
            "",
            "## Validity and deployment constraints",
            "",
            "- E2–E6 are MRI+clinical inference models, not MRI-only models. They must be compared against multimodal baselines and labeled accordingly.",
            "- The 29-variable set excludes MMSE but includes memory and functional variables. Because some may overlap diagnostic criteria, the paper must disclose them explicitly and retain E0 as a shortcut/upper-bound audit.",
            "- Missing values are represented with train-fold-fitted imputation plus missingness indicators. The separate E7/E8 audit shows substantial degradation under added missingness; see `missingness_audit/MISSINGNESS_AUDIT.md`.",
            "- E4 does not justify claiming that surface/ribbon/graph privileged knowledge raises classification accuracy over ordinary clinical KD. Its defensible role is anatomical correspondence and mechanism-oriented regularization.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_arm = {arm: {} for arm in ["E0", *ARMS]}
    fold_rows = []
    pooled = []
    for fold in range(3):
        reference_ids = reference_labels = None
        loaded = {}
        for arm, (_, run_dir) in ARMS.items():
            pred_path = run_dir / f"fold_{fold}/validation_predictions.csv"
            metric_path = run_dir / f"fold_{fold}/development_metrics.json"
            metadata = json.loads(metric_path.read_text(encoding="utf-8"))
            if metadata.get("held_out_test_evaluated") is not False:
                raise ValueError(f"held-out test flag is not false: {metric_path}")
            ids, labels, probabilities = load_predictions(pred_path)
            if reference_ids is None:
                reference_ids, reference_labels = ids, labels
            elif ids != reference_ids or not np.array_equal(labels, reference_labels):
                raise ValueError(f"fold pairing mismatch for {arm}/{fold}")
            loaded[arm] = probabilities
        loaded["E0"] = clinical_only_fold(
            fold, reference_ids, reference_labels, args.seed
        )
        for arm in ["E0", *ARMS]:
            probabilities = loaded[arm]
            metrics = metric_values(reference_labels, probabilities)
            by_arm[arm][fold] = (reference_ids, reference_labels, probabilities)
            fold_rows.append(
                {"arm": arm, "fold": fold, **{key: float(metrics[key]) for key in METRICS}}
            )
        for index, filename in enumerate(reference_ids):
            record = {"fold": fold, "filename": filename, "label": int(reference_labels[index])}
            for arm in ["E0", *ARMS]:
                record[arm] = loaded[arm][index].tolist()
            pooled.append(record)
    summary_rows = []
    for arm in ["E0", *ARMS]:
        rows = [row for row in fold_rows if row["arm"] == arm]
        summary = {"arm": arm, "folds": 3}
        for key in METRICS:
            mean, sd = mean_sd([row[key] for row in rows])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd
        summary_rows.append(summary)
    comparisons = []
    for index, (candidate, baseline) in enumerate(
        [("E4", "E2"), ("E4", "E3"), ("E4", "E6"), ("E4", "E5"), ("E4", "E0")]
    ):
        comparisons.append(
            paired_bootstrap(
                pooled,
                candidate,
                baseline,
                args.bootstrap_replicates,
                args.seed + index,
            )
        )
    write_csv(args.output_dir / "unified_metrics_by_fold.csv", fold_rows)
    write_csv(args.output_dir / "unified_metrics_summary.csv", summary_rows)
    write_csv(args.output_dir / "unified_paired_comparisons.csv", comparisons)
    payload = {
        "held_out_test_accessed": False,
        "mmse_included": False,
        "folds": [0, 1, 2],
        "summary": summary_rows,
        "paired_comparisons": comparisons,
    }
    (args.output_dir / "unified_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "UNIFIED_CLINICAL_COMPARISON_REPORT.md").write_text(
        make_report(summary_rows, comparisons), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "held_out_test_accessed": False}))


if __name__ == "__main__":
    main()
