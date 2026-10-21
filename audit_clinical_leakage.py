#!/usr/bin/env python3
"""Development-only clinical-label leakage audit for the COG3 task.

The audit deliberately reads only ``train.csv`` and ``valid.csv`` from each
cross-validation fold.  It never imports the imaging dataset, opens a test
split, or loads a neural-network checkpoint.  Every preprocessing operation is
fit on the current fold's training rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import compute_classification_metrics


CURRENT_CLINICAL_FEATURES = (
    "age",
    "education",
    "apoe",
    "animal",
    "gds",
    "lm_imm",
    "lm_del",
    "faq_BILLS",
    "faq_TAXES",
    "faq_SHOPPING",
    "faq_GAMES",
    "faq_STOVE",
    "faq_MEALPREP",
    "faq_EVENTS",
    "faq_PAYATTN",
    "faq_REMDATES",
    "faq_TRAVEL",
    "npiq_DEL",
    "npiq_HALL",
    "npiq_AGIT",
    "npiq_DEPD",
    "npiq_ANX",
    "npiq_ELAT",
    "npiq_APA",
    "npiq_DISN",
    "npiq_IRR",
    "npiq_MOT",
    "npiq_NITE",
    "npiq_APP",
)

SAFE_BACKGROUND_FEATURES = ("age", "gender", "education", "apoe")
HIGH_RISK_COGNITIVE_FEATURES = (
    "animal",
    "lm_imm",
    "lm_del",
    "faq_BILLS",
    "faq_TAXES",
    "faq_SHOPPING",
    "faq_GAMES",
    "faq_STOVE",
    "faq_MEALPREP",
    "faq_EVENTS",
    "faq_PAYATTN",
    "faq_REMDATES",
    "faq_TRAVEL",
)
NEUROPSYCHIATRIC_FEATURES = (
    "gds",
    "npiq_DEL",
    "npiq_HALL",
    "npiq_AGIT",
    "npiq_DEPD",
    "npiq_ANX",
    "npiq_ELAT",
    "npiq_APA",
    "npiq_DISN",
    "npiq_IRR",
    "npiq_MOT",
    "npiq_NITE",
    "npiq_APP",
)
SAFE_NEUROPSYCHIATRIC_FEATURES = SAFE_BACKGROUND_FEATURES + NEUROPSYCHIATRIC_FEATURES

FEATURE_SETS = {
    "safe_background": SAFE_BACKGROUND_FEATURES,
    "safe_neuropsychiatric": SAFE_NEUROPSYCHIATRIC_FEATURES,
    "high_risk_cognition": HIGH_RISK_COGNITIVE_FEATURES,
    "neuropsychiatric": NEUROPSYCHIATRIC_FEATURES,
    "current_clinical_29": CURRENT_CLINICAL_FEATURES,
}

LABEL_COLUMNS = ("NC", "MCI", "AD", "DE", "ADD", "MCIC")


@dataclass(frozen=True)
class Row:
    fold: int
    split: str
    filename: str
    subject: str
    label: int
    raw: Mapping[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        default="lookupcsv-full-huashan-adni-teacher-clean",
        help="Root containing cross{fold}/train.csv and valid.csv.",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--output-dir",
        default="outputs/vnext_clinical_leakage_audit_20261018",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def subject_from_filename(filename: str) -> str:
    return str(filename).split("_", 1)[0]


def _parse_label(raw: str) -> int | None:
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    return value if value in (0, 1, 2) else None


def read_rows(path: Path, fold: int, split: str) -> List[Row]:
    if path.name == "test.csv" or split == "test":
        raise RuntimeError("The clinical leakage audit must never read a test split")
    rows: List[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            filename = str(raw.get("filename", "")).strip()
            if not filename.startswith("sub-ADNI"):
                continue
            label = _parse_label(raw.get("COG", ""))
            if not filename or label is None:
                continue
            rows.append(
                Row(
                    fold=fold,
                    split=split,
                    filename=filename,
                    subject=subject_from_filename(filename),
                    label=label,
                    raw=dict(raw),
                )
            )
    if not rows:
        raise RuntimeError(f"No valid ADNI COG3 rows in {path}")
    return rows


def numeric_value(raw: object) -> float:
    text = str(raw if raw is not None else "").strip().lower()
    if text in {"male", "m"}:
        return 1.0
    if text in {"female", "f"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def feature_matrix(rows: Sequence[Row], features: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[numeric_value(row.raw.get(name, "")) for name in features] for row in rows],
        dtype=np.float64,
    )


def labels(rows: Sequence[Row]) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=np.int64)


def build_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def predict_arm(
    train_rows: Sequence[Row],
    valid_rows: Sequence[Row],
    features: Sequence[str],
    seed: int,
    shuffle_labels: bool = False,
) -> np.ndarray:
    x_train = feature_matrix(train_rows, features)
    x_valid = feature_matrix(valid_rows, features)
    y_train = labels(train_rows)
    if shuffle_labels:
        y_train = np.random.default_rng(seed).permutation(y_train)
    model = build_classifier(seed)
    model.fit(x_train, y_train)
    raw_prob = model.predict_proba(x_valid)
    prob = np.zeros((len(valid_rows), 3), dtype=np.float64)
    for source_idx, class_idx in enumerate(model.named_steps["classifier"].classes_):
        prob[:, int(class_idx)] = raw_prob[:, source_idx]
    return prob


def metric_subset(metrics: Mapping[str, object]) -> Dict[str, float | list]:
    keys = ("acc", "balanced_acc", "auc", "mcc", "f1_macro", "confusion_matrix")
    return {key: metrics[key] for key in keys if key in metrics}


def summarize_fold_metrics(fold_metrics: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    keys = ("acc", "balanced_acc", "auc", "mcc", "f1_macro")
    result: Dict[str, object] = {}
    for key in keys:
        values = np.asarray([float(item[key]) for item in fold_metrics], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "fold_values": values.tolist(),
        }
    return result


def direct_label(row: Row) -> str:
    positive = []
    for name in LABEL_COLUMNS:
        value = numeric_value(row.raw.get(name, ""))
        if np.isfinite(value) and value > 0.5:
            positive.append(name)
    return "+".join(positive) if positive else "none"


def label_provenance(rows_by_fold: Mapping[int, Mapping[str, Sequence[Row]]]) -> List[Dict[str, object]]:
    records = []
    expected = {0: "NC", 1: "MCI", 2: "AD"}
    for fold, splits in sorted(rows_by_fold.items()):
        for split, rows in splits.items():
            counts = Counter((row.label, direct_label(row)) for row in rows)
            for (label, source), count in sorted(counts.items()):
                records.append(
                    {
                        "fold": fold,
                        "split": split,
                        "cog_label": label,
                        "expected_group": expected[label],
                        "direct_indicator_pattern": source,
                        "count": count,
                        "matches_expected_indicator": source == expected[label],
                    }
                )
    return records


def bootstrap_metric_difference(
    records: Sequence[Mapping[str, object]],
    candidate: str,
    baseline: str,
    metric: str,
    replicates: int,
    seed: int,
) -> Dict[str, object]:
    by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_subject[str(record["subject"])].append(record)
    subjects = np.asarray(sorted(by_subject), dtype=object)
    rng = np.random.default_rng(seed)

    def evaluate(selected_subjects: Iterable[str]) -> float:
        selected = [item for subject in selected_subjects for item in by_subject[str(subject)]]
        y = np.asarray([int(item["label"]) for item in selected], dtype=np.int64)
        candidate_prob = np.asarray([item[f"prob_{candidate}"] for item in selected], dtype=np.float64)
        baseline_prob = np.asarray([item[f"prob_{baseline}"] for item in selected], dtype=np.float64)
        return float(compute_classification_metrics(y, candidate_prob)[metric]) - float(
            compute_classification_metrics(y, baseline_prob)[metric]
        )

    point = evaluate(subjects)
    samples = np.empty(int(replicates), dtype=np.float64)
    for idx in range(int(replicates)):
        samples[idx] = evaluate(rng.choice(subjects, size=len(subjects), replace=True))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "difference": point,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
        "subjects": int(len(subjects)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{float(value):.5f}"


def report_markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# CortexBridge vNext — Clinical Label-Leakage Audit",
        "",
        "## Scope",
        "",
        "This audit is development-only. It reads each fold's training and validation CSVs only; no test CSV, MRI, surface, checkpoint, or held-out outcome is loaded.",
        "",
        "## Label provenance",
        "",
        f"- Rows audited: `{result['row_count']}` validation rows across `{result['fold_count']}` folds.",
        f"- Train/validation subject overlaps: `{result['train_valid_subject_overlap_total']}`.",
        f"- Direct NC/MCI/AD indicator agreement with COG: `{fmt(result['direct_indicator_agreement'])}`.",
        "- This indicator agreement documents the CSV encoding; it does not independently verify the clinical adjudication protocol that produced the source columns.",
        "",
        "## Validation-only clinical baselines",
        "",
        "| Arm | Features | ACC | Balanced ACC | AUC | MCC | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, summary in result["arm_summaries"].items():
        features = len(result["feature_sets"][arm]) if arm in result["feature_sets"] else len(CURRENT_CLINICAL_FEATURES)
        lines.append(
            f"| `{arm}` | {features} | {fmt(summary['acc']['mean'])} ± {fmt(summary['acc']['sample_sd'])} | "
            f"{fmt(summary['balanced_acc']['mean'])} ± {fmt(summary['balanced_acc']['sample_sd'])} | "
            f"{fmt(summary['auc']['mean'])} ± {fmt(summary['auc']['sample_sd'])} | "
            f"{fmt(summary['mcc']['mean'])} ± {fmt(summary['mcc']['sample_sd'])} | "
            f"{fmt(summary['f1_macro']['mean'])} ± {fmt(summary['f1_macro']['sample_sd'])} |"
        )
    lines.extend(["", "## Paired OOF bootstrap contrasts", ""])
    lines.extend(
        [
            "| Candidate − baseline | Metric | Difference | 95% CI | Subjects |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in result["bootstrap_contrasts"]:
        lines.append(
            f"| `{item['candidate']} − {item['baseline']}` | {item['metric']} | "
            f"{fmt(item['difference'])} | [{fmt(item['ci_low'])}, {fmt(item['ci_high'])}] | {item['subjects']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen interpretation boundary",
            "",
            "- High clinical-only performance is evidence of label-proximal predictive information, not proof of improper leakage by itself.",
            "- FAQ, episodic-memory, and fluency measures are treated as high-risk label-proximal variables until the source diagnostic adjudication protocol is documented.",
            "- The disease-aware graph experiments must use the same frozen clinical setting across arms. A clinical-off or safe-background main analysis is required if current clinical variables dominate the endpoint.",
            "- Held-out test accessed: `false`.",
            "",
            "## Artifacts",
            "",
            "- `audit_metrics.json`",
            "- `fold_metrics.csv`",
            "- `oof_predictions.csv`",
            "- `label_provenance.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    rows_by_fold: Dict[int, Dict[str, List[Row]]] = {}
    for fold in folds:
        fold_root = split_root / f"cross{fold}"
        rows_by_fold[fold] = {
            "train": read_rows(fold_root / "train.csv", fold, "train"),
            "valid": read_rows(fold_root / "valid.csv", fold, "valid"),
        }

    overlaps = {}
    for fold, splits in rows_by_fold.items():
        train_subjects = {row.subject for row in splits["train"]}
        valid_subjects = {row.subject for row in splits["valid"]}
        overlaps[fold] = sorted(train_subjects & valid_subjects)
        if overlaps[fold]:
            raise RuntimeError(f"Fold {fold} has train/validation subject overlap: {overlaps[fold][:5]}")

    arms = dict(FEATURE_SETS)
    arms["current_clinical_29_shuffled_label"] = CURRENT_CLINICAL_FEATURES
    fold_metric_rows: List[Dict[str, object]] = []
    oof_records: List[Dict[str, object]] = []
    arm_fold_metrics: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for fold in folds:
        train_rows = rows_by_fold[fold]["train"]
        valid_rows = rows_by_fold[fold]["valid"]
        fold_probs = {}
        for arm, features in arms.items():
            fold_probs[arm] = predict_arm(
                train_rows,
                valid_rows,
                features,
                seed=int(args.seed) + fold,
                shuffle_labels=arm.endswith("shuffled_label"),
            )
            metrics = metric_subset(compute_classification_metrics(labels(valid_rows), fold_probs[arm]))
            arm_fold_metrics[arm].append(metrics)
            fold_metric_rows.append({"fold": fold, "arm": arm, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        for row_idx, row in enumerate(valid_rows):
            record: Dict[str, object] = {
                "fold": fold,
                "filename": row.filename,
                "subject": row.subject,
                "label": row.label,
            }
            for arm, probs in fold_probs.items():
                record[f"prob_{arm}"] = probs[row_idx].tolist()
                record[f"pred_{arm}"] = int(np.argmax(probs[row_idx]))
            oof_records.append(record)

    provenance = label_provenance(rows_by_fold)
    direct_total = sum(int(item["count"]) for item in provenance)
    direct_match = sum(int(item["count"]) for item in provenance if item["matches_expected_indicator"])
    summaries = {arm: summarize_fold_metrics(values) for arm, values in arm_fold_metrics.items()}
    contrasts = []
    for candidate, baseline in (
        ("current_clinical_29", "safe_background"),
        ("high_risk_cognition", "safe_background"),
        ("current_clinical_29", "current_clinical_29_shuffled_label"),
    ):
        for metric in ("mcc", "auc", "balanced_acc"):
            contrasts.append(
                bootstrap_metric_difference(
                    oof_records,
                    candidate=candidate,
                    baseline=baseline,
                    metric=metric,
                    replicates=args.bootstrap_replicates,
                    seed=int(args.seed) + len(contrasts),
                )
            )

    result = {
        "run_id": output_dir.name,
        "split_root": str(split_root),
        "folds": folds,
        "fold_count": len(folds),
        "row_count": len(oof_records),
        "unique_oof_subjects": len({str(record["subject"]) for record in oof_records}),
        "train_valid_subject_overlap": overlaps,
        "train_valid_subject_overlap_total": sum(len(value) for value in overlaps.values()),
        "feature_sets": {name: list(values) for name, values in arms.items()},
        "arm_summaries": summaries,
        "bootstrap_contrasts": contrasts,
        "direct_indicator_agreement": direct_match / max(1, direct_total),
        "held_out_test_accessed": False,
        "imaging_loaded": False,
        "checkpoint_loaded": False,
        "seed": int(args.seed),
        "bootstrap_replicates": int(args.bootstrap_replicates),
    }

    with (output_dir / "audit_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    serial_oof = []
    for record in oof_records:
        flat = dict(record)
        for key, value in list(flat.items()):
            if key.startswith("prob_"):
                flat[key] = json.dumps(value)
        serial_oof.append(flat)
    write_csv(output_dir / "fold_metrics.csv", fold_metric_rows)
    write_csv(output_dir / "oof_predictions.csv", serial_oof)
    write_csv(output_dir / "label_provenance.csv", provenance)
    (output_dir / "CLINICAL_LEAKAGE_AUDIT.md").write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "held_out_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
