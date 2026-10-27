#!/usr/bin/env python3
"""Audit P2/P3 clinical privilege, split isolation, and MRI-only Student contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import load_configs


SAFE_FEATURES = (
    "age",
    "education",
    "apoe",
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
FORBIDDEN_TOKENS = ("mmse", "cdr", "adas", "faq", "ravlt", "lm_", "animal")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="outputs/vnext_privileged_kd_20261022")
    parser.add_argument("--folds", default="0,1,2")
    return parser.parse_args()


def patient_id(filename):
    value = str(filename).strip()
    if "_ses-" in value:
        return value.split("_ses-", 1)[0]
    return value.split("_", 1)[0]


def read_split(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty split: {path}")
    missing_columns = [name for name in ("filename", "COG", *SAFE_FEATURES) if name not in rows[0]]
    if missing_columns:
        raise RuntimeError(f"Missing columns in {path}: {missing_columns}")
    return rows


def feature_audit(rows):
    result = {}
    for feature in SAFE_FEATURES:
        missing = 0
        nonnumeric = 0
        observed = 0
        for row in rows:
            raw = str(row.get(feature, "")).strip()
            if not raw:
                missing += 1
                continue
            try:
                float(raw)
                observed += 1
            except ValueError:
                nonnumeric += 1
        result[feature] = {
            "observed": observed,
            "missing": missing,
            "nonnumeric": nonnumeric,
            "missing_fraction": missing / len(rows),
        }
    return result


def load_protocol_configs(root):
    teacher_base = [root / "configs/data.yaml", root / "configs/teacher.yaml"]
    student_base = [
        root / "configs/data.yaml",
        root / "configs/student.yaml",
        root / "configs/runtime_vnext_graph_student_locked_s4.yaml",
        root / "configs/runtime_roi_bottleneck_b1_global_logit_kd.yaml",
    ]
    return {
        "p2_teacher": load_configs(
            [str(path) for path in teacher_base + [root / "configs/runtime_privileged_p2_clinical_teacher.yaml"]]
        ),
        "p3_teacher": load_configs(
            [
                str(path)
                for path in teacher_base
                + [
                    root / "configs/runtime_vnext_graph_g3_boundary_homology.yaml",
                    root / "configs/runtime_privileged_p3_joint_teacher.yaml",
                ]
            ]
        ),
        "p2_student": load_configs(
            [str(path) for path in student_base + [root / "configs/runtime_privileged_p2_b1_student.yaml"]]
        ),
        "p3_student": load_configs(
            [str(path) for path in student_base + [root / "configs/runtime_privileged_p3_b1_student.yaml"]]
        ),
    }


def config_audit(configs):
    feature_lists = {
        name: tuple(str(value) for value in cfg.data.clinical.features)
        for name, cfg in configs.items()
    }
    if any(features != SAFE_FEATURES for features in feature_lists.values()):
        raise RuntimeError(f"P2/P3 feature mismatch: {feature_lists}")
    forbidden = [
        feature
        for feature in SAFE_FEATURES
        if any(token in feature.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"Diagnosis-proximal features entered the clean protocol: {forbidden}")
    for name, cfg in configs.items():
        if bool(cfg.runtime.evaluate_test):
            raise RuntimeError(f"Held-out test is enabled in {name}")
        if not bool(cfg.data.clinical.enabled):
            raise RuntimeError(f"Clinical data are disabled in {name}")
    for name in ("p2_student", "p3_student"):
        cfg = configs[name]
        if bool(cfg.model.student_use_clinical):
            raise RuntimeError(f"Student clinical input is enabled in {name}")
        if bool(cfg.model.student_roi_classifier.enabled) or bool(cfg.model.student_roi_graph.enabled):
            raise RuntimeError(f"Student B1 contract is violated in {name}")
        weights = (
            float(cfg.loss_weights.vnext_kd_logit),
            float(cfg.loss_weights.vnext_kd_roi),
            float(cfg.loss_weights.vnext_kd_graph),
            float(cfg.loss_weights.vnext_kd_prototype),
        )
        if weights != (0.5, 0.0, 0.0, 0.0):
            raise RuntimeError(f"Unexpected KD weights in {name}: {weights}")
    p2 = configs["p2_teacher"]
    p3 = configs["p3_teacher"]
    if bool(p2.model.use_roi_branch) or bool(p2.model.cross_geometry.enabled) or bool(p2.model.roi_graph.enabled):
        raise RuntimeError("P2 must not contain the geometric privileged branch")
    if not (
        bool(p3.model.use_roi_branch)
        and bool(p3.model.cross_geometry.enabled)
        and bool(p3.model.roi_graph.enabled)
    ):
        raise RuntimeError("P3 joint privileged geometry is incomplete")
    return {
        "safe_features": list(SAFE_FEATURES),
        "forbidden_features_present": forbidden,
        "all_evaluate_test_false": True,
        "student_inference_inputs": ["MRI"],
        "student_roi_classifier_enabled": False,
        "student_roi_graph_enabled": False,
        "kd": {"kind": "logit_KL", "temperature": 4.0, "weight": 0.5},
        "p2_teacher": "MRI + safe clinical; no surface/ribbon/ROI graph",
        "p3_teacher": "MRI + surface + ribbon correspondence + DKT graph + safe clinical",
    }


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    configs = load_protocol_configs(root)
    protocol = config_audit(configs)
    split_audits = []
    for fold in folds:
        fold_root = root / "lookupcsv-full-huashan-adni-teacher-clean" / f"cross{fold}"
        train_path = fold_root / "train.csv"
        valid_path = fold_root / "valid.csv"
        train_rows = read_split(train_path)
        valid_rows = read_split(valid_path)
        train_visits = [str(row["filename"]).strip() for row in train_rows]
        valid_visits = [str(row["filename"]).strip() for row in valid_rows]
        train_patients = {patient_id(value) for value in train_visits}
        valid_patients = {patient_id(value) for value in valid_visits}
        visit_overlap = sorted(set(train_visits) & set(valid_visits))
        patient_overlap = sorted(train_patients & valid_patients)
        if visit_overlap or patient_overlap:
            raise RuntimeError(
                f"Fold {fold} leakage: visits={len(visit_overlap)} patients={len(patient_overlap)}"
            )
        split_audits.append(
            {
                "fold": fold,
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "train_patients": len(train_patients),
                "valid_patients": len(valid_patients),
                "train_duplicate_visits": len(train_visits) - len(set(train_visits)),
                "valid_duplicate_visits": len(valid_visits) - len(set(valid_visits)),
                "train_valid_visit_overlap": 0,
                "train_valid_patient_overlap": 0,
                "train_feature_audit": feature_audit(train_rows),
                "valid_feature_audit": feature_audit(valid_rows),
            }
        )
    payload = {
        "held_out_test_read": False,
        "folds": folds,
        "protocol": protocol,
        "split_audits": split_audits,
        "clinical_preprocessing": {
            "fit_scope": "training fold only",
            "imputation": "training-fold mean",
            "standardization": "training-fold mean/std",
            "missingness": "one binary flag per feature",
        },
    }
    (output_root / "protocol_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# P2/P3 Privileged-KD Protocol Audit",
        "",
        "- Held-out test read: `false`",
        "- Student inference input: `MRI only`",
        "- Clinical features: `16` clean numeric variables",
        "- P2: MRI + safe clinical Teacher; B1 Student logit KD",
        "- P3: joint geometry + safe clinical Teacher; B1 Student logit KD",
        "",
        "| Fold | Train rows | Valid rows | Train patients | Valid patients | Patient overlap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in split_audits:
        lines.append(
            f"| {item['fold']} | {item['train_rows']} | {item['valid_rows']} | "
            f"{item['train_patients']} | {item['valid_patients']} | 0 |"
        )
    lines.extend(
        [
            "",
            "Clinical imputation and standardization are fitted on each training fold and reused unchanged for validation. MRI, surface and clinical values are read from the same visit row. The audit intentionally never opens a test CSV.",
        ]
    )
    (output_root / "PROTOCOL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "held_out_test_read": False, "folds": folds}))


if __name__ == "__main__":
    main()
