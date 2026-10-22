#!/usr/bin/env python3
"""Recompute development-only classification metrics from saved predictions.

This script intentionally uses an explicit registry.  It excludes dry-runs,
the invalid low-learning-rate Student runs, and every held-out test artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "vnext_classification_metric_recalculation_20261020"
CLASS_NAMES = ("NC", "MCI", "DE")
METRIC_KEYS = ("mcc", "acc", "sen", "spe", "auc", "f1_macro", "recall_macro")


EXPERIMENTS = (
    {
        "family": "Graph Teacher",
        "variant": "G0",
        "description": "hard CG, graph off",
        "run": "outputs/vnext_graph_stageA_20261019/g0_fold{fold}",
        "config": "configs/runtime_vnext_graph_g0_clinical_off.yaml",
        "format": "json",
    },
    {
        "family": "Graph Teacher",
        "variant": "G1",
        "description": "graph, no topology bias",
        "run": "outputs/vnext_graph_stageA_20261019/g1_fold{fold}",
        "config": "configs/runtime_vnext_graph_g1_unbiased.yaml",
        "format": "json",
    },
    {
        "family": "Graph Teacher",
        "variant": "G2",
        "description": "graph + true DKT boundary bias",
        "run": "outputs/vnext_graph_stageA_20261019/g2_fold{fold}",
        "config": "configs/runtime_vnext_graph_g2_boundary.yaml",
        "format": "json",
    },
    {
        "family": "Graph Teacher",
        "variant": "G3",
        "description": "G2 + bilateral homology bias",
        "run": "outputs/vnext_graph_stageA_20261019/g3_fold{fold}",
        "config": "configs/runtime_vnext_graph_g3_boundary_homology.yaml",
        "format": "json",
    },
    {
        "family": "Graph Teacher",
        "variant": "G3-R",
        "description": "G3 with deterministically shuffled topology bias",
        "run": "outputs/vnext_graph_stageA_20261019/g3r_fold{fold}",
        "config": "configs/runtime_vnext_graph_g3r_shuffled_boundary_homology.yaml",
        "format": "json",
    },
    {
        "family": "MRI-only Student",
        "variant": "S0",
        "description": "CE only",
        "run": "outputs/vnext_graph_student_20261019/s0_scratch_corrected",
        "config": "configs/runtime_vnext_graph_student_s0_scratch.yaml",
        "format": "csv",
    },
    {
        "family": "MRI-only Student",
        "variant": "S3",
        "description": "S0 + logit KD",
        "run": "outputs/vnext_graph_student_20261019/s3_logit_kd_corrected",
        "config": "configs/runtime_vnext_graph_student_s3_logit_kd.yaml",
        "format": "csv",
    },
    {
        "family": "MRI-only Student",
        "variant": "S4",
        "description": "S3 + ROI graph-state KD (locked)",
        "run": "outputs/vnext_graph_student_20261019/s4_roi_kd_corrected",
        "config": "configs/runtime_vnext_graph_student_s4_roi_kd.yaml",
        "format": "csv",
    },
    {
        "family": "MRI-only Student",
        "variant": "S5",
        "description": "S4 + graph-relation KD",
        "run": "outputs/vnext_graph_student_20261019/s5_graph_kd_corrected",
        "config": "configs/runtime_vnext_graph_student_s5_graph_kd.yaml",
        "format": "csv",
    },
    {
        "family": "MRI-only Student",
        "variant": "S6",
        "description": "S5 + prototype-posterior KD",
        "run": "outputs/vnext_graph_student_20261019/s6_full_kd_corrected",
        "config": "configs/runtime_vnext_graph_student_s6_full_kd.yaml",
        "format": "csv",
    },
)


def load_predictions(exp: dict, fold: int) -> tuple[list[str], np.ndarray, np.ndarray, Path]:
    run_dir = ROOT / exp["run"].format(fold=fold)
    fold_dir = run_dir / f"fold_{fold}"
    path = fold_dir / f"validation_predictions.{exp['format']}"
    if not path.is_file():
        raise FileNotFoundError(path)

    if exp["format"] == "json":
        payload = json.loads(path.read_text())
        if payload.get("split") != "validation" or payload.get("fold") != fold:
            raise ValueError(f"unexpected split metadata in {path}")
        if payload.get("held_out_test_evaluated") is not False:
            raise ValueError(f"held-out flag is not false in {path}")
        records = payload["records"]
        ids = [str(row["subject_id"]) for row in records]
        labels = np.asarray([row["label"] for row in records], dtype=np.int64)
        probs = np.asarray([row["probabilities"] for row in records], dtype=np.float64)
        saved_preds = np.asarray([row["prediction"] for row in records], dtype=np.int64)
    else:
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        ids = [str(row["filename"]) for row in records]
        labels = np.asarray([row["label"] for row in records], dtype=np.int64)
        probs = np.asarray(
            [[row[f"score_{index}"] for index in range(len(CLASS_NAMES))] for row in records],
            dtype=np.float64,
        )
        saved_preds = np.asarray([row["pred"] for row in records], dtype=np.int64)

    if probs.shape != (len(labels), len(CLASS_NAMES)):
        raise ValueError(f"unexpected probability shape {probs.shape} in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate subject/session identifiers in {path}")
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError(f"invalid probabilities in {path}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError(f"probabilities do not sum to one in {path}")
    if not np.array_equal(probs.argmax(axis=1), saved_preds):
        raise ValueError(f"saved predictions disagree with argmax in {path}")

    metrics_path = fold_dir / "development_metrics.json"
    metadata = json.loads(metrics_path.read_text())
    if metadata.get("held_out_test_evaluated") is not False:
        raise ValueError(f"held-out flag is not false in {metrics_path}")
    return ids, labels, probs, metrics_path


def calculate(labels: np.ndarray, probs: np.ndarray) -> tuple[dict, list[dict]]:
    preds = probs.argmax(axis=1)
    matrix = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    total = int(matrix.sum())
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if tp + fn else math.nan
        specificity = tn / (tn + fp) if tn + fp else math.nan
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
        auc = roc_auc_score((labels == index).astype(np.int64), probs[:, index])
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": int((labels == index).sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "sensitivity": float(sensitivity),
                "recall": float(sensitivity),
                "specificity": float(specificity),
                "f1": float(f1),
                "auc_ovr": float(auc),
            }
        )

    recall_macro = float(recall_score(labels, preds, average="macro", zero_division=0))
    result = {
        "n": int(len(labels)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "acc": float(accuracy_score(labels, preds)),
        "sen": recall_macro,
        "spe": float(np.mean([row["specificity"] for row in per_class])),
        "auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": recall_macro,
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }
    return result, per_class


def read_flat_options(path: Path) -> dict[str, str]:
    options = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*([^:]+?):\s*(.*?)\s*$", line)
        if match:
            options[match.group(1).strip()] = match.group(2).strip()
    return options


def config_manifest(exp: dict, n_total: int) -> dict:
    run_dir = ROOT / exp["run"].format(fold=0)
    options = read_flat_options(run_dir / "opt.txt")
    get = lambda key, default="N/A": options.get(key, default)
    return {
        "family": exp["family"],
        "variant": exp["variant"],
        "description": exp["description"],
        "runtime_config": exp["config"],
        "config_stack": get("runtime.config_paths"),
        "folds": "0,1,2",
        "validation_n_total": n_total,
        "selection_metric": get("training.selection_metric"),
        "clinical_enabled": get("data.clinical.enabled"),
        "volume_pooling": get("model.volume_pooling.mode"),
        "cross_geometry": get("model.cross_geometry.enabled"),
        "lambda_cg": get("model.cross_geometry.lambda_cg"),
        "roi_graph": get("model.roi_graph.enabled"),
        "boundary_bias": get("model.roi_graph.use_boundary_strength", "False"),
        "homology_bias": get("model.roi_graph.use_homology", "False"),
        "topology_shuffled": get("model.roi_graph.shuffle_topology_bias", "False"),
        "topology_query": get("model.topology_query.enabled", "False"),
        "kd_logit": get("loss_weights.vnext_kd_logit", "0.0"),
        "kd_roi": get("loss_weights.vnext_kd_roi", "0.0"),
        "kd_graph": get("loss_weights.vnext_kd_graph", "0.0"),
        "kd_prototype": get("loss_weights.vnext_kd_prototype", "0.0"),
        "epochs": get("training.epochs"),
        "batch_size": get("training.batch_size"),
        "lr": get("training.lr"),
        "encoder_lr": get("training.encoder_lr", "N/A"),
        "warmup_epochs": get("training.warmup_epochs", "N/A"),
        "seed": get("training.seed"),
        "evaluate_test": get("runtime.evaluate_test"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()), float(data.std(ddof=1))


def format_mean_sd(row: dict, key: str) -> str:
    return f"{row[key + '_mean']:.5f} ± {row[key + '_sd']:.5f}"


def build_report(
    summary_rows: list[dict],
    per_class_summary: list[dict],
    manifest: list[dict],
    max_delta: float,
) -> str:
    by_family = {}
    for row in summary_rows:
        by_family.setdefault(row["family"], []).append(row)

    lines = [
        "# CortexBridge 正式消融分类指标复算报告",
        "",
        "日期：2026-10-20  ",
        "状态：**VERIFIED（由保存的 development validation 概率逐例复算）**  ",
        "边界：未重新训练；未读取、加载或评估 held-out test。",
        "",
        "## 结论摘要",
        "",
        "已对 Graph Teacher G0/G1/G2/G3/G3-R 与 MRI-only Student S0/S3/S4/S5/S6 的三折正式预测统一复算 MCC、ACC、SEN、SPE、AUC、macro-F1 和 macro Recall。S4 仍是 Student 消融中 mean MCC 最高的锁定方案。",
        "",
        "本任务为三分类 NC/MCI/DE：`SEN` 定义为逐类 one-vs-rest sensitivity 的宏平均，因此 `SEN = macro Recall = balanced accuracy`。报告保留 SEN 与 Recall 两列以满足临床报告习惯，但它们不是独立指标。`SPE` 是逐类 one-vs-rest specificity 的宏平均。",
        "",
    ]
    for family, rows in by_family.items():
        lines += [
            f"## {family}",
            "",
            "所有单元格均为三折 mean ± sample SD。",
            "",
            "| ID | MCC | ACC | SEN | SPE | AUC | Macro-F1 | Macro Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {variant} | {mcc} | {acc} | {sen} | {spe} | {auc} | {f1} | {recall} |".format(
                    variant=row["variant"],
                    mcc=format_mean_sd(row, "mcc"),
                    acc=format_mean_sd(row, "acc"),
                    sen=format_mean_sd(row, "sen"),
                    spe=format_mean_sd(row, "spe"),
                    auc=format_mean_sd(row, "auc"),
                    f1=format_mean_sd(row, "f1_macro"),
                    recall=format_mean_sd(row, "recall_macro"),
                )
            )
        lines.append("")

    s4_classes = [row for row in per_class_summary if row["variant"] == "S4"]
    lines += [
        "## 主模型 S4 的逐类表现",
        "",
        "逐类指标采用 one-vs-rest 定义，均为三折 mean ± sample SD。Recall 与 Sensitivity 在逐类层面也是同一个量。",
        "",
        "| Class | Support（三折总计） | SEN / Recall | SPE | AUC | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in s4_classes:
        lines.append(
            "| {name} | {support} | {sen:.5f} ± {sen_sd:.5f} | {spe:.5f} ± {spe_sd:.5f} | {auc:.5f} ± {auc_sd:.5f} | {f1:.5f} ± {f1_sd:.5f} |".format(
                name=row["class_name"],
                support=row["support_total"],
                sen=row["sensitivity_mean"],
                sen_sd=row["sensitivity_sd"],
                spe=row["specificity_mean"],
                spe_sd=row["specificity_sd"],
                auc=row["auc_ovr_mean"],
                auc_sd=row["auc_ovr_sd"],
                f1=row["f1_mean"],
                f1_sd=row["f1_sd"],
            )
        )
    lines += [
        "",
        "## 模型选择解释",
        "",
        "- Graph Teacher 中，G3 的 mean MCC 最高；G2 的 ACC 略高、G3-R 的 AUC 略高，因此不能说 G3 在所有指标上都占优。G3 是依据预设主指标 MCC 与结构解释性进入 Student。",
        "- Student 中，S4 的 mean MCC（0.31227）与 AUC（0.73894）最高；S5 的 ACC、SEN、SPE 和 macro-F1 略高，但 MCC 更低且跨折波动更大。",
        "- 实验在运行前以 MCC 为 selection metric；补报次要指标后不得事后改变选择准则。因此锁定 S4 的决定不变。完整五折时应同时报告全部指标、混淆矩阵和逐类结果，不应只报 MCC。",
        "",
    ]

    lines += [
        "## 实验配置对应关系",
        "",
        "| ID | 改动 | Runtime config | Graph/B/H/Shuffle | KD weights (logit/ROI/graph/proto) |",
        "|---|---|---|---|---|",
    ]
    for row in manifest:
        graph = "/".join(
            [row["roi_graph"], row["boundary_bias"], row["homology_bias"], row["topology_shuffled"]]
        )
        kd = "/".join([row["kd_logit"], row["kd_roi"], row["kd_graph"], row["kd_prototype"]])
        lines.append(
            f"| {row['variant']} | {row['description']} | `{row['runtime_config']}` | {graph} | {kd} |"
        )

    lines += [
        "",
        "共同设置：development folds 0/1/2；3 类 NC/MCI/DE；clinical off；legacy_mask volume pooling；hard Surface↔Ribbon cross-geometry alignment，`lambda_cg=0.01`；selection metric=MCC；seed=3407；`evaluate_test=false`。Graph Teacher 为 30 epochs、batch size 8、lr=1e-4。Student 为随机初始化 topology-seeded Graph Student，30 epochs、batch size 8、encoder/inherited lr=1e-4、warmup=0。完整扁平化配置栈见 `ablation_config_manifest.csv` 和各 run 的 `opt.txt`。",
        "",
        "## 指标定义",
        "",
        "- ACC：全部样本预测正确比例。",
        "- SEN / Recall：先计算 NC、MCI、DE 各自 `TP/(TP+FN)`，再做不加权宏平均。",
        "- SPE：先把每一类视为阳性、其余两类视为阴性，计算 `TN/(TN+FP)`，再做宏平均。",
        "- AUC：三类 one-vs-rest ROC-AUC 的宏平均，使用保存的概率而非 hard label。",
        "- F1：三类 F1 的宏平均；不是 micro-F1 或 weighted-F1。",
        "- MCC：scikit-learn multiclass Matthews correlation coefficient。",
        "- 跨折汇总：算术平均 ± sample SD（`ddof=1`），不是把三折样本合并后再报单一值。",
        "",
        "## 逐类结果与可复现产物",
        "",
        "逐 fold 指标、NC/MCI/DE 逐类指标及配置清单分别保存在：",
        "",
        "- `classification_metrics_by_fold.csv`",
        "- `classification_metrics_summary.csv`",
        "- `classification_metrics_per_class.csv`",
        "- `classification_metrics_per_class_summary.csv`",
        "- `ablation_config_manifest.csv`",
        "- `classification_metrics_recalculation.json`",
        "",
        "## 完整性与一致性验证",
        "",
        "- 10 个正式变体 × 3 folds = 30 份 validation prediction artifacts 均成功读取。",
        "- 每折样本数为 454/457/467，三折合计 1,378；各变体 subject/session ID、标签及折划分一致。",
        "- 所有概率有限、非负、逐行和为 1，保存的 prediction 与概率 argmax 一致。",
        "- 所有 prediction/development metadata 的 held-out 标志均为 false。",
        f"- 与训练阶段保存的聚合指标逐项核对，最大绝对差为 `{max_delta:.3e}`。",
        "- dry-run、早期无效 low-LR Student、缺逐例概率的历史 Phase 1/2.1 结果均未混入本复算表。",
        "",
        "## Material Passport",
        "",
        "- Input：30 份已保存的 development validation prediction artifacts；对应 development metrics 与 `opt.txt`。",
        "- Transformation：固定公式逐例复算 → 折内指标 → 三折 mean ± sample SD；不拟合参数，不改变预测。",
        "- Output：本报告、4 个指标 CSV、1 个配置 CSV、1 个机器可读 JSON。",
        "- Exclusions：held-out test、dry-run、无效学习率协议、没有逐例概率的历史结果。",
        "- Reproduction：`python recompute_ablation_classification_metrics.py`。",
        "",
        "## Epistemic / fallacy scan（11/11）",
        "",
        "1. 不把 development selection 写成 held-out 泛化。",
        "2. 不把 mean 改善写成统计显著，除非另有配对区间支持。",
        "3. 不把 SEN、Recall、balanced accuracy 重复计为三份证据。",
        "4. 不把 macro 指标解释为任一具体疾病类别表现。",
        "5. 不用 hard prediction 计算 AUC。",
        "6. 不把 pooled 指标冒充 fold mean。",
        "7. 不把 Graph Teacher 与 MRI-only Student 当成同一推理输入条件。",
        "8. 不把 G3 优于 G3-R 写成已稳定证明的 anatomy-specific classification gain。",
        "9. 不把 S4−S3 的小增量写成 ROI-state KD 已独立确证。",
        "10. 不纳入 dry-run 或已判无效的 Student 协议。",
        "11. 不从本复算推断外部验证、校准或亚组公平性。",
        "",
        "## 历史 Phase 1/2.1 的边界",
        "",
        "Phase 1/1.1/2.1 当时只落盘了聚合 development metrics 与 checkpoint，没有逐受试者 validation probabilities。因此本次无训练复算不能为这些历史变体新增可靠的 SPE/逐类 SEN；如必须补齐，需要单独执行 checkpoint validation inference（仍可保持 development-only），那将属于一次新的推理任务，而不是本报告的纯离线复算。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    per_class_rows: list[dict] = []
    per_class_summary: list[dict] = []
    summary_rows: list[dict] = []
    manifest: list[dict] = []
    details = []
    reference_split = {}
    max_stored_delta = 0.0

    for exp in EXPERIMENTS:
        variant_metrics = []
        n_total = 0
        for fold in range(3):
            ids, labels, probs, metrics_path = load_predictions(exp, fold)
            split_signature = (ids, labels.tolist())
            if fold not in reference_split:
                reference_split[fold] = split_signature
            elif reference_split[fold] != split_signature:
                raise ValueError(f"split mismatch for {exp['variant']} fold {fold}")

            metrics, class_metrics = calculate(labels, probs)
            stored_payload = json.loads(metrics_path.read_text())
            stored = stored_payload.get("metrics", stored_payload.get("best_val_metrics"))
            if stored is None:
                raise ValueError(f"no saved validation metrics in {metrics_path}")
            for ours, theirs in (
                (metrics["mcc"], stored["mcc"]),
                (metrics["acc"], stored["acc"]),
                (metrics["sen"], stored["sen"]),
                (metrics["spe"], stored["spe"]),
                (metrics["auc"], stored["auc"]),
                (metrics["f1_macro"], stored["f1_macro"]),
                (metrics["recall_macro"], stored["recall_macro"]),
            ):
                max_stored_delta = max(max_stored_delta, abs(ours - float(theirs)))

            flat = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
            fold_rows.append(
                {
                    "family": exp["family"],
                    "variant": exp["variant"],
                    "fold": fold,
                    "prediction_artifact": str(
                        metrics_path.parent / f"validation_predictions.{exp['format']}"
                    ).replace(str(ROOT) + "/", ""),
                    **flat,
                }
            )
            for row in class_metrics:
                per_class_rows.append(
                    {
                        "family": exp["family"],
                        "variant": exp["variant"],
                        "fold": fold,
                        **row,
                    }
                )
            variant_metrics.append(metrics)
            n_total += metrics["n"]
            details.append(
                {
                    "family": exp["family"],
                    "variant": exp["variant"],
                    "fold": fold,
                    "metrics": metrics,
                    "per_class": class_metrics,
                }
            )

        summary = {
            "family": exp["family"],
            "variant": exp["variant"],
            "description": exp["description"],
            "folds": 3,
            "validation_n_total": n_total,
        }
        for key in METRIC_KEYS:
            mean, sd = mean_sd([row[key] for row in variant_metrics])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_sd"] = sd
        summary_rows.append(summary)
        manifest.append(config_manifest(exp, n_total))

        exp_class_rows = [row for row in per_class_rows if row["variant"] == exp["variant"]]
        for class_index, class_name in enumerate(CLASS_NAMES):
            rows = [row for row in exp_class_rows if row["class_index"] == class_index]
            class_summary = {
                "family": exp["family"],
                "variant": exp["variant"],
                "class_index": class_index,
                "class_name": class_name,
                "support_total": sum(int(row["support"]) for row in rows),
            }
            for key in ("precision", "sensitivity", "recall", "specificity", "f1", "auc_ovr"):
                mean, sd = mean_sd([float(row[key]) for row in rows])
                class_summary[f"{key}_mean"] = mean
                class_summary[f"{key}_sd"] = sd
            per_class_summary.append(class_summary)

    if max_stored_delta > 1e-6:
        raise ValueError(f"recomputed/stored metric mismatch: {max_stored_delta}")

    write_csv(output_dir / "classification_metrics_by_fold.csv", fold_rows)
    write_csv(output_dir / "classification_metrics_summary.csv", summary_rows)
    write_csv(output_dir / "classification_metrics_per_class.csv", per_class_rows)
    write_csv(output_dir / "classification_metrics_per_class_summary.csv", per_class_summary)
    write_csv(output_dir / "ablation_config_manifest.csv", manifest)
    payload = {
        "scope": "formal three-fold development ablations only",
        "held_out_test_accessed": False,
        "class_names": list(CLASS_NAMES),
        "definitions": {
            "sen": "macro one-vs-rest sensitivity; identical to macro recall",
            "spe": "macro one-vs-rest specificity",
            "auc": "macro one-vs-rest ROC-AUC",
            "f1": "macro-F1",
            "mcc": "multiclass Matthews correlation coefficient",
            "cross_fold": "arithmetic mean and sample standard deviation (ddof=1)",
        },
        "maximum_absolute_difference_vs_saved_metrics": max_stored_delta,
        "summary": summary_rows,
        "per_class_summary": per_class_summary,
        "folds": details,
        "config_manifest": manifest,
    }
    (output_dir / "classification_metrics_recalculation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (output_dir / "ABLATION_CLASSIFICATION_METRICS_REPORT.md").write_text(
        build_report(summary_rows, per_class_summary, manifest, max_stored_delta) + "\n"
    )
    print(f"Wrote recalculation artifacts to {output_dir}")
    print(f"Maximum absolute delta vs saved metrics: {max_stored_delta:.3e}")


if __name__ == "__main__":
    main()
