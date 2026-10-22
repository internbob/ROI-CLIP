#!/usr/bin/env python3
"""Compare the selected true-topology graph with its shuffled-topology control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_graph_stage_a import mcnemar, paired_subject_bootstrap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-dir", default="outputs/vnext_graph_stageA_20261019")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pair(root, selected, randomized, folds):
    metrics = {selected: [], randomized: []}
    predictions = {selected: {}, randomized: {}}
    for variant in (selected, randomized):
        for fold in folds:
            fold_root = root / f"{variant}_fold{fold}" / f"fold_{fold}"
            metric_path = fold_root / "development_metrics.json"
            prediction_path = fold_root / "validation_predictions.json"
            if not metric_path.is_file() or not prediction_path.is_file():
                raise FileNotFoundError(f"Randomization control is incomplete under {fold_root}")
            metric_payload = read_json(metric_path)
            if bool(metric_payload.get("held_out_test_evaluated", True)):
                raise RuntimeError(f"Held-out test flag is not false: {metric_path}")
            metrics[variant].append(metric_payload["best_val_metrics"])
            prediction_payload = read_json(prediction_path)
            if prediction_payload.get("split") != "validation":
                raise RuntimeError(f"Non-validation predictions found: {prediction_path}")
            for record in prediction_payload["records"]:
                predictions[variant][record["filename"]] = record
    if set(predictions[selected]) != set(predictions[randomized]):
        raise RuntimeError("True and randomized topology prediction keys differ")
    for key in predictions[selected]:
        if int(predictions[selected][key]["label"]) != int(predictions[randomized][key]["label"]):
            raise RuntimeError(f"Prediction label mismatch: {key}")
    return metrics, predictions


def metric_values(metrics, variant, name):
    return np.asarray([float(item[name]) for item in metrics[variant]], dtype=np.float64)


def fmt(value):
    return f"{float(value):.5f}"


def main():
    args = parse_args()
    root = Path(args.stage_a_dir).resolve()
    stage_summary = read_json(root / "graph_stage_a_summary.json")
    stage_gate = stage_summary.get("gate", {})
    if not bool(stage_gate.get("graph_teacher_gate_passed", False)):
        raise RuntimeError("Stage A Graph gate did not pass")
    selected = str(stage_gate.get("selected_topology_variant", ""))
    if selected not in {"g2", "g3"}:
        raise RuntimeError(f"Unsupported selected topology variant: {selected!r}")
    randomized = f"{selected}r"
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    metrics, predictions = load_pair(root, selected, randomized, folds)

    fold_deltas = {}
    for name in (
        "mcc",
        "auc",
        "acc",
        "f1_macro",
        "balanced_acc",
        "graph_anatomical_mass",
        "graph_homology_mass",
        "graph_boundary_correlation",
        "graph_remote_mass",
    ):
        fold_deltas[name] = (
            metric_values(metrics, selected, name) - metric_values(metrics, randomized, name)
        ).tolist()

    aligned = {"g0": predictions[randomized], selected: predictions[selected]}
    bootstrap = []
    for index, metric in enumerate(("mcc", "auc", "balanced_acc")):
        bootstrap.append(
            paired_subject_bootstrap(
                aligned,
                candidate=selected,
                metric=metric,
                replicates=int(args.bootstrap_replicates),
                seed=int(args.seed) + index,
            )
        )
    mcnemar_result = mcnemar(aligned, selected)

    mcc_delta = np.asarray(fold_deltas["mcc"])
    boundary_delta = np.asarray(fold_deltas["graph_boundary_correlation"])
    anatomy_delta = np.asarray(fold_deltas["graph_anatomical_mass"])
    homology_delta = np.asarray(fold_deltas["graph_homology_mass"])
    no_randomization_disadvantage = bool(mcc_delta.mean() >= -0.005 and np.sum(mcc_delta >= 0.0) >= 1)
    anatomy_specific_evidence = bool(
        mcc_delta.mean() > 0.0
        and np.sum(mcc_delta > 0.0) >= 2
        and boundary_delta.mean() > 0.0
        and anatomy_delta.mean() > 0.0
        and (selected != "g3" or homology_delta.mean() > 0.0)
    )
    gate = {
        "selected_true_topology_variant": selected,
        "randomized_control": randomized,
        "mean_mcc_delta_true_minus_randomized": float(mcc_delta.mean()),
        "improved_folds": int(np.sum(mcc_delta > 0.0)),
        "no_randomization_disadvantage": no_randomization_disadvantage,
        "anatomy_specific_evidence": anatomy_specific_evidence,
        "student_stage_authorized": bool(
            stage_gate.get("student_stage_authorized", False) and no_randomization_disadvantage
        ),
    }
    payload = {
        "folds": folds,
        "held_out_test_evaluated": False,
        "fold_deltas_true_minus_randomized": fold_deltas,
        "paired_bootstrap": bootstrap,
        "mcnemar": mcnemar_result,
        "gate": gate,
    }
    (root / "graph_randomization_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# CortexBridge Graph Topology-Randomization Control",
        "",
        f"- True topology variant: `{selected}`.",
        f"- Shuffled control: `{randomized}`.",
        "- Clinical input: `off`.",
        "- Held-out test loaded/evaluated: `false`.",
        "",
        "## Fold-level true − randomized deltas",
        "",
        "| Metric | Fold deltas | Mean delta |",
        "|---|---:|---:|",
    ]
    for name, values in fold_deltas.items():
        lines.append(
            f"| {name} | {', '.join(fmt(value) for value in values)} | {fmt(np.mean(values))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- No material shuffled-topology advantage: `{str(no_randomization_disadvantage).lower()}`.",
            f"- Anatomy-specific evidence: `{str(anatomy_specific_evidence).lower()}`.",
            f"- Student stage authorized: `{str(gate['student_stage_authorized']).lower()}`.",
            "",
            "Anatomy-specific evidence requires true topology to improve MCC in at least two folds and to increase true boundary/anatomy attention diagnostics. If this condition fails, the manuscript must not attribute a graph benefit specifically to the DKT prior.",
        ]
    )
    (root / "GRAPH_RANDOMIZATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
