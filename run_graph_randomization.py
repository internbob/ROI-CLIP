#!/usr/bin/env python3
"""Run the selected topology-randomization control after the Stage A gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "g2": ROOT / "configs" / "runtime_vnext_graph_g2r_shuffled_boundary.yaml",
    "g3": ROOT / "configs" / "runtime_vnext_graph_g3r_shuffled_boundary_homology.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--stage-a-dir",
        default=str(ROOT / "outputs" / "vnext_graph_stageA_20261019"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def selected_variant(summary_path):
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    gate = payload.get("gate", {})
    if not bool(gate.get("graph_teacher_gate_passed", False)):
        raise RuntimeError("Stage A Graph gate did not pass; randomization training is not authorized")
    selected = str(gate.get("selected_topology_variant", ""))
    if selected not in CONFIGS:
        raise RuntimeError(f"Unsupported selected topology variant: {selected!r}")
    return selected


def main():
    args = parse_args()
    stage_a_dir = Path(args.stage_a_dir).resolve()
    summary_path = stage_a_dir / "graph_stage_a_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Stage A summary not found: {summary_path}")
    selected = selected_variant(summary_path)
    randomized_name = f"{selected}r"
    output_dir = stage_a_dir / f"{randomized_name}_fold{args.fold}"
    command = [
        sys.executable,
        str(ROOT / "train_teacher.py"),
        "--config",
        str(ROOT / "configs" / "data.yaml"),
        str(ROOT / "configs" / "teacher.yaml"),
        str(CONFIGS[selected]),
        "--fold",
        str(args.fold),
        "--device",
        str(args.device),
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
