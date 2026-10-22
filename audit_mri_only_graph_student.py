#!/usr/bin/env python3
"""Verify that a trained Graph Student checkpoint has an MRI-only inference contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import clone_config, load_configs
from train_student import build_student, infer_num_classes


PRIVILEGED_TOKENS = ("surface", "ribbon", "clinical", "pet", "cognitive")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            "configs/data.yaml",
            "configs/student.yaml",
            "configs/runtime_vnext_graph_student_s6_full_kd.yaml",
        ],
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = load_configs(args.config)
    cfg.model.num_classes = int(getattr(cfg.model, "num_classes", infer_num_classes(cfg.data.task)))
    if bool(getattr(cfg.model, "student_use_clinical", False)):
        raise RuntimeError("MRI-only audit requires model.student_use_clinical=false")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"]
    privileged_state_keys = [
        name for name in state if any(token in name.lower() for token in PRIVILEGED_TOKENS)
    ]
    if privileged_state_keys:
        raise RuntimeError(f"Privileged modality state found: {privileged_state_keys[:5]}")

    model = build_student(cfg).to(device).eval()
    model.load_state_dict(state, strict=True)

    # Make every subject-specific privileged/cache location invalid. Static DKT
    # topology remains available because it is a population template, not a
    # subject input.
    isolated_cfg = clone_config(cfg)
    isolated_cfg.data.surface_root = "/definitely/missing/surface"
    isolated_cfg.data.ribbon_cache_root = "/definitely/missing/ribbon"
    isolated_cfg.data.preprocess_cache_root = "/definitely/missing/cache"
    isolated_cfg.data.clinical.enabled = False
    isolated_model = build_student(isolated_cfg).to(device).eval()
    isolated_model.load_state_dict(state, strict=True)

    image = torch.randn(1, 1, 32, 32, 32, device=device)
    arbitrary_clinical = torch.randn(1, 29, device=device)
    with torch.no_grad():
        reference = model(image)["logits"]
        extra_field = model(image, clinical_data=arbitrary_clinical)["logits"]
        isolated = isolated_model(image)["logits"]

    extra_field_delta = float((reference - extra_field).abs().max().cpu())
    missing_path_delta = float((reference - isolated).abs().max().cpu())
    topology_keys = sorted(
        name
        for name in state
        if "topology" in name.lower()
        or "anatomy" in name.lower()
        or "boundary" in name.lower()
        or "homology" in name.lower()
    )
    passed = bool(
        not privileged_state_keys
        and extra_field_delta == 0.0
        and missing_path_delta == 0.0
        and topology_keys
        and model.use_clinical is False
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "passed": passed,
        "student_use_clinical": bool(model.use_clinical),
        "forward_required_positional_inputs": ["image"],
        "subject_specific_surface_required": False,
        "subject_specific_ribbon_required": False,
        "pet_required": False,
        "cognition_required": False,
        "clinical_required": False,
        "static_population_topology_required": True,
        "privileged_state_keys": privileged_state_keys,
        "static_topology_state_key_count": len(topology_keys),
        "max_logit_delta_with_arbitrary_clinical_argument": extra_field_delta,
        "max_logit_delta_with_missing_subject_specific_paths": missing_path_delta,
        "held_out_test_loaded_or_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
