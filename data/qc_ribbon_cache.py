import argparse
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import CortexWeaveROIClipDataset
from data.ribbon_geometry import ribbon_bundle_path
from data.visualize_ribbon_bundle import render_ribbon_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Create reproducible multi-subject ribbon-coordinate QC.")
    parser.add_argument("--surface-root", default="/public_bme/data/zhouwl2023/ADNI/BrainSurf_Results")
    parser.add_argument("--split-root", default=os.path.join(ROOT, "lookupcsv-full-huashan-adni-teacher-clean"))
    parser.add_argument("--ribbon-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-subjects", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-vertices", type=int, default=5124)
    parser.add_argument("--num-rois", type=int, default=62)
    parser.add_argument("--mri-shape", type=int, nargs=3, default=[160, 192, 160])
    parser.add_argument("--atlas-name", default="DKTatlas")
    return parser.parse_args()


def collect_samples(args):
    kwargs = {
        "surface_root": args.surface_root,
        "task": "AD",
        "num_vertices": args.num_vertices,
        "num_rois": args.num_rois,
        "mri_shape": tuple(args.mri_shape),
        "normalize_surface": True,
        "train_mode": False,
        "require_teacher_inputs": True,
        "allowed_subject_prefixes": ["sub-ADNI"],
        "atlas_enabled": True,
        "atlas_name": args.atlas_name,
    }
    split_dir = os.path.join(args.split_root, f"cross{args.fold}")
    samples = {}
    for split_name in ("train.csv", "valid.csv", "test.csv"):
        dataset = CortexWeaveROIClipDataset(split_csv=os.path.join(split_dir, split_name), **kwargs)
        for sample in dataset.samples:
            samples[str(sample["subject_id"])] = sample
    return samples


def summarize_bundle(path):
    with np.load(path, allow_pickle=False) as payload:
        grid = payload["ribbon_grid"].astype(np.float32)
        depth_valid = payload["ribbon_valid_mask"].astype(bool)
        vertex_valid = payload["vertex_valid_mask"].astype(bool)
        roi_index = payload["vertex_roi_index"].astype(np.int64)
        white = payload["white_voxel"].astype(np.float32)
        pial = payload["pial_voxel"].astype(np.float32)
    thickness = np.linalg.norm(pial - white, axis=-1)
    return {
        "finite_grid": bool(np.isfinite(grid).all()),
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "valid_depth_fraction": float(depth_valid.mean()),
        "valid_vertex_fraction": float(vertex_valid.mean()),
        "roi_assigned_fraction": float((roi_index >= 0).mean()),
        "ribbon_thickness_voxels_mean": float(thickness.mean()),
        "ribbon_thickness_voxels_median": float(np.median(thickness)),
    }


def main():
    args = parse_args()
    samples = collect_samples(args)
    available = [
        sample_id
        for sample_id in sorted(samples)
        if os.path.exists(ribbon_bundle_path(args.ribbon_cache_root, sample_id))
    ]
    if len(available) < args.num_subjects:
        raise RuntimeError(f"Only {len(available)} cached subjects are available; need {args.num_subjects}")
    selected = random.Random(args.seed).sample(available, args.num_subjects)
    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    for sample_id in selected:
        sample = samples[sample_id]
        bundle_path = ribbon_bundle_path(args.ribbon_cache_root, sample_id)
        image_name = f"{sample_id}_ribbon_qc.png"
        slice_indices = render_ribbon_overlay(
            mri_path=str(sample["mri_path"]),
            bundle_path=bundle_path,
            output_path=os.path.join(args.output_dir, image_name),
            mri_shape=tuple(args.mri_shape),
        )
        record = {
            "sample_id": sample_id,
            "mri_path": os.path.abspath(str(sample["mri_path"])),
            "bundle_path": os.path.abspath(bundle_path),
            "overlay": image_name,
            "slice_indices_ijk": slice_indices,
            **summarize_bundle(bundle_path),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "fold": args.fold,
        "seed": args.seed,
        "sampling": "seeded random sample without replacement from sorted cached fold union",
        "num_available": len(available),
        "num_subjects": len(records),
        "all_finite": all(record["finite_grid"] for record in records),
        "mean_valid_depth_fraction": float(np.mean([record["valid_depth_fraction"] for record in records])),
        "mean_valid_vertex_fraction": float(np.mean([record["valid_vertex_fraction"] for record in records])),
        "subjects": records,
    }
    summary_path = os.path.join(args.output_dir, "qc_summary.json")
    temp_path = f"{summary_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, summary_path)
    print(f"Saved QC summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
