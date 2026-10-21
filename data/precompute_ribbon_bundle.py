import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.atlas_utils import load_surface_roi_masks_from_atlas
from data.dataset import CortexWeaveROIClipDataset
from data.ribbon_geometry import (
    RIBBON_CACHE_VERSION,
    build_ribbon_bundle,
    load_ribbon_bundle,
    ribbon_bundle_path,
    save_ribbon_bundle,
)


_WORKER_DATASET = None
_WORKER_DEPTHS = None
_WORKER_CACHE_ROOT = ""
_WORKER_OVERWRITE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute real white-pial cortical ribbon grids.")
    parser.add_argument(
        "--surface-root",
        default="/public_bme/data/zhouwl2023/ADNI/BrainSurf_Results",
    )
    parser.add_argument(
        "--split-root",
        default=os.path.join(ROOT, "lookupcsv-full-huashan-adni-teacher-clean"),
    )
    parser.add_argument("--ribbon-cache-root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-vertices", type=int, default=5124)
    parser.add_argument("--num-rois", type=int, default=62)
    parser.add_argument("--mri-shape", type=int, nargs=3, default=[160, 192, 160])
    parser.add_argument("--atlas-name", default="DKTatlas")
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--allowed-subject-prefixes", nargs="+", default=["sub-ADNI"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_kwargs(args):
    return {
        "surface_root": args.surface_root,
        "task": "AD",
        "num_vertices": args.num_vertices,
        "num_rois": args.num_rois,
        "mri_shape": tuple(args.mri_shape),
        "normalize_surface": True,
        "train_mode": False,
        "require_teacher_inputs": True,
        "allowed_subject_prefixes": list(args.allowed_subject_prefixes),
        "atlas_enabled": True,
        "atlas_name": args.atlas_name,
    }


def collect_samples(args):
    kwargs = dataset_kwargs(args)
    split_dir = os.path.join(args.split_root, f"cross{args.fold}")
    samples = {}
    for split_name in ("train.csv", "valid.csv", "test.csv"):
        dataset = CortexWeaveROIClipDataset(split_csv=os.path.join(split_dir, split_name), **kwargs)
        for sample in dataset.samples:
            samples[str(sample["subject_id"])] = sample
    ordered = [samples[key] for key in sorted(samples)]
    if args.limit > 0:
        ordered = ordered[: args.limit]
    return ordered, kwargs, os.path.join(split_dir, "train.csv")


def init_worker(kwargs, split_csv, depths, cache_root, overwrite):
    global _WORKER_DATASET, _WORKER_DEPTHS, _WORKER_CACHE_ROOT, _WORKER_OVERWRITE
    _WORKER_DATASET = CortexWeaveROIClipDataset(split_csv=split_csv, **kwargs)
    _WORKER_DEPTHS = tuple(float(value) for value in depths)
    _WORKER_CACHE_ROOT = str(cache_root)
    _WORKER_OVERWRITE = bool(overwrite)


def cache_one(sample):
    started = time.perf_counter()
    sample_id = str(sample["subject_id"])
    try:
        target_affine = _WORKER_DATASET._load_volume_geometry(str(sample["mri_path"]))
        path = ribbon_bundle_path(_WORKER_CACHE_ROOT, sample_id)
        if os.path.exists(path) and not _WORKER_OVERWRITE:
            load_ribbon_bundle(
                _WORKER_CACHE_ROOT,
                sample_id,
                num_vertices=_WORKER_DATASET.num_vertices,
                num_depths=len(_WORKER_DEPTHS),
                target_shape=_WORKER_DATASET.mri_shape,
                target_affine=target_affine,
                expected_depths=_WORKER_DEPTHS,
            )
            return sample_id, "skipped", time.perf_counter() - started, ""

        fs_subject_dir = _WORKER_DATASET._fs_subject_dir(
            str(sample["subject_root"]), str(sample.get("freesurfer_subject", ""))
        )
        surface_roi_masks, _ = load_surface_roi_masks_from_atlas(
            fs_subject_dir=fs_subject_dir,
            num_vertices=_WORKER_DATASET.num_vertices,
            atlas_name=_WORKER_DATASET.atlas_name,
        )
        bundle = build_ribbon_bundle(
            fs_subject_dir=fs_subject_dir,
            target_shape=_WORKER_DATASET.mri_shape,
            target_affine=target_affine,
            num_vertices=_WORKER_DATASET.num_vertices,
            surface_roi_masks=surface_roi_masks,
            depths=_WORKER_DEPTHS,
        )
        path = save_ribbon_bundle(
            _WORKER_CACHE_ROOT,
            sample_id,
            bundle,
            overwrite=_WORKER_OVERWRITE or os.path.exists(path),
        )
        load_ribbon_bundle(
            _WORKER_CACHE_ROOT,
            sample_id,
            num_vertices=_WORKER_DATASET.num_vertices,
            num_depths=len(_WORKER_DEPTHS),
            target_shape=_WORKER_DATASET.mri_shape,
            target_affine=target_affine,
            expected_depths=_WORKER_DEPTHS,
        )
        return sample_id, "written", time.perf_counter() - started, ""
    except Exception as exc:
        return sample_id, "failed", time.perf_counter() - started, repr(exc)


def main():
    args = parse_args()
    if str(args.atlas_name).lower() not in {"dkt", "dktatlas", "aparc.dktatlas"}:
        raise ValueError("Phase 1 ribbon preprocessing currently requires subject-specific DKTatlas surfaces")
    samples, kwargs, worker_split_csv = collect_samples(args)
    os.makedirs(args.ribbon_cache_root, exist_ok=True)
    print(f"Samples: {len(samples)}", flush=True)
    print(f"Ribbon cache root: {args.ribbon_cache_root}", flush=True)
    print(f"Depths: {tuple(args.depths)}", flush=True)

    counts = {"written": 0, "skipped": 0, "failed": 0}
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.num_workers)),
        initializer=init_worker,
        initargs=(kwargs, worker_split_csv, args.depths, args.ribbon_cache_root, args.overwrite),
    ) as executor:
        for index, (sample_id, status, elapsed, error) in enumerate(
            executor.map(cache_one, samples), start=1
        ):
            counts[status] += 1
            if status == "failed" or index % 10 == 0 or index == len(samples):
                message = (
                    f"[{index}/{len(samples)}] {sample_id} {status} "
                    f"sample_time={elapsed:.1f}s totals={counts}"
                )
                if error:
                    message += f" error={error}"
                print(message, flush=True)

    manifest = {
        "ribbon_cache_version": RIBBON_CACHE_VERSION,
        "coordinate_chain": "surface-RAS -> FreeSurfer reference voxel/world -> cropped MRI voxel",
        "grid_order": "grid_sample(x=W/y=H/z=D), align_corners=False",
        "atlas_name": args.atlas_name,
        "target_mri_shape": list(args.mri_shape),
        "num_vertices": args.num_vertices,
        "num_rois": args.num_rois,
        "depths": list(args.depths),
        "counts": counts,
        "num_samples": len(samples),
        "elapsed_seconds": time.perf_counter() - started,
        "surface_root": os.path.abspath(args.surface_root),
        "split_root": os.path.abspath(args.split_root),
    }
    manifest_path = os.path.join(args.ribbon_cache_root, "ribbon_manifest.json")
    temp_path = f"{manifest_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, manifest_path)
    print(f"Finished: {counts}", flush=True)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
