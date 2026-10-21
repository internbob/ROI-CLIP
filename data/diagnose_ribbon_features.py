import argparse
import copy
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, load_configs
from data import build_dataloaders
from models.cortical_ribbon_sampler import CorticalRibbonSampler
from train_teacher import build_model
from utils import move_to_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Phase 1 ribbon geometry and token statistics.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--ribbon-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def to_config(value):
    if isinstance(value, dict):
        return Config({key: to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config(item) for item in value]
    return value


def load_state(model, path, device):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    return int(payload.get("epoch", -1))


def grid_to_feature_voxel(grid, feature_shape):
    depth, height, width = (int(value) for value in feature_shape)
    size_xyz = grid.new_tensor([width, height, depth])
    return (grid + 1.0) * size_xyz / 2.0 - 0.5


def adjacent_distance(grid, valid, feature_shape):
    voxel = grid_to_feature_voxel(grid, feature_shape)
    distance = torch.linalg.vector_norm(voxel[:, :, 1:] - voxel[:, :, :-1], dim=-1)
    pair_valid = valid[:, :, 1:] & valid[:, :, :-1]
    return distance[pair_valid].detach().cpu()


def adjacent_cosine(depth_features, valid):
    cosine = F.cosine_similarity(depth_features[:, :, 1:], depth_features[:, :, :-1], dim=-1)
    pair_valid = valid[:, :, 1:] & valid[:, :, :-1]
    return cosine[pair_valid].detach().cpu()


def distribution(values):
    values = torch.cat(values).float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "fraction_below_0_5": float((values < 0.5).float().mean()),
    }


def token_statistics(tokens):
    tokens = torch.cat(tokens, dim=0).float()
    return {
        "shape": list(tokens.shape),
        "mean": float(tokens.mean()),
        "std": float(tokens.std(unbiased=False)),
        "mean_l2_norm": float(torch.linalg.vector_norm(tokens, dim=-1).mean()),
        "channel_variance": float(tokens.var(dim=-1, unbiased=False).mean()),
        "inter_subject_variance": float(tokens.var(dim=0, unbiased=False).mean()),
        "inter_roi_variance": float(tokens.var(dim=1, unbiased=False).mean()),
    }


def main():
    args = parse_args()
    cfg = load_configs(args.config)
    cfg.runtime.fold = int(args.fold)
    cfg.runtime.device = args.device
    cfg.runtime.dry_run = False
    cfg.model.volume_pooling.mode = "ribbon"
    seed_everything(int(cfg.training.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = build_dataloaders(
        cfg,
        fold=args.fold,
        require_teacher_inputs=True,
        filter_teacher_available=True,
    )
    legacy_cfg = to_config(copy.deepcopy(dict(cfg)))
    legacy_cfg.model.volume_pooling.mode = "legacy_mask"
    ribbon_cfg = to_config(copy.deepcopy(dict(cfg)))
    ribbon_cfg.model.volume_pooling.mode = "ribbon"
    legacy_model = build_model(legacy_cfg).to(device).eval()
    ribbon_model = build_model(ribbon_cfg).to(device).eval()
    legacy_epoch = load_state(legacy_model, args.legacy_checkpoint, device)
    ribbon_epoch = load_state(ribbon_model, args.ribbon_checkpoint, device)

    current_distances = []
    highres_inner_distances = []
    current_cosines = []
    highres_inner_cosines = []
    legacy_tokens = []
    ribbon_tokens = []
    roi_valid_ratios = []
    roi_vertex_counts = []
    observed_shapes = None
    inner_depths = torch.tensor([0.2, 0.35, 0.5, 0.65, 0.8], device=device)

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= int(args.num_batches):
                break
            batch = move_to_device(batch, device)
            image = batch["image"]
            ribbon_feat, ribbon_multiscale, _ = ribbon_model.volume_encoder(image)
            legacy_feat, _, _ = legacy_model.volume_encoder(image)
            highres_feat = ribbon_multiscale[0]
            if observed_shapes is None:
                observed_shapes = {
                    "input": list(image.shape),
                    "current_ribbon_feature": list(ribbon_feat.shape),
                    "high_resolution_feature": list(highres_feat.shape),
                    "multi_scale": [list(feature.shape) for feature in ribbon_multiscale],
                    "current_effective_stride_dhw": [
                        float(image.shape[-3] / ribbon_feat.shape[-3]),
                        float(image.shape[-2] / ribbon_feat.shape[-2]),
                        float(image.shape[-1] / ribbon_feat.shape[-1]),
                    ],
                    "highres_effective_stride_dhw": [
                        float(image.shape[-3] / highres_feat.shape[-3]),
                        float(image.shape[-2] / highres_feat.shape[-2]),
                        float(image.shape[-1] / highres_feat.shape[-1]),
                    ],
                }

            current = ribbon_model.ribbon_sampler(
                ribbon_feat,
                batch["ribbon_grid"],
                batch["vertex_roi_index"],
                batch["ribbon_valid_mask"],
                batch["vertex_valid_mask"],
            )
            legacy = legacy_model.volume_pool(
                legacy_feat,
                batch["local_ribbon_masks"],
                batch["roi_volume_masks"],
            )[0]

            endpoint_grid = batch["ribbon_grid"][:, :, [0, -1]]
            inner_grid = (
                (1.0 - inner_depths[None, None, :, None]) * endpoint_grid[:, :, :1]
                + inner_depths[None, None, :, None] * endpoint_grid[:, :, 1:]
            )
            inner_valid = (inner_grid >= -1.0).all(dim=-1) & (inner_grid <= 1.0).all(dim=-1)
            highres_sampler = CorticalRibbonSampler(
                feature_dim=highres_feat.shape[1],
                num_rois=int(cfg.model.num_rois),
                depth_attention=False,
            ).to(device)
            highres = highres_sampler(
                highres_feat,
                inner_grid,
                batch["vertex_roi_index"],
                inner_valid,
                batch["vertex_valid_mask"],
            )

            current_distances.append(
                adjacent_distance(batch["ribbon_grid"], batch["ribbon_valid_mask"], ribbon_feat.shape[-3:])
            )
            highres_inner_distances.append(
                adjacent_distance(inner_grid, inner_valid, highres_feat.shape[-3:])
            )
            current_cosines.append(adjacent_cosine(current["depth_features"], batch["ribbon_valid_mask"]))
            highres_inner_cosines.append(adjacent_cosine(highres["depth_features"], inner_valid))
            legacy_tokens.append(legacy.detach().cpu())
            ribbon_tokens.append(current["roi_features"].detach().cpu())

            roi_index = batch["vertex_roi_index"]
            assigned = (roi_index >= 0) & (roi_index < int(cfg.model.num_rois))
            safe_index = roi_index.clamp(0, int(cfg.model.num_rois) - 1)
            total = image.new_zeros((image.shape[0], int(cfg.model.num_rois)))
            valid = image.new_zeros((image.shape[0], int(cfg.model.num_rois)))
            total.scatter_add_(1, safe_index, assigned.float())
            valid.scatter_add_(1, safe_index, (assigned & current["vertex_valid_mask"]).float())
            roi_valid_ratios.append((valid / total.clamp_min(1.0)).cpu())
            roi_vertex_counts.append(total.cpu())

    if observed_shapes is None:
        raise RuntimeError("No validation batches were available")
    ratios = torch.cat(roi_valid_ratios, dim=0).numpy()
    counts = torch.cat(roi_vertex_counts, dim=0).numpy()
    region_names = list(getattr(val_loader.dataset, "atlas_region_names", []))
    if len(region_names) != int(cfg.model.num_rois):
        region_names = [f"roi_{index:02d}" for index in range(int(cfg.model.num_rois))]

    os.makedirs(args.output_dir, exist_ok=True)
    roi_rows = []
    for roi_index, region_name in enumerate(region_names):
        values = ratios[:, roi_index]
        roi_rows.append(
            {
                "roi_index": roi_index,
                "roi": region_name,
                "mean_valid_ratio": float(values.mean()),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "mean_vertex_count": float(counts[:, roi_index].mean()),
            }
        )
    with open(os.path.join(args.output_dir, "roi_validity.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(roi_rows[0]))
        writer.writeheader()
        writer.writerows(roi_rows)

    current_distance = distribution(current_distances)
    highres_distance = distribution(highres_inner_distances)
    current_cosine = distribution(current_cosines)
    highres_cosine = distribution(highres_inner_cosines)
    for stats in (current_cosine, highres_cosine):
        stats.pop("fraction_below_0_5")
    result = {
        "fold": int(args.fold),
        "num_batches": min(int(args.num_batches), len(val_loader)),
        "num_subjects": int(sum(tokens.shape[0] for tokens in legacy_tokens)),
        "checkpoints": {
            "legacy": os.path.abspath(args.legacy_checkpoint),
            "legacy_epoch": legacy_epoch,
            "ribbon": os.path.abspath(args.ribbon_checkpoint),
            "ribbon_epoch": ribbon_epoch,
        },
        "augmentation_audit": {
            "spatial_transform": "zero-fill voxel shift only",
            "max_voxel_shift": int(cfg.data.mri_aug.max_voxel_shift),
            "coordinates_synchronized": int(cfg.data.mri_aug.max_voxel_shift) == 0,
            "intensity_only_when_shift_zero": ["scale", "shift", "gamma", "Gaussian noise"],
        },
        "grid_sample": {
            "coordinate_order": "grid_sample x=W,y=H,z=D",
            "align_corners": False,
            "normalized_formula": "2*(voxel+0.5)/shape-1",
        },
        "feature_shapes": observed_shapes,
        "current_depth_distance_feature_voxels": current_distance,
        "highres_inner_depth_distance_feature_voxels": highres_distance,
        "current_adjacent_depth_cosine": current_cosine,
        "highres_inner_adjacent_depth_cosine": highres_cosine,
        "token_statistics": {
            "legacy": token_statistics(legacy_tokens),
            "ribbon": token_statistics(ribbon_tokens),
        },
        "roi_validity": {
            "minimum_mean_valid_ratio": min(row["mean_valid_ratio"] for row in roi_rows),
            "median_mean_valid_ratio": float(np.median([row["mean_valid_ratio"] for row in roi_rows])),
            "maximum_mean_valid_ratio": max(row["mean_valid_ratio"] for row in roi_rows),
            "csv": "roi_validity.csv",
        },
    }
    output_path = os.path.join(args.output_dir, "diagnostics.json")
    temp_path = f"{output_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, output_path)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
