from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_5d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 4:
        return mask.unsqueeze(1)
    return mask


def binary_dilate_3d(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    mask5d = _as_5d(mask).float()
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool3d(mask5d, kernel_size=kernel, stride=1, padding=radius)
    return dilated.squeeze(1)


def binary_erode_3d(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    inv = 1.0 - _as_5d(mask).float()
    kernel = 2 * int(radius) + 1
    eroded = 1.0 - F.max_pool3d(inv, kernel_size=kernel, stride=1, padding=radius)
    return eroded.squeeze(1)


def compute_brain_mask(volume: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    return (volume > float(threshold)).astype(np.float32)


def choose_grid_shape(num_rois: int) -> Tuple[int, int, int]:
    z_bins = max(1, int(round(num_rois ** (1.0 / 3.0))))
    y_bins = z_bins
    x_bins = max(1, int(np.ceil(num_rois / float(z_bins * y_bins))))
    while x_bins * y_bins * z_bins < num_rois:
        x_bins += 1
    return x_bins, y_bins, z_bins


def infer_spatial_roi_masks(brain_mask: np.ndarray, num_rois: int) -> np.ndarray:
    brain_mask = brain_mask.astype(np.float32, copy=False)
    coords = np.argwhere(brain_mask > 0.0)
    if coords.size == 0:
        return np.zeros((num_rois,) + brain_mask.shape, dtype=np.float32)

    x_bins, y_bins, z_bins = choose_grid_shape(num_rois)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    x_edges = np.linspace(mins[0], maxs[0], x_bins + 1)
    y_edges = np.linspace(mins[1], maxs[1], y_bins + 1)
    z_edges = np.linspace(mins[2], maxs[2], z_bins + 1)

    masks = []
    for ix in range(x_bins):
        for iy in range(y_bins):
            for iz in range(z_bins):
                mask = np.zeros_like(brain_mask, dtype=np.float32)
                x_sel = (coords[:, 0] >= x_edges[ix]) & (coords[:, 0] < x_edges[ix + 1] + 1e-6)
                y_sel = (coords[:, 1] >= y_edges[iy]) & (coords[:, 1] < y_edges[iy + 1] + 1e-6)
                z_sel = (coords[:, 2] >= z_edges[iz]) & (coords[:, 2] < z_edges[iz + 1] + 1e-6)
                selected = coords[x_sel & y_sel & z_sel]
                if selected.size == 0:
                    continue
                mask[selected[:, 0], selected[:, 1], selected[:, 2]] = 1.0
                masks.append(mask)
                if len(masks) >= num_rois:
                    break
            if len(masks) >= num_rois:
                break
        if len(masks) >= num_rois:
            break

    while len(masks) < num_rois:
        masks.append(brain_mask.copy())
    return np.stack(masks[:num_rois], axis=0)


def infer_local_and_roi_masks(volume: np.ndarray, num_rois: int, dilation_radius: int = 1):
    brain_mask = compute_brain_mask(volume)
    roi_masks = infer_spatial_roi_masks(brain_mask, num_rois)

    brain_mask_t = torch.from_numpy(brain_mask[None, ...]).float()
    eroded = binary_erode_3d(brain_mask_t, radius=1).squeeze(0)
    shell = (brain_mask_t.squeeze(0) - eroded).clamp_min(0.0).numpy()

    local_masks = []
    for roi_mask in roi_masks:
        local = roi_mask * shell
        local_t = torch.from_numpy(local[None, ...]).float()
        local_t = binary_dilate_3d(local_t, radius=dilation_radius).squeeze(0)
        local_masks.append((local_t.numpy() > 0).astype(np.float32))
    return np.stack(local_masks, axis=0), roi_masks.astype(np.float32), brain_mask.astype(np.float32)


def downsample_dense_masks(masks: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    masks = masks.float()
    if masks.ndim == 4:
        masks = masks.unsqueeze(0)
    if masks.ndim != 5:
        raise ValueError(f"Expected masks to be [B,K,H,W,D], got {masks.shape}")
    bsz, num_rois = masks.shape[:2]
    flat = masks.reshape(bsz * num_rois, 1, *masks.shape[-3:])
    down = F.interpolate(flat, size=tuple(int(v) for v in target_shape), mode="nearest")
    return down.reshape(bsz, num_rois, *target_shape)


def downsample_label_map(label_map: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    if label_map.ndim == 3:
        label_map = label_map.unsqueeze(0)
    if label_map.ndim != 4:
        raise ValueError(f"Expected label_map to be [B,H,W,D], got {label_map.shape}")
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(label_map.shape[-3:]) == target_shape:
        return label_map.long()
    label_map = F.interpolate(label_map.unsqueeze(1).float(), size=target_shape, mode="nearest")
    return label_map.squeeze(1).round().long()


def label_map_to_one_hot(label_map: torch.Tensor, num_rois: int) -> torch.Tensor:
    if label_map.ndim != 4:
        raise ValueError(f"Expected label_map to be [B,H,W,D], got {label_map.shape}")
    safe_labels = label_map.clamp(min=0, max=max(int(num_rois) - 1, 0)).long()
    one_hot = F.one_hot(safe_labels, num_classes=int(num_rois)).permute(0, 4, 1, 2, 3).float()
    valid = ((label_map >= 0) & (label_map < int(num_rois))).unsqueeze(1).float()
    return one_hot * valid


def prepare_pooling_masks(roi_inputs: torch.Tensor, target_shape: Sequence[int], num_rois: int) -> torch.Tensor:
    if roi_inputs.ndim == 5:
        return downsample_dense_masks(roi_inputs, target_shape)
    if roi_inputs.ndim == 4:
        return label_map_to_one_hot(downsample_label_map(roi_inputs, target_shape), num_rois=num_rois)
    raise ValueError(f"Unsupported ROI input shape: {roi_inputs.shape}")


def masked_pool_3d(feat_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bsz, channels, h, w, d = feat_map.shape
    if masks.shape[-3:] != (h, w, d):
        masks = downsample_dense_masks(masks, (h, w, d))
    feat_flat = feat_map.flatten(2)
    masks_flat = masks.flatten(2)
    weights = masks_flat / masks_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pooled = torch.einsum("bkn,bcn->bkc", weights, feat_flat)
    return pooled


def normalize_attention_maps(masks: torch.Tensor) -> torch.Tensor:
    flat = masks.flatten(2)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return flat.reshape_as(masks)
