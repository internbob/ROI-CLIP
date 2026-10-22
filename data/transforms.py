from typing import List, Sequence, Tuple

import numpy as np


def center_crop_or_pad_voxel_offset(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> Tuple[int, int, int]:
    """Return the source-voxel origin represented by target voxel ``(0, 0, 0)``."""
    source = tuple(int(v) for v in source_shape)
    target = tuple(int(v) for v in target_shape)
    if len(source) != 3 or len(target) != 3:
        raise ValueError(f"Expected 3D shapes, got {source} -> {target}")

    offsets = []
    for dim, tgt in zip(source, target):
        if dim <= 0 or tgt <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {source} -> {target}")
        before_pad = max(tgt - dim, 0) // 2
        padded_dim = max(dim, tgt)
        crop_start = max((padded_dim - tgt) // 2, 0)
        offsets.append(int(crop_start - before_pad))
    return tuple(offsets)


def center_crop_or_pad_3d(volume: np.ndarray, target_shape: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    target = tuple(int(v) for v in target_shape)
    if volume.ndim != 3 or len(target) != 3:
        raise ValueError(f"Expected a 3D volume and target shape, got {volume.shape} -> {target}")
    voxel_offset = center_crop_or_pad_voxel_offset(volume.shape, target)
    out = volume.astype(np.float32, copy=False)

    pads: List[Tuple[int, int]] = []
    for dim, tgt in zip(out.shape, target):
        total_pad = max(tgt - dim, 0)
        before = total_pad // 2
        after = total_pad - before
        pads.append((before, after))
    if any(before > 0 or after > 0 for before, after in pads):
        out = np.pad(out, pads, mode="constant")

    slices = []
    for dim, tgt in zip(out.shape, target):
        start = max((dim - tgt) // 2, 0)
        slices.append(slice(start, start + tgt))
    return out[tuple(slices)], voxel_offset


def adjust_affine_for_voxel_offset(affine: np.ndarray, voxel_offset: Sequence[int]) -> np.ndarray:
    adjusted = affine.astype(np.float32).copy()
    offset = np.asarray(voxel_offset, dtype=np.float32)
    adjusted[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]
    return adjusted


def normalize_volume_minmax(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    lo = float(volume.min())
    hi = float(volume.max())
    if hi - lo < 1e-6:
        return volume - lo
    return (volume - lo) / (hi - lo)


def shift_volume_zero_fill(volume: np.ndarray, shifts: Sequence[int]) -> np.ndarray:
    out = np.zeros_like(volume, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for axis, shift in enumerate(shifts, start=1):
        dim = volume.shape[axis]
        shift = int(shift)
        if shift >= 0:
            src_start = 0
            src_end = max(dim - shift, 0)
            dst_start = min(shift, dim)
            dst_end = dim
        else:
            src_start = min(-shift, dim)
            src_end = dim
            dst_start = 0
            dst_end = max(dim + shift, 0)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[(slice(None),) + tuple(dst_slices)] = volume[(slice(None),) + tuple(src_slices)]
    return out


def random_augment_volume(
    volume: np.ndarray,
    prob: float = 0.0,
    max_voxel_shift: int = 0,
    intensity_scale: float = 0.0,
    intensity_shift: float = 0.0,
    gamma: float = 0.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    if prob <= 0.0 or np.random.rand() > prob:
        return volume.astype(np.float32, copy=False)

    out = volume.astype(np.float32, copy=True)
    if max_voxel_shift > 0:
        shifts = [np.random.randint(-max_voxel_shift, max_voxel_shift + 1) for _ in range(3)]
        if any(shift != 0 for shift in shifts):
            out = shift_volume_zero_fill(out, shifts)
    if intensity_scale > 0.0:
        out *= np.float32(1.0 + np.random.uniform(-intensity_scale, intensity_scale))
    if intensity_shift > 0.0:
        out += np.float32(np.random.uniform(-intensity_shift, intensity_shift))
    out = np.clip(out, 0.0, 1.0)
    if gamma > 0.0:
        gamma_value = np.random.uniform(max(0.5, 1.0 - gamma), 1.0 + gamma)
        out = np.power(out, np.float32(gamma_value), dtype=np.float32)
    if noise_std > 0.0:
        sigma = np.random.uniform(0.0, noise_std)
        out += np.random.normal(loc=0.0, scale=sigma, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
