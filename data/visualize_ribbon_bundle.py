import argparse
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.transforms import center_crop_or_pad_3d, normalize_volume_minmax


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay precomputed white-pial points on cropped T1 MRI.")
    parser.add_argument("--mri", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mri-shape", type=int, nargs=3, default=[160, 192, 160])
    parser.add_argument("--slice-tolerance", type=float, default=0.75)
    parser.add_argument("--point-stride", type=int, default=4)
    return parser.parse_args()


def _scatter_near_slice(axis, points, fixed_axis, slice_index, x_axis, y_axis, color, label, tolerance, stride):
    selected = np.abs(points[:, fixed_axis] - float(slice_index)) <= float(tolerance)
    selected_points = points[selected][:: max(1, int(stride))]
    if len(selected_points):
        axis.scatter(
            selected_points[:, x_axis],
            selected_points[:, y_axis],
            s=3,
            c=color,
            alpha=0.65,
            linewidths=0,
            label=label,
        )


def render_ribbon_overlay(
    mri_path,
    bundle_path,
    output_path,
    mri_shape=(160, 192, 160),
    slice_tolerance=0.75,
    point_stride=4,
):
    """Render three orthogonal MRI/ribbon panels and return their slice indices."""
    image = nib.load(mri_path).get_fdata().astype(np.float32)
    image, _ = center_crop_or_pad_3d(image, tuple(mri_shape))
    image = normalize_volume_minmax(image)
    with np.load(bundle_path, allow_pickle=False) as payload:
        white = payload["white_voxel"].astype(np.float32)
        pial = payload["pial_voxel"].astype(np.float32)
        valid = payload["vertex_valid_mask"].astype(bool)
    white = white[valid]
    pial = pial[valid]
    middle = 0.5 * (white + pial)
    slice_indices = np.rint(np.median(middle, axis=0)).astype(int)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        (0, slice_indices[0], 1, 2, image[slice_indices[0], :, :].T, "Sagittal"),
        (1, slice_indices[1], 0, 2, image[:, slice_indices[1], :].T, "Coronal"),
        (2, slice_indices[2], 0, 1, image[:, :, slice_indices[2]].T, "Axial"),
    ]
    for axis, (fixed_axis, index, x_axis, y_axis, background, title) in zip(axes, panels):
        axis.imshow(background, cmap="gray", origin="lower")
        _scatter_near_slice(
            axis, white, fixed_axis, index, x_axis, y_axis, "cyan", "white", slice_tolerance, point_stride
        )
        _scatter_near_slice(
            axis, middle, fixed_axis, index, x_axis, y_axis, "yellow", "mid", slice_tolerance, point_stride
        )
        _scatter_near_slice(
            axis, pial, fixed_axis, index, x_axis, y_axis, "magenta", "pial", slice_tolerance, point_stride
        )
        axis.set_title(f"{title} slice {index}")
        axis.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=3)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return slice_indices.tolist()


def main():
    args = parse_args()
    render_ribbon_overlay(
        mri_path=args.mri,
        bundle_path=args.bundle,
        output_path=args.output,
        mri_shape=tuple(args.mri_shape),
        slice_tolerance=args.slice_tolerance,
        point_stride=args.point_stride,
    )


if __name__ == "__main__":
    main()
