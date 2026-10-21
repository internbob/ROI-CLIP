import numpy as np
import pytest

from data.ribbon_geometry import (
    load_ribbon_bundle,
    save_ribbon_bundle,
    voxel_to_grid_sample_coordinates,
)


def test_voxel_coordinates_are_reordered_for_5d_grid_sample():
    voxel = np.asarray([[[0.0, 0.0, 0.0], [7.0, 5.0, 3.0]]], dtype=np.float32)
    grid = voxel_to_grid_sample_coordinates(voxel, target_shape=(8, 6, 4))

    np.testing.assert_allclose(grid[0, 0], [-0.75, -5.0 / 6.0, -0.875], atol=1e-6)
    np.testing.assert_allclose(grid[0, 1], [0.75, 5.0 / 6.0, 0.875], atol=1e-6)


def test_ribbon_cache_validates_shape_and_affine(tmp_path):
    affine = np.eye(4, dtype=np.float32)
    bundle = {
        "ribbon_grid": np.zeros((8, 5, 3), dtype=np.float32),
        "ribbon_valid_mask": np.ones((8, 5), dtype=np.bool_),
        "vertex_valid_mask": np.ones((8,), dtype=np.bool_),
        "vertex_roi_index": np.arange(8, dtype=np.int64) % 2,
        "white_voxel": np.zeros((8, 3), dtype=np.float32),
        "pial_voxel": np.ones((8, 3), dtype=np.float32),
        "depths": np.linspace(0.0, 1.0, 5, dtype=np.float32),
        "target_mri_shape": np.asarray((8, 8, 8), dtype=np.int32),
        "target_mri_affine": affine,
    }
    save_ribbon_bundle(str(tmp_path), "subject", bundle)

    depths = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    loaded = load_ribbon_bundle(
        str(tmp_path), "subject", 8, 5, (8, 8, 8), affine, expected_depths=depths
    )
    assert loaded["ribbon_grid"].shape == (8, 5, 3)
    assert loaded["vertex_roi_index"].dtype == np.int64

    shifted_affine = affine.copy()
    shifted_affine[0, 3] = 1.0
    with pytest.raises(ValueError, match="affine mismatch"):
        load_ribbon_bundle(str(tmp_path), "subject", 8, 5, (8, 8, 8), shifted_affine)

    inner_depths = [0.2, 0.35, 0.5, 0.65, 0.8]
    resampled = load_ribbon_bundle(
        str(tmp_path),
        "subject",
        8,
        5,
        (8, 8, 8),
        affine,
        expected_depths=inner_depths,
    )
    expected_voxel = np.asarray(inner_depths, dtype=np.float32)[None, :, None]
    expected_voxel = np.broadcast_to(expected_voxel, (8, 5, 3))
    np.testing.assert_allclose(
        resampled["ribbon_grid"],
        voxel_to_grid_sample_coordinates(expected_voxel, (8, 8, 8)),
        atol=1e-6,
    )
