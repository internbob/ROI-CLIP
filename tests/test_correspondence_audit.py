import numpy as np
import torch

from evaluate_correspondence_audit import (
    classify_mismatches,
    deterministic_derangement,
    subject_similarity_analysis,
)


def test_subject_similarity_analysis_separates_identity_and_subject_components():
    # One-hot ROI identity gives a large ROI component. A small subject code in
    # the last two dimensions gives a positive matched-subject increment.
    surface = torch.tensor(
        [
            [[1.0, 0.0, 0.2, 0.0], [0.0, 1.0, 0.2, 0.0]],
            [[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.2]],
        ]
    )
    surface = torch.nn.functional.normalize(surface, dim=-1)
    result = subject_similarity_analysis(surface, surface.clone())
    assert result["g_subject"] > 0.0
    assert result["roi_identity_component"] > result["g_subject"]
    np.testing.assert_allclose(
        result["roi_identity_share"] + result["subject_specific_share"], 1.0
    )


def test_deterministic_derangement_has_no_fixed_points():
    first = deterministic_derangement(100, seed=123)
    second = deterministic_derangement(100, seed=123)
    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(100))
    np.testing.assert_array_equal(np.sort(first), np.arange(100))


def test_mismatch_decomposition_uses_required_precedence():
    # target 0 predicts 1: both mesh and homology, so mesh must win.
    # target 1 predicts 2: spatial only. target 2 predicts 0: remote.
    top1 = np.asarray([[1, 2, 0]])
    anatomical = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    homology = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    spatial = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    result = classify_mismatches(top1, anatomical, homology, spatial)
    assert result["count"]["true_mesh_neighbor"] == 1
    assert result["count"]["bilateral_homolog"] == 0
    assert result["count"]["spatially_close_not_mesh_adjacent"] == 1
    assert result["count"]["remote_roi"] == 1
