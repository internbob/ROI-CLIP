import torch

from models.cortical_ribbon_sampler import CorticalRibbonSampler


def test_ribbon_sampler_shapes_normalization_and_gradient():
    torch.manual_seed(3)
    sampler = CorticalRibbonSampler(feature_dim=32, num_rois=62, depth_attention=True)
    feature_map = torch.randn(2, 32, 8, 8, 8, requires_grad=True)
    ribbon_grid = torch.rand(2, 100, 5, 3) * 1.8 - 0.9
    vertex_roi_index = torch.arange(100).remainder(62).unsqueeze(0).expand(2, -1)
    ribbon_valid_mask = torch.ones(2, 100, 5, dtype=torch.bool)
    vertex_valid_mask = torch.ones(2, 100, dtype=torch.bool)

    output = sampler(
        feature_map,
        ribbon_grid,
        vertex_roi_index,
        ribbon_valid_mask,
        vertex_valid_mask,
    )

    assert output["vertex_features"].shape == (2, 100, 32)
    assert output["roi_features"].shape == (2, 62, 32)
    assert output["depth_attention"].shape == (2, 100, 5)
    assert torch.allclose(output["depth_attention"].sum(-1), torch.ones(2, 100), atol=1e-5)
    assert torch.isfinite(output["roi_features"]).all()

    output["roi_features"].square().mean().backward()
    assert feature_map.grad is not None
    assert sampler.depth_score[-1].weight.grad is not None


def test_invalid_depths_and_vertices_do_not_enter_roi_pooling():
    sampler = CorticalRibbonSampler(feature_dim=4, num_rois=2, depth_attention=False)
    feature_map = torch.ones(1, 4, 4, 4, 4)
    ribbon_grid = torch.zeros(1, 3, 3, 3)
    vertex_roi_index = torch.tensor([[0, 1, -1]])
    ribbon_valid_mask = torch.tensor([[[True, False, False], [False, False, False], [True, True, True]]])
    vertex_valid_mask = torch.tensor([[True, True, True]])

    output = sampler(
        feature_map,
        ribbon_grid,
        vertex_roi_index,
        ribbon_valid_mask,
        vertex_valid_mask,
    )

    assert torch.allclose(output["depth_attention"][0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(output["depth_attention"][0, 1], torch.zeros(3))
    assert output["roi_valid_mask"].tolist() == [[True, False]]
    assert output["roi_valid_ratio"].tolist() == [[1.0, 0.0]]
    assert torch.allclose(output["roi_features"][0, 0], torch.ones(4))
    assert torch.allclose(output["roi_features"][0, 1], torch.zeros(4))


def test_roi_valid_ratio_uses_assigned_vertices_as_denominator():
    sampler = CorticalRibbonSampler(feature_dim=2, num_rois=2, depth_attention=False)
    feature_map = torch.ones(1, 2, 3, 3, 3)
    ribbon_grid = torch.zeros(1, 5, 1, 3)
    roi_index = torch.tensor([[0, 0, 0, 1, -1]])
    depth_valid = torch.ones(1, 5, 1, dtype=torch.bool)
    vertex_valid = torch.tensor([[True, False, True, True, True]])

    output = sampler(feature_map, ribbon_grid, roi_index, depth_valid, vertex_valid)

    assert torch.allclose(output["roi_total_vertex_count"], torch.tensor([[3.0, 1.0]]))
    assert torch.allclose(output["roi_vertex_count"], torch.tensor([[2.0, 1.0]]))
    assert torch.allclose(output["roi_valid_ratio"], torch.tensor([[2.0 / 3.0, 1.0]]))
