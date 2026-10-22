import torch

from models.disease_aware_graph import DiseaseAwareROIGraph


def _topology(num_rois=6):
    anatomy = torch.zeros(num_rois, num_rois)
    for index in range(num_rois - 1):
        anatomy[index, index + 1] = 1.0
        anatomy[index + 1, index] = 1.0
    boundary = anatomy / anatomy.sum(dim=1, keepdim=True).clamp_min(1.0)
    homology = torch.zeros_like(anatomy)
    for index in range(num_rois // 2):
        homology[index, index + num_rois // 2] = 1.0
        homology[index + num_rois // 2, index] = 1.0
    return anatomy, boundary, homology


def test_graph_shapes_and_soft_remote_connections():
    anatomy, boundary, homology = _topology()
    graph = DiseaseAwareROIGraph(
        feature_dim=16,
        num_rois=6,
        num_heads=4,
        anatomy_adjacency=anatomy,
        boundary_strength=boundary,
        homology=homology,
        use_boundary_strength=True,
        use_homology=True,
    ).eval()
    outputs = graph(torch.randn(3, 6, 16))
    assert outputs["tokens"].shape == (3, 6, 16)
    assert outputs["attention"].shape == (3, 4, 6, 6)
    assert float(outputs["remote_mass"]) > 0.0
    assert float(outputs["attention_uniform_kl"]) >= 0.0
    assert torch.isfinite(outputs["boundary_correlation"])
    assert float(outputs["uniform_anatomical_mass"]) > 0.0
    assert 0.0 < float(outputs["residual_gate"]) < 1.0
    assert torch.allclose(outputs["mean_attention"].sum(dim=-1), torch.ones(3, 6))


def test_topology_bias_changes_attention_without_hard_masking():
    anatomy, boundary, homology = _topology()
    torch.manual_seed(7)
    unbiased = DiseaseAwareROIGraph(
        feature_dim=16,
        num_rois=6,
        num_heads=4,
        anatomy_adjacency=anatomy,
        boundary_strength=boundary,
        homology=homology,
    ).eval()
    torch.manual_seed(7)
    biased = DiseaseAwareROIGraph(
        feature_dim=16,
        num_rois=6,
        num_heads=4,
        anatomy_adjacency=anatomy,
        boundary_strength=boundary,
        homology=homology,
        use_boundary_strength=True,
        use_homology=True,
    ).eval()
    tokens = torch.randn(2, 6, 16)
    unbiased_attention = unbiased(tokens)["attention"]
    biased_attention = biased(tokens)["attention"]
    assert not torch.allclose(unbiased_attention, biased_attention)
    assert torch.all(biased_attention > 0.0)


def test_residual_gate_initialization():
    anatomy, boundary, homology = _topology()
    graph = DiseaseAwareROIGraph(
        feature_dim=16,
        num_rois=6,
        num_heads=4,
        residual_gate_init=0.1,
        anatomy_adjacency=anatomy,
        boundary_strength=boundary,
        homology=homology,
    ).eval()
    output = graph(torch.randn(1, 6, 16))
    assert torch.allclose(output["residual_gate"], torch.tensor(0.1), atol=1e-6)
