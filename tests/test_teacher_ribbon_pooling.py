import torch

from models.teacher import CortexWeaveROIClipTeacher


def _teacher(mode, **kwargs):
    return CortexWeaveROIClipTeacher(
        surface_in_channels=3,
        surface_hidden_dim=8,
        num_vertices=84,
        feature_dim=16,
        proj_dim=16,
        num_rois=4,
        num_classes=2,
        base_channels=4,
        dropout=0.0,
        volume_pooling_mode=mode,
        **kwargs,
    )


def _inputs():
    image = torch.randn(2, 1, 32, 32, 32)
    surface = torch.randn(2, 84, 3)
    surface_masks = torch.zeros(2, 4, 84)
    for roi in range(4):
        surface_masks[:, roi, roi * 21 : (roi + 1) * 21] = 1.0
    volume_masks = torch.rand(2, 4, 4, 4, 4)
    return image, surface, surface_masks, volume_masks


def test_legacy_mode_does_not_add_ribbon_parameters():
    teacher = _teacher("legacy_mask")
    assert teacher.ribbon_sampler is None
    assert not any(name.startswith("ribbon_sampler") for name in teacher.state_dict())


def test_teacher_ribbon_mode_replaces_only_volume_roi_tokens():
    torch.manual_seed(5)
    teacher = _teacher("ribbon")
    image, surface, surface_masks, volume_masks = _inputs()
    ribbon_grid = torch.rand(2, 84, 5, 3) * 1.6 - 0.8
    ribbon_valid = torch.ones(2, 84, 5, dtype=torch.bool)
    vertex_valid = torch.ones(2, 84, dtype=torch.bool)
    vertex_roi = torch.arange(84).remainder(4).unsqueeze(0).expand(2, -1)

    output = teacher(
        image,
        surface,
        surface_masks,
        volume_masks,
        volume_masks,
        ribbon_grid=ribbon_grid,
        ribbon_valid_mask=ribbon_valid,
        vertex_valid_mask=vertex_valid,
        vertex_roi_index=vertex_roi,
    )
    output["logits"].sum().backward()

    assert output["mri_ribbon_tokens"].shape == (2, 4, 16)
    assert output["volume_tokens"].shape == (2, 4, 16)
    assert output["z_volume"].shape == (2, 4, 16)
    assert output["depth_attention"].shape == (2, 84, 5)
    assert output["legacy_volume_tokens"].shape == (2, 4, 16)
    assert teacher.ribbon_sampler.depth_score[-1].weight.grad is not None


def test_high_resolution_uniform_ribbon_projects_to_roi_feature_dim():
    teacher = _teacher("ribbon", ribbon_feature_stage="high_res", ribbon_depth_attention=False)
    image, surface, surface_masks, volume_masks = _inputs()
    ribbon_grid = torch.rand(2, 84, 5, 3) * 1.6 - 0.8
    vertex_roi = torch.arange(84).remainder(4).unsqueeze(0).expand(2, -1)
    output = teacher(
        image,
        surface,
        surface_masks,
        volume_masks,
        volume_masks,
        ribbon_grid=ribbon_grid,
        ribbon_valid_mask=torch.ones(2, 84, 5, dtype=torch.bool),
        vertex_valid_mask=torch.ones(2, 84, dtype=torch.bool),
        vertex_roi_index=vertex_roi,
    )

    assert teacher.ribbon_sampler.feature_dim == 4
    assert output["mri_ribbon_tokens_raw"].shape == (2, 4, 4)
    assert output["mri_ribbon_tokens"].shape == (2, 4, 16)
    assert torch.allclose(output["depth_attention"], torch.full((2, 84, 5), 0.2), atol=1e-6)


def test_hybrid_ribbon_is_exact_legacy_at_initialization():
    torch.manual_seed(11)
    teacher = _teacher(
        "hybrid_ribbon",
        ribbon_feature_stage="high_res",
        ribbon_depth_attention=False,
        ribbon_gate_init=0.1,
    )
    image, surface, surface_masks, volume_masks = _inputs()
    ribbon_grid = torch.rand(2, 84, 5, 3) * 1.6 - 0.8
    vertex_roi = torch.arange(84).remainder(4).unsqueeze(0).expand(2, -1)
    output = teacher(
        image,
        surface,
        surface_masks,
        volume_masks,
        volume_masks,
        ribbon_grid=ribbon_grid,
        ribbon_valid_mask=torch.ones(2, 84, 5, dtype=torch.bool),
        vertex_valid_mask=torch.ones(2, 84, dtype=torch.bool),
        vertex_roi_index=vertex_roi,
    )

    assert torch.equal(output["volume_tokens"], output["legacy_volume_tokens"])
    assert torch.count_nonzero(output["hybrid_ribbon_delta"]) == 0
    assert torch.allclose(output["hybrid_ribbon_gate"], torch.full((2, 4, 1), 0.1), atol=1e-6)
    output["logits"].sum().backward()
    assert teacher.hybrid_ribbon_adapter[-1].weight.grad is not None


def test_hybrid_additions_do_not_change_common_same_seed_initialization():
    torch.manual_seed(29)
    legacy = _teacher("legacy_mask")
    torch.manual_seed(29)
    hybrid = _teacher(
        "hybrid_ribbon",
        ribbon_feature_stage="high_res",
        ribbon_depth_attention=False,
    )

    legacy_state = legacy.state_dict()
    hybrid_state = hybrid.state_dict()
    common_names = [
        name
        for name in legacy_state
        if not name.startswith("ribbon") and not name.startswith("hybrid")
    ]
    assert common_names
    for name in common_names:
        assert torch.equal(legacy_state[name], hybrid_state[name]), name


def test_cross_geometry_anchor_does_not_change_legacy_classification_path():
    image, surface, surface_masks, volume_masks = _inputs()
    ribbon_grid = torch.rand(2, 84, 5, 3) * 1.6 - 0.8
    ribbon_valid = torch.ones(2, 84, 5, dtype=torch.bool)
    vertex_valid = torch.ones(2, 84, dtype=torch.bool)
    vertex_roi = torch.arange(84).remainder(4).unsqueeze(0).expand(2, -1)

    torch.manual_seed(37)
    baseline = _teacher("legacy_mask")
    torch.manual_seed(37)
    aligned = _teacher(
        "legacy_mask",
        cross_geometry_enabled=True,
        cross_geometry_dim=12,
    )
    baseline.eval()
    aligned.eval()

    baseline_output = baseline(image, surface, surface_masks, volume_masks, volume_masks)
    aligned_output = aligned(
        image,
        surface,
        surface_masks,
        volume_masks,
        volume_masks,
        ribbon_grid=ribbon_grid,
        ribbon_valid_mask=ribbon_valid,
        vertex_valid_mask=vertex_valid,
        vertex_roi_index=vertex_roi,
    )

    assert torch.equal(aligned_output["volume_tokens"], aligned_output["legacy_volume_tokens"])
    assert torch.equal(aligned_output["logits"], baseline_output["logits"])
    assert aligned_output["ribbon_anchor_tokens"].shape == (2, 4, 4)
    assert aligned_output["cross_geometry_surface"].shape == (2, 4, 12)
    assert aligned_output["cross_geometry_ribbon"].shape == (2, 4, 12)
    assert aligned_output["hybrid_ribbon_gate"] is None

    similarity = torch.matmul(
        aligned_output["cross_geometry_surface"],
        aligned_output["cross_geometry_ribbon"].transpose(-1, -2),
    )
    similarity.diagonal(dim1=-2, dim2=-1).mean().backward()
    assert aligned.cross_geometry_surface_projector[-1].weight.grad is not None
    assert aligned.cross_geometry_ribbon_projector[-1].weight.grad is not None
    assert any(parameter.grad is not None for parameter in aligned.surface_encoder.parameters())
    assert any(parameter.grad is not None for parameter in aligned.volume_encoder.parameters())


def test_cross_geometry_modules_preserve_same_seed_common_initialization():
    torch.manual_seed(41)
    baseline = _teacher("legacy_mask")
    torch.manual_seed(41)
    aligned = _teacher("legacy_mask", cross_geometry_enabled=True, cross_geometry_dim=12)

    baseline_state = baseline.state_dict()
    aligned_state = aligned.state_dict()
    common_names = [name for name in baseline_state if not name.startswith("cross_geometry")]
    assert common_names
    for name in common_names:
        assert torch.equal(baseline_state[name], aligned_state[name]), name


def test_roi_graph_preserves_same_seed_common_initialization():
    anatomy = torch.zeros(4, 4)
    anatomy[0, 1] = anatomy[1, 0] = 1.0
    anatomy[2, 3] = anatomy[3, 2] = 1.0
    boundary = anatomy / anatomy.sum(dim=1, keepdim=True).clamp_min(1.0)
    homology = torch.zeros_like(anatomy)
    homology[0, 2] = homology[2, 0] = 1.0
    homology[1, 3] = homology[3, 1] = 1.0

    torch.manual_seed(43)
    baseline = _teacher("legacy_mask")
    torch.manual_seed(43)
    graph_teacher = _teacher(
        "legacy_mask",
        roi_graph_enabled=True,
        roi_graph_num_heads=4,
        roi_graph_anatomy_adjacency=anatomy,
        roi_graph_boundary_strength=boundary,
        roi_graph_homology=homology,
    )

    baseline_state = baseline.state_dict()
    graph_state = graph_teacher.state_dict()
    common_names = [name for name in baseline_state if not name.startswith("roi_graph")]
    assert common_names
    for name in common_names:
        assert torch.equal(baseline_state[name], graph_state[name]), name


def test_roi_graph_off_preserves_legacy_forward_and_graph_on_is_diagnostic():
    image, surface, surface_masks, volume_masks = _inputs()
    anatomy = torch.zeros(4, 4)
    anatomy[0, 1] = anatomy[1, 0] = 1.0
    anatomy[2, 3] = anatomy[3, 2] = 1.0
    boundary = anatomy / anatomy.sum(dim=1, keepdim=True).clamp_min(1.0)
    homology = torch.zeros_like(anatomy)
    homology[0, 2] = homology[2, 0] = 1.0
    homology[1, 3] = homology[3, 1] = 1.0

    torch.manual_seed(47)
    legacy = _teacher("legacy_mask")
    torch.manual_seed(47)
    explicit_off = _teacher("legacy_mask", roi_graph_enabled=False)
    torch.manual_seed(47)
    graph_teacher = _teacher(
        "legacy_mask",
        roi_graph_enabled=True,
        roi_graph_num_heads=4,
        roi_graph_use_boundary_strength=True,
        roi_graph_use_homology=True,
        roi_graph_anatomy_adjacency=anatomy,
        roi_graph_boundary_strength=boundary,
        roi_graph_homology=homology,
    )
    legacy.eval()
    explicit_off.eval()
    graph_teacher.eval()

    legacy_output = legacy(image, surface, surface_masks, volume_masks, volume_masks)
    off_output = explicit_off(image, surface, surface_masks, volume_masks, volume_masks)
    graph_output = graph_teacher(image, surface, surface_masks, volume_masks, volume_masks)

    assert torch.equal(legacy_output["logits"], off_output["logits"])
    assert torch.equal(legacy_output["z_teacher"], off_output["z_teacher"])
    assert graph_output["z_teacher_graph"].shape == (2, 4, 16)
    assert graph_output["roi_graph_attention"].shape == (2, 4, 4, 4)
    assert float(graph_output["roi_graph_remote_mass"]) > 0.0
    assert float(graph_output["roi_graph_output_delta_l2"]) > 0.0
