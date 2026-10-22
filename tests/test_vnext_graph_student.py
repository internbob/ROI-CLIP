from types import SimpleNamespace

import torch

from config import load_configs
from models.student import MRIOnlyStudent
from options import Option
from train_student import build_topology_query_pe, distillation_enabled


def _ring_topology(num_rois=6):
    adjacency = torch.zeros(num_rois, num_rois)
    for index in range(num_rois):
        adjacency[index, (index - 1) % num_rois] = 1.0
        adjacency[index, (index + 1) % num_rois] = 1.0
    homology = torch.zeros_like(adjacency)
    half = num_rois // 2
    for index in range(half):
        homology[index, index + half] = 1.0
        homology[index + half, index] = 1.0
    boundary = adjacency / adjacency.sum(dim=1, keepdim=True)
    return {
        "anatomy_adjacency": adjacency,
        "boundary_strength": boundary,
        "evaluation_boundary_strength": boundary,
        "homology": homology,
        "evaluation_homology": homology,
    }


def test_topology_query_pe_is_deterministic_and_nonconstant():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            topology_query=SimpleNamespace(enabled=True, components=3, include_homology=True)
        )
    )
    topology = _ring_topology()
    first = build_topology_query_pe(cfg, topology)
    second = build_topology_query_pe(cfg, topology)
    assert first.shape == (6, 3)
    assert torch.equal(first, second)
    assert torch.all(first.std(dim=0) > 0)


def test_vnext_scratch_ignores_legacy_cli_kd_defaults():
    cfg = SimpleNamespace(
        training=SimpleNamespace(vnext_graph_student=True),
        loss_weights=SimpleNamespace(
            kd_logit=0.25,
            kd_token=0.1,
            kd_relation=0.01,
            kd_prototype=0.01,
            vnext_kd_logit=0.0,
            vnext_kd_roi=0.0,
            vnext_kd_graph=0.0,
            vnext_kd_prototype=0.0,
        ),
    )
    assert distillation_enabled(cfg) is False


def test_student_cli_defaults_do_not_override_yaml_schedule():
    option = Option(stage="student", default_paths=[])
    parser = option.initialize(option.parser)
    args = parser.parse_args([])
    cfg = load_configs(
        [
            "configs/data.yaml",
            "configs/student.yaml",
            "configs/runtime_vnext_graph_student_s0_scratch.yaml",
        ]
    )
    cfg = option._apply_overrides(cfg, args)
    assert cfg.training.warmup_epochs == 0
    assert cfg.training.encoder_lr == 1e-4
    assert cfg.training.inherited_lr == 1e-4


def test_graph_student_forward_requires_only_mri():
    topology = _ring_topology()
    model = MRIOnlyStudent(
        feature_dim=16,
        proj_dim=16,
        num_rois=6,
        num_classes=3,
        base_channels=4,
        dropout=0.0,
        volume_norm="group",
        topology_query_pe=torch.randn(6, 3),
        roi_graph_enabled=True,
        roi_graph_num_heads=4,
        roi_graph_anatomy_adjacency=topology["anatomy_adjacency"],
        roi_graph_boundary_strength=topology["boundary_strength"],
        roi_graph_evaluation_boundary_strength=topology["evaluation_boundary_strength"],
        roi_graph_homology=topology["homology"],
        roi_graph_evaluation_homology=topology["evaluation_homology"],
        roi_graph_use_boundary_strength=True,
        roi_graph_use_homology=True,
    ).eval()
    image = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        outputs = model(image)
        outputs_with_unused_privileged_field = model(
            image,
            clinical_data=torch.randn(2, 11),
        )
    assert outputs["logits"].shape == (2, 3)
    assert torch.equal(outputs["logits"], outputs_with_unused_privileged_field["logits"])
    assert outputs["z_student_base"].shape == (2, 6, 16)
    assert outputs["z_student_graph"].shape == (2, 6, 16)
    assert outputs["roi_graph_attention"] is not None
    assert model.topology_query_pe.shape == (1, 6, 3)
    assert "topology_query_pe" in dict(model.named_buffers())
    assert model.use_clinical is False
    assert not any(
        token in name.lower()
        for name in model.state_dict()
        for token in ("surface", "ribbon", "clinical", "pet")
    )


def test_roi_bottleneck_switches_preserve_global_path_and_separate_raw_tokens():
    topology = _ring_topology()
    common = dict(
        feature_dim=16,
        proj_dim=16,
        num_rois=6,
        num_classes=3,
        base_channels=4,
        dropout=0.0,
        volume_norm="group",
        topology_query_pe=torch.randn(6, 3),
        roi_graph_enabled=True,
        roi_graph_num_heads=4,
        roi_graph_anatomy_adjacency=topology["anatomy_adjacency"],
        roi_graph_boundary_strength=topology["boundary_strength"],
        roi_graph_evaluation_boundary_strength=topology["evaluation_boundary_strength"],
        roi_graph_homology=topology["homology"],
        roi_graph_evaluation_homology=topology["evaluation_homology"],
    )
    historical = MRIOnlyStudent(**common).eval()
    global_only = MRIOnlyStudent(
        **common,
        roi_classification_enabled=False,
        roi_classifier_use_raw_tokens=True,
    ).eval()
    global_only.load_state_dict(historical.state_dict())
    image = torch.randn(2, 1, 32, 32, 32)

    with torch.no_grad():
        old = historical(image)
        new = global_only(image)

    assert torch.allclose(old["logits"], old["global_logits"] + old["roi_weight"] * old["roi_logits"])
    assert torch.equal(new["logits"], new["global_logits"])
    assert torch.allclose(new["z_student_graph"].norm(dim=-1), torch.ones(2, 6), atol=1e-5)
    assert not torch.allclose(
        new["z_student_graph_raw"].norm(dim=-1),
        torch.ones(2, 6),
        atol=1e-3,
    )


def test_roi_bottleneck_configs_form_the_declared_single_factor_sequence():
    base = [
        "configs/data.yaml",
        "configs/student.yaml",
        "configs/runtime_vnext_graph_student_locked_s4.yaml",
    ]
    names = (
        "runtime_roi_bottleneck_b0_global_ce.yaml",
        "runtime_roi_bottleneck_b1_global_logit_kd.yaml",
        "runtime_roi_bottleneck_b2_global_raw_roi.yaml",
        "runtime_roi_bottleneck_b3_global_raw_roi_graph.yaml",
        "runtime_roi_bottleneck_b4_global_raw_roi_graph_kd.yaml",
    )
    configs = [load_configs(base + [f"configs/{name}"]) for name in names]
    observed = [
        (
            cfg.model.student_roi_classifier.enabled,
            cfg.model.student_roi_graph.enabled,
            cfg.loss_weights.vnext_kd_logit,
            cfg.loss_weights.vnext_kd_roi,
        )
        for cfg in configs
    ]
    assert observed == [
        (False, False, 0.0, 0.0),
        (False, False, 0.5, 0.0),
        (True, False, 0.5, 0.0),
        (True, True, 0.5, 0.0),
        (True, True, 0.5, 1.0),
    ]
    assert all(cfg.model.student_roi_classifier.use_raw_tokens for cfg in configs)
    assert all(cfg.runtime.evaluate_test is False for cfg in configs)
    assert all(cfg.model.roi_graph.enabled is True for cfg in configs)


def test_graph_toggle_preserves_all_common_parameter_initialization():
    topology = _ring_topology()
    common = dict(
        feature_dim=16,
        proj_dim=16,
        num_rois=6,
        num_classes=3,
        base_channels=4,
        dropout=0.0,
        volume_norm="group",
        topology_query_pe=torch.randn(6, 3),
        roi_graph_num_heads=4,
        roi_graph_anatomy_adjacency=topology["anatomy_adjacency"],
        roi_graph_boundary_strength=topology["boundary_strength"],
        roi_graph_evaluation_boundary_strength=topology["evaluation_boundary_strength"],
        roi_graph_homology=topology["homology"],
        roi_graph_evaluation_homology=topology["evaluation_homology"],
    )
    torch.manual_seed(41)
    graph_off = MRIOnlyStudent(**common, roi_graph_enabled=False)
    torch.manual_seed(41)
    graph_on = MRIOnlyStudent(**common, roi_graph_enabled=True)

    graph_on_state = graph_on.state_dict()
    for name, value in graph_off.state_dict().items():
        if name == "topology_query_pe":
            continue
        assert torch.equal(value, graph_on_state[name]), name


def test_progressive_component_configs_change_only_declared_kd_weights():
    names = ("s3_logit_kd", "s4_roi_kd", "s5_graph_kd", "s6_full_kd")
    expected = (
        (0.5, 0.0, 0.0, 0.0),
        (0.5, 1.0, 0.0, 0.0),
        (0.5, 1.0, 0.2, 0.0),
        (0.5, 1.0, 0.2, 0.2),
    )
    configs = [
        load_configs(
            [
                "configs/data.yaml",
                "configs/student.yaml",
                f"configs/runtime_vnext_graph_student_{name}.yaml",
            ]
        )
        for name in names
    ]
    for cfg, weights in zip(configs, expected):
        actual = (
            cfg.loss_weights.vnext_kd_logit,
            cfg.loss_weights.vnext_kd_roi,
            cfg.loss_weights.vnext_kd_graph,
            cfg.loss_weights.vnext_kd_prototype,
        )
        assert actual == weights
        assert cfg.training.warmup_epochs == 0
        assert cfg.training.encoder_lr == 1e-4
        assert cfg.training.inherited_lr == 1e-4
        assert cfg.training.initialize_from_teacher is False
        assert cfg.training.match_teacher_available is True
        assert cfg.runtime.evaluate_test is False
        assert cfg.model.student_use_clinical is False


def test_locked_s4_config_matches_selected_component_protocol():
    paths = ["configs/data.yaml", "configs/student.yaml"]
    selected = load_configs(paths + ["configs/runtime_vnext_graph_student_s4_roi_kd.yaml"])
    locked = load_configs(paths + ["configs/runtime_vnext_graph_student_locked_s4.yaml"])

    assert locked.loss_weights.vnext_kd_logit == selected.loss_weights.vnext_kd_logit == 0.5
    assert locked.loss_weights.vnext_kd_roi == selected.loss_weights.vnext_kd_roi == 1.0
    assert locked.loss_weights.vnext_kd_graph == selected.loss_weights.vnext_kd_graph == 0.0
    assert locked.loss_weights.vnext_kd_prototype == selected.loss_weights.vnext_kd_prototype == 0.0
    assert locked.training.warmup_epochs == selected.training.warmup_epochs == 0
    assert locked.training.encoder_lr == selected.training.encoder_lr == 1e-4
    assert locked.training.inherited_lr == selected.training.inherited_lr == 1e-4
    assert locked.training.initialize_from_teacher is selected.training.initialize_from_teacher is False
    assert locked.model.student_use_clinical is selected.model.student_use_clinical is False
    assert locked.model.volume_pooling.mode == selected.model.volume_pooling.mode == "legacy_mask"
    assert locked.model.topology_query.enabled is selected.model.topology_query.enabled is True
    assert locked.model.roi_graph.enabled is selected.model.roi_graph.enabled is True
    assert locked.model.roi_graph.topology_path == selected.model.roi_graph.topology_path
    assert locked.runtime.evaluate_test is selected.runtime.evaluate_test is False


def test_s4c_configs_extend_locked_s4_without_mmse_or_teacher_clinical_input():
    base_paths = [
        "configs/data.yaml",
        "configs/student.yaml",
        "configs/runtime_vnext_graph_student_locked_s4.yaml",
    ]
    variants = {
        "safe_gated": 4,
        "safe_neuro_gated": 17,
        "cognition_gated": 13,
        "full29_no_mmse_gated": 29,
    }
    for suffix, expected_features in variants.items():
        cfg = load_configs(
            base_paths + [f"configs/runtime_vnext_graph_student_s4c_{suffix}.yaml"]
        )
        features = [str(value) for value in cfg.data.clinical.features]
        assert cfg.data.clinical.enabled is True
        assert cfg.model.use_clinical is False
        assert cfg.model.student_use_clinical is True
        assert cfg.model.clinical_fusion_mode == "gated"
        assert cfg.runtime.evaluate_test is False
        assert len(features) == expected_features
        assert not any("mmse" in feature.lower() for feature in features)
        assert cfg.loss_weights.vnext_kd_logit == 0.5
        assert cfg.loss_weights.vnext_kd_roi == 1.0
        assert cfg.loss_weights.vnext_kd_graph == 0.0
        assert cfg.loss_weights.vnext_kd_prototype == 0.0
        assert cfg.training.warmup_epochs == 0
        assert cfg.training.encoder_lr == 1e-4
        assert cfg.training.inherited_lr == 1e-4


def test_locked_s4c_matches_selected_safe_neuro_overlay():
    shared = ["configs/data.yaml", "configs/student.yaml"]
    composed = load_configs(
        shared
        + [
            "configs/runtime_vnext_graph_student_locked_s4.yaml",
            "configs/runtime_vnext_graph_student_s4c_safe_neuro_gated.yaml",
        ]
    )
    def select(cfg, path):
        value = cfg
        for key in path.split("."):
            value = value[key]
        return value

    locked = load_configs(
        shared + ["configs/runtime_vnext_graph_student_locked_s4c_safe_neuro.yaml"]
    )
    paths = (
        "data.clinical.enabled",
        "data.clinical.features",
        "model.use_clinical",
        "model.student_use_clinical",
        "model.clinical_fusion_mode",
        "model.volume_pooling.mode",
        "model.cross_geometry.lambda_cg",
        "model.topology_query.enabled",
        "model.roi_graph.topology_path",
        "model.roi_graph.use_boundary_strength",
        "model.roi_graph.use_homology",
        "loss_weights.vnext_kd_logit",
        "loss_weights.vnext_kd_roi",
        "loss_weights.vnext_kd_graph",
        "loss_weights.vnext_kd_prototype",
        "training.warmup_epochs",
        "training.encoder_lr",
        "training.inherited_lr",
        "runtime.evaluate_test",
    )
    for path in paths:
        assert select(locked, path) == select(composed, path)
    assert not any("mmse" in str(value).lower() for value in locked.data.clinical.features)
