from types import SimpleNamespace

import torch

from config import load_configs
from options import Option
from train_student import shuffle_teacher_roi_outputs, student_clinical_input


def _cfg(enabled):
    return SimpleNamespace(training=SimpleNamespace(student_clinical_shuffle=enabled))


def test_student_clinical_shuffle_is_deterministic_derangement_for_batch():
    clinical = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    shuffled = student_clinical_input(clinical, _cfg(True))
    assert torch.equal(shuffled, torch.roll(clinical, shifts=1, dims=0))
    assert not torch.any(torch.all(shuffled == clinical, dim=1))


def test_student_clinical_shuffle_disabled_is_identity():
    clinical = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    assert student_clinical_input(clinical, _cfg(False)) is clinical


def test_student_clinical_shuffle_keeps_singleton_batch_unchanged():
    clinical = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    assert student_clinical_input(clinical, _cfg(True)) is clinical


def test_roi_shuffle_breaks_graph_state_correspondence_but_preserves_logits():
    torch.manual_seed(7)
    logits = torch.randn(3, 3)
    graph_states = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4)
    outputs = {
        "logits": logits,
        "z_teacher_graph": graph_states,
        "z_volume": graph_states + 1000.0,
    }

    shuffled = shuffle_teacher_roi_outputs(outputs)

    assert shuffled["logits"] is logits
    assert torch.equal(shuffled["logits"], logits)
    assert not torch.equal(shuffled["z_teacher_graph"], graph_states)
    # One shared ROI permutation must be applied consistently to both states.
    assert torch.equal(
        shuffled["z_teacher_graph"] + 1000.0,
        shuffled["z_volume"],
    )
    assert torch.equal(
        shuffled["z_teacher_graph"].sort(dim=1).values,
        graph_states.sort(dim=1).values,
    )


def test_clinical_two_level_configs_are_matched_and_development_only():
    base = [
        "configs/data.yaml",
        "configs/student.yaml",
        "configs/runtime_unified_clinical_common.yaml",
        "configs/runtime_unified_student_e4_full_kd.yaml",
        "configs/runtime_clinical_two_level_student_graph.yaml",
    ]
    d0 = load_configs(base + ["configs/runtime_clinical_two_level_d0_logit_only.yaml"])
    d1 = load_configs(base + ["configs/runtime_clinical_two_level_d1_roi100.yaml"])

    assert d0.data.clinical.features == d1.data.clinical.features
    assert len(d1.data.clinical.features) == 29
    assert "mmse" not in {str(name).lower() for name in d1.data.clinical.features}
    assert d1.model.use_clinical is True
    assert d1.model.student_use_clinical is True
    assert d1.model.student_roi_classifier.enabled is False
    assert d1.model.student_roi_graph.enabled is True
    assert d1.model.topology_query.enabled is True
    assert d0.loss_weights.vnext_kd_logit == d1.loss_weights.vnext_kd_logit == 0.5
    assert d0.loss_weights.vnext_kd_roi == 0.0
    assert d1.loss_weights.vnext_kd_roi == 1.0
    assert d0.runtime.evaluate_test is False
    assert d1.runtime.evaluate_test is False


def test_student_cli_does_not_silently_disable_yaml_roi_shuffle_control():
    option = Option(stage="student", default_paths=[])
    parser = option.initialize(option.parser)
    args = parser.parse_args([])
    cfg = load_configs(
        [
            "configs/data.yaml",
            "configs/student.yaml",
            "configs/runtime_clinical_two_level_d3_roi_shuffle.yaml",
        ]
    )
    cfg = option._apply_overrides(cfg, args)
    assert cfg.training.control_shuffle_roi is True
