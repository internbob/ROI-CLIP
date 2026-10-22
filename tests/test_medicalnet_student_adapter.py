from pathlib import Path

import torch

from config import load_configs
from models.teacher import CortexWeaveROIClipTeacher
from models.volume_encoder import MedicalNetResNet10Encoder3D
from train_student import build_student, build_teacher


def test_medicalnet_resnet10_adapter_preserves_cortexbridge_shapes():
    encoder = MedicalNetResNet10Encoder3D(out_dim=16).eval()
    with torch.no_grad():
        roi_map, multi_scale, global_feature = encoder(torch.randn(2, 1, 32, 40, 32))
    assert roi_map.shape == (2, 16, 4, 5, 4)
    assert global_feature.shape == (2, 16)
    assert len(multi_scale) == 5
    assert multi_scale[0].shape[-3:] == (16, 20, 16)


def test_medicalnet_checkpoint_loader_accepts_module_prefix(tmp_path):
    source = MedicalNetResNet10Encoder3D(out_dim=16)
    backbone_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
    state = {
        f"module.{key}": value.clone()
        for key, value in source.state_dict().items()
        if key.startswith(backbone_prefixes)
    }
    checkpoint_path = tmp_path / "medicalnet.pth"
    torch.save({"state_dict": state}, checkpoint_path)
    loaded = MedicalNetResNet10Encoder3D(out_dim=16, pretrained_path=str(checkpoint_path))
    assert loaded.pretrained_report["loaded"] is True
    assert loaded.pretrained_report["loaded_tensors"] == len(state)
    assert torch.equal(loaded.conv1.weight, source.conv1.weight)


def test_med3d_student_overlay_does_not_replace_locked_teacher():
    root = Path(__file__).resolve().parents[1]
    cfg = load_configs(
        [
            root / "configs/data.yaml",
            root / "configs/teacher.yaml",
            root / "configs/student.yaml",
            root / "configs/runtime_vnext_graph_student_locked_s4.yaml",
            root / "configs/runtime_med3d_m4_resnet10_pretrained_roi_kd.yaml",
        ]
    )
    # Avoid loading the real weight in this construction-only contract check.
    cfg.model.student_backbone.pretrained_path = ""
    student = build_student(cfg)
    teacher = build_teacher(cfg)
    assert isinstance(student.volume_encoder, MedicalNetResNet10Encoder3D)
    assert isinstance(teacher, CortexWeaveROIClipTeacher)
    assert teacher.volume_encoder.__class__.__name__ == "SFCNVolumeEncoder3D"


def test_med3d_m1_to_m4_objective_matrix():
    root = Path(__file__).resolve().parents[1]
    names = (
        "runtime_med3d_m1_resnet10_scratch_ce.yaml",
        "runtime_med3d_m2_resnet10_pretrained_ce.yaml",
        "runtime_med3d_m3_resnet10_pretrained_logit_kd.yaml",
        "runtime_med3d_m4_resnet10_pretrained_roi_kd.yaml",
    )
    expected = ((0.0, 0.0), (0.0, 0.0), (0.5, 0.0), (0.5, 1.0))
    for name, weights in zip(names, expected):
        cfg = load_configs(
            [
                root / "configs/data.yaml",
                root / "configs/student.yaml",
                root / "configs/runtime_vnext_graph_student_locked_s4.yaml",
                root / f"configs/{name}",
            ]
        )
        assert cfg.model.student_backbone.name == "medicalnet_resnet10"
        assert cfg.runtime.evaluate_test is False
        assert cfg.training.initialize_from_teacher is False
        assert (cfg.loss_weights.vnext_kd_logit, cfg.loss_weights.vnext_kd_roi) == weights
        if "m1_" in name:
            assert cfg.model.student_backbone.pretrained_path == ""
            assert cfg.training.encoder_lr == 1e-4
        else:
            assert cfg.model.student_backbone.pretrained_path.endswith("resnet_10_23dataset.pth")
            assert cfg.training.encoder_lr == 1e-5
