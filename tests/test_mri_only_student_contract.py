from types import SimpleNamespace

import torch

from train_student import build_student


def test_student_defaults_to_mri_only_when_teacher_uses_clinical_data():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            feature_dim=16,
            proj_dim=16,
            num_rois=4,
            num_classes=2,
            base_channels=4,
            dropout=0.0,
            roi_residual_init=0.1,
            volume_norm="batch",
            use_clinical=True,
            clinical_input_dim=6,
            clinical_fusion_mode="gated",
            clinical_hidden_dim=16,
            clinical_dropout=0.0,
            clinical_num_heads=4,
            clinical_num_layers=1,
        ),
        temperature=SimpleNamespace(prototype=0.1),
    )

    student = build_student(cfg)
    output = student(torch.randn(2, 1, 32, 32, 32))

    assert student.use_clinical is False
    assert output["logits"].shape == (2, 2)
