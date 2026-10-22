from types import SimpleNamespace

import torch
import torch.nn.functional as F

from data.dataset import CortexWeaveROIClipDataset
from models.student import MRIOnlyStudent
from models.teacher import CortexWeaveROIClipTeacher
from train_teacher import auxiliary_loss_scale_for_epoch, compute_class_weights
from train_student import (
    build_student_optimizer,
    initialize_student_from_teacher,
    set_inherited_trainable,
)


def build_models():
    common = {
        "feature_dim": 16,
        "proj_dim": 16,
        "num_rois": 4,
        "num_classes": 2,
        "base_channels": 4,
        "dropout": 0.0,
        "prototype_temperature": 0.1,
        "roi_residual_init": 0.1,
    }
    teacher = CortexWeaveROIClipTeacher(
        surface_in_channels=3,
        surface_hidden_dim=8,
        num_vertices=84,
        **common,
    )
    student = MRIOnlyStudent(**common)
    initialize_student_from_teacher(student, teacher)
    return teacher, student


def test_global_branch_inheritance_and_forward():
    torch.manual_seed(7)
    teacher, student = build_models()
    teacher.eval()
    student.eval()

    image = torch.randn(2, 1, 32, 32, 32)
    surface = torch.randn(2, 84, 3)
    surface_masks = torch.zeros(2, 4, 84)
    for roi in range(4):
        surface_masks[:, roi, roi * 21 : (roi + 1) * 21] = 1.0
    volume_masks = torch.rand(2, 4, 4, 4, 4)

    with torch.no_grad():
        teacher_out = teacher(image, surface, surface_masks, volume_masks, volume_masks)
        student_out = student(image)

    assert teacher_out["logits"].shape == (2, 2)
    assert student_out["logits"].shape == (2, 2)
    assert student_out["z_student"].shape == (2, 4, 16)
    assert student_out["multi_scale_features"][2].shape == (2, 16, 4, 4, 4)
    assert student_out["multi_scale_features"][-1].shape == (2, 8, 1, 1, 1)
    assert sum(isinstance(module, torch.nn.BatchNorm3d) for module in student.volume_encoder.modules()) == 6
    assert not any(isinstance(module, torch.nn.GroupNorm) for module in student.volume_encoder.modules())
    assert torch.allclose(teacher_out["global_logits"], student_out["global_logits"], atol=1e-6)
    assert torch.allclose(teacher_out["z_volume"].norm(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_atlas_guided_surface_refiner_is_optional_and_shape_preserving():
    torch.manual_seed(17)
    teacher = CortexWeaveROIClipTeacher(
        surface_in_channels=3,
        surface_hidden_dim=8,
        num_vertices=84,
        feature_dim=16,
        proj_dim=16,
        num_rois=4,
        num_classes=2,
        base_channels=4,
        dropout=0.0,
        surface_refiner="agsr",
        surface_refiner_hidden_dim=12,
        surface_refiner_dropout=0.0,
        surface_refiner_residual_init=0.2,
    )
    image = torch.randn(2, 1, 32, 32, 32)
    surface = torch.randn(2, 84, 3)
    surface_masks = torch.zeros(2, 4, 84)
    for roi in range(4):
        surface_masks[:, roi, roi * 21 : (roi + 1) * 21] = 1.0
    volume_masks = torch.rand(2, 4, 4, 4, 4)

    output = teacher(image, surface, surface_masks, volume_masks, volume_masks)
    loss = output["logits"].sum()
    loss.backward()

    assert output["surface_tokens"].shape == (2, 4, 16)
    assert teacher.surface_refiner.roi_embedding.grad is not None


def test_warmup_only_updates_new_student_heads():
    torch.manual_seed(11)
    _, student = build_models()
    set_inherited_trainable(student, trainable=False)
    cfg = SimpleNamespace(
        training=SimpleNamespace(encoder_lr=1e-5, inherited_lr=1e-5, lr=1e-4, weight_decay=1e-5)
    )
    optimizer = build_student_optimizer(student, cfg)

    output = student(torch.randn(2, 1, 32, 32, 32))
    loss = F.cross_entropy(output["logits"], torch.tensor([0, 1]))
    optimizer.zero_grad()
    loss.backward()

    assert all(param.grad is None for param in student.volume_encoder.parameters())
    assert any(param.grad is not None for param in student.to_q.parameters())


def test_mri_only_teacher_skips_roi_branch():
    torch.manual_seed(13)
    teacher = CortexWeaveROIClipTeacher(
        surface_in_channels=3,
        surface_hidden_dim=8,
        num_vertices=84,
        feature_dim=16,
        proj_dim=16,
        num_rois=4,
        num_classes=3,
        base_channels=4,
        dropout=0.0,
        use_roi_branch=False,
    )
    output = teacher(torch.randn(2, 1, 32, 32, 32), None, None, None, None)
    loss = F.cross_entropy(output["logits"], torch.tensor([0, 2]))
    loss.backward()

    assert output["logits"].shape == (2, 3)
    assert output["roi_weight"].item() == 0.0
    assert all(param.grad is None for param in teacher.surface_encoder.parameters())
    assert any(param.grad is not None for param in teacher.volume_encoder.parameters())
    assert any(param.grad is not None for param in teacher.global_head.parameters())


def test_sqrt_balanced_weights_are_milder_than_balanced_weights():
    dataset = SimpleNamespace(samples=[{"label": 0}] * 9 + [{"label": 1}] * 4 + [{"label": 2}])
    balanced = compute_class_weights(dataset, num_classes=3, mode="balanced")
    sqrt_balanced = compute_class_weights(dataset, num_classes=3, mode="sqrt_balanced")

    assert abs(balanced[2] / balanced[0] - 9.0) < 1e-6
    assert abs(sqrt_balanced[2] / sqrt_balanced[0] - 3.0) < 1e-6


def test_teacher_auxiliary_losses_start_after_five_ce_epochs():
    assert [auxiliary_loss_scale_for_epoch(epoch, 5) for epoch in range(7)] == [0.0] * 5 + [1.0, 1.0]


def test_clinical_preprocessing_uses_values_and_missing_indicators():
    dataset = CortexWeaveROIClipDataset.__new__(CortexWeaveROIClipDataset)
    dataset.clinical_features = ("first", "second")
    dataset.samples = [
        {"clinical_values": torch.tensor([1.0, float("nan")]).numpy()},
        {"clinical_values": torch.tensor([3.0, 4.0]).numpy()},
    ]
    stats = dataset.fit_clinical_stats()
    dataset.clinical_mean = stats["mean"]
    dataset.clinical_std = stats["std"]

    encoded = dataset._encode_clinical(dataset.samples[0])

    assert torch.allclose(torch.from_numpy(stats["mean"]), torch.tensor([2.0, 4.0]))
    assert torch.allclose(torch.from_numpy(encoded), torch.tensor([-1.0, 0.0, 0.0, 1.0]))


def test_student_inherits_clinical_cross_transformer_from_teacher():
    common = {
        "feature_dim": 16,
        "proj_dim": 16,
        "num_rois": 4,
        "num_classes": 3,
        "base_channels": 4,
        "dropout": 0.0,
        "prototype_temperature": 0.1,
        "roi_residual_init": 0.1,
        "use_clinical": True,
        "clinical_input_dim": 6,
        "clinical_fusion_mode": "cross",
        "clinical_num_heads": 4,
        "clinical_num_layers": 2,
    }
    teacher = CortexWeaveROIClipTeacher(
        surface_in_channels=3,
        surface_hidden_dim=8,
        num_vertices=84,
        **common,
    )
    student = MRIOnlyStudent(**common)
    initialize_student_from_teacher(student, teacher)
    teacher.eval()
    student.eval()

    image = torch.randn(2, 1, 32, 32, 32)
    clinical_data = torch.randn(2, 6)
    surface = torch.randn(2, 84, 3)
    surface_masks = torch.zeros(2, 4, 84)
    volume_masks = torch.rand(2, 4, 4, 4, 4)
    with torch.no_grad():
        teacher_output = teacher(
            image,
            surface,
            surface_masks,
            volume_masks,
            volume_masks,
            clinical_data=clinical_data,
        )
        student_output = student(image, clinical_data=clinical_data)

    for name, teacher_value in teacher.clinical_cross_fusion.state_dict().items():
        assert torch.allclose(teacher_value, student.clinical_cross_fusion.state_dict()[name])
    assert teacher_output["image_to_clinical_attention"].shape == (2, 5, 3)
    assert teacher_output["clinical_to_image_attention"].shape == (2, 3, 5)
    assert student_output["image_to_clinical_attention"].shape == (2, 5, 3)


def test_student_inherits_gated_clinical_fusion_from_teacher():
    common = {
        "feature_dim": 16,
        "proj_dim": 16,
        "num_rois": 4,
        "num_classes": 3,
        "base_channels": 4,
        "dropout": 0.0,
        "prototype_temperature": 0.1,
        "roi_residual_init": 0.1,
        "use_clinical": True,
        "clinical_input_dim": 6,
        "clinical_fusion_mode": "gated",
        "clinical_hidden_dim": 8,
        "clinical_dropout": 0.0,
    }
    teacher = CortexWeaveROIClipTeacher(
        surface_in_channels=3,
        surface_hidden_dim=8,
        num_vertices=84,
        **common,
    )
    student = MRIOnlyStudent(**common)
    initialize_student_from_teacher(student, teacher)
    teacher.eval()
    student.eval()

    image = torch.randn(2, 1, 32, 32, 32)
    clinical_data = torch.randn(2, 6)
    surface = torch.randn(2, 84, 3)
    surface_masks = torch.zeros(2, 4, 84)
    volume_masks = torch.rand(2, 4, 4, 4, 4)
    with torch.no_grad():
        teacher_output = teacher(
            image,
            surface,
            surface_masks,
            volume_masks,
            volume_masks,
            clinical_data=clinical_data,
        )
        student_output = student(image, clinical_data=clinical_data)

    assert torch.allclose(teacher_output["global_logits"], student_output["global_logits"], atol=1e-6)
    assert teacher_output["clinical_gate"].shape == (2, 16)
    assert teacher_output["cross_image_tokens"] is None
