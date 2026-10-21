import torch
import torch.nn.functional as F


def _flatten_spatial(tensor):
    if tensor.ndim < 4:
        raise ValueError(f"Expected [B,K,...] spatial tensor, got {tuple(tensor.shape)}")
    return tensor.flatten(2)


def _valid_rois(target_masks):
    return _flatten_spatial(target_masks).sum(dim=-1) > 0


def _masked_mean(values, valid):
    valid = valid.to(device=values.device, dtype=values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def normalize_spatial_distribution(spatial, eps=1e-8):
    flat = _flatten_spatial(spatial).clamp_min(0.0)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return flat.reshape_as(spatial)


def roi_attention_mass_loss(student_attention, target_masks, eps=1e-6):
    """Encourage every ROI query to place probability mass inside its atlas ROI."""
    student = normalize_spatial_distribution(student_attention, eps=eps).flatten(2)
    target = (_flatten_spatial(target_masks) > 0).to(student.dtype)
    valid = target.sum(dim=-1) > 0
    mass = (student * target).sum(dim=-1).clamp(min=float(eps), max=1.0)
    return _masked_mean(-mass.log(), valid)


def roi_occupancy_loss(spatial_logits, target_masks, dice_weight=0.5, eps=1e-6):
    """Balanced voxelwise occupancy loss plus soft Dice for small cortical ROIs."""
    logits = _flatten_spatial(spatial_logits)
    target = (_flatten_spatial(target_masks) > 0).to(logits.dtype)
    valid = target.sum(dim=-1) > 0

    positive = target.sum(dim=-1).clamp_min(1.0)
    negative = (1.0 - target).sum(dim=-1).clamp_min(1.0)
    positive_bce = (F.softplus(-logits) * target).sum(dim=-1) / positive
    negative_bce = (F.softplus(logits) * (1.0 - target)).sum(dim=-1) / negative
    balanced_bce = 0.5 * (positive_bce + negative_bce)

    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=-1)
    dice = (2.0 * intersection + float(eps)) / (
        probability.sum(dim=-1) + target.sum(dim=-1) + float(eps)
    )
    combined = balanced_bce + float(dice_weight) * (1.0 - dice)
    return _masked_mean(combined, valid)


def spatial_attention_consistency_loss(student_attention, teacher_attention, eps=1e-6):
    """Symmetric Jensen-Shannon consistency between teacher and student ROI maps."""
    student = normalize_spatial_distribution(student_attention, eps=eps).flatten(2)
    teacher = normalize_spatial_distribution(teacher_attention.detach(), eps=eps).flatten(2)
    valid = _flatten_spatial(teacher_attention).sum(dim=-1) > 0
    student = student.clamp_min(float(eps))
    teacher = teacher.clamp_min(float(eps))
    mixture = 0.5 * (student + teacher)
    js = 0.5 * (
        (student * (student.log() - mixture.log())).sum(dim=-1)
        + (teacher * (teacher.log() - mixture.log())).sum(dim=-1)
    )
    return _masked_mean(js, valid)


def _coordinate_grid(shape, device, dtype):
    axes = [torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype) for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(shape))


def spatial_centers(spatial, eps=1e-8):
    flat = _flatten_spatial(spatial).clamp_min(0.0)
    weights = flat / flat.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    grid = _coordinate_grid(spatial.shape[2:], spatial.device, spatial.dtype)
    return weights @ grid


def roi_center_correspondence_loss(student_attention, target_masks):
    """Match ROI-query centers to affine-aligned atlas ROI centers in normalized coordinates."""
    valid = _valid_rois(target_masks)
    student_center = spatial_centers(student_attention)
    target_center = spatial_centers(target_masks.to(student_attention.dtype)).detach()
    distance = F.smooth_l1_loss(student_center, target_center, reduction="none").mean(dim=-1)
    return _masked_mean(distance, valid)


def roi_spatial_metrics(student_attention, spatial_logits, target_masks, eps=1e-6):
    """Detached interpretable metrics used for validation/test reporting."""
    with torch.no_grad():
        student = normalize_spatial_distribution(student_attention, eps=eps).flatten(2)
        target = (_flatten_spatial(target_masks) > 0).to(student.dtype)
        valid = target.sum(dim=-1) > 0
        mass = (student * target).sum(dim=-1)

        probability = torch.sigmoid(_flatten_spatial(spatial_logits))
        intersection = (probability * target).sum(dim=-1)
        dice = (2.0 * intersection + float(eps)) / (
            probability.sum(dim=-1) + target.sum(dim=-1) + float(eps)
        )

        student_center = spatial_centers(student_attention)
        target_center = spatial_centers(target_masks.to(student_attention.dtype))
        center_distance = torch.linalg.vector_norm(student_center - target_center, dim=-1)
        return {
            "roi_attention_mass": float(_masked_mean(mass, valid)),
            "roi_soft_dice": float(_masked_mean(dice, valid)),
            "roi_center_distance": float(_masked_mean(center_distance, valid)),
        }
