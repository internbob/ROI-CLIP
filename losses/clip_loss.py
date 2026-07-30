import torch
import torch.nn.functional as F


def roi_clip_loss(z_surface, z_volume, temperature=0.07):
    surface_flat = z_surface.flatten(0, 1)
    volume_flat = z_volume.flatten(0, 1)
    logits = surface_flat @ volume_flat.transpose(0, 1) / max(float(temperature), 1e-6)
    target = torch.arange(logits.size(0), device=logits.device)
    loss_sv = F.cross_entropy(logits, target)
    loss_vs = F.cross_entropy(logits.transpose(0, 1), target)
    return 0.5 * (loss_sv + loss_vs)


def _flattened_logits(z_surface, z_volume, temperature):
    surface_flat = z_surface.flatten(0, 1)
    volume_flat = z_volume.flatten(0, 1)
    return surface_flat @ volume_flat.transpose(0, 1) / max(float(temperature), 1e-6)


def _block_diagonal_transport_target(batch_size, roi_transport, device, dtype):
    num_rois = int(roi_transport.size(0))
    target = torch.zeros(
        batch_size,
        num_rois,
        batch_size,
        num_rois,
        device=device,
        dtype=dtype,
    )
    subject_eye = torch.eye(batch_size, device=device, dtype=dtype).view(batch_size, 1, batch_size, 1)
    target = target + subject_eye * roi_transport.view(1, num_rois, 1, num_rois)
    return target.reshape(batch_size * num_rois, batch_size * num_rois)


def anatomical_transport_roi_clip_loss(
    z_surface,
    z_volume,
    roi_affinity,
    temperature=0.07,
    epsilon=0.1,
):
    """Soft ROI-CLIP where same-subject target mass follows an anatomical ROI graph."""
    if roi_affinity is None:
        return roi_clip_loss(z_surface, z_volume, temperature=temperature)
    batch_size, num_rois = z_surface.shape[:2]
    affinity = roi_affinity.to(device=z_surface.device, dtype=z_surface.dtype)
    affinity = affinity[:num_rois, :num_rois].clone()
    affinity.fill_diagonal_(0.0)
    affinity = affinity.clamp_min(0.0)
    affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    epsilon = min(max(float(epsilon), 0.0), 1.0)
    roi_transport = (1.0 - epsilon) * torch.eye(num_rois, device=z_surface.device, dtype=z_surface.dtype)
    roi_transport = roi_transport + epsilon * affinity
    roi_transport = roi_transport / roi_transport.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    logits = _flattened_logits(z_surface, z_volume, temperature)
    target_sv = _block_diagonal_transport_target(batch_size, roi_transport, logits.device, logits.dtype)
    loss_sv = -(target_sv * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss_vs = -(target_sv.transpose(0, 1) * F.log_softmax(logits.transpose(0, 1), dim=-1)).sum(dim=-1).mean()
    return 0.5 * (loss_sv + loss_vs)


def distance_weighted_roi_clip_loss(
    z_surface,
    z_volume,
    roi_affinity,
    temperature=0.07,
    min_negative_weight=0.05,
):
    """Hard-positive ROI-CLIP with weaker same-subject negatives for nearby ROIs."""
    if roi_affinity is None:
        return roi_clip_loss(z_surface, z_volume, temperature=temperature)
    batch_size, num_rois = z_surface.shape[:2]
    logits = _flattened_logits(z_surface, z_volume, temperature)
    affinity = roi_affinity.to(device=logits.device, dtype=logits.dtype)[:num_rois, :num_rois].clone()
    affinity.fill_diagonal_(0.0)
    same_subject_weight = (1.0 - affinity.clamp(0.0, 1.0)).clamp_min(float(min_negative_weight))
    same_subject_weight.fill_diagonal_(1.0)
    weights = torch.ones(
        batch_size,
        num_rois,
        batch_size,
        num_rois,
        device=logits.device,
        dtype=logits.dtype,
    )
    subject_eye = torch.eye(batch_size, device=logits.device, dtype=logits.dtype).view(batch_size, 1, batch_size, 1)
    weights = weights * (1.0 - subject_eye) + subject_eye * same_subject_weight.view(1, num_rois, 1, num_rois)
    weights = weights.reshape(batch_size * num_rois, batch_size * num_rois).clamp_min(1e-6)
    target = torch.arange(logits.size(0), device=logits.device)
    log_denom_sv = torch.logsumexp(logits + weights.log(), dim=-1)
    loss_sv = (log_denom_sv - logits[torch.arange(logits.size(0), device=logits.device), target]).mean()
    logits_t = logits.transpose(0, 1)
    log_denom_vs = torch.logsumexp(logits_t + weights.transpose(0, 1).log(), dim=-1)
    loss_vs = (log_denom_vs - logits_t[torch.arange(logits_t.size(0), device=logits.device), target]).mean()
    return 0.5 * (loss_sv + loss_vs)
