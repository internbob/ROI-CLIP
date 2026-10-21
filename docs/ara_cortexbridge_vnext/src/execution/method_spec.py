"""Typed research specification; this is not a production model implementation."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def hard_cross_geometry_loss(
    surface_tokens: torch.Tensor,
    ribbon_tokens: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Subject-local symmetric same-ROI loss for tensors shaped [B, K, D]."""
    surface = F.normalize(surface_tokens, dim=-1)
    ribbon = F.normalize(ribbon_tokens, dim=-1)
    similarity = surface @ ribbon.transpose(-1, -2) / max(float(temperature), 1e-6)
    batch, rois, _ = similarity.shape
    target = torch.arange(rois, device=similarity.device).repeat(batch)
    loss_sr = F.cross_entropy(similarity.reshape(batch * rois, rois), target)
    loss_rs = F.cross_entropy(similarity.transpose(-1, -2).reshape(batch * rois, rois), target)
    return 0.5 * (loss_sr + loss_rs)


def topology_biased_attention(
    tokens: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    boundary_strength: torch.Tensor,
    homology: torch.Tensor,
    boundary_scale: torch.Tensor,
    homology_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Planned graph core: topology is additive bias and remote edges remain possible."""
    query = tokens @ q_weight
    key = tokens @ k_weight
    value = tokens @ v_weight
    scale = query.shape[-1] ** -0.5
    logits = (query @ key.transpose(-1, -2)) * scale
    boundary_bias = torch.log(boundary_strength.clamp_min(1e-6))
    logits = logits + boundary_scale * boundary_bias + homology_scale * homology
    attention = torch.softmax(logits, dim=-1)
    return attention @ value, attention


def graph_state_distillation(
    teacher: Dict[str, torch.Tensor],
    student: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Planned MRI-only KD targets using detached Teacher graph outputs."""
    teacher_tokens = F.normalize(teacher["z_teacher_graph"].detach(), dim=-1)
    student_tokens = F.normalize(student["z_student_graph"], dim=-1)
    roi_loss = (1.0 - (teacher_tokens * student_tokens).sum(dim=-1)).mean()

    teacher_graph = teacher["teacher_graph"].detach().clamp_min(1e-6)
    student_graph = student["student_graph"].clamp_min(1e-6)
    graph_loss = F.kl_div(student_graph.log(), teacher_graph, reduction="batchmean")

    teacher_proto = F.softmax(teacher["prototype_logits"].detach(), dim=-1)
    student_proto = F.log_softmax(student["prototype_logits"], dim=-1)
    prototype_loss = F.kl_div(student_proto, teacher_proto, reduction="batchmean")
    return {"roi": roi_loss, "graph": graph_loss, "prototype": prototype_loss}

