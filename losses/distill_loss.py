import torch
import torch.nn.functional as F


def _mean_distribution_kl(student_log_prob, teacher_prob):
    """Average KL per distribution, independent of batch or ROI count."""
    per_distribution = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="none",
    ).sum(dim=-1)
    return per_distribution.mean()


def kd_logit_loss(teacher_logits, student_logits, temperature=4.0):
    tau = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits.detach() / tau, dim=-1)
    student_log_prob = F.log_softmax(student_logits / tau, dim=-1)
    return _mean_distribution_kl(student_log_prob, teacher_prob) * (tau ** 2)


def token_distill_loss(teacher_tokens, student_tokens):
    teacher = F.normalize(teacher_tokens.detach(), dim=-1)
    student = F.normalize(student_tokens, dim=-1)
    return (1.0 - (teacher * student).sum(dim=-1)).mean()


def attention_distill_loss(teacher_attention, student_attention):
    teacher = teacher_attention.detach().flatten(2)
    student = student_attention.flatten(2)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    teacher = teacher.clamp_min(1e-6)
    student = student.clamp_min(1e-6)
    return _mean_distribution_kl(student.log(), teacher)


def relation_distill_loss(teacher_relation, student_relation):
    return F.mse_loss(student_relation, teacher_relation.detach())


def _weighted_relation_mse(teacher_relation, student_relation, weight):
    teacher = teacher_relation.detach()
    weight = weight.to(device=student_relation.device, dtype=student_relation.dtype)
    weight = weight / weight.sum().clamp_min(1e-6)
    return (((student_relation - teacher) ** 2) * weight.unsqueeze(0)).sum(dim=(1, 2)).mean()


def multiscale_relation_distill_loss(
    teacher_relation,
    student_relation,
    local_weight,
    remote_weight,
    local_scale=0.7,
    remote_scale=0.3,
):
    local_scale = float(local_scale)
    remote_scale = float(remote_scale)
    normalizer = max(local_scale + remote_scale, 1e-6)
    local_loss = _weighted_relation_mse(teacher_relation, student_relation, local_weight)
    remote_loss = _weighted_relation_mse(teacher_relation, student_relation, remote_weight)
    return (local_scale * local_loss + remote_scale * remote_loss) / normalizer


def prototype_distill_loss(teacher_logits, student_logits, temperature=1.0):
    tau = max(float(temperature), 1e-6)
    teacher_prob = F.softmax(teacher_logits.detach() / tau, dim=-1)
    student_log_prob = F.log_softmax(student_logits / tau, dim=-1)
    return _mean_distribution_kl(student_log_prob, teacher_prob) * (tau ** 2)
