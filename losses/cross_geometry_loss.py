import torch
import torch.nn.functional as F


def _cross_geometry_similarity(surface_tokens, ribbon_tokens):
    if surface_tokens.ndim != 3 or ribbon_tokens.ndim != 3:
        raise ValueError("Cross-geometry tokens must be [B,K,C]")
    if surface_tokens.shape != ribbon_tokens.shape:
        raise ValueError(
            "Cross-geometry token shapes must match, "
            f"got {tuple(surface_tokens.shape)} and {tuple(ribbon_tokens.shape)}"
        )
    surface_tokens = F.normalize(surface_tokens, dim=-1)
    ribbon_tokens = F.normalize(ribbon_tokens, dim=-1)
    return torch.matmul(surface_tokens, ribbon_tokens.transpose(-1, -2))


def cross_geometry_correspondence_loss(surface_tokens, ribbon_tokens, temperature=0.1):
    """Symmetric, subject-local hard same-ROI correspondence loss."""
    similarity = _cross_geometry_similarity(surface_tokens, ribbon_tokens)
    batch_size, num_rois = similarity.shape[:2]
    logits = similarity / max(float(temperature), 1e-6)
    targets = torch.arange(num_rois, device=logits.device).repeat(batch_size)
    loss_sr = F.cross_entropy(logits.reshape(batch_size * num_rois, num_rois), targets)
    loss_rs = F.cross_entropy(
        logits.transpose(-1, -2).reshape(batch_size * num_rois, num_rois),
        targets,
    )
    return 0.5 * (loss_sr + loss_rs)


def _directional_retrieval_metrics(similarity, topology_mask=None):
    batch_size, num_rois, _ = similarity.shape
    targets = torch.arange(num_rois, device=similarity.device).view(1, num_rois, 1)
    targets = targets.expand(batch_size, -1, -1)
    ranking = similarity.argsort(dim=-1, descending=True)
    matches = ranking.eq(targets)
    ranks = matches.float().argmax(dim=-1) + 1
    top1 = ranking[..., 0]

    metrics = {
        "roi_retrieval_acc1": (ranks == 1).float().mean(),
        "roi_retrieval_acc5": (ranks <= min(5, num_rois)).float().mean(),
        "roi_retrieval_mrr": ranks.float().reciprocal().mean(),
    }
    if topology_mask is not None:
        allowed = topology_mask.to(device=similarity.device, dtype=torch.bool)[:num_rois, :num_rois]
        allowed = allowed.clone()
        allowed.fill_diagonal_(True)
        target_rows = torch.arange(num_rois, device=similarity.device).view(1, num_rois)
        target_rows = target_rows.expand(batch_size, -1)
        metrics["topology_retrieval_acc"] = allowed[target_rows, top1].float().mean()
    else:
        metrics["topology_retrieval_acc"] = metrics["roi_retrieval_acc1"]
    return metrics


@torch.no_grad()
def cross_geometry_retrieval_metrics(surface_tokens, ribbon_tokens, topology_affinity=None):
    """Bidirectional retrieval and cosine-separation diagnostics."""
    similarity = _cross_geometry_similarity(surface_tokens, ribbon_tokens)
    topology_mask = topology_affinity > 0 if topology_affinity is not None else None
    forward = _directional_retrieval_metrics(similarity, topology_mask=topology_mask)
    reverse_mask = topology_mask.transpose(0, 1) if topology_mask is not None else None
    reverse = _directional_retrieval_metrics(
        similarity.transpose(-1, -2),
        topology_mask=reverse_mask,
    )

    num_rois = int(similarity.size(1))
    diagonal = similarity.diagonal(dim1=-2, dim2=-1).mean()
    if num_rois > 1:
        eye = torch.eye(num_rois, device=similarity.device, dtype=torch.bool).unsqueeze(0)
        off_diagonal = similarity.masked_select(~eye).mean()
    else:
        off_diagonal = diagonal.new_zeros(())

    return {
        key: 0.5 * (forward[key] + reverse[key])
        for key in forward
    } | {
        "same_roi_similarity": diagonal,
        "off_diagonal_similarity": off_diagonal,
        "diagonal_similarity_gap": diagonal - off_diagonal,
    }
