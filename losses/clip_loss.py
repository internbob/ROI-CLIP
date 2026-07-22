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

