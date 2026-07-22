import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIClipProjector(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.surface_proj = nn.Linear(in_dim, proj_dim)
        self.volume_proj = nn.Linear(in_dim, proj_dim)

    def forward(self, surface_tokens, volume_tokens):
        z_surface = F.normalize(self.surface_proj(surface_tokens), dim=-1)
        z_volume = F.normalize(self.volume_proj(volume_tokens), dim=-1)
        similarity = z_surface.flatten(0, 1) @ z_volume.flatten(0, 1).transpose(0, 1)
        return z_surface, z_volume, similarity

