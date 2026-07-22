import math

import torch
import torch.nn as nn


class IdentitySurfaceRefiner(nn.Module):
    def forward(self, surface_features, surface_roi_masks):
        del surface_roi_masks
        return surface_features


class AtlasGuidedSurfaceRefiner(nn.Module):
    """Inject atlas membership priors into vertex-level surface features."""

    def __init__(self, num_rois, feature_dim, hidden_dim=None, dropout=0.0, residual_init=0.1):
        super().__init__()
        hidden_dim = int(hidden_dim or feature_dim)
        self.roi_embedding = nn.Parameter(torch.randn(1, int(num_rois), int(feature_dim)) * 0.02)
        self.context_proj = nn.Sequential(
            nn.Linear(int(feature_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(feature_dim)),
        )
        self.gate = nn.Sequential(
            nn.Linear(int(feature_dim) * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(feature_dim)),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(int(feature_dim))
        residual_init = min(max(float(residual_init), 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))

    def forward(self, surface_features, surface_roi_masks):
        vertex_weights = surface_roi_masks.transpose(1, 2).to(dtype=surface_features.dtype)
        vertex_weights = vertex_weights / vertex_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        atlas_context = vertex_weights @ self.roi_embedding.expand(surface_features.size(0), -1, -1)
        update = self.context_proj(atlas_context)
        gate = self.gate(torch.cat([surface_features, atlas_context], dim=-1))
        residual_scale = torch.sigmoid(self.residual_logit)
        return self.norm(surface_features + residual_scale * gate * update)


def build_surface_refiner(name, num_rois, feature_dim, hidden_dim=None, dropout=0.0, residual_init=0.1):
    name = str(name or "none").strip().lower()
    if name in {"", "none", "identity", "off", "false", "0"}:
        return IdentitySurfaceRefiner()
    if name in {"agsr", "atlas_guided", "atlas-guided", "atlas_guided_surface_refiner"}:
        return AtlasGuidedSurfaceRefiner(
            num_rois=num_rois,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            residual_init=residual_init,
        )
    raise ValueError(f"Unsupported surface_refiner: {name}")
