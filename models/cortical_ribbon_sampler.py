import torch
import torch.nn as nn
import torch.nn.functional as F


class CorticalRibbonSampler(nn.Module):
    """Sample multi-depth cortical features and pool vertices into ROI tokens."""

    def __init__(self, feature_dim: int, num_rois: int, depth_attention: bool = True):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_rois = int(num_rois)
        self.use_depth_attention = bool(depth_attention)
        hidden_dim = max(1, self.feature_dim // 2)
        self.depth_score = (
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.use_depth_attention
            else None
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        ribbon_grid: torch.Tensor,
        vertex_roi_index: torch.Tensor,
        ribbon_valid_mask: torch.Tensor = None,
        vertex_valid_mask: torch.Tensor = None,
    ):
        if feature_map.ndim != 5:
            raise ValueError(f"feature_map must be [B,C,D,H,W], got {tuple(feature_map.shape)}")
        if ribbon_grid.ndim != 4 or ribbon_grid.shape[-1] != 3:
            raise ValueError(f"ribbon_grid must be [B,N,L,3], got {tuple(ribbon_grid.shape)}")
        batch_size, channels = feature_map.shape[:2]
        grid_batch, num_vertices, num_depths, _ = ribbon_grid.shape
        if grid_batch != batch_size or channels != self.feature_dim:
            raise ValueError(
                f"Ribbon/feature mismatch: feature={tuple(feature_map.shape)}, grid={tuple(ribbon_grid.shape)}"
            )
        if vertex_roi_index.shape != (batch_size, num_vertices):
            raise ValueError(
                f"vertex_roi_index must be {(batch_size, num_vertices)}, got {tuple(vertex_roi_index.shape)}"
            )
        if not torch.isfinite(ribbon_grid).all():
            raise ValueError("ribbon_grid contains NaN or Inf")

        if ribbon_valid_mask is None:
            ribbon_valid_mask = (ribbon_grid >= -1.0).all(dim=-1) & (ribbon_grid <= 1.0).all(dim=-1)
        else:
            ribbon_valid_mask = ribbon_valid_mask.bool()
        if ribbon_valid_mask.shape != (batch_size, num_vertices, num_depths):
            raise ValueError(
                "ribbon_valid_mask must be [B,N,L], "
                f"got {tuple(ribbon_valid_mask.shape)}"
            )

        grid = ribbon_grid.reshape(batch_size, num_vertices * num_depths, 1, 1, 3)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        depth_features = sampled[..., 0, 0].transpose(1, 2)
        depth_features = depth_features.reshape(batch_size, num_vertices, num_depths, channels)

        if self.depth_score is None:
            depth_logits = torch.zeros_like(depth_features[..., 0])
        else:
            depth_logits = self.depth_score(depth_features).squeeze(-1)
        has_valid_depth = ribbon_valid_mask.any(dim=-1, keepdim=True)
        masked_logits = depth_logits.masked_fill(~ribbon_valid_mask, torch.finfo(depth_logits.dtype).min)
        masked_logits = torch.where(has_valid_depth, masked_logits, torch.zeros_like(masked_logits))
        depth_attention = torch.softmax(masked_logits, dim=-1)
        depth_attention = depth_attention * ribbon_valid_mask.to(depth_attention.dtype)
        depth_attention = depth_attention / depth_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        vertex_features = (depth_attention.unsqueeze(-1) * depth_features).sum(dim=2)

        assigned_vertex = (vertex_roi_index >= 0) & (
            vertex_roi_index < self.num_rois
        )
        valid_vertex = has_valid_depth.squeeze(-1) & assigned_vertex
        if vertex_valid_mask is not None:
            if vertex_valid_mask.shape != (batch_size, num_vertices):
                raise ValueError(
                    "vertex_valid_mask must be [B,N], "
                    f"got {tuple(vertex_valid_mask.shape)}"
                )
            valid_vertex = valid_vertex & vertex_valid_mask.bool()

        safe_index = vertex_roi_index.clamp(min=0, max=self.num_rois - 1).long()
        weighted_vertices = vertex_features * valid_vertex.unsqueeze(-1).to(vertex_features.dtype)
        roi_sum = feature_map.new_zeros((batch_size, self.num_rois, channels))
        roi_sum.scatter_add_(1, safe_index.unsqueeze(-1).expand(-1, -1, channels), weighted_vertices)
        roi_count = feature_map.new_zeros((batch_size, self.num_rois, 1))
        roi_count.scatter_add_(
            1,
            safe_index.unsqueeze(-1),
            valid_vertex.unsqueeze(-1).to(feature_map.dtype),
        )
        roi_features = roi_sum / roi_count.clamp_min(1.0)
        roi_total_count = feature_map.new_zeros((batch_size, self.num_rois, 1))
        roi_total_count.scatter_add_(
            1,
            safe_index.unsqueeze(-1),
            assigned_vertex.unsqueeze(-1).to(feature_map.dtype),
        )
        roi_valid_ratio = roi_count / roi_total_count.clamp_min(1.0)

        return {
            "depth_features": depth_features,
            "vertex_features": vertex_features,
            "roi_features": roi_features,
            "depth_attention": depth_attention,
            "vertex_valid_mask": valid_vertex,
            "roi_valid_mask": roi_count.squeeze(-1) > 0,
            "roi_vertex_count": roi_count.squeeze(-1),
            "roi_total_vertex_count": roi_total_count.squeeze(-1),
            "roi_valid_ratio": roi_valid_ratio.squeeze(-1),
        }
