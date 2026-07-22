import torch
import torch.nn as nn

from data.mask_utils import masked_pool_3d, normalize_attention_maps, prepare_pooling_masks


class SurfaceROIPooling(nn.Module):
    def forward(self, surface_features, surface_roi_masks):
        weights = surface_roi_masks / surface_roi_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        tokens = torch.einsum("bkn,bnd->bkd", weights, surface_features)
        return tokens, weights


class MaskedVolumeTokenPooling(nn.Module):
    def __init__(self, in_dim, out_dim, num_rois):
        super().__init__()
        self.num_rois = int(num_rois)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, feat_map, local_masks, roi_masks):
        target_shape = feat_map.shape[-3:]
        local_masks_l = prepare_pooling_masks(local_masks, target_shape, num_rois=self.num_rois)
        roi_masks_l = prepare_pooling_masks(roi_masks, target_shape, num_rois=self.num_rois)
        local_tokens = masked_pool_3d(feat_map, local_masks_l)
        roi_tokens = masked_pool_3d(feat_map, roi_masks_l)
        fused = self.fusion(torch.cat([local_tokens, roi_tokens], dim=-1))
        aux = {
            "local_masks": local_masks_l,
            "roi_masks": roi_masks_l,
            "local_attention": normalize_attention_maps(local_masks_l),
            "roi_attention": normalize_attention_maps(roi_masks_l),
        }
        return fused, local_tokens, roi_tokens, aux
