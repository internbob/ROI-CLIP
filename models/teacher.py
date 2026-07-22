import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import GlobalClassifierHead, ROIClassifierHead
from .clinical_encoder import ClinicalFeatureTokenizer, ClinicalGatedFusion, ImagingClinicalCrossTransformer
from .prototypes import DiseasePrototypeBank
from .relation import pairwise_cosine_matrix
from .roi_clip import ROIClipProjector
from .surface_encoder import SurfaceEncoder
from .surface_refiner import build_surface_refiner
from .token_pooling import MaskedVolumeTokenPooling, SurfaceROIPooling
from .volume_encoder import SFCNVolumeEncoder3D


class SurfaceOnlyTransformerBaseline(nn.Module):
    """Surface-only comparison baseline with ROI tokens and a compact Transformer."""

    def __init__(
        self,
        surface_in_channels,
        feature_dim,
        proj_dim,
        num_rois,
        num_classes,
        num_vertices,
        surface_hidden_dim=128,
        dropout=0.1,
        surface_refiner="none",
        surface_refiner_hidden_dim=0,
        surface_refiner_dropout=0.0,
        surface_refiner_residual_init=0.1,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.surface_encoder = SurfaceEncoder(surface_in_channels, surface_hidden_dim, feature_dim, num_vertices=num_vertices)
        self.surface_refiner = build_surface_refiner(
            name=surface_refiner,
            num_rois=num_rois,
            feature_dim=feature_dim,
            hidden_dim=int(surface_refiner_hidden_dim) if int(surface_refiner_hidden_dim or 0) > 0 else None,
            dropout=float(surface_refiner_dropout),
            residual_init=float(surface_refiner_residual_init),
        )
        self.surface_pool = SurfaceROIPooling()
        self.token_proj = nn.Linear(feature_dim, proj_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, int(num_rois) + 1, proj_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=int(num_heads),
            dim_feedforward=proj_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = GlobalClassifierHead(feature_dim=proj_dim, num_classes=num_classes, dropout=dropout)

    def forward(
        self,
        image,
        surface_data,
        surface_roi_masks,
        local_ribbon_masks,
        roi_volume_masks,
        labels=None,
        clinical_data=None,
    ):
        del image, local_ribbon_masks, roi_volume_masks, labels, clinical_data
        surface_features = self.surface_encoder(surface_data)
        surface_features = self.surface_refiner(surface_features, surface_roi_masks)
        surface_tokens, surface_roi_weights = self.surface_pool(surface_features, surface_roi_masks)
        z_surface = self.token_proj(surface_tokens)
        cls = self.cls_token.expand(z_surface.size(0), -1, -1)
        tokens = torch.cat([cls, z_surface], dim=1) + self.pos_embed[:, : z_surface.size(1) + 1]
        encoded = self.norm(self.transformer(tokens))
        logits = self.classifier(encoded[:, 0])
        return {
            "logits": logits,
            "global_logits": logits,
            "roi_weight": logits.new_zeros(()),
            "prob": torch.softmax(logits, dim=-1),
            "surface_tokens": surface_tokens,
            "z_surface": F.normalize(z_surface, dim=-1),
            "relation": pairwise_cosine_matrix(F.normalize(z_surface, dim=-1)),
            "surface_roi_weights": surface_roi_weights,
        }


class MRIClinicalCLIPBaseline(nn.Module):
    """Contrastive MRI-clinical CLIP comparison baseline."""

    def __init__(
        self,
        feature_dim,
        proj_dim,
        num_classes,
        base_channels=32,
        dropout=0.1,
        volume_norm="batch",
        clinical_input_dim=0,
        clinical_hidden_dim=256,
        clinical_num_heads=4,
        clinical_num_layers=2,
    ):
        super().__init__()
        if int(clinical_input_dim) <= 0:
            raise ValueError("clinical_input_dim must be positive for MRIClinicalCLIPBaseline")
        self.volume_encoder = SFCNVolumeEncoder3D(
            in_channels=1,
            base_channels=base_channels,
            out_dim=feature_dim,
            norm=volume_norm,
        )
        self.mri_projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, proj_dim),
        )
        self.clinical_tokenizer = ClinicalFeatureTokenizer(int(clinical_input_dim), proj_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=int(clinical_num_heads),
            dim_feedforward=proj_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.clinical_transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(clinical_num_layers))
        self.clinical_norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim * 4),
            nn.Linear(proj_dim * 4, clinical_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clinical_hidden_dim, num_classes),
        )

    def forward(
        self,
        image,
        surface_data,
        surface_roi_masks,
        local_ribbon_masks,
        roi_volume_masks,
        labels=None,
        clinical_data=None,
    ):
        del surface_data, surface_roi_masks, local_ribbon_masks, roi_volume_masks, labels
        if clinical_data is None:
            raise ValueError("clinical_data is required for MRIClinicalCLIPBaseline")
        _, multi_scale_features, global_feat = self.volume_encoder(image)
        mri_embed = F.normalize(self.mri_projector(global_feat), dim=-1)
        clinical_tokens = self.clinical_transformer(self.clinical_tokenizer(clinical_data))
        clinical_embed = F.normalize(self.clinical_norm(clinical_tokens.mean(dim=1)), dim=-1)
        fused = torch.cat(
            [mri_embed, clinical_embed, torch.abs(mri_embed - clinical_embed), mri_embed * clinical_embed],
            dim=-1,
        )
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "global_logits": logits,
            "roi_weight": logits.new_zeros(()),
            "prob": torch.softmax(logits, dim=-1),
            "z_mri": mri_embed.unsqueeze(1),
            "z_clinical": clinical_embed.unsqueeze(1),
            "multi_scale_features": multi_scale_features,
            "global_volume_feat": global_feat,
            "clinical_tokens": clinical_tokens,
        }


class CortexWeaveROIClipTeacher(nn.Module):
    def __init__(
        self,
        surface_in_channels,
        feature_dim,
        proj_dim,
        num_rois,
        num_classes,
        num_vertices,
        base_channels=32,
        surface_hidden_dim=128,
        dropout=0.1,
        prototype_temperature=0.1,
        roi_residual_init=0.1,
        use_roi_branch=True,
        volume_norm="batch",
        use_clinical=False,
        clinical_input_dim=0,
        clinical_fusion_mode="gated",
        clinical_hidden_dim=256,
        clinical_dropout=0.1,
        clinical_num_heads=4,
        clinical_num_layers=2,
        surface_refiner="none",
        surface_refiner_hidden_dim=0,
        surface_refiner_dropout=0.0,
        surface_refiner_residual_init=0.1,
    ):
        super().__init__()
        self.use_roi_branch = bool(use_roi_branch)
        self.use_clinical = bool(use_clinical)
        self.clinical_fusion_mode = str(clinical_fusion_mode).lower()
        if self.clinical_fusion_mode not in {"gated", "cross"}:
            raise ValueError(f"Unsupported clinical fusion mode: {clinical_fusion_mode}")
        self.surface_encoder = SurfaceEncoder(surface_in_channels, surface_hidden_dim, feature_dim, num_vertices=num_vertices)
        self.surface_refiner = build_surface_refiner(
            name=surface_refiner,
            num_rois=num_rois,
            feature_dim=feature_dim,
            hidden_dim=int(surface_refiner_hidden_dim) if int(surface_refiner_hidden_dim or 0) > 0 else None,
            dropout=float(surface_refiner_dropout),
            residual_init=float(surface_refiner_residual_init),
        )
        self.surface_pool = SurfaceROIPooling()
        self.volume_encoder = SFCNVolumeEncoder3D(
            in_channels=1,
            base_channels=base_channels,
            out_dim=feature_dim,
            norm=volume_norm,
        )
        self.volume_pool = MaskedVolumeTokenPooling(in_dim=feature_dim, out_dim=feature_dim, num_rois=num_rois)
        self.roi_clip = ROIClipProjector(in_dim=feature_dim, proj_dim=proj_dim)
        self.prototype_bank = DiseasePrototypeBank(
            num_classes=num_classes,
            num_rois=num_rois,
            proto_dim=proj_dim,
            temperature=prototype_temperature,
        )
        self.teacher_fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.classifier = ROIClassifierHead(token_dim=proj_dim, num_classes=num_classes, dropout=dropout)
        self.global_head = GlobalClassifierHead(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        self.clinical_fusion = (
            ClinicalGatedFusion(
                input_dim=int(clinical_input_dim),
                feature_dim=feature_dim,
                hidden_dim=int(clinical_hidden_dim),
                dropout=float(clinical_dropout),
            )
            if self.use_clinical and self.clinical_fusion_mode == "gated"
            else None
        )
        self.global_to_cross = (
            nn.Linear(feature_dim, proj_dim)
            if self.use_clinical and self.clinical_fusion_mode == "cross"
            else None
        )
        self.cross_global_head = (
            GlobalClassifierHead(feature_dim=proj_dim, num_classes=num_classes, dropout=dropout)
            if self.use_clinical and self.clinical_fusion_mode == "cross"
            else None
        )
        self.clinical_cross_fusion = (
            ImagingClinicalCrossTransformer(
                clinical_input_dim=int(clinical_input_dim),
                embed_dim=proj_dim,
                max_imaging_tokens=num_rois + 1,
                num_heads=int(clinical_num_heads),
                num_layers=int(clinical_num_layers),
                dropout=dropout,
            )
            if self.use_clinical and self.clinical_fusion_mode == "cross"
            else None
        )
        roi_residual_init = min(max(float(roi_residual_init), 1e-4), 1.0 - 1e-4)
        self.roi_logit_scale = nn.Parameter(torch.tensor(math.log(roi_residual_init / (1.0 - roi_residual_init))))

    def forward(
        self,
        image,
        surface_data,
        surface_roi_masks,
        local_ribbon_masks,
        roi_volume_masks,
        labels=None,
        clinical_data=None,
    ):
        del labels
        feat_map, multi_scale_features, global_feat = self.volume_encoder(image)
        if not self.use_roi_branch:
            cross_outputs = None
            clinical_feature = None
            clinical_gate = None
            if self.use_clinical:
                if clinical_data is None:
                    raise ValueError("clinical_data is required when use_clinical=True")
                clinical_data = clinical_data.to(global_feat.device)
                if self.clinical_fusion_mode == "gated":
                    fused_global_feat, clinical_feature, clinical_gate = self.clinical_fusion(
                        global_feat,
                        clinical_data,
                    )
                    global_logits = self.global_head(fused_global_feat)
                else:
                    cross_outputs = self.clinical_cross_fusion(
                        self.global_to_cross(global_feat).unsqueeze(1),
                        clinical_data,
                    )
                    fused_global_feat = cross_outputs["image_tokens"][:, 0]
                    global_logits = self.cross_global_head(fused_global_feat)
            else:
                fused_global_feat = global_feat
                global_logits = self.global_head(global_feat)
            roi_weight = global_logits.new_zeros(())
            return {
                "logits": global_logits,
                "global_logits": global_logits,
                "roi_weight": roi_weight,
                "prob": torch.softmax(global_logits, dim=-1),
                "multi_scale_features": multi_scale_features,
                "global_volume_feat": global_feat,
                "fused_global_feat": fused_global_feat,
                "clinical_feature": clinical_feature,
                "clinical_gate": clinical_gate,
                "clinical_tokens": cross_outputs["clinical_tokens"] if cross_outputs is not None else None,
                "image_to_clinical_attention": (
                    cross_outputs["image_to_clinical_attention"] if cross_outputs is not None else None
                ),
                "clinical_to_image_attention": (
                    cross_outputs["clinical_to_image_attention"] if cross_outputs is not None else None
                ),
            }

        surface_features = self.surface_encoder(surface_data)
        surface_features = self.surface_refiner(surface_features, surface_roi_masks)
        surface_tokens, surface_roi_weights = self.surface_pool(surface_features, surface_roi_masks)

        volume_tokens, local_tokens, roi_tokens, aux = self.volume_pool(feat_map, local_ribbon_masks, roi_volume_masks)

        z_surface, z_volume, similarity_matrix = self.roi_clip(surface_tokens, volume_tokens)
        z_teacher = self.teacher_fusion(torch.cat([z_surface, z_volume], dim=-1))
        cross_outputs = None
        clinical_feature = None
        clinical_gate = None
        classifier_tokens = z_teacher
        if self.use_clinical:
            if clinical_data is None:
                raise ValueError("clinical_data is required when use_clinical=True")
            clinical_data = clinical_data.to(global_feat.device)
            if self.clinical_fusion_mode == "gated":
                fused_global_feat, clinical_feature, clinical_gate = self.clinical_fusion(
                    global_feat,
                    clinical_data,
                )
                global_logits = self.global_head(fused_global_feat)
            else:
                imaging_tokens = torch.cat([self.global_to_cross(global_feat).unsqueeze(1), z_teacher], dim=1)
                cross_outputs = self.clinical_cross_fusion(imaging_tokens, clinical_data)
                fused_global_feat = cross_outputs["image_tokens"][:, 0]
                classifier_tokens = cross_outputs["image_tokens"][:, 1:]
                global_logits = self.cross_global_head(fused_global_feat)
        else:
            fused_global_feat = global_feat
            global_logits = self.global_head(global_feat)
        roi_logits = self.classifier(classifier_tokens)
        roi_weight = torch.sigmoid(self.roi_logit_scale)
        logits = global_logits + roi_weight * roi_logits
        prob = torch.softmax(logits, dim=-1)

        surface_relation = pairwise_cosine_matrix(z_surface)
        volume_relation = pairwise_cosine_matrix(z_volume)
        teacher_relation = pairwise_cosine_matrix(z_teacher)

        outputs = {
            "logits": logits,
            "global_logits": global_logits,
            "roi_logits": roi_logits,
            "roi_weight": roi_weight,
            "prob": prob,
            "surface_tokens": surface_tokens,
            "volume_tokens": volume_tokens,
            "local_tokens": local_tokens,
            "roi_tokens": roi_tokens,
            "z_surface": z_surface,
            "z_volume": z_volume,
            "z_teacher": z_teacher,
            "surface_relation": surface_relation,
            "volume_relation": volume_relation,
            "relation": teacher_relation,
            "surface_prototype_logits": self.prototype_bank(z_surface),
            "volume_prototype_logits": self.prototype_bank(z_volume),
            "prototype_logits": self.prototype_bank(z_teacher),
            "mask_attention": aux["local_attention"],
            "similarity_matrix": similarity_matrix,
            "surface_roi_weights": surface_roi_weights,
            "multi_scale_features": multi_scale_features,
            "global_volume_feat": global_feat,
            "fused_global_feat": fused_global_feat,
            "clinical_feature": clinical_feature,
            "clinical_gate": clinical_gate,
            "cross_image_tokens": cross_outputs["image_tokens"] if cross_outputs is not None else None,
            "clinical_tokens": cross_outputs["clinical_tokens"] if cross_outputs is not None else None,
            "image_to_clinical_attention": (
                cross_outputs["image_to_clinical_attention"] if cross_outputs is not None else None
            ),
            "clinical_to_image_attention": (
                cross_outputs["clinical_to_image_attention"] if cross_outputs is not None else None
            ),
        }
        return outputs
