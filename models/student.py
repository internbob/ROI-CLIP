import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import GlobalClassifierHead, ROIClassifierHead
from .clinical_encoder import ClinicalGatedFusion, ImagingClinicalCrossTransformer
from .prototypes import DiseasePrototypeBank
from .relation import pairwise_cosine_matrix
from .volume_encoder import SFCNVolumeEncoder3D


class MRIOnlyStudent(nn.Module):
    def __init__(
        self,
        feature_dim,
        proj_dim,
        num_rois,
        num_classes,
        base_channels=32,
        dropout=0.1,
        prototype_temperature=0.1,
        roi_residual_init=0.1,
        volume_norm="batch",
        use_clinical=False,
        clinical_input_dim=0,
        clinical_fusion_mode="gated",
        clinical_hidden_dim=256,
        clinical_dropout=0.1,
        clinical_num_heads=4,
        clinical_num_layers=2,
    ):
        super().__init__()
        self.use_clinical = bool(use_clinical)
        self.clinical_fusion_mode = str(clinical_fusion_mode).lower()
        if self.clinical_fusion_mode not in {"gated", "cross"}:
            raise ValueError(f"Unsupported clinical fusion mode: {clinical_fusion_mode}")
        self.volume_encoder = SFCNVolumeEncoder3D(
            in_channels=1,
            base_channels=base_channels,
            out_dim=feature_dim,
            norm=volume_norm,
        )
        self.roi_queries = nn.Parameter(torch.randn(1, num_rois, feature_dim) * 0.02)
        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(feature_dim, feature_dim)
        self.to_v = nn.Linear(feature_dim, feature_dim)
        self.scale = 1.0 / math.sqrt(float(feature_dim))
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.projector = nn.Linear(feature_dim, proj_dim)
        self.prototype_bank = DiseasePrototypeBank(
            num_classes=num_classes,
            num_rois=num_rois,
            proto_dim=proj_dim,
            temperature=prototype_temperature,
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

    def forward(self, image, clinical_data=None):
        feat_map, multi_scale_features, global_feat = self.volume_encoder(image)
        bsz, channels, h, w, d = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)

        queries = self.roi_queries.expand(bsz, -1, -1)
        q = self.to_q(queries)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        student_tokens = self.out_proj(attn @ v)
        z_student = F.normalize(self.projector(student_tokens), dim=-1)
        classifier_tokens = z_student
        cross_outputs = None
        clinical_feature = None
        clinical_gate = None
        if self.use_clinical:
            if clinical_data is None:
                raise ValueError("clinical_data is required when use_clinical=True")
            if self.clinical_fusion_mode == "gated":
                fused_global_feat, clinical_feature, clinical_gate = self.clinical_fusion(
                    global_feat,
                    clinical_data,
                )
                global_logits = self.global_head(fused_global_feat)
            else:
                imaging_tokens = torch.cat([self.global_to_cross(global_feat).unsqueeze(1), z_student], dim=1)
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
        attention_maps = attn.reshape(bsz, queries.size(1), h, w, d)

        return {
            "logits": logits,
            "global_logits": global_logits,
            "roi_logits": roi_logits,
            "roi_weight": roi_weight,
            "prob": prob,
            "student_tokens": student_tokens,
            "z_student": z_student,
            "attention_maps": attention_maps,
            "relation": pairwise_cosine_matrix(z_student),
            "prototype_logits": self.prototype_bank(z_student),
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
