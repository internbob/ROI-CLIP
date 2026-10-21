import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import GlobalClassifierHead, ROIClassifierHead
from .clinical_encoder import (
    ClinicalAtlasPromptFusion,
    ClinicalFeatureTokenizer,
    ClinicalGatedFusion,
    ImagingClinicalCrossTransformer,
)
from .cortical_ribbon_sampler import CorticalRibbonSampler
from .disease_aware_graph import DiseaseAwareROIGraph
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


class SurfaceVolumeConcatBaseline(nn.Module):
    """Directly concatenate cortical surface and voxel features for classification."""

    def __init__(
        self,
        surface_in_channels,
        feature_dim,
        num_classes,
        num_vertices,
        base_channels=32,
        surface_hidden_dim=128,
        dropout=0.1,
        volume_norm="batch",
        use_clinical=False,
        clinical_input_dim=0,
        clinical_hidden_dim=256,
        surface_refiner="none",
        surface_refiner_hidden_dim=0,
        surface_refiner_dropout=0.0,
        surface_refiner_residual_init=0.1,
        num_rois=62,
    ):
        super().__init__()
        self.use_clinical = bool(use_clinical)
        self.surface_encoder = SurfaceEncoder(
            surface_in_channels,
            surface_hidden_dim,
            feature_dim,
            num_vertices=num_vertices,
        )
        self.surface_refiner = build_surface_refiner(
            name=surface_refiner,
            num_rois=num_rois,
            feature_dim=feature_dim,
            hidden_dim=int(surface_refiner_hidden_dim) if int(surface_refiner_hidden_dim or 0) > 0 else None,
            dropout=float(surface_refiner_dropout),
            residual_init=float(surface_refiner_residual_init),
        )
        self.surface_pool = SurfaceROIPooling()
        self.surface_summary = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.volume_encoder = SFCNVolumeEncoder3D(
            in_channels=1,
            base_channels=base_channels,
            out_dim=feature_dim,
            norm=volume_norm,
        )
        self.clinical_encoder = (
            nn.Sequential(
                nn.LayerNorm(int(clinical_input_dim)),
                nn.Linear(int(clinical_input_dim), int(clinical_hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(clinical_hidden_dim), feature_dim),
                nn.LayerNorm(feature_dim),
            )
            if self.use_clinical and int(clinical_input_dim) > 0
            else None
        )
        concat_dim = feature_dim * (3 if self.clinical_encoder is not None else 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, int(clinical_hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(clinical_hidden_dim), num_classes),
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
        del local_ribbon_masks, roi_volume_masks, labels
        _, multi_scale_features, global_feat = self.volume_encoder(image)
        surface_features = self.surface_encoder(surface_data)
        surface_features = self.surface_refiner(surface_features, surface_roi_masks)
        surface_tokens, surface_roi_weights = self.surface_pool(surface_features, surface_roi_masks)
        surface_mean = surface_tokens.mean(dim=1)
        surface_max = surface_tokens.max(dim=1).values
        surface_global = self.surface_summary(torch.cat([surface_mean, surface_max], dim=-1))

        features = [global_feat, surface_global]
        clinical_feature = None
        if self.clinical_encoder is not None:
            if clinical_data is None:
                raise ValueError("clinical_data is required when use_clinical=True")
            clinical_feature = self.clinical_encoder(clinical_data.to(global_feat.device))
            features.append(clinical_feature)
        fused = torch.cat(features, dim=-1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "global_logits": logits,
            "roi_weight": logits.new_zeros(()),
            "prob": torch.softmax(logits, dim=-1),
            "surface_tokens": surface_tokens,
            "surface_roi_weights": surface_roi_weights,
            "surface_global_feat": surface_global,
            "global_volume_feat": global_feat,
            "fused_global_feat": fused,
            "clinical_feature": clinical_feature,
            "multi_scale_features": multi_scale_features,
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
        volume_pooling_mode="legacy_mask",
        ribbon_depth_attention=True,
        ribbon_feature_stage="roi",
        ribbon_gate_init=0.1,
        cross_geometry_enabled=False,
        cross_geometry_dim=128,
        cross_geometry_feature_stage="high_res",
        roi_graph_enabled=False,
        roi_graph_num_layers=1,
        roi_graph_num_heads=4,
        roi_graph_dropout=0.1,
        roi_graph_residual_gate_init=0.1,
        roi_graph_use_boundary_strength=False,
        roi_graph_use_homology=False,
        roi_graph_anatomy_adjacency=None,
        roi_graph_boundary_strength=None,
        roi_graph_evaluation_boundary_strength=None,
        roi_graph_homology=None,
        roi_graph_evaluation_homology=None,
    ):
        super().__init__()
        self.use_roi_branch = bool(use_roi_branch)
        self.use_clinical = bool(use_clinical)
        self.volume_pooling_mode = str(volume_pooling_mode).lower()
        if self.volume_pooling_mode not in {"legacy_mask", "ribbon", "hybrid_ribbon"}:
            raise ValueError(f"Unsupported volume pooling mode: {volume_pooling_mode}")
        self.ribbon_feature_stage = str(ribbon_feature_stage).lower()
        if self.ribbon_feature_stage not in {"roi", "high_res"}:
            raise ValueError(f"Unsupported ribbon feature stage: {ribbon_feature_stage}")
        self.cross_geometry_enabled = bool(cross_geometry_enabled)
        self.roi_graph_enabled = bool(roi_graph_enabled)
        self.cross_geometry_feature_stage = str(cross_geometry_feature_stage).lower()
        if self.cross_geometry_feature_stage != "high_res":
            raise ValueError(
                "Phase 2.1 cross-geometry anchors must use the high_res MRI feature stage"
            )
        self.clinical_fusion_mode = str(clinical_fusion_mode).lower()
        if self.clinical_fusion_mode not in {"gated", "cross", "atlas_prompt"}:
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
        ribbon_enabled = self.volume_pooling_mode in {"ribbon", "hybrid_ribbon"}
        ribbon_feature_dim = base_channels if self.ribbon_feature_stage == "high_res" else feature_dim
        # Do not let optional ribbon modules advance the RNG seen by the
        # unchanged legacy/common layers. This keeps same-seed ablations
        # matched outside the pooling branch.
        optional_ribbon_enabled = ribbon_enabled or self.cross_geometry_enabled
        common_rng_state = torch.random.get_rng_state() if optional_ribbon_enabled else None
        self.ribbon_sampler = (
            CorticalRibbonSampler(
                feature_dim=ribbon_feature_dim,
                num_rois=num_rois,
                depth_attention=bool(ribbon_depth_attention),
            )
            if ribbon_enabled
            else None
        )
        self.pure_ribbon_projection = (
            nn.Sequential(nn.LayerNorm(ribbon_feature_dim), nn.Linear(ribbon_feature_dim, feature_dim))
            if self.volume_pooling_mode == "ribbon" and ribbon_feature_dim != feature_dim
            else nn.Identity()
            if self.volume_pooling_mode == "ribbon"
            else None
        )
        if self.volume_pooling_mode == "hybrid_ribbon":
            self.hybrid_legacy_norm = nn.LayerNorm(feature_dim)
            self.hybrid_ribbon_norm = nn.LayerNorm(ribbon_feature_dim)
            self.hybrid_ribbon_projection = (
                nn.Linear(ribbon_feature_dim, feature_dim)
                if ribbon_feature_dim != feature_dim
                else nn.Identity()
            )
            self.hybrid_ribbon_adapter = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
            nn.init.zeros_(self.hybrid_ribbon_adapter[-1].weight)
            nn.init.zeros_(self.hybrid_ribbon_adapter[-1].bias)
            gate_init = min(max(float(ribbon_gate_init), 1e-4), 1.0 - 1e-4)
            self.hybrid_ribbon_gate = nn.Linear(feature_dim * 2 + 1, 1)
            nn.init.zeros_(self.hybrid_ribbon_gate.weight)
            nn.init.constant_(self.hybrid_ribbon_gate.bias, math.log(gate_init / (1.0 - gate_init)))
        else:
            self.hybrid_legacy_norm = None
            self.hybrid_ribbon_norm = None
            self.hybrid_ribbon_projection = None
            self.hybrid_ribbon_adapter = None
            self.hybrid_ribbon_gate = None
        cross_geometry_dim = int(cross_geometry_dim)
        if cross_geometry_dim <= 0:
            raise ValueError("cross_geometry_dim must be positive")
        if self.cross_geometry_enabled:
            self.cross_geometry_ribbon_sampler = CorticalRibbonSampler(
                feature_dim=base_channels,
                num_rois=num_rois,
                depth_attention=False,
            )
            self.cross_geometry_surface_projector = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, cross_geometry_dim),
            )
            self.cross_geometry_ribbon_projector = nn.Sequential(
                nn.LayerNorm(base_channels),
                nn.Linear(base_channels, cross_geometry_dim),
            )
        else:
            self.cross_geometry_ribbon_sampler = None
            self.cross_geometry_surface_projector = None
            self.cross_geometry_ribbon_projector = None
        if common_rng_state is not None:
            torch.random.set_rng_state(common_rng_state)
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
        graph_rng_state = torch.random.get_rng_state() if self.roi_graph_enabled else None
        self.roi_graph = (
            DiseaseAwareROIGraph(
                feature_dim=proj_dim,
                num_rois=num_rois,
                num_heads=roi_graph_num_heads,
                num_layers=roi_graph_num_layers,
                dropout=roi_graph_dropout,
                residual_gate_init=roi_graph_residual_gate_init,
                use_boundary_strength=roi_graph_use_boundary_strength,
                use_homology=roi_graph_use_homology,
                anatomy_adjacency=roi_graph_anatomy_adjacency,
                boundary_strength=roi_graph_boundary_strength,
                evaluation_boundary_strength=roi_graph_evaluation_boundary_strength,
                homology=roi_graph_homology,
                evaluation_homology=roi_graph_evaluation_homology,
            )
            if self.roi_graph_enabled
            else None
        )
        if graph_rng_state is not None:
            torch.random.set_rng_state(graph_rng_state)
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
        self.clinical_atlas_prompt = (
            ClinicalAtlasPromptFusion(
                clinical_input_dim=int(clinical_input_dim),
                feature_dim=feature_dim,
                roi_dim=proj_dim,
                num_rois=num_rois,
                num_domain_prompts=max(2, int(clinical_num_layers) * 2),
                num_heads=int(clinical_num_heads),
                dropout=float(clinical_dropout),
            )
            if self.use_clinical and self.clinical_fusion_mode == "atlas_prompt"
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
        ribbon_grid=None,
        ribbon_valid_mask=None,
        vertex_valid_mask=None,
        vertex_roi_index=None,
    ):
        del labels
        feat_map, multi_scale_features, global_feat = self.volume_encoder(image)
        if not self.use_roi_branch:
            cross_outputs = None
            prompt_outputs = None
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
                elif self.clinical_fusion_mode == "cross":
                    cross_outputs = self.clinical_cross_fusion(
                        self.global_to_cross(global_feat).unsqueeze(1),
                        clinical_data,
                    )
                    fused_global_feat = cross_outputs["image_tokens"][:, 0]
                    global_logits = self.cross_global_head(fused_global_feat)
                else:
                    prompt_outputs = self.clinical_atlas_prompt(
                        global_feature=global_feat,
                        clinical_data=clinical_data,
                    )
                    fused_global_feat = prompt_outputs["global_feature"]
                    clinical_feature = prompt_outputs["clinical_feature"]
                    clinical_gate = prompt_outputs["clinical_gate"]
                    global_logits = self.global_head(fused_global_feat)
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
                "clinical_tokens": (
                    cross_outputs["clinical_tokens"]
                    if cross_outputs is not None
                    else prompt_outputs["clinical_tokens"]
                    if prompt_outputs is not None
                    else None
                ),
                "image_to_clinical_attention": (
                    cross_outputs["image_to_clinical_attention"] if cross_outputs is not None else None
                ),
                "clinical_to_image_attention": (
                    cross_outputs["clinical_to_image_attention"] if cross_outputs is not None else None
                ),
                "clinical_domain_prompts": (
                    prompt_outputs["clinical_domain_prompts"] if prompt_outputs is not None else None
                ),
                "clinical_domain_attention": (
                    prompt_outputs["clinical_domain_attention"] if prompt_outputs is not None else None
                ),
                "roi_prompt_attention": prompt_outputs["roi_prompt_attention"] if prompt_outputs is not None else None,
            }

        surface_features = self.surface_encoder(surface_data)
        surface_features = self.surface_refiner(surface_features, surface_roi_masks)
        surface_tokens, surface_roi_weights = self.surface_pool(surface_features, surface_roi_masks)

        cross_geometry_outputs = None
        cross_geometry_surface = None
        cross_geometry_ribbon = None
        ribbon_anchor_tokens = None
        if self.cross_geometry_enabled:
            required = {
                "ribbon_grid": ribbon_grid,
                "ribbon_valid_mask": ribbon_valid_mask,
                "vertex_valid_mask": vertex_valid_mask,
                "vertex_roi_index": vertex_roi_index,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"Cross-geometry alignment requires inputs: {', '.join(missing)}")
            cross_geometry_outputs = self.cross_geometry_ribbon_sampler(
                feature_map=multi_scale_features[0],
                ribbon_grid=ribbon_grid,
                vertex_roi_index=vertex_roi_index,
                ribbon_valid_mask=ribbon_valid_mask,
                vertex_valid_mask=vertex_valid_mask,
            )
            ribbon_anchor_tokens = cross_geometry_outputs["roi_features"]
            cross_geometry_surface = F.normalize(
                self.cross_geometry_surface_projector(surface_tokens), dim=-1
            )
            cross_geometry_ribbon = F.normalize(
                self.cross_geometry_ribbon_projector(ribbon_anchor_tokens), dim=-1
            )

        ribbon_outputs = None
        hybrid_gate = None
        hybrid_delta = None
        ribbon_context_tokens = None
        if self.volume_pooling_mode in {"ribbon", "hybrid_ribbon"}:
            # Preserve legacy mask diagnostics for the existing spatial-KD code,
            # without retaining an unused trainable legacy pooling graph.
            if self.volume_pooling_mode == "hybrid_ribbon":
                legacy_volume_tokens, local_tokens, roi_tokens, aux = self.volume_pool(
                    feat_map, local_ribbon_masks, roi_volume_masks
                )
            else:
                with torch.no_grad():
                    legacy_volume_tokens, local_tokens, roi_tokens, aux = self.volume_pool(
                        feat_map, local_ribbon_masks, roi_volume_masks
                    )
            required = {
                "ribbon_grid": ribbon_grid,
                "ribbon_valid_mask": ribbon_valid_mask,
                "vertex_valid_mask": vertex_valid_mask,
                "vertex_roi_index": vertex_roi_index,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"Ribbon pooling requires inputs: {', '.join(missing)}")
            ribbon_feature_map = (
                multi_scale_features[0]
                if self.ribbon_feature_stage == "high_res"
                else feat_map
            )
            ribbon_outputs = self.ribbon_sampler(
                feature_map=ribbon_feature_map,
                ribbon_grid=ribbon_grid,
                vertex_roi_index=vertex_roi_index,
                ribbon_valid_mask=ribbon_valid_mask,
                vertex_valid_mask=vertex_valid_mask,
            )
            if self.volume_pooling_mode == "hybrid_ribbon":
                legacy_context_tokens = self.hybrid_legacy_norm(legacy_volume_tokens)
                ribbon_context_tokens = self.hybrid_ribbon_projection(
                    self.hybrid_ribbon_norm(ribbon_outputs["roi_features"])
                )
                validity = ribbon_outputs["roi_valid_ratio"].unsqueeze(-1)
                hybrid_gate = torch.sigmoid(
                    self.hybrid_ribbon_gate(
                        torch.cat([legacy_context_tokens, ribbon_context_tokens, validity], dim=-1)
                    )
                )
                hybrid_delta = self.hybrid_ribbon_adapter(ribbon_context_tokens)
                # The zero-initialized adapter makes the initial hybrid model
                # exactly equal to the legacy pooling baseline.
                volume_tokens = legacy_volume_tokens + validity * hybrid_gate * hybrid_delta
            else:
                ribbon_context_tokens = self.pure_ribbon_projection(ribbon_outputs["roi_features"])
                volume_tokens = ribbon_context_tokens
        else:
            legacy_volume_tokens, local_tokens, roi_tokens, aux = self.volume_pool(
                feat_map, local_ribbon_masks, roi_volume_masks
            )
            volume_tokens = legacy_volume_tokens

        z_surface, z_volume, similarity_matrix = self.roi_clip(surface_tokens, volume_tokens)
        cross_outputs = None
        prompt_outputs = None
        clinical_feature = None
        clinical_gate = None
        if self.use_clinical:
            if clinical_data is None:
                raise ValueError("clinical_data is required when use_clinical=True")
            clinical_data = clinical_data.to(global_feat.device)
            if self.clinical_fusion_mode == "atlas_prompt":
                prompt_outputs = self.clinical_atlas_prompt(
                    global_feature=global_feat,
                    clinical_data=clinical_data,
                    volume_tokens=z_volume,
                    surface_tokens=z_surface,
                )
                z_surface = prompt_outputs["surface_tokens"]
                z_volume = prompt_outputs["volume_tokens"]
                fused_global_feat = prompt_outputs["global_feature"]
                clinical_feature = prompt_outputs["clinical_feature"]
                clinical_gate = prompt_outputs["clinical_gate"]
                global_logits = self.global_head(fused_global_feat)
        z_teacher_base = self.teacher_fusion(torch.cat([z_surface, z_volume], dim=-1))
        graph_outputs = self.roi_graph(z_teacher_base) if self.roi_graph is not None else None
        z_teacher_graph = graph_outputs["tokens"] if graph_outputs is not None else z_teacher_base
        classifier_tokens = z_teacher_graph
        if self.use_clinical:
            if self.clinical_fusion_mode == "gated":
                fused_global_feat, clinical_feature, clinical_gate = self.clinical_fusion(
                    global_feat,
                    clinical_data,
                )
                global_logits = self.global_head(fused_global_feat)
            elif self.clinical_fusion_mode == "cross":
                imaging_tokens = torch.cat(
                    [self.global_to_cross(global_feat).unsqueeze(1), z_teacher_graph], dim=1
                )
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
        teacher_relation = pairwise_cosine_matrix(z_teacher_graph)

        outputs = {
            "logits": logits,
            "global_logits": global_logits,
            "roi_logits": roi_logits,
            "roi_weight": roi_weight,
            "prob": prob,
            "surface_tokens": surface_tokens,
            "volume_tokens": volume_tokens,
            "legacy_volume_tokens": legacy_volume_tokens,
            "mri_ribbon_tokens": ribbon_context_tokens,
            "mri_ribbon_tokens_raw": ribbon_outputs["roi_features"] if ribbon_outputs is not None else None,
            "ribbon_vertex_features": ribbon_outputs["vertex_features"] if ribbon_outputs is not None else None,
            "depth_attention": ribbon_outputs["depth_attention"] if ribbon_outputs is not None else None,
            "ribbon_vertex_valid_mask": (
                ribbon_outputs["vertex_valid_mask"] if ribbon_outputs is not None else None
            ),
            "ribbon_roi_valid_mask": ribbon_outputs["roi_valid_mask"] if ribbon_outputs is not None else None,
            "ribbon_roi_vertex_count": (
                ribbon_outputs["roi_vertex_count"] if ribbon_outputs is not None else None
            ),
            "ribbon_roi_valid_ratio": (
                ribbon_outputs["roi_valid_ratio"] if ribbon_outputs is not None else None
            ),
            "hybrid_ribbon_gate": hybrid_gate,
            "hybrid_ribbon_delta": hybrid_delta,
            "ribbon_anchor_tokens": ribbon_anchor_tokens,
            "cross_geometry_surface": cross_geometry_surface,
            "cross_geometry_ribbon": cross_geometry_ribbon,
            "cross_geometry_ribbon_valid_ratio": (
                cross_geometry_outputs["roi_valid_ratio"] if cross_geometry_outputs is not None else None
            ),
            "local_tokens": local_tokens,
            "roi_tokens": roi_tokens,
            "z_surface": z_surface,
            "z_volume": z_volume,
            "z_teacher": z_teacher_graph,
            "z_teacher_base": z_teacher_base,
            "z_teacher_graph": z_teacher_graph,
            "roi_graph_attention": graph_outputs["attention"] if graph_outputs is not None else None,
            "roi_graph_mean_attention": (
                graph_outputs["mean_attention"] if graph_outputs is not None else None
            ),
            "roi_graph_anatomical_mass": (
                graph_outputs["anatomical_mass"] if graph_outputs is not None else None
            ),
            "roi_graph_homology_mass": (
                graph_outputs["homology_mass"] if graph_outputs is not None else None
            ),
            "roi_graph_identity_mass": (
                graph_outputs["identity_mass"] if graph_outputs is not None else None
            ),
            "roi_graph_remote_mass": (
                graph_outputs["remote_mass"] if graph_outputs is not None else None
            ),
            "roi_graph_attention_entropy": (
                graph_outputs["attention_entropy"] if graph_outputs is not None else None
            ),
            "roi_graph_attention_uniform_kl": (
                graph_outputs["attention_uniform_kl"] if graph_outputs is not None else None
            ),
            "roi_graph_attention_std": (
                graph_outputs["attention_std"] if graph_outputs is not None else None
            ),
            "roi_graph_boundary_correlation": (
                graph_outputs["boundary_correlation"] if graph_outputs is not None else None
            ),
            "roi_graph_uniform_anatomical_mass": (
                graph_outputs["uniform_anatomical_mass"] if graph_outputs is not None else None
            ),
            "roi_graph_uniform_homology_mass": (
                graph_outputs["uniform_homology_mass"] if graph_outputs is not None else None
            ),
            "roi_graph_residual_gate": (
                graph_outputs["residual_gate"] if graph_outputs is not None else None
            ),
            "roi_graph_output_delta_l2": (
                graph_outputs["output_delta_l2"] if graph_outputs is not None else None
            ),
            "surface_relation": surface_relation,
            "volume_relation": volume_relation,
            "relation": teacher_relation,
            "surface_prototype_logits": self.prototype_bank(z_surface),
            "volume_prototype_logits": self.prototype_bank(z_volume),
            "prototype_logits": self.prototype_bank(z_teacher_graph),
            "mask_attention": aux["local_attention"],
            "local_mask_attention": aux["local_attention"],
            "roi_mask_attention": aux["roi_attention"],
            "roi_occupancy": aux["roi_masks"],
            "similarity_matrix": similarity_matrix,
            "surface_roi_weights": surface_roi_weights,
            "multi_scale_features": multi_scale_features,
            "global_volume_feat": global_feat,
            "fused_global_feat": fused_global_feat,
            "clinical_feature": clinical_feature,
            "clinical_gate": clinical_gate,
            "cross_image_tokens": cross_outputs["image_tokens"] if cross_outputs is not None else None,
            "clinical_tokens": (
                cross_outputs["clinical_tokens"]
                if cross_outputs is not None
                else prompt_outputs["clinical_tokens"]
                if prompt_outputs is not None
                else None
            ),
            "image_to_clinical_attention": (
                cross_outputs["image_to_clinical_attention"] if cross_outputs is not None else None
            ),
            "clinical_to_image_attention": (
                cross_outputs["clinical_to_image_attention"] if cross_outputs is not None else None
            ),
            "clinical_domain_prompts": (
                prompt_outputs["clinical_domain_prompts"] if prompt_outputs is not None else None
            ),
            "clinical_domain_attention": (
                prompt_outputs["clinical_domain_attention"] if prompt_outputs is not None else None
            ),
            "roi_prompt_attention": prompt_outputs["roi_prompt_attention"] if prompt_outputs is not None else None,
        }
        return outputs
