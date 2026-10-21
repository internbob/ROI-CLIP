import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import GlobalClassifierHead, ROIClassifierHead
from .clinical_encoder import ClinicalAtlasPromptFusion, ClinicalGatedFusion, ImagingClinicalCrossTransformer
from .disease_aware_graph import DiseaseAwareROIGraph
from .prototypes import DiseasePrototypeBank
from .relation import pairwise_cosine_matrix
from .volume_encoder import build_student_volume_encoder


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
        topology_query_pe=None,
        topology_query_trainable_residual=True,
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
        volume_encoder_name="sfcn",
        volume_encoder_pretrained_path="",
        volume_encoder_freeze_bn_stats=False,
        roi_classification_enabled=True,
        roi_classifier_use_raw_tokens=False,
    ):
        super().__init__()
        self.use_clinical = bool(use_clinical)
        self.clinical_fusion_mode = str(clinical_fusion_mode).lower()
        if self.clinical_fusion_mode not in {"gated", "cross", "atlas_prompt"}:
            raise ValueError(f"Unsupported clinical fusion mode: {clinical_fusion_mode}")
        self.volume_encoder_name = str(volume_encoder_name).lower()
        self.roi_classification_enabled = bool(roi_classification_enabled)
        self.roi_classifier_use_raw_tokens = bool(roi_classifier_use_raw_tokens)
        self.volume_encoder = build_student_volume_encoder(
            name=self.volume_encoder_name,
            in_channels=1,
            base_channels=base_channels,
            out_dim=feature_dim,
            norm=volume_norm,
            pretrained_path=volume_encoder_pretrained_path,
            freeze_bn_stats=volume_encoder_freeze_bn_stats,
        )
        if topology_query_pe is None:
            self.register_buffer("topology_query_pe", None)
            self.topology_query_proj = None
            self.roi_queries = nn.Parameter(torch.randn(1, num_rois, feature_dim) * 0.02)
            self.roi_query_delta = None
        else:
            topology_query_pe = torch.as_tensor(topology_query_pe, dtype=torch.float32)
            if topology_query_pe.dim() != 2 or int(topology_query_pe.shape[0]) != int(num_rois):
                raise ValueError(
                    f"topology_query_pe must have shape [{num_rois}, P], got "
                    f"{tuple(topology_query_pe.shape)}"
                )
            self.register_buffer("topology_query_pe", topology_query_pe.unsqueeze(0))
            self.topology_query_proj = nn.Linear(int(topology_query_pe.shape[1]), feature_dim, bias=False)
            self.roi_queries = None
            self.roi_query_delta = nn.Parameter(
                torch.zeros(1, num_rois, feature_dim),
                requires_grad=bool(topology_query_trainable_residual),
            )
        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(feature_dim, feature_dim)
        self.to_v = nn.Linear(feature_dim, feature_dim)
        self.scale = 1.0 / math.sqrt(float(feature_dim))
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.projector = nn.Linear(feature_dim, proj_dim)
        # Graph construction must not change initialization of the shared
        # prototype/classifier heads.  This keeps graph-on/off ablations
        # matched under the same seed.
        rng_state = torch.random.get_rng_state()
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
            if bool(roi_graph_enabled)
            else None
        )
        torch.random.set_rng_state(rng_state)
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

    def forward(self, image, clinical_data=None):
        feat_map, multi_scale_features, global_feat = self.volume_encoder(image)
        bsz, channels, h, w, d = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)

        if self.topology_query_pe is None:
            queries = self.roi_queries.expand(bsz, -1, -1)
        else:
            queries = self.topology_query_proj(self.topology_query_pe)
            if self.roi_query_delta is not None:
                queries = queries + self.roi_query_delta
            queries = queries.expand(bsz, -1, -1)
        q = self.to_q(queries)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        spatial_logits = (q @ k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(spatial_logits, dim=-1)
        student_tokens = self.out_proj(attn @ v)
        z_student_base_raw = self.projector(student_tokens)
        z_student_base = F.normalize(z_student_base_raw, dim=-1)
        graph_input = z_student_base_raw if self.roi_classifier_use_raw_tokens else z_student_base
        graph_outputs = self.roi_graph(graph_input) if self.roi_graph is not None else None
        z_student_graph_raw = graph_outputs["tokens"] if graph_outputs is not None else graph_input
        # Preserve the historical representation exactly by default.  In the
        # raw-token ablation, classification retains magnitude while the KD,
        # relation, and prototype interfaces remain normalized.
        z_student_graph = (
            F.normalize(z_student_graph_raw, dim=-1)
            if self.roi_classifier_use_raw_tokens
            else z_student_graph_raw
        )
        z_student = z_student_graph
        classifier_tokens = z_student_graph_raw
        cross_outputs = None
        prompt_outputs = None
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
            elif self.clinical_fusion_mode == "cross":
                imaging_tokens = torch.cat([self.global_to_cross(global_feat).unsqueeze(1), z_student], dim=1)
                cross_outputs = self.clinical_cross_fusion(imaging_tokens, clinical_data)
                fused_global_feat = cross_outputs["image_tokens"][:, 0]
                classifier_tokens = cross_outputs["image_tokens"][:, 1:]
                global_logits = self.cross_global_head(fused_global_feat)
            else:
                prompt_outputs = self.clinical_atlas_prompt(
                    global_feature=global_feat,
                    clinical_data=clinical_data,
                    volume_tokens=z_student,
                )
                fused_global_feat = prompt_outputs["global_feature"]
                clinical_feature = prompt_outputs["clinical_feature"]
                clinical_gate = prompt_outputs["clinical_gate"]
                z_student = prompt_outputs["volume_tokens"]
                classifier_tokens = z_student
                global_logits = self.global_head(fused_global_feat)
        else:
            fused_global_feat = global_feat
            global_logits = self.global_head(global_feat)
        roi_logits = self.classifier(classifier_tokens)
        roi_weight = torch.sigmoid(self.roi_logit_scale)
        logits = (
            global_logits + roi_weight * roi_logits
            if self.roi_classification_enabled
            else global_logits
        )
        prob = torch.softmax(logits, dim=-1)
        attention_maps = attn.reshape(bsz, queries.size(1), h, w, d)
        spatial_logits = spatial_logits.reshape(bsz, queries.size(1), h, w, d)

        return {
            "logits": logits,
            "global_logits": global_logits,
            "roi_logits": roi_logits,
            "roi_weight": roi_weight,
            "prob": prob,
            "student_tokens": student_tokens,
            "z_student": z_student,
            "z_student_base": z_student_base,
            "z_student_base_raw": z_student_base_raw,
            "z_student_graph": z_student_graph,
            "z_student_graph_raw": z_student_graph_raw,
            "roi_graph_attention": graph_outputs["attention"] if graph_outputs is not None else None,
            "roi_graph_mean_attention": (
                graph_outputs["mean_attention"] if graph_outputs is not None else None
            ),
            "attention_maps": attention_maps,
            "spatial_logits": spatial_logits,
            "relation": pairwise_cosine_matrix(z_student_graph),
            "prototype_logits": self.prototype_bank(z_student_graph),
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
