import torch
import torch.nn as nn


class ClinicalGatedFusion(nn.Module):
    """Encode one clinical vector and inject it into the global MRI feature."""

    def __init__(self, input_dim, feature_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, image_feature, clinical_data):
        clinical_feature = self.encoder(clinical_data)
        gate = self.gate(torch.cat([image_feature, clinical_feature], dim=-1))
        fused_feature = self.output_norm(image_feature + gate * clinical_feature)
        return fused_feature, clinical_feature, gate


class ClinicalFeatureTokenizer(nn.Module):
    """Convert each normalized value and missing flag into a clinical token."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        if int(input_dim) % 2 != 0:
            raise ValueError("clinical input_dim must contain equal value and missing-mask halves")
        self.num_features = int(input_dim) // 2
        self.value_projection = nn.Linear(2, embed_dim)
        self.feature_embedding = nn.Parameter(torch.randn(1, self.num_features, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, clinical_data):
        values, missing = clinical_data.chunk(2, dim=-1)
        paired = torch.stack([values, missing], dim=-1)
        return self.norm(self.value_projection(paired) + self.feature_embedding)


class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, ffn_multiplier=2):
        super().__init__()
        self.image_query_norm = nn.LayerNorm(embed_dim)
        self.clinical_key_norm = nn.LayerNorm(embed_dim)
        self.clinical_query_norm = nn.LayerNorm(embed_dim)
        self.image_key_norm = nn.LayerNorm(embed_dim)
        self.image_to_clinical = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.clinical_to_image = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(embed_dim) * int(ffn_multiplier)
        self.image_ffn_norm = nn.LayerNorm(embed_dim)
        self.clinical_ffn_norm = nn.LayerNorm(embed_dim)
        self.image_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.clinical_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, image_tokens, clinical_tokens):
        image_source = image_tokens
        clinical_source = clinical_tokens
        image_delta, image_attention = self.image_to_clinical(
            query=self.image_query_norm(image_source),
            key=self.clinical_key_norm(clinical_source),
            value=self.clinical_key_norm(clinical_source),
            need_weights=True,
            average_attn_weights=True,
        )
        clinical_delta, clinical_attention = self.clinical_to_image(
            query=self.clinical_query_norm(clinical_source),
            key=self.image_key_norm(image_source),
            value=self.image_key_norm(image_source),
            need_weights=True,
            average_attn_weights=True,
        )
        image_tokens = image_source + self.residual_dropout(image_delta)
        clinical_tokens = clinical_source + self.residual_dropout(clinical_delta)
        image_tokens = image_tokens + self.image_ffn(self.image_ffn_norm(image_tokens))
        clinical_tokens = clinical_tokens + self.clinical_ffn(self.clinical_ffn_norm(clinical_tokens))
        return image_tokens, clinical_tokens, image_attention, clinical_attention


class ImagingClinicalCrossTransformer(nn.Module):
    """Bidirectionally fuse imaging tokens with per-feature clinical tokens."""

    def __init__(
        self,
        clinical_input_dim,
        embed_dim,
        max_imaging_tokens,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.clinical_tokenizer = ClinicalFeatureTokenizer(clinical_input_dim, embed_dim)
        self.imaging_position = nn.Parameter(torch.randn(1, int(max_imaging_tokens), embed_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                BidirectionalCrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.image_norm = nn.LayerNorm(embed_dim)
        self.clinical_norm = nn.LayerNorm(embed_dim)

    def forward(self, imaging_tokens, clinical_data):
        if imaging_tokens.size(1) > self.imaging_position.size(1):
            raise ValueError("imaging token count exceeds configured maximum")
        image_tokens = imaging_tokens + self.imaging_position[:, : imaging_tokens.size(1)]
        clinical_tokens = self.clinical_tokenizer(clinical_data)
        image_attention = None
        clinical_attention = None
        for layer in self.layers:
            image_tokens, clinical_tokens, image_attention, clinical_attention = layer(
                image_tokens,
                clinical_tokens,
            )
        return {
            "image_tokens": self.image_norm(image_tokens),
            "clinical_tokens": self.clinical_norm(clinical_tokens),
            "image_to_clinical_attention": image_attention,
            "clinical_to_image_attention": clinical_attention,
        }


class ClinicalAtlasPromptFusion(nn.Module):
    """Inject clinical prompts into global imaging features and ROI tokens."""

    def __init__(
        self,
        clinical_input_dim,
        feature_dim,
        roi_dim,
        num_rois,
        num_domain_prompts=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.clinical_tokenizer = ClinicalFeatureTokenizer(clinical_input_dim, roi_dim)
        self.domain_queries = nn.Parameter(torch.randn(1, int(num_domain_prompts), roi_dim) * 0.02)
        self.domain_attention = nn.MultiheadAttention(
            embed_dim=roi_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.roi_position = nn.Parameter(torch.randn(1, int(num_rois), roi_dim) * 0.02)
        self.roi_prompt_attention = nn.MultiheadAttention(
            embed_dim=roi_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.roi_update = nn.Sequential(
            nn.LayerNorm(roi_dim),
            nn.Linear(roi_dim, roi_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(roi_dim, roi_dim),
        )
        self.volume_gate = nn.Sequential(
            nn.Linear(roi_dim * 2, roi_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(roi_dim, roi_dim),
            nn.Sigmoid(),
        )
        self.surface_gate = nn.Sequential(
            nn.Linear(roi_dim * 3, roi_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(roi_dim, roi_dim),
            nn.Sigmoid(),
        )
        self.volume_norm = nn.LayerNorm(roi_dim)
        self.surface_norm = nn.LayerNorm(roi_dim)
        self.global_context = nn.Sequential(
            nn.LayerNorm(roi_dim),
            nn.Linear(roi_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(feature_dim, feature_dim),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        self.global_norm = nn.LayerNorm(feature_dim)

    def forward(self, global_feature, clinical_data, volume_tokens=None, surface_tokens=None):
        batch_size = global_feature.size(0)
        clinical_tokens = self.clinical_tokenizer(clinical_data)
        domain_queries = self.domain_queries.expand(batch_size, -1, -1)
        domain_prompts, domain_attention = self.domain_attention(
            query=domain_queries,
            key=clinical_tokens,
            value=clinical_tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        clinical_feature = self.global_context(domain_prompts.mean(dim=1))
        clinical_gate = self.global_gate(torch.cat([global_feature, clinical_feature], dim=-1))
        fused_global_feature = self.global_norm(global_feature + clinical_gate * clinical_feature)

        outputs = {
            "global_feature": fused_global_feature,
            "clinical_feature": clinical_feature,
            "clinical_gate": clinical_gate,
            "clinical_tokens": clinical_tokens,
            "clinical_domain_prompts": domain_prompts,
            "clinical_domain_attention": domain_attention,
            "roi_prompt_attention": None,
            "volume_tokens": volume_tokens,
            "surface_tokens": surface_tokens,
        }
        if volume_tokens is None:
            return outputs

        if volume_tokens.size(1) > self.roi_position.size(1):
            raise ValueError("ROI token count exceeds configured maximum")
        query_tokens = volume_tokens if surface_tokens is None else 0.5 * (volume_tokens + surface_tokens)
        query_tokens = query_tokens + self.roi_position[:, : query_tokens.size(1)]
        roi_prompt_delta, roi_prompt_attention = self.roi_prompt_attention(
            query=query_tokens,
            key=domain_prompts,
            value=domain_prompts,
            need_weights=True,
            average_attn_weights=True,
        )
        roi_update = self.roi_update(roi_prompt_delta)
        volume_gate = self.volume_gate(torch.cat([volume_tokens, roi_update], dim=-1))
        outputs["volume_tokens"] = self.volume_norm(volume_tokens + volume_gate * roi_update)
        outputs["roi_prompt_attention"] = roi_prompt_attention

        if surface_tokens is not None:
            surface_gate = self.surface_gate(torch.cat([surface_tokens, volume_tokens, roi_update], dim=-1))
            outputs["surface_tokens"] = self.surface_norm(surface_tokens + surface_gate * roi_update)

        return outputs
