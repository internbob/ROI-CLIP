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
