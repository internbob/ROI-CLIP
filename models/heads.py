import torch
import torch.nn as nn


class GlobalClassifierHead(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, global_feature):
        return self.net(global_feature)


class ROIClassifierHead(nn.Module):
    def __init__(self, token_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, roi_tokens):
        pooled_mean = roi_tokens.mean(dim=1)
        pooled_max = roi_tokens.max(dim=1).values
        return self.net(torch.cat([pooled_mean, pooled_max], dim=-1))
