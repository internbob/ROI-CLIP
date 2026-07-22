import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseasePrototypeBank(nn.Module):
    def __init__(self, num_classes, num_rois, proto_dim, temperature=0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_rois, proto_dim) * 0.02)
        self.temperature = float(temperature)

    def forward(self, tokens):
        tokens = F.normalize(tokens, dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        logits = torch.einsum("bkd,ckd->bkc", tokens, protos) / max(self.temperature, 1e-6)
        return logits

