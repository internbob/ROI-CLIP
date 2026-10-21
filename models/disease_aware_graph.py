import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyBiasedROIAttentionLayer(nn.Module):
    """Full ROI self-attention with optional soft anatomical biases."""

    def __init__(
        self,
        feature_dim,
        num_heads,
        dropout=0.1,
        residual_gate_init=0.1,
        use_boundary_strength=False,
        use_homology=False,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        num_heads = int(num_heads)
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_boundary_strength = bool(use_boundary_strength)
        self.use_homology = bool(use_homology)

        self.norm = nn.LayerNorm(feature_dim)
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.boundary_scale = nn.Parameter(torch.ones(num_heads))
        self.homology_scale = nn.Parameter(torch.ones(num_heads))

        gate = min(max(float(residual_gate_init), 1e-4), 1.0 - 1e-4)
        self.residual_gate_logit = nn.Parameter(
            torch.tensor(math.log(gate / (1.0 - gate)), dtype=torch.float32)
        )

    def forward(self, tokens, boundary_strength, homology):
        batch_size, num_rois, _ = tokens.shape
        normalized = self.norm(tokens)
        qkv = self.qkv(normalized).reshape(
            batch_size, num_rois, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.use_boundary_strength:
            scores = scores + self.boundary_scale.view(1, -1, 1, 1) * boundary_strength.view(
                1, 1, num_rois, num_rois
            )
        if self.use_homology:
            scores = scores + self.homology_scale.view(1, -1, 1, 1) * homology.view(
                1, 1, num_rois, num_rois
            )

        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(attention), value)
        attended = attended.transpose(1, 2).reshape(batch_size, num_rois, self.feature_dim)
        delta = self.out_proj(attended)
        gate = torch.sigmoid(self.residual_gate_logit)
        output = tokens + gate * self.dropout(delta)
        return output, attention, delta, gate


class DiseaseAwareROIGraph(nn.Module):
    """Disease-sensitive ROI reasoning with template anatomy as a soft prior."""

    def __init__(
        self,
        feature_dim,
        num_rois,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        residual_gate_init=0.1,
        use_boundary_strength=False,
        use_homology=False,
        anatomy_adjacency=None,
        boundary_strength=None,
        evaluation_boundary_strength=None,
        homology=None,
        evaluation_homology=None,
    ):
        super().__init__()
        self.num_rois = int(num_rois)
        shape = (self.num_rois, self.num_rois)

        def topology_tensor(value, name):
            tensor = torch.zeros(shape, dtype=torch.float32) if value is None else torch.as_tensor(value).float()
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
            return tensor

        anatomy = topology_tensor(anatomy_adjacency, "anatomy_adjacency")
        boundary = topology_tensor(boundary_strength, "boundary_strength")
        evaluation_boundary = topology_tensor(
            evaluation_boundary_strength if evaluation_boundary_strength is not None else boundary,
            "evaluation_boundary_strength",
        )
        homolog = topology_tensor(homology, "homology")
        evaluation_homolog = topology_tensor(
            evaluation_homology if evaluation_homology is not None else homolog,
            "evaluation_homology",
        )
        self.register_buffer("anatomy_adjacency", anatomy, persistent=True)
        self.register_buffer("boundary_strength", boundary, persistent=True)
        self.register_buffer("evaluation_boundary_strength", evaluation_boundary, persistent=True)
        self.register_buffer("homology", homolog, persistent=True)
        self.register_buffer("evaluation_homology", evaluation_homolog, persistent=True)
        self.layers = nn.ModuleList(
            [
                AnatomyBiasedROIAttentionLayer(
                    feature_dim=feature_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    residual_gate_init=residual_gate_init,
                    use_boundary_strength=use_boundary_strength,
                    use_homology=use_homology,
                )
                for _ in range(int(num_layers))
            ]
        )

    def _diagnostics(self, attention, input_tokens, output_tokens, gates):
        # Average heads but retain subject-specific attention before aggregation.
        mean_attention = attention.mean(dim=1)
        eye = torch.eye(self.num_rois, device=attention.device, dtype=attention.dtype)
        anatomy = (self.anatomy_adjacency > 0).to(attention.dtype)
        homology = (self.evaluation_homology > 0).to(attention.dtype)
        remote = (1.0 - torch.clamp(eye + anatomy + homology, max=1.0)).to(attention.dtype)
        entropy = -(mean_attention.clamp_min(1e-8) * mean_attention.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, self.num_rois))
        uniform_kl = (
            mean_attention.clamp_min(1e-8)
            * (mean_attention.clamp_min(1e-8).log() + math.log(self.num_rois))
        ).sum(dim=-1)
        group_attention = mean_attention.mean(dim=0)
        off_diagonal = ~torch.eye(self.num_rois, device=attention.device, dtype=torch.bool)
        attention_flat = group_attention[off_diagonal]
        boundary_flat = self.evaluation_boundary_strength[off_diagonal].to(attention.dtype)
        attention_centered = attention_flat - attention_flat.mean()
        boundary_centered = boundary_flat - boundary_flat.mean()
        correlation = (attention_centered * boundary_centered).mean() / (
            attention_centered.square().mean().sqrt()
            * boundary_centered.square().mean().sqrt()
        ).clamp_min(1e-8)
        return {
            "attention": attention,
            "mean_attention": mean_attention,
            "anatomical_mass": (mean_attention * anatomy).sum(dim=-1).mean(),
            "homology_mass": (mean_attention * homology).sum(dim=-1).mean(),
            "identity_mass": (mean_attention * eye).sum(dim=-1).mean(),
            "remote_mass": (mean_attention * remote).sum(dim=-1).mean(),
            "attention_entropy": entropy.mean(),
            "attention_uniform_kl": uniform_kl.mean(),
            "attention_std": mean_attention.std(unbiased=False),
            "boundary_correlation": correlation,
            "uniform_anatomical_mass": anatomy.sum(dim=-1).mean() / self.num_rois,
            "uniform_homology_mass": homology.sum(dim=-1).mean() / self.num_rois,
            "residual_gate": torch.stack(gates).mean(),
            "output_delta_l2": (output_tokens - input_tokens).norm(dim=-1).mean(),
        }

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.size(1) != self.num_rois:
            raise ValueError(
                f"Expected ROI tokens [B,{self.num_rois},D], got {tuple(tokens.shape)}"
            )
        initial = tokens
        attentions = []
        gates = []
        for layer in self.layers:
            tokens, attention, _, gate = layer(
                tokens,
                boundary_strength=self.boundary_strength,
                homology=self.homology,
            )
            attentions.append(attention)
            gates.append(gate)
        attention = torch.stack(attentions, dim=0).mean(dim=0)
        outputs = self._diagnostics(attention, initial, tokens, gates)
        outputs["tokens"] = tokens
        return outputs
