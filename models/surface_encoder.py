import torch
import torch.nn as nn

from data.ico_utils import get_ico_topology, infer_ico_level


class OneRingConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, one_ring_index):
        super().__init__()
        self.register_buffer("one_ring_index", torch.as_tensor(one_ring_index, dtype=torch.long), persistent=False)
        self.linear = nn.Linear(in_channels * int(self.one_ring_index.shape[1]), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        support = x[:, self.one_ring_index]
        support = support.reshape(x.shape[0], x.shape[1], -1)
        return self.linear(support)


class SphericalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, one_ring_index):
        super().__init__()
        self.conv = OneRingConv(in_channels, out_channels, one_ring_index)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class HemisphereEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, one_ring_index):
        super().__init__()
        self.block1 = SphericalConvBlock(in_channels, hidden_dim, one_ring_index)
        self.block2 = SphericalConvBlock(hidden_dim, hidden_dim, one_ring_index)
        self.block3 = SphericalConvBlock(hidden_dim, out_dim, one_ring_index)
        self.skip = nn.Linear(in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x + residual


class SurfaceEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim, num_vertices):
        super().__init__()
        per_hemi = max(1, int(num_vertices) // 2)
        level = infer_ico_level(per_hemi)
        _, one_ring_index = get_ico_topology(level)
        self.per_hemi = per_hemi
        self.left_encoder = HemisphereEncoder(in_channels, hidden_dim, out_dim, one_ring_index)
        self.right_encoder = HemisphereEncoder(in_channels, hidden_dim, out_dim, one_ring_index)

    def forward(self, surface_data):
        left = surface_data[:, : self.per_hemi]
        right = surface_data[:, self.per_hemi : self.per_hemi * 2]
        left_feat = self.left_encoder(left)
        right_feat = self.right_encoder(right)
        return torch.cat([left_feat, right_feat], dim=1)
