import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VolumeEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, out_dim=128):
        super().__init__()
        self.stem = ConvBlock3D(in_channels, base_channels, pool=True)
        self.block1 = ConvBlock3D(base_channels, base_channels * 2, pool=True)
        self.block2 = ConvBlock3D(base_channels * 2, base_channels * 4, pool=True)
        self.block3 = ConvBlock3D(base_channels * 4, out_dim, pool=False)
        self.out_dim = out_dim

    def forward(self, x):
        multi_scale = []
        x = self.stem(x)
        multi_scale.append(x)
        x = self.block1(x)
        multi_scale.append(x)
        x = self.block2(x)
        multi_scale.append(x)
        feat_map = self.block3(x)
        global_feat = feat_map.flatten(2).mean(dim=-1)
        return feat_map, multi_scale, global_feat


def _normalization_3d(kind, channels):
    kind = str(kind).lower()
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind != "group":
        raise ValueError(f"Unsupported 3D normalization: {kind}")
    groups = min(8, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SFCNSingleConvBlock3D(nn.Module):
    """SFCN-style single-convolution block with configurable normalization."""

    def __init__(self, in_channels, out_channels, pool=True, kernel_size=3, padding=1, norm="batch"):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            _normalization_3d(norm, out_channels),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SFCNVolumeEncoder3D(nn.Module):
    """Six-layer SFCN with an intermediate high-resolution ROI feature map."""

    def __init__(self, in_channels=1, base_channels=32, out_dim=128, norm="batch"):
        super().__init__()
        self.norm = str(norm).lower()
        self.stem = SFCNSingleConvBlock3D(in_channels, base_channels, pool=True, norm=self.norm)
        self.block1 = SFCNSingleConvBlock3D(base_channels, base_channels * 2, pool=True, norm=self.norm)
        self.block2 = SFCNSingleConvBlock3D(base_channels * 2, out_dim, pool=True, norm=self.norm)
        self.block3 = SFCNSingleConvBlock3D(out_dim, base_channels * 8, pool=True, norm=self.norm)
        self.block4 = SFCNSingleConvBlock3D(base_channels * 8, base_channels * 8, pool=True, norm=self.norm)
        self.block5 = SFCNSingleConvBlock3D(
            base_channels * 8,
            base_channels * 2,
            pool=False,
            kernel_size=1,
            padding=0,
            norm=self.norm,
        )
        self.global_projection = nn.Linear(base_channels * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        multi_scale = []
        x = self.stem(x)
        multi_scale.append(x)
        x = self.block1(x)
        multi_scale.append(x)
        roi_feat_map = self.block2(x)
        multi_scale.append(roi_feat_map)
        x = self.block3(roi_feat_map)
        multi_scale.append(x)
        x = self.block4(x)
        multi_scale.append(x)
        x = self.block5(x)
        multi_scale.append(x)
        global_feat = self.global_projection(x.flatten(2).mean(dim=-1))
        return roi_feat_map, multi_scale, global_feat
