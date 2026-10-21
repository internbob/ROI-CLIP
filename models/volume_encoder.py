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


class MedicalNetBasicBlock3D(nn.Module):
    """BasicBlock matching Tencent MedicalNet's 3D ResNet checkpoints."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNet10Encoder3D(nn.Module):
    """Student-only Med3D ResNet10 adapter with the CortexBridge encoder API.

    MedicalNet keeps layer2-layer4 at output stride 8 using dilated
    convolutions.  Layer2 therefore replaces the SFCN ROI feature map while
    the layer4 global average is projected to the configured feature size.
    The segmentation decoder from the original repository is intentionally
    excluded.
    """

    def __init__(
        self,
        in_channels=1,
        out_dim=128,
        pretrained_path="",
        freeze_bn_stats=False,
    ):
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("MedicalNet ResNet10 checkpoints require one input channel")
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(MedicalNetBasicBlock3D(64, 64))
        self.layer2 = nn.Sequential(MedicalNetBasicBlock3D(64, 128, stride=2))
        self.layer3 = nn.Sequential(MedicalNetBasicBlock3D(128, 256, dilation=2))
        self.layer4 = nn.Sequential(MedicalNetBasicBlock3D(256, 512, dilation=4))
        self.roi_projection = nn.Identity() if int(out_dim) == 128 else nn.Conv3d(128, out_dim, 1)
        self.global_projection = nn.Linear(512, out_dim)
        self.out_dim = int(out_dim)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.pretrained_report = {
            "loaded": False,
            "path": "",
            "loaded_tensors": 0,
            "missing_backbone_keys": [],
            "unexpected_keys": [],
        }
        self._initialize_weights()
        if str(pretrained_path or ""):
            self.load_medicalnet_pretrained(str(pretrained_path))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_medicalnet_pretrained(self, path):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility.
            checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for key, value in state.items():
            key = str(key)
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("conv_seg."):
                continue
            cleaned[key] = value

        incompatible = self.load_state_dict(cleaned, strict=False)
        backbone_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
        missing_backbone = [
            key for key in incompatible.missing_keys if key.startswith(backbone_prefixes)
        ]
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("conv_seg.")]
        if missing_backbone or unexpected:
            raise RuntimeError(
                "MedicalNet checkpoint is incompatible: "
                f"missing_backbone_keys={missing_backbone}, unexpected_keys={unexpected}"
            )
        loaded_tensors = sum(key.startswith(backbone_prefixes) for key in cleaned)
        if loaded_tensors == 0:
            raise RuntimeError(f"No MedicalNet backbone tensors were loaded from {path}")
        self.pretrained_report = {
            "loaded": True,
            "path": path,
            "loaded_tensors": int(loaded_tensors),
            "missing_backbone_keys": missing_backbone,
            "unexpected_keys": unexpected,
        }
        return self.pretrained_report

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        return self

    def forward(self, x):
        high_res = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(high_res)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        roi_feat_map = self.roi_projection(layer2)
        global_feat = self.global_projection(layer4.flatten(2).mean(dim=-1))
        return roi_feat_map, [high_res, layer1, roi_feat_map, layer3, layer4], global_feat


def build_student_volume_encoder(
    name,
    in_channels=1,
    base_channels=32,
    out_dim=128,
    norm="batch",
    pretrained_path="",
    freeze_bn_stats=False,
):
    name = str(name or "sfcn").lower()
    if name == "sfcn":
        if str(pretrained_path or ""):
            raise ValueError("pretrained_path is only supported for a pretrained Student backbone")
        return SFCNVolumeEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            out_dim=out_dim,
            norm=norm,
        )
    if name in {"medicalnet_resnet10", "med3d_resnet10"}:
        return MedicalNetResNet10Encoder3D(
            in_channels=in_channels,
            out_dim=out_dim,
            pretrained_path=pretrained_path,
            freeze_bn_stats=freeze_bn_stats,
        )
    raise ValueError(f"Unsupported Student volume encoder: {name}")
