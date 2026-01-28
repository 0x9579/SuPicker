import torch
import torch.nn as nn
import torch.nn.functional as F

from supicker.config import BackboneConfig, ConvNeXtVariant


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise conv and inverted bottleneck."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) for regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# Model configurations for each variant
CONVNEXT_CONFIGS = {
    ConvNeXtVariant.TINY: {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    ConvNeXtVariant.SMALL: {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    ConvNeXtVariant.BASE: {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
}


class ConvNeXt(nn.Module):
    """ConvNeXt backbone for feature extraction."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        variant_config = CONVNEXT_CONFIGS[config.variant]
        depths = variant_config["depths"]
        dims = variant_config["dims"]

        # Stem: patchify with 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Build stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # Downsampling layers between stages
        self.downsamples = nn.ModuleList()
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsamples.append(downsample)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of feature maps [C1, C2, C3, C4] at strides [4, 8, 16, 32]
        """
        features = []
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < 3:
                x = self.downsamples[i](x)

        return features
