import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from supicker.config import BackboneConfig, ConvNeXtVariant


# Pretrained weights URLs from torchvision
CONVNEXT_PRETRAINED_URLS = {
    ConvNeXtVariant.TINY: "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    ConvNeXtVariant.SMALL: "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    ConvNeXtVariant.BASE: "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
}


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

        # Load pretrained weights if requested
        if config.pretrained:
            self._load_pretrained(config.pretrained_path)

    def _load_pretrained(self, pretrained_path: Optional[str] = None) -> None:
        """Load pretrained weights from URL or local path.

        Args:
            pretrained_path: Optional local path to weights file.
                           If None, downloads from PyTorch hub.
        """
        if pretrained_path is not None:
            # Load from local path
            state_dict = torch.load(pretrained_path, map_location="cpu")
        else:
            # Download from PyTorch hub
            url = CONVNEXT_PRETRAINED_URLS.get(self.config.variant)
            if url is None:
                print(f"No pretrained weights available for {self.config.variant}")
                return
            state_dict = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", progress=True
            )

        # Map pretrained keys to our model keys
        mapped_state_dict = self._map_pretrained_keys(state_dict)

        # Adapt input channels if needed (RGB -> grayscale)
        if self.config.in_channels != 3:
            mapped_state_dict = self._adapt_input_channels(mapped_state_dict)

        # Load with strict=False to handle missing/extra keys
        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)

        if missing:
            print(f"Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading pretrained weights: {unexpected}")

    def _adapt_input_channels(self, state_dict: dict) -> dict:
        """Adapt pretrained RGB weights to grayscale input.

        Converts 3-channel stem weights to 1-channel by averaging.

        Args:
            state_dict: Pretrained state dict with RGB weights

        Returns:
            State dict with adapted input channel weights
        """
        stem_key = "stem.0.weight"
        if stem_key in state_dict:
            rgb_weight = state_dict[stem_key]
            # Average across RGB channels: (out_ch, 3, H, W) -> (out_ch, 1, H, W)
            gray_weight = rgb_weight.mean(dim=1, keepdim=True)
            # Repeat for multi-channel input if needed
            if self.config.in_channels > 1:
                gray_weight = gray_weight.repeat(1, self.config.in_channels, 1, 1)
            state_dict[stem_key] = gray_weight

        return state_dict

    def _map_pretrained_keys(self, state_dict: dict) -> dict:
        """Map torchvision ConvNeXt keys to SuPicker keys.

        Args:
            state_dict: Pretrained state dict from torchvision

        Returns:
            State dict with mapped keys
        """
        mapped = {}
        key_mapping = {
            # Stem mappings
            "features.0.0.weight": "stem.0.weight",
            "features.0.0.bias": "stem.0.bias",
            "features.0.1.weight": "stem.1.weight",
            "features.0.1.bias": "stem.1.bias",
        }

        # Stage and block mappings
        # torchvision: features.{stage_idx}.{block_idx}.{layer}
        # Our model: stages.{stage_idx}.{block_idx}.{layer}
        # Downsample: features.{2,4,6}.{0,1} -> downsamples.{0,1,2}.{0,1}
        stage_indices = [1, 3, 5, 7]  # torchvision stage indices
        downsample_indices = [2, 4, 6]  # torchvision downsample indices

        for key, value in state_dict.items():
            if key in key_mapping:
                mapped[key_mapping[key]] = value
                continue

            # Skip classifier head
            if key.startswith("classifier"):
                continue

            # Map stage blocks
            matched = False
            for our_idx, tv_idx in enumerate(stage_indices):
                prefix = f"features.{tv_idx}."
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    # Map block layer names
                    suffix = suffix.replace(".block.0.", ".dwconv.")
                    suffix = suffix.replace(".block.1.", ".norm.")
                    suffix = suffix.replace(".block.3.", ".pwconv1.")
                    suffix = suffix.replace(".block.5.", ".pwconv2.")
                    suffix = suffix.replace(".layer_scale", ".gamma")
                    new_key = f"stages.{our_idx}.{suffix}"
                    mapped[new_key] = value
                    matched = True
                    break

            if matched:
                continue

            # Map downsample layers
            for our_idx, tv_idx in enumerate(downsample_indices):
                prefix = f"features.{tv_idx}."
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    new_key = f"downsamples.{our_idx}.{suffix}"
                    mapped[new_key] = value
                    break

        return mapped

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
