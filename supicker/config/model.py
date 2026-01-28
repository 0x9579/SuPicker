from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .base import Config


class ConvNeXtVariant(Enum):
    """ConvNeXt model variants."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"


@dataclass
class BackboneConfig(Config):
    """ConvNeXt backbone configuration."""
    variant: ConvNeXtVariant = ConvNeXtVariant.TINY
    pretrained: bool = True
    pretrained_path: Optional[str] = None
    in_channels: int = 1  # Grayscale cryo-EM images


@dataclass
class FPNConfig(Config):
    """Feature Pyramid Network configuration."""
    in_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    out_channels: int = 256


@dataclass
class HeadConfig(Config):
    """CenterNet head configuration."""
    num_classes: int = 1
    feat_channels: int = 256
    heatmap_loss: str = "focal"
    heatmap_loss_params: dict = field(default_factory=lambda: {"alpha": 2.0, "beta": 4.0})
    size_loss: str = "l1"
    size_loss_params: dict = field(default_factory=dict)
    offset_loss: str = "l1"
    offset_loss_params: dict = field(default_factory=dict)


@dataclass
class ModelConfig(Config):
    """Complete model configuration."""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
