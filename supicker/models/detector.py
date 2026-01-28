import torch
import torch.nn as nn

from supicker.config import ModelConfig
from supicker.models.backbone.convnext import ConvNeXt
from supicker.models.neck.fpn import FPN
from supicker.models.head.centernet import CenterNetHead


class Detector(nn.Module):
    """Complete particle detector: ConvNeXt + FPN + CenterNet."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build backbone
        self.backbone = ConvNeXt(config.backbone)

        # Build FPN neck
        self.neck = FPN(config.fpn)

        # Build detection head (uses P2 feature at 1/4 resolution)
        self.head = CenterNetHead(config.head, in_channels=config.fpn.out_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the detector.

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            Dictionary with 'heatmap', 'size', 'offset' predictions
        """
        # Extract multi-scale features
        features = self.backbone(x)

        # Fuse features with FPN
        fpn_features = self.neck(features)

        # Detect on P2 (highest resolution FPN output)
        outputs = self.head(fpn_features[0])

        return outputs
