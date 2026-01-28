import torch
import torch.nn as nn
import torch.nn.functional as F

from supicker.config import FPNConfig


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, config: FPNConfig):
        super().__init__()
        self.config = config
        in_channels = config.in_channels
        out_channels = config.out_channels

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Output convolutions (3x3 conv to smooth features)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse multi-scale features with top-down pathway.

        Args:
            features: List of feature maps [C1, C2, C3, C4] from backbone

        Returns:
            List of fused feature maps [P2, P3, P4, P5]
        """
        # Apply lateral convolutions
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return outputs
