import torch
import torch.nn as nn

from supicker.config import HeadConfig


class CenterNetHead(nn.Module):
    """CenterNet detection head for heatmap, size, and offset prediction."""

    def __init__(self, config: HeadConfig, in_channels: int):
        super().__init__()
        self.config = config
        feat_channels = config.feat_channels

        # Shared feature layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # Heatmap branch (class probability at each location)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, config.num_classes, kernel_size=1),
        )

        # Size branch (width, height prediction)
        self.size_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 2, kernel_size=1),
        )

        # Offset branch (sub-pixel offset)
        self.offset_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 2, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize heatmap bias for focal loss (low initial confidence)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)  # sigmoid(-2.19) ≈ 0.1

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict heatmap, size, and offset from feature map.

        Args:
            feature: Feature map from FPN (B, C, H, W)

        Returns:
            Dictionary with 'heatmap', 'size', 'offset' tensors
        """
        x = self.shared_conv(feature)

        heatmap = self.heatmap_head(x)
        heatmap = torch.sigmoid(heatmap)

        size = self.size_head(x)
        size = torch.relu(size)  # Size must be positive

        offset = self.offset_head(x)

        return {
            "heatmap": heatmap,
            "size": size,
            "offset": offset,
        }
