import pytest
import torch
from supicker.config import HeadConfig


def test_centernet_head_output_shapes():
    from supicker.models.head.centernet import CenterNetHead

    config = HeadConfig(num_classes=3, feat_channels=256)
    head = CenterNetHead(config, in_channels=256)

    # Single scale feature map (P2 from FPN)
    feature = torch.randn(2, 256, 64, 64)
    outputs = head(feature)

    assert "heatmap" in outputs
    assert "size" in outputs
    assert "offset" in outputs
    assert outputs["heatmap"].shape == (2, 3, 64, 64)   # num_classes channels
    assert outputs["size"].shape == (2, 2, 64, 64)      # width, height
    assert outputs["offset"].shape == (2, 2, 64, 64)    # offset_x, offset_y


def test_centernet_head_heatmap_activation():
    from supicker.models.head.centernet import CenterNetHead

    config = HeadConfig(num_classes=1)
    head = CenterNetHead(config, in_channels=256)

    feature = torch.randn(2, 256, 32, 32)
    outputs = head(feature)

    # Heatmap should be in [0, 1] after sigmoid
    assert outputs["heatmap"].min() >= 0.0
    assert outputs["heatmap"].max() <= 1.0
