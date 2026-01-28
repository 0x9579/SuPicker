import pytest
import torch
from supicker.config import FPNConfig


def test_fpn_output_shapes():
    from supicker.models.neck.fpn import FPN

    config = FPNConfig(in_channels=[96, 192, 384, 768], out_channels=256)
    fpn = FPN(config)

    # Simulate backbone outputs at different scales
    features = [
        torch.randn(2, 96, 64, 64),   # C1
        torch.randn(2, 192, 32, 32),  # C2
        torch.randn(2, 384, 16, 16),  # C3
        torch.randn(2, 768, 8, 8),    # C4
    ]

    outputs = fpn(features)

    assert len(outputs) == 4
    assert outputs[0].shape == (2, 256, 64, 64)   # P2
    assert outputs[1].shape == (2, 256, 32, 32)   # P3
    assert outputs[2].shape == (2, 256, 16, 16)   # P4
    assert outputs[3].shape == (2, 256, 8, 8)     # P5
