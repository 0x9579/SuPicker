import pytest
import torch
from supicker.config import ModelConfig, BackboneConfig, HeadConfig, ConvNeXtVariant


def test_detector_forward():
    from supicker.models.detector import Detector

    config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False),
        head=HeadConfig(num_classes=3),
    )
    model = Detector(config)

    x = torch.randn(2, 1, 256, 256)
    outputs = model(x)

    assert "heatmap" in outputs
    assert "size" in outputs
    assert "offset" in outputs
    # Output at 1/4 resolution
    assert outputs["heatmap"].shape == (2, 3, 64, 64)
    assert outputs["size"].shape == (2, 2, 64, 64)
    assert outputs["offset"].shape == (2, 2, 64, 64)


def test_detector_different_input_sizes():
    from supicker.models.detector import Detector

    config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False),
        head=HeadConfig(num_classes=1),
    )
    model = Detector(config)

    # Test various input sizes (must be divisible by 32)
    for size in [128, 256, 512]:
        x = torch.randn(1, 1, size, size)
        outputs = model(x)
        expected_out_size = size // 4
        assert outputs["heatmap"].shape == (1, 1, expected_out_size, expected_out_size)
