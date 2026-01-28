import pytest
import torch
from supicker.models.backbone.convnext import ConvNeXtBlock, LayerNorm2d
from supicker.config import BackboneConfig, ConvNeXtVariant


def test_layer_norm_2d_shape():
    norm = LayerNorm2d(96)
    x = torch.randn(2, 96, 32, 32)
    out = norm(x)
    assert out.shape == (2, 96, 32, 32)


def test_convnext_block_shape():
    block = ConvNeXtBlock(dim=96)
    x = torch.randn(2, 96, 32, 32)
    out = block(x)
    assert out.shape == (2, 96, 32, 32)


def test_convnext_tiny_output_shapes():
    from supicker.models.backbone.convnext import ConvNeXt

    config = BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    model = ConvNeXt(config)
    x = torch.randn(2, 1, 256, 256)
    features = model(x)

    assert len(features) == 4
    assert features[0].shape == (2, 96, 64, 64)    # C1: H/4
    assert features[1].shape == (2, 192, 32, 32)   # C2: H/8
    assert features[2].shape == (2, 384, 16, 16)   # C3: H/16
    assert features[3].shape == (2, 768, 8, 8)     # C4: H/32


def test_convnext_single_channel_input():
    from supicker.models.backbone.convnext import ConvNeXt

    config = BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False, in_channels=1)
    model = ConvNeXt(config)
    x = torch.randn(2, 1, 128, 128)
    features = model(x)
    assert features[0].shape[1] == 96
