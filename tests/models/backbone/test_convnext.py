import pytest
import torch
from supicker.models.backbone.convnext import ConvNeXtBlock, LayerNorm2d


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
