import pytest
from supicker.config.model import (
    ConvNeXtVariant,
    BackboneConfig,
    FPNConfig,
    HeadConfig,
    ModelConfig,
)


def test_convnext_variant_values():
    assert ConvNeXtVariant.TINY.value == "tiny"
    assert ConvNeXtVariant.SMALL.value == "small"
    assert ConvNeXtVariant.BASE.value == "base"


def test_backbone_config_defaults():
    config = BackboneConfig()
    assert config.variant == ConvNeXtVariant.TINY
    assert config.pretrained is True
    assert config.pretrained_path is None
    assert config.in_channels == 1


def test_fpn_config_defaults():
    config = FPNConfig()
    assert config.in_channels == [96, 192, 384, 768]
    assert config.out_channels == 256


def test_head_config_defaults():
    config = HeadConfig()
    assert config.num_classes == 1
    assert config.feat_channels == 256


def test_model_config_composition():
    config = ModelConfig()
    assert isinstance(config.backbone, BackboneConfig)
    assert isinstance(config.fpn, FPNConfig)
    assert isinstance(config.head, HeadConfig)
