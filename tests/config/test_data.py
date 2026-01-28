import pytest
from supicker.config.data import AugmentationConfig, DataConfig


def test_augmentation_config_defaults():
    config = AugmentationConfig()
    assert config.horizontal_flip is True
    assert config.vertical_flip is True
    assert config.rotation_90 is True
    assert config.random_rotation is True
    assert config.rotation_range == (-180.0, 180.0)
    assert config.gaussian_noise is True
    assert config.ctf_simulation is False


def test_data_config_defaults():
    config = DataConfig()
    assert config.train_image_dir == ""
    assert config.train_star_file == ""
    assert isinstance(config.augmentation, AugmentationConfig)
