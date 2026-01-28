import pytest
from supicker.config.training import LossConfig, TrainingConfig


def test_loss_config_defaults():
    config = LossConfig()
    assert config.heatmap_type == "focal"
    assert config.heatmap_weight == 1.0
    assert config.focal_alpha == 2.0
    assert config.focal_beta == 4.0
    assert config.size_type == "l1"
    assert config.size_weight == 0.1


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.batch_size == 8
    assert config.epochs == 100
    assert config.optimizer == "adamw"
    assert config.learning_rate == 1e-4
    assert config.scheduler == "cosine"
