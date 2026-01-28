import pytest
from supicker.config.inference import InferenceConfig


def test_inference_config_defaults():
    config = InferenceConfig()
    assert config.checkpoint_path == ""
    assert config.score_threshold == 0.3
    assert config.nms_enabled is True
    assert config.nms_radius == 20.0
    assert config.output_format == "star"
    assert config.batch_size == 1
