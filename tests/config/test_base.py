import pytest
from supicker.config.base import Config


def test_config_to_dict():
    config = Config()
    result = config.to_dict()
    assert isinstance(result, dict)


def test_config_from_dict():
    data = {"key": "value"}
    config = Config.from_dict(data)
    assert isinstance(config, Config)
