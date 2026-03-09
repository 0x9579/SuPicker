import pytest
import torch
import tempfile
from pathlib import Path


def test_predictor_extract_peaks():
    from supicker.engine.predictor import Predictor
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, InferenceConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    inference_config = InferenceConfig(score_threshold=0.1)

    model = Detector(model_config)
    predictor = Predictor(model=model, config=inference_config)

    # Create dummy heatmap with peaks
    heatmap = torch.zeros(1, 1, 32, 32)
    heatmap[0, 0, 10, 10] = 0.9
    heatmap[0, 0, 20, 20] = 0.8

    peaks = predictor.extract_peaks(heatmap)

    assert len(peaks) == 2
    assert abs(peaks[0]["score"] - 0.9) < 0.01
    assert peaks[0]["x"] == 10
    assert peaks[0]["y"] == 10


def test_predictor_nms():
    from supicker.engine.predictor import Predictor
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, InferenceConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    inference_config = InferenceConfig(nms_enabled=True, nms_radius=5.0)

    model = Detector(model_config)
    predictor = Predictor(model=model, config=inference_config)

    # Two peaks very close together - should be merged by NMS
    particles = [
        {"x": 10.0, "y": 10.0, "score": 0.9, "class_id": 0},
        {"x": 12.0, "y": 12.0, "score": 0.8, "class_id": 0},  # Within NMS radius
        {"x": 50.0, "y": 50.0, "score": 0.7, "class_id": 0},  # Far away
    ]

    filtered = predictor.apply_nms(particles)

    # Should keep highest score from first cluster and the far one
    assert len(filtered) == 2
    assert filtered[0]["score"] == 0.9


def test_predictor_nms_keeps_different_classes():
    from supicker.engine.predictor import Predictor

    particles = [
        {"x": 10.0, "y": 10.0, "score": 0.9, "class_id": 0, "batch_idx": 0},
        {"x": 11.0, "y": 11.0, "score": 0.8, "class_id": 1, "batch_idx": 0},
    ]

    filtered = Predictor.apply_nms_to_particles(particles, radius=5.0)

    assert len(filtered) == 2


def test_predictor_predict():
    from supicker.engine.predictor import Predictor
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, InferenceConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    inference_config = InferenceConfig(score_threshold=0.01)

    model = Detector(model_config)
    predictor = Predictor(model=model, config=inference_config, device="cpu")

    # Create dummy image
    image = torch.randn(1, 1, 128, 128)

    particles = predictor.predict(image)

    # Should return list of particles
    assert isinstance(particles, list)
    # Each particle should have required fields
    for p in particles:
        assert "x" in p
        assert "y" in p
        assert "score" in p
