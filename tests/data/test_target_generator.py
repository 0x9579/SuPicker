import pytest
import torch


def test_generate_gaussian_heatmap():
    from supicker.data.target_generator import TargetGenerator

    generator = TargetGenerator(num_classes=1, output_stride=4, gaussian_sigma=2.0)

    # Single particle at center
    particles = [{"x": 128.0, "y": 128.0, "class_id": 0, "width": 64, "height": 64}]
    heatmap = generator.generate_heatmap(particles, image_size=(256, 256))

    assert heatmap.shape == (1, 64, 64)  # num_classes, H/4, W/4
    # Peak should be at (32, 32) in output space
    assert heatmap[0, 32, 32] == 1.0
    # Values should decrease away from center
    assert heatmap[0, 32, 30] < 1.0


def test_generate_size_map():
    from supicker.data.target_generator import TargetGenerator

    generator = TargetGenerator(num_classes=1, output_stride=4)

    particles = [{"x": 128.0, "y": 128.0, "class_id": 0, "width": 64, "height": 48}]
    size_map, mask = generator.generate_size_map(particles, image_size=(256, 256))

    assert size_map.shape == (2, 64, 64)  # width, height channels
    assert mask.shape == (64, 64)
    # Size at particle location
    assert size_map[0, 32, 32] == 64.0  # width
    assert size_map[1, 32, 32] == 48.0  # height
    assert mask[32, 32] == 1.0


def test_generate_offset_map():
    from supicker.data.target_generator import TargetGenerator

    generator = TargetGenerator(num_classes=1, output_stride=4)

    # Particle at non-integer output location
    particles = [{"x": 130.0, "y": 132.0, "class_id": 0, "width": 64, "height": 64}]
    offset_map, mask = generator.generate_offset_map(particles, image_size=(256, 256))

    assert offset_map.shape == (2, 64, 64)  # offset_x, offset_y
    # 130/4 = 32.5, so offset_x = 0.5
    # 132/4 = 33.0, so offset_y = 0.0
    assert abs(offset_map[0, 33, 32] - 0.5) < 0.01
    assert abs(offset_map[1, 33, 32] - 0.0) < 0.01


def test_generate_targets():
    from supicker.data.target_generator import TargetGenerator

    generator = TargetGenerator(num_classes=2, output_stride=4)

    particles = [
        {"x": 64.0, "y": 64.0, "class_id": 0, "width": 50, "height": 50},
        {"x": 192.0, "y": 192.0, "class_id": 1, "width": 60, "height": 60},
    ]
    targets = generator(particles, image_size=(256, 256))

    assert "heatmap" in targets
    assert "size" in targets
    assert "offset" in targets
    assert "mask" in targets
    assert targets["heatmap"].shape == (2, 64, 64)
