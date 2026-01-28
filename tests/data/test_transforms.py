import pytest
import torch
import numpy as np


def test_horizontal_flip():
    from supicker.data.transforms import HorizontalFlip

    transform = HorizontalFlip(p=1.0)  # Always flip
    image = torch.arange(16).reshape(1, 4, 4).float()
    particles = [{"x": 1.0, "y": 2.0, "width": 10, "height": 10}]

    new_image, new_particles = transform(image, particles)

    # Image should be flipped horizontally
    assert new_image[0, 0, 0] == image[0, 0, 3]
    # x coordinate should be flipped: new_x = width - x = 4 - 1 = 3
    assert new_particles[0]["x"] == 3.0
    assert new_particles[0]["y"] == 2.0  # y unchanged


def test_vertical_flip():
    from supicker.data.transforms import VerticalFlip

    transform = VerticalFlip(p=1.0)
    image = torch.arange(16).reshape(1, 4, 4).float()
    particles = [{"x": 1.0, "y": 1.0, "width": 10, "height": 10}]

    new_image, new_particles = transform(image, particles)

    # y coordinate should be flipped: new_y = height - y = 4 - 1 = 3
    assert new_particles[0]["x"] == 1.0  # x unchanged
    assert new_particles[0]["y"] == 3.0


def test_random_rotation_90():
    from supicker.data.transforms import RandomRotation90

    transform = RandomRotation90(p=1.0)
    image = torch.randn(1, 64, 64)
    particles = [{"x": 16.0, "y": 32.0, "width": 10, "height": 10}]

    new_image, new_particles = transform(image, particles)

    # Shape should be preserved for square images
    assert new_image.shape == image.shape
    # Particles should be transformed
    assert len(new_particles) == 1


def test_gaussian_noise():
    from supicker.data.transforms import GaussianNoise

    transform = GaussianNoise(std=0.1, p=1.0)
    image = torch.zeros(1, 32, 32)
    particles = [{"x": 16.0, "y": 16.0}]

    new_image, new_particles = transform(image, particles)

    # Image should have noise added
    assert new_image.std() > 0
    # Particles should be unchanged
    assert new_particles[0]["x"] == 16.0


def test_compose_transforms():
    from supicker.data.transforms import Compose, HorizontalFlip, VerticalFlip

    transform = Compose([
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
    ])
    image = torch.randn(1, 32, 32)
    particles = [{"x": 8.0, "y": 8.0, "width": 10, "height": 10}]

    new_image, new_particles = transform(image, particles)

    # Both flips applied
    assert new_particles[0]["x"] == 32.0 - 8.0  # 24
    assert new_particles[0]["y"] == 32.0 - 8.0  # 24
