import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_data_dir():
    """Create a temporary directory with sample data."""
    import tifffile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample images
        for i in range(3):
            img = np.random.rand(256, 256).astype(np.float32)
            tifffile.imwrite(tmpdir / f"image_{i:03d}.tiff", img)

        # Create sample STAR file
        star_content = """data_

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
_rlnClassNumber
image_000.tiff 100.0 100.0 1
image_000.tiff 150.0 150.0 1
image_001.tiff 128.0 128.0 1
"""
        (tmpdir / "particles.star").write_text(star_content)

        yield tmpdir


def test_dataset_length(sample_data_dir):
    from supicker.data.dataset import ParticleDataset

    dataset = ParticleDataset(
        image_dir=sample_data_dir,
        star_file=sample_data_dir / "particles.star",
    )

    # Should have 2 images with particles (image_000 and image_001)
    assert len(dataset) == 2


def test_dataset_getitem(sample_data_dir):
    from supicker.data.dataset import ParticleDataset

    dataset = ParticleDataset(
        image_dir=sample_data_dir,
        star_file=sample_data_dir / "particles.star",
        output_stride=4,
    )

    sample = dataset[0]

    assert "image" in sample
    assert "heatmap" in sample
    assert "size" in sample
    assert "offset" in sample
    assert "mask" in sample

    # Check shapes
    assert sample["image"].shape == (1, 256, 256)  # C, H, W
    assert sample["heatmap"].shape == (1, 64, 64)  # num_classes, H/4, W/4


def test_dataset_with_transforms(sample_data_dir):
    from supicker.data.dataset import ParticleDataset
    from supicker.data.transforms import Compose, Normalize

    transforms = Compose([Normalize(p=1.0)])

    dataset = ParticleDataset(
        image_dir=sample_data_dir,
        star_file=sample_data_dir / "particles.star",
        transforms=transforms,
    )

    sample = dataset[0]

    # Image should be normalized (roughly zero mean)
    assert abs(sample["image"].mean()) < 0.5
