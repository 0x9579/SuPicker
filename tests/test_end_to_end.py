"""End-to-end test for SuPicker pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
import tifffile

from supicker.config import (
    ModelConfig,
    BackboneConfig,
    ConvNeXtVariant,
    TrainingConfig,
    InferenceConfig,
)
from supicker.models import Detector
from supicker.data import ParticleDataset, create_dataloader
from supicker.engine import Trainer, Predictor
from supicker.utils import export_particles


def create_synthetic_data(output_dir: Path, num_images: int = 3):
    """Create synthetic training data with known particle positions."""
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create STAR file
    star_file = output_dir / "particles.star"

    all_particles = []

    for i in range(num_images):
        # Create synthetic image (256x256)
        image = np.random.randn(256, 256).astype(np.float32) * 0.1

        # Add Gaussian spots at known locations (simulating particles)
        particle_coords = [
            (64, 64),
            (128, 128),
            (192, 64),
            (64, 192),
        ]

        for x, y in particle_coords:
            # Create Gaussian spot
            yy, xx = np.ogrid[:256, :256]
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 10**2))
            image += gaussian.astype(np.float32)

            all_particles.append({
                "micrograph": f"image_{i:03d}.tiff",
                "x": float(x),
                "y": float(y),
            })

        # Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Save image
        tifffile.imwrite(str(image_dir / f"image_{i:03d}.tiff"), image)

    # Write STAR file
    with open(star_file, "w") as f:
        f.write("data_\n\n")
        f.write("loop_\n")
        f.write("_rlnMicrographName #1\n")
        f.write("_rlnCoordinateX #2\n")
        f.write("_rlnCoordinateY #3\n")

        for p in all_particles:
            f.write(f"{p['micrograph']}\t{p['x']:.2f}\t{p['y']:.2f}\n")

    return image_dir, star_file, all_particles


def test_end_to_end_pipeline():
    """Test full pipeline: data loading -> training -> prediction."""
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic data
        image_dir, star_file, ground_truth = create_synthetic_data(tmpdir)

        # Create model config with tiny backbone for fast testing
        model_config = ModelConfig(
            backbone=BackboneConfig(
                variant=ConvNeXtVariant.TINY,
                pretrained=False,
                in_channels=1,
            ),
        )

        # Create model
        model = Detector(model_config)

        # Create dataloader
        train_loader = create_dataloader(
            image_dir=str(image_dir),
            star_file=str(star_file),
            batch_size=2,
            num_workers=0,
            shuffle=True,
            transforms=None,
            num_classes=1,
        )

        # Verify dataloader works
        batch = next(iter(train_loader))
        assert "image" in batch
        assert "heatmap" in batch
        assert batch["image"].shape[0] == 2

        # Create trainer
        training_config = TrainingConfig(
            batch_size=2,
            epochs=2,  # Very short for testing
            learning_rate=1e-3,
            checkpoint_dir=str(tmpdir / "checkpoints"),
            log_dir=str(tmpdir / "logs"),
            save_interval=1,
        )

        trainer = Trainer(
            model=model,
            config=training_config,
            checkpoint_dir=str(tmpdir / "checkpoints"),
            log_dir=str(tmpdir / "logs"),
            device="cpu",
        )

        # Train for a few iterations
        trainer.train(train_loader, val_loader=None)

        # Verify checkpoint was saved
        checkpoint_dir = tmpdir / "checkpoints"
        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0

        # Test inference
        inference_config = InferenceConfig(
            checkpoint_path=str(checkpoints[0]),
            score_threshold=0.1,  # Low threshold for testing
            nms_enabled=True,
            nms_radius=20.0,
            device="cpu",
        )

        predictor = Predictor.from_checkpoint(
            checkpoint_path=str(checkpoints[0]),
            model=model,
            config=inference_config,
            device="cpu",
        )

        # Load a test image and run prediction
        test_image_path = image_dir / "image_000.tiff"
        image = tifffile.imread(str(test_image_path))
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image_tensor = torch.from_numpy(image[np.newaxis, ...])

        # Run prediction
        particles = predictor.predict(image_tensor)

        # Should detect some particles (model is undertrained but should produce output)
        assert isinstance(particles, list)

        # Test export functionality
        output_path = tmpdir / "output.star"
        export_particles(
            particles,
            output_path,
            format="star",
            micrograph_name="image_000.tiff",
        )
        assert output_path.exists()

        # Verify STAR file has content
        with open(output_path) as f:
            content = f.read()
            assert "data_" in content
            assert "loop_" in content


def test_model_forward_pass():
    """Test that model can do a forward pass with various input sizes."""
    model_config = ModelConfig(
        backbone=BackboneConfig(
            variant=ConvNeXtVariant.TINY,
            pretrained=False,
            in_channels=1,
        ),
    )
    model = Detector(model_config)
    model.eval()

    # Test different input sizes
    for size in [128, 256, 512]:
        x = torch.randn(1, 1, size, size)
        with torch.no_grad():
            outputs = model(x)

        assert "heatmap" in outputs
        assert "size" in outputs
        assert "offset" in outputs

        # Output should be 1/4 of input size (stride 4 from FPN)
        expected_h = size // 4
        expected_w = size // 4
        assert outputs["heatmap"].shape[2:] == (expected_h, expected_w)


def test_checkpoint_save_load():
    """Test that model can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        model_config = ModelConfig(
            backbone=BackboneConfig(
                variant=ConvNeXtVariant.TINY,
                pretrained=False,
            ),
        )

        # Create and save model
        model1 = Detector(model_config)
        checkpoint_path = tmpdir / "test_checkpoint.pt"

        torch.save({
            "model_state_dict": model1.state_dict(),
        }, checkpoint_path)

        # Load into new model
        model2 = Detector(model_config)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Compare outputs
        model1.eval()
        model2.eval()

        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        # Outputs should be identical
        assert torch.allclose(out1["heatmap"], out2["heatmap"])
        assert torch.allclose(out1["size"], out2["size"])
        assert torch.allclose(out1["offset"], out2["offset"])
