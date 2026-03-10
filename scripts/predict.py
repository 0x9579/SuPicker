#!/usr/bin/env python
"""Prediction script for SuPicker particle detector."""

import argparse
from pathlib import Path
from typing import Optional

import torch
import tifffile
import numpy as np

from supicker.config import (
    ModelConfig,
    BackboneConfig,
    ConvNeXtVariant,
    InferenceConfig,
)
from supicker.data.transforms import Normalize
from supicker.models import Detector
from supicker.engine import Predictor
from supicker.utils import export_particles


def parse_args():
    parser = argparse.ArgumentParser(description="Run SuPicker particle detection")

    # Input arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image or directory of images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="ConvNeXt backbone variant",
    )
    parser.add_argument(
        "--num-classes", type=int, default=1, help="Number of particle classes"
    )

    # Inference arguments
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Score threshold for detection"
    )
    parser.add_argument(
        "--nms-radius", type=float, default=20.0, help="NMS radius in pixels"
    )
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS")

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="./predictions",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="star",
        choices=["star", "json", "csv"],
        help="Output format",
    )

    # Other arguments
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    return parser.parse_args()


def load_image(path: Path) -> torch.Tensor:
    """Load an image file."""
    if path.suffix.lower() in [".tiff", ".tif"]:
        image = tifffile.imread(str(path))
    elif path.suffix.lower() == ".mrc":
        import mrcfile
        with mrcfile.open(str(path), permissive=True) as mrc:
            image = mrc.data.copy()
    else:
        from PIL import Image
        image = np.array(Image.open(path))

    # Convert to float32
    image = image.astype(np.float32)

    # Match dataset loading
    if image.max() > 1.0:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Add channel dimension if needed
    if image.ndim == 2:
        image = image[np.newaxis, ...]

    tensor = torch.from_numpy(image)
    tensor, _ = Normalize(p=1.0).apply(tensor, [])
    return tensor


def main():
    args = parse_args()

    # Create configs
    variant_map = {
        "tiny": ConvNeXtVariant.TINY,
        "small": ConvNeXtVariant.SMALL,
        "base": ConvNeXtVariant.BASE,
    }

    model_config = ModelConfig(
        backbone=BackboneConfig(
            variant=variant_map[args.backbone],
            pretrained=False,
            in_channels=1,
        ),
    )
    model_config.head.num_classes = args.num_classes

    inference_config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        score_threshold=args.threshold,
        nms_enabled=not args.no_nms,
        nms_radius=args.nms_radius,
        output_format=args.format,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Create model
    print(f"Loading model from {args.checkpoint}...")
    model = Detector(model_config)

    # Create predictor
    predictor = Predictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        config=inference_config,
        device=args.device,
    )

    # Find input images
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = list(input_path.glob("*.tiff")) + list(input_path.glob("*.tif"))
        image_paths += list(input_path.glob("*.mrc"))
    else:
        image_paths = [input_path]

    print(f"Found {len(image_paths)} images to process")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    total_particles = 0
    for img_path in image_paths:
        print(f"Processing {img_path.name}...")

        # Load image
        image = load_image(img_path)

        # Run prediction
        particles = predictor.predict(image)

        print(f"  Found {len(particles)} particles")
        total_particles += len(particles)

        # Export results
        output_path = output_dir / f"{img_path.stem}.{args.format}"
        export_particles(
            particles,
            output_path,
            format=args.format,
            micrograph_name=img_path.name,
        )

    print(f"\nProcessed {len(image_paths)} images")
    print(f"Total particles detected: {total_particles}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
