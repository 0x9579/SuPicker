from pathlib import Path
from typing import Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from supicker.config import InferenceConfig


class Predictor:
    """Predictor for particle detection inference."""

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize predictor.

        Args:
            model: Trained detection model
            config: Inference configuration
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.output_stride = 4  # Default output stride

    def predict(
        self,
        image: torch.Tensor,
    ) -> list[dict]:
        """Run inference on an image.

        Args:
            image: Input image tensor (1, C, H, W) or (C, H, W)

        Returns:
            List of detected particles with x, y, score, class_id, width, height
        """
        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(image)

        # Extract peaks from heatmap
        heatmap = outputs["heatmap"]
        size = outputs["size"]
        offset = outputs["offset"]

        particles = self.extract_peaks(heatmap)

        # Add size and offset information
        for p in particles:
            x_out, y_out = int(p["x"]), int(p["y"])
            h_out, w_out = heatmap.shape[2], heatmap.shape[3]

            # Clamp to valid range
            x_out = max(0, min(x_out, w_out - 1))
            y_out = max(0, min(y_out, h_out - 1))

            # Get size predictions
            if size is not None:
                p["width"] = float(size[0, 0, y_out, x_out])
                p["height"] = float(size[0, 1, y_out, x_out])
            else:
                p["width"] = 64.0
                p["height"] = 64.0

            # Get offset and apply to coordinates
            if offset is not None:
                offset_x = float(offset[0, 0, y_out, x_out])
                offset_y = float(offset[0, 1, y_out, x_out])
            else:
                offset_x, offset_y = 0.0, 0.0

            # Convert to original image coordinates
            p["x"] = (p["x"] + offset_x) * self.output_stride
            p["y"] = (p["y"] + offset_y) * self.output_stride

        # Apply NMS
        if self.config.nms_enabled:
            particles = self.apply_nms(particles)

        return particles

    def extract_peaks(
        self,
        heatmap: torch.Tensor,
        min_distance: int = 1,
    ) -> list[dict]:
        """Extract local maxima from heatmap.

        Args:
            heatmap: Heatmap tensor (B, C, H, W)
            min_distance: Minimum distance between peaks

        Returns:
            List of peak dictionaries with x, y, score, class_id
        """
        batch_size, num_classes, h, w = heatmap.shape
        particles = []

        # Use max pooling to find local maxima
        kernel_size = 2 * min_distance + 1
        padding = min_distance

        hmax = F.max_pool2d(
            heatmap,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        # Keep only local maxima above threshold
        keep = (heatmap == hmax) & (heatmap >= self.config.score_threshold)

        for b in range(batch_size):
            for c in range(num_classes):
                # Find peak locations
                y_coords, x_coords = torch.where(keep[b, c])

                for y, x in zip(y_coords.tolist(), x_coords.tolist()):
                    score = float(heatmap[b, c, y, x])
                    particles.append({
                        "x": x,
                        "y": y,
                        "score": score,
                        "class_id": c,
                    })

        # Sort by score descending
        particles.sort(key=lambda p: p["score"], reverse=True)

        return particles

    def apply_nms(
        self,
        particles: list[dict],
    ) -> list[dict]:
        """Apply non-maximum suppression.

        Args:
            particles: List of particle dictionaries

        Returns:
            Filtered list of particles
        """
        if not particles:
            return []

        radius = self.config.nms_radius
        radius_sq = radius ** 2

        # Sort by score descending
        particles = sorted(particles, key=lambda p: p["score"], reverse=True)

        keep = []
        suppressed = set()

        for i, p in enumerate(particles):
            if i in suppressed:
                continue

            keep.append(p)

            # Suppress nearby particles
            for j in range(i + 1, len(particles)):
                if j in suppressed:
                    continue

                other = particles[j]
                dist_sq = (p["x"] - other["x"]) ** 2 + (p["y"] - other["y"]) ** 2

                if dist_sq < radius_sq:
                    suppressed.add(j)

        return keep

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "Predictor":
        """Create predictor from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model architecture (weights will be loaded)
            config: Inference configuration
            device: Device to run inference on

        Returns:
            Predictor instance
        """
        if config is None:
            config = InferenceConfig()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model=model, config=config, device=device)
