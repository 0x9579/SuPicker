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

        return self.decode_outputs(
            outputs,
            score_threshold=self.config.score_threshold,
            nms_enabled=self.config.nms_enabled,
            nms_radius=self.config.nms_radius,
            output_stride=self.output_stride,
        )

    @staticmethod
    def decode_outputs(
        outputs: dict[str, torch.Tensor],
        score_threshold: float = 0.3,
        nms_enabled: bool = True,
        nms_radius: float = 20.0,
        output_stride: int = 4,
        min_distance: int = 1,
    ) -> list[dict]:
        """Decode model outputs into particle predictions."""
        heatmap = outputs["heatmap"]
        size = outputs.get("size")
        offset = outputs.get("offset")

        particles = Predictor.extract_peaks_from_heatmap(
            heatmap,
            score_threshold=score_threshold,
            min_distance=min_distance,
        )

        for p in particles:
            batch_idx = int(p.get("batch_idx", 0))
            x_out, y_out = int(p["x"]), int(p["y"])
            h_out, w_out = heatmap.shape[2], heatmap.shape[3]

            x_out = max(0, min(x_out, w_out - 1))
            y_out = max(0, min(y_out, h_out - 1))

            if size is not None:
                p["width"] = float(size[batch_idx, 0, y_out, x_out])
                p["height"] = float(size[batch_idx, 1, y_out, x_out])
            else:
                p["width"] = 64.0
                p["height"] = 64.0

            if offset is not None:
                offset_x = float(offset[batch_idx, 0, y_out, x_out])
                offset_y = float(offset[batch_idx, 1, y_out, x_out])
            else:
                offset_x, offset_y = 0.0, 0.0

            p["x"] = (p["x"] + offset_x) * output_stride
            p["y"] = (p["y"] + offset_y) * output_stride

        if nms_enabled:
            particles = Predictor.apply_nms_to_particles(particles, radius=nms_radius)

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
        return self.extract_peaks_from_heatmap(
            heatmap,
            score_threshold=self.config.score_threshold,
            min_distance=min_distance,
        )

    @staticmethod
    def extract_peaks_from_heatmap(
        heatmap: torch.Tensor,
        score_threshold: float,
        min_distance: int = 1,
    ) -> list[dict]:
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
        keep = (heatmap == hmax) & (heatmap >= score_threshold)

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
                        "batch_idx": b,
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
        return self.apply_nms_to_particles(particles, radius=self.config.nms_radius)

    @staticmethod
    def apply_nms_to_particles(
        particles: list[dict],
        radius: float,
    ) -> list[dict]:
        if not particles:
            return []

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
                if p.get("batch_idx", 0) != other.get("batch_idx", 0):
                    continue
                if p.get("class_id", 0) != other.get("class_id", 0):
                    continue
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
