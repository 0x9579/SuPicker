from dataclasses import dataclass

from .base import Config


@dataclass
class InferenceConfig(Config):
    """Inference configuration."""
    checkpoint_path: str = ""
    score_threshold: float = 0.3
    nms_enabled: bool = True
    nms_radius: float = 20.0  # pixels
    output_format: str = "star"  # star, json, csv
    batch_size: int = 1
    device: str = "cuda"
