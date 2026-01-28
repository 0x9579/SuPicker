from dataclasses import dataclass, field
from typing import Optional

from .base import Config


@dataclass
class AugmentationConfig(Config):
    """Data augmentation configuration."""
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_90: bool = True
    random_rotation: bool = True
    rotation_range: tuple[float, float] = (-180.0, 180.0)
    brightness: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast: bool = True
    contrast_range: tuple[float, float] = (0.8, 1.2)
    gaussian_noise: bool = True
    noise_std: float = 0.02
    ctf_simulation: bool = False


@dataclass
class DataConfig(Config):
    """Data loading and preprocessing configuration."""
    train_image_dir: str = ""
    train_star_file: str = ""
    val_image_dir: Optional[str] = None
    val_star_file: Optional[str] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    num_workers: int = 4
    pin_memory: bool = True
