from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import tifffile
import numpy as np

from .star_parser import parse_star_file
from .target_generator import TargetGenerator
from .transforms import Compose


class ParticleDataset(Dataset):
    """Dataset for loading micrographs and particle annotations."""

    def __init__(
        self,
        image_dir: Union[str, Path],
        star_file: Union[str, Path],
        num_classes: int = 1,
        output_stride: int = 4,
        gaussian_sigma: float = 2.0,
        default_particle_size: int = 64,
        transforms: Optional[Compose] = None,
    ):
        """Initialize dataset.

        Args:
            image_dir: Directory containing micrograph images
            star_file: Path to STAR file with particle annotations
            num_classes: Number of particle classes
            output_stride: Downsampling factor for output maps
            gaussian_sigma: Sigma for Gaussian heatmap
            default_particle_size: Default particle size if not in annotations
            transforms: Optional transforms to apply
        """
        self.image_dir = Path(image_dir)
        self.star_file = Path(star_file)
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.default_particle_size = default_particle_size
        self.transforms = transforms

        # Parse STAR file
        self.particles_by_micrograph = parse_star_file(star_file)

        # Filter to only images with particles that exist
        self.image_names = []
        for name in self.particles_by_micrograph.keys():
            image_path = self._find_image(name)
            if image_path is not None:
                self.image_names.append(name)

        # Create target generator
        self.target_generator = TargetGenerator(
            num_classes=num_classes,
            output_stride=output_stride,
            gaussian_sigma=gaussian_sigma,
        )

    def _find_image(self, name: str) -> Optional[Path]:
        """Find image file by name."""
        # Try exact match first
        path = self.image_dir / name
        if path.exists():
            return path

        # Try without extension
        stem = Path(name).stem
        for ext in [".tiff", ".tif", ".mrc", ".png", ".jpg"]:
            path = self.image_dir / f"{stem}{ext}"
            if path.exists():
                return path

        return None

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dictionary with 'image', 'heatmap', 'size', 'offset', 'mask'
        """
        image_name = self.image_names[idx]
        image_path = self._find_image(image_name)

        # Load image
        image = self._load_image(image_path)
        h, w = image.shape[-2], image.shape[-1]

        # Get particles for this image
        particles = self.particles_by_micrograph[image_name]

        # Add default size and normalize class_id
        for p in particles:
            if "width" not in p:
                p["width"] = self.default_particle_size
            if "height" not in p:
                p["height"] = self.default_particle_size
            
            # If we only want one class, force all particles to class 0
            if self.num_classes == 1:
                p["class_id"] = 0
            # Otherwise ensure class_id is within bounds
            elif p.get("class_id", 0) >= self.num_classes:
                p["class_id"] = 0 # Default to first class if out of bounds

        # Apply transforms
        if self.transforms is not None:
            image, particles = self.transforms(image, particles)

        # Generate targets
        targets = self.target_generator(particles, image_size=(h, w))

        return {
            "image": image,
            "heatmap": targets["heatmap"],
            "size": targets["size"],
            "offset": targets["offset"],
            "mask": targets["mask"],
            "particles": particles,  # Include original particles for evaluation
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from file."""
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

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Add channel dimension if needed
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        return torch.from_numpy(image)


def particle_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function that handles variable-length particle lists.

    Tensor fields (image, heatmap, size, offset, mask) are stacked normally.
    The 'particles' field is kept as a list of lists since each image
    may have a different number of particles.
    """
    collated = {}
    tensor_keys = ["image", "heatmap", "size", "offset", "mask"]

    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([sample[key] for sample in batch])

    # Keep particles as list of lists (variable length per image)
    if "particles" in batch[0]:
        collated["particles"] = [sample["particles"] for sample in batch]

    return collated


def create_dataloader(
    image_dir: Union[str, Path],
    star_file: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    transforms: Optional[Compose] = None,
    distributed: bool = False,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for training.

    Args:
        image_dir: Directory containing images
        star_file: Path to STAR file
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        transforms: Optional transforms
        distributed: Whether to use DistributedSampler for multi-GPU training
        **dataset_kwargs: Additional arguments for ParticleDataset

    Returns:
        DataLoader instance
    """
    dataset = ParticleDataset(
        image_dir=image_dir,
        star_file=star_file,
        transforms=transforms,
        **dataset_kwargs,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        # When using DistributedSampler, shuffle must be False in DataLoader
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=particle_collate_fn,
    )
