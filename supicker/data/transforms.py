import random
import copy
from typing import Callable
import torch


class BaseTransform:
    """Base class for transforms that modify both image and particles."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        if random.random() < self.p:
            return self.apply(image, copy.deepcopy(particles))
        return image, particles

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        raise NotImplementedError


class HorizontalFlip(BaseTransform):
    """Flip image and particles horizontally."""

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        # Flip image along width axis
        image = torch.flip(image, dims=[-1])

        # Flip particle x coordinates
        width = image.shape[-1]
        for p in particles:
            p["x"] = width - p["x"]

        return image, particles


class VerticalFlip(BaseTransform):
    """Flip image and particles vertically."""

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        # Flip image along height axis
        image = torch.flip(image, dims=[-2])

        # Flip particle y coordinates
        height = image.shape[-2]
        for p in particles:
            p["y"] = height - p["y"]

        return image, particles


class RandomRotation90(BaseTransform):
    """Rotate image and particles by 90, 180, or 270 degrees."""

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        # Random rotation: 1, 2, or 3 times 90 degrees
        k = random.randint(1, 3)

        # Rotate image (k * 90 degrees counter-clockwise)
        image = torch.rot90(image, k, dims=[-2, -1])

        # Rotate particle coordinates
        height, width = image.shape[-2], image.shape[-1]
        for p in particles:
            x, y = p["x"], p["y"]
            for _ in range(k):
                # 90 degree counter-clockwise rotation
                new_x = y
                new_y = width - x
                x, y = new_x, new_y
                # After rotation, dimensions swap
                height, width = width, height
            p["x"] = x
            p["y"] = y
            # Swap width/height if rotated 90 or 270
            if k in (1, 3) and "width" in p and "height" in p:
                p["width"], p["height"] = p["height"], p["width"]

        return image, particles


class RandomRotation(BaseTransform):
    """Rotate image and particles by a random angle."""

    def __init__(self, angle_range: tuple[float, float] = (-180.0, 180.0), p: float = 0.5):
        super().__init__(p)
        self.angle_range = angle_range

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        import math

        angle = random.uniform(*self.angle_range)
        rad = math.radians(angle)

        # Rotate image using grid_sample
        h, w = image.shape[-2], image.shape[-1]
        cx, cy = w / 2, h / 2

        # Create rotation matrix
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # For simplicity, use torch functional if available
        # This is a basic implementation - for production, use torchvision
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=image.dtype).unsqueeze(0)

        grid = torch.nn.functional.affine_grid(theta, image.unsqueeze(0).shape, align_corners=False)
        image = torch.nn.functional.grid_sample(
            image.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros'
        ).squeeze(0)

        # Rotate particle coordinates around center
        for p in particles:
            x, y = p["x"] - cx, p["y"] - cy
            new_x = x * cos_a - y * sin_a + cx
            new_y = x * sin_a + y * cos_a + cy
            p["x"] = new_x
            p["y"] = new_y

        return image, particles


class GaussianNoise(BaseTransform):
    """Add Gaussian noise to image."""

    def __init__(self, std: float = 0.02, p: float = 0.5):
        super().__init__(p)
        self.std = std

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        noise = torch.randn_like(image) * self.std
        image = image + noise
        return image, particles


class BrightnessContrast(BaseTransform):
    """Adjust brightness and contrast."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)

        # Apply brightness and contrast
        mean = image.mean()
        image = (image - mean) * contrast + mean * brightness

        return image, particles


class RandomCrop(BaseTransform):
    """Randomly crop a patch from the image and filter particles.

    Essential for training on large cryo-EM micrographs that don't fit
    in GPU memory at full resolution.
    """

    def __init__(self, crop_size: int = 1024, p: float = 1.0):
        super().__init__(p)
        self.crop_size = crop_size

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        _, h, w = image.shape
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        # Random top-left corner
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # Crop image
        image = image[:, top:top + crop_h, left:left + crop_w]

        # Filter and adjust particles
        cropped_particles = []
        for p in particles:
            new_x = p["x"] - left
            new_y = p["y"] - top
            # Keep particle only if its center is inside the crop
            if 0 <= new_x < crop_w and 0 <= new_y < crop_h:
                new_p = p.copy()
                new_p["x"] = new_x
                new_p["y"] = new_y
                cropped_particles.append(new_p)

        return image, cropped_particles


class Normalize(BaseTransform):
    """Normalize image to zero mean and unit variance."""

    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        return image, particles


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, particles: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        for t in self.transforms:
            image, particles = t(image, particles)
        return image, particles


def build_transforms(config) -> Compose:
    """Build transforms from AugmentationConfig.

    Args:
        config: AugmentationConfig instance

    Returns:
        Composed transforms
    """
    transforms = []

    # Crop first to reduce memory usage before other transforms
    crop_size = getattr(config, 'crop_size', 1024)
    if crop_size > 0:
        transforms.append(RandomCrop(crop_size=crop_size, p=1.0))

    if config.horizontal_flip:
        transforms.append(HorizontalFlip(p=0.5))
    if config.vertical_flip:
        transforms.append(VerticalFlip(p=0.5))
    if config.rotation_90:
        transforms.append(RandomRotation90(p=0.5))
    if config.random_rotation:
        transforms.append(RandomRotation(angle_range=config.rotation_range, p=0.5))
    if config.brightness or config.contrast:
        transforms.append(BrightnessContrast(
            brightness_range=config.brightness_range,
            contrast_range=config.contrast_range,
            p=0.5,
        ))
    if config.gaussian_noise:
        transforms.append(GaussianNoise(std=config.noise_std, p=0.5))

    # Always normalize
    transforms.append(Normalize(p=1.0))

    return Compose(transforms)
