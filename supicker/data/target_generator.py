import math
import torch
import numpy as np


class TargetGenerator:
    """Generate training targets from particle annotations."""

    def __init__(
        self,
        num_classes: int = 1,
        output_stride: int = 4,
        gaussian_sigma: float = 2.0,
    ):
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.gaussian_sigma = gaussian_sigma

    def generate_heatmap(
        self, particles: list[dict], image_size: tuple[int, int]
    ) -> torch.Tensor:
        """Generate Gaussian heatmap from particle locations.

        Args:
            particles: List of particle dicts with 'x', 'y', 'class_id'
            image_size: (height, width) of input image

        Returns:
            Heatmap tensor (num_classes, H/stride, W/stride)
        """
        h, w = image_size
        out_h, out_w = h // self.output_stride, w // self.output_stride
        heatmap = torch.zeros(self.num_classes, out_h, out_w)

        for p in particles:
            # Convert to output coordinates
            cx = p["x"] / self.output_stride
            cy = p["y"] / self.output_stride
            class_id = p.get("class_id", 0)

            # Skip particles outside the output map
            if cx < 0 or cx >= out_w or cy < 0 or cy >= out_h:
                continue

            # Ensure class_id is within bounds
            if class_id < 0 or class_id >= self.num_classes:
                continue

            # Determine Gaussian radius based on particle size or default
            width = p.get("width", 64) / self.output_stride
            height = p.get("height", 64) / self.output_stride
            radius = max(int(self.gaussian_sigma * 3), 1)

            # Draw Gaussian
            self._draw_gaussian(heatmap[class_id], cx, cy, radius)

        return heatmap

    def _draw_gaussian(
        self, heatmap: torch.Tensor, cx: float, cy: float, radius: int
    ) -> None:
        """Draw a Gaussian peak on the heatmap."""
        height, width = heatmap.shape

        # Integer center
        cx_int, cy_int = int(cx), int(cy)

        # Gaussian range
        left = min(cx_int, radius)
        right = min(width - cx_int, radius + 1)
        top = min(cy_int, radius)
        bottom = min(height - cy_int, radius + 1)

        # Skip if the Gaussian range is empty
        if left + right <= 0 or top + bottom <= 0:
            return

        # Create Gaussian kernel
        sigma = self.gaussian_sigma
        y = torch.arange(-top, bottom, dtype=torch.float32)
        x = torch.arange(-left, right, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # In CenterNet, the peak must be EXACTLY 1.0 at the integer center. 
        # So we omit the sub-pixel offsets (cx-cx_int) from the Gaussian kernel.
        # The offset regression head will handle the precise sub-pixel coordinates.
        gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

        # Apply to heatmap (take max with existing values)
        y_slice = slice(cy_int - top, cy_int + bottom)
        x_slice = slice(cx_int - left, cx_int + right)
        heatmap[y_slice, x_slice] = torch.maximum(heatmap[y_slice, x_slice], gaussian)

    def generate_size_map(
        self, particles: list[dict], image_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate size regression targets.

        Args:
            particles: List of particle dicts with 'x', 'y', 'width', 'height'
            image_size: (height, width) of input image

        Returns:
            Tuple of (size_map, mask) where size_map is (2, H/stride, W/stride)
        """
        h, w = image_size
        out_h, out_w = h // self.output_stride, w // self.output_stride
        size_map = torch.zeros(2, out_h, out_w)
        mask = torch.zeros(out_h, out_w)

        for p in particles:
            cx_int = int(p["x"] / self.output_stride)
            cy_int = int(p["y"] / self.output_stride)

            if 0 <= cx_int < out_w and 0 <= cy_int < out_h:
                size_map[0, cy_int, cx_int] = p.get("width", 64)
                size_map[1, cy_int, cx_int] = p.get("height", 64)
                mask[cy_int, cx_int] = 1.0

        return size_map, mask

    def generate_offset_map(
        self, particles: list[dict], image_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate offset regression targets for sub-pixel localization.

        Args:
            particles: List of particle dicts with 'x', 'y'
            image_size: (height, width) of input image

        Returns:
            Tuple of (offset_map, mask) where offset_map is (2, H/stride, W/stride)
        """
        h, w = image_size
        out_h, out_w = h // self.output_stride, w // self.output_stride
        offset_map = torch.zeros(2, out_h, out_w)
        mask = torch.zeros(out_h, out_w)

        for p in particles:
            cx = p["x"] / self.output_stride
            cy = p["y"] / self.output_stride
            cx_int = int(cx)
            cy_int = int(cy)

            if 0 <= cx_int < out_w and 0 <= cy_int < out_h:
                offset_map[0, cy_int, cx_int] = cx - cx_int  # x offset
                offset_map[1, cy_int, cx_int] = cy - cy_int  # y offset
                mask[cy_int, cx_int] = 1.0

        return offset_map, mask

    def __call__(
        self, particles: list[dict], image_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Generate all training targets.

        Args:
            particles: List of particle dicts
            image_size: (height, width) of input image

        Returns:
            Dictionary with 'heatmap', 'size', 'offset', 'mask' tensors
        """
        heatmap = self.generate_heatmap(particles, image_size)
        size_map, size_mask = self.generate_size_map(particles, image_size)
        offset_map, offset_mask = self.generate_offset_map(particles, image_size)

        # Use same mask for size and offset
        mask = size_mask  # They should be identical

        return {
            "heatmap": heatmap,
            "size": size_map,
            "offset": offset_map,
            "mask": mask,
        }
