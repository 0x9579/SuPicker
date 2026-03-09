from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile

from supicker.data.star_parser import parse_star_file


def flip_y_coordinates(particles: list[dict], image_height: int) -> list[dict]:
    flipped = []
    for particle in particles:
        new_particle = deepcopy(particle)
        new_particle["y"] = image_height - particle["y"]
        flipped.append(new_particle)
    return flipped


def compute_coordinate_stats(particles: list[dict], image_width: int, image_height: int) -> dict:
    xs = [float(p["x"]) for p in particles]
    ys = [float(p["y"]) for p in particles]
    out_of_bounds_count = sum(
        x < 0 or x >= image_width or y < 0 or y >= image_height
        for x, y in zip(xs, ys)
    )
    return {
        "particle_count": len(particles),
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "out_of_bounds_count": out_of_bounds_count,
    }


def _load_image_array(image_path: Path) -> np.ndarray:
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(str(image_path))
    elif image_path.suffix.lower() == ".mrc":
        try:
            import mrcfile
        except ImportError as exc:
            raise ImportError(
                'mrcfile is required for .mrc support; install with pip install -e ".[mrc]"'
            ) from exc
        with mrcfile.open(str(image_path), permissive=True) as mrc:
            image = mrc.data.copy()
    else:
        image = np.array(Image.open(image_path))

    image = image.astype(np.float32)
    if image.ndim > 2:
        image = image.squeeze()
    return image


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max > image_min:
        normalized = (image - image_min) / (image_max - image_min)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def generate_coordinate_overlay(
    image_path: str | Path,
    star_path: str | Path,
    output_path: str | Path,
    micrograph_name: str | None = None,
    flip_y: bool = False,
    radius: int = 6,
) -> dict:
    image_path = Path(image_path)
    star_path = Path(star_path)
    output_path = Path(output_path)

    image = _load_image_array(image_path)
    image_height, image_width = image.shape[:2]

    particles_by_micrograph = parse_star_file(star_path)
    resolved_micrograph_name = micrograph_name or image_path.name
    if resolved_micrograph_name not in particles_by_micrograph:
        raise ValueError(f"No STAR entry found for micrograph: {resolved_micrograph_name}")

    particles = particles_by_micrograph[resolved_micrograph_name]
    if flip_y:
        particles = flip_y_coordinates(particles, image_height=image_height)

    canvas = Image.fromarray(_normalize_to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for particle in particles:
        x = float(particle["x"])
        y = float(particle["y"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    stats = compute_coordinate_stats(particles, image_width=image_width, image_height=image_height)
    stats.update(
        {
            "resolved_micrograph_name": resolved_micrograph_name,
            "flip_y": flip_y,
            "image_width": image_width,
            "image_height": image_height,
            "output_path": str(output_path),
        }
    )
    return stats
