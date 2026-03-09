from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile


def test_flip_y_coordinates_uses_image_height():
    from supicker.utils.coordinate_validation import flip_y_coordinates

    particles = [{"x": 10.0, "y": 20.0}, {"x": 5.0, "y": 99.5}]

    flipped = flip_y_coordinates(particles, image_height=100)

    assert flipped[0]["x"] == 10.0
    assert flipped[0]["y"] == 80.0
    assert flipped[1]["y"] == 0.5


def test_compute_coordinate_stats_counts_out_of_bounds():
    from supicker.utils.coordinate_validation import compute_coordinate_stats

    particles = [
        {"x": 10.0, "y": 20.0},
        {"x": -1.0, "y": 5.0},
        {"x": 30.0, "y": 41.0},
    ]

    stats = compute_coordinate_stats(particles, image_width=40, image_height=40)

    assert stats["particle_count"] == 3
    assert stats["out_of_bounds_count"] == 2
    assert stats["x_min"] == -1.0
    assert stats["y_max"] == 41.0


def test_generate_coordinate_overlay_creates_png(tmp_path: Path):
    from supicker.utils.coordinate_validation import generate_coordinate_overlay

    image = np.zeros((32, 32), dtype=np.float32)
    image_path = tmp_path / "micrograph_001.tiff"
    tifffile.imwrite(image_path, image)

    star_path = tmp_path / "particles.star"
    star_path.write_text(
        """data_particles

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
micrograph_001.tiff 8.0 9.0
micrograph_001.tiff 16.0 20.0
"""
    )

    output_path = tmp_path / "overlay.png"

    result = generate_coordinate_overlay(
        image_path=image_path,
        star_path=star_path,
        output_path=output_path,
        micrograph_name=None,
        flip_y=False,
        radius=3,
    )

    assert output_path.exists()
    assert result["particle_count"] == 2
    assert result["resolved_micrograph_name"] == "micrograph_001.tiff"
    assert result["flip_y"] is False


def test_validate_coords_cli_generates_overlay(tmp_path: Path):
    image = np.zeros((24, 24), dtype=np.float32)
    image_path = tmp_path / "cli_image.tiff"
    tifffile.imwrite(image_path, image)

    star_path = tmp_path / "cli.star"
    star_path.write_text(
        """data_particles

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
cli_image.tiff 5.0 6.0
"""
    )

    output_path = tmp_path / "cli_overlay.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_coords.py",
            "--image",
            str(image_path),
            "--star",
            str(star_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.exists()
    assert "particle_count" in result.stdout


def test_generate_coordinate_overlay_requires_matching_star_entry(tmp_path: Path):
    from supicker.utils.coordinate_validation import generate_coordinate_overlay

    image = np.zeros((16, 16), dtype=np.float32)
    image_path = tmp_path / "missing.tiff"
    tifffile.imwrite(image_path, image)

    star_path = tmp_path / "missing.star"
    star_path.write_text(
        """data_particles

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
other.tiff 1.0 2.0
"""
    )

    with pytest.raises(ValueError, match="micrograph"):
        generate_coordinate_overlay(
            image_path=image_path,
            star_path=star_path,
            output_path=tmp_path / "unused.png",
        )
