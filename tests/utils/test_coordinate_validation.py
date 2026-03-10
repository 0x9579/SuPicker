from pathlib import Path
import subprocess
import sys
import json
import csv

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


def test_validate_coords_cli_directory_mode_generates_multiple_overlays(tmp_path: Path):
    image_dir = tmp_path / "images"
    star_dir = tmp_path / "stars"
    output_dir = tmp_path / "overlays"
    image_dir.mkdir()
    star_dir.mkdir()

    for stem in ["a", "b"]:
        tifffile.imwrite(image_dir / f"{stem}.tiff", np.zeros((16, 16), dtype=np.float32))
        (star_dir / f"{stem}.star").write_text(
            f"""data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
{stem}.tiff 5.0 6.0
"""
        )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_coords.py",
            "--image-dir",
            str(image_dir),
            "--star-dir",
            str(star_dir),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (output_dir / "a.tiff.png").exists()
    assert (output_dir / "b.tiff.png").exists()
    assert "processed_files: 2" in result.stdout


def test_validate_coords_cli_directory_mode_rejects_micrograph_name(tmp_path: Path):
    image_dir = tmp_path / "images"
    star_dir = tmp_path / "stars"
    output_dir = tmp_path / "overlays"
    image_dir.mkdir()
    star_dir.mkdir()

    tifffile.imwrite(image_dir / "a.tiff", np.zeros((16, 16), dtype=np.float32))
    (star_dir / "a.star").write_text(
        """data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
a.tiff 5.0 6.0
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_coords.py",
            "--image-dir", str(image_dir),
            "--star-dir", str(star_dir),
            "--output-dir", str(output_dir),
            "--micrograph-name", "forced_name.mrc",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "micrograph-name" in result.stderr


def test_validate_coords_cli_directory_mode_writes_json_summary(tmp_path: Path):
    image_dir = tmp_path / "images"
    star_dir = tmp_path / "stars"
    output_dir = tmp_path / "overlays"
    summary_path = tmp_path / "summary.json"
    image_dir.mkdir()
    star_dir.mkdir()

    tifffile.imwrite(image_dir / "a.tiff", np.zeros((16, 16), dtype=np.float32))
    (star_dir / "a.star").write_text(
        """data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
a.tiff 5.0 6.0
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_coords.py",
            "--image-dir",
            str(image_dir),
            "--star-dir",
            str(star_dir),
            "--output-dir",
            str(output_dir),
            "--summary-output",
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    data = json.loads(summary_path.read_text())
    assert data["processed_files"] == 1
    assert data["files"][0]["resolved_micrograph_name"] == "a.tiff"


def test_validate_coords_cli_directory_mode_writes_csv_summary(tmp_path: Path):
    image_dir = tmp_path / "images"
    star_dir = tmp_path / "stars"
    output_dir = tmp_path / "overlays"
    summary_path = tmp_path / "summary.csv"
    image_dir.mkdir()
    star_dir.mkdir()

    tifffile.imwrite(image_dir / "a.tiff", np.zeros((16, 16), dtype=np.float32))
    (star_dir / "a.star").write_text(
        """data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
a.tiff 5.0 6.0
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_coords.py",
            "--image-dir",
            str(image_dir),
            "--star-dir",
            str(star_dir),
            "--output-dir",
            str(output_dir),
            "--summary-output",
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["resolved_micrograph_name"] == "a.tiff"


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
