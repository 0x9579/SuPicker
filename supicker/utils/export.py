import json
import csv
from pathlib import Path
from typing import Union

from supicker.data.star_parser import write_star_file


def export_to_json(
    particles: list[dict],
    output_path: Union[str, Path],
    micrograph_name: str = "micrograph.tiff",
    indent: int = 2,
) -> None:
    """Export particles to JSON format.

    Args:
        particles: List of particle dictionaries
        output_path: Output file path
        micrograph_name: Name of the micrograph
        indent: JSON indentation
    """
    output_path = Path(output_path)

    data = {
        "micrograph": micrograph_name,
        "num_particles": len(particles),
        "particles": particles,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent)


def export_to_csv(
    particles: list[dict],
    output_path: Union[str, Path],
    micrograph_name: str = "micrograph.tiff",
) -> None:
    """Export particles to CSV format.

    Args:
        particles: List of particle dictionaries
        output_path: Output file path
        micrograph_name: Name of the micrograph
    """
    output_path = Path(output_path)

    if not particles:
        # Write empty file with header
        with open(output_path, "w", newline="") as f:
            f.write("micrograph,x,y,score,class_id,width,height\n")
        return

    # Get all fields from first particle
    fieldnames = ["micrograph", "x", "y", "score", "class_id", "width", "height"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for p in particles:
            row = {
                "micrograph": p.get("micrograph", micrograph_name),
                "x": p.get("x", 0),
                "y": p.get("y", 0),
                "score": p.get("score", 1.0),
                "class_id": p.get("class_id", 0),
                "width": p.get("width", 64),
                "height": p.get("height", 64),
            }
            writer.writerow(row)


def export_to_star(
    particles: list[dict],
    output_path: Union[str, Path],
    micrograph_name: str = "micrograph.tiff",
) -> None:
    """Export particles to STAR format.

    Args:
        particles: List of particle dictionaries
        output_path: Output file path
        micrograph_name: Name of the micrograph
    """
    write_star_file(particles, output_path, micrograph_name)


def export_particles(
    particles: list[dict],
    output_path: Union[str, Path],
    format: str = "star",
    micrograph_name: str = "micrograph.tiff",
) -> None:
    """Export particles to specified format.

    Args:
        particles: List of particle dictionaries
        output_path: Output file path
        format: Output format ('star', 'json', 'csv')
        micrograph_name: Name of the micrograph
    """
    format = format.lower()

    if format == "star":
        export_to_star(particles, output_path, micrograph_name)
    elif format == "json":
        export_to_json(particles, output_path, micrograph_name)
    elif format == "csv":
        export_to_csv(particles, output_path, micrograph_name)
    else:
        raise ValueError(f"Unknown format: {format}. Supported: star, json, csv")
