from pathlib import Path
from typing import Union


def parse_star_file(star_path: Union[str, Path]) -> dict[str, list[dict]]:
    """Parse STAR file and group particles by micrograph.

    Supports both RELION and cryoSPARC formats.

    Args:
        star_path: Path to STAR file

    Returns:
        Dictionary mapping micrograph names to list of particles.
        Each particle has keys: 'x', 'y', 'class_id'
    """
    star_path = Path(star_path)
    particles_by_micrograph: dict[str, list[dict]] = {}

    with open(star_path, "r") as f:
        lines = f.readlines()

    # Find column indices
    column_indices = {}
    in_data_particles = False
    in_loop = False
    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "data_particles":
            in_data_particles = True
            continue
        if in_data_particles and line.startswith("loop_"):
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            # Parse column name
            parts = line.split()
            col_name = parts[0]
            col_idx = len(column_indices)
            column_indices[col_name] = col_idx
        elif in_loop and line and not line.startswith("_") and not line.startswith("#"):
            data_start = i
            break

    # Map common column name variants
    mic_col = None
    x_col = None
    y_col = None
    class_col = None

    for name, idx in column_indices.items():
        name_lower = name.lower()
        if "micrograph" in name_lower:
            mic_col = idx
        elif "coordinatex" in name_lower:
            x_col = idx
        elif "coordinatey" in name_lower:
            y_col = idx
        elif "class" in name_lower:
            class_col = idx

    if mic_col is None or x_col is None or y_col is None:
        raise ValueError(f"Missing required columns in STAR file: {star_path}")

    # Parse data lines
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("data_"):
            continue

        parts = line.split()
        if len(parts) < len(column_indices):
            continue

        # micrograph = Path(parts[mic_col]).name
        micrograph = parts[mic_col]
        x = float(parts[x_col])
        y = float(parts[y_col])
        class_id = int(parts[class_col]) - 1 if class_col is not None else 0

        if micrograph not in particles_by_micrograph:
            particles_by_micrograph[micrograph] = []

        particles_by_micrograph[micrograph].append({
            "x": x,
            "y": y,
            "class_id": class_id,
        })

    return particles_by_micrograph


def write_star_file(
    particles: list[dict],
    output_path: Union[str, Path],
    micrograph_name: str = "micrograph.tiff",
) -> None:
    """Write particles to STAR file format.

    Args:
        particles: List of particle dicts with 'x', 'y', 'class_id', 'score', 'width', 'height'
        output_path: Output file path
        micrograph_name: Name of the micrograph
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("data_particles\n\n")
        f.write("loop_\n")
        f.write("_rlnMicrographName #1\n")
        f.write("_rlnCoordinateX #2\n")
        f.write("_rlnCoordinateY #3\n")
        f.write("_rlnClassNumber #4\n")
        f.write("_rlnAutopickFigureOfMerit #5\n")
        f.write("_rlnParticleBoxSize #6\n")

        for p in particles:
            mic = p.get("micrograph", micrograph_name)
            x = p["x"]
            y = p["y"]
            cls = p.get("class_id", 0) + 1  # 1-indexed in STAR
            score = p.get("score", 1.0)
            box_size = max(p.get("width", 100), p.get("height", 100))
            f.write(f"{mic} {x:.2f} {y:.2f} {cls} {score:.4f} {box_size:.0f}\n")
