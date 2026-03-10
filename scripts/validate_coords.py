#!/usr/bin/env python

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supicker.utils.coordinate_validation import generate_coordinate_overlay


def find_image_pairs(image_dir: Path, star_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = []
    for pattern in ("*.tiff", "*.tif", "*.mrc", "*.png", "*.jpg"):
        image_paths.extend(sorted(image_dir.glob(pattern)))

    pairs = []
    for image_path in image_paths:
        star_path = star_dir / f"{image_path.stem}.star"
        if star_path.exists():
            pairs.append((image_path, star_path))
    return pairs


def write_summary(summary_path: Path, file_stats: list[dict], totals: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = summary_path.suffix.lower()
    if suffix == ".json":
        payload = dict(totals)
        payload["files"] = file_stats
        summary_path.write_text(json.dumps(payload, indent=2))
        return
    if suffix == ".csv":
        if not file_stats:
            summary_path.write_text("resolved_micrograph_name,particle_count,out_of_bounds_count,output_path\n")
            return
        fieldnames = [
            "resolved_micrograph_name",
            "particle_count",
            "out_of_bounds_count",
            "image_width",
            "image_height",
            "output_path",
        ]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in file_stats:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return
    raise ValueError("summary output must end with .json or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate STAR coordinates on one micrograph")
    single_group = parser.add_argument_group("single-file mode")
    single_group.add_argument("--image")
    single_group.add_argument("--star")
    single_group.add_argument("--output")
    batch_group = parser.add_argument_group("directory mode")
    batch_group.add_argument("--image-dir")
    batch_group.add_argument("--star-dir")
    batch_group.add_argument("--output-dir")
    parser.add_argument("--summary-output")
    parser.add_argument("--micrograph-name")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--radius", type=int, default=6)
    args = parser.parse_args()

    if args.image and args.star and args.output:
        stats = generate_coordinate_overlay(
            image_path=args.image,
            star_path=args.star,
            output_path=args.output,
            micrograph_name=args.micrograph_name,
            flip_y=args.flip_y,
            radius=args.radius,
        )

        for key, value in stats.items():
            print(f"{key}: {value}")
        return

    if args.image_dir and args.star_dir and args.output_dir:
        if args.micrograph_name is not None:
            parser.error("--micrograph-name is only supported in single-file mode")
        pairs = find_image_pairs(Path(args.image_dir), Path(args.star_dir))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_files = 0
        total_particles = 0
        total_out_of_bounds = 0
        file_stats = []
        for image_path, star_path in pairs:
            output_path = output_dir / f"{image_path.name}.png"
            stats = generate_coordinate_overlay(
                image_path=image_path,
                star_path=star_path,
                output_path=output_path,
                micrograph_name=image_path.name,
                flip_y=args.flip_y,
                radius=args.radius,
            )
            processed_files += 1
            total_particles += int(stats["particle_count"])
            total_out_of_bounds += int(stats["out_of_bounds_count"])
            file_stats.append(stats)

        totals = {
            "processed_files": processed_files,
            "total_particles": total_particles,
            "total_out_of_bounds": total_out_of_bounds,
        }

        if args.summary_output:
            write_summary(Path(args.summary_output), file_stats, totals)

        print(f"processed_files: {processed_files}")
        print(f"total_particles: {total_particles}")
        print(f"total_out_of_bounds: {total_out_of_bounds}")
        return

    parser.error(
        "Provide either --image/--star/--output or --image-dir/--star-dir/--output-dir"
    )


if __name__ == "__main__":
    main()
