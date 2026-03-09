#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supicker.utils.coordinate_validation import generate_coordinate_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate STAR coordinates on one micrograph")
    parser.add_argument("--image", required=True)
    parser.add_argument("--star", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--micrograph-name")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--radius", type=int, default=6)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
