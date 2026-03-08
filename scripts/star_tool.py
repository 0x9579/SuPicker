#!/usr/bin/env python
"""STAR file inspection and splitting tool.

Usage:
    # Check how many images are referenced in a STAR file
    python scripts/star_tool.py info particles.star

    # Split a STAR file to keep only the first N images
    python scripts/star_tool.py split particles.star --num-images 10 --output val.star

    # Split a STAR file to keep the last N images (for train/val split)
    python scripts/star_tool.py split particles.star --num-images 10 --from-end --output val.star

    # Split into train and val sets at once
    python scripts/star_tool.py split-trainval particles.star --val-images 50 \
        --train-output train.star --val-output val.star
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supicker.data.star_parser import parse_star_file


def cmd_info(args):
    """Show info about a STAR file."""
    star_path = Path(args.star_file)
    if not star_path.exists():
        print(f"Error: File not found: {star_path}")
        sys.exit(1)

    particles_by_mic = parse_star_file(star_path)

    micrographs = list(particles_by_mic.keys())
    total_particles = sum(len(p) for p in particles_by_mic.values())
    counts = [len(p) for p in particles_by_mic.values()]

    print(f"STAR file: {star_path}")
    print(f"  Micrographs: {len(micrographs)}")
    print(f"  Total particles: {total_particles}")
    print(f"  Particles per image: min={min(counts)}, max={max(counts)}, avg={total_particles / len(counts):.1f}")
    print()

    if args.list:
        print("Micrograph list:")
        for i, mic in enumerate(micrographs, 1):
            print(f"  {i:4d}. {mic} ({len(particles_by_mic[mic])} particles)")


def _read_star_raw(star_path: Path) -> tuple[list[str], int, int, dict[str, int]]:
    """Read a STAR file and return raw lines, header info, and column indices.

    Returns:
        (all_lines, header_end_line, mic_col, column_indices)
        - all_lines: all lines from file
        - data_start: index of first data line
        - mic_col: column index for micrograph name
        - column_indices: dict of column_name -> column_index
    """
    with open(star_path, "r") as f:
        lines = f.readlines()

    column_indices = {}
    in_data = False
    in_loop = False
    data_start = 0
    header_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("data_particles", "data_"):
            in_data = True
            continue
        if in_data and stripped.startswith("loop_"):
            in_loop = True
            continue
        if in_loop and stripped.startswith("_"):
            parts = stripped.split()
            col_name = parts[0]
            col_idx = len(column_indices)
            column_indices[col_name] = col_idx
        elif in_loop and stripped and not stripped.startswith("_") and not stripped.startswith("#"):
            data_start = i
            break

    # Find micrograph column
    mic_col = None
    for name, idx in column_indices.items():
        if "micrograph" in name.lower():
            mic_col = idx
            break

    if mic_col is None:
        raise ValueError(f"No micrograph column found in {star_path}")

    return lines, data_start, mic_col, column_indices


def _write_star_subset(
    original_lines: list[str],
    data_start: int,
    mic_col: int,
    column_indices: dict[str, int],
    keep_micrographs: set[str],
    output_path: Path,
):
    """Write a subset of a STAR file, keeping only specified micrographs."""
    with open(output_path, "w") as f:
        # Write header (everything before data lines)
        for line in original_lines[:data_start]:
            f.write(line)

        # Write matching data lines
        kept = 0
        for line in original_lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("data_"):
                continue
            parts = stripped.split()
            if len(parts) < len(column_indices):
                continue
            mic_name = parts[mic_col]
            if mic_name in keep_micrographs:
                f.write(line)
                kept += 1

    return kept


def cmd_split(args):
    """Split a STAR file to keep only N images."""
    star_path = Path(args.star_file)
    if not star_path.exists():
        print(f"Error: File not found: {star_path}")
        sys.exit(1)

    # Parse to get micrograph list (ordered)
    particles_by_mic = parse_star_file(star_path)
    micrographs = list(particles_by_mic.keys())

    n = args.num_images
    if n > len(micrographs):
        print(f"Warning: Requested {n} images but STAR file only has {len(micrographs)}")
        n = len(micrographs)

    # Select images
    if args.from_end:
        selected = micrographs[-n:]
    else:
        selected = micrographs[:n]

    selected_set = set(selected)
    total_particles = sum(len(particles_by_mic[m]) for m in selected)

    # Read original file and write subset
    lines, data_start, mic_col, col_indices = _read_star_raw(star_path)
    output_path = Path(args.output)
    kept = _write_star_subset(lines, data_start, mic_col, col_indices, selected_set, output_path)

    print(f"Input:  {star_path}")
    print(f"  Total micrographs: {len(micrographs)}")
    print(f"Output: {output_path}")
    print(f"  Selected micrographs: {n} ({'last' if args.from_end else 'first'} {n})")
    print(f"  Particles written: {kept}")


def cmd_split_trainval(args):
    """Split a STAR file into train and val sets."""
    star_path = Path(args.star_file)
    if not star_path.exists():
        print(f"Error: File not found: {star_path}")
        sys.exit(1)

    particles_by_mic = parse_star_file(star_path)
    micrographs = list(particles_by_mic.keys())

    val_n = args.val_images
    if val_n >= len(micrographs):
        print(f"Error: val-images ({val_n}) must be less than total images ({len(micrographs)})")
        sys.exit(1)

    # Optionally shuffle before splitting
    if args.shuffle:
        import random
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(micrographs)

    val_mics = set(micrographs[-val_n:])
    train_mics = set(micrographs[:-val_n])

    lines, data_start, mic_col, col_indices = _read_star_raw(star_path)

    train_out = Path(args.train_output)
    val_out = Path(args.val_output)

    train_kept = _write_star_subset(lines, data_start, mic_col, col_indices, train_mics, train_out)
    val_kept = _write_star_subset(lines, data_start, mic_col, col_indices, val_mics, val_out)

    train_particles = sum(len(particles_by_mic[m]) for m in train_mics)
    val_particles = sum(len(particles_by_mic[m]) for m in val_mics)

    print(f"Input: {star_path} ({len(micrographs)} micrographs)")
    print(f"  Train: {train_out} ({len(train_mics)} micrographs, {train_kept} particles)")
    print(f"  Val:   {val_out} ({len(val_mics)} micrographs, {val_kept} particles)")


def main():
    parser = argparse.ArgumentParser(
        description="STAR file inspection and splitting tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show STAR file info")
    info_parser.add_argument("star_file", help="Path to STAR file")
    info_parser.add_argument("--list", action="store_true", help="List all micrographs")

    # split command
    split_parser = subparsers.add_parser("split", help="Extract N images from STAR file")
    split_parser.add_argument("star_file", help="Path to STAR file")
    split_parser.add_argument("--num-images", "-n", type=int, required=True, help="Number of images to keep")
    split_parser.add_argument("--from-end", action="store_true", help="Take images from end instead of beginning")
    split_parser.add_argument("--output", "-o", type=str, required=True, help="Output STAR file path")

    # split-trainval command
    tv_parser = subparsers.add_parser("split-trainval", help="Split into train/val sets")
    tv_parser.add_argument("star_file", help="Path to STAR file")
    tv_parser.add_argument("--val-images", type=int, required=True, help="Number of images for validation set")
    tv_parser.add_argument("--train-output", type=str, required=True, help="Output train STAR file")
    tv_parser.add_argument("--val-output", type=str, required=True, help="Output val STAR file")
    tv_parser.add_argument("--shuffle", action="store_true", help="Shuffle images before splitting")
    tv_parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle (default: 42)")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "split-trainval":
        cmd_split_trainval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
