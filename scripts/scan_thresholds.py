#!/usr/bin/env python
"""Scan validation metrics across multiple score thresholds."""

import argparse
from pathlib import Path
from typing import Optional

import torch

from supicker.config import BackboneConfig, ConvNeXtVariant, ModelConfig
from supicker.config.data import AugmentationConfig
from supicker.data import build_transforms, create_dataloader
from supicker.engine import Predictor
from supicker.models import Detector
from supicker.utils.metrics import MetricAggregator


DEFAULT_THRESHOLDS = "0.03,0.05,0.08,0.1,0.12,0.15,0.2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan validation score thresholds")
    parser.add_argument("--val-images", type=str, required=True, help="Validation image directory")
    parser.add_argument("--val-star", type=str, required=True, help="Validation STAR file")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=str, help="Checkpoint path")
    checkpoint_group.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument(
        "--backbone",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="ConvNeXt backbone variant",
    )
    parser.add_argument("--num-classes", type=int, default=1, help="Number of particle classes")
    parser.add_argument("--batch-size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS, help="Comma-separated score thresholds")
    parser.add_argument("--checkpoint-pattern", type=str, default="*.pt", help="Glob pattern when scanning checkpoint directory")
    parser.add_argument("--distance-threshold", type=float, default=20.0, help="Matching distance threshold in pixels")
    parser.add_argument("--nms-radius", type=float, default=20.0, help="NMS radius in pixels")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("No thresholds provided")
    return thresholds


def format_scan_results(rows: list[dict]) -> str:
    include_checkpoint = any("checkpoint" in row for row in rows)
    prefix = f"{'Checkpoint':>20} " if include_checkpoint else ""
    header = f"{prefix}{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg Preds':>10}"
    lines = [header, "-" * len(header)]
    for row in rows:
        checkpoint = f"{row['checkpoint']:>20} " if include_checkpoint else ""
        lines.append(
            f"{checkpoint}{row['threshold']:>10.3f} {row['precision']:>10.3f} {row['recall']:>10.3f} "
            f"{row['f1_score']:>10.3f} {row['avg_predictions']:>10.2f}"
        )
    return "\n".join(lines)


def format_best_summary(rows: list[dict]) -> str:
    header = f"{'Checkpoint':>20} {'Best Th':>10} {'Best F1':>10} {'Precision':>10} {'Recall':>10} {'Avg Preds':>10}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['checkpoint']:>20} {row['threshold']:>10.3f} {row['f1_score']:>10.3f} "
            f"{row['precision']:>10.3f} {row['recall']:>10.3f} {row['avg_predictions']:>10.2f}"
        )
    return "\n".join(lines)


def evaluate_thresholds(
    batches: list[dict],
    thresholds: list[float],
    distance_threshold: float,
    nms_radius: float,
    checkpoint_name: Optional[str] = None,
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        aggregator = MetricAggregator(distance_threshold=distance_threshold)
        prediction_counts = []

        for batch in batches:
            predictions = Predictor.decode_outputs(
                batch["outputs"],
                score_threshold=threshold,
                nms_enabled=nms_radius > 0,
                nms_radius=nms_radius,
            )
            per_batch_predictions = split_predictions_by_batch(
                predictions,
                batch_size=len(batch["particles"]),
            )
            for preds, targets in zip(per_batch_predictions, batch["particles"]):
                aggregator.add_image(preds, targets)
                prediction_counts.append(len(preds))

        metrics = aggregator.compute_aggregate(compute_ap=False)
        avg_predictions = sum(prediction_counts) / max(len(prediction_counts), 1)
        rows.append(
            {
                **({"checkpoint": checkpoint_name} if checkpoint_name is not None else {}),
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "avg_predictions": avg_predictions,
            }
        )

    return rows


def find_checkpoints(checkpoint_dir: Path, checkpoint_pattern: str) -> list[Path]:
    checkpoints = sorted(path for path in checkpoint_dir.glob(checkpoint_pattern) if path.is_file())
    if not checkpoints:
        raise ValueError(
            f"No checkpoints found in {checkpoint_dir} matching pattern {checkpoint_pattern}"
        )
    return checkpoints


def summarize_best_rows(rows: list[dict]) -> list[dict]:
    best_by_checkpoint = {}
    for row in rows:
        checkpoint = row["checkpoint"]
        best = best_by_checkpoint.get(checkpoint)
        if best is None or row["f1_score"] > best["f1_score"]:
            best_by_checkpoint[checkpoint] = row
    return [best_by_checkpoint[key] for key in sorted(best_by_checkpoint)]


def split_predictions_by_batch(predictions: list[dict], batch_size: int) -> list[list[dict]]:
    result = [[] for _ in range(batch_size)]
    for pred in predictions:
        batch_idx = int(pred.get("batch_idx", 0))
        if batch_idx < batch_size:
            result[batch_idx].append(pred)
    return result


def collect_validation_batches(model, dataloader, device: str) -> list[dict]:
    model.eval()
    batches = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            outputs = model(image)
            batches.append(
                {
                    "outputs": {name: tensor.detach().cpu() for name, tensor in outputs.items()},
                    "particles": batch.get("particles", []),
                }
            )
    return batches


def build_eval_transforms():
    config = AugmentationConfig(
        crop_size=0,
        horizontal_flip=False,
        vertical_flip=False,
        rotation_90=False,
        random_rotation=False,
        brightness=False,
        contrast=False,
        gaussian_noise=False,
    )
    return build_transforms(config)


def main() -> None:
    args = build_parser().parse_args()
    variant_map = {
        "tiny": ConvNeXtVariant.TINY,
        "small": ConvNeXtVariant.SMALL,
        "base": ConvNeXtVariant.BASE,
    }

    model_config = ModelConfig(
        backbone=BackboneConfig(
            variant=variant_map[args.backbone],
            pretrained=False,
            in_channels=1,
        ),
    )
    model_config.head.num_classes = args.num_classes
    model = Detector(model_config).to(args.device)

    dataloader = create_dataloader(
        image_dir=args.val_images,
        star_file=args.val_star,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        transforms=build_eval_transforms(),
        num_classes=args.num_classes,
        distributed=False,
    )

    thresholds = parse_thresholds(args.thresholds)
    checkpoint_paths = (
        [Path(args.checkpoint)]
        if args.checkpoint is not None
        else find_checkpoints(Path(args.checkpoint_dir), args.checkpoint_pattern)
    )

    all_rows = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        batches = collect_validation_batches(model, dataloader, device=args.device)
        all_rows.extend(
            evaluate_thresholds(
                batches=batches,
                thresholds=thresholds,
                distance_threshold=args.distance_threshold,
                nms_radius=args.nms_radius,
                checkpoint_name=checkpoint_path.name,
            )
        )

    print(format_scan_results(all_rows))
    if len(checkpoint_paths) > 1:
        print()
        print(format_best_summary(summarize_best_rows(all_rows)))


if __name__ == "__main__":
    main()
