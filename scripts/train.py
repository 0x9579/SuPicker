#!/usr/bin/env python
"""Training script for SuPicker particle detector."""

import argparse
from pathlib import Path

import torch

from supicker.config import (
    ModelConfig,
    BackboneConfig,
    ConvNeXtVariant,
    DataConfig,
    TrainingConfig,
    AugmentationConfig,
)
from supicker.models import Detector
from supicker.data import ParticleDataset, create_dataloader, build_transforms
from supicker.engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SuPicker particle detector")

    # Data arguments
    parser.add_argument(
        "--train-images", type=str, required=True, help="Directory containing training images"
    )
    parser.add_argument(
        "--train-star", type=str, required=True, help="STAR file with training particle annotations"
    )
    parser.add_argument(
        "--val-images", type=str, default=None, help="Directory containing validation images"
    )
    parser.add_argument(
        "--val-star", type=str, default=None, help="STAR file with validation particle annotations"
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="ConvNeXt backbone variant",
    )
    parser.add_argument(
        "--num-classes", type=int, default=1, help="Number of particle classes"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained backbone"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")

    # Augmentation arguments
    parser.add_argument("--no-augmentation", action="store_true", help="Disable augmentation")

    # Output arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Checkpoint save interval (epochs)"
    )

    # Other arguments
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Distributed training arguments
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend (nccl for GPU, gloo for CPU)",
    )
    parser.add_argument(
        "--no-sync-bn",
        action="store_true",
        help="Disable SyncBatchNorm in distributed training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create configs
    variant_map = {
        "tiny": ConvNeXtVariant.TINY,
        "small": ConvNeXtVariant.SMALL,
        "base": ConvNeXtVariant.BASE,
    }

    model_config = ModelConfig(
        backbone=BackboneConfig(
            variant=variant_map[args.backbone],
            pretrained=args.pretrained,
            in_channels=1,
        ),
    )
    model_config.head.num_classes = args.num_classes

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        distributed=args.distributed,
        dist_backend=args.dist_backend,
        sync_bn=not args.no_sync_bn,
    )

    # Create model
    print(f"Creating model with {args.backbone} backbone...")
    model = Detector(model_config)

    # Create transforms
    if args.no_augmentation:
        transforms = None
    else:
        aug_config = AugmentationConfig()
        transforms = build_transforms(aug_config)

    # Determine if we're the main process for printing
    import os
    if args.distributed:
        import torch.distributed as dist
        if not dist.is_initialized():
            # Set defaults for single-node if not set (helps when not using torchrun)
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))

            dist.init_process_group(
                backend=args.dist_backend,
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # Create dataloaders
    if is_main:
        print(f"Loading training data from {args.train_images}...")
    train_loader = create_dataloader(
        image_dir=args.train_images,
        star_file=args.train_star,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        transforms=transforms,
        num_classes=args.num_classes,
        distributed=args.distributed,
    )
    if is_main:
        print(f"  Found {len(train_loader.dataset)} training images")

    val_loader = None
    if args.val_images and args.val_star:
        if is_main:
            print(f"Loading validation data from {args.val_images}...")
        val_loader = create_dataloader(
            image_dir=args.val_images,
            star_file=args.val_star,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            transforms=None,  # No augmentation for validation
            num_classes=args.num_classes,
            distributed=args.distributed,
        )
        if is_main:
            print(f"  Found {len(val_loader.dataset)} validation images")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    if is_main:
        print(f"Starting training for {args.epochs} epochs...")
        print(f"  Device: {args.device}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        if args.distributed:
            print(f"  Distributed: True (backend={args.dist_backend})")
        print()

    trainer.train(train_loader, val_loader)

    if is_main:
        print("Training complete!")


if __name__ == "__main__":
    main()
