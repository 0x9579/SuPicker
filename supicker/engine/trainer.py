from pathlib import Path
from typing import Optional, Union, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from supicker.config import TrainingConfig
from supicker.losses import CombinedLoss
from supicker.utils.checkpoint import CheckpointManager
from supicker.utils.logger import Logger


class Trainer:
    """Trainer for particle detection model."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        checkpoint_dir: Union[str, Path],
        log_dir: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function
        self.criterion = CombinedLoss(config.loss)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Logger
        self.logger = Logger(log_dir, use_tensorboard=True)

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform a single training step.

        Args:
            batch: Dictionary with 'image', 'heatmap', 'size', 'offset', 'mask'

        Returns:
            Tuple of (loss, loss_dict)
        """
        self.model.train()

        # Move to device
        image = batch["image"].to(self.device)
        targets = {
            "heatmap": batch["heatmap"].to(self.device),
            "size": batch["size"].to(self.device),
            "offset": batch["offset"].to(self.device),
            "mask": batch["mask"].to(self.device),
        }

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(image)

        # Compute loss
        loss, loss_dict = self.criterion(outputs, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss, loss_dict

    def validate(
        self, val_loader: Union[DataLoader, list[dict[str, torch.Tensor]]]
    ) -> float:
        """Run validation.

        Args:
            val_loader: Validation dataloader or list of batches

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                image = batch["image"].to(self.device)
                targets = {
                    "heatmap": batch["heatmap"].to(self.device),
                    "size": batch["size"].to(self.device),
                    "offset": batch["offset"].to(self.device),
                    "mask": batch["mask"].to(self.device),
                }

                # Forward pass
                outputs = self.model(image)
                loss, _ = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training dataloader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            loss, _ = self.train_step(batch)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> None:
        """Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            epochs: Number of epochs (overrides config if provided)
        """
        epochs = epochs or self.config.epochs

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = None
            if val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Get current learning rate
            lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.logger.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
            )

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                loss_for_save = val_loss if val_loss is not None else train_loss
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=loss_for_save,
                    scheduler=self.scheduler,
                )

        self.logger.close()

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint and resume training.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = self.checkpoint_manager.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("loss", float("inf"))
