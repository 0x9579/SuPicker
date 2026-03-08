from pathlib import Path
from typing import Optional, Union, Iterator
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
        self.config = config
        self.device = device

        # Distributed training setup
        self.is_distributed = config.distributed
        self.local_rank = config.local_rank
        self.world_size = config.world_size

        if self.is_distributed:
            self._setup_distributed()
            # Only override device with LOCAL_RANK when launched via torchrun
            # (i.e. LOCAL_RANK is explicitly set in the environment).
            # Otherwise, respect the user's --device argument.
            if "LOCAL_RANK" in os.environ:
                self.device = f"cuda:{self.local_rank}"

        # Move model to device
        self.model = model.to(self.device)

        # Convert BatchNorm to SyncBatchNorm for distributed training
        if self.is_distributed and config.sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Wrap model with DDP for distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Loss function
        self.criterion = CombinedLoss(config.loss)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Checkpoint manager (only on main process)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir) if self.is_main_process else None

        # Logger (only on main process)
        self.logger = Logger(log_dir, use_tensorboard=True) if self.is_main_process else None

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return not self.is_distributed or self.local_rank == 0

    def _setup_distributed(self) -> None:
        """Initialize distributed training process group."""
        if not dist.is_initialized():
            # Set defaults for single-node if not set
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            self.local_rank = int(os.environ.get("LOCAL_RANK", self.config.local_rank))
            self.world_size = int(os.environ.get("WORLD_SIZE", self.config.world_size))
            rank = int(os.environ.get("RANK", self.local_rank))

            backend = self.config.dist_backend
            if backend == "nccl" and not torch.cuda.is_available():
                backend = "gloo"

            dist.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=self.world_size,
                rank=rank,
            )

            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)

    def _reduce_value(self, value: torch.Tensor, average: bool = True) -> torch.Tensor:
        """Reduce a value across all processes.

        Args:
            value: Value to reduce
            average: Whether to average (True) or sum (False)

        Returns:
            Reduced value
        """
        if not self.is_distributed:
            return value

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=self.device)

        dist.all_reduce(value, op=dist.ReduceOp.SUM)

        if average:
            value = value / self.world_size

        return value

    def _get_model_for_saving(self) -> nn.Module:
        """Get the underlying model (unwrap DDP if needed)."""
        if self.is_distributed:
            return self.model.module
        return self.model

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
        self,
        val_loader: Union[DataLoader, list[dict[str, torch.Tensor]]],
        compute_metrics: bool = False,
        distance_threshold: float = 10.0,
    ) -> Union[float, tuple[float, Optional[dict]]]:
        """Run validation.

        Args:
            val_loader: Validation dataloader or list of batches
            compute_metrics: Whether to compute detection metrics
            distance_threshold: Distance threshold for metric matching

        Returns:
            Average validation loss, or tuple of (loss, metrics_dict) if compute_metrics=True
        """
        from supicker.utils.metrics import MetricAggregator

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        metric_aggregator = MetricAggregator(distance_threshold) if compute_metrics else None

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

                # Compute detection metrics if requested
                if compute_metrics and metric_aggregator is not None:
                    predictions = self._extract_predictions(outputs)
                    ground_truth = batch.get("particles", [])
                    # Handle batched particles (list of lists)
                    if ground_truth and isinstance(ground_truth[0], list):
                        for preds_batch, gts_batch in zip(
                            self._split_predictions_by_batch(predictions, image.shape[0]),
                            ground_truth,
                        ):
                            metric_aggregator.add_image(preds_batch, gts_batch)
                    else:
                        metric_aggregator.add_image(predictions, ground_truth)

        avg_loss = total_loss / max(num_batches, 1)

        if compute_metrics and metric_aggregator is not None:
            metrics = metric_aggregator.compute_aggregate()
            metrics_dict = {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "avg_distance": metrics.avg_distance,
            }
            if metrics.ap is not None:
                metrics_dict["ap"] = metrics.ap
            return avg_loss, metrics_dict

        return avg_loss

    def _extract_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """Extract particle predictions from model outputs.

        Args:
            outputs: Model outputs with 'heatmap', 'size', 'offset' keys
            score_threshold: Minimum score threshold

        Returns:
            List of particle dictionaries with x, y, score keys
        """
        import torch.nn.functional as F

        heatmap = outputs["heatmap"]
        batch_size, num_classes, h, w = heatmap.shape

        # Find local maxima using max pooling
        hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        keep = (heatmap == hmax) & (heatmap >= score_threshold)

        particles = []
        output_stride = 4  # Default output stride

        for b in range(batch_size):
            for c in range(num_classes):
                y_coords, x_coords = torch.where(keep[b, c])
                for y, x in zip(y_coords.tolist(), x_coords.tolist()):
                    score = float(heatmap[b, c, y, x])

                    # Get offset if available
                    offset_x, offset_y = 0.0, 0.0
                    if "offset" in outputs and outputs["offset"] is not None:
                        offset_x = float(outputs["offset"][b, 0, y, x])
                        offset_y = float(outputs["offset"][b, 1, y, x])

                    # Convert to image coordinates
                    img_x = (x + offset_x) * output_stride
                    img_y = (y + offset_y) * output_stride

                    particles.append({
                        "x": img_x,
                        "y": img_y,
                        "score": score,
                        "class_id": c,
                        "batch_idx": b,
                    })

        return particles

    def _split_predictions_by_batch(
        self,
        predictions: list[dict],
        batch_size: int,
    ) -> list[list[dict]]:
        """Split predictions into per-batch lists.

        Args:
            predictions: List of predictions with batch_idx
            batch_size: Number of batches

        Returns:
            List of prediction lists, one per batch
        """
        result = [[] for _ in range(batch_size)]
        for pred in predictions:
            batch_idx = pred.get("batch_idx", 0)
            if batch_idx < batch_size:
                result[batch_idx].append(pred)
        return result

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

        avg_loss = total_loss / max(num_batches, 1)

        # Synchronize loss across processes
        if self.is_distributed:
            avg_loss_tensor = self._reduce_value(
                torch.tensor(avg_loss, device=self.device)
            )
            avg_loss = avg_loss_tensor.item()

        return avg_loss

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

        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch

                # Set epoch for distributed sampler (ensures different shuffling per epoch)
                if self.is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

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

                # Log (only on main process)
                if self.is_main_process and self.logger is not None:
                    self.logger.log_epoch(
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        lr=lr,
                    )

                # Save checkpoint (only on main process)
                if self.is_main_process and (epoch + 1) % self.config.save_interval == 0:
                    loss_for_save = val_loss if val_loss is not None else train_loss
                    self.checkpoint_manager.save(
                        model=self._get_model_for_saving(),
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        loss=loss_for_save,
                        scheduler=self.scheduler,
                    )

        finally:
            # Cleanup
            if self.is_main_process and self.logger is not None:
                self.logger.close()

            # Destroy process group for distributed training
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint and resume training.

        Args:
            path: Path to checkpoint file
        """
        # For distributed training, we need a checkpoint manager temporarily
        if self.checkpoint_manager is None:
            temp_manager = CheckpointManager(Path(path).parent)
            checkpoint = temp_manager.load(
                path,
                model=self._get_model_for_saving(),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        else:
            checkpoint = self.checkpoint_manager.load(
                path,
                model=self._get_model_for_saving(),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("loss", float("inf"))
