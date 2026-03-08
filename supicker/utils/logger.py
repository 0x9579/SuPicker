from pathlib import Path
from typing import Optional, Union
from datetime import datetime


class Logger:
    """Logger for training with optional TensorBoard support."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        print_freq: int = 1,
        experiment_name: Optional[str] = None,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard logging
            print_freq: Frequency of console output (epochs)
            experiment_name: Optional experiment name for TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.print_freq = print_freq
        self.writer = None

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the metric
            value: Value to log
            step: Current step/epoch
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """Log multiple scalar values.

        Args:
            metrics: Dictionary of metric names to values
            step: Current step/epoch
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, value, step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[dict[str, float]] = None,
        val_metrics: Optional[dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Log epoch summary with console output.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Optional validation loss
            train_metrics: Optional training metrics
            val_metrics: Optional validation metrics
            lr: Optional learning rate
        """
        # Console output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"

        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"

        if val_metrics:
            if "precision" in val_metrics:
                msg += f" | P: {val_metrics['precision']:.3f}"
            if "recall" in val_metrics:
                msg += f" | R: {val_metrics['recall']:.3f}"
            if "f1_score" in val_metrics:
                msg += f" | F1: {val_metrics['f1_score']:.3f}"
            if "max_score" in val_metrics:
                msg += f" | Max_Score: {val_metrics['max_score']:.3f}"

        if lr is not None:
            msg += f" | LR: {lr:.2e}"

        print(msg)

        # TensorBoard logging
        self.log_scalar("train/loss", train_loss, epoch)
        if val_loss is not None:
            self.log_scalar("val/loss", val_loss, epoch)
        if lr is not None:
            self.log_scalar("train/lr", lr, epoch)

        if train_metrics:
            self.log_scalars(train_metrics, epoch, prefix="train")
        if val_metrics:
            self.log_scalars(val_metrics, epoch, prefix="val")

    def log_image(self, tag: str, image, step: int) -> None:
        """Log an image.

        Args:
            tag: Name of the image
            image: Image tensor (C, H, W) or (H, W)
            step: Current step
        """
        if self.writer is not None:
            self.writer.add_image(tag, image, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram.

        Args:
            tag: Name of the histogram
            values: Values to create histogram from
            step: Current step
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def close(self) -> None:
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
