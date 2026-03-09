from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
from datetime import datetime


class CheckpointManager:
    """Manage model checkpoints for saving and loading."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_best: bool = True,
        max_checkpoints: Optional[int] = 5,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to track and save best checkpoint
            max_checkpoints: Maximum number of checkpoints to keep (None for unlimited)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.max_checkpoints = max_checkpoints
        self.best_loss = float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            loss: Current loss value
            scheduler: Optional learning rate scheduler
            **kwargs: Additional data to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **kwargs,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save regular checkpoint with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epoch_{epoch:03d}_loss_{loss:.4f}_{timestamp}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        # Save best checkpoint if this is the best
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(path)

    def load(
        self,
        path: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
    ) -> dict:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint file
            model: Optional model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to

        Returns:
            Dictionary with checkpoint data
        """
        checkpoint = torch.load(path, map_location=device)

        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if not found
        """
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return str(best_path)
        return None

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None if not found
        """
        checkpoints = list(self.checkpoint_dir.glob("epoch_*.pt"))
        if not checkpoints:
            return None

        # Sort by modification time to get the truly latest file
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        return str(checkpoints[-1])

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        if self.max_checkpoints is None:
            return

        checkpoints = list(self.checkpoint_dir.glob("epoch_*.pt"))
        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by modification time to keep the most recent checkpoints
        # regardless of epoch number (useful when restarting training)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest checkpoints
        for ckpt in checkpoints[: -self.max_checkpoints]:
            ckpt.unlink()
