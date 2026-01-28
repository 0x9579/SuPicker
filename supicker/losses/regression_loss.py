import torch
import torch.nn as nn


class RegL1Loss(nn.Module):
    """L1 loss computed only at specified locations."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute L1 loss at masked locations.

        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)
            mask: Binary mask (B, H, W) indicating valid locations

        Returns:
            Scalar loss value
        """
        # Expand mask to match prediction channels
        mask = mask.unsqueeze(1).expand_as(pred)

        loss = torch.abs(pred - target) * mask
        num_pos = mask.sum()

        if num_pos == 0:
            return loss.sum()
        return loss.sum() / num_pos


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss computed only at specified locations."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute Smooth L1 loss at masked locations.

        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)
            mask: Binary mask (B, H, W) indicating valid locations

        Returns:
            Scalar loss value
        """
        mask = mask.unsqueeze(1).expand_as(pred)
        diff = torch.abs(pred - target)

        # Smooth L1 formula
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )
        loss = loss * mask

        num_pos = mask.sum()
        if num_pos == 0:
            return loss.sum()
        return loss.sum() / num_pos
