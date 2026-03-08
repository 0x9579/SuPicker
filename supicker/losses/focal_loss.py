import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss for heatmap prediction in CenterNet.

    Based on CornerNet: https://arxiv.org/abs/1808.01244
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted heatmap (B, C, H, W), values in [0, 1]
            target: Target heatmap (B, C, H, W), values in [0, 1]

        Returns:
            Scalar loss value
        """
        # Cast to float32 for numerical stability (critical for AMP/FP16)
        pred = pred.float()
        target = target.float()

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # Positive samples (target == 1)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive loss
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask

        # Negative loss with reduced weight near positive samples
        neg_weight = torch.pow(1 - target, self.beta)
        neg_loss = -neg_weight * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_mask

        # Normalize by number of positive samples
        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


class GaussianFocalLoss(nn.Module):
    """Gaussian focal loss variant with quality-aware weighting."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian focal loss."""
        # Cast to float32 for numerical stability (critical for AMP/FP16)
        pred = pred.float()
        target = target.float()

        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        pos_mask = target.ge(0.99).float()
        neg_mask = target.lt(0.99).float()

        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask
        neg_weight = torch.pow(1 - target, self.beta)
        neg_loss = -neg_weight * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_mask

        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss
