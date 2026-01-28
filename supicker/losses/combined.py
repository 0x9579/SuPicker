import torch
import torch.nn as nn

from supicker.config import LossConfig
from .focal_loss import FocalLoss, GaussianFocalLoss
from .regression_loss import RegL1Loss, SmoothL1Loss


class CombinedLoss(nn.Module):
    """Combined loss for CenterNet training."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

        # Build heatmap loss
        if config.heatmap_type == "focal":
            self.heatmap_loss = FocalLoss(
                alpha=config.focal_alpha, beta=config.focal_beta
            )
        elif config.heatmap_type == "gaussian_focal":
            self.heatmap_loss = GaussianFocalLoss(
                alpha=config.focal_alpha, beta=config.focal_beta
            )
        else:
            self.heatmap_loss = nn.MSELoss(reduction="mean")

        # Build size loss
        if config.size_type == "l1":
            self.size_loss = RegL1Loss()
        else:
            self.size_loss = SmoothL1Loss(beta=config.smooth_l1_beta)

        # Build offset loss
        if config.offset_type == "l1":
            self.offset_loss = RegL1Loss()
        else:
            self.offset_loss = SmoothL1Loss(beta=config.smooth_l1_beta)

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            outputs: Model outputs with 'heatmap', 'size', 'offset'
            targets: Ground truth with 'heatmap', 'size', 'offset', 'mask'

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        heatmap_loss = self.heatmap_loss(outputs["heatmap"], targets["heatmap"])
        size_loss = self.size_loss(outputs["size"], targets["size"], targets["mask"])
        offset_loss = self.offset_loss(outputs["offset"], targets["offset"], targets["mask"])

        total_loss = (
            self.config.heatmap_weight * heatmap_loss
            + self.config.size_weight * size_loss
            + self.config.offset_weight * offset_loss
        )

        loss_dict = {
            "heatmap_loss": heatmap_loss.item(),
            "size_loss": size_loss.item(),
            "offset_loss": offset_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
