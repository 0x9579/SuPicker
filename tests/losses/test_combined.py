import pytest
import torch
from supicker.config import LossConfig


def test_combined_loss():
    from supicker.losses.combined import CombinedLoss

    config = LossConfig()
    loss_fn = CombinedLoss(config)

    outputs = {
        "heatmap": torch.sigmoid(torch.randn(2, 3, 64, 64)),
        "size": torch.abs(torch.randn(2, 2, 64, 64)),
        "offset": torch.randn(2, 2, 64, 64),
    }

    targets = {
        "heatmap": torch.zeros(2, 3, 64, 64),
        "size": torch.zeros(2, 2, 64, 64),
        "offset": torch.zeros(2, 2, 64, 64),
        "mask": torch.zeros(2, 64, 64),
    }
    # Add positive sample
    targets["heatmap"][0, 0, 32, 32] = 1.0
    targets["mask"][0, 32, 32] = 1.0

    loss, loss_dict = loss_fn(outputs, targets)

    assert loss.ndim == 0
    assert "heatmap_loss" in loss_dict
    assert "size_loss" in loss_dict
    assert "offset_loss" in loss_dict
