import pytest
import torch


def test_l1_loss_at_locations():
    from supicker.losses.regression_loss import RegL1Loss

    loss_fn = RegL1Loss()
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    mask = torch.zeros(2, 64, 64)
    mask[0, 32, 32] = 1
    mask[1, 16, 16] = 1

    loss = loss_fn(pred, target, mask)
    assert loss.ndim == 0
    assert loss >= 0


def test_smooth_l1_loss_at_locations():
    from supicker.losses.regression_loss import SmoothL1Loss

    loss_fn = SmoothL1Loss(beta=1.0)
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    mask = torch.zeros(2, 64, 64)
    mask[0, 32, 32] = 1

    loss = loss_fn(pred, target, mask)
    assert loss.ndim == 0
    assert loss >= 0
