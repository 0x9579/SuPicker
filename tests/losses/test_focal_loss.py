import pytest
import torch


def test_focal_loss_shape():
    from supicker.losses.focal_loss import FocalLoss

    loss_fn = FocalLoss(alpha=2.0, beta=4.0)
    pred = torch.sigmoid(torch.randn(2, 3, 64, 64))
    target = torch.zeros(2, 3, 64, 64)
    # Add some positive samples
    target[0, 0, 32, 32] = 1.0
    target[1, 1, 16, 16] = 1.0

    loss = loss_fn(pred, target)
    assert loss.ndim == 0  # Scalar
    assert loss >= 0


def test_focal_loss_positive_sample():
    from supicker.losses.focal_loss import FocalLoss

    loss_fn = FocalLoss(alpha=2.0, beta=4.0)

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 2, 2] = 1.0

    # Good prediction: high confidence at positive, low at negatives
    pred_good = torch.ones(1, 1, 4, 4) * 0.1  # Low at negatives (correct)
    pred_good[0, 0, 2, 2] = 0.9  # High at positive (correct)
    loss_good = loss_fn(pred_good, target)

    # Bad prediction: low confidence at positive, low at negatives
    pred_bad = torch.ones(1, 1, 4, 4) * 0.1  # Low at negatives (correct)
    pred_bad[0, 0, 2, 2] = 0.1  # Low at positive (incorrect)
    loss_bad = loss_fn(pred_bad, target)

    assert loss_bad > loss_good
