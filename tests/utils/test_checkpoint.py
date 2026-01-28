import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


def test_save_checkpoint():
    from supicker.utils.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.5,
        )

        assert Path(path).exists()
        assert "epoch_005" in path


def test_load_checkpoint():
    from supicker.utils.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        path = manager.save(model=model, optimizer=optimizer, epoch=10, loss=0.25)

        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Load checkpoint
        checkpoint = manager.load(path, model=new_model, optimizer=new_optimizer)

        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.25


def test_load_best_checkpoint():
    from supicker.utils.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir, save_best=True)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save multiple checkpoints
        manager.save(model=model, optimizer=optimizer, epoch=1, loss=0.5)
        manager.save(model=model, optimizer=optimizer, epoch=2, loss=0.3)
        manager.save(model=model, optimizer=optimizer, epoch=3, loss=0.4)

        # Best should be epoch 2 with loss 0.3
        best_path = manager.get_best_checkpoint()
        assert best_path is not None
        assert "best" in best_path


def test_load_latest_checkpoint():
    from supicker.utils.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        manager.save(model=model, optimizer=optimizer, epoch=1, loss=0.5)
        manager.save(model=model, optimizer=optimizer, epoch=2, loss=0.3)

        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "epoch_002" in latest
