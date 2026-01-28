import pytest
import torch
import tempfile
from pathlib import Path


def test_trainer_train_step():
    from supicker.engine.trainer import Trainer
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, TrainingConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    training_config = TrainingConfig(batch_size=2)

    model = Detector(model_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            config=training_config,
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
        )

        # Create dummy batch
        batch = {
            "image": torch.randn(2, 1, 128, 128),
            "heatmap": torch.zeros(2, 1, 32, 32),
            "size": torch.zeros(2, 2, 32, 32),
            "offset": torch.zeros(2, 2, 32, 32),
            "mask": torch.zeros(2, 32, 32),
        }
        batch["heatmap"][:, 0, 16, 16] = 1.0
        batch["mask"][:, 16, 16] = 1.0

        loss, loss_dict = trainer.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "heatmap_loss" in loss_dict


def test_trainer_validate():
    from supicker.engine.trainer import Trainer
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, TrainingConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    training_config = TrainingConfig(batch_size=2)

    model = Detector(model_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            config=training_config,
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
        )

        # Create dummy dataloader
        batches = [
            {
                "image": torch.randn(2, 1, 128, 128),
                "heatmap": torch.zeros(2, 1, 32, 32),
                "size": torch.zeros(2, 2, 32, 32),
                "offset": torch.zeros(2, 2, 32, 32),
                "mask": torch.zeros(2, 32, 32),
            }
        ]
        batches[0]["heatmap"][:, 0, 16, 16] = 1.0
        batches[0]["mask"][:, 16, 16] = 1.0

        val_loss = trainer.validate(batches)

        assert isinstance(val_loss, float)
        assert val_loss >= 0
