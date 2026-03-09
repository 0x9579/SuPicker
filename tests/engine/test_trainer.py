import pytest
import torch
import tempfile
from pathlib import Path
import torch.nn as nn


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


class DummyModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, x):
        return self.outputs


class DummyCriterion:
    def __call__(self, outputs, targets):
        return torch.tensor(0.0), {}


def test_training_config_validation_metric_defaults():
    from supicker.config import TrainingConfig

    config = TrainingConfig()

    assert config.val_score_threshold == 0.1
    assert config.val_distance_threshold == 20.0
    assert config.val_nms_radius == 20.0


def test_validate_uses_configurable_score_threshold_and_nms():
    from supicker.engine.trainer import Trainer
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, TrainingConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    training_config = TrainingConfig(
        batch_size=1,
        val_score_threshold=0.1,
        val_distance_threshold=20.0,
        val_nms_radius=20.0,
    )

    model = Detector(model_config)

    outputs = {
        "heatmap": torch.zeros(1, 1, 32, 32),
        "size": torch.zeros(1, 2, 32, 32),
        "offset": torch.zeros(1, 2, 32, 32),
    }
    outputs["heatmap"][0, 0, 10, 10] = 0.20
    outputs["heatmap"][0, 0, 10, 12] = 0.19

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            config=training_config,
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
            device="cpu",
        )
        trainer.model = DummyModel(outputs)
        trainer.criterion = DummyCriterion()

        batches = [{
            "image": torch.randn(1, 1, 128, 128),
            "heatmap": torch.zeros(1, 1, 32, 32),
            "size": torch.zeros(1, 2, 32, 32),
            "offset": torch.zeros(1, 2, 32, 32),
            "mask": torch.zeros(1, 32, 32),
            "particles": [[{"x": 40.0, "y": 40.0}]],
        }]

        _, metrics = trainer.validate(batches, compute_metrics=True)

        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1_score"] == pytest.approx(1.0)


def test_validate_respects_configurable_distance_threshold():
    from supicker.engine.trainer import Trainer
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, TrainingConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    training_config = TrainingConfig(
        batch_size=1,
        val_score_threshold=0.1,
        val_distance_threshold=20.0,
        val_nms_radius=20.0,
    )

    model = Detector(model_config)

    outputs = {
        "heatmap": torch.zeros(1, 1, 32, 32),
        "size": torch.zeros(1, 2, 32, 32),
        "offset": torch.zeros(1, 2, 32, 32),
    }
    outputs["heatmap"][0, 0, 10, 10] = 0.95
    outputs["offset"][0, 0, 10, 10] = 0.0
    outputs["offset"][0, 1, 10, 10] = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            config=training_config,
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
            device="cpu",
        )
        trainer.model = DummyModel(outputs)
        trainer.criterion = DummyCriterion()

        batches = [{
            "image": torch.randn(1, 1, 128, 128),
            "heatmap": torch.zeros(1, 1, 32, 32),
            "size": torch.zeros(1, 2, 32, 32),
            "offset": torch.zeros(1, 2, 32, 32),
            "mask": torch.zeros(1, 32, 32),
            "particles": [[{"x": 52.0, "y": 40.0}]],
        }]

        _, metrics = trainer.validate(batches, compute_metrics=True)

        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)


def test_extract_predictions_applies_max_per_image_per_batch():
    from supicker.engine.trainer import Trainer
    from supicker.models import Detector
    from supicker.config import ModelConfig, BackboneConfig, ConvNeXtVariant, TrainingConfig

    model_config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    )
    training_config = TrainingConfig(batch_size=2)
    model = Detector(model_config)

    outputs = {
        "heatmap": torch.zeros(2, 1, 8, 8),
        "size": torch.zeros(2, 2, 8, 8),
        "offset": torch.zeros(2, 2, 8, 8),
    }
    outputs["heatmap"][0, 0, 1, 1] = 0.95
    outputs["heatmap"][0, 0, 2, 2] = 0.90
    outputs["heatmap"][1, 0, 3, 3] = 0.85
    outputs["heatmap"][1, 0, 4, 4] = 0.80

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            config=training_config,
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
            device="cpu",
        )

        predictions = trainer._extract_predictions(
            outputs,
            score_threshold=0.1,
            max_per_image=1,
            nms_radius=0.0,
        )
        per_batch = trainer._split_predictions_by_batch(predictions, batch_size=2)

        assert len(per_batch[0]) == 1
        assert len(per_batch[1]) == 1
        assert per_batch[0][0]["score"] == pytest.approx(0.95)
        assert per_batch[1][0]["score"] == pytest.approx(0.85)
