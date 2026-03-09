import pytest
import tempfile
from pathlib import Path


def test_logger_log_scalar():
    from supicker.utils.logger import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_tensorboard=False)

        logger.log_scalar("loss", 0.5, step=1)
        logger.log_scalar("accuracy", 0.8, step=1)

        # Should not raise
        logger.close()


def test_logger_log_scalars():
    from supicker.utils.logger import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_tensorboard=False)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        logger.log_scalars(metrics, step=1)

        logger.close()


def test_logger_with_tensorboard():
    from supicker.utils.logger import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_tensorboard=True)

        logger.log_scalar("train/loss", 0.5, step=1)

        logger.close()

        # TensorBoard files should exist
        events_files = list(Path(tmpdir).glob("events.out.tfevents.*"))
        assert len(events_files) > 0


def test_logger_console_output(capsys):
    from supicker.utils.logger import Logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_tensorboard=False, print_freq=1)

        logger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.3)

        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "0.5" in captured.out or "0.50" in captured.out


def test_format_validation_threshold_logging():
    from supicker.config import TrainingConfig
    from scripts.train import format_validation_thresholds

    config = TrainingConfig(
        val_interval=5,
        val_score_threshold=0.1,
        val_distance_threshold=20.0,
        val_nms_radius=20.0,
    )

    output = format_validation_thresholds(config)

    assert "Val interval: 5" in output
    assert "Val score threshold: 0.1" in output
    assert "Val distance threshold: 20.0" in output
    assert "Val NMS radius: 20.0" in output
