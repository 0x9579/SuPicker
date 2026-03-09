import pytest
import torch
from pathlib import Path


def test_parse_thresholds_parses_csv_values():
    from scripts.scan_thresholds import parse_thresholds

    assert parse_thresholds("0.03, 0.1,0.2") == [0.03, 0.1, 0.2]


def test_format_scan_results_includes_metric_columns():
    from scripts.scan_thresholds import format_scan_results

    rows = [
        {
            "threshold": 0.1,
            "precision": 0.25,
            "recall": 0.5,
            "f1_score": 0.3333,
            "avg_predictions": 12.0,
        }
    ]

    output = format_scan_results(rows)

    assert "Threshold" in output
    assert "Precision" in output
    assert "Recall" in output
    assert "F1" in output
    assert "Avg Preds" in output
    assert "0.100" in output


def test_evaluate_thresholds_returns_metrics_for_each_threshold():
    from scripts.scan_thresholds import evaluate_thresholds

    outputs = {
        "heatmap": torch.zeros(1, 1, 8, 8),
        "size": torch.zeros(1, 2, 8, 8),
        "offset": torch.zeros(1, 2, 8, 8),
    }
    outputs["heatmap"][0, 0, 2, 2] = 0.3
    outputs["heatmap"][0, 0, 4, 4] = 0.12

    batches = [
        {
            "outputs": outputs,
            "particles": [[{"x": 8.0, "y": 8.0}]],
        }
    ]

    rows = evaluate_thresholds(
        batches=batches,
        thresholds=[0.1, 0.2],
        distance_threshold=20.0,
        nms_radius=20.0,
    )

    assert [row["threshold"] for row in rows] == [0.1, 0.2]
    assert rows[0]["recall"] >= rows[1]["recall"]
    assert rows[0]["avg_predictions"] >= rows[1]["avg_predictions"]


def test_build_parser_default_thresholds():
    from scripts.scan_thresholds import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--val-images", "images",
        "--val-star", "particles.star",
        "--checkpoint", "model.pt",
    ])

    assert args.thresholds == "0.03,0.05,0.08,0.1,0.12,0.15,0.2"


def test_build_parser_accepts_checkpoint_dir():
    from scripts.scan_thresholds import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--val-images", "images",
        "--val-star", "particles.star",
        "--checkpoint-dir", "checkpoints",
    ])

    assert args.checkpoint_dir == "checkpoints"
    assert args.checkpoint is None


def test_find_checkpoints_in_directory(tmp_path: Path):
    from scripts.scan_thresholds import find_checkpoints

    (tmp_path / "epoch_10.pt").write_text("x")
    (tmp_path / "epoch_40.pt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    paths = find_checkpoints(checkpoint_dir=tmp_path, checkpoint_pattern="*.pt")

    assert [path.name for path in paths] == ["epoch_10.pt", "epoch_40.pt"]


def test_summarize_best_rows_picks_highest_f1():
    from scripts.scan_thresholds import summarize_best_rows

    rows = [
        {"checkpoint": "epoch_10.pt", "threshold": 0.05, "f1_score": 0.2, "precision": 0.1, "recall": 0.8, "avg_predictions": 20.0},
        {"checkpoint": "epoch_10.pt", "threshold": 0.10, "f1_score": 0.3, "precision": 0.2, "recall": 0.6, "avg_predictions": 10.0},
        {"checkpoint": "epoch_20.pt", "threshold": 0.05, "f1_score": 0.1, "precision": 0.05, "recall": 0.5, "avg_predictions": 30.0},
    ]

    summary = summarize_best_rows(rows)

    assert len(summary) == 2
    assert summary[0]["checkpoint"] == "epoch_10.pt"
    assert summary[0]["threshold"] == 0.10
    assert summary[1]["checkpoint"] == "epoch_20.pt"


def test_format_best_summary_includes_best_threshold_and_f1():
    from scripts.scan_thresholds import format_best_summary

    summary = [
        {"checkpoint": "epoch_40.pt", "threshold": 0.1, "f1_score": 0.25, "precision": 0.2, "recall": 0.33, "avg_predictions": 12.0}
    ]

    output = format_best_summary(summary)

    assert "Checkpoint" in output
    assert "Best Th" in output
    assert "Best F1" in output
    assert "epoch_40.pt" in output


def test_parse_thresholds_rejects_empty_input():
    from scripts.scan_thresholds import parse_thresholds

    with pytest.raises(ValueError, match="No thresholds provided"):
        parse_thresholds(" , ")


def test_find_checkpoints_raises_when_no_match(tmp_path: Path):
    from scripts.scan_thresholds import find_checkpoints

    with pytest.raises(ValueError, match="No checkpoints found"):
        find_checkpoints(checkpoint_dir=tmp_path, checkpoint_pattern="*.pt")


def test_build_eval_transforms_disables_random_crop():
    from scripts.scan_thresholds import build_eval_transforms

    transforms = build_eval_transforms()

    assert transforms.transforms[0].__class__.__name__ == "Normalize"
