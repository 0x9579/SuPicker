import numpy as np
import pytest
import tifffile
import torch


def test_load_image_applies_zero_mean_unit_variance_normalization(tmp_path):
    from scripts.predict import load_image
    from supicker.data.transforms import Normalize

    image = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    image_path = tmp_path / "sample.tif"
    tifffile.imwrite(image_path, image)

    tensor = load_image(image_path)
    expected, _ = Normalize(p=1.0).apply(torch.from_numpy((image / 6.0)[np.newaxis, ...]), [])

    assert tensor.shape == (1, 2, 2)
    assert torch.allclose(tensor, expected)


def test_load_image_skips_zscore_for_constant_input(tmp_path):
    from scripts.predict import load_image

    image = np.full((2, 2), 5.0, dtype=np.float32)
    image_path = tmp_path / "constant.tif"
    tifffile.imwrite(image_path, image)

    tensor = load_image(image_path)

    assert tensor.shape == (1, 2, 2)
    assert torch.allclose(tensor, torch.zeros_like(tensor))


def test_predict_parser_accepts_merge_output(monkeypatch):
    from scripts import predict

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict.py",
            "--input", "images",
            "--checkpoint", "model.pt",
            "--merge-output", "merged.star",
        ],
    )

    args = predict.parse_args()

    assert args.merge_output == "merged.star"


def test_export_merged_particles_writes_single_star(tmp_path):
    from scripts.predict import export_merged_particles

    merged_path = tmp_path / "merged.star"
    particles_by_image = {
        "a.mrc": [{"x": 1.0, "y": 2.0, "score": 0.9, "class_id": 0}],
        "b.mrc": [{"x": 3.0, "y": 4.0, "score": 0.8, "class_id": 0}],
    }

    export_merged_particles(particles_by_image, merged_path, format="star")

    content = merged_path.read_text()
    assert "a.mrc 1.00 2.00" in content
    assert "b.mrc 3.00 4.00" in content


def test_export_merged_particles_creates_parent_directory(tmp_path):
    from scripts.predict import export_merged_particles

    merged_path = tmp_path / "nested" / "merged.star"
    particles_by_image = {"a.mrc": [{"x": 1.0, "y": 2.0, "score": 0.9, "class_id": 0}]}

    export_merged_particles(particles_by_image, merged_path, format="star")

    assert merged_path.exists()
