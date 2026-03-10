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
