import pytest
from pathlib import Path


def test_parse_star_file():
    from supicker.data.star_parser import parse_star_file

    star_path = Path(__file__).parent / "fixtures" / "test.star"
    result = parse_star_file(star_path)

    assert "image_001.tiff" in result
    assert "image_002.tiff" in result
    assert len(result["image_001.tiff"]) == 2
    assert len(result["image_002.tiff"]) == 1

    particle = result["image_001.tiff"][0]
    assert particle["x"] == 100.5
    assert particle["y"] == 200.3
    assert particle["class_id"] == 0  # 0-indexed


def test_parse_star_cryosparc_format():
    from supicker.data.star_parser import parse_star_file

    # cryoSPARC uses different column names
    star_content = """data_

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
mic_001.tiff 50.0 60.0
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".star", delete=False) as f:
        f.write(star_content)
        f.flush()
        result = parse_star_file(f.name)

    assert "mic_001.tiff" in result
    assert result["mic_001.tiff"][0]["class_id"] == 0  # Default class
