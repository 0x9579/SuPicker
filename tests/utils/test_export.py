import pytest
import tempfile
import json
from pathlib import Path


def test_export_to_json():
    from supicker.utils.export import export_to_json

    particles = [
        {"x": 100.0, "y": 200.0, "score": 0.95, "class_id": 0},
        {"x": 150.0, "y": 250.0, "score": 0.85, "class_id": 1},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_to_json(particles, f.name, micrograph_name="test.tiff")

        # Read back and verify
        with open(f.name, "r") as rf:
            data = json.load(rf)

        assert data["micrograph"] == "test.tiff"
        assert len(data["particles"]) == 2
        assert data["particles"][0]["x"] == 100.0


def test_export_to_csv():
    from supicker.utils.export import export_to_csv

    particles = [
        {"x": 100.0, "y": 200.0, "score": 0.95, "class_id": 0, "width": 64, "height": 64},
        {"x": 150.0, "y": 250.0, "score": 0.85, "class_id": 1, "width": 64, "height": 64},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        export_to_csv(particles, f.name)

        # Read back and verify
        with open(f.name, "r") as rf:
            lines = rf.readlines()

        # Should have header + 2 data lines
        assert len(lines) == 3
        assert "x,y" in lines[0]


def test_export_to_star():
    from supicker.utils.export import export_to_star

    particles = [
        {"x": 100.0, "y": 200.0, "score": 0.95, "class_id": 0, "width": 64, "height": 64},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".star", delete=False) as f:
        export_to_star(particles, f.name, micrograph_name="test.tiff")

        # Read back and verify
        with open(f.name, "r") as rf:
            content = rf.read()

        assert "_rlnCoordinateX" in content
        assert "100.00" in content
