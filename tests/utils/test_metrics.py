"""Tests for detection metrics module."""

import pytest
import numpy as np

from supicker.utils.metrics import (
    DetectionMetrics,
    match_particles_by_distance,
    compute_detection_metrics,
    compute_average_precision,
    MetricAggregator,
)


class TestMatchParticlesByDistance:
    """Tests for Hungarian matching function."""

    def test_perfect_match(self):
        """Test perfect 1:1 matching."""
        predictions = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        matched, fp, fn = match_particles_by_distance(predictions, ground_truth, threshold=5.0)

        assert len(matched) == 2
        assert len(fp) == 0
        assert len(fn) == 0

    def test_close_match(self):
        """Test matching with small offsets."""
        predictions = [{"x": 11, "y": 10}, {"x": 19, "y": 21}]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        matched, fp, fn = match_particles_by_distance(predictions, ground_truth, threshold=5.0)

        assert len(matched) == 2
        assert len(fp) == 0
        assert len(fn) == 0

    def test_threshold_filtering(self):
        """Test that matches beyond threshold are rejected."""
        predictions = [{"x": 10, "y": 10}, {"x": 50, "y": 50}]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        matched, fp, fn = match_particles_by_distance(predictions, ground_truth, threshold=5.0)

        assert len(matched) == 1
        assert len(fp) == 1
        assert len(fn) == 1

    def test_empty_predictions(self):
        """Test with no predictions."""
        predictions = []
        ground_truth = [{"x": 10, "y": 10}]

        matched, fp, fn = match_particles_by_distance(predictions, ground_truth, threshold=5.0)

        assert len(matched) == 0
        assert len(fp) == 0
        assert len(fn) == 1

    def test_empty_ground_truth(self):
        """Test with no ground truth."""
        predictions = [{"x": 10, "y": 10}]
        ground_truth = []

        matched, fp, fn = match_particles_by_distance(predictions, ground_truth, threshold=5.0)

        assert len(matched) == 0
        assert len(fp) == 1
        assert len(fn) == 0

    def test_both_empty(self):
        """Test with both empty."""
        matched, fp, fn = match_particles_by_distance([], [], threshold=5.0)

        assert len(matched) == 0
        assert len(fp) == 0
        assert len(fn) == 0


class TestComputeDetectionMetrics:
    """Tests for metric computation."""

    def test_perfect_detection(self):
        """Test with perfect predictions."""
        predictions = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        metrics = compute_detection_metrics(predictions, ground_truth, distance_threshold=5.0)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.true_positives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.avg_distance == 0.0

    def test_partial_detection(self):
        """Test with partial overlap."""
        predictions = [{"x": 10, "y": 10}, {"x": 100, "y": 100}]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        metrics = compute_detection_metrics(predictions, ground_truth, distance_threshold=5.0)

        assert metrics.precision == 0.5
        assert metrics.recall == 0.5
        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1

    def test_all_false_positives(self):
        """Test when all predictions are wrong."""
        predictions = [{"x": 100, "y": 100}, {"x": 200, "y": 200}]
        ground_truth = [{"x": 10, "y": 10}]

        metrics = compute_detection_metrics(predictions, ground_truth, distance_threshold=5.0)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.false_positives == 2
        assert metrics.false_negatives == 1

    def test_average_distance(self):
        """Test average distance calculation."""
        predictions = [{"x": 13, "y": 10}, {"x": 20, "y": 24}]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        metrics = compute_detection_metrics(predictions, ground_truth, distance_threshold=10.0)

        expected_dist = (3.0 + 4.0) / 2
        assert abs(metrics.avg_distance - expected_dist) < 1e-6


class TestComputeAveragePrecision:
    """Tests for AP computation."""

    def test_perfect_ap(self):
        """Test AP with perfect high-confidence predictions."""
        predictions = [
            {"x": 10, "y": 10, "score": 0.9},
            {"x": 20, "y": 20, "score": 0.8},
        ]
        ground_truth = [{"x": 10, "y": 10}, {"x": 20, "y": 20}]

        ap = compute_average_precision(predictions, ground_truth, distance_threshold=5.0)

        assert ap == pytest.approx(1.0, abs=0.02)

    def test_empty_predictions_ap(self):
        """Test AP with no predictions."""
        ap = compute_average_precision([], [{"x": 10, "y": 10}], distance_threshold=5.0)
        assert ap == 0.0

    def test_empty_ground_truth_ap(self):
        """Test AP with no ground truth."""
        ap = compute_average_precision([{"x": 10, "y": 10, "score": 0.9}], [], distance_threshold=5.0)
        assert ap == 0.0

    def test_both_empty_ap(self):
        """Test AP with both empty."""
        ap = compute_average_precision([], [], distance_threshold=5.0)
        assert ap == 1.0


class TestMetricAggregator:
    """Tests for MetricAggregator class."""

    def test_single_image(self):
        """Test aggregation with single image."""
        aggregator = MetricAggregator(distance_threshold=5.0)

        predictions = [{"x": 10, "y": 10}]
        ground_truth = [{"x": 10, "y": 10}]

        aggregator.add_image(predictions, ground_truth)
        metrics = aggregator.compute_aggregate(compute_ap=False)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.true_positives == 1

    def test_multiple_images(self):
        """Test aggregation across multiple images."""
        aggregator = MetricAggregator(distance_threshold=5.0)

        # Image 1: 1 TP
        aggregator.add_image([{"x": 10, "y": 10}], [{"x": 10, "y": 10}])
        # Image 2: 1 FP
        aggregator.add_image([{"x": 100, "y": 100}], [{"x": 10, "y": 10}])

        metrics = aggregator.compute_aggregate(compute_ap=False)

        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5

    def test_reset(self):
        """Test reset functionality."""
        aggregator = MetricAggregator(distance_threshold=5.0)

        aggregator.add_image([{"x": 10, "y": 10}], [{"x": 10, "y": 10}])
        aggregator.reset()

        metrics = aggregator.compute_aggregate(compute_ap=False)

        assert metrics.true_positives == 0
        assert metrics.precision == 0.0


class TestDetectionMetrics:
    """Tests for DetectionMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test dataclass instantiation."""
        metrics = DetectionMetrics(
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            true_positives=10,
            false_positives=2,
            false_negatives=3,
            avg_distance=2.5,
            ap=0.85,
        )

        assert metrics.precision == 0.8
        assert metrics.recall == 0.7
        assert metrics.f1_score == 0.75
        assert metrics.true_positives == 10
        assert metrics.false_positives == 2
        assert metrics.false_negatives == 3
        assert metrics.avg_distance == 2.5
        assert metrics.ap == 0.85

    def test_optional_ap(self):
        """Test that AP is optional."""
        metrics = DetectionMetrics(
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            true_positives=10,
            false_positives=2,
            false_negatives=3,
            avg_distance=2.5,
        )

        assert metrics.ap is None
