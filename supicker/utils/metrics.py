"""Evaluation metrics for particle detection."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    avg_distance: float
    ap: Optional[float] = None


def match_particles_by_distance(
    predictions: list[dict],
    ground_truth: list[dict],
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predictions to ground truth using Hungarian algorithm.

    Args:
        predictions: List of predicted particles with 'x', 'y' keys
        ground_truth: List of ground truth particles with 'x', 'y' keys
        threshold: Maximum distance for valid match

    Returns:
        Tuple of:
            - matched_pairs: List of (pred_idx, gt_idx) pairs
            - unmatched_preds: List of unmatched prediction indices (FP)
            - unmatched_gts: List of unmatched ground truth indices (FN)
    """
    if len(predictions) == 0 or len(ground_truth) == 0:
        unmatched_preds = list(range(len(predictions)))
        unmatched_gts = list(range(len(ground_truth)))
        return [], unmatched_preds, unmatched_gts

    # Build cost matrix (distances)
    n_preds = len(predictions)
    n_gts = len(ground_truth)
    cost_matrix = np.zeros((n_preds, n_gts))

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            dx = pred["x"] - gt["x"]
            dy = pred["y"] - gt["y"]
            cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

    # Run Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by threshold
    matched_pairs = []
    matched_pred_set = set()
    matched_gt_set = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] <= threshold:
            matched_pairs.append((pred_idx, gt_idx))
            matched_pred_set.add(pred_idx)
            matched_gt_set.add(gt_idx)

    # Find unmatched
    unmatched_preds = [i for i in range(n_preds) if i not in matched_pred_set]
    unmatched_gts = [i for i in range(n_gts) if i not in matched_gt_set]

    return matched_pairs, unmatched_preds, unmatched_gts


def compute_detection_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    distance_threshold: float = 10.0,
) -> DetectionMetrics:
    """Compute detection metrics for a single image.

    Args:
        predictions: List of predicted particles with 'x', 'y' keys
        ground_truth: List of ground truth particles with 'x', 'y' keys
        distance_threshold: Maximum distance for valid match

    Returns:
        DetectionMetrics with precision, recall, F1, etc.
    """
    matched_pairs, unmatched_preds, unmatched_gts = match_particles_by_distance(
        predictions, ground_truth, distance_threshold
    )

    tp = len(matched_pairs)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute average distance for matched pairs
    if matched_pairs:
        total_distance = 0.0
        for pred_idx, gt_idx in matched_pairs:
            pred = predictions[pred_idx]
            gt = ground_truth[gt_idx]
            dx = pred["x"] - gt["x"]
            dy = pred["y"] - gt["y"]
            total_distance += np.sqrt(dx * dx + dy * dy)
        avg_distance = total_distance / len(matched_pairs)
    else:
        avg_distance = 0.0

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        avg_distance=avg_distance,
    )


def compute_average_precision(
    predictions: list[dict],
    ground_truth: list[dict],
    distance_threshold: float = 10.0,
    num_score_thresholds: int = 101,
) -> float:
    """Compute Average Precision (AP) across score thresholds.

    Uses 11-point interpolation method common in object detection.

    Args:
        predictions: List of predicted particles with 'x', 'y', 'score' keys
        ground_truth: List of ground truth particles with 'x', 'y' keys
        distance_threshold: Maximum distance for valid match
        num_score_thresholds: Number of score thresholds to evaluate

    Returns:
        Average Precision value in range [0, 1]
    """
    if len(ground_truth) == 0:
        return 1.0 if len(predictions) == 0 else 0.0

    if len(predictions) == 0:
        return 0.0

    # Sort predictions by score descending
    sorted_preds = sorted(predictions, key=lambda p: p.get("score", 0), reverse=True)

    # Compute precision-recall at each score threshold
    score_thresholds = np.linspace(0, 1, num_score_thresholds)
    precisions = []
    recalls = []

    for score_thresh in score_thresholds:
        filtered_preds = [p for p in sorted_preds if p.get("score", 0) >= score_thresh]
        metrics = compute_detection_metrics(filtered_preds, ground_truth, distance_threshold)
        precisions.append(metrics.precision)
        recalls.append(metrics.recall)

    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall (ascending) for proper PR curve
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Remove duplicate recall values, keeping highest precision
    unique_recalls = []
    unique_precisions = []
    prev_recall = -1

    for r, p in zip(recalls, precisions):
        if r != prev_recall:
            unique_recalls.append(r)
            unique_precisions.append(p)
            prev_recall = r
        else:
            # Update precision if higher
            unique_precisions[-1] = max(unique_precisions[-1], p)

    recalls = np.array(unique_recalls)
    precisions = np.array(unique_precisions)

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP using trapezoidal integration
    # Prepend (0, first_precision) and append (1, 0) for full curve
    recalls_ext = np.concatenate([[0], recalls, [1]])
    precisions_ext = np.concatenate([[precisions[0] if len(precisions) > 0 else 0], precisions, [0]])

    # AP = sum of (recall_diff * precision)
    recall_diffs = np.diff(recalls_ext)
    ap = np.sum(recall_diffs * precisions_ext[1:])

    return float(np.clip(ap, 0.0, 1.0))


class MetricAggregator:
    """Aggregates detection metrics across multiple images."""

    def __init__(self, distance_threshold: float = 20.0):
        """Initialize aggregator.

        Args:
            distance_threshold: Maximum distance for valid match
        """
        self.distance_threshold = distance_threshold
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_distance = 0.0
        self.matched_count = 0
        self.all_predictions = []
        self.all_ground_truths = []

    def add_image(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
    ) -> None:
        """Add metrics from a single image.

        Args:
            predictions: List of predicted particles
            ground_truth: List of ground truth particles
        """
        matched_pairs, unmatched_preds, unmatched_gts = match_particles_by_distance(
            predictions, ground_truth, self.distance_threshold
        )

        self.total_tp += len(matched_pairs)
        self.total_fp += len(unmatched_preds)
        self.total_fn += len(unmatched_gts)

        # Accumulate distances for matched pairs
        for pred_idx, gt_idx in matched_pairs:
            pred = predictions[pred_idx]
            gt = ground_truth[gt_idx]
            dx = pred["x"] - gt["x"]
            dy = pred["y"] - gt["y"]
            self.total_distance += np.sqrt(dx * dx + dy * dy)
            self.matched_count += 1

        # Store for AP computation
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truth)

    def compute_aggregate(self, compute_ap: bool = True) -> DetectionMetrics:
        """Compute aggregated metrics across all images.

        Args:
            compute_ap: Whether to compute Average Precision

        Returns:
            Aggregated DetectionMetrics
        """
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_distance = self.total_distance / self.matched_count if self.matched_count > 0 else 0.0

        ap = None
        if compute_ap and self.all_predictions and self.all_ground_truths:
            ap = compute_average_precision(
                self.all_predictions,
                self.all_ground_truths,
                self.distance_threshold,
            )

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            avg_distance=avg_distance,
            ap=ap,
        )
