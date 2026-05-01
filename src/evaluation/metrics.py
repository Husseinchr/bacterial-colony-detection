"""Evaluation metrics for segmentation and colony counting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CountMetrics:
    """Counting error metrics for one image or an aggregate dataset."""

    mae: float
    rmse: float
    mean_error: float
    mean_absolute_percentage_error: float


def binary_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute intersection over union for binary masks."""

    pred = pred_mask > 0
    true = true_mask > 0
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return 1.0 if union == 0 else float(intersection / union)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute Dice coefficient for binary masks."""

    pred = pred_mask > 0
    true = true_mask > 0
    denominator = pred.sum() + true.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * np.logical_and(pred, true).sum()) / denominator)


def pixel_precision_recall_f1(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    """Compute pixel-level precision, recall, and F1 for binary segmentation."""

    pred = pred_mask > 0
    true = true_mask > 0
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def count_metrics(pred_counts: list[int], true_counts: list[int]) -> CountMetrics:
    """Compute aggregate count metrics from predicted and ground-truth counts."""

    if len(pred_counts) != len(true_counts):
        raise ValueError("pred_counts and true_counts must have the same length")
    if not pred_counts:
        raise ValueError("At least one count pair is required")

    pred = np.asarray(pred_counts, dtype=float)
    true = np.asarray(true_counts, dtype=float)
    errors = pred - true
    abs_errors = np.abs(errors)
    percentage = np.array([_safe_div(abs_err, max(t, 1.0)) for abs_err, t in zip(abs_errors, true)])
    return CountMetrics(
        mae=float(abs_errors.mean()),
        rmse=float(math.sqrt(np.mean(errors * errors))),
        mean_error=float(errors.mean()),
        mean_absolute_percentage_error=float(percentage.mean() * 100.0),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)

