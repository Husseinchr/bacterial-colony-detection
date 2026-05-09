from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import binary_iou, count_metrics, dice_score, pixel_precision_recall_f1


def test_binary_segmentation_metrics_handle_overlap() -> None:
    pred = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    true = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)

    assert binary_iou(pred, true) == pytest.approx(0.5)
    assert dice_score(pred, true) == pytest.approx(2.0 / 3.0)

    scores = pixel_precision_recall_f1(pred, true)
    assert scores["precision"] == pytest.approx(2.0 / 3.0)
    assert scores["recall"] == pytest.approx(2.0 / 3.0)
    assert scores["f1"] == pytest.approx(2.0 / 3.0)


def test_count_metrics_compute_expected_values() -> None:
    metrics = count_metrics(pred_counts=[10, 12, 8], true_counts=[9, 15, 8])

    assert metrics.mean_error == pytest.approx(-2.0 / 3.0)
    assert metrics.mae == pytest.approx(4.0 / 3.0)
    assert metrics.rmse == pytest.approx((10.0 / 3.0) ** 0.5)
    assert metrics.mean_absolute_percentage_error == pytest.approx(
        ((1.0 / 9.0) + (3.0 / 15.0) + 0.0) / 3.0 * 100.0
    )


def test_count_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        count_metrics(pred_counts=[1], true_counts=[1, 2])
