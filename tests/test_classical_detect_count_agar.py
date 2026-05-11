from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classical.detect_count import (
    ClassicalCountConfig,
    CountPrediction,
    count_colonies,
    evaluate_count_predictions,
)


def test_count_colonies_detects_separated_dark_colonies() -> None:
    image = np.full((220, 260, 3), 230, dtype=np.uint8)
    centers = [(60, 70), (130, 80), (190, 150)]
    for center in centers:
        cv2.circle(image, center, 14, (45, 45, 45), -1)

    count, mask, contours = count_colonies(
        image,
        ClassicalCountConfig(
            background_kernel=41,
            min_area=120,
            max_area=1200,
            min_circularity=0.45,
        ),
    )

    assert count == 3
    assert mask.dtype == np.uint8
    assert len(contours) == 3


def test_count_colonies_returns_zero_for_empty_plate() -> None:
    image = np.full((180, 220, 3), 230, dtype=np.uint8)

    count, _, contours = count_colonies(
        image,
        ClassicalCountConfig(
            background_kernel=41,
            min_area=120,
            max_area=1200,
            min_circularity=0.45,
        ),
    )

    assert count == 0
    assert contours == []


def test_evaluate_count_predictions_saves_core_metrics() -> None:
    predictions = [
        CountPrediction("a.jpg", 10, 12, "countable", "A", 2, 2, 4),
        CountPrediction("b.jpg", 0, 0, "empty", "", 0, 0, 0),
        CountPrediction("c.jpg", 5, 4, "countable", "B", -1, 1, 1),
    ]

    evaluation = evaluate_count_predictions(predictions, ClassicalCountConfig())

    assert evaluation.metrics["image_count"] == 3
    assert evaluation.metrics["true_total"] == 15
    assert evaluation.metrics["predicted_total"] == 16
    assert evaluation.metrics["mae"] == 1.0
    assert evaluation.metrics["zero_exact_accuracy"] == 1.0
    assert "config" in evaluation.metrics


def test_prediction_rows_are_csv_serializable(tmp_path: Path) -> None:
    predictions = [
        CountPrediction("a.jpg", 1, 2, "countable", "A", 1, 1, 1),
    ]

    path = tmp_path / "predictions.csv"
    pd.DataFrame([prediction.__dict__ for prediction in predictions]).to_csv(path, index=False)
    loaded = pd.read_csv(path)

    assert loaded.loc[0, "predicted_count"] == 2
