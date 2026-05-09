from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.create_figshare_splits import validate_ratios
from scripts.evaluate_yolo_count_predictions import compute_metrics, count_prediction_file
from scripts.export_figshare_yolo_dataset import sorted_species, yolo_rows_for_image
from scripts.sweep_yolo_count_thresholds import thresholds


def test_count_prediction_file_applies_confidence_threshold(tmp_path: Path) -> None:
    label_path = tmp_path / "pred.txt"
    label_path.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.1 0.1 0.95",
                "0 0.4 0.4 0.1 0.1 0.31",
                "0 0.3 0.3 0.1 0.1 0.05",
                "0 0.2 0.2 0.1 0.1",
            ]
        ),
        encoding="utf-8",
    )

    assert count_prediction_file(label_path, conf_threshold=0.32) == 2
    assert count_prediction_file(label_path, conf_threshold=0.01) == 4
    assert count_prediction_file(tmp_path / "missing.txt", conf_threshold=0.01) == 0


def test_compute_metrics_matches_expected_count_errors() -> None:
    results = pd.DataFrame(
        {
            "true_count": [10, 20, 5],
            "pred_count": [12, 14, 5],
        }
    )

    metrics = compute_metrics(results)

    assert metrics["images"] == 3.0
    assert metrics["true_colonies"] == 35.0
    assert metrics["pred_colonies"] == 31.0
    assert metrics["mean_error"] == pytest.approx(-4.0 / 3.0)
    assert metrics["mae"] == pytest.approx(8.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((40.0 / 3.0) ** 0.5)
    assert metrics["mape_percent"] == pytest.approx(((2.0 / 10.0) + (6.0 / 20.0)) / 3.0 * 100.0)


def test_yolo_rows_for_image_normalizes_and_clips_boxes() -> None:
    annotations = pd.DataFrame(
        [
            {
                "label_name": "sp02",
                "bbox_x": 90,
                "bbox_y": 40,
                "bbox_width": 30,
                "bbox_height": 20,
                "image_width": 100,
                "image_height": 80,
            }
        ]
    )

    rows = yolo_rows_for_image(annotations, {"sp02": 3}, single_class=False)
    class_id, x_center, y_center, width, height = rows[0].split()

    assert class_id == "3"
    assert float(x_center) == 1.0
    assert float(y_center) == pytest.approx(0.625)
    assert float(width) == pytest.approx(0.3)
    assert float(height) == pytest.approx(0.25)

    single_class_rows = yolo_rows_for_image(annotations, {}, single_class=True)
    assert single_class_rows[0].split()[0] == "0"


def test_sorted_species_uses_numeric_order() -> None:
    labels = pd.Series(["sp10", "sp02", "sp1", "sp03"])

    assert sorted_species(labels) == ["sp1", "sp02", "sp03", "sp10"]


def test_validate_ratios_rejects_invalid_sum() -> None:
    validate_ratios(0.7, 0.15, 0.15)

    with pytest.raises(ValueError):
        validate_ratios(0.7, 0.2, 0.2)


def test_threshold_generation_includes_upper_bound() -> None:
    assert thresholds(0.1, 0.3, 0.1) == [0.1, 0.2, 0.3]
