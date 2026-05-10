from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.yolo.common import read_yolo_label_file
from src.yolo.detect_count import evaluate_yolo_detect_count
from src.yolo.species_classification import classification_outputs, evaluate_yolo_species_classification


def test_read_yolo_label_file_filters_by_confidence(tmp_path: Path) -> None:
    label_path = tmp_path / "image.txt"
    label_path.write_text(
        "0 0.5 0.5 0.1 0.1 0.9\n1 0.4 0.4 0.1 0.1 0.2\n",
        encoding="utf-8",
    )

    labels = read_yolo_label_file(label_path, conf_threshold=0.5)

    assert len(labels) == 1
    assert labels.iloc[0]["class_id"] == 0


def test_evaluate_yolo_detect_count_outputs_metrics(tmp_path: Path) -> None:
    split_csv = tmp_path / "split.csv"
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pd.DataFrame(
        [
            {"image_name": "a.jpg", "colony_count": 2},
            {"image_name": "b.jpg", "colony_count": 1},
        ]
    ).to_csv(split_csv, index=False)
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n0 0.4 0.4 0.1 0.1 0.8\n", encoding="utf-8")
    (label_dir / "b.txt").write_text("", encoding="utf-8")

    evaluation = evaluate_yolo_detect_count(split_csv, label_dir, conf_threshold=0.5)

    assert evaluation.summary["images"] == 2
    assert evaluation.summary["true_colonies"] == 3
    assert evaluation.summary["pred_colonies"] == 2
    assert evaluation.summary["mae"] == pytest.approx(0.5)


def test_evaluate_yolo_species_classification_uses_majority_class(tmp_path: Path) -> None:
    split_csv = tmp_path / "split.csv"
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    class_names = tmp_path / "classes.txt"
    class_names.write_text("sp01\nsp02\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"image_name": "a.jpg", "label_name": "sp02"},
            {"image_name": "b.jpg", "label_name": "sp01"},
        ]
    ).to_csv(split_csv, index=False)
    (label_dir / "a.txt").write_text(
        "1 0.5 0.5 0.1 0.1 0.9\n1 0.4 0.4 0.1 0.1 0.8\n0 0.3 0.3 0.1 0.1 0.7\n",
        encoding="utf-8",
    )
    (label_dir / "b.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")

    evaluation = evaluate_yolo_species_classification(split_csv, label_dir, class_names, conf_threshold=0.5)

    assert evaluation.metrics["accuracy"] == 1.0
    assert evaluation.predictions.loc[0, "pred_label"] == "sp02"
    assert len(evaluation.class_distribution) == 3


def test_classification_outputs_handles_errors() -> None:
    predictions = pd.DataFrame(
        [
            {"true_label": "sp01", "pred_label": "sp01", "correct": True},
            {"true_label": "sp02", "pred_label": "sp01", "correct": False},
        ]
    )

    report, confusion, metrics = classification_outputs(predictions)

    assert metrics["accuracy"] == 0.5
    assert set(report["label"]) == {"sp01", "sp02"}
    assert confusion.loc["sp02", "sp01"] == 1
