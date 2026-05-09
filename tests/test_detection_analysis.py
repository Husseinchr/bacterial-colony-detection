from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import build_density_grid, read_yolo_detections, summarize_detections


def test_read_yolo_detections_filters_and_converts_boxes(tmp_path: Path) -> None:
    label_path = tmp_path / "image.txt"
    label_path.write_text(
        "\n".join(
            [
                "0 0.500000 0.500000 0.200000 0.100000 0.90",
                "0 0.100000 0.100000 0.100000 0.100000 0.20",
                "0 1.050000 0.500000 0.200000 0.200000 0.95",
            ]
        ),
        encoding="utf-8",
    )

    detections = read_yolo_detections(label_path, image_width=100, image_height=80, conf_threshold=0.5)

    assert len(detections) == 2
    assert detections.iloc[0]["x1"] == 40
    assert detections.iloc[0]["y1"] == 36
    assert detections.iloc[0]["x2"] == 60
    assert detections.iloc[0]["y2"] == 44
    assert detections.iloc[0]["bbox_area"] == 160.0
    assert detections.iloc[1]["x2"] == 100


def test_read_yolo_detections_returns_empty_frame_for_missing_file(tmp_path: Path) -> None:
    detections = read_yolo_detections(tmp_path / "missing.txt", image_width=100, image_height=80)

    assert detections.empty
    assert "equivalent_diameter" in detections.columns


def test_build_density_grid_counts_centroids() -> None:
    detections = read_yolo_detections_from_text(
        "0 0.250000 0.250000 0.100000 0.100000 0.90\n"
        "0 0.750000 0.750000 0.100000 0.100000 0.90\n",
        image_width=100,
        image_height=100,
    )

    grid = build_density_grid(detections, image_width=100, image_height=100, rows=2, cols=2)

    counts = grid.pivot(index="grid_row", columns="grid_col", values="count").to_numpy()
    assert counts.tolist() == [[1, 0], [0, 1]]
    assert grid["count"].sum() == 2


def test_build_density_grid_rejects_invalid_shape() -> None:
    detections = read_yolo_detections_from_text("", image_width=100, image_height=100)

    with pytest.raises(ValueError):
        build_density_grid(detections, image_width=100, image_height=100, rows=0, cols=2)


def test_summarize_detections_reports_count_and_density() -> None:
    detections = read_yolo_detections_from_text(
        "0 0.500000 0.500000 0.200000 0.100000 0.90\n",
        image_width=100,
        image_height=80,
    )

    summary = summarize_detections(detections, image_width=100, image_height=80)

    assert summary["colony_count"] == 1
    assert summary["image_area_pixels"] == 8000
    assert summary["density_per_100k_pixels"] == pytest.approx(12.5)
    assert summary["mean_bbox_area"] == 160.0
    assert summary["mean_confidence"] == 0.9


def read_yolo_detections_from_text(text: str, image_width: int, image_height: int):
    path = Path("/tmp/test_detection_analysis_labels.txt")
    path.write_text(text, encoding="utf-8")
    try:
        return read_yolo_detections(path, image_width=image_width, image_height=image_height)
    finally:
        path.unlink(missing_ok=True)
