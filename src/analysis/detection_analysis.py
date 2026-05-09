from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DETECTION_COLUMNS = [
    "detection_id",
    "class_id",
    "confidence",
    "x_center_norm",
    "y_center_norm",
    "width_norm",
    "height_norm",
    "x1",
    "y1",
    "x2",
    "y2",
    "centroid_x",
    "centroid_y",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "aspect_ratio",
    "equivalent_diameter",
]


def read_yolo_detections(
    label_path: str | Path,
    image_width: int,
    image_height: int,
    conf_threshold: float = 0.0,
) -> pd.DataFrame:
    label_path = Path(label_path)
    if not label_path.exists():
        return pd.DataFrame(columns=DETECTION_COLUMNS)

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame(columns=DETECTION_COLUMNS)

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])
        confidence = float(parts[5]) if len(parts) >= 6 else 1.0

        if confidence < conf_threshold:
            continue

        x_center = x_center_norm * image_width
        y_center = y_center_norm * image_height
        box_width = width_norm * image_width
        box_height = height_norm * image_height

        x1 = int(round(x_center - box_width / 2.0))
        y1 = int(round(y_center - box_height / 2.0))
        x2 = int(round(x_center + box_width / 2.0))
        y2 = int(round(y_center + box_height / 2.0))

        x1 = min(max(x1, 0), image_width)
        y1 = min(max(y1, 0), image_height)
        x2 = min(max(x2, 0), image_width)
        y2 = min(max(y2, 0), image_height)

        clipped_width = max(0, x2 - x1)
        clipped_height = max(0, y2 - y1)
        if clipped_width == 0 or clipped_height == 0:
            continue

        area = float(clipped_width * clipped_height)
        rows.append(
            {
                "detection_id": len(rows) + 1,
                "class_id": class_id,
                "confidence": confidence,
                "x_center_norm": x_center_norm,
                "y_center_norm": y_center_norm,
                "width_norm": width_norm,
                "height_norm": height_norm,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "centroid_x": (x1 + x2) / 2.0,
                "centroid_y": (y1 + y2) / 2.0,
                "bbox_width": clipped_width,
                "bbox_height": clipped_height,
                "bbox_area": area,
                "aspect_ratio": clipped_width / clipped_height,
                "equivalent_diameter": float(np.sqrt(4.0 * area / np.pi)),
            }
        )

    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def build_density_grid(
    detections: pd.DataFrame,
    image_width: int,
    image_height: int,
    rows: int = 4,
    cols: int = 4,
) -> pd.DataFrame:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    grid_rows = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            x1 = round(col_idx * image_width / cols)
            x2 = round((col_idx + 1) * image_width / cols)
            y1 = round(row_idx * image_height / rows)
            y2 = round((row_idx + 1) * image_height / rows)
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            area = width * height

            if detections.empty:
                count = 0
            else:
                in_cell = (
                    (detections["centroid_x"] >= x1)
                    & (detections["centroid_x"] < x2)
                    & (detections["centroid_y"] >= y1)
                    & (detections["centroid_y"] < y2)
                )
                if col_idx == cols - 1:
                    in_cell = in_cell | (
                        (detections["centroid_x"] == image_width)
                        & (detections["centroid_y"] >= y1)
                        & (detections["centroid_y"] < y2)
                    )
                if row_idx == rows - 1:
                    in_cell = in_cell | (
                        (detections["centroid_y"] == image_height)
                        & (detections["centroid_x"] >= x1)
                        & (detections["centroid_x"] < x2)
                    )
                count = int(in_cell.sum())

            density = 0.0 if area == 0 else count / area * 100_000.0
            grid_rows.append(
                {
                    "grid_row": row_idx,
                    "grid_col": col_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cell_area": area,
                    "count": count,
                    "density_per_100k_pixels": density,
                }
            )

    return pd.DataFrame(grid_rows)


def summarize_detections(detections: pd.DataFrame, image_width: int, image_height: int) -> dict[str, float | int]:
    image_area = int(image_width * image_height)
    count = int(len(detections))
    if detections.empty:
        return {
            "colony_count": 0,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_area_pixels": image_area,
            "density_per_100k_pixels": 0.0,
            "mean_bbox_area": 0.0,
            "median_bbox_area": 0.0,
            "mean_equivalent_diameter": 0.0,
            "median_equivalent_diameter": 0.0,
            "mean_aspect_ratio": 0.0,
            "centroid_std_x": 0.0,
            "centroid_std_y": 0.0,
            "mean_confidence": 0.0,
        }

    return {
        "colony_count": count,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_area_pixels": image_area,
        "density_per_100k_pixels": float(count / image_area * 100_000.0) if image_area else 0.0,
        "mean_bbox_area": float(detections["bbox_area"].mean()),
        "median_bbox_area": float(detections["bbox_area"].median()),
        "mean_equivalent_diameter": float(detections["equivalent_diameter"].mean()),
        "median_equivalent_diameter": float(detections["equivalent_diameter"].median()),
        "mean_aspect_ratio": float(detections["aspect_ratio"].mean()),
        "centroid_std_x": float(detections["centroid_x"].std(ddof=0)),
        "centroid_std_y": float(detections["centroid_y"].std(ddof=0)),
        "mean_confidence": float(detections["confidence"].mean()),
    }
