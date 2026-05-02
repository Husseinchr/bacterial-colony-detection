"""Render ground-truth and predicted boxes for images with large count errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Dataset root containing images and annot_tab.csv.")
    parser.add_argument("--count-errors-csv", required=True, type=Path)
    parser.add_argument("--prediction-label-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-threshold", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=1800)
    return parser.parse_args()


def read_predictions(label_path: Path, image_width: int, image_height: int, conf_threshold: float) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        confidence = float(parts[5]) if len(parts) >= 6 else 1.0
        if confidence < conf_threshold:
            continue

        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height
        x1 = round(x_center - width / 2.0)
        y1 = round(y_center - height / 2.0)
        x2 = round(x_center + width / 2.0)
        y2 = round(y_center + height / 2.0)
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_box(image, box: tuple[int, int, int, int], color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def resize_for_review(image, max_side: int):
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(args.dataset_dir / "annot_tab.csv")
    errors = pd.read_csv(args.count_errors_csv)
    worst = errors.sort_values("absolute_error", ascending=False).head(args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in worst.iterrows():
        image_name = row["image_name"]
        image_path = args.dataset_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_height, image_width = image.shape[:2]
        gt = annotations[annotations["image_name"] == image_name]
        pred_boxes = read_predictions(
            args.prediction_label_dir / f"{Path(image_name).stem}.txt",
            image_width,
            image_height,
            args.conf_threshold,
        )

        rendered = image.copy()
        for _, gt_row in gt.iterrows():
            box = (
                int(gt_row["bbox_x"]),
                int(gt_row["bbox_y"]),
                int(gt_row["bbox_x"] + gt_row["bbox_width"]),
                int(gt_row["bbox_y"] + gt_row["bbox_height"]),
            )
            draw_box(rendered, box, (0, 255, 0), 2)

        for box in pred_boxes:
            draw_box(rendered, box, (0, 0, 255), 1)

        title = (
            f"GT {int(row['true_count'])} | Pred {int(row['pred_count'])} | "
            f"Err {int(row['error'])} | green=GT red=pred"
        )
        cv2.putText(rendered, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6)
        cv2.putText(rendered, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        rendered = resize_for_review(rendered, args.max_side)
        output_path = args.output_dir / f"{Path(image_name).stem}_gt_pred_error.jpg"
        ok = cv2.imwrite(str(output_path), rendered)
        if not ok:
            raise IOError(f"Failed to save {output_path}")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

