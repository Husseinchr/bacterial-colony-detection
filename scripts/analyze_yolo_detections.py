from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import build_density_grid, read_yolo_detections, summarize_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO colony detections for count, morphology, and density.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prediction-label", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=1800)
    return parser.parse_args()


def draw_detection_overlay(image, detections):
    output = image.copy()
    for _, row in detections.iterrows():
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        cx = int(round(row["centroid_x"]))
        cy = int(round(row["centroid_y"]))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(output, (cx, cy), 3, (0, 255, 255), -1)
    return output


def resize_for_review(image, max_side: int):
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def save_size_distribution(path: Path, detections) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if detections.empty:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(detections["equivalent_diameter"], bins=20, color="#2563eb", edgecolor="white")
        ax.set_xlabel("Equivalent diameter from bounding box (pixels)")
        ax.set_ylabel("Colony count")
        ax.set_title("Detected colony size distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_density_heatmap(path: Path, density_grid, rows: int, cols: int) -> None:
    matrix = density_grid.pivot(index="grid_row", columns="grid_col", values="count").to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="magma")
    ax.set_title("Spatial colony density")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    for row_idx in range(rows):
        for col_idx in range(cols):
            ax.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Detections")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    image_height, image_width = image.shape[:2]
    detections = read_yolo_detections(args.prediction_label, image_width, image_height, args.conf_threshold)
    density_grid = build_density_grid(detections, image_width, image_height, args.grid_rows, args.grid_cols)
    summary = summarize_detections(detections, image_width, image_height)
    summary["conf_threshold"] = float(args.conf_threshold)
    summary["prediction_label"] = str(args.prediction_label)
    summary["image"] = str(args.image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections.to_csv(args.output_dir / "detections.csv", index=False)
    density_grid.to_csv(args.output_dir / "density_grid.csv", index=False)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    overlay = draw_detection_overlay(image, detections)
    overlay = resize_for_review(overlay, args.max_side)
    ok = cv2.imwrite(str(args.output_dir / "overlay.png"), overlay)
    if not ok:
        raise IOError(f"Failed to save overlay to {args.output_dir / 'overlay.png'}")

    save_size_distribution(args.output_dir / "size_distribution.png", detections)
    save_density_heatmap(args.output_dir / "density_heatmap.png", density_grid, args.grid_rows, args.grid_cols)

    print(f"Detected colonies: {summary['colony_count']}")
    print(f"Saved analysis outputs to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
