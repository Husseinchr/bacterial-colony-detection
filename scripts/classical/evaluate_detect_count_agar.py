from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.detect_count import (
    ClassicalCountConfig,
    CountPrediction,
    count_colonies,
    evaluate_count_predictions,
    read_image,
    render_count_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold-method", default="otsu", choices=("otsu", "adaptive"))
    parser.add_argument("--threshold-polarity", default="dark", choices=("dark", "bright"))
    parser.add_argument("--gaussian-kernel", default=5, type=int)
    parser.add_argument("--clahe-clip-limit", default=2.0, type=float)
    parser.add_argument("--clahe-tile-grid-size", default=8, type=int)
    parser.add_argument("--background-kernel", default=61, type=int)
    parser.add_argument("--adaptive-block-size", default=51, type=int)
    parser.add_argument("--adaptive-c", default=2.0, type=float)
    parser.add_argument("--morph-open-kernel", default=3, type=int)
    parser.add_argument("--morph-close-kernel", default=3, type=int)
    parser.add_argument("--min-area", default=12.0, type=float)
    parser.add_argument("--max-area", default=10000.0, type=float)
    parser.add_argument("--min-circularity", default=0.05, type=float)
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", default=40, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ClassicalCountConfig(
        gaussian_kernel=args.gaussian_kernel,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
        background_kernel=args.background_kernel,
        threshold_method=args.threshold_method,
        threshold_polarity=args.threshold_polarity,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        morph_open_kernel=args.morph_open_kernel,
        morph_close_kernel=args.morph_close_kernel,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
    )
    split_df = pd.read_csv(args.split_csv)
    if "detection_eligible" in split_df.columns:
        split_df = split_df[split_df["detection_eligible"].map(is_truthy)].copy()
    if args.max_images > 0:
        split_df = split_df.head(args.max_images).copy()
    if split_df.empty:
        raise ValueError(f"No detection/counting rows found in {args.split_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for index, row in split_df.reset_index(drop=True).iterrows():
        image_relative = str(row["image_path"])
        image_path = args.dataset_dir / image_relative
        image = read_image(image_path)
        predicted_count, mask, contours = count_colonies(image, config)
        true_count = int(row["colonies_number"])
        prediction = CountPrediction(
            image_path=image_relative,
            true_count=true_count,
            predicted_count=int(predicted_count),
            category=str(row.get("category", "")),
            primary_class=str(row.get("primary_class", "")),
            error=int(predicted_count - true_count),
            abs_error=int(abs(predicted_count - true_count)),
            squared_error=int((predicted_count - true_count) ** 2),
        )
        predictions.append(prediction)
        if args.save_overlays and index < args.overlay_limit:
            overlay = render_count_overlay(image, predicted_count, contours)
            output_name = safe_output_name(image_relative, ".jpg")
            cv2.imwrite(str(overlay_dir / output_name), overlay)
            cv2.imwrite(str(overlay_dir / safe_output_name(image_relative, "_mask.png")), mask)

    evaluation = evaluate_count_predictions(predictions, config)
    prediction_rows = [asdict(prediction) for prediction in evaluation.predictions]
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.metrics, f, indent=2)

    print("Classical AGAR detection/counting validation summary")
    for key, value in evaluation.metrics.items():
        if key != "config":
            print(f"{key}: {value}")
    print(f"Saved predictions and metrics to: {args.output_dir}")


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_output_name(path: str, suffix: str) -> str:
    source = Path(path)
    stem = "__".join(source.with_suffix("").parts)
    return f"{stem}{suffix}"


if __name__ == "__main__":
    main()
