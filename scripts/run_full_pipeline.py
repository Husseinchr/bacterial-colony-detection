from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (
    build_density_grid,
    classify_growth,
    load_class_names,
    read_yolo_detections,
    summarize_class_distribution,
    summarize_detections,
)
from src.counting import summarize_count
from src.features import extract_region_features, summarize_features
from src.preprocessing import load_image, preprocess_image
from src.segmentation import segment_colonies
from src.visualization import (
    draw_colony_markers,
    labels_to_color,
    make_overlay,
    save_debug_panel,
    save_density_heatmap,
    save_image,
    save_size_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete bacterial colony analysis pipeline.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("classical", "yolo"), default="yolo")
    parser.add_argument("--config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--prediction-label", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--class-names", type=Path, default=None, help="Optional classes.txt or YOLO data.yaml.")
    parser.add_argument("--conf-threshold", type=float, default=0.32)
    parser.add_argument("--predict-conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--moderate-count", type=int, default=50)
    parser.add_argument("--high-count", type=int, default=200)
    parser.add_argument("--moderate-density", type=float, default=1.0)
    parser.add_argument("--high-density", type=float, default=3.0)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_yolo_prediction(image_path: Path, weights_path: Path, output_dir: Path, imgsz: int, predict_conf: float, max_det: int) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required when --weights is used") from exc

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=predict_conf,
        max_det=max_det,
        verbose=False,
    )
    result = results[0]
    label_path = output_dir / "yolo_prediction.txt"
    rows = []

    if result.boxes is not None and len(result.boxes) > 0:
        xywhn = result.boxes.xywhn.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for class_id, box, confidence in zip(classes, xywhn, confidences):
            rows.append(
                f"{int(class_id)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {float(confidence):.6f}"
            )

    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return label_path


def add_growth(summary: dict, args: argparse.Namespace) -> dict:
    summary["growth_level"] = classify_growth(
        int(summary["colony_count"]),
        float(summary["density_per_100k_pixels"]),
        moderate_count=args.moderate_count,
        high_count=args.high_count,
        moderate_density=args.moderate_density,
        high_density=args.high_density,
    )
    summary["growth_thresholds"] = {
        "moderate_count": args.moderate_count,
        "high_count": args.high_count,
        "moderate_density": args.moderate_density,
        "high_density": args.high_density,
    }
    return summary


def run_yolo_mode(args: argparse.Namespace) -> dict:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    if args.prediction_label is None and args.weights is None:
        raise ValueError("YOLO mode requires either --prediction-label or --weights")

    label_path = args.prediction_label
    if label_path is None:
        label_path = run_yolo_prediction(
            args.image,
            args.weights,
            args.output_dir,
            args.imgsz,
            args.predict_conf,
            args.max_det,
        )

    image_height, image_width = image.shape[:2]
    class_names = load_class_names(args.class_names)
    detections = read_yolo_detections(label_path, image_width, image_height, args.conf_threshold, class_names)
    density_grid = build_density_grid(detections, image_width, image_height, args.grid_rows, args.grid_cols)
    class_distribution = summarize_class_distribution(detections)
    summary = summarize_detections(detections, image_width, image_height)
    summary["mode"] = "yolo"
    summary["conf_threshold"] = args.conf_threshold
    summary["prediction_label"] = str(label_path)
    summary["image"] = str(args.image)
    summary = add_growth(summary, args)

    detections.to_csv(args.output_dir / "detections.csv", index=False)
    density_grid.to_csv(args.output_dir / "density_grid.csv", index=False)
    class_distribution.to_csv(args.output_dir / "class_distribution.csv", index=False)

    overlay = draw_yolo_overlay(image, detections, summary)
    save_image(args.output_dir / "overlay.png", overlay)
    save_size_distribution(
        args.output_dir / "size_distribution.png",
        detections["equivalent_diameter"] if not detections.empty else pd.Series(dtype=float),
        title="Detected colony size distribution",
        xlabel="Equivalent diameter from bounding box (pixels)",
    )
    save_density_heatmap(args.output_dir / "density_heatmap.png", density_grid, args.grid_rows, args.grid_cols)
    return summary


def draw_yolo_overlay(image, detections: pd.DataFrame, summary: dict):
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

    label = f"Count: {summary['colony_count']} | Growth: {summary['growth_level']} | conf >= {summary['conf_threshold']:.2f}"
    cv2.putText(output, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6)
    cv2.putText(output, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return output


def run_classical_mode(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    image = load_image(args.image)
    preprocessing = preprocess_image(image, **config["preprocessing"])
    segmentation = segment_colonies(
        preprocessing.normalized,
        threshold_method=config["pipeline"]["threshold_method"],
        split_touching=config["pipeline"]["split_touching"],
        **config["segmentation"],
    )
    features = extract_region_features(segmentation.labels)
    count_summary = summarize_count(segmentation.mask, segmentation.labels)
    feature_summary = summarize_features(features)
    image_height, image_width = image.shape[:2]
    density_grid = build_density_grid(features, image_width, image_height, args.grid_rows, args.grid_cols)

    overlay = make_overlay(preprocessing.image_bgr, segmentation.mask, alpha=config["outputs"]["overlay_alpha"])
    marked_overlay = draw_colony_markers(overlay, features)

    save_image(args.output_dir / "mask.png", segmentation.mask)
    save_image(args.output_dir / "labels.png", labels_to_color(segmentation.labels))
    save_image(args.output_dir / "overlay.png", marked_overlay)
    features.to_csv(args.output_dir / "features.csv", index=False)
    density_grid.to_csv(args.output_dir / "density_grid.csv", index=False)
    save_size_distribution(args.output_dir / "size_distribution.png", features["equivalent_diameter"])
    save_density_heatmap(args.output_dir / "density_heatmap.png", density_grid, args.grid_rows, args.grid_cols)

    if config["pipeline"].get("save_debug_panel", True):
        save_debug_panel(
            args.output_dir / "debug_panel.png",
            preprocessing.image_bgr,
            preprocessing.grayscale,
            preprocessing.enhanced,
            preprocessing.normalized,
            segmentation.mask,
            marked_overlay,
        )

    summary = {
        "mode": "classical",
        "image": str(args.image),
        "count": asdict(count_summary),
        "colony_count": count_summary.colony_count,
        "density_per_100k_pixels": count_summary.density_per_100k_pixels,
        "features": feature_summary,
        "config": config,
    }
    return add_growth(summary, args)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "classical":
        summary = run_classical_mode(args)
    else:
        summary = run_yolo_mode(args)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Mode: {args.mode}")
    print(f"Detected colonies: {summary['colony_count']}")
    print(f"Growth level: {summary['growth_level']}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
