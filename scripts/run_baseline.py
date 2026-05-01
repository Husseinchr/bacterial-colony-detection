"""Run the classical bacterial colony detection baseline on one image."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.counting import summarize_count
from src.features import extract_region_features, summarize_features
from src.preprocessing import load_image, preprocess_image
from src.segmentation import segment_colonies
from src.visualization import (
    draw_colony_markers,
    labels_to_color,
    make_overlay,
    save_debug_panel,
    save_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Path to a petri dish image.")
    parser.add_argument("--config", default=Path("configs/default.yaml"), type=Path, help="YAML config path.")
    parser.add_argument("--output-dir", default=Path("outputs/baseline"), type=Path, help="Directory for outputs.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    overlay = make_overlay(preprocessing.image_bgr, segmentation.mask, alpha=config["outputs"]["overlay_alpha"])
    marked_overlay = draw_colony_markers(overlay, features)
    label_vis = labels_to_color(segmentation.labels)

    save_image(args.output_dir / "mask.png", segmentation.mask)
    save_image(args.output_dir / "labels.png", label_vis)
    save_image(args.output_dir / "overlay.png", marked_overlay)
    features.to_csv(args.output_dir / "features.csv", index=False)

    summary = {
        "image": str(args.image),
        "count": asdict(count_summary),
        "features": feature_summary,
        "config": config,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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

    print(f"Detected colonies: {count_summary.colony_count}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
