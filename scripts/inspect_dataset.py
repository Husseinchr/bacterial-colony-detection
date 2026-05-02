"""Inspect a colony dataset folder and summarize image/annotation structure."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOTATION_EXTENSIONS = {".json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Root directory of the downloaded dataset.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for a CSV inventory. Example: data/annotations/dataset_inventory.csv",
    )
    parser.add_argument("--max-json-samples", type=int, default=5, help="Number of JSON files to sample.")
    return parser.parse_args()


def find_files(root: Path, extensions: set[str]) -> list[Path]:
    """Find files with matching extensions under a root directory."""

    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def image_info(path: Path) -> dict[str, Any]:
    """Read basic image metadata without loading it into the project pipeline."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"width": None, "height": None, "channels": None, "readable": False}

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {"width": width, "height": height, "channels": channels, "readable": True}


def summarize_json(path: Path) -> dict[str, Any]:
    """Return a lightweight summary of one JSON annotation file."""

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"path": str(path), "error": f"{exc.__class__.__name__}: {exc}"}

    summary: dict[str, Any] = {
        "path": str(path),
        "top_level_type": type(data).__name__,
        "top_level_keys": [],
        "annotation_count_guess": None,
        "candidate_annotation_keys": [],
    }

    if isinstance(data, dict):
        summary["top_level_keys"] = list(data.keys())
        for key, value in data.items():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value[:10]):
                summary["candidate_annotation_keys"].append(key)
                if summary["annotation_count_guess"] is None:
                    summary["annotation_count_guess"] = len(value)
    elif isinstance(data, list):
        summary["annotation_count_guess"] = len(data)
        if data and isinstance(data[0], dict):
            summary["candidate_annotation_keys"] = ["<root_list>"]

    return summary


def build_inventory(images: list[Path], json_files: list[Path], dataset_dir: Path) -> pd.DataFrame:
    """Build one row per image with matching JSON information when available."""

    json_by_stem = {path.stem: path for path in json_files}
    rows = []

    for image_path in images:
        info = image_info(image_path)
        annotation_path = json_by_stem.get(image_path.stem)
        rows.append(
            {
                "image_path": str(image_path.relative_to(dataset_dir)),
                "image_stem": image_path.stem,
                "image_extension": image_path.suffix.lower(),
                "width": info["width"],
                "height": info["height"],
                "channels": info["channels"],
                "readable": info["readable"],
                "has_json_annotation": annotation_path is not None,
                "json_path": "" if annotation_path is None else str(annotation_path.relative_to(dataset_dir)),
            }
        )

    return pd.DataFrame(rows)


def print_json_samples(json_files: list[Path], max_samples: int) -> None:
    """Print compact JSON structure samples for annotation analysis."""

    print("\nJSON annotation samples:")
    if not json_files:
        print("  No JSON files found.")
        return

    for path in json_files[:max_samples]:
        summary = summarize_json(path)
        print(f"\n- {summary['path']}")
        for key, value in summary.items():
            if key != "path":
                print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    images = find_files(dataset_dir, IMAGE_EXTENSIONS)
    json_files = find_files(dataset_dir, ANNOTATION_EXTENSIONS)
    inventory = build_inventory(images, json_files, dataset_dir)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Images found: {len(images)}")
    print(f"JSON files found: {len(json_files)}")

    if not inventory.empty:
        print("\nImage extensions:")
        for extension, count in Counter(inventory["image_extension"]).most_common():
            print(f"  {extension}: {count}")

        print("\nReadable images:")
        print(f"  {int(inventory['readable'].sum())} / {len(inventory)}")

        print("\nImage dimensions:")
        print(inventory[["width", "height", "channels"]].drop_duplicates().head(20).to_string(index=False))

        print("\nImage/JSON pairing:")
        print(f"  images with matching JSON by filename stem: {int(inventory['has_json_annotation'].sum())}")
        print(f"  images without matching JSON: {int((~inventory['has_json_annotation']).sum())}")

    print_json_samples(json_files, args.max_json_samples)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(args.output_csv, index=False)
        print(f"\nSaved inventory CSV to: {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

