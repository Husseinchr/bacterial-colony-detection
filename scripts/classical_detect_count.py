from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical import run_classical_detect_count
from src.preprocessing import load_image
from src.visualization import save_debug_panel, save_density_heatmap, save_image, save_size_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--grid-rows", default=4, type=int)
    parser.add_argument("--grid-cols", default=4, type=int)
    parser.add_argument("--moderate-count", default=50, type=int)
    parser.add_argument("--high-count", default=200, type=int)
    parser.add_argument("--moderate-density", default=1.0, type=float)
    parser.add_argument("--high-density", default=3.0, type=float)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image = load_image(args.image)
    result = run_classical_detect_count(
        image,
        config,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        moderate_count=args.moderate_count,
        high_count=args.high_count,
        moderate_density=args.moderate_density,
        high_density=args.high_density,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(args.output_dir / "mask.png", result.mask)
    save_image(args.output_dir / "labels.png", result.labels_visualization)
    save_image(args.output_dir / "overlay.png", result.overlay)
    result.features.to_csv(args.output_dir / "features.csv", index=False)
    result.density_grid.to_csv(args.output_dir / "density_grid.csv", index=False)
    save_size_distribution(args.output_dir / "size_distribution.png", result.features["equivalent_diameter"])
    save_density_heatmap(args.output_dir / "density_heatmap.png", result.density_grid, args.grid_rows, args.grid_cols)
    save_debug_panel(
        args.output_dir / "debug_panel.png",
        result.debug_images["original_bgr"],
        result.debug_images["grayscale"],
        result.debug_images["enhanced"],
        result.debug_images["normalized"],
        result.mask,
        result.overlay,
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2)

    print(f"Model family: {result.summary['model_family']}")
    print(f"Task: {result.summary['task']}")
    print(f"Detected colonies: {result.summary['colony_count']}")
    print(f"Growth level: {result.summary['growth_level']}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
