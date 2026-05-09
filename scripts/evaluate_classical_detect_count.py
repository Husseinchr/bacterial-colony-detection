from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical import evaluate_classical_detect_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--grid-rows", default=4, type=int)
    parser.add_argument("--grid-cols", default=4, type=int)
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    evaluation = evaluate_classical_detect_count(
        args.dataset_dir,
        args.split_csv,
        config,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        limit=args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.per_image.to_csv(args.output_dir / "per_image_count_errors.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.summary, f, indent=2)

    print("Classical detection/counting evaluation")
    for key, value in evaluation.summary.items():
        print(f"{key}: {value}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
