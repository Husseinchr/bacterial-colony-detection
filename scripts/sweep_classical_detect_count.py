from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical import sweep_classical_detect_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--min-area-values", default="50,100,200,400")
    parser.add_argument("--min-circularity-values", default="0.20,0.35,0.50")
    parser.add_argument("--morph-kernel-values", default="3,5")
    parser.add_argument("--opening-iterations-values", default="1,2")
    parser.add_argument("--closing-iterations-values", default="1,2")
    parser.add_argument("--grid-rows", default=4, type=int)
    parser.add_argument("--grid-cols", default=4, type=int)
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_int_values(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_float_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def build_search_space(args: argparse.Namespace) -> list[dict]:
    values = product(
        parse_int_values(args.min_area_values),
        parse_float_values(args.min_circularity_values),
        parse_int_values(args.morph_kernel_values),
        parse_int_values(args.opening_iterations_values),
        parse_int_values(args.closing_iterations_values),
    )
    return [
        {
            "min_area": min_area,
            "min_circularity": min_circularity,
            "morph_kernel_size": morph_kernel_size,
            "opening_iterations": opening_iterations,
            "closing_iterations": closing_iterations,
        }
        for min_area, min_circularity, morph_kernel_size, opening_iterations, closing_iterations in values
    ]


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    search_space = build_search_space(args)
    results, best_config = sweep_classical_detect_count(
        args.dataset_dir,
        args.split_csv,
        base_config,
        search_space,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        limit=args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "sweep_results.csv", index=False)
    with (args.output_dir / "best_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)

    best = results.iloc[0].to_dict()
    print("Best classical detection/counting trial")
    for key, value in best.items():
        print(f"{key}: {value}")
    print(f"Saved sweep results to: {args.output_dir}")


if __name__ == "__main__":
    main()
