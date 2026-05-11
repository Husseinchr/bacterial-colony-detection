from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import build_inventory, split_inventory, write_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", default=0.70, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--test-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--skip-image-read", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory, summary = build_inventory(args.dataset_dir, read_images=not args.skip_image_read)
    if inventory.empty:
        raise ValueError(
            f"No AGAR image records found under {args.dataset_dir}. "
            "Check that the path contains data/countable, data/empty, and data/uncountable."
        )
    split_df = split_inventory(
        inventory,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_split_files(split_df, args.output_dir)
    split_summary = {
        "dataset": summary,
        "splits": split_df.groupby("split").size().astype(int).to_dict(),
        "categories_by_split": {
            split: values.astype(int).to_dict()
            for split, values in split_df.groupby("split")["category"].value_counts().groupby(level=0)
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print("AGAR-primary split summary")
    print(split_df.groupby("split").size().to_string())
    print("")
    print(split_df.groupby(["split", "category"]).size().to_string())
    print(f"Saved splits to: {args.output_dir}")


if __name__ == "__main__":
    main()
