"""Create image-level train/validation/test splits for the Figshare colony dataset."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Root of Figshare dataset containing JPG files and annot_tab.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where train.csv, val.csv, test.csv, and all.csv will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


def load_image_level_table(dataset_dir: Path) -> pd.DataFrame:
    """Aggregate Figshare box annotations into one row per image."""

    annotation_path = dataset_dir / "annot_tab.csv"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation CSV: {annotation_path}")

    annotations = pd.read_csv(annotation_path)
    required_columns = {
        "label_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "image_name",
        "image_width",
        "image_height",
    }
    missing = required_columns - set(annotations.columns)
    if missing:
        raise ValueError(f"Missing required columns in annot_tab.csv: {sorted(missing)}")

    grouped = (
        annotations.groupby("image_name")
        .agg(
            label_name=("label_name", "first"),
            colony_count=("label_name", "size"),
            image_width=("image_width", "first"),
            image_height=("image_height", "first"),
        )
        .reset_index()
    )
    grouped["image_path"] = grouped["image_name"]
    grouped["annotation_path"] = "annot_tab.csv"
    grouped["image_exists"] = grouped["image_name"].map(lambda name: (dataset_dir / name).exists())

    missing_images = grouped.loc[~grouped["image_exists"], "image_name"].tolist()
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} images listed in CSV are missing. Examples: {preview}")

    return grouped.drop(columns=["image_exists"])


def stratified_image_split(
    images: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Split images within each species label to reduce class imbalance drift."""

    rng = random.Random(seed)
    rows = []

    for _, group in images.groupby("label_name", sort=True):
        records = group.to_dict("records")
        rng.shuffle(records)

        n_total = len(records)
        n_train = round(n_total * train_ratio)
        n_val = round(n_total * val_ratio)

        # Keep tiny classes represented in train when possible.
        if n_total >= 3:
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))
        elif n_total == 2:
            n_train = 1
            n_val = 0
        else:
            n_train = 1
            n_val = 0

        for idx, record in enumerate(records):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            record["split"] = split
            rows.append(record)

    split_df = pd.DataFrame(rows)
    return split_df.sort_values(["split", "label_name", "image_name"]).reset_index(drop=True)


def write_splits(split_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_dir / "all.csv", index=False)
    for split in ["train", "val", "test"]:
        split_df[split_df["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)


def print_summary(split_df: pd.DataFrame) -> None:
    print("Split summary:")
    print(split_df.groupby("split").agg(images=("image_name", "count"), colonies=("colony_count", "sum")).to_string())
    print("\nSpecies coverage by split:")
    print(pd.crosstab(split_df["label_name"], split_df["split"]).to_string())


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    images = load_image_level_table(dataset_dir)
    split_df = stratified_image_split(images, args.train_ratio, args.val_ratio, args.seed)
    write_splits(split_df, args.output_dir)
    print_summary(split_df)
    print(f"\nSaved splits to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

