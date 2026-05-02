"""Summarize per-image colony count errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-errors-csv", required=True, type=Path, help="CSV from evaluate_yolo_count_predictions.py.")
    parser.add_argument("--split-csv", required=True, type=Path, help="Split CSV with label_name and colony_count.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for error analysis CSV files.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of worst cases to save.")
    return parser.parse_args()


def compute_group_metrics(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = df.groupby(group_column)
    metrics = grouped.agg(
        images=("image_name", "count"),
        true_colonies=("true_count", "sum"),
        pred_colonies=("pred_count", "sum"),
        mean_error=("error", "mean"),
        mae=("absolute_error", "mean"),
        max_absolute_error=("absolute_error", "max"),
    ).reset_index()
    metrics["mape_percent"] = grouped.apply(
        lambda group: (group["absolute_error"] / group["true_count"].clip(lower=1)).mean() * 100.0,
        include_groups=False,
    ).to_numpy()
    return metrics.sort_values("mae", ascending=False)


def main() -> None:
    args = parse_args()
    errors = pd.read_csv(args.count_errors_csv)
    split_df = pd.read_csv(args.split_csv)

    merged = errors.merge(
        split_df[["image_name", "label_name", "image_width", "image_height"]],
        on="image_name",
        how="left",
        validate="one_to_one",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    worst = merged.sort_values("absolute_error", ascending=False).head(args.top_k)
    overcounts = merged.sort_values("error", ascending=False).head(args.top_k)
    undercounts = merged.sort_values("error", ascending=True).head(args.top_k)
    by_species = compute_group_metrics(merged, "label_name")

    worst.to_csv(args.output_dir / "worst_absolute_errors.csv", index=False)
    overcounts.to_csv(args.output_dir / "worst_overcounts.csv", index=False)
    undercounts.to_csv(args.output_dir / "worst_undercounts.csv", index=False)
    by_species.to_csv(args.output_dir / "count_metrics_by_species.csv", index=False)

    print("Overall:")
    print(f"  images: {len(merged)}")
    print(f"  true_colonies: {merged['true_count'].sum()}")
    print(f"  pred_colonies: {merged['pred_count'].sum()}")
    print(f"  mean_error: {merged['error'].mean():.3f}")
    print(f"  mae: {merged['absolute_error'].mean():.3f}")
    print(f"  rmse: {((merged['error'] ** 2).mean() ** 0.5):.3f}")
    print("\nWorst species by MAE:")
    print(by_species.head(10).to_string(index=False))
    print(f"\nSaved analysis CSVs to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

