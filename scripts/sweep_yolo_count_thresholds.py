"""Sweep YOLO prediction confidence thresholds and evaluate count error."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path, help="Split CSV with image_name and colony_count.")
    parser.add_argument(
        "--prediction-label-dir",
        required=True,
        type=Path,
        help="Directory with YOLO prediction TXT files saved with confidence values.",
    )
    parser.add_argument("--output-csv", required=True, type=Path, help="Path for threshold sweep CSV.")
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--max-conf", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def read_prediction_confidences(path: Path) -> list[float]:
    """Read confidence scores from YOLO prediction labels.

    Supports lines in either:
      class x y w h
      class x y w h confidence

    Files without confidence scores are treated as confidence 1.0 for all rows.
    """

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    confidences = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6:
            confidences.append(float(parts[5]))
        else:
            confidences.append(1.0)
    return confidences


def thresholds(min_conf: float, max_conf: float, step: float) -> list[float]:
    values = []
    current = min_conf
    while current <= max_conf + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def metric_row(split_df: pd.DataFrame, predictions: dict[str, list[float]], threshold: float) -> dict[str, float]:
    rows = []
    for _, row in split_df.iterrows():
        image_name = row["image_name"]
        true_count = int(row["colony_count"])
        pred_count = sum(conf >= threshold for conf in predictions.get(image_name, []))
        rows.append((true_count, pred_count))

    counts = pd.DataFrame(rows, columns=["true_count", "pred_count"])
    errors = counts["pred_count"] - counts["true_count"]
    abs_errors = errors.abs()
    return {
        "threshold": threshold,
        "true_colonies": float(counts["true_count"].sum()),
        "pred_colonies": float(counts["pred_count"].sum()),
        "mean_error": float(errors.mean()),
        "mae": float(abs_errors.mean()),
        "rmse": float((errors * errors).mean() ** 0.5),
        "mape_percent": float((abs_errors / counts["true_count"].clip(lower=1)).mean() * 100.0),
    }


def main() -> None:
    args = parse_args()
    split_df = pd.read_csv(args.split_csv)
    predictions = {
        image_name: read_prediction_confidences(args.prediction_label_dir / f"{Path(image_name).stem}.txt")
        for image_name in split_df["image_name"]
    }

    sweep = pd.DataFrame(
        [
            metric_row(split_df, predictions, threshold)
            for threshold in thresholds(args.min_conf, args.max_conf, args.step)
        ]
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_csv, index=False)

    best = sweep.sort_values(["mae", "rmse"]).iloc[0]
    print("Best threshold by MAE:")
    for column in sweep.columns:
        print(f"  {column}: {best[column]:.3f}")
    print(f"\nSaved threshold sweep to: {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

