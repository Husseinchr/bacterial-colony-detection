"""Evaluate YOLO prediction text files as colony count estimates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path, help="Split CSV with image_name and colony_count columns.")
    parser.add_argument(
        "--prediction-label-dir",
        required=True,
        type=Path,
        help="Directory containing YOLO prediction TXT files from predict save_txt=True.",
    )
    parser.add_argument("--output-csv", required=True, type=Path, help="Path for per-image count error CSV.")
    return parser.parse_args()


def count_prediction_file(path: Path) -> int:
    """Count YOLO detections in one prediction label file."""

    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    return len(text.splitlines())


def compute_metrics(results: pd.DataFrame) -> dict[str, float]:
    errors = results["pred_count"] - results["true_count"]
    abs_errors = errors.abs()
    squared_errors = errors * errors
    percentage_errors = abs_errors / results["true_count"].clip(lower=1)
    return {
        "images": float(len(results)),
        "true_colonies": float(results["true_count"].sum()),
        "pred_colonies": float(results["pred_count"].sum()),
        "mean_error": float(errors.mean()),
        "mae": float(abs_errors.mean()),
        "rmse": float(squared_errors.mean() ** 0.5),
        "mape_percent": float(percentage_errors.mean() * 100.0),
    }


def main() -> None:
    args = parse_args()
    split_df = pd.read_csv(args.split_csv)

    required = {"image_name", "colony_count"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    rows = []
    for _, row in split_df.iterrows():
        image_name = row["image_name"]
        label_path = args.prediction_label_dir / f"{Path(image_name).stem}.txt"
        true_count = int(row["colony_count"])
        pred_count = count_prediction_file(label_path)
        rows.append(
            {
                "image_name": image_name,
                "true_count": true_count,
                "pred_count": pred_count,
                "error": pred_count - true_count,
                "absolute_error": abs(pred_count - true_count),
            }
        )

    results = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)

    metrics = compute_metrics(results)
    print("Count metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    print(f"\nSaved per-image count errors to: {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

