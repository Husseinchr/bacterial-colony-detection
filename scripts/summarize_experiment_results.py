"""Create a compact CSV summary for model count-evaluation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        nargs=3,
        metavar=("MODEL_NAME", "COUNT_ERRORS_CSV", "CONF_THRESHOLD"),
        required=True,
        help="Add one result row. Can be repeated.",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def metrics_for_file(model_name: str, csv_path: Path, conf_threshold: str) -> dict[str, float | str]:
    df = pd.read_csv(csv_path)
    errors = df["pred_count"] - df["true_count"]
    abs_errors = errors.abs()
    return {
        "model_name": model_name,
        "conf_threshold": float(conf_threshold),
        "images": int(len(df)),
        "true_colonies": int(df["true_count"].sum()),
        "pred_colonies": int(df["pred_count"].sum()),
        "mean_error": float(errors.mean()),
        "mae": float(abs_errors.mean()),
        "rmse": float((errors * errors).mean() ** 0.5),
        "mape_percent": float((abs_errors / df["true_count"].clip(lower=1)).mean() * 100.0),
    }


def main() -> None:
    args = parse_args()
    rows = [metrics_for_file(name, Path(path), threshold) for name, path, threshold in args.result]
    summary = pd.DataFrame(rows).sort_values("mae")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

