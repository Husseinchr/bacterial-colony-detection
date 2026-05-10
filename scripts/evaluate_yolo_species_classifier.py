from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo import evaluate_yolo_species_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--prediction-label-dir", required=True, type=Path)
    parser.add_argument("--class-names", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-threshold", required=True, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = evaluate_yolo_species_classification(
        args.split_csv,
        args.prediction_label_dir,
        args.class_names,
        args.conf_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    evaluation.class_distribution.to_csv(args.output_dir / "class_distribution.csv", index=False)
    evaluation.report.to_csv(args.output_dir / "classification_report.csv", index=False)
    evaluation.confusion.to_csv(args.output_dir / "confusion_matrix.csv")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.metrics, f, indent=2)

    print("YOLO species classification evaluation")
    for key, value in evaluation.metrics.items():
        print(f"{key}: {value}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
