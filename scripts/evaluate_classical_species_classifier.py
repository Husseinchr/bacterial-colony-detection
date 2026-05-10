from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.species_classification import (
    evaluate_classical_species_model,
    extract_species_feature_table,
    load_classical_species_model,
    save_species_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_classical_species_model(args.model)
    features = extract_species_feature_table(
        args.dataset_dir,
        args.split_csv,
        model.detection_config,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "features.csv", index=False)
    evaluation = evaluate_classical_species_model(model, features)
    save_species_evaluation(evaluation, args.output_dir)

    print("Classical species classifier evaluation metrics")
    for key, value in evaluation.metrics.items():
        print(f"{key}: {value}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
