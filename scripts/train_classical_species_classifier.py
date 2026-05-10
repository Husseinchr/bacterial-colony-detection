from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.species_classification import (
    evaluate_classical_species_model,
    extract_species_feature_table,
    save_classical_species_model,
    save_species_evaluation,
    train_classical_species_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--detect-count-config", default=Path("configs/classical_detect_count_tuned.yaml"), type=Path)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--n-estimators", default=400, type=int)
    parser.add_argument("--limit-train", default=None, type=int)
    parser.add_argument("--limit-val", default=None, type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detection_config = load_config(args.detect_count_config)

    train_features = extract_species_feature_table(
        args.dataset_dir,
        args.train_csv,
        detection_config,
        limit=args.limit_train,
    )
    val_features = extract_species_feature_table(
        args.dataset_dir,
        args.val_csv,
        detection_config,
        limit=args.limit_val,
    )

    train_features.to_csv(args.output_dir / "train_features.csv", index=False)
    val_features.to_csv(args.output_dir / "val_features.csv", index=False)

    model = train_classical_species_model(
        train_features,
        detection_config,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    save_classical_species_model(model, args.output_dir / "model.pkl")

    evaluation = evaluate_classical_species_model(model, val_features)
    save_species_evaluation(evaluation, args.output_dir / "validation")

    metadata = {
        "model_family": "classical",
        "task": "species_classification",
        "dataset_dir": str(args.dataset_dir),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "detect_count_config": str(args.detect_count_config),
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "train_images": int(len(train_features)),
        "val_images": int(len(val_features)),
        "class_count": int(len(model.class_names)),
        "class_names": model.class_names,
        "validation_metrics": evaluation.metrics,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Classical species classifier validation metrics")
    for key, value in evaluation.metrics.items():
        print(f"{key}: {value}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
