from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.classical.detect_count import run_classical_detect_count
from src.evaluation import count_metrics
from src.preprocessing import load_image


@dataclass(frozen=True)
class ClassicalCountEvaluation:
    per_image: pd.DataFrame
    summary: dict[str, Any]


def evaluate_classical_detect_count(
    dataset_dir: Path,
    split_csv: Path,
    config: dict[str, Any],
    grid_rows: int = 4,
    grid_cols: int = 4,
    limit: int | None = None,
) -> ClassicalCountEvaluation:
    split_df = pd.read_csv(split_csv)
    required = {"image_name", "colony_count"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    if limit is not None:
        split_df = split_df.head(limit)

    rows = []
    for _, row in split_df.iterrows():
        image_name = row["image_name"]
        image = load_image(dataset_dir / image_name)
        result = run_classical_detect_count(image, config, grid_rows=grid_rows, grid_cols=grid_cols)
        true_count = int(row["colony_count"])
        pred_count = int(result.summary["colony_count"])
        rows.append(
            {
                "image_name": image_name,
                "true_count": true_count,
                "pred_count": pred_count,
                "error": pred_count - true_count,
                "absolute_error": abs(pred_count - true_count),
                "growth_level": result.summary["growth_level"],
            }
        )

    per_image = pd.DataFrame(rows)
    metrics = count_metrics(per_image["pred_count"].tolist(), per_image["true_count"].tolist())
    summary = {
        "model_family": "classical",
        "task": "detect_count",
        "dataset_dir": str(dataset_dir),
        "split_csv": str(split_csv),
        "images": int(len(per_image)),
        "true_colonies": int(per_image["true_count"].sum()),
        "pred_colonies": int(per_image["pred_count"].sum()),
        **asdict(metrics),
    }
    return ClassicalCountEvaluation(per_image=per_image, summary=summary)
