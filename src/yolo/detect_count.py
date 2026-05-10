from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import count_metrics
from src.yolo.common import read_yolo_label_file


@dataclass(frozen=True)
class YoloCountEvaluation:
    per_image: pd.DataFrame
    summary: dict[str, Any]


def evaluate_yolo_detect_count(
    split_csv: Path,
    prediction_label_dir: Path,
    conf_threshold: float,
) -> YoloCountEvaluation:
    split_df = pd.read_csv(split_csv)
    required = {"image_name", "colony_count"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    rows = []
    for _, row in split_df.iterrows():
        image_name = row["image_name"]
        labels = read_yolo_label_file(prediction_label_dir / f"{Path(image_name).stem}.txt", conf_threshold)
        true_count = int(row["colony_count"])
        pred_count = int(len(labels))
        rows.append(
            {
                "image_name": image_name,
                "true_count": true_count,
                "pred_count": pred_count,
                "error": pred_count - true_count,
                "absolute_error": abs(pred_count - true_count),
            }
        )

    per_image = pd.DataFrame(rows)
    metrics = count_metrics(per_image["pred_count"].tolist(), per_image["true_count"].tolist())
    summary = {
        "model_family": "yolo",
        "task": "detect_count",
        "split_csv": str(split_csv),
        "prediction_label_dir": str(prediction_label_dir),
        "conf_threshold": float(conf_threshold),
        "images": int(len(per_image)),
        "true_colonies": int(per_image["true_count"].sum()),
        "pred_colonies": int(per_image["pred_count"].sum()),
        **asdict(metrics),
    }
    return YoloCountEvaluation(per_image=per_image, summary=summary)
