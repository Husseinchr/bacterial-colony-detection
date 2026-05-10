from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.yolo.common import class_name_for_id, load_optional_class_names, read_yolo_label_file


@dataclass(frozen=True)
class YoloSpeciesEvaluation:
    predictions: pd.DataFrame
    class_distribution: pd.DataFrame
    report: pd.DataFrame
    confusion: pd.DataFrame
    metrics: dict[str, Any]


def evaluate_yolo_species_classification(
    split_csv: Path,
    prediction_label_dir: Path,
    class_names_path: Path | None,
    conf_threshold: float,
) -> YoloSpeciesEvaluation:
    split_df = pd.read_csv(split_csv)
    required = {"image_name", "label_name"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    class_names = load_optional_class_names(class_names_path)
    prediction_rows = []
    distribution_rows = []

    for _, row in split_df.iterrows():
        image_name = row["image_name"]
        labels = read_yolo_label_file(prediction_label_dir / f"{Path(image_name).stem}.txt", conf_threshold)
        true_label = row["label_name"]

        if labels.empty:
            pred_label = ""
            pred_count = 0
            mean_confidence = 0.0
        else:
            labels["class_name"] = labels["class_id"].map(lambda class_id: class_name_for_id(int(class_id), class_names))
            counts = labels["class_name"].value_counts()
            pred_label = str(counts.index[0])
            pred_count = int(counts.iloc[0])
            mean_confidence = float(labels.loc[labels["class_name"] == pred_label, "confidence"].mean())
            class_summary = (
                labels.groupby(["class_id", "class_name"], as_index=False)
                .agg(count=("confidence", "size"), mean_confidence=("confidence", "mean"))
                .sort_values(["count", "mean_confidence"], ascending=[False, False])
            )
            for _, class_row in class_summary.iterrows():
                distribution_rows.append(
                    {
                        "image_name": image_name,
                        "class_id": int(class_row["class_id"]),
                        "class_name": class_row["class_name"],
                        "count": int(class_row["count"]),
                        "mean_confidence": float(class_row["mean_confidence"]),
                    }
                )

        prediction_rows.append(
            {
                "image_name": image_name,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_label_count": pred_count,
                "pred_label_mean_confidence": mean_confidence,
                "correct": pred_label == true_label,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    class_distribution = pd.DataFrame(distribution_rows)
    report, confusion, metrics = classification_outputs(predictions)
    metrics.update(
        {
            "model_family": "yolo",
            "task": "species_classification",
            "split_csv": str(split_csv),
            "prediction_label_dir": str(prediction_label_dir),
            "class_names": "" if class_names_path is None else str(class_names_path),
            "conf_threshold": float(conf_threshold),
        }
    )
    return YoloSpeciesEvaluation(
        predictions=predictions,
        class_distribution=class_distribution,
        report=report,
        confusion=confusion,
        metrics=metrics,
    )


def classification_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels = sorted(set(predictions["true_label"]).union(set(predictions["pred_label"])))
    labels = [label for label in labels if label != ""]
    rows = []

    for label in labels:
        true_positive = int(((predictions["true_label"] == label) & (predictions["pred_label"] == label)).sum())
        false_positive = int(((predictions["true_label"] != label) & (predictions["pred_label"] == label)).sum())
        false_negative = int(((predictions["true_label"] == label) & (predictions["pred_label"] != label)).sum())
        support = int((predictions["true_label"] == label).sum())
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )

    report = pd.DataFrame(rows)
    accuracy = float(predictions["correct"].mean()) if len(predictions) else 0.0
    macro_f1 = float(report["f1_score"].mean()) if not report.empty else 0.0
    balanced_accuracy = float(report["recall"].mean()) if not report.empty else 0.0
    weighted_f1 = safe_div(float((report["f1_score"] * report["support"]).sum()), float(report["support"].sum())) if not report.empty else 0.0

    confusion = pd.crosstab(
        pd.Categorical(predictions["true_label"], categories=labels),
        pd.Categorical(predictions["pred_label"], categories=labels),
        rownames=["true_label"],
        colnames=["pred_label"],
        dropna=False,
    )
    metrics = {
        "images": int(len(predictions)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    return report, confusion, metrics


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)
