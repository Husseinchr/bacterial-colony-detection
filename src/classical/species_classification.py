from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.classical.detect_count import run_classical_detect_count
from src.preprocessing import load_image, to_grayscale


@dataclass(frozen=True)
class ClassicalSpeciesModel:
    classifier: Any
    feature_columns: list[str]
    class_names: list[str]
    detection_config: dict[str, Any]


@dataclass(frozen=True)
class ClassicalSpeciesEvaluation:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    report: pd.DataFrame
    confusion: pd.DataFrame


def extract_species_features(image_bgr: np.ndarray, detection_config: dict[str, Any]) -> dict[str, float]:
    gray = to_grayscale(image_bgr)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    features: dict[str, float] = {}

    for index, name in enumerate(["b", "g", "r"]):
        features.update(channel_stats(f"color_{name}", image_bgr[:, :, index]))

    for index, name in enumerate(["h", "s", "v"]):
        features.update(channel_stats(f"hsv_{name}", hsv[:, :, index]))

    features.update(channel_stats("gray", gray))
    features.update(histogram_features("gray_hist", gray, bins=16, value_range=(0, 256)))
    features.update(histogram_features("hue_hist", hsv[:, :, 0], bins=18, value_range=(0, 180)))

    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = float(np.count_nonzero(edges) / edges.size)

    result = run_classical_detect_count(image_bgr, detection_config)
    colony_features = result.features
    features["classical_colony_count"] = float(result.summary["colony_count"])
    features["classical_density_per_100k"] = float(result.summary["count"]["density_per_100k_pixels"])
    features.update(frame_stats("colony_area", colony_features, "area"))
    features.update(frame_stats("colony_circularity", colony_features, "circularity"))
    features.update(frame_stats("colony_equivalent_diameter", colony_features, "equivalent_diameter"))
    features.update(frame_stats("colony_eccentricity", colony_features, "eccentricity"))
    return features


def channel_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    flat = values.astype(np.float32).ravel()
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_p10": float(np.percentile(flat, 10)),
        f"{prefix}_p50": float(np.percentile(flat, 50)),
        f"{prefix}_p90": float(np.percentile(flat, 90)),
    }


def histogram_features(prefix: str, values: np.ndarray, bins: int, value_range: tuple[int, int]) -> dict[str, float]:
    hist, _ = np.histogram(values.ravel(), bins=bins, range=value_range)
    total = hist.sum()
    normalized = hist / total if total else hist
    return {f"{prefix}_{index:02d}": float(value) for index, value in enumerate(normalized)}


def frame_stats(prefix: str, frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame.empty or column not in frame:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
        }
    values = frame[column].astype(float)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_median": float(values.median()),
    }


def extract_species_feature_table(
    dataset_dir: Path,
    split_csv: Path,
    detection_config: dict[str, Any],
    limit: int | None = None,
) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    required = {"image_name", "label_name"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    if limit is not None:
        split_df = split_df.head(limit)

    rows = []
    for _, row in split_df.iterrows():
        image_name = row["image_name"]
        image = load_image(dataset_dir / image_name)
        feature_row = extract_species_features(image, detection_config)
        feature_row["image_name"] = image_name
        feature_row["label_name"] = row["label_name"]
        rows.append(feature_row)

    return pd.DataFrame(rows)


def train_classical_species_model(
    train_features: pd.DataFrame,
    detection_config: dict[str, Any],
    random_state: int = 42,
    n_estimators: int = 400,
) -> ClassicalSpeciesModel:
    feature_columns = sorted(column for column in train_features.columns if column not in {"image_name", "label_name"})
    class_names = sorted(train_features["label_name"].unique().tolist())
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(train_features[feature_columns], train_features["label_name"])
    return ClassicalSpeciesModel(
        classifier=classifier,
        feature_columns=feature_columns,
        class_names=class_names,
        detection_config=detection_config,
    )


def evaluate_classical_species_model(model: ClassicalSpeciesModel, features: pd.DataFrame) -> ClassicalSpeciesEvaluation:
    y_true = features["label_name"]
    y_pred = model.classifier.predict(features[model.feature_columns])
    predictions = pd.DataFrame(
        {
            "image_name": features["image_name"],
            "true_label": y_true,
            "pred_label": y_pred,
            "correct": y_true.to_numpy() == y_pred,
        }
    )
    metrics = {
        "images": int(len(features)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).transpose()
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=model.class_names),
        index=model.class_names,
        columns=model.class_names,
    )
    return ClassicalSpeciesEvaluation(metrics=metrics, predictions=predictions, report=report, confusion=confusion)


def save_classical_species_model(model: ClassicalSpeciesModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
            "classifier": model.classifier,
            "feature_columns": model.feature_columns,
            "class_names": model.class_names,
            "detection_config": model.detection_config,
            },
            f,
        )


def load_classical_species_model(path: Path) -> ClassicalSpeciesModel:
    with path.open("rb") as f:
        data = pickle.load(f)
    return ClassicalSpeciesModel(
        classifier=data["classifier"],
        feature_columns=data["feature_columns"],
        class_names=data["class_names"],
        detection_config=data["detection_config"],
    )


def save_species_evaluation(evaluation: ClassicalSpeciesEvaluation, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.predictions.to_csv(output_dir / "predictions.csv", index=False)
    evaluation.report.to_csv(output_dir / "classification_report.csv")
    evaluation.confusion.to_csv(output_dir / "confusion_matrix.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.metrics, f, indent=2)
