from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.classical.detect_count import read_image


@dataclass(frozen=True)
class SpeciesFeatureConfig:
    image_size: int = 256
    hue_bins: int = 16
    saturation_bins: int = 8
    value_bins: int = 8
    gray_bins: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeciesPrediction:
    image_path: str
    true_class: str
    predicted_class: str
    distance: float
    correct: bool
    category: str


@dataclass(frozen=True)
class SpeciesEvaluation:
    metrics: dict[str, Any]
    predictions: list[SpeciesPrediction]


class NearestCentroidSpeciesClassifier:
    def __init__(
        self,
        classes: list[str],
        feature_names: list[str],
        mean: list[float],
        scale: list[float],
        centroids: dict[str, list[float]],
        config: SpeciesFeatureConfig,
    ) -> None:
        self.classes = classes
        self.feature_names = feature_names
        self.mean = np.array(mean, dtype=np.float64)
        self.scale = np.array(scale, dtype=np.float64)
        self.centroids = {label: np.array(values, dtype=np.float64) for label, values in centroids.items()}
        self.config = config

    def predict_features(self, features: dict[str, float]) -> tuple[str, float]:
        vector = vectorize_features(features, self.feature_names)
        normalized = (vector - self.mean) / self.scale
        distances = {
            label: float(np.linalg.norm(normalized - centroid))
            for label, centroid in self.centroids.items()
        }
        predicted = min(distances, key=distances.get)
        return predicted, distances[predicted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "nearest_centroid_classical_species",
            "classes": self.classes,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "centroids": {label: centroid.tolist() for label, centroid in self.centroids.items()},
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NearestCentroidSpeciesClassifier":
        return cls(
            classes=[str(label) for label in data["classes"]],
            feature_names=[str(name) for name in data["feature_names"]],
            mean=[float(value) for value in data["mean"]],
            scale=[float(value) for value in data["scale"]],
            centroids={str(label): [float(value) for value in values] for label, values in data["centroids"].items()},
            config=SpeciesFeatureConfig(**data.get("config", {})),
        )


def train_species_classifier(
    dataset_dir: Path,
    train_csv: Path,
    config: SpeciesFeatureConfig | None = None,
    max_images: int = 0,
) -> tuple[NearestCentroidSpeciesClassifier, pd.DataFrame]:
    active_config = config or SpeciesFeatureConfig()
    rows = load_species_split(train_csv, max_images=max_images)
    feature_rows = extract_feature_rows(dataset_dir, rows, active_config)
    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        raise ValueError(f"No species-image rows found in {train_csv}")
    labels = sorted(feature_frame["true_class"].unique().tolist())
    feature_names = [column for column in feature_frame.columns if column.startswith("feature_")]
    matrix = feature_frame[feature_names].to_numpy(dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (matrix - mean) / scale
    centroids = {}
    label_array = feature_frame["true_class"].to_numpy()
    for label in labels:
        centroids[label] = normalized[label_array == label].mean(axis=0).tolist()
    model = NearestCentroidSpeciesClassifier(
        classes=labels,
        feature_names=feature_names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        centroids=centroids,
        config=active_config,
    )
    return model, feature_frame


def evaluate_species_classifier(
    dataset_dir: Path,
    split_csv: Path,
    model: NearestCentroidSpeciesClassifier,
    max_images: int = 0,
) -> SpeciesEvaluation:
    rows = load_species_split(split_csv, max_images=max_images)
    predictions = []
    for row in rows.to_dict("records"):
        image_path = dataset_dir / str(row["image_path"])
        image = read_image(image_path)
        features = extract_species_features(image, model.config)
        predicted_class, distance = model.predict_features(features)
        true_class = str(row["primary_class"])
        predictions.append(
            SpeciesPrediction(
                image_path=str(row["image_path"]),
                true_class=true_class,
                predicted_class=predicted_class,
                distance=float(distance),
                correct=predicted_class == true_class,
                category=str(row.get("category", "")),
            )
        )
    return SpeciesEvaluation(metrics=classification_metrics(predictions, model.classes), predictions=predictions)


def extract_species_features(image: np.ndarray, config: SpeciesFeatureConfig | None = None) -> dict[str, float]:
    active_config = config or SpeciesFeatureConfig()
    resized = resize_square(image, active_config.image_size)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = {}
    append_channel_stats(features, "b", resized[:, :, 0])
    append_channel_stats(features, "g", resized[:, :, 1])
    append_channel_stats(features, "r", resized[:, :, 2])
    append_channel_stats(features, "h", hsv[:, :, 0])
    append_channel_stats(features, "s", hsv[:, :, 1])
    append_channel_stats(features, "v", hsv[:, :, 2])
    append_channel_stats(features, "lab_l", lab[:, :, 0])
    append_channel_stats(features, "lab_a", lab[:, :, 1])
    append_channel_stats(features, "lab_b", lab[:, :, 2])
    append_histogram(features, "hue", hsv[:, :, 0], active_config.hue_bins, 0, 180)
    append_histogram(features, "saturation", hsv[:, :, 1], active_config.saturation_bins, 0, 256)
    append_histogram(features, "value", hsv[:, :, 2], active_config.value_bins, 0, 256)
    append_histogram(features, "gray", gray, active_config.gray_bins, 0, 256)
    append_texture_features(features, gray, hsv)
    return {f"feature_{key}": float(value) for key, value in features.items()}


def extract_feature_rows(dataset_dir: Path, rows: pd.DataFrame, config: SpeciesFeatureConfig) -> list[dict[str, Any]]:
    feature_rows = []
    for row in rows.to_dict("records"):
        image_path = dataset_dir / str(row["image_path"])
        image = read_image(image_path)
        features = extract_species_features(image, config)
        feature_rows.append(
            {
                "image_path": str(row["image_path"]),
                "true_class": str(row["primary_class"]),
                "category": str(row.get("category", "")),
                **features,
            }
        )
    return feature_rows


def load_species_split(split_csv: Path, max_images: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    if "species_image_eligible" in frame.columns:
        frame = frame[frame["species_image_eligible"].map(is_truthy)].copy()
    frame = frame[frame["primary_class"].fillna("").astype(str) != ""].copy()
    if max_images > 0:
        frame = frame.head(max_images).copy()
    return frame


def classification_metrics(predictions: list[SpeciesPrediction], classes: list[str]) -> dict[str, Any]:
    if not predictions:
        return {"image_count": 0}
    total = len(predictions)
    correct = sum(1 for prediction in predictions if prediction.correct)
    per_class = {}
    recalls = []
    precisions = []
    f1_scores = []
    for label in classes:
        true_positive = sum(
            1 for prediction in predictions if prediction.true_class == label and prediction.predicted_class == label
        )
        false_positive = sum(
            1 for prediction in predictions if prediction.true_class != label and prediction.predicted_class == label
        )
        false_negative = sum(
            1 for prediction in predictions if prediction.true_class == label and prediction.predicted_class != label
        )
        support = true_positive + false_negative
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if support == 0 else true_positive / support
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        if support > 0:
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)
    return {
        "image_count": int(total),
        "accuracy": float(correct / total),
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(predictions, classes),
    }


def confusion_matrix(predictions: list[SpeciesPrediction], classes: list[str]) -> dict[str, dict[str, int]]:
    matrix = {true_label: {predicted_label: 0 for predicted_label in classes} for true_label in classes}
    for prediction in predictions:
        if prediction.true_class in matrix and prediction.predicted_class in matrix[prediction.true_class]:
            matrix[prediction.true_class][prediction.predicted_class] += 1
    return matrix


def save_species_model(model: NearestCentroidSpeciesClassifier, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)


def load_species_model(path: Path) -> NearestCentroidSpeciesClassifier:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return NearestCentroidSpeciesClassifier.from_dict(data)


def save_species_evaluation(evaluation: SpeciesEvaluation, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(prediction) for prediction in evaluation.predictions]).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.metrics, f, indent=2)


def resize_square(image: np.ndarray, image_size: int) -> np.ndarray:
    size = max(16, int(image_size))
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def append_channel_stats(features: dict[str, float], prefix: str, channel: np.ndarray) -> None:
    values = channel.astype(np.float32)
    features[f"{prefix}_mean"] = float(values.mean())
    features[f"{prefix}_std"] = float(values.std())
    features[f"{prefix}_p10"] = float(np.percentile(values, 10))
    features[f"{prefix}_p50"] = float(np.percentile(values, 50))
    features[f"{prefix}_p90"] = float(np.percentile(values, 90))


def append_histogram(
    features: dict[str, float],
    prefix: str,
    channel: np.ndarray,
    bins: int,
    minimum: float,
    maximum: float,
) -> None:
    hist, _ = np.histogram(channel, bins=max(1, int(bins)), range=(minimum, maximum))
    total = max(1, int(hist.sum()))
    for index, value in enumerate(hist.astype(np.float64) / total):
        features[f"{prefix}_hist_{index:02d}"] = float(value)


def append_texture_features(features: dict[str, float], gray: np.ndarray, hsv: np.ndarray) -> None:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edges = cv2.Canny(gray, 60, 160)
    features["laplacian_var"] = float(laplacian.var())
    features["sobel_mean"] = float(magnitude.mean())
    features["sobel_std"] = float(magnitude.std())
    features["edge_density"] = float((edges > 0).mean())
    features["dark_fraction"] = float((gray < 80).mean())
    features["bright_fraction"] = float((gray > 200).mean())
    features["saturated_fraction"] = float((hsv[:, :, 1] > 80).mean())


def vectorize_features(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.array([float(features[name]) for name in feature_names], dtype=np.float64)


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
