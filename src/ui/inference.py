from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.classical.agar_evaluation import load_count_config
from src.classical.detect_count import ClassicalCountConfig, count_colonies, read_image, render_count_overlay
from src.classical.species_classification import extract_species_features, load_species_model


@dataclass(frozen=True)
class CountingInferenceResult:
    predicted_count: int
    overlay_bgr: np.ndarray
    mask: np.ndarray
    config: dict[str, Any]


@dataclass(frozen=True)
class ClassificationInferenceResult:
    predicted_class: str
    classifier_type: str
    distance: float
    config: dict[str, Any]


CLASSICAL_COUNT_PRESETS = {
    "Config 23 baseline": ClassicalCountConfig(
        threshold_method="otsu",
        threshold_polarity="dark",
        background_kernel=101,
        min_area=500,
        max_area=10000.0,
        min_circularity=0.15,
        morph_open_kernel=5,
        morph_close_kernel=3,
    ),
    "Conservative dense alternative": ClassicalCountConfig(
        threshold_method="otsu",
        threshold_polarity="dark",
        background_kernel=101,
        min_area=500,
        max_area=50000.0,
        min_circularity=0.15,
        morph_open_kernel=5,
        morph_close_kernel=3,
        use_peak_count_estimation=True,
        large_component_min_area=6000.0,
        peak_blur_kernel=5,
        peak_local_max_kernel=7,
        peak_relative_threshold=0.75,
        peak_min_distance=3.5,
        use_area_count_estimation=True,
        area_count_scale=2.8,
    ),
}


CLASSICAL_SPECIES_MODEL_PRESETS = {
    "Latest sweep best run": "/content/drive/MyDrive/bacterial_colony_detection/outputs/classical_species_image/val_sweep_001/best_run/model.json",
}


def load_image_input(image_path: str = "", uploaded_bytes: bytes | None = None) -> tuple[np.ndarray, str]:
    if uploaded_bytes:
        return decode_uploaded_image(uploaded_bytes), "uploaded image"
    if image_path.strip():
        return read_image(Path(image_path.strip())), image_path.strip()
    raise ValueError("Provide either an image path or an uploaded image")


def decode_uploaded_image(uploaded_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")
    return image


def count_with_classical_model(
    image: np.ndarray,
    preset_name: str = "Config 23 baseline",
    config_json_path: str = "",
) -> CountingInferenceResult:
    config = load_count_config(Path(config_json_path.strip())) if config_json_path.strip() else preset_count_config(preset_name)
    predicted_count, mask, contours = count_colonies(image, config)
    overlay = render_count_overlay(image, predicted_count, contours)
    return CountingInferenceResult(
        predicted_count=int(predicted_count),
        overlay_bgr=overlay,
        mask=mask,
        config=config.to_dict(),
    )


def classify_with_classical_species_model(
    image: np.ndarray,
    model_json_path: str,
) -> ClassificationInferenceResult:
    if not model_json_path.strip():
        raise ValueError("Provide a species model JSON path")
    model = load_species_model(Path(model_json_path.strip()))
    features = extract_species_features(image, model.config)
    predicted_class, distance = model.predict_features(features)
    return ClassificationInferenceResult(
        predicted_class=predicted_class,
        classifier_type=model.classifier_type,
        distance=float(distance),
        config=model.config.to_dict(),
    )


def preset_count_config(preset_name: str) -> ClassicalCountConfig:
    if preset_name not in CLASSICAL_COUNT_PRESETS:
        raise ValueError(f"Unknown counting preset: {preset_name}")
    return CLASSICAL_COUNT_PRESETS[preset_name]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def config_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    return [{"Parameter": key, "Value": str(value)} for key, value in config.items()]
