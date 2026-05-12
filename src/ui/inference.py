from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.classical.agar_evaluation import load_count_config
from src.classical.detect_count import ClassicalCountConfig, count_colonies, read_image, render_count_overlay
from src.classical.species_classification import extract_species_features, load_species_model, load_species_model_bytes


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
    accepted: bool
    rejection_reason: str
    rejection_distance_threshold: float | None


@dataclass(frozen=True)
class EmptyPlateGateResult:
    rejected: bool
    reason: str
    estimated_count: int
    foreground_ratio: float
    center_saturated_fraction: float
    center_s_std: float
    center_lab_a_std: float
    center_lab_b_std: float
    center_edge_density: float


@dataclass(frozen=True)
class SpeciesModelPathStatus:
    path: str
    exists: bool
    ready: bool
    is_colab_path: bool
    message: str
    local_candidates: tuple[str, ...]


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


EMPTY_PLATE_GATE_CONFIG = ClassicalCountConfig(
    threshold_method="otsu",
    threshold_polarity="dark",
    background_kernel=101,
    min_area=180.0,
    max_area=9000.0,
    min_circularity=0.62,
    morph_open_kernel=5,
    morph_close_kernel=3,
    use_plate_mask=True,
    plate_margin_ratio=0.08,
)


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
    model_json_path: str = "",
    model_json_bytes: bytes | None = None,
    reject_empty_unsupported: bool = False,
) -> ClassificationInferenceResult:
    if model_json_bytes is not None:
        upload_status = inspect_uploaded_species_model(model_json_bytes)
        if not upload_status.ready:
            raise ValueError(upload_status.message)
        model = load_species_model_bytes(model_json_bytes)
    else:
        status = inspect_species_model_path(model_json_path)
        if not status.ready:
            raise FileNotFoundError(status.message)
        model = load_species_model(Path(status.path))
    features = extract_species_features(image, model.config)
    predicted_class, distance = model.predict_features(features)
    accepted = True
    rejection_reason = ""
    if reject_empty_unsupported:
        empty_gate = evaluate_empty_plate_gate(image, features)
        if empty_gate.rejected:
            accepted = False
            predicted_class = "Empty / unsupported"
            rejection_reason = empty_gate.reason
    return ClassificationInferenceResult(
        predicted_class=predicted_class,
        classifier_type=model.classifier_type,
        distance=float(distance),
        config=model.config.to_dict(),
        accepted=accepted,
        rejection_reason=rejection_reason,
        rejection_distance_threshold=None,
    )


def preset_count_config(preset_name: str) -> ClassicalCountConfig:
    if preset_name not in CLASSICAL_COUNT_PRESETS:
        raise ValueError(f"Unknown counting preset: {preset_name}")
    return CLASSICAL_COUNT_PRESETS[preset_name]


def inspect_species_model_path(model_json_path: str, search_root: Path | None = None) -> SpeciesModelPathStatus:
    raw_path = model_json_path.strip()
    if not raw_path:
        return SpeciesModelPathStatus(
            path="",
            exists=False,
            ready=False,
            is_colab_path=False,
            message="Provide a local species model JSON path.",
            local_candidates=find_local_species_model_candidates(search_root),
        )

    path = Path(raw_path)
    exists = path.exists()
    is_colab = raw_path.startswith("/content/")
    local_candidates = find_local_species_model_candidates(search_root)
    if exists:
        return SpeciesModelPathStatus(
            path=str(path),
            exists=True,
            ready=True,
            is_colab_path=is_colab,
            message=f"Local model is ready: {path}",
            local_candidates=local_candidates,
        )
    if is_colab:
        return SpeciesModelPathStatus(
            path=str(path),
            exists=False,
            ready=False,
            is_colab_path=True,
            message=(
                "This preset points to a Colab or Google Drive path under /content/. "
                "The local Streamlit app cannot read that location. Use a local model.json path."
            ),
            local_candidates=local_candidates,
        )
    return SpeciesModelPathStatus(
        path=str(path),
        exists=False,
        ready=False,
        is_colab_path=False,
        message=f"Model file not found: {path}",
        local_candidates=local_candidates,
    )


def inspect_uploaded_species_model(model_json_bytes: bytes | None, uploaded_name: str = "") -> SpeciesModelPathStatus:
    display_name = uploaded_name or "uploaded model.json"
    if model_json_bytes is None:
        return SpeciesModelPathStatus(
            path=display_name,
            exists=False,
            ready=False,
            is_colab_path=False,
            message="Upload a species model JSON or provide a local path.",
            local_candidates=(),
        )
    try:
        load_species_model_bytes(model_json_bytes)
    except Exception as exc:
        return SpeciesModelPathStatus(
            path=display_name,
            exists=False,
            ready=False,
            is_colab_path=False,
            message=f"Uploaded species model is invalid: {exc}",
            local_candidates=(),
        )
    return SpeciesModelPathStatus(
        path=display_name,
        exists=True,
        ready=True,
        is_colab_path=False,
        message=f"Uploaded model is ready: {display_name}",
        local_candidates=(),
    )


def find_local_species_model_candidates(search_root: Path | None = None) -> tuple[str, ...]:
    if search_root is None or not search_root.exists():
        return ()
    candidates = []
    for path in search_root.rglob("model.json"):
        try:
            relative_parts = [part.lower() for part in path.relative_to(search_root).parts]
        except ValueError:
            relative_parts = [part.lower() for part in path.parts]
        if not any("species" in part for part in relative_parts[:-1]):
            continue
        candidates.append(str(path))
    return tuple(sorted(candidates)[:8])


def evaluate_empty_plate_gate(
    image: np.ndarray,
    features: dict[str, float],
    config: ClassicalCountConfig | None = None,
) -> EmptyPlateGateResult:
    gate_config = config or EMPTY_PLATE_GATE_CONFIG
    estimated_count, mask, _ = count_colonies(image, gate_config)
    foreground_ratio = float((mask > 0).mean())
    center_saturated_fraction = float(features.get("feature_center_saturated_fraction", 0.0))
    center_s_std = float(features.get("feature_center_s_std", 0.0))
    center_lab_a_std = float(features.get("feature_center_lab_a_std", 0.0))
    center_lab_b_std = float(features.get("feature_center_lab_b_std", 0.0))
    center_edge_density = float(features.get("feature_center_edge_density", 0.0))
    rejected = (
        estimated_count <= 1
        and foreground_ratio < 0.0024
        and center_saturated_fraction < 0.03
        and center_s_std < 14.0
        and center_lab_a_std < 4.0
        and center_lab_b_std < 4.0
        and center_edge_density < 0.03
    )
    reason = ""
    if rejected:
        reason = "The image has no colony-like evidence inside the plate and is treated as empty or unsupported."
    return EmptyPlateGateResult(
        rejected=rejected,
        reason=reason,
        estimated_count=int(estimated_count),
        foreground_ratio=foreground_ratio,
        center_saturated_fraction=center_saturated_fraction,
        center_s_std=center_s_std,
        center_lab_a_std=center_lab_a_std,
        center_lab_b_std=center_lab_b_std,
        center_edge_density=center_edge_density,
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def config_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    return [{"Parameter": key, "Value": str(value)} for key, value in config.items()]
