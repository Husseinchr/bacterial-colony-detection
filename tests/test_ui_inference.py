from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.classical.species_classification import SpeciesFeatureConfig, fit_species_classifier, save_species_model
from src.ui.inference import (
    CLASSICAL_COUNT_PRESETS,
    CLASSICAL_SPECIES_MODEL_PRESETS,
    classify_with_classical_species_model,
    config_rows,
    count_with_classical_model,
    decode_uploaded_image,
)


def test_decode_uploaded_image_roundtrip() -> None:
    image = np.full((32, 32, 3), (20, 80, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)

    assert ok
    decoded = decode_uploaded_image(encoded.tobytes())
    assert decoded.shape == image.shape


def test_count_with_classical_model_uses_presets() -> None:
    image = np.full((220, 260, 3), 230, dtype=np.uint8)
    for center in [(60, 70), (130, 80), (190, 150)]:
        cv2.circle(image, center, 14, (45, 45, 45), -1)

    result = count_with_classical_model(image, preset_name="Config 23 baseline")

    assert "min_area" in result.config
    assert result.config["min_area"] == CLASSICAL_COUNT_PRESETS["Config 23 baseline"].min_area
    assert result.overlay_bgr.shape == image.shape


def test_classify_with_classical_species_model_predicts_saved_model(tmp_path: Path) -> None:
    rows = [
        {
            "image_path": "a.png",
            "true_class": "A",
            "category": "countable",
            "feature_global_b_mean": 1.0,
            "feature_global_hue_hist_00": 0.8,
        },
        {
            "image_path": "b.png",
            "true_class": "B",
            "category": "countable",
            "feature_global_b_mean": 9.0,
            "feature_global_hue_hist_00": 0.2,
        },
    ]
    import pandas as pd

    feature_frame = pd.DataFrame(rows)
    config = SpeciesFeatureConfig()
    model = fit_species_classifier(feature_frame, config, "centroid_l2")
    model_path = tmp_path / "model.json"
    save_species_model(model, model_path)

    image = np.full((64, 64, 3), (1, 1, 240), dtype=np.uint8)
    result = classify_with_classical_species_model(image, str(model_path))

    assert result.classifier_type == "centroid_l2"
    assert result.predicted_class in {"A", "B"}


def test_config_rows_are_displayable() -> None:
    rows = config_rows({"alpha": 1, "beta": True})

    assert rows == [{"Parameter": "alpha", "Value": "1"}, {"Parameter": "beta", "Value": "True"}]


def test_species_model_preset_points_to_latest_sweep_best_run() -> None:
    assert CLASSICAL_SPECIES_MODEL_PRESETS["Latest sweep best run"].endswith(
        "outputs/classical_species_image/val_sweep_001/best_run/model.json"
    )
