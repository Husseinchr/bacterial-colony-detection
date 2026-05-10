from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.classical.species_classification import (
    evaluate_classical_species_model,
    extract_species_features,
    train_classical_species_model,
)


def test_extract_species_features_returns_numeric_schema() -> None:
    image = np.full((80, 80, 3), 30, dtype=np.uint8)
    image[20:40, 20:40] = (80, 120, 180)
    config = {
        "pipeline": {
            "threshold_method": "otsu",
            "split_touching": False,
        },
        "preprocessing": {
            "gaussian_kernel_size": 3,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": 8,
            "background_kernel_size": 11,
        },
        "segmentation": {
            "invert_threshold": False,
            "adaptive_block_size": 51,
            "adaptive_c": 2,
            "morph_kernel_size": 3,
            "opening_iterations": 0,
            "closing_iterations": 0,
            "min_area": 20,
            "max_area": 1000,
            "min_circularity": 0.0,
        },
        "outputs": {
            "overlay_alpha": 0.45,
        },
    }

    features = extract_species_features(image, config)

    assert "color_b_mean" in features
    assert "hsv_h_mean" in features
    assert "gray_hist_00" in features
    assert "classical_colony_count" in features
    assert all(np.isfinite(value) for value in features.values())


def test_train_and_evaluate_classical_species_model() -> None:
    features = pd.DataFrame(
        [
            {"image_name": "a.jpg", "label_name": "sp01", "f1": 0.0, "f2": 1.0},
            {"image_name": "b.jpg", "label_name": "sp01", "f1": 0.1, "f2": 1.1},
            {"image_name": "c.jpg", "label_name": "sp02", "f1": 5.0, "f2": 6.0},
            {"image_name": "d.jpg", "label_name": "sp02", "f1": 5.1, "f2": 6.1},
        ]
    )

    model = train_classical_species_model(features, detection_config={}, random_state=0, n_estimators=20)
    evaluation = evaluate_classical_species_model(model, features)

    assert model.class_names == ["sp01", "sp02"]
    assert evaluation.metrics["images"] == 4
    assert "accuracy" in evaluation.metrics
    assert set(evaluation.predictions.columns) == {"image_name", "true_label", "pred_label", "correct"}
    assert evaluation.confusion.shape == (2, 2)
