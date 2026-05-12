from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.classical.species_classification import (
    SpeciesFeatureConfig,
    extract_species_features,
    fit_species_classifier,
    save_species_model,
)
from src.ui.inference import (
    CLASSICAL_COUNT_PRESETS,
    CLASSICAL_SPECIES_MODEL_PRESETS,
    classify_with_classical_species_model,
    config_rows,
    count_with_classical_model,
    decode_uploaded_image,
    evaluate_empty_plate_gate,
    find_local_species_model_candidates,
    inspect_species_model_path,
    inspect_uploaded_species_model,
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


def test_classify_with_classical_species_model_accepts_uploaded_model_bytes(tmp_path: Path) -> None:
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
    model_bytes = model_path.read_bytes()
    result = classify_with_classical_species_model(image, model_json_bytes=model_bytes)

    assert result.classifier_type == "centroid_l2"
    assert result.predicted_class in {"A", "B"}


def test_classify_with_classical_species_model_can_reject_empty_like_input(tmp_path: Path) -> None:
    config = SpeciesFeatureConfig()
    train_rows = []
    for index, bgr, label in [
        (0, (25, 40, 220), "A"),
        (1, (35, 55, 210), "A"),
        (2, (40, 60, 200), "A"),
        (3, (220, 60, 30), "B"),
        (4, (210, 70, 40), "B"),
        (5, (200, 80, 50), "B"),
    ]:
        image = np.full((96, 96, 3), 235, dtype=np.uint8)
        cv2.circle(image, (48, 48), 14 + index % 3, bgr, -1)
        features = extract_species_features(image, config)
        train_rows.append(
            {
                "image_path": f"{label}_{index}.png",
                "true_class": label,
                "category": "countable",
                **features,
            }
        )

    import pandas as pd

    model = fit_species_classifier(pd.DataFrame(train_rows), config, "knn_3")
    model_path = tmp_path / "model.json"
    save_species_model(model, model_path)

    empty_like = np.full((96, 96, 3), 240, dtype=np.uint8)
    result = classify_with_classical_species_model(
        empty_like,
        model_json_path=str(model_path),
        reject_empty_unsupported=True,
    )

    assert not result.accepted
    assert result.predicted_class == "Empty / unsupported"
    assert result.rejection_reason


def test_evaluate_empty_plate_gate_rejects_empty_like_plate() -> None:
    config = SpeciesFeatureConfig()
    empty_like = np.full((96, 96, 3), 240, dtype=np.uint8)
    features = extract_species_features(empty_like, config)

    gate = evaluate_empty_plate_gate(empty_like, features)

    assert gate.rejected
    assert gate.estimated_count == 0


def test_evaluate_empty_plate_gate_keeps_clear_colony_plate() -> None:
    config = SpeciesFeatureConfig()
    colony_like = np.full((128, 128, 3), 235, dtype=np.uint8)
    cv2.circle(colony_like, (44, 48), 12, (40, 50, 170), -1)
    cv2.circle(colony_like, (84, 76), 11, (45, 60, 165), -1)
    features = extract_species_features(colony_like, config)

    gate = evaluate_empty_plate_gate(colony_like, features)

    assert not gate.rejected


def test_evaluate_empty_plate_gate_keeps_dense_colony_plate() -> None:
    config = SpeciesFeatureConfig()
    dense_plate = np.full((160, 160, 3), 232, dtype=np.uint8)
    for y in range(28, 132, 18):
        for x in range(28, 132, 18):
            cv2.circle(dense_plate, (x, y), 7, (58, 78, 180), -1)
    features = extract_species_features(dense_plate, config)

    gate = evaluate_empty_plate_gate(dense_plate, features)

    assert not gate.rejected


def test_config_rows_are_displayable() -> None:
    rows = config_rows({"alpha": 1, "beta": True})

    assert rows == [{"Parameter": "alpha", "Value": "1"}, {"Parameter": "beta", "Value": "True"}]


def test_species_model_preset_points_to_latest_sweep_best_run() -> None:
    assert CLASSICAL_SPECIES_MODEL_PRESETS["Latest sweep best run"].endswith(
        "outputs/classical_species_image/val_sweep_001/best_run/model.json"
    )


def test_inspect_species_model_path_marks_colab_path_unavailable() -> None:
    status = inspect_species_model_path(
        "/content/drive/MyDrive/bacterial_colony_detection/outputs/classical_species_image/val_sweep_001/best_run/model.json"
    )

    assert not status.ready
    assert status.is_colab_path
    assert "local Streamlit app cannot read that location" in status.message


def test_inspect_species_model_path_marks_existing_local_model_ready(tmp_path: Path) -> None:
    model_path = tmp_path / "species_model.json"
    model_path.write_text("{}", encoding="utf-8")

    status = inspect_species_model_path(str(model_path), tmp_path)

    assert status.ready
    assert status.exists
    assert status.path == str(model_path)


def test_inspect_uploaded_species_model_marks_valid_upload_ready(tmp_path: Path) -> None:
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

    status = inspect_uploaded_species_model(model_path.read_bytes(), "model.json")

    assert status.ready
    assert status.exists
    assert status.path == "model.json"


def test_inspect_uploaded_species_model_marks_invalid_upload_not_ready() -> None:
    status = inspect_uploaded_species_model(b"{bad json", "broken.json")

    assert not status.ready
    assert "invalid" in status.message.lower()


def test_find_local_species_model_candidates_filters_model_paths(tmp_path: Path) -> None:
    species_model = tmp_path / "outputs" / "classical_species_image" / "best_run" / "model.json"
    generic_model = tmp_path / "outputs" / "other" / "model.json"
    species_model.parent.mkdir(parents=True)
    generic_model.parent.mkdir(parents=True)
    species_model.write_text("{}", encoding="utf-8")
    generic_model.write_text("{}", encoding="utf-8")

    candidates = find_local_species_model_candidates(tmp_path)

    assert candidates == (str(species_model),)
