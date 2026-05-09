from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classical import evaluate_classical_detect_count


def test_evaluate_classical_detect_count_writes_expected_summary_inputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image = np.full((80, 80, 3), 20, dtype=np.uint8)
    image[25:40, 25:40] = 240
    cv2.imwrite(str(dataset_dir / "image_a.jpg"), image)

    split_csv = tmp_path / "split.csv"
    pd.DataFrame([{"image_name": "image_a.jpg", "colony_count": 1}]).to_csv(split_csv, index=False)

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

    evaluation = evaluate_classical_detect_count(dataset_dir, split_csv, config)

    assert len(evaluation.per_image) == 1
    assert evaluation.summary["model_family"] == "classical"
    assert evaluation.summary["task"] == "detect_count"
    assert evaluation.summary["images"] == 1
    assert evaluation.summary["true_colonies"] == 1
    assert "mae" in evaluation.summary
    assert "rmse" in evaluation.summary
