from __future__ import annotations

import numpy as np

from src.classical import run_classical_detect_count


def test_classical_detect_count_returns_expected_artifacts() -> None:
    image = np.full((120, 160, 3), 20, dtype=np.uint8)
    image[45:65, 55:75] = 240
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

    result = run_classical_detect_count(image, config, grid_rows=2, grid_cols=2)

    assert result.summary["model_family"] == "classical"
    assert result.summary["task"] == "detect_count"
    assert result.mask.shape == (120, 160)
    assert result.labels.shape == (120, 160)
    assert result.overlay.shape == image.shape
    assert len(result.density_grid) == 4
    assert "colony_count" in result.summary
    assert "growth_level" in result.summary
