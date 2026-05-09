from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import build_density_grid, classify_growth
from src.counting import summarize_count
from src.features import extract_region_features, summarize_features
from src.preprocessing import preprocess_image
from src.segmentation import segment_colonies
from src.visualization import draw_colony_markers, labels_to_color, make_overlay


@dataclass(frozen=True)
class ClassicalDetectCountResult:
    summary: dict[str, Any]
    features: pd.DataFrame
    density_grid: pd.DataFrame
    mask: np.ndarray
    labels: np.ndarray
    labels_visualization: np.ndarray
    overlay: np.ndarray
    debug_images: dict[str, np.ndarray]


def run_classical_detect_count(
    image_bgr: np.ndarray,
    config: dict[str, Any],
    grid_rows: int = 4,
    grid_cols: int = 4,
    moderate_count: int = 50,
    high_count: int = 200,
    moderate_density: float = 1.0,
    high_density: float = 3.0,
) -> ClassicalDetectCountResult:
    preprocessing = preprocess_image(image_bgr, **config["preprocessing"])
    segmentation = segment_colonies(
        preprocessing.normalized,
        threshold_method=config["pipeline"]["threshold_method"],
        split_touching=config["pipeline"]["split_touching"],
        **config["segmentation"],
    )
    features = extract_region_features(segmentation.labels)
    count_summary = summarize_count(segmentation.mask, segmentation.labels)
    feature_summary = summarize_features(features)
    image_height, image_width = image_bgr.shape[:2]
    density_grid = build_density_grid(features, image_width, image_height, grid_rows, grid_cols)
    growth_level = classify_growth(
        count_summary.colony_count,
        count_summary.density_per_100k_pixels,
        moderate_count=moderate_count,
        high_count=high_count,
        moderate_density=moderate_density,
        high_density=high_density,
    )
    overlay = make_overlay(image_bgr, segmentation.mask, alpha=config["outputs"]["overlay_alpha"])
    marked_overlay = draw_colony_markers(overlay, features)
    summary = {
        "task": "detect_count",
        "model_family": "classical",
        "colony_count": count_summary.colony_count,
        "growth_level": growth_level,
        "count": asdict(count_summary),
        "features": feature_summary,
        "density_grid": {
            "rows": grid_rows,
            "cols": grid_cols,
        },
        "growth_thresholds": {
            "moderate_count": moderate_count,
            "high_count": high_count,
            "moderate_density": moderate_density,
            "high_density": high_density,
        },
    }
    return ClassicalDetectCountResult(
        summary=summary,
        features=features,
        density_grid=density_grid,
        mask=segmentation.mask,
        labels=segmentation.labels,
        labels_visualization=labels_to_color(segmentation.labels),
        overlay=marked_overlay,
        debug_images={
            "original_bgr": preprocessing.image_bgr,
            "grayscale": preprocessing.grayscale,
            "enhanced": preprocessing.enhanced,
            "normalized": preprocessing.normalized,
        },
    )
