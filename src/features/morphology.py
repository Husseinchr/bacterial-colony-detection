"""Morphological feature extraction for colony candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def extract_region_features(labels: np.ndarray) -> pd.DataFrame:
    """Extract per-colony morphology features from a connected-component label map."""

    if labels.max() == 0:
        return pd.DataFrame(
            columns=[
                "label",
                "area",
                "perimeter",
                "centroid_y",
                "centroid_x",
                "equivalent_diameter",
                "eccentricity",
                "circularity",
            ]
        )

    props = regionprops_table(
        labels,
        properties=(
            "label",
            "area",
            "perimeter",
            "centroid",
            "equivalent_diameter",
            "eccentricity",
        ),
    )
    df = pd.DataFrame(props)
    df = df.rename(columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"})
    perimeter = df["perimeter"].replace(0, np.nan)
    df["circularity"] = (4.0 * np.pi * df["area"]) / (perimeter * perimeter)
    # Pixel-grid perimeter estimates can produce values slightly above 1.0.
    df["circularity"] = df["circularity"].fillna(0.0).clip(lower=0.0, upper=1.0)
    return df


def summarize_features(features: pd.DataFrame) -> dict[str, float]:
    """Create aggregate morphology statistics suitable for JSON output."""

    if features.empty:
        return {
            "mean_area": 0.0,
            "median_area": 0.0,
            "mean_circularity": 0.0,
            "mean_equivalent_diameter": 0.0,
        }

    return {
        "mean_area": float(features["area"].mean()),
        "median_area": float(features["area"].median()),
        "mean_circularity": float(features["circularity"].mean()),
        "mean_equivalent_diameter": float(features["equivalent_diameter"].mean()),
    }
