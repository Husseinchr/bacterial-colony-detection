"""Counting utilities for segmented bacterial colonies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CountSummary:
    """Summary of colony counting results."""

    colony_count: int
    mask_area_pixels: int
    image_area_pixels: int
    density_per_100k_pixels: float


def count_from_labels(labels: np.ndarray) -> int:
    """Count connected components in a label image, excluding background label 0."""

    unique_labels = np.unique(labels)
    return int(np.sum(unique_labels > 0))


def summarize_count(mask: np.ndarray, labels: np.ndarray) -> CountSummary:
    """Compute count and coarse density summary from segmentation outputs."""

    image_area = int(mask.shape[0] * mask.shape[1])
    mask_area = int(np.count_nonzero(mask))
    count = count_from_labels(labels)
    density = 0.0 if image_area == 0 else count / image_area * 100_000.0
    return CountSummary(
        colony_count=count,
        mask_area_pixels=mask_area,
        image_area_pixels=image_area,
        density_per_100k_pixels=density,
    )

