"""Classical segmentation methods for bacterial colony candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

ThresholdMethod = Literal["otsu", "adaptive"]


@dataclass(frozen=True)
class SegmentationResult:
    """Outputs from the classical segmentation stage."""

    mask: np.ndarray
    labels: np.ndarray
    num_labels: int


def threshold_image(
    gray: np.ndarray,
    method: ThresholdMethod = "otsu",
    invert: bool = False,
    adaptive_block_size: int = 51,
    adaptive_c: int = 2,
) -> np.ndarray:
    """Create a binary mask using Otsu or adaptive thresholding."""

    if method == "otsu":
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif method == "adaptive":
        block_size = _ensure_odd(adaptive_block_size)
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            block_size,
            int(adaptive_c),
        )
    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    return mask.astype(np.uint8)


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
) -> np.ndarray:
    """Remove isolated noise and close small holes in segmented colonies."""

    kernel_size = _ensure_odd(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
    return closed


def filter_components(
    mask: np.ndarray,
    min_area: int = 20,
    max_area: int = 20000,
    min_circularity: float = 0.15,
) -> np.ndarray:
    """Filter connected components by area and circularity."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        component = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], closed=True)
        circularity = 0.0 if perimeter == 0 else (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity >= min_circularity:
            filtered[labels == label_id] = 255

    return filtered


def split_touching_colonies(mask: np.ndarray, min_distance: int = 8) -> np.ndarray:
    """Split touching colonies with distance-transform watershed.

    This is intentionally optional because watershed can over-split noisy plates.
    """

    binary = mask > 0
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)

    markers = np.zeros(distance.shape, dtype=np.int32)
    for marker_id, (row, col) in enumerate(coords, start=1):
        markers[row, col] = marker_id

    if markers.max() == 0:
        return mask

    labels = watershed(-distance, markers, mask=binary)
    return (labels > 0).astype(np.uint8) * 255


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected colony candidates."""

    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels.astype(np.int32), num_labels - 1


def segment_colonies(
    gray: np.ndarray,
    threshold_method: ThresholdMethod = "otsu",
    invert_threshold: bool = False,
    adaptive_block_size: int = 51,
    adaptive_c: int = 2,
    morph_kernel_size: int = 3,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
    min_area: int = 20,
    max_area: int = 20000,
    min_circularity: float = 0.15,
    split_touching: bool = False,
) -> SegmentationResult:
    """Run thresholding, cleanup, filtering, and connected-component labeling."""

    raw_mask = threshold_image(
        gray,
        method=threshold_method,
        invert=invert_threshold,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )
    cleaned = clean_mask(raw_mask, morph_kernel_size, opening_iterations, closing_iterations)
    filtered = filter_components(cleaned, min_area, max_area, min_circularity)
    if split_touching:
        filtered = split_touching_colonies(filtered)

    labels, count = label_mask(filtered)
    return SegmentationResult(mask=filtered, labels=labels, num_labels=count)


def _ensure_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1

