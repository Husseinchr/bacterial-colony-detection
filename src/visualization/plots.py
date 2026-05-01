"""Visualization helpers for masks, labels, overlays, and debug panels."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a binary mask on top of the original image."""

    overlay = image_bgr.copy()
    color_mask = np.zeros_like(image_bgr)
    color_mask[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(color_mask, alpha, overlay, 1.0 - alpha, 0)


def draw_colony_markers(image_bgr: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Draw colony centroids and labels on an image."""

    output = image_bgr.copy()
    for _, row in features.iterrows():
        x = int(round(row["centroid_x"]))
        y = int(round(row["centroid_y"]))
        label = int(row["label"])
        cv2.circle(output, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(output, str(label), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return output


def labels_to_color(labels: np.ndarray) -> np.ndarray:
    """Convert integer labels to a BGR color visualization."""

    rgb = label2rgb(labels, bg_label=0, image_alpha=1.0)
    uint8_rgb = (rgb * 255).astype(np.uint8)
    return cv2.cvtColor(uint8_rgb, cv2.COLOR_RGB2BGR)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image, creating parent directories as needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def save_debug_panel(
    path: str | Path,
    original_bgr: np.ndarray,
    grayscale: np.ndarray,
    enhanced: np.ndarray,
    normalized: np.ndarray,
    mask: np.ndarray,
    overlay_bgr: np.ndarray,
) -> None:
    """Save a compact panel of important intermediate outputs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [
        cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
        grayscale,
        enhanced,
        normalized,
        mask,
        cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
    ]
    titles = ["Original", "Gray", "CLAHE", "Background normalized", "Mask", "Overlay"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, image, title in zip(axes.ravel(), images, titles):
        cmap = "gray" if image.ndim == 2 else None
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

