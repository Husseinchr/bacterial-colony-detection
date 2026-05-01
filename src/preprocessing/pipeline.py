"""Image loading and preprocessing utilities for colony detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    """Container for preprocessing outputs used by later pipeline stages."""

    image_bgr: np.ndarray
    grayscale: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    normalized: np.ndarray


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR format.

    Raises:
        FileNotFoundError: If the image path does not exist or cannot be decoded.
    """

    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to uint8 grayscale."""

    if image_bgr.ndim == 2:
        return image_bgr.astype(np.uint8, copy=False)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got {image_bgr.shape}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def gaussian_denoise(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Reduce sensor noise and small texture variations with Gaussian blur."""

    kernel_size = _ensure_odd(kernel_size)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """Enhance local contrast using CLAHE."""

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(gray)


def normalize_background(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Correct uneven illumination using a blurred background estimate."""

    kernel_size = _ensure_odd(kernel_size)
    background = cv2.medianBlur(gray, kernel_size)
    corrected = cv2.subtract(gray, background)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_image(
    image_bgr: np.ndarray,
    gaussian_kernel_size: int = 5,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    background_kernel_size: int = 51,
) -> PreprocessingResult:
    """Run the default preprocessing pipeline."""

    gray = to_grayscale(image_bgr)
    denoised = gaussian_denoise(gray, gaussian_kernel_size)
    enhanced = apply_clahe(denoised, clahe_clip_limit, clahe_tile_grid_size)
    normalized = normalize_background(enhanced, background_kernel_size)
    return PreprocessingResult(
        image_bgr=image_bgr,
        grayscale=gray,
        denoised=denoised,
        enhanced=enhanced,
        normalized=normalized,
    )


def _ensure_odd(value: int) -> int:
    """Return a positive odd integer suitable for OpenCV kernels."""

    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1

