"""Tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np

from src.preprocessing import apply_clahe, gaussian_denoise, normalize_background, preprocess_image, to_grayscale


def test_to_grayscale_converts_bgr_image() -> None:
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    gray = to_grayscale(image)
    assert gray.shape == (10, 12)
    assert gray.dtype == np.uint8


def test_preprocess_image_preserves_image_shape() -> None:
    image = np.full((20, 30, 3), 127, dtype=np.uint8)
    result = preprocess_image(image)
    assert result.grayscale.shape == (20, 30)
    assert result.denoised.shape == (20, 30)
    assert result.enhanced.shape == (20, 30)
    assert result.normalized.shape == (20, 30)


def test_preprocessing_steps_return_uint8() -> None:
    gray = np.random.default_rng(0).integers(0, 255, size=(25, 25), dtype=np.uint8)
    assert gaussian_denoise(gray).dtype == np.uint8
    assert apply_clahe(gray).dtype == np.uint8
    assert normalize_background(gray).dtype == np.uint8

