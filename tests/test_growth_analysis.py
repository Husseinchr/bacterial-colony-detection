from __future__ import annotations

from src.analysis import classify_growth


def test_classify_growth_from_count() -> None:
    assert classify_growth(10, 0.2) == "low"
    assert classify_growth(50, 0.2) == "moderate"
    assert classify_growth(200, 0.2) == "high"


def test_classify_growth_from_density() -> None:
    assert classify_growth(10, 1.0) == "moderate"
    assert classify_growth(10, 3.0) == "high"


def test_classify_growth_accepts_custom_thresholds() -> None:
    assert classify_growth(80, 0.2, moderate_count=100, high_count=300) == "low"
    assert classify_growth(120, 0.2, moderate_count=100, high_count=300) == "moderate"
