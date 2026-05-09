from __future__ import annotations


def classify_growth(
    colony_count: int,
    density_per_100k_pixels: float,
    moderate_count: int = 50,
    high_count: int = 200,
    moderate_density: float = 1.0,
    high_density: float = 3.0,
) -> str:
    if colony_count >= high_count or density_per_100k_pixels >= high_density:
        return "high"
    if colony_count >= moderate_count or density_per_100k_pixels >= moderate_density:
        return "moderate"
    return "low"
