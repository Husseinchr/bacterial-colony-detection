from __future__ import annotations

import pytest

from src.classical import config_with_segmentation_overrides


def test_config_with_segmentation_overrides_updates_supported_sections() -> None:
    base_config = {
        "pipeline": {
            "threshold_method": "otsu",
            "split_touching": False,
        },
        "preprocessing": {
            "gaussian_kernel_size": 5,
        },
        "segmentation": {
            "min_area": 20,
            "min_circularity": 0.15,
        },
    }

    config = config_with_segmentation_overrides(
        base_config,
        {
            "threshold_method": "adaptive",
            "gaussian_kernel_size": 7,
            "min_area": 100,
            "min_circularity": 0.35,
        },
    )

    assert config["pipeline"]["threshold_method"] == "adaptive"
    assert config["preprocessing"]["gaussian_kernel_size"] == 7
    assert config["segmentation"]["min_area"] == 100
    assert config["segmentation"]["min_circularity"] == 0.35
    assert base_config["segmentation"]["min_area"] == 20


def test_config_with_segmentation_overrides_rejects_unknown_key() -> None:
    base_config = {
        "pipeline": {},
        "preprocessing": {},
        "segmentation": {},
    }

    with pytest.raises(KeyError):
        config_with_segmentation_overrides(base_config, {"unknown": 1})
