from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.classical.evaluate_detect_count import evaluate_classical_detect_count


def config_with_segmentation_overrides(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base_config)
    for key, value in overrides.items():
        if key in {"threshold_method", "split_touching"}:
            config["pipeline"][key] = value
        elif key in config["segmentation"]:
            config["segmentation"][key] = value
        elif key in config["preprocessing"]:
            config["preprocessing"][key] = value
        else:
            raise KeyError(f"Unsupported config override: {key}")
    return config


def sweep_classical_detect_count(
    dataset_dir: Path,
    split_csv: Path,
    base_config: dict[str, Any],
    search_space: list[dict[str, Any]],
    grid_rows: int = 4,
    grid_cols: int = 4,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    best_config = deepcopy(base_config)

    for index, overrides in enumerate(search_space, start=1):
        config = config_with_segmentation_overrides(base_config, overrides)
        evaluation = evaluate_classical_detect_count(
            dataset_dir,
            split_csv,
            config,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            limit=limit,
        )
        row = {"trial": index, **overrides, **evaluation.summary}
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    if not results.empty:
        best_overrides = search_space[int(results.iloc[0]["trial"]) - 1]
        best_config = config_with_segmentation_overrides(base_config, best_overrides)

    return results, best_config
