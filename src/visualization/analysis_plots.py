from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_size_distribution(
    path: str | Path,
    values: pd.Series,
    title: str = "Colony size distribution",
    xlabel: str = "Equivalent diameter (pixels)",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    values = values.dropna()
    if values.empty:
        ax.text(0.5, 0.5, "No colonies", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=20, color="#2563eb", edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Colony count")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_density_heatmap(path: str | Path, density_grid: pd.DataFrame, rows: int, cols: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    matrix = density_grid.pivot(index="grid_row", columns="grid_col", values="count").to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="magma")
    ax.set_title("Spatial colony density")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    for row_idx in range(rows):
        for col_idx in range(cols):
            ax.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Detections")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
