from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import load_class_names


def read_yolo_label_file(path: Path, conf_threshold: float = 0.0) -> pd.DataFrame:
    columns = ["class_id", "x_center", "y_center", "width", "height", "confidence"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame(columns=columns)

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        confidence = float(parts[5]) if len(parts) >= 6 else 1.0
        if confidence < conf_threshold:
            continue
        rows.append(
            {
                "class_id": int(float(parts[0])),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def class_name_for_id(class_id: int, class_names: list[str] | None) -> str:
    if class_names is not None and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def load_optional_class_names(path: Path | None) -> list[str] | None:
    return load_class_names(path) if path is not None else None
