from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.datasets import build_inventory, load_agar_annotation, split_inventory


def write_sample(root: Path, category: str, sample_id: int, colonies_number: int, class_name: str, labels: list[dict]) -> None:
    category_dir = root / "data" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 50, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(category_dir / f"{sample_id}.png"), image)
    payload = {
        "background": "bright",
        "classes": [class_name],
        "colonies_number": colonies_number,
        "labels": labels,
        "sample_id": sample_id,
    }
    (category_dir / f"{sample_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_agar_annotation_normalizes_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"sample_id": 1, "labels": None, "classes": None}), encoding="utf-8")

    data = load_agar_annotation(path)

    assert data["labels"] == []
    assert data["classes"] == []


def test_build_inventory_reads_categories_and_eligibility(tmp_path: Path) -> None:
    write_sample(
        tmp_path,
        "countable",
        1,
        1,
        "P.aeruginosa",
        [{"class": "P.aeruginosa", "x": 1, "y": 2, "width": 3, "height": 4, "id": 1}],
    )
    write_sample(tmp_path, "empty", 2, 0, "P.aeruginosa", [])
    write_sample(tmp_path, "uncountable", 3, -1, "E.coli", [])

    inventory, summary = build_inventory(tmp_path)

    assert len(inventory) == 3
    assert summary["total_images"] == 3
    assert summary["detection_eligible"] == 2
    assert summary["species_image_eligible"] == 2
    assert inventory.loc[inventory["category"] == "uncountable", "detection_eligible"].iloc[0] == False


def test_split_inventory_preserves_rows_and_writes_split_column(tmp_path: Path) -> None:
    for index in range(6):
        write_sample(tmp_path, "countable", index, 1, "A", [])
    inventory, _ = build_inventory(tmp_path, read_images=False)

    split_df = split_inventory(inventory, seed=7)

    assert len(split_df) == len(inventory)
    assert set(split_df["split"]) == {"train", "val", "test"}
