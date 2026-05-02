"""Export Figshare colony annotations into a YOLO training directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Figshare dataset root.")
    parser.add_argument("--split-dir", required=True, type=Path, help="Directory containing train.csv, val.csv, test.csv.")
    parser.add_argument("--output-dir", required=True, type=Path, help="YOLO dataset output directory.")
    parser.add_argument(
        "--image-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Use copy for Google Drive portability. Symlink is faster but less portable.",
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Export all colonies as one class named by --single-class-name.",
    )
    parser.add_argument("--single-class-name", default="colony")
    return parser.parse_args()


def sorted_species(labels: pd.Series) -> list[str]:
    """Sort labels such as sp01, sp02, ..., sp24 in natural numeric order."""

    def key(label: str) -> tuple[str, int]:
        prefix = "".join(ch for ch in label if not ch.isdigit())
        digits = "".join(ch for ch in label if ch.isdigit())
        return prefix, int(digits or 0)

    return sorted(labels.unique().tolist(), key=key)


def load_split(split_dir: Path, split: str) -> pd.DataFrame:
    path = split_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    return pd.read_csv(path)


def prepare_dirs(output_dir: Path) -> None:
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def link_or_copy_image(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def yolo_rows_for_image(
    image_annotations: pd.DataFrame,
    class_to_id: dict[str, int],
    single_class: bool,
) -> list[str]:
    rows = []
    for _, row in image_annotations.iterrows():
        image_width = float(row["image_width"])
        image_height = float(row["image_height"])

        x = float(row["bbox_x"])
        y = float(row["bbox_y"])
        width = float(row["bbox_width"])
        height = float(row["bbox_height"])

        x_center = (x + width / 2.0) / image_width
        y_center = (y + height / 2.0) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        values = [
            0 if single_class else class_to_id[row["label_name"]],
            _clip01(x_center),
            _clip01(y_center),
            _clip01(norm_width),
            _clip01(norm_height),
        ]
        rows.append(f"{values[0]} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f} {values[4]:.6f}")
    return rows


def export_split(
    dataset_dir: Path,
    output_dir: Path,
    split_name: str,
    split_df: pd.DataFrame,
    annotations: pd.DataFrame,
    class_to_id: dict[str, int],
    image_mode: str,
    single_class: bool,
) -> tuple[int, int]:
    image_count = 0
    box_count = 0

    for image_name in sorted(split_df["image_name"].unique()):
        source_image = dataset_dir / image_name
        if not source_image.exists():
            raise FileNotFoundError(f"Missing image: {source_image}")

        destination_image = output_dir / "images" / split_name / image_name
        link_or_copy_image(source_image, destination_image, image_mode)

        image_annotations = annotations[annotations["image_name"] == image_name]
        label_rows = yolo_rows_for_image(image_annotations, class_to_id, single_class)

        label_path = output_dir / "labels" / split_name / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(label_rows) + "\n", encoding="utf-8")

        image_count += 1
        box_count += len(label_rows)

    return image_count, box_count


def write_dataset_metadata(output_dir: Path, class_names: list[str]) -> None:
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    annotation_path = dataset_dir / "annot_tab.csv"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    annotations = pd.read_csv(annotation_path)
    class_names = [args.single_class_name] if args.single_class else sorted_species(annotations["label_name"])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    prepare_dirs(output_dir)
    summary = {}
    for split in SPLITS:
        split_df = load_split(split_dir, split)
        image_count, box_count = export_split(
            dataset_dir,
            output_dir,
            split,
            split_df,
            annotations,
            class_to_id,
            args.image_mode,
            args.single_class,
        )
        summary[split] = {"images": image_count, "boxes": box_count}

    write_dataset_metadata(output_dir, class_names)

    print(f"Exported YOLO dataset to: {output_dir}")
    print(f"Classes: {len(class_names)}")
    for split, values in summary.items():
        print(f"{split}: {values['images']} images, {values['boxes']} boxes")
    print(f"Dataset config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
