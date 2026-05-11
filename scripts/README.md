# Script Entrypoints

Scripts are grouped by project phase and task.

## `datasets/`

Dataset inspection, split creation, and annotation QA utilities for AGAR-primary.

## `classical/`

Classical computer-vision model evaluation and tuning entrypoints.

- `evaluate_detect_count_agar.py` runs one classical counting configuration.
- `sweep_detect_count_agar.py` tunes classical counting parameters on a validation split.
- `train_species_image_agar.py` trains and validates the classical image-level species classifier.
- `evaluate_species_image_agar.py` evaluates a saved classical image-level species classifier.

## Planned Folders

- `yolo/` for YOLO export, training, prediction, and evaluation scripts.
- `unet/` for U-Net data preparation, training, prediction, and evaluation scripts.
