# Script Entrypoints

Scripts are grouped by project phase and task.

## `datasets/`

Dataset inspection, split creation, and annotation QA utilities for AGAR-primary.

## `classical/`

Classical computer-vision model evaluation and tuning entrypoints.

- `evaluate_detect_count_agar.py` runs one classical counting configuration.
- `analyze_detect_count_errors_agar.py` summarizes saved counting prediction errors.
- `sweep_detect_count_agar.py` tunes classical counting parameters on a validation split, writes incremental progress, and now supports dense-colony estimation parameters.
- `train_species_image_agar.py` trains and validates the classical image-level species classifier.
- `train_species_image_agar.py` now selects the best classical species classifier on validation from multiple candidate rules.
- `sweep_species_image_agar.py` sweeps classical species feature-view settings and saves the best validation-selected model.
- `evaluate_species_image_agar.py` evaluates a saved classical image-level species classifier.

## Planned Folders

- `yolo/` for YOLO export, training, prediction, and evaluation scripts.
- `unet/` for U-Net data preparation, training, prediction, and evaluation scripts.
