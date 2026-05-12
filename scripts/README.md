# Script Entrypoints

Scripts are grouped by project phase and task.

## UI

- `ui/streamlit_app.py` is the local Streamlit workbench for testing saved models on single images.
- The UI includes classical counting presets, direct upload support for classical species `model.json` files, a classical species-model preset for the current Colab sweep best-run path, and placeholder status panels for later YOLO and U-Net integration.

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

## `unet/`

- `prepare_unet_count_agar.py` converts AGAR detection/counting splits into pseudo-mask manifests for U-Net segmentation.
- `train_unet_count_agar.py` trains a U-Net segmentation model for colony-mask prediction on prepared pseudo masks.
- `evaluate_unet_count_agar.py` evaluates a saved U-Net counting segmenter on a prepared manifest and derives count metrics from connected components.
- `train_unet_species_image_agar.py` trains a U-Net encoder image-level species classifier on `species_image` splits.
- `evaluate_unet_species_image_agar.py` evaluates a saved U-Net image-level species classifier.

## Planned Folders

- `yolo/` for YOLO export, training, prediction, and evaluation scripts.
