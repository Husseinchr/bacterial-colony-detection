# Automated Bacterial Colony Analysis on AGAR-Primary

This repository contains a university computer vision project for automated bacterial colony analysis on the AGAR-primary dataset.

The project implements three modeling families:

- Classical computer vision
- YOLO object detection
- U-Net-based deep learning

Across those families, the repository contains six final model tracks:

1. Classical counting
2. Classical image-level species classification
3. YOLO counting
4. YOLO object-level species detection
5. U-Net counting
6. U-Net image-level species classification

The codebase is designed so that:

- local development happens in this repository
- heavy training and split evaluation run in Google Colab
- validation is used for model selection
- test is used only for locked final reporting

## Current Status

The AGAR-primary workflow is complete through:

- Phase 1: dataset inspection, split creation, and annotation QA
- Phase 2: classical baselines
- Phase 3: YOLO models
- Phase 4: U-Net models

The local Streamlit workbench supports single-image testing for the locked:

- classical models
- YOLO models
- U-Net models

## Final Locked Results

### Counting

| Model | Test split | MAE | RMSE | Bias | Exact match | Within 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classical counting | `detect_count/test.csv` | 33.5333 | 60.8307 | -18.6410 | 0.1026 | 0.3538 |
| U-Net counting | `detect_count/test.csv` | 11.3333 | 27.9593 | -10.9949 | 0.4205 | 0.7077 |
| YOLO counting | `detect_count/test.csv` | 1.1077 | 3.2360 | -0.6462 | 0.6769 | 0.9487 |

Best counting model:

- **YOLO counting**

### Image-Level Species Classification

| Model | Test split | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Classical species | `species_image/test.csv` | 0.6045 | 0.6036 |
| U-Net species | `species_image/test.csv` | 0.7388 | 0.7160 |

Best image-level species model:

- **U-Net species classification**

### Object-Level Species Detection

| Model | Test split | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLO species | `countable_only/test` | 0.9630 | 0.9440 | 0.9689 | 0.8479 | 0.6945 |

Important reporting note:

- YOLO species is an **object-level countable-only** model
- Classical and U-Net species models are **image-level** models on `species_image`
- These species results should not be compared as if they solve the same task

## Dataset

Dataset root in Colab:

`/content/drive/MyDrive/bacterial_colony_detection/data/raw/agar_primary`

Expected structure:

- `data/countable/`
- `data/empty/`
- `data/uncountable/`

Each image has a paired JSON annotation file with the same stem.

### Category Semantics

- `countable`
  - individual colonies are annotated with boxes
  - `colonies_number >= 0`
- `empty`
  - no colonies are present
  - `colonies_number = 0`
- `uncountable`
  - visible dense growth exists
  - individual colonies are not annotated
  - `colonies_number = -1`

### Core Dataset Policy

- detection and counting use `countable` and `empty`
- uncountable images are not used as object-detection negatives
- image-level species classification uses countable and uncountable images
- object-level species detection uses countable-only images

This policy is central to the project. Uncountable plates may contain visible colonies without colony-level boxes, so using them as negatives would create false supervision.

### Inspected Dataset Summary

- total images: `2089`
- countable: `994`
- empty: `309`
- uncountable: `786`
- readable images: `2089`
- paired JSON files: `2089`
- detection-eligible images: `1303`
- image-level species-eligible images: `1780`

### Split Summary

Global split:

| Split | Images |
| --- | ---: |
| Train | 1461 |
| Validation | 315 |
| Test | 313 |

Category split:

| Split | Countable | Empty | Uncountable |
| --- | ---: | ---: | ---: |
| Train | 695 | 216 | 550 |
| Validation | 149 | 48 | 118 |
| Test | 150 | 45 | 118 |

Task-specific split counts:

| Task | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Detection and counting | 911 | 197 | 195 |
| Image-level species classification | 1245 | 267 | 268 |
| Countable-only object-level species | 695 | 149 | 150 |

## Repository Layout

```text
final_report.md          Final doctor-facing report
handwritten_notes.md     Development log from start to completion
AGENTS.md                Project workflow and status rules for agents
scripts/
  datasets/              AGAR inspection, splitting, and annotation QA
  classical/             Classical evaluation and tuning entrypoints
  yolo/                  YOLO export, training, and evaluation entrypoints
  unet/                  U-Net preparation, training, and evaluation entrypoints
  ui/                    Local Streamlit workbench
src/
  datasets/              AGAR dataset discovery and split logic
  classical/             Classical counting and species modules
  yolo/                  YOLO export and count-evaluation utilities
  unet/                  U-Net backends and helpers
  ui/                    Streamlit inference helpers
tests/                   Focused unit tests
testing_images/          Manual smoke-test images for UI checks
models_trained/          Local model organization helpers
```

For script-level details, see:

- [scripts/README.md](scripts/README.md)

## Main Entrypoints

### Dataset

- `scripts/datasets/inspect_agar_dataset.py`
- `scripts/datasets/create_agar_splits.py`
- `scripts/datasets/visualize_agar_annotations.py`

### Classical

- `scripts/classical/evaluate_detect_count_agar.py`
- `scripts/classical/analyze_detect_count_errors_agar.py`
- `scripts/classical/sweep_detect_count_agar.py`
- `scripts/classical/train_species_image_agar.py`
- `scripts/classical/sweep_species_image_agar.py`
- `scripts/classical/evaluate_species_image_agar.py`

### YOLO

- `scripts/yolo/export_yolo_agar.py`
- `scripts/yolo/train_yolo_agar.py`
- `scripts/yolo/evaluate_yolo_count_agar.py`
- `scripts/yolo/sweep_yolo_count_thresholds_agar.py`
- `scripts/yolo/evaluate_yolo_species_agar.py`

### U-Net

- `scripts/unet/prepare_unet_count_agar.py`
- `scripts/unet/train_unet_count_agar.py`
- `scripts/unet/evaluate_unet_count_agar.py`
- `scripts/unet/sweep_unet_count_postprocess_agar.py`
- `scripts/unet/train_unet_species_image_agar.py`
- `scripts/unet/evaluate_unet_species_image_agar.py`

### Local UI

- `scripts/ui/streamlit_app.py`

## Setup

Create the local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Local Streamlit Workbench

```bash
.venv/bin/python -m streamlit run scripts/ui/streamlit_app.py --server.port 8501 --server.headless true
```

Then open:

- `http://localhost:8501`

### Local Dependency Note

The YOLO panels require a working local installation of:

- `torch`
- `torchvision`
- `ultralytics`

If YOLO inference fails locally with a `torchvision::nms` error, reinstall the PyTorch stack as a matched set inside `.venv`.

## Model Artifacts

Locked Colab-side model paths:

### Classical

- Classical species model:
  - `/content/drive/MyDrive/bacterial_colony_detection/outputs/classical_species_image/val_sweep_001/best_run/model.json`
- Classical counting:
  - no learned checkpoint file
  - this baseline is a deterministic config-driven pipeline

### U-Net

- U-Net counting:
  - `/content/drive/MyDrive/bacterial_colony_detection/outputs/unet_count/train_run_001/model.pt`
- U-Net species:
  - `/content/drive/MyDrive/bacterial_colony_detection/outputs/unet_species_image/train_run_001/model.pt`

### YOLO

- YOLO counting:
  - `/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_count/train_run_001/weights/best.pt`
- YOLO species:
  - `/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_species/train_run_001/weights/best.pt`

For the local Streamlit app, these files must exist on the local machine. The app cannot directly read Colab `/content/...` paths.

## Development Workflow Rules

- keep raw datasets, model binaries, and generated outputs out of git
- use validation for tuning and test only for final locked reporting
- keep reusable logic in `src/` and `scripts/`
- keep notebooks exploratory rather than authoritative
- preserve the distinction between:
  - object-level `countable_only` species detection
  - image-level `species_image` classification

## Final Deliverables in This Repository

- [final_report.md](final_report.md): final project report
- [handwritten_notes.md](handwritten_notes.md): full implementation history
- [report.md](report.md): internal technical project summary
