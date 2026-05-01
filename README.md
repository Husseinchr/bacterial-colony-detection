# Automated Bacterial Colony Detection, Counting, and Morphology Analysis

This project implements a computer vision pipeline for petri dish images. The first milestone is a reliable classical baseline for colony detection and counting. Deep learning segmentation, classification, and richer morphology analysis are planned as later extensions.

## Engineering Scope

Core deliverables:

- Load and preprocess petri dish images.
- Segment likely colony regions using deterministic OpenCV methods.
- Count colony candidates after size and shape filtering.
- Extract basic morphology features such as area, perimeter, circularity, and centroid.
- Save overlays, masks, feature tables, and metrics-ready outputs.

Optional deliverables:

- U-Net segmentation in PyTorch.
- Growth-level classification.
- Density maps and advanced spatial statistics.
- Experiment tracking and richer CLI workflows.

## Repository Layout

```text
data/
  raw/              Original datasets, never edited
  interim/          Temporary converted files
  processed/        Cleaned images or generated masks
  annotations/      Ground-truth masks, count CSVs, metadata
configs/            YAML configuration files
src/                Reusable Python packages
  preprocessing/    Loading, grayscale conversion, denoising, contrast correction
  segmentation/     Classical segmentation now, U-Net integration later
  counting/         Component counting and density summaries
  features/         Morphology feature extraction
  evaluation/       Segmentation and count metrics
  visualization/    Overlays, label images, and debug panels
scripts/            Runnable entrypoints
tests/              Unit tests
outputs/            Generated masks, overlays, tables, metrics
models/             Trained model checkpoints and exports
notebooks/          Exploratory notebooks only
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Classical Baseline

```bash
python scripts/run_baseline.py \
  --image data/raw/example_plate.jpg \
  --config configs/default.yaml \
  --output-dir outputs/baseline_example
```

Expected outputs:

- `mask.png`: binary colony segmentation mask
- `labels.png`: connected-component label visualization
- `overlay.png`: detected colonies drawn on the original image
- `features.csv`: per-colony morphology features
- `summary.json`: count and aggregate statistics

## Dataset Policy

Keep raw datasets unchanged under `data/raw/`. Store annotations separately under `data/annotations/`. Splits must be image-level or plate-level, not crop-level, to avoid leakage.

Primary dataset:

- Microbial Colony Recognition Dataset from Kaggle for detection, counting, and analysis.

Optional datasets:

- Bacteria Data for Machine Vision and Digital Biology for morphology/classification experiments.
- Selected BBBC datasets for segmentation pretraining or method comparison.

## Baseline Pipeline

The default classical pipeline is:

1. Load image.
2. Convert to grayscale.
3. Apply Gaussian denoising.
4. Enhance contrast with CLAHE.
5. Normalize uneven background.
6. Threshold with Otsu by default.
7. Clean mask with morphology.
8. Optionally split touching regions with watershed.
9. Extract connected components.
10. Filter by area and circularity.
11. Save visualizations and feature tables.

## Evaluation

Use segmentation metrics when masks are available:

- IoU
- Dice
- precision / recall / F1 at the pixel level

Use counting metrics when ground-truth counts are available:

- absolute count error
- MAE
- RMSE
- percentage error

Feature extraction should be validated by comparing extracted properties with annotation-derived measurements when masks or manually reviewed component labels exist.
