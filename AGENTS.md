# Project Context for Agents

## Project

Automated Bacterial Colony Detection, Counting, and Morphology Analysis.

This repository is for a university computer vision project by Hussein Chreif and Mostafa Younes. The system should detect bacterial colonies in petri dish images, segment individual colonies, count colonies, extract morphology features, and provide quantitative analysis such as size distribution and spatial density.

## Current Engineering Direction

The first milestone is a deterministic classical computer vision baseline. Do not jump to U-Net, YOLO, Mask R-CNN, or classification before counting and segmentation evaluation are working.

Current local machine constraint observed on 2026-05-01:

- `nvidia-smi` is unavailable.
- `lspci` shows `VMware SVGA II Adapter`, not an NVIDIA CUDA GPU.
- RAM is about 7.5 GiB, with about 4.0 GiB available during inspection.
- Disk free space in the project filesystem is about 55 GiB.
- CPU shown to the VM: Intel Core i7-10750H class, 4 visible CPUs.

Implication: run the OpenCV baseline locally. Treat local U-Net training as CPU-only experimentation with very small images and tiny batches, or use Google Colab/Kaggle/cloud GPU for serious segmentation training.

Preferred execution workflow:

- GitHub/local repository is the source of truth for code, configs, documentation, and reports.
- Google Colab is an execution environment for GPU experiments and future U-Net training.
- Local machine is used for code editing, quick tests, dataset organization, and classical baseline checks.
- Colab notebooks should call scripts and modules from this repository. Do not let core pipeline logic live only inside notebooks.
- Large datasets should live in Google Drive or Kaggle/Colab storage, not in git.
- Generated outputs and model checkpoints should be saved outside git, then only selected final figures/tables should be copied into reports.

Default baseline pipeline:

1. Load BGR image with OpenCV.
2. Convert to grayscale.
3. Apply Gaussian denoising.
4. Apply CLAHE contrast enhancement.
5. Normalize uneven illumination with background subtraction.
6. Segment colonies with Otsu thresholding by default.
7. Clean binary mask with morphological opening and closing.
8. Filter connected components by area and circularity.
9. Optionally split touching colonies with watershed.
10. Count connected components.
11. Extract area, perimeter, centroid, equivalent diameter, eccentricity, and circularity.
12. Save mask, labels, overlay, feature CSV, debug panel, and summary JSON.

## Repository Structure

- `configs/`: YAML pipeline configuration.
- `data/raw/`: original datasets, never edited.
- `data/interim/`: temporary converted data.
- `data/processed/`: generated cleaned images or masks.
- `data/annotations/`: ground-truth masks, counts, metadata, and split files.
- `src/`: reusable Python packages.
- `src/preprocessing/`: image loading, grayscale conversion, denoising, CLAHE, background normalization.
- `src/segmentation/`: classical segmentation now and future model-backed segmentation.
- `src/counting/`: connected-component counting and density summaries.
- `src/features/`: morphology feature extraction.
- `src/evaluation/`: segmentation and counting metrics.
- `src/visualization/`: overlays, label maps, and debug panels.
- `scripts/`: runnable command-line entrypoints.
- `tests/`: unit tests.
- `outputs/`: generated baseline outputs.
- `models/`: future trained model checkpoints and exports.
- `notebooks/`: exploratory analysis only, not production pipeline logic.

## Datasets

Primary dataset:

- Microbial Colony Recognition Dataset from Kaggle for detection, counting, and quantitative analysis.

Optional datasets:

- Bacteria Data for Machine Vision and Digital Biology from Mendeley for morphology or classification.
- Selected Broad Bioimage Benchmark Collection datasets for optional segmentation support or pretraining.

Use image-level or plate-level train/validation/test splits. Avoid leakage from crops or augmented variants of the same plate appearing across different splits.

## Evaluation Plan

Segmentation metrics when masks exist:

- IoU
- Dice
- pixel precision, recall, and F1

Counting metrics when ground-truth counts exist:

- absolute count error
- MAE
- RMSE
- mean signed error
- mean absolute percentage error

Feature validation:

- Compare extracted feature distributions against mask-derived or manually reviewed annotations.
- Review outliers visually, especially very small, very large, or low-circularity detections.

## Real-World Image Issues

Expected failure modes include touching colonies, overlapping colonies, uneven illumination, glare, low contrast, plate borders, agar texture, dust, and imaging noise.

Preferred baseline techniques:

- CLAHE for local contrast.
- Background normalization for uneven illumination.
- Gaussian blur for noise reduction.
- Otsu thresholding as the default global threshold.
- Adaptive thresholding for strongly uneven plates.
- Morphological opening and closing for cleanup.
- Connected components and contour filtering for colony candidate extraction.
- Area and circularity filtering to remove artifacts.
- Watershed only as an optional post-processing step because it can over-split.

## Coding Rules

- Keep code modular and readable.
- Prefer OpenCV, NumPy, scikit-image, pandas, matplotlib, and PyYAML for the baseline.
- Use PyTorch only when adding the future U-Net pipeline.
- Keep pipeline logic out of notebooks.
- Do not commit raw datasets, generated outputs, or model binaries.
- Add focused tests for reusable functions.
