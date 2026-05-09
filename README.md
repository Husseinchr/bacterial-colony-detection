# Automated Bacterial Colony Detection, Counting, and Morphology Analysis

This repository contains a computer vision project for detecting, counting, and analyzing bacterial colonies in petri dish images. It includes a deterministic OpenCV segmentation baseline, a YOLO object-detection/counting workflow, post-detection morphology and density analysis, and growth-level classification.

The project is developed by Hussein Chreif and Mostafa Younes for a university computer vision project.

## Current Status

The classical baseline is available for deterministic segmentation/counting experiments. The current strongest result comes from a single-class YOLOv8n detector trained on the Figshare Scientific Data bacterial colony dataset, where every colony annotation is exported as the class `colony`.

Current best clean held-out test result:

| Model | Image size | Classes | Threshold source | Test MAE | Test RMSE | Test MAPE |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| YOLOv8n | 1536 | 1 colony class | validation-selected `conf=0.32` | 10.603 | 24.895 | 11.787% |

The 1536 image-size single-class YOLOv8n model is now the best result by MAE and RMSE. The 1024 model still has lower MAPE, so both should be reported with a clear metric tradeoff.

The complete system output is not limited to a count. YOLO detections can be converted into per-colony geometry features, spatial density grids, size-distribution plots, overlays, and JSON/CSV summaries with `scripts/analyze_yolo_detections.py`.

The unified full pipeline is available in `scripts/run_full_pipeline.py`. It supports:

- `--mode classical`: preprocessing, classical segmentation, connected components, mask-derived features, density, growth level, and visual outputs.
- `--mode yolo`: YOLO detection or existing YOLO labels, count, class/species distribution when class names are supplied, detection-derived morphology, density, growth level, and visual outputs.

## Recommended Next Step

Freeze the current results for reporting. The final result summary is in `docs/final_results_summary.md`.

Do not tune the threshold further on the test set; `conf=0.32` was selected on validation and must remain fixed for reporting. If there is time for one more experiment, make it validation-driven tiled inference or tiled training, not another test-tuned threshold sweep.

## Submission Drafts

- `docs/final_report.md`: report draft aligned with the course deliverable structure.
- `docs/final_results_summary.md`: compact result and error-analysis summary.
- `docs/demo_video_script.md`: 2-4 minute demo narration and screen plan.
- `docs/defense_talking_points.md`: questions and answers to prepare for the live defense.

## Repository Layout

```text
configs/            YAML configuration files
data/
  raw/              Original datasets, never edited
  interim/          Temporary converted files
  processed/        Cleaned images, generated masks, or exported YOLO datasets
  annotations/      Ground-truth masks, counts, metadata, and split files
models/             Future checkpoints and exports; large binaries stay out of git
notebooks/          Exploratory notebooks only
outputs/            Generated masks, overlays, tables, metrics, and summaries
scripts/            Runnable command-line entrypoints
src/                Reusable Python packages
  preprocessing/    Loading, grayscale conversion, denoising, contrast correction
  segmentation/     Classical segmentation
  counting/         Connected-component counting and density summaries
  features/         Morphology feature extraction
  evaluation/       Segmentation and count metrics
  visualization/    Overlays, label maps, and debug panels
tests/              Unit tests
```

Keep core pipeline logic in `src/` and `scripts/`. Colab notebooks should call repository scripts instead of becoming the only place where training or evaluation logic exists.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local execution is intended for code editing, quick tests, dataset inspection, and the classical OpenCV baseline. The observed local machine does not have an NVIDIA CUDA GPU, so serious YOLO or future U-Net training should run in Colab, Kaggle, or another GPU environment.

## Datasets

Preferred production-oriented dataset:

- AGAR, Annotated Germs for Automated Recognition. Access has been requested but is not available yet.

Current practical main dataset:

- Annotated Dataset for Deep-Learning-Based Bacterial Colony Detection from Scientific Data / Figshare.
- Local inspected path: `/home/hchreif/Downloads/22022540`
- Size: about 630 MB
- Images: 369 `.jpg` files
- Annotations: `annot_tab.csv`, `annot_tab.tsv`, `annot_COCO.json`, `annot_YOLO/`, and `annot_VOC_XML/`
- CSV boxes: 56,865 colonies across 24 species labels
- Split with seed 42:
  - train: 254 images / 39,594 colonies
  - validation: 57 images / 9,410 colonies
  - test: 58 images / 7,861 colonies

Small smoke-test dataset:

- Kaggle Microbial Colony Recognition Dataset.
- It has 40 images and paired JSON annotations, so it is useful for smoke tests and demos but too small to be the main deep learning dataset.

Optional stress-test dataset:

- BBBC004 Synthetic Cells, only after the core baseline is working. It is useful for overlap/counting stress tests but is not a bacterial colony dataset.

## Classical Baseline

The deterministic OpenCV baseline follows this pipeline:

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

Run it with:

```bash
python scripts/run_baseline.py \
  --image data/raw/example_plate.jpg \
  --config configs/default.yaml \
  --output-dir outputs/baseline_example
```

Expected outputs:

- `mask.png`
- `labels.png`
- `overlay.png`
- `features.csv`
- `debug_panel.png`
- `summary.json`

## Full Pipeline

Run the complete system in YOLO mode using existing prediction labels:

```bash
python scripts/run_full_pipeline.py \
  --mode yolo \
  --image /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection/sp09_img12.jpg \
  --prediction-label /content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_predictions/figshare_yolov8n_single_class_img1536_test_conf001_maxdet1000-2/labels/sp09_img12.txt \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/full_pipeline/sp09_img12_yolo \
  --conf-threshold 0.32
```

Run the complete system in YOLO mode directly from trained weights:

```bash
python scripts/run_full_pipeline.py \
  --mode yolo \
  --image /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection/sp09_img12.jpg \
  --weights /content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_training/figshare_yolov8n_single_class_img1536/weights/best.pt \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/full_pipeline/sp09_img12_yolo_from_weights \
  --conf-threshold 0.32 \
  --predict-conf 0.01 \
  --imgsz 1536 \
  --max-det 1000
```

Run the complete system with the 24-class species model:

```bash
python scripts/run_full_pipeline.py \
  --mode yolo \
  --image /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection/sp09_img12.jpg \
  --weights /content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_training/figshare_yolov8n_baseline/weights/best.pt \
  --class-names /content/drive/MyDrive/bacterial_colony_detection/data/processed/figshare_yolo/data.yaml \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/full_pipeline/sp09_img12_yolo_24class \
  --conf-threshold 0.10 \
  --predict-conf 0.01 \
  --imgsz 1024 \
  --max-det 1000
```

Run the complete system in classical mode:

```bash
python scripts/run_full_pipeline.py \
  --mode classical \
  --image /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection/sp09_img12.jpg \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/full_pipeline/sp09_img12_classical \
  --config configs/default.yaml
```

YOLO-mode outputs:

- `detections.csv`
- `class_distribution.csv`
- `density_grid.csv`
- `summary.json`
- `overlay.png`
- `size_distribution.png`
- `density_heatmap.png`
- `yolo_prediction.txt`, only when `--weights` is used

Classical-mode outputs:

- `features.csv`
- `density_grid.csv`
- `summary.json`
- `mask.png`
- `labels.png`
- `overlay.png`
- `debug_panel.png`
- `size_distribution.png`
- `density_heatmap.png`

The growth level in `summary.json` is a rule-based low/moderate/high label derived from colony count and density. It is growth-intensity classification, not bacterial species classification.

Species/type classification is supported when the 24-class YOLO model is used with `--class-names`. The single-class model remains the best counting model, while the 24-class model provides species labels with weaker detection/counting performance.

## Figshare Workflow

Inspect the dataset:

```bash
python scripts/inspect_dataset.py \
  --dataset-dir /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection
```

Create reproducible image-level splits:

```bash
python scripts/create_figshare_splits.py \
  --dataset-dir /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/data/splits/figshare \
  --seed 42
```

Visually check annotations before training:

```bash
python scripts/visualize_figshare_annotations.py \
  --dataset-dir /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection \
  --split-csv /content/drive/MyDrive/bacterial_colony_detection/data/splits/figshare/train.csv \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/annotation_qa/train \
  --max-boxes 0
```

Export a single-class YOLO dataset:

```bash
python scripts/export_figshare_yolo_dataset.py \
  --dataset-dir /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection \
  --split-dir /content/drive/MyDrive/bacterial_colony_detection/data/splits/figshare \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/data/processed/figshare_yolo_single_class \
  --image-mode copy \
  --single-class \
  --single-class-name colony
```

## YOLO Training

The first 24-class YOLOv8n model worked but was not aligned with the main counting objective. A single-class detector is much better for colony counting because it uses model capacity to find colonies instead of separating species labels.

Available configs:

- `configs/yolo_figshare.yaml`: 24-class YOLOv8n, `imgsz=1024`, `batch=4`
- `configs/yolo_figshare_single_class.yaml`: single-class YOLOv8n, `imgsz=1024`, `batch=4`
- `configs/yolo_figshare_single_class_1536.yaml`: single-class YOLOv8n, `imgsz=1536`, `batch=2`

Example Colab training command:

```bash
yolo cfg=configs/yolo_figshare_single_class.yaml
```

For prediction/count evaluation, use `max_det=1000`. The dataset contains dense plates, and YOLO's default `max_det=300` can cap predictions and cause undercounting.

## Count Evaluation

Prediction labels must be saved with confidences:

```bash
yolo detect predict \
  model=/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_training/figshare_yolov8n_single_class/weights/best.pt \
  source=/content/drive/MyDrive/bacterial_colony_detection/data/processed/figshare_yolo_single_class/images/val \
  imgsz=1024 \
  conf=0.01 \
  max_det=1000 \
  save_txt=True \
  save_conf=True \
  project=/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_predictions \
  name=figshare_yolov8n_single_class_val_conf001
```

Sweep thresholds on validation:

```bash
python scripts/sweep_yolo_count_thresholds.py \
  --split-csv /content/drive/MyDrive/bacterial_colony_detection/data/splits/figshare/val.csv \
  --prediction-label-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_predictions/figshare_yolov8n_single_class_val_conf001/labels \
  --output-csv /content/drive/MyDrive/bacterial_colony_detection/outputs/evaluation/figshare_yolov8n_single_class_val_threshold_sweep.csv \
  --min-conf 0.01 \
  --max-conf 0.50 \
  --step 0.01
```

Evaluate held-out test counts with the validation-selected threshold:

```bash
python scripts/evaluate_yolo_count_predictions.py \
  --split-csv /content/drive/MyDrive/bacterial_colony_detection/data/splits/figshare/test.csv \
  --prediction-label-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_predictions/figshare_yolov8n_single_class_test/labels \
  --output-csv /content/drive/MyDrive/bacterial_colony_detection/outputs/evaluation/figshare_yolov8n_single_class_test_count_errors.csv \
  --conf-threshold 0.32
```

Summarize model count results:

```bash
python scripts/summarize_experiment_results.py \
  --result yolo24_1024 /content/drive/MyDrive/bacterial_colony_detection/outputs/evaluation/figshare_yolov8n_24class_test_count_errors.csv 0.10 \
  --result yolo1_1024 /content/drive/MyDrive/bacterial_colony_detection/outputs/evaluation/figshare_yolov8n_single_class_test_count_errors.csv 0.32 \
  --result yolo1_1536 /content/drive/MyDrive/bacterial_colony_detection/outputs/evaluation/figshare_yolov8n_single_class_img1536_test_count_errors_conf032_rerun.csv 0.32 \
  --output-csv /content/drive/MyDrive/bacterial_colony_detection/outputs/evaluation/model_count_comparison.csv
```

## Detection Analysis Outputs

After generating YOLO prediction labels with `save_txt=True` and `save_conf=True`, run a full per-image analysis:

```bash
python scripts/analyze_yolo_detections.py \
  --image /content/drive/MyDrive/bacterial_colony_detection/data/raw/figshare_bacterial_colony_detection/sp09_img12.jpg \
  --prediction-label /content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_predictions/figshare_yolov8n_single_class_img1536_test_conf001_maxdet1000-2/labels/sp09_img12.txt \
  --output-dir /content/drive/MyDrive/bacterial_colony_detection/outputs/analysis/sp09_img12 \
  --conf-threshold 0.32
```

Expected outputs:

- `detections.csv`: one row per predicted colony with confidence, centroid, bounding-box size, equivalent diameter, aspect ratio, and area.
- `density_grid.csv`: spatial grid counts and density per 100k pixels.
- `summary.json`: colony count, average size, median size, density, confidence, and centroid spread.
- `overlay.png`: predicted boxes and centroids.
- `size_distribution.png`: detected colony size histogram.
- `density_heatmap.png`: spatial colony density heatmap.

These features are detection-derived morphology approximations. They support quantitative colony analysis, but they are not a substitute for true mask-derived segmentation metrics.

## Results So Far

| Experiment | Split | Main result |
| --- | --- | --- |
| 24-class YOLOv8n, 1024 | validation | precision 0.467, recall 0.467, mAP50 0.378, mAP50-95 0.208 |
| 24-class YOLOv8n, 1024 | test detection | precision 0.578, recall 0.421, mAP50 0.397, mAP50-95 0.216 |
| 24-class YOLOv8n, 1024 | test count, fixed `conf=0.10` | MAE 44.293, RMSE 94.616, MAPE 32.020% |
| single-class YOLOv8n, 1024 | validation detection | precision 0.917, recall 0.900, mAP50 0.934, mAP50-95 0.536 |
| single-class YOLOv8n, 1024 | test count, fixed `conf=0.32` | MAE 11.776, RMSE 30.032, MAPE 9.636% |
| single-class YOLOv8n, 1536 | validation detection | precision 0.934, recall 0.912, mAP50 0.954, mAP50-95 0.551 |
| single-class YOLOv8n, 1536 | validation count, selected `conf=0.32` | MAE 9.246, RMSE 18.744, MAPE 9.291% |
| single-class YOLOv8n, 1536 | test count, fixed `conf=0.32` | MAE 10.603, RMSE 24.895, MAPE 11.787% |

The 1536 model improves test MAE and RMSE over 1024, but 1024 has better MAPE. This suggests 1536 reduces larger absolute errors on dense plates while making proportionally larger mistakes on some low-count plates.

The 24-class model is the species/type classifier. It is retained as a secondary result because it predicts species labels, but it is not the best model for the core counting objective.

For a report-ready interpretation, see `docs/final_results_summary.md`.

## Evaluation Metrics

Use segmentation metrics only when true masks exist:

- IoU
- Dice
- pixel precision, recall, and F1

Use counting metrics when ground-truth object counts or boxes exist:

- absolute count error
- MAE
- RMSE
- mean signed error
- MAPE

Most current public colony datasets are bounding-box datasets, not mask datasets. If the final report claims segmentation performance, it needs true mask annotations or a clear weak-supervision caveat.

## Development Rules

- Keep raw datasets, generated outputs, and model binaries out of git.
- Use image-level or plate-level train/validation/test splits to avoid leakage.
- Choose thresholds and hyperparameters on validation, then report test once.
- Keep scripts reusable and deterministic.
- Keep notebooks exploratory; production logic belongs in scripts and modules.
- Add focused tests for reusable functions.
