# vidqc — Video Artifact Detection System

Automated detection and classification of visual artifacts in AI-generated video clips.

## Version

v0.1.0 (Milestones 1–7 complete)

## Features

- **Two artifact categories detected:**
  - **TEXT_INCONSISTENCY**: OCR-based text stability analysis (character mutations, jitter, erasure)
  - **TEMPORAL_FLICKER**: Frame-to-frame pixel/structural instability detection
- **Feature engineering + XGBoost classifier** (works on small datasets, CPU-only)
- **Explainable predictions** with frame-level evidence and probability scores
- **Scene-aware analysis** with automatic scene cut detection and per-segment analysis
- **Smart caching** for feature extraction (106x faster on repeated runs)
- **CLI** with four commands: train, eval, predict, batch
- **Docker** multi-stage build for deployment

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
# Clone repository
git clone <repository-url>
cd vidqc

# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev

# Install dataset generation tools (Sora/Veo-3 API clients)
uv sync --group dataset

# Verify installation
uv run python -m vidqc --help
```

## Quick Start

### 1. Generate Synthetic Dataset

```bash
uv run python tools/make_dataset.py --output samples/ --count 30 --seed 42
```

This creates 30 video clips across 6 types with ground truth labels in `samples/labels.csv`.
> Notice that `--seed` is optional.

### 2. Train

```bash
uv run python -m vidqc train --labels samples/labels.csv --output-dir models/
```

Output:
- `models/model.json` — Trained XGBoost model
- `models/training_manifest.json` — Training metadata and CV metrics
- `models/feature_importances.json` — Feature importance rankings
- `features_cache.pkl` — Cached features for fast retraining

### 3. Evaluate

```bash
uv run python -m vidqc eval --labels samples/labels.csv --model-dir models/
```

Checks §0.5 success criteria: binary recall >= 0.85, binary precision >= 0.70, per-category F1 >= 0.70.

### 4. Predict (Single Clip)

Any resolution MP4 is accepted as input. Videos taller than 720p are automatically downscaled (preserving aspect ratio) before processing. Videos at 720p or below are used as-is. The downscale target is configurable via `video.max_resolution` in `config.yaml`.

```bash
uv run python -m vidqc predict --input /path/to/your/video.mp4
```
This last assumes a trained model exists at `models/model.json` (which is the default). If the model is elsewhere then:
```bash
uv run python -m vidqc predict --input /path/to/your/video.mp4 --model-dir /path/to/models/
```
Outputs JSON to stdout:
```json
{
  "clip_id": "text_001.mp4",
  "artifact": true,
  "confidence": 0.92,
  "artifact_category": "TEXT_INCONSISTENCY",
  "class_probabilities": {
    "NONE": 0.05,
    "TEXT_INCONSISTENCY": 0.92,
    "TEMPORAL_FLICKER": 0.03
  },
  "evidence": {
    "notes": "Probabilities: NONE=0.050, TEXT_INCONSISTENCY=0.920, TEMPORAL_FLICKER=0.030"
  }
}
```
#### 4.1. Interactive BBox Review

Generate a self-contained HTML report for human QA of predicted evidence and OCR boxes:

```bash
uv run python tools/review_bboxes.py --input path/to/your/video --output path/to/output/report.html --model-dir path/to/models/ --config path/to/config.yaml
```
> Note: `--output`, `--model-dir` and `--config` are optional if you want to use the default paths.

This creates `reports/bbox_review_<clip>.html` with:
- frame-by-frame navigation
- evidence bbox overlays
- optional OCR bbox overlays
- prediction summary and raw JSON

### 5. Batch Predict

```bash
uv run python -m vidqc batch --input samples/ --output predictions.jsonl --model-dir models/
```

Processes all `.mp4` files in the directory, outputs one JSON per line to JSONL.

## CLI Reference

```
vidqc train
    --config PATH          Config file (default: config.yaml)
    --labels PATH          Labels CSV (default: samples/labels.csv)
    --output-dir PATH      Where to save models (default: models/)
    --cache / --no-cache   Cache extracted features (default: --cache)

vidqc eval
    --config PATH          Config file (default: config.yaml)
    --labels PATH          Labels CSV (default: samples/labels.csv)
    --model-dir PATH       Trained model directory (default: models/)

vidqc predict
    --input PATH           Single video file (required)
    --model-dir PATH       Model directory (default: models/)
    --config PATH          Config file (default: config.yaml)

vidqc batch
    --input PATH           Directory of .mp4 files (required)
    --output PATH          Output JSONL path (default: predictions.jsonl)
    --model-dir PATH       Model directory (default: models/)
    --config PATH          Config file (default: config.yaml)
```

## Fetching Real AI Clips

To generate real AI-generated clips for training/validation:

```bash
# OpenAI Sora
export OPENAI_API_KEY=sk-...
uv run python tools/fetch_sora.py --prompts tools/prompts.txt --output samples/ --count 15

# Google Veo-3
export GOOGLE_API_KEY=...
uv run python tools/fetch_veo3.py --prompts tools/prompts.txt --output samples/ --count 15
```

After fetching, label clips in `samples/labels.csv` and retrain.

## Feature Distribution Comparison

After adding real clips, compare feature distributions between synthetic and real data:

```bash
uv run python tools/compare_distributions.py \
    --labels samples/labels.csv \
    --output reports/distribution_comparison.json
```

## Docker

### Build

```bash
docker build -t vidqc .
```

### Run

```bash
# Predict on a single clip
docker run -v $(pwd)/samples:/data vidqc predict --input /data/text_001.mp4

# Batch predict
docker run -v $(pwd)/samples:/data vidqc batch --input /data --output /data/predictions.jsonl
```

## Feature Extraction

vidqc extracts **43 features** from each video clip:

### Text Features (29 features)
- Text presence gate (`has_text_regions`, `frames_with_text_ratio`)
- Per-region features: edit distance, confidence delta, bbox shift, SSIM crop, char count delta
- False positive suppression: global SSIM, confidence variance, substitution ratio
- Aggregations: mean, max, std for each metric

### Temporal Features (14 features)
- Frame-to-frame SSIM statistics (mean, min, std)
- Mean Absolute Error (MAE) statistics
- Histogram correlation
- Patch-based analysis (4x4 grid, 16 patches)
- Diff acceleration (second derivative of changes)
- Instability duration and spike isolation

### Scene-Aware Processing
- Automatic scene cut detection (SSIM < 0.4 + stability check)
- Per-segment feature extraction
- Worst-case segment reporting for multi-segment videos

## Classifier Details

### Decision Logic
1. **Binary decision**: `artifact = P(NONE) < binary_threshold`
2. **Category selection**: `argmax(P(TEXT), P(FLICKER))` when artifact detected
3. **Hard constraint**: No text detected -> cannot predict TEXT_INCONSISTENCY
4. **Threshold fallback**: If both `P(TEXT) <= 0.1` and `P(FLICKER) <= 0.1` -> NONE

### Training Features
- 5-fold stratified cross-validation (auto-adjusts for small datasets)
- Feature caching with modification-time invalidation
- Progress bars for real-time feedback
- Deterministic training (seed=42)

## Development

### Run Tests

```bash
uv run pytest -v
```

### Lint

```bash
uv run ruff check vidqc/ tests/
```

### Type Check

```bash
uv run mypy vidqc/
```

### Full Development Workflow

```bash
# Setup
uv sync
uv sync --group dev --group dataset

# Generate synthetic dataset
uv run python tools/make_dataset.py --output samples/ --count 30

# Train
uv run python -m vidqc train

# Evaluate
uv run python -m vidqc eval

# Predict
uv run python -m vidqc predict --input samples/text_001.mp4

# Batch
uv run python -m vidqc batch --input samples/ --output predictions.jsonl

# Run tests
uv run pytest -v

# Lint
uv run ruff check vidqc/ tests/
```

## Project Structure

```
vidqc/
├── vidqc/                          # Main package
│   ├── features/                   # Feature extraction modules
│   │   ├── text_inconsistency.py   # OCR-based text analysis
│   │   ├── temporal_flicker.py     # Frame instability detection
│   │   ├── scene_segmentation.py   # Scene cut detection
│   │   ├── feature_manifest.py     # Feature name registry
│   │   ├── extract.py              # Combined feature extraction
│   │   └── common.py               # Shared utilities (SSIM, IoU, etc.)
│   ├── models/                     # Classifier implementation
│   │   └── classifier.py           # XGBoost wrapper with constraints
│   ├── utils/                      # Utilities
│   │   ├── video.py                # Frame extraction
│   │   └── logging.py              # Logging configuration
│   ├── train.py                    # Training pipeline
│   ├── predict.py                  # Prediction interface
│   ├── eval.py                     # Evaluation & success criteria
│   ├── schemas.py                  # Pydantic output schemas
│   ├── config.py                   # YAML configuration loader
│   ├── cli.py                      # CLI interface (4 commands)
│   └── __main__.py                 # Entry point
├── tools/                          # Dataset & analysis tools
│   ├── make_dataset.py             # Synthetic clip generator
│   ├── fetch_sora.py               # OpenAI Sora API client
│   ├── fetch_veo3.py               # Google Veo-3 API client
│   ├── compare_distributions.py    # Synthetic vs. real comparison
│   └── prompts.txt                 # Curated generation prompts
├── samples/                        # Video clips + labels.csv
├── models/                         # Trained models (after training)
├── tests/                          # Test suite (50 tests)
├── config.yaml                     # Configuration
├── Dockerfile                      # Multi-stage Docker build
├── SPEC.md                         # Technical specification
├── PRODUCTION_PLAN.md              # AWS deployment plan
└── pyproject.toml                  # Dependencies
```

## Configuration

Key settings in `config.yaml`:

```yaml
video:
  sample_fps: 5              # Frames per second to sample
  max_resolution: 720        # Downscale to 720p

text:
  ocr_skip_ssim_threshold: 0.995  # Skip OCR if frame similarity > threshold
  edit_distance_threshold: 2       # Min edit distance for artifact
  bbox_iou_threshold: 0.3          # Min IoU for region matching

temporal:
  scene_cut:
    cut_ssim_threshold: 0.4        # SSIM below this = scene cut
    post_cut_stability_window: 3   # Frames to verify stability after cut

model:
  binary_threshold: 0.5            # P(NONE) < threshold = artifact
  seed: 42                         # For reproducibility
```

## Performance

- **Training time**: ~1-2 minutes per clip (OCR-bound)
- **Prediction time**: ~75-140 seconds per clip on CPU (EasyOCR is the bottleneck; text-heavy clips ~136s, flicker-only ~90s, clean ~75s)
- **Cache speedup**: 106x faster on repeated training runs
- **Model size**: ~100KB (XGBoost JSON)
- **Target latency**: < 60s per clip (aspirational — requires GPU or OCR optimization)

## Documentation

- **SPEC.md**: Full technical specification
- **CLAUDE.md**: Agent implementation instructions
- **PROGRESS.md**: Current milestone status
- **PRODUCTION_PLAN.md**: AWS deployment architecture

## License

(To be determined)
