# vidqc — Video Artifact Detection System

Automated detection and classification of visual artifacts in AI-generated video clips.

## Version

v0.1.0 (in development - Milestones 1-5 complete)

## Features

- **Two artifact categories detected:**
  - **TEXT_INCONSISTENCY**: OCR-based text stability analysis (character mutations, jitter, erasure)
  - **TEMPORAL_FLICKER**: Frame-to-frame pixel/structural instability detection
- **Feature engineering + XGBoost classifier** (works on small datasets, CPU-only)
- **Explainable predictions** with frame-level evidence and probability scores
- **Scene-aware analysis** with automatic scene cut detection and per-segment analysis
- **Smart caching** for feature extraction (106x faster on repeated runs)
- **Progress bars** for real-time feedback during training
- **CLI-based workflow** for training, evaluation, and prediction

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

# Verify installation
uv run python -m vidqc --help
```

## Quick Start

### 1. Generate Synthetic Dataset

```bash
# Generate 30 synthetic clips for development
uv run python tools/make_dataset.py --output samples/ --count 30

# This creates:
# - samples/*.mp4 (video clips with 6 artifact types)
# - samples/labels.csv (ground truth labels)
```

**Clip types generated:**
- `text_*` — TEXT_INCONSISTENCY artifacts (character swaps, jitter, erasure)
- `flicker_*` — TEMPORAL_FLICKER artifacts (brightness oscillation, color shifts)
- `clean_text_*` — Clean clips with stable text
- `clean_notext_*` — Clean clips without text
- `clean_scenecut_*` — Clean clips with scene transitions
- `clean_blur_*` — Clean clips with motion blur

### 2. Train Classifier

```bash
# Train on synthetic dataset
uv run python -m vidqc.train --labels samples/labels.csv --output models/

# With custom cache location
uv run python -m vidqc.train --labels samples/labels.csv --output models/ --cache my_cache.pkl
```

**Training output:**
```
Extracting features:  50%|█████     | 15/30 clips [12:30<12:30, 50.0s/clip]
  OCR frames:  71%|███████   | 17/24 frames [00:54<00:22, 3.2s/frame]

Cross-validation:  60%|██████    | 3/5 folds [00:09<00:06, 3.0s/fold]
Fold 3 F1: 0.850

CV Mean F1: 0.847 ± 0.023
Training complete!
Model: models/model.json
```

**Output files:**
- `models/model.json` — Trained XGBoost model
- `models/training_manifest.json` — Training metadata and metrics
- `models/feature_importances.json` — Feature importance rankings
- `features_cache.pkl` — Cached features for fast retraining

### 3. Predict on a Single Clip

```bash
# Single clip prediction
uv run python -m vidqc.predict --video samples/text_001.mp4 --model models/model.json
```

**Output (JSON):**
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
    "timestamps": [],
    "bounding_boxes": [],
    "frames": [],
    "notes": "Probabilities: NONE=0.050, TEXT_INCONSISTENCY=0.920, TEMPORAL_FLICKER=0.030"
  }
}
```

### 4. Batch Prediction

```bash
# Batch prediction from CSV
uv run python -m vidqc.predict --batch samples/labels.csv --output predictions.csv --model models/model.json
```

**Output CSV columns:**
- `clip_path`, `artifact`, `category`, `confidence`
- `prob_none`, `prob_text`, `prob_flicker`

### 5. Evaluate Model

```bash
# Evaluate on labeled dataset
uv run python -m vidqc.eval --labels samples/labels.csv --model models/model.json
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

## Development

### Run Tests

```bash
# Full test suite
uv run pytest -v

# With coverage
uv run pytest --cov=vidqc

# Specific test file
uv run pytest tests/test_text_features.py -v
```

**Test coverage (50 tests):**
- Schema validation (9 tests)
- Frame extraction (7 tests)
- Text feature extraction (3 tests)
- Blur suppression (3 tests)
- No-text gate (3 tests)
- Temporal features (4 tests)
- Scene cut handling (5 tests)
- Scene cut with flicker (5 tests)
- Category constraint (7 tests)
- Determinism (5 tests)

### Lint

```bash
uv run ruff check vidqc/ tests/
```

### Type Check

```bash
uv run mypy vidqc/
```

## Project Structure

```
vidqc/
├── vidqc/                      # Main package
│   ├── features/               # Feature extraction modules
│   │   ├── text_inconsistency.py   # OCR-based text analysis
│   │   ├── temporal_flicker.py     # Frame instability detection
│   │   ├── scene_segmentation.py   # Scene cut detection
│   │   ├── feature_manifest.py     # Feature name registry
│   │   ├── extract.py              # Combined feature extraction
│   │   └── common.py               # Shared utilities (SSIM, IoU, etc.)
│   ├── models/                 # Classifier implementation
│   │   └── classifier.py           # XGBoost wrapper with constraints
│   ├── utils/                  # Utilities
│   │   ├── video.py                # Frame extraction
│   │   └── logging.py              # Logging configuration
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Prediction interface
│   ├── eval.py                 # Evaluation metrics
│   ├── schemas.py              # Pydantic output schemas
│   ├── config.py               # YAML configuration loader
│   └── cli.py                  # CLI interface
├── tools/                      # Dataset generation
│   └── make_dataset.py             # Synthetic clip generator
├── samples/                    # Video clips + labels.csv
├── models/                     # Trained models (after training)
├── tests/                      # Test suite (50 tests)
├── config.yaml                 # Configuration
└── pyproject.toml              # Dependencies
```

## Configuration

Key settings in `config.yaml`:

```yaml
# Frame extraction
video:
  fps: 5                    # Frames per second to sample
  max_dimension: 720        # Downscale to 720p

# Text analysis
text:
  ocr_skip_ssim_threshold: 0.995  # Skip OCR if frame similarity > threshold
  edit_distance_threshold: 2       # Min edit distance for artifact
  bbox_iou_threshold: 0.3          # Min IoU for region matching

# Temporal analysis
temporal:
  scene_cut_ssim_threshold: 0.4    # SSIM below this = scene cut
  post_cut_stability_window: 3     # Frames to verify stability after cut

# Classifier
classifier:
  binary_threshold: 0.5            # P(NONE) < threshold = artifact
  seed: 42                         # For reproducibility
```

## Classifier Details

### Decision Logic
1. **Binary decision**: `artifact = P(NONE) < binary_threshold`
2. **Category selection**: `argmax(P(TEXT), P(FLICKER))` when artifact detected
3. **Hard constraint**: No text detected → cannot predict TEXT_INCONSISTENCY
4. **Threshold fallback**: If both `P(TEXT) ≤ 0.1` and `P(FLICKER) ≤ 0.1` → NONE

### Training Features
- 5-fold stratified cross-validation (auto-adjusts for small datasets)
- Feature caching with modification-time invalidation
- Progress bars for real-time feedback
- Deterministic training (seed=42)

## Documentation

- **SPEC.md**: Full technical specification
- **CLAUDE.md**: Agent implementation instructions
- **PROGRESS.md**: Current milestone status

## Status

**Completed Milestones:**
- [x] Milestone 1: Scaffolding & Synthetic Data
- [x] Milestone 2: Frame Extraction & Core Utilities
- [x] Milestone 3: Text Feature Extraction
- [x] Milestone 4: Temporal Feature Extraction
- [x] Milestone 5: Classifier & Training Pipeline

**Current**: Milestone 6 - Real Data & Validation

See PROGRESS.md for details.

## Performance

- **Training time**: ~1-2 minutes per clip (OCR-bound)
- **Prediction time**: ~30-60 seconds per clip
- **Cache speedup**: 106x faster on repeated training runs
- **Model size**: ~100KB (XGBoost JSON)

## License

(To be determined)
