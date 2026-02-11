# vidqc — Video Artifact Detection System

Automated detection and classification of visual artifacts in AI-generated video clips.

## Version

v0.1.0 (in development - Milestone 1 complete)

## Features

- Detects two artifact categories:
  - **TEXT_INCONSISTENCY**: OCR-based text stability analysis
  - **TEMPORAL_FLICKER**: Frame-to-frame pixel/structural instability
- Feature engineering + XGBoost classifier (works on small datasets)
- Explainable predictions with frame-level evidence
- CLI-based workflow

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
# - samples/*.mp4 (video clips)
# - labels.csv (ground truth labels)
```

### 2. Train Classifier (Coming in Milestone 5)

```bash
uv run python -m vidqc train --config config.yaml
```

### 3. Predict on a Clip (Coming in Milestone 5)

```bash
uv run python -m vidqc predict --input samples/text_001.mp4
```

### 4. Batch Prediction (Coming in Milestone 5)

```bash
uv run python -m vidqc batch --input samples/ --output predictions.jsonl
```

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

## Project Structure

```
vidqc/
├── vidqc/              # Main package
│   ├── features/       # Feature extraction modules
│   ├── models/         # Classifier implementation
│   ├── utils/          # Video I/O, logging
│   └── schemas.py      # Pydantic output schemas
├── tools/              # Dataset generation scripts
├── samples/            # Video clips
├── tests/              # Test suite
├── config.yaml         # Configuration
└── labels.csv          # Ground truth labels
```

## Documentation

- **SPEC.md**: Full technical specification
- **CLAUDE.md**: Agent implementation instructions
- **PROGRESS.md**: Current milestone status

## Status

**Current Milestone**: 1 - Scaffolding & Synthetic Data

See PROGRESS.md for details.

## License

(To be determined)
