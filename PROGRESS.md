# vidqc Progress

## Current Milestone: 7 (Complete)

## Completed Milestones
- [x] Milestone 1 — 2026-02-11 — Scaffolding & synthetic dataset generator
- [x] Milestone 2 — 2026-02-11 — Frame extraction & core utilities
- [x] Milestone 3 — 2026-02-11 — Text feature extraction (OCR-based)
- [x] Milestone 4 — 2026-02-11 — Temporal feature extraction (scene segmentation)
- [x] Milestone 5 — 2026-02-11 — Classifier & Training pipeline
- [x] Milestone 6 — 2026-02-11 — Real data tooling (code deliverables; real data pending API keys)
- [x] Milestone 7 — 2026-02-11 — CLI, Docker & Documentation

## Current Status
All code milestones (1-7) complete. Milestone 6 real-data steps pending API keys.
- CLI: 4 commands (train, eval, predict, batch) working end-to-end
- Docker: image builds (3.02GB), predict verified in container
- Latency: avg 100s/clip (target 60s); EasyOCR on CPU is bottleneck
- All 50 tests pass, ruff check clean

## Key Decisions
- Fixed uint8 overflow in hue rotation by casting to int16 first (M1)
- Config uses singleton with lazy loading (M2)
- Evidence contract enforced via Pydantic model_validator in after mode (M2)
- OCR skip threshold 0.995 (not 0.98) to catch small text mutations (M3)
- Substitution ratio uses python-Levenshtein editops decomposition (M3)
- EasyOCR singleton pattern prevents redundant model loading (M3)
- Scene cut: SSIM < 0.4 AND post-cut stability check (M4)
- Combined feature manifest: 43 features (29 text + 14 temporal) (M4)
- Binary decision uses P(NONE) < threshold, not argmax (M5)
- Hard constraint: has_text=0 → cannot predict TEXT_INCONSISTENCY (M5)
- Threshold fallback: P(TEXT)≤0.1 AND P(FLICKER)≤0.1 → NONE (M5)
- eval.py: §0.5 criteria (binary recall≥0.85, precision≥0.70, F1≥0.70, CV std<0.15) (M6)
- CLI uses argparse.BooleanOptionalAction for --cache/--no-cache (M7)
- Batch outputs JSONL (one JSON per line) (M7)
- Docker: libgl1 replaces obsolete libgl1-mesa-glx (M7)
- Latency exceeds 60s target — EasyOCR at ~3-5s/frame on CPU is bottleneck (M7)

## Known Issues
- Latency: avg 100s/clip vs 60s target. Needs GPU or reduced frame count for production.
- Docker image: 3.02GB (just over 3GB target). EasyOCR models are the main size contributor.
- Milestone 6 real data steps require OPENAI_API_KEY / GOOGLE_API_KEY.

## Test Results
Milestone 7: 50 passed, 1 skipped, ruff check clean
