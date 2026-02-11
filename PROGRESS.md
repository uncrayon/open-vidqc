# vidqc Progress

## Current Milestone: 4

## Completed Milestones
- [x] Milestone 1 — 2026-02-11 — Scaffolding & synthetic dataset generator
- [x] Milestone 2 — 2026-02-11 — Frame extraction & core utilities
- [x] Milestone 3 — 2026-02-11 — Text feature extraction (OCR-based)

## Current Status
Milestone 3 complete. Text features extracted using 6-step pipeline with OCR skip optimization.
All exit tests pass: feature extraction on synthetic clips, no NaN/Inf, consistent 29-feature vector.
Ready to start Milestone 4: Temporal Feature Extraction.
No blockers.

## Key Decisions
- Fixed uint8 overflow in hue rotation by casting to int16 first (Milestone 1)
- Synthetic clips generate 6 types with balanced distribution (Milestone 1)
- Config uses singleton with lazy loading (Milestone 2)
- Frame extraction returns partial results on corruption (Milestone 2)
- Evidence contract enforced via Pydantic model_validator in after mode (Milestone 2)
- OCR skip threshold 0.995 (not 0.98) to catch small text mutations (Milestone 3)
- Substitution ratio uses python-Levenshtein editops decomposition (Milestone 3)
- Feature vector always 29 elements, zeros when no text (Milestone 3)
- EasyOCR singleton pattern prevents redundant model loading (Milestone 3)

## Known Issues
(None)

## Test Results
Milestone 3 exit tests: 24 passed, 1 skipped (9 text features + 3 blur suppression + 3 no-text gate)
