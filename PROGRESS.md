# vidqc Progress

## Current Milestone: 5

## Completed Milestones
- [x] Milestone 1 — 2026-02-11 — Scaffolding & synthetic dataset generator
- [x] Milestone 2 — 2026-02-11 — Frame extraction & core utilities
- [x] Milestone 3 — 2026-02-11 — Text feature extraction (OCR-based)
- [x] Milestone 4 — 2026-02-11 — Temporal feature extraction (scene segmentation)

## Current Status
Milestone 4 complete. Temporal flicker detection implemented with scene cut suppression.
All exit tests pass: scene cuts detected, segments analyzed separately, worst-case reported.
Combined feature vector: 43 features (29 text + 14 temporal).
Ready to start Milestone 5: Classifier & Training.
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
- Scene cut detection uses SSIM < 0.4 AND post-cut stability check (Milestone 4)
- Late-clip edge case handled: stop checking cuts when < stability window frames remain (Milestone 4)
- Multi-segment videos report worst-case segment features (Milestone 4)
- Combined feature manifest: 43 features (29 text + 14 temporal) (Milestone 4)

## Known Issues
(None)

## Test Results
Milestone 4 exit tests: 38 passed, 1 skipped (14 temporal + 24 previous)
