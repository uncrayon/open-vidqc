# vidqc Progress

## Current Milestone: 3

## Completed Milestones
- [x] Milestone 1 — 2026-02-11 — Scaffolding & synthetic dataset generator
- [x] Milestone 2 — 2026-02-11 — Frame extraction & core utilities

## Current Status
Milestone 2 complete. Frame extraction, config management, and schema validation working.
Ready to start Milestone 3: Text Feature Extraction.
No blockers.

## Key Decisions
- Fixed uint8 overflow in hue rotation by casting to int16 first (Milestone 1)
- Synthetic clips generate 6 types with balanced distribution (Milestone 1)
- Config uses singleton with lazy loading (Milestone 2)
- Frame extraction returns partial results on corruption (Milestone 2)
- Evidence contract enforced via Pydantic model_validator in after mode (Milestone 2)

## Known Issues
(None)

## Test Results
Milestone 2 exit tests: 15 passed, 1 skipped (9 schema + 6 frame extraction)
