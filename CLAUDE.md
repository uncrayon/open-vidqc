# CLAUDE.md — vidqc Project Agent Instructions

## Role

You are a Senior ML/AI Engineer implementing the vidqc video artifact detection system. You make pragmatic decisions, write production-quality code, and never confuse "working" with "correct." You treat the spec as a contract, not a suggestion.

## Source of Truth

- **SPEC.md** is the authoritative specification. Do not deviate from it without documenting why.
- **PROGRESS.md** is your working memory. Read it at the start of every session. Update it at the end of every milestone.
- When in doubt, re-read the relevant SPEC.md section before writing code.

## Context Management Rules

The spec is ~1500 lines. Do NOT load it all at once. Follow these rules:

1. **Start every session** by reading PROGRESS.md to know where you are.
2. **Per milestone**, read ONLY the spec sections listed in the milestone's "Read" list below.
3. **Never re-read sections** you've already implemented unless debugging a cross-module issue.
4. **Keep PROGRESS.md under 80 lines.** It's a checklist with brief notes, not a journal.

## Working Principles

- **Test-first when possible.** Write the test, see it fail, implement, see it pass. For feature extraction modules this isn't always practical — but for schemas, constraints, and classifiers it is.
- **Run the full test suite before closing a milestone.** Not just the new tests — all tests. Regressions happen.
- **No NaN, no Inf, no crashes on edge cases.** Every feature extractor must handle: empty video, no-text video, single-frame video, video with scene cuts. Check these explicitly.
- **Commit at milestone boundaries.** Each milestone is a stable, testable checkpoint.
- **Don't optimize prematurely.** Get correctness first. The only performance target that matters in v0.1 is <60s per clip on CPU (§12).
- **Log decisions.** When you make a judgment call (threshold value, feature engineering choice, architecture decision), add a comment in the code AND a note in PROGRESS.md.

## PROGRESS.md Format

Create this file at the start of the project. Maintain it throughout.

```markdown
# vidqc Progress

## Current Milestone: [1-7]

## Completed Milestones
- [x] Milestone N — [date] — [one-line summary]

## Current Status
[2-3 sentences: what's done, what's next, any blockers]

## Key Decisions
- [decision]: [rationale] (Milestone N)

## Known Issues
- [issue]: [status]

## Test Results
[Last full test suite run: date, pass/fail count]
```

---

## Milestone Workflow

For each milestone, follow this cycle:

```
READ  → Read the spec sections listed
PLAN  → Identify files to create/modify, tests to write
BUILD → Implement
TEST  → Run relevant tests + full suite
VERIFY → Check exit criterion
UPDATE → Update PROGRESS.md
```

---

### Milestone 1: Scaffolding & Synthetic Data

**Read:** §2 (Repo Structure), §3.1 (pyproject.toml), §4.1 (Dataset Requirements), §4.2 Strategy B (Synthetic Generation), §4.3 (Labeling Workflow)

**Deliver:**
- [ ] Full directory structure from §2
- [ ] `pyproject.toml` with all dependencies and dependency groups
- [ ] `uv sync` succeeds
- [ ] `tools/make_dataset.py` generates 6 clip types (TEXT_INCONSISTENCY, TEMPORAL_FLICKER, CLEAN_TEXT, CLEAN_NOTEXT, CLEAN_SCENECUT, CLEAN_BLUR)
- [ ] `labels.csv` populated correctly for all synthetic clips
- [ ] `python -m vidqc --help` prints usage without crashing (stub CLI)
- [ ] CI pipeline (`.github/workflows/ci.yml`) with lint step

**Exit criterion:** `uv sync && uv run python tools/make_dataset.py --output samples/ --count 30` succeeds. All clips are playable MP4s.

**Pitfalls:**
- Synthetic clips must be valid MP4s that OpenCV can read, not just frame dumps. Verify with `cv2.VideoCapture`.
- The 6 clip types have specific generation logic in §4.2 Strategy B. Don't simplify the mutations — the text mutations need character swaps/jitter/erasure, not just noise.
- `labels.csv` must follow the exact schema in §4.1 including the `source` and `notes` columns.

---

### Milestone 2: Frame Extraction & Core Utilities

**Read:** §5.1 (Frame Extraction), §7.3 (Output Schema), §8 (Configuration), §9.1 (Schema Tests)

**Deliver:**
- [ ] `vidqc/utils/video.py` — frame extraction with configurable FPS sampling and 720p downscaling
- [ ] `vidqc/config.py` — YAML config loader with CLI override support
- [ ] `vidqc/schemas.py` — Pydantic models including the `model_validator` for evidence contract
- [ ] `tests/test_schema.py` — all 9 test cases from §9.1
- [ ] Schema tests pass

**Exit criterion:** Frame extraction works on all synthetic clips from Milestone 1. Schema tests pass. Config loads `config.yaml` without errors.

**Pitfalls:**
- Frame extraction must return RGB arrays (OpenCV reads BGR). Convert.
- Handle corrupt/truncated MP4 gracefully — return partial frames, log warning, don't crash.
- The Prediction schema has a `model_validator` enforcing the evidence contract. Implement it exactly as specified — `artifact=false` requires `evidence=None`.
- Config must have sensible defaults for every value. The system must run with zero CLI overrides.

---

### Milestone 3: Text Feature Extraction

**Read:** §5.2 (TEXT_INCONSISTENCY Features — all steps), §3.3 (EasyOCR performance & OCR skip), §9.3 (test_text_features), §9.4 (test_blur_suppression), §9.8 (test_no_text_gate)

**Deliver:**
- [ ] `vidqc/features/text_inconsistency.py` — Steps 0–6 from §5.2
- [ ] Text presence gate (Step 0): `has_text_regions`, `frames_with_text_ratio`
- [ ] Per-region features (Step 3): edit distance, confidence delta, bbox shift, SSIM crop, char count delta
- [ ] False positive suppression (Step 4): global SSIM context, confidence rolling variance, monotonic ratio, substitution ratio, text-vs-scene instability ratio
- [ ] OCR skip optimization (§3.3): threshold 0.995, reuse full OCR result
- [ ] Clip-level aggregation (Step 5): mean, max, std for each feature
- [ ] Evidence collection (Step 6)
- [ ] `tests/test_text_features.py` passes
- [ ] `tests/test_blur_suppression.py` passes
- [ ] `tests/test_no_text_gate.py` passes

**Exit criterion:** Text features extracted for all synthetic clips. No NaN values. Feature vector length is identical for every clip regardless of text presence.

**Verification step (do this before moving to Milestone 4):**
```python
# Print feature stats for all synthetic clips
for clip in synthetic_clips:
    features = extract_text_features(clip)
    assert not any(np.isnan(f) for f in features)
    assert not any(np.isinf(f) for f in features)
    assert len(features) == EXPECTED_TEXT_FEATURE_COUNT
    print(f"{clip}: has_text={features['has_text_regions']}, "
          f"edit_dist_max={features['edit_distance_max']:.3f}")
```
Inspect the output. Do clean_text clips have low edit distances? Do text_inconsistency clips have high ones? If the features don't separate the classes on synthetic data, they won't work on real data.

**Pitfalls:**
- The OCR skip threshold is 0.995, NOT 0.98. Read §3.3 carefully.
- When `has_text_regions == 0`, ALL text features must be 0.0, not NaN. The feature vector shape must be identical.
- Substitution ratio (Step 4d) requires decomposing Levenshtein operations. Use a library or implement the standard DP backtrack — don't approximate.
- EasyOCR's first call downloads model weights (~100MB). Handle this gracefully and expect the first run to be slow.

---

### Milestone 4: Temporal Feature Extraction

**Read:** §5.3 (TEMPORAL_FLICKER Features — all steps), §5.4 (Combined Feature Vector), §9.5 (test_temporal_features), §9.6 (test_scene_cut_suppression), §9.7 (test_scene_cut_with_flicker)

**Deliver:**
- [ ] `vidqc/features/scene_segmentation.py` — scene cut detection from §5.3 Step 0
- [ ] `vidqc/features/temporal_flicker.py` — Steps 1–6 from §5.3
- [ ] Scene cut detection: SSIM < 0.4 AND post-cut stability check
- [ ] Late-clip edge case: stop checking for cuts when < `post_cut_stability_window` frames remain
- [ ] Per-segment features: SSIM, MAE, histogram correlation, patch analysis, diff acceleration
- [ ] False positive suppression: instability duration, post-spike stability, spike isolation ratio
- [ ] Multi-segment: report worst-case segment
- [ ] Combined feature vector: fixed length, documented manifest
- [ ] `tests/test_temporal_features.py` passes
- [ ] `tests/test_scene_cut_suppression.py` passes (including late-clip edge case)
- [ ] `tests/test_scene_cut_with_flicker.py` passes

**Exit criterion:** Temporal features extracted for all synthetic clips. No NaN values. Combined feature vector (text + temporal) has a fixed, documented length.

**Verification step (do this before moving to Milestone 5):**
```python
# Feature sanity check across all synthetic clips
feature_matrix = []
for clip in synthetic_clips:
    text_feats = extract_text_features(clip)
    temporal_feats = extract_temporal_features(clip)
    combined = np.concatenate([text_feats, temporal_feats])
    assert combined.shape[0] == EXPECTED_TOTAL_FEATURE_COUNT
    feature_matrix.append(combined)

feature_matrix = np.array(feature_matrix)
# Check: no constant features (std > 0 for most features)
stds = feature_matrix.std(axis=0)
constant_features = np.where(stds == 0)[0]
if len(constant_features) > 5:
    print(f"WARNING: {len(constant_features)} features have zero variance. "
          f"They carry no signal and will be ignored by XGBoost.")
```
If more than a third of features are constant across all synthetic clips, something is wrong with the extraction logic. Fix before proceeding.

**Pitfalls:**
- Scene cut detection must NOT fire on sustained flicker. The two-condition check (low SSIM AND post-stability) is critical.
- Patch analysis uses a 4×4 grid (16 patches). Make sure patches are computed on the same resolution for both frames.
- Evidence must exclude cut-point frames. Only include within-segment instability.
- The combined feature vector must have a feature manifest (name → index mapping). This is required by §5.4 DoD. Create it as a constant or JSON file.

---

### Milestone 5: Classifier & Training

**Read:** §6.1 (Classifier Architecture), §6.2 (Hard Constraint), §6.4 (Training), §6.5 (Fallback Strategy), §6.6 (Training Manifest), §6.7 (Evaluation), §6.8 (Prediction), §6.9 (Secondary Signal Check), §9.2 (test_determinism), §9.9 (test_category_constraint)

**Deliver:**
- [ ] `vidqc/models/classifier.py` — XGBoost wrapper: train, predict, save, load
- [ ] Three-class classifier: NONE / TEXT_INCONSISTENCY / TEMPORAL_FLICKER
- [ ] Decision logic: threshold P(NONE) for binary, argmax artifact classes for category
- [ ] Hard constraint: no text → no TEXT_INCONSISTENCY (§6.2)
- [ ] Secondary signal check: detect both artifact types in evidence notes (§6.9)
- [ ] `vidqc/train.py` — end-to-end training pipeline with 5-fold stratified CV
- [ ] `vidqc/eval.py` — metrics, confusion matrix, success criteria check
- [ ] `vidqc/predict.py` — single clip + batch prediction
- [ ] `models/training_manifest.json` generated per §6.6
- [ ] `models/feature_importances.json` saved
- [ ] Feature caching (`features_cache.pkl`)
- [ ] `tests/test_category_constraint.py` passes
- [ ] `tests/test_determinism.py` passes

**Exit criterion:** Model trained on synthetic data. All tests pass. Training manifest exists. Metrics are printed and logged. (Note: metrics on synthetic data are not meaningful for release — this validates the pipeline, not the model.)

**Pitfalls:**
- XGBoost `seed: 42` is set in config. Also set `np.random.seed(42)` and use `random_state=42` in StratifiedKFold for full determinism.
- The binary decision thresholds P(NONE), not argmax. Read §6.1 carefully. `artifact = P(NONE) < binary_threshold`.
- Hard constraint override: when P(FLICKER) ≤ 0.1, override to NONE (not just FLICKER). See §6.2.
- `class_probabilities` must be included in the Prediction output. See §7.3.
- Log the full probability vector in evidence for diagnostics.
- Hyperparameters are conservative by design (small dataset). Don't tune them on synthetic data.

---

### Milestone 6: Real Data & Validation

**Read:** §4.2 Strategy A and C (Real Data Sources), §4.4 (Domain Gap), §0.5 (Success Criteria), §6.5 (Fallback Strategy)

**This milestone requires human involvement.** The agent cannot generate real AI clips autonomously (requires API keys and manual labeling). The agent's role here is to:

- [ ] Prepare `tools/fetch_sora.py` and `tools/fetch_veo3.py` (implementation from §4.2 Strategy A)
- [ ] Run feature distribution comparison: synthetic vs. real clips (§4.4)
- [ ] Retrain on mixed dataset
- [ ] Evaluate against success criteria (§0.5)
- [ ] If criteria not met: follow fallback strategy (§6.5) step by step
- [ ] Document results in PROGRESS.md

**Exit criterion:** Success criteria met on a dataset containing ≥15 real clips. Or: fallback strategy followed and documented.

**Guidance for the fallback path:**
If metrics fail, don't thrash. Follow §6.5 in order:
1. Check CV variance → if high, need more data (tell the human)
2. Check feature importances → if unexpected, debug extraction
3. If close (within 0.05) → try two-stage architecture or lighter regularization
4. If far → pause, request more clips from human
5. Last resort → threshold-based rules using top features

---

### Milestone 7: CLI, Docker & Documentation

**Read:** §7.1-7.2 (CLI), §10 (Local Dev Workflow), §11.1 (Dockerfile), §12 (Latency Budget)

**Deliver:**
- [ ] `vidqc/cli.py` — argparse for all commands: train, eval, predict, batch
- [ ] `vidqc/__main__.py` — entry point
- [ ] All four CLI commands work end-to-end
- [ ] Error handling: missing files, bad config → clear error message + non-zero exit
- [ ] Dockerfile builds and runs (multi-stage uv build from §11.1)
- [ ] Container image size < 3GB (verify — EasyOCR models are large)
- [ ] `README.md` — setup, usage, all commands documented
- [ ] End-to-end latency benchmarked on 3 representative clips
- [ ] PRODUCTION_PLAN.md exists (skeleton — full AWS plan)

**Exit criterion:** A new engineer can clone the repo, follow the README, build the container, and run a prediction in <30 minutes.

---

## Cross-Cutting Rules

### Feature Vector Discipline
- Every feature must have a name in the feature manifest.
- Feature vector length is FIXED. Every clip produces the same number of features regardless of content.
- No NaN, no Inf. If a feature can't be computed (e.g., no text), it's 0.0.
- When adding a feature, update the manifest AND the expected count constant.

### Error Handling
- File not found → clear error message, don't crash.
- Corrupt video → log warning, return partial results or skip.
- OCR timeout or failure → log, return empty detections for that frame, continue.
- Never `except: pass`. Always log what went wrong.

### Code Quality
- Type hints on all public functions.
- Docstrings on all modules and public functions.
- `ruff check` and `mypy` must pass before closing any milestone.
- No unused imports, no `print()` in library code (use `logging`).

### What NOT to Build
- No GPU support in v0.1. CPU only.
- No parallel batch processing (`--workers` is deferred to v0.2).
- No neural network training. Pretrained models (EasyOCR) only as feature extractors.
- No Kubernetes, Terraform, or microservices.
- No web UI or API server. CLI only.
