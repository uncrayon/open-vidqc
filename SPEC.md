# vidqc — Video Artifact Detection System

## Technical Specification v0.2

---

## 0. Product Context

### 0.1 Problem Statement

AI video generators (Sora, Veo-3, Kling, etc.) produce clips with visual artifacts that degrade perceived quality. Teams publishing, curating, or benchmarking AI-generated video need an automated, explainable way to detect these artifacts before content reaches downstream consumers.

### 0.2 Target Users

| User | Workflow | What they need from vidqc |
|------|----------|---------------------------|
| **AI video platform engineer** | Automated quality gate in a generation pipeline. Clip is generated → vidqc runs → clip is published or rejected. | High recall (don't let bad clips through). False positives are tolerable — they trigger re-generation, which is cheap. |
| **AI research team** | Benchmarking artifact rates across models/prompts. Run vidqc on batches of clips, aggregate results. | Consistent, deterministic scoring. Confidence scores that are comparable across clips. |
| **Content ops / QA analyst** | Manual review triage. vidqc flags suspicious clips → human reviews flagged subset only. | High precision (don't waste human time on false positives). Evidence output must be actionable — specific frames + bounding boxes. |

### 0.3 Primary Optimization Target

**Recall-first for v0.1.** The cost of a missed artifact reaching production is higher than the cost of a false positive triggering a human review or re-generation. We target ≥ 0.85 recall on the binary detection task (artifact yes/no), accepting precision as low as 0.70.

This can be adjusted per-deployment via the `binary_threshold` config parameter, which thresholds on P(NONE) — the model's estimated probability that no artifact is present. If P(NONE) < `binary_threshold`, the clip is flagged as containing an artifact. Lower threshold → higher precision (only flag when model is very confident). Higher threshold → higher recall (flag even uncertain clips). The default (0.5) balances both; users who want a strict quality gate should raise it to 0.6–0.7 to catch more artifacts at the cost of more false positives.

### 0.4 Scope & Deliberate Exclusions

**v0.1 covers two artifact categories:**

| ID | Category | Why included first |
|----|----------|-------------------|
| 1 | `TEXT_INCONSISTENCY` | Detectable with OCR + frame comparison. No spatial reasoning required. High frequency in current AI generators. |
| 2 | `TEMPORAL_FLICKER` | Detectable with pixel/structural similarity. Classical CV features are sufficient. Highly visible to end users. |

**Known artifact types NOT covered in v0.1:**

| Artifact | Why deferred |
|----------|-------------|
| Anatomical deformation (hands, faces) | Requires spatial/pose estimation models — not feasible with feature engineering on 30 clips. |
| Physics violations (floating objects, impossible motion) | Requires scene understanding / common-sense reasoning. Research-grade problem. |
| Object persistence failure (objects appearing/disappearing) | Requires object tracking across frames. Significant additional pipeline complexity. |
| Spatial coherence breakdown (warping, morphing) | Requires dense optical flow or learned representations. Beyond classical CV. |

**Extensibility path:** The pipeline architecture (feature extraction → tabular features → classifier) supports adding new categories by writing a new feature extractor module in `vidqc/features/` and retraining. Each new category requires its own labeled samples and feature engineering. The classifier, schema, and CLI are designed to be category-count agnostic.

### 0.5 Success Criteria (v0.1 Release Gate)

The following must ALL be true before v0.1 is declared shippable:

| # | Criterion | Measurement |
|---|-----------|-------------|
| 1 | Binary recall ≥ 0.85 on real AI-generated clips (not synthetic-only) | 5-fold stratified CV mean |
| 2 | Binary precision ≥ 0.70 | 5-fold stratified CV mean |
| 3 | Per-category F1 ≥ 0.70 for both TEXT_INCONSISTENCY and TEMPORAL_FLICKER | 5-fold stratified CV mean |
| 4 | Cross-validation standard deviation < 0.15 for all metrics | Across 5 folds |
| 5 | All tests pass (§9) | `pytest -v` exits 0 |
| 6 | End-to-end latency < 60s per clip (10s, 720p, CPU) | Benchmarked on 3 representative clips |
| 7 | Deterministic output (same input → same output) | `test_determinism.py` |

**If criteria 1–4 are not met** after training on the full dataset, the fallback strategy is documented in §6.5.

---

## 1. Technical Overview

**Goal:** Build a CLI-based ML pipeline that detects and classifies visual artifacts in short AI-generated video clips.

**Artifact Categories (2):**

| ID | Category | Detection Strategy |
|----|----------|-------------------|
| 1 | `TEXT_INCONSISTENCY` | OCR-based frame-to-frame text stability analysis |
| 2 | `TEMPORAL_FLICKER` | Pixel/structural similarity spikes between consecutive frames |

**Core Principle:** Feature engineering + classical ML. No neural network training on 30 videos. Use pretrained models only as feature extractors (OCR, embeddings), then classify with interpretable, small-data-friendly models.

---

## 2. Repository Structure

```
vidqc/
├── vidqc/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry (python -m vidqc)
│   ├── cli.py                   # argparse definitions
│   ├── config.py                # Config loader (YAML + CLI overrides)
│   ├── train.py                 # Train classifier on extracted features
│   ├── eval.py                  # Evaluate model on labeled data
│   ├── predict.py               # Single clip + batch prediction
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_inconsistency.py   # OCR-based feature extraction
│   │   ├── temporal_flicker.py     # Frame-diff feature extraction
│   │   ├── scene_segmentation.py   # Scene cut detection, clip segmentation
│   │   └── common.py              # Shared frame utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py          # XGBoost wrapper (train/predict/save/load)
│   ├── schemas.py               # Pydantic models for output validation
│   └── utils/
│       ├── __init__.py
│       ├── video.py             # Frame extraction, video I/O
│       └── logging.py           # Logging configuration
├── tools/
│   ├── make_dataset.py          # Synthetic dataset generator
│   ├── fetch_sora.py            # Download/generate clips via Sora API
│   └── fetch_veo3.py            # Download/generate clips via Veo-3 API
├── samples/                     # 20-40 video clips
│   ├── text_001.mp4
│   ├── flicker_001.mp4
│   ├── clean_001.mp4
│   └── ...
├── tests/
│   ├── test_schema.py              # Output JSONL schema validation
│   ├── test_determinism.py         # Same input → same output
│   ├── test_text_features.py       # Unit test: OCR instability metric
│   ├── test_blur_suppression.py    # False positive: blur ≠ text artifact
│   ├── test_temporal_features.py
│   ├── test_scene_cut_suppression.py   # False positive: scene cut ≠ flicker
│   ├── test_scene_cut_with_flicker.py  # Scene cut + flicker in same clip
│   ├── test_no_text_gate.py        # No text → all text features zero, not NaN
│   └── test_category_constraint.py # No text → can't predict TEXT_INCONSISTENCY
├── config.yaml                  # Default configuration
├── labels.csv                   # Ground truth labels
├── pyproject.toml               # Project definition (PEP 621)
├── uv.lock
├── Dockerfile
├── README.md
├── PRODUCTION_PLAN.md           # Scaling to 500 concurrent executions
└── .github/
    └── workflows/
        └── ci.yml               # Lint + test on push
```

**Definition of Done — Repository Structure:**
- [ ] All directories and placeholder files exist
- [ ] `uv sync` succeeds without errors
- [ ] `python -m vidqc --help` prints usage without crashing
- [ ] CI pipeline (`.github/workflows/ci.yml`) runs lint + test on push

---

## 3. Tooling & Dependencies

### 3.1 Project Management — uv

```toml
[project]
name = "vidqc"
version = "0.1.0"
description = "Video artifact detection for AI-generated clips"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "opencv-python-headless>=4.9",
    "easyocr>=1.7",
    "scikit-image>=0.22",
    "xgboost>=2.0",
    "scikit-learn>=1.4",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "Pillow>=10.2",
    "tqdm>=4.66",
]

[project.scripts]
vidqc = "vidqc.__main__:main"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.3", "mypy>=1.8"]
dataset = ["openai>=1.12", "google-genai>=1.0", "requests>=2.31"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3.2 Dependency Rationale

| Dependency | Why |
|---|---|
| `opencv-python-headless` | Frame extraction, image ops. Headless = no GUI deps (server-friendly). |
| `easyocr` | Pretrained OCR. Returns bounding boxes + confidence per detection. DL-based internally but we don't train it. |
| `scikit-image` | SSIM computation (`structural_similarity`). |
| `xgboost` | Classifier. Works well on small tabular data, handles feature importance natively. |
| `pydantic` | Output schema validation. Guarantees JSONL contract. |
| `pyyaml` | Config file parsing. |

### 3.3 EasyOCR Performance Characteristics & Contingency

EasyOCR is the expected performance bottleneck. Engineers should be aware of these numbers when implementing and testing.

**Expected latency (CPU, single-threaded):**

| Frame Resolution | EasyOCR per frame | 50 frames (5fps × 10s clip) |
|-----------------|-------------------|------------------------------|
| 720p (1280×720) | 0.5–1.5s | 25–75s |
| 1080p (1920×1080) | 1.0–2.5s | 50–125s |

**Mitigation strategies (implement in order of priority):**

1. **Frame-level OCR skip (implement in v0.1):** Before running OCR on frame N, compute SSIM between frame N and frame N-1. If SSIM > 0.995, skip OCR for frame N and reuse the previous frame's full OCR result — whether that result was "no text detected" or a list of text detections with bounding boxes and content. This avoids re-running OCR on near-identical frames regardless of whether they contain text. The threshold is set at 0.995 (not 0.98) to avoid skipping frames where a small text region mutated on a stable background — a character swap in a ~100×30px text region on a 720p frame changes ~0.3% of pixels, which drops SSIM below 0.995 but not below 0.98. This ensures the skip optimization never masks the exact artifact we're trying to detect.
2. **Resolution downscale (implement in v0.1):** Downscale frames to 720p before OCR if source is higher resolution. OCR accuracy loss is negligible for the text sizes we care about.
3. **PaddleOCR swap (contingency, not v0.1):** If EasyOCR latency is unacceptable even with mitigations, PaddleOCR is a drop-in alternative with 2–5× faster CPU inference. The feature extraction interface (`list of (bbox, text, confidence)`) is the same. Document the swap path but don't implement unless needed.

**Thread safety note (v0.2):** EasyOCR is NOT safe for Python multiprocessing with `fork`. If parallel batch processing is added in a future version, use `spawn` multiprocessing context or process-level isolation (one EasyOCR reader per worker process, initialized after fork). OpenCV `VideoCapture` has similar constraints. v0.1 runs single-threaded only.

**Definition of Done — Tooling & Dependencies:**
- [ ] `uv sync` completes on a clean environment (Python 3.10+)
- [ ] `uv sync --group dev --group dataset` includes all optional groups
- [ ] `ruff check vidqc/ tests/` passes with zero violations
- [ ] `mypy vidqc/` passes with zero errors
- [ ] EasyOCR initializes successfully on CPU (`easyocr.Reader(["en"], gpu=False)`)
- [ ] Latency of EasyOCR on a single 720p frame measured and documented in README

---

## 4. Dataset

### 4.1 Requirements

- **Total clips:** 20–40 (aim for 35)
- **Minimum TEXT_INCONSISTENCY:** 10 clips
- **Minimum TEMPORAL_FLICKER:** 8 clips
- **Clean (no artifact):** 10–14 clips, with intentional diversity:
  - At least **4 clips with NO text** (nature scenes, abstract motion)
    — trains `has_text_regions=0` doesn't mean `artifact=0`
  - At least **2 clips with stable, readable text**
    — trains that text presence alone isn't an artifact signal
  - At least **3 clips with hard scene cuts but NO flicker**
    — trains the scene cut suppression (most critical for avoiding false positives)
  - At least **1 clip with blur/defocus transitions**
    — trains the blur suppression for text features
- **Format:** MP4, 3–10 seconds, 720p preferred
- **Label file:** `labels.csv`

```csv
clip_id,path,artifact,artifact_category,source,notes
text_001,samples/text_001.mp4,true,TEXT_INCONSISTENCY,sora,
flicker_001,samples/flicker_001.mp4,true,TEMPORAL_FLICKER,veo3,
flicker_notext_001,samples/flicker_notext_001.mp4,true,TEMPORAL_FLICKER,synthetic,
clean_001,samples/clean_001.mp4,false,NONE,sora,
clean_notext_001,samples/clean_notext_001.mp4,false,NONE,synthetic,
clean_text_001,samples/clean_text_001.mp4,false,NONE,veo3,
clean_scenecut_001,samples/clean_scenecut_001.mp4,false,NONE,synthetic,
clean_blur_001,samples/clean_blur_001.mp4,false,NONE,synthetic,
```

**Note:** The `artifact` field uses `true`/`false` strings to align with the Pydantic `bool` type in the output schema (§7.3). The `source` column tracks provenance for the domain gap analysis described in §4.4. The `notes` column is optional free-text — use it to document multi-label clips (e.g., "Also exhibits temporal flicker") so the labeling judgment is traceable.

### 4.2 Dataset Sources

#### Strategy A: AI-Generated Clips via API (preferred — real artifacts)

**Sora (OpenAI) — `tools/fetch_sora.py`**

```python
"""
Fetches video clips from OpenAI's Sora API.

Usage:
    uv run python tools/fetch_sora.py --prompts prompts.txt --output samples/ --count 15

Prompts should request scenes with visible text to trigger TEXT_INCONSISTENCY:
  - "A storefront with neon signs at night in a busy city"
  - "A person reading a newspaper in a park, text clearly visible"
  - "A classroom whiteboard with equations being written"

And scenes likely to produce flicker:
  - "A candle flickering in a dark room with reflections on glass"
  - "Ocean waves crashing on rocks at sunset"

Environment:
    OPENAI_API_KEY must be set.
"""
```

Implementation approach:
1. Read prompts from file (one per line)
2. Call Sora API endpoint for each prompt
3. Save resulting MP4 to `samples/` with incremental naming
4. Log prompt → filename mapping for labeling

**Veo-3 (Google) — `tools/fetch_veo3.py`**

```python
"""
Fetches video clips from Google's Veo-3 (Vertex AI / Gemini API).

Usage:
    uv run python tools/fetch_veo3.py --prompts prompts.txt --output samples/ --count 15

Environment:
    GOOGLE_API_KEY must be set (or use gcloud auth for Vertex AI).
"""
```

Implementation approach:
1. Use `google-genai` SDK
2. Request video generation with text-heavy and motion-heavy prompts
3. Poll for completion, download result
4. Save to `samples/`

**Prompt strategy for maximizing artifact yield:**

| Artifact Target | Prompt Pattern | Why |
|---|---|---|
| TEXT_INCONSISTENCY | Scenes with visible text: signs, books, screens, menus | AI models struggle with text consistency across frames |
| TEMPORAL_FLICKER | High-frequency detail: water, fire, foliage, reflections | Temporal coherence breaks down in complex textures |
| CLEAN (with cuts) | "First show a forest, then cut to a city street" or two-scene prompts | Produces hard scene cuts — critical negative examples for flicker detection |
| CLEAN (with blur) | "Camera slowly goes out of focus" or rack-focus prompts | Produces defocus transitions — critical negative examples for text detection |
| CLEAN (simple) | Simple, slow scenes: still landscape, single object on table | Low complexity = fewer artifacts |

#### Strategy B: Synthetic Generation (development & CI fallback)

**`tools/make_dataset.py`** — Does not require API keys. Useful for development iteration and CI.

```python
"""
Generates synthetic video clips with controllable artifacts.

Usage:
    uv run python tools/make_dataset.py --output samples/ --count 35

Generates six types:
  1. TEXT_INCONSISTENCY: Renders text on frames, applies per-frame random
     mutations (character swaps, jitter, thickness changes, partial erasure).
  2. TEMPORAL_FLICKER: Renders a stable scene, injects random brightness/hue
     shifts and texture swaps on random frame subsets.
  3. CLEAN_TEXT: Renders stable text and scenes with no perturbation.
  4. CLEAN_NOTEXT: Renders scenes without any text.
  5. CLEAN_SCENECUT: Renders two distinct stable scenes with a hard cut.
  6. CLEAN_BLUR: Renders text that progressively blurs (defocus transition).
"""
```

Synthetic generation logic:

```
TEXT_INCONSISTENCY:
  1. Create base frame (solid/gradient background)
  2. Render text string with PIL (e.g., "OPEN 24 HOURS")
  3. For each frame (30fps × 5sec = 150 frames):
     - With probability p=0.3, mutate:
       a. Swap 1-2 characters randomly
       b. Shift text position by ±2px (jitter)
       c. Change font weight/size slightly
       d. Drop characters (partial erasure)
  4. Write frames to MP4 via OpenCV VideoWriter

TEMPORAL_FLICKER:
  1. Create base scene (geometric shapes, gradient)
  2. For each frame:
     - With probability p=0.25, apply:
       a. Brightness shift (±30%)
       b. Hue rotation (±15°)
       c. Local patch replacement from different frame
  3. Write to MP4

CLEAN_TEXT:
  1. Render stable text and scene
  2. Only add minimal natural noise (Gaussian, σ=2)
  3. Write to MP4

CLEAN_NOTEXT:
  1. Render geometric shapes / gradients with NO text
  2. Add minimal noise (Gaussian, σ=2)
  3. Write to MP4

CLEAN_SCENECUT:
  1. Render Scene A for frames 0 to N/2 (e.g., green background + circle)
  2. Render Scene B for frames N/2+1 to N (e.g., grey background + rectangle)
  3. Hard cut between them — no transition, no crossfade
  4. Both scenes are internally stable (no flicker, no text mutation)
  5. Write to MP4
  → This is the critical negative example for temporal flicker.
    The cut produces a massive SSIM spike, but the system must NOT flag it.

CLEAN_BLUR:
  1. Render text "HELLO WORLD" on a stable background
  2. For frames 0 to N/2: sharp rendering
  3. For frames N/2+1 to N: apply progressive Gaussian blur
     (kernel size increasing each frame: 3, 5, 7, 9, ...)
  4. Write to MP4
  → Critical negative example for text inconsistency.
    OCR confidence drops, characters disappear, but it's a blur, not a mutation.
```

#### Strategy C: Public Datasets (supplement)

If API access is limited, supplement with clips from:
- HuggingFace `AIGC-Video` datasets
- YouTube compilations of AI-generated videos (download specific clips with `yt-dlp`)

### 4.3 Labeling Workflow

1. Generate/download all clips into `samples/`
2. Manually review each clip and assign label
3. For synthetic clips, labels are deterministic (assigned at generation time)
4. Write to `labels.csv`
5. Commit both clips and labels to repo (use Git LFS if total size > 50MB)

**Multi-label convention:** If a clip exhibits both TEXT_INCONSISTENCY and TEMPORAL_FLICKER, label it with the **more severe** artifact — the one a human reviewer would notice first. This is a judgment call, but it must be consistent: when in doubt, prefer TEXT_INCONSISTENCY (text artifacts are more visually jarring and more likely to be noticed by end users). Document any multi-label clips in a comment column or a separate note so the ambiguity is traceable. The three-class model can only predict one category per clip; §6.8 describes how secondary signals are surfaced in the evidence notes.

### 4.4 Domain Gap: Synthetic vs. Real Artifacts

**Critical awareness for engineers:** Synthetic artifacts (random character swaps, uniform brightness shifts) have different statistical signatures than real AI-generated artifacts (diffusion model hallucinations, temporal coherence failures). A model trained exclusively on synthetic data may not generalize to real clips.

**Rules:**
1. **Synthetic-only is acceptable for development and CI.** Use it to validate the pipeline works end-to-end, that feature extractors produce sensible values, and that tests pass.
2. **The final training set for any model claiming v0.1 readiness MUST include at minimum 15 real AI-generated clips** (from Strategy A or C). Synthetic clips may supplement but not replace real clips.
3. **After assembling the full dataset, run a feature distribution comparison:** for each feature, compare the mean/std between synthetic and real clips within the same label category. If distributions diverge significantly (e.g., mean differs by > 2 standard deviations), document which features are affected and whether this impacts classification.

This comparison should be output as part of the training log and reviewed before signing off on the model.

**Definition of Done — Dataset:**
- [ ] `samples/` contains ≥ 20 clips meeting the distribution requirements in §4.1
- [ ] At least 15 clips are from real AI generators (Sora, Veo-3, or public datasets)
- [ ] `labels.csv` exists, is valid CSV, and every row's `path` resolves to an existing file
- [ ] `source` column is populated for every clip
- [ ] `make_dataset.py` generates a valid synthetic dataset when run with `--count 35`
- [ ] Synthetic clips are playable MP4s (verified via `ffprobe` or OpenCV)
- [ ] Feature distribution comparison between synthetic and real clips has been run and documented

---

## 5. Feature Extraction Pipeline

### 5.1 Frame Extraction (`vidqc/utils/video.py`)

```
Input: video path, config (fps sampling rate)
Output: list of (frame_index, timestamp, numpy_array) tuples

- Use OpenCV VideoCapture
- Sample at N fps (default: 5fps — configurable)
- Why 5fps: balances granularity vs. compute. Most artifacts persist across
  multiple frames, so we don't need every frame.
- Return frames as RGB numpy arrays (H, W, 3)
- If source resolution > 720p, downscale to 720p before returning
  (see §3.3 — OCR performance mitigation)
```

**Definition of Done — Frame Extraction:**
- [ ] Given a 10s 720p MP4 at 30fps source, returns exactly 50 frames at 5fps
- [ ] Frame indices and timestamps are correct (frame 0 = 0.0s, frame 1 = 0.2s, etc.)
- [ ] Handles corrupt/truncated MP4 gracefully (returns partial frames, logs warning, does not crash)
- [ ] Frames are RGB numpy arrays with shape (H, W, 3) and dtype uint8
- [ ] 1080p source is downscaled to 720p

### 5.2 TEXT_INCONSISTENCY Features (`vidqc/features/text_inconsistency.py`)

```
Step 0: Text Presence Gate
  - Run EasyOCR on ALL sampled frames
  - Count frames where at least one text region was detected
  - Compute: frames_with_text_ratio = frames_with_text / total_frames

  IF frames_with_text_ratio == 0 (no text detected in any frame):
    → Set ALL text features to 0.0 (not NaN — zeros are meaningful here)
    → Set has_text_regions = 0
    → Set frames_with_text_ratio = 0.0
    → Skip Steps 1–6 entirely
    → TEXT_INCONSISTENCY is logically impossible — no text means no text artifact
    → Clip can still be evaluated for TEMPORAL_FLICKER normally

  IF frames_with_text_ratio > 0:
    → Set has_text_regions = 1
    → Proceed to Step 1

  Why this matters:
    Without the gate, a no-text clip produces a feature vector of all zeros
    that looks identical to a clip with perfectly stable text. The classifier
    can't distinguish "no text existed" from "text existed and was rock solid."
    The has_text_regions flag resolves this ambiguity explicitly.
```

For each consecutive frame pair (f_i, f_{i+1}) where text was detected:

```
Step 1: OCR Detection
  - Run EasyOCR on f_i and f_{i+1}
  - Get: list of (bbox, text, confidence) per frame
  - Apply frame-level OCR skip optimization (§3.3):
    If SSIM(f_i, f_{i-1}) > 0.995 → reuse previous frame's full OCR result

Step 2: Text Region Matching
  - Match text regions across frames by IoU of bounding boxes (threshold ≥ 0.3)
  - For matched pairs, compute:

Step 3: Per-Region Features (raw text change signals)
  a. edit_distance:        Levenshtein distance between OCR text outputs
  b. confidence_delta:     |conf_i - conf_{i+1}|
  c. bbox_shift:           L2 distance between bbox centers (jitter)
  d. ssim_text_crop:       SSIM of the cropped text region between frames
  e. char_count_delta:     |len(text_i) - len(text_{i+1})|

Step 4: False Positive Suppression Features
  These features distinguish genuine AI text artifacts from legitimate scene
  changes (blur, defocus, camera motion, transitions) that also degrade OCR.

  The core insight: a blur/transition degrades the ENTIRE frame uniformly,
  while an AI text artifact mutates text while the surrounding scene stays stable.

  a. global_ssim_at_instability:
     - Compute full-frame SSIM for the same frame pairs where text instability
       was detected (edit_distance > threshold)
     - HIGH global SSIM + LOW text-region SSIM = real artifact
       (scene is stable, but text is glitching)
     - LOW global SSIM + LOW text-region SSIM = likely scene change
       (everything is changing, not just text)
     - This is the single most important false-positive discriminator.

  b. confidence_variance_rolling:
     - Rolling variance of OCR confidence over a window of 5-7 frames
     - Blur/defocus causes SMOOTH, MONOTONIC confidence decline
       (readable → blurry → unreadable)
     - AI artifacts cause ERRATIC, NON-MONOTONIC confidence
       (high on frame 30, low on 31, high again on 32)
     - High rolling variance = artifact. Low variance = transition.

  c. confidence_monotonic_ratio:
     - Fraction of consecutive frame transitions where confidence moves in
       a consistent direction (always decreasing or always increasing)
     - Close to 1.0 = smooth transition (blur, fade). Close to 0.5 = erratic (artifact)
     - Computed over the window where text instability was detected.

  d. substitution_ratio:
     - Decompose edit distance into operation types: substitutions vs. deletions vs. insertions
     - Blur causes characters to DISAPPEAR (deletions):
       "HELLO" → "HE" → "" (progressive loss)
     - AI artifacts cause characters to CHANGE INTO OTHER CHARACTERS (substitutions):
       "HELLO" → "HGLLO" → "HELLO" → "HFLQO" (mutation)
     - substitution_ratio = substitutions / total_edit_ops
     - High substitution ratio + stable char count = strong artifact signal

  e. text_vs_scene_instability_ratio:
     - Ratio of text-region instability to global-frame instability
     - text_instability = 1 - ssim_text_crop (mean across regions)
     - scene_instability = 1 - ssim_global
     - ratio = text_instability / max(scene_instability, epsilon)
     - ratio >> 1 = text is changing much more than the scene = artifact
     - ratio ≈ 1 = text and scene changing equally = scene transition

Step 5: Clip-Level Aggregation
  - For each feature (Steps 3 + 4), compute: mean, max, std across all frame pairs
  - Additional:
    * has_text_regions (boolean → 0/1): from Step 0
    * frames_with_text_ratio: from Step 0
    * num_text_regions_detected (total across all frames)
  - Total features: ~27-32 numeric values per clip

Step 6: Evidence Collection (for prediction output)
  - Store frame indices and timestamps where:
    * edit_distance > threshold AND global_ssim > scene_stability_threshold
    (only flag frames where text is unstable but the scene is stable)
  - Store bounding boxes of unstable text regions
  - Include in notes: substitution_ratio and confidence_monotonic_ratio
    to explain why the detection was triggered
```

> **Design Decision — Why not just filter out low-global-SSIM frames?**
>
> Hard-filtering (discarding all frame pairs where global SSIM < threshold) would
> lose information. Instead, we include `global_ssim_at_instability` and
> `text_vs_scene_instability_ratio` as *features* and let XGBoost learn the
> boundary. This is more robust because:
> - The "right" threshold varies by content (a slow pan has lower SSIM than a static shot, but neither is a blur)
> - The classifier can learn interactions (e.g., low global SSIM is fine if substitution_ratio is very high)
> - We avoid manual threshold tuning that breaks on edge cases

**Definition of Done — TEXT_INCONSISTENCY Features:**
- [ ] Text presence gate correctly returns `has_text_regions=0` and all-zero features for a clip with no text
- [ ] Feature vector length is identical regardless of whether text is present (consistent shape)
- [ ] Edit distance, confidence delta, bbox shift, SSIM crop, char count delta are computed correctly for a known synthetic frame pair (verified against hand-calculated values)
- [ ] False positive suppression features (global SSIM context, confidence monotonicity, substitution ratio, text-vs-scene ratio) are computed and non-NaN for all clips with text
- [ ] Frame-level OCR skip optimization is implemented and reduces total OCR calls by ≥ 20% on a typical clean clip
- [ ] Evidence collection produces valid frame indices, timestamps, and bounding boxes
- [ ] No NaN values appear in any feature for any clip in the dataset

### 5.3 TEMPORAL_FLICKER Features (`vidqc/features/temporal_flicker.py`)

```
Step 0: Scene Cut Detection & Segmentation
  A hard scene cut (e.g., landscape → F1 car) produces the same signals as
  temporal flicker: massive SSIM drop, pixel difference spike, optical flow
  explosion. But the pattern is fundamentally different:
    - Scene cut: ONE massive spike, then the new scene stabilizes
    - Flicker: REPEATED smaller spikes, scene never stabilizes

  Rather than teaching the classifier to ignore cuts (fragile on 30 samples),
  we detect cuts first and evaluate segments independently.

  Detection:
    - Compute SSIM between every consecutive frame pair
    - A scene cut = frame pair where:
      * ssim_global < cut_threshold (default: 0.4) — very low similarity
      * AND the next 3 frame pairs have ssim_global > stability_threshold
        (default: 0.8) — the new scene stabilizes quickly
    - This two-condition check avoids misidentifying sustained flicker as a cut
    - Edge case: stop checking for cuts when fewer than
      post_cut_stability_window (default: 3) frame pairs remain in the clip.
      A cut that can't be stability-verified is not confirmed. The remaining
      frames stay in the current segment. This also avoids producing trailing
      segments shorter than min_segment_frames, which would be skipped anyway.

  Segmentation:
    - Split the clip at detected cut points into segments
    - Each segment is evaluated independently for flicker
    - Example: [landscape (frames 0-40)] [CUT] [F1 car (frames 42-150)]
      → evaluate frames 0-40 as segment A, frames 42-150 as segment B
    - If no cuts detected → entire clip is one segment

  Output:
    - num_scene_cuts: count of detected cuts (feature for classifier)
    - segments: list of frame ranges to evaluate
```

For each segment, for each consecutive frame pair (f_i, f_{i+1}):

```
Step 1: Global Frame Comparison
  a. ssim_global:          SSIM between full frames
  b. mae_global:           Mean Absolute Error (pixel-level)
  c. hist_correlation:     Histogram correlation (color distribution shift)

Step 2: Local Patch Analysis (divide frame into 4x4 grid = 16 patches)
  a. ssim_patch_min:       Minimum SSIM across patches
  b. ssim_patch_std:       Std of SSIM across patches (localized flicker)
  c. mae_patch_max:        Max MAE across patches

Step 3: Temporal Gradient
  a. diff_acceleration:    |diff(i,i+1) - diff(i-1,i)| — sudden change in
                           rate of change indicates flicker vs. smooth motion

Step 4: False Positive Suppression Features (scene cut handling)
  These features distinguish flicker from legitimate scene changes
  (cuts, fast pans, zoom transitions) that also produce frame-to-frame spikes.

  a. instability_duration:
     - Count of CONSECUTIVE frame pairs where ssim_global < threshold
     - Scene cut: 1-2 frames. Flicker: 5, 10, 20+.
     - This is the strongest single discriminator.

  b. post_spike_stability:
     - After the worst SSIM drop in the segment, compute mean SSIM
       of the next 3-5 frame pairs
     - High post-spike SSIM = scene stabilized = not flicker
     - Low post-spike SSIM = still unstable = flicker

  c. spike_isolation_ratio:
     - spike_frames / total_segment_frames
     - Scene cut: tiny fraction (1-2 frames out of 50+)
     - Flicker: distributed across a large portion of the segment
     - High ratio = flicker. Low ratio = isolated event (cut, bump).

  d. histogram_shift_persistence:
     - At the worst SSIM spike, compare color histograms of frame N-1
       (before spike) and frame N+5 (after spike)
     - Scene cut: PERMANENT color shift (landscape greens → track greys)
       → low histogram correlation between before and after
     - Flicker: OSCILLATING shifts around the same distribution
       → high histogram correlation between before and after

  e. content_change_persistence:
     - Compare frame N-5 (well before spike) with frame N+5 (well after)
     - Very low SSIM = the scene genuinely changed (cut)
     - High SSIM = same scene, it was just flickering

Step 5: Clip-Level Aggregation
  - For each feature (Steps 1-4), compute per-segment: mean, max, std
  - If multiple segments: report the WORST segment (highest flicker signal)
    This handles clips that have a legitimate cut AND flicker in one segment.
  - Additional:
    * num_scene_cuts (from Step 0)
    * total_spike_frames (across all segments)
    * worst_segment_spike_ratio
  - Total features: ~22-28 numeric values per clip

Step 6: Evidence Collection
  - Store frame pairs where ssim_global < threshold, EXCLUDING cut points
    (only report frames within segments, not the cuts themselves)
  - Store patch grid locations with highest instability
  - Convert patch coords to bounding boxes
  - Include in notes: which segment the flicker was detected in,
    and whether scene cuts were present
```

> **Design Decision — Why segment at cuts instead of just adding features?**
>
> A clip with a scene cut AND flicker in one segment is a real scenario.
> If you evaluate the whole clip as one unit, the cut spike dominates the
> aggregated statistics and drowns out the flicker signal in the other segment.
> Segmentation lets you evaluate each piece independently, then report the
> worst-case segment. It's a small amount of extra logic that prevents a
> whole class of false negatives (missed flicker because a cut was louder).

**Definition of Done — TEMPORAL_FLICKER Features:**
- [ ] Scene cut detection correctly identifies a hard cut in a two-scene synthetic clip
- [ ] Scene cut detection does NOT misidentify sustained flicker as a cut
- [ ] Scene cut detection does NOT attempt to verify cuts within the last `post_cut_stability_window` frames (edge case: late-clip SSIM drops are absorbed into the current segment)
- [ ] Segmentation splits the clip at cut points and produces correct frame ranges
- [ ] A clip with no cuts produces exactly one segment spanning all frames
- [ ] All per-segment features (SSIM, MAE, histogram, patch analysis, temporal gradient) are computed and non-NaN
- [ ] False positive suppression features (instability duration, post-spike stability, spike isolation, histogram persistence, content persistence) are present
- [ ] Multi-segment clips report the worst-case segment's features
- [ ] Evidence excludes cut-point frames and only includes within-segment instability frames
- [ ] No NaN values in any feature for any clip

### 5.4 Combined Feature Vector

```
Per clip:
  [text_features (27-32)] + [flicker_features (22-28)] = ~50-60 features total

  Text features include:
    - Raw text change signals (edit distance, confidence, SSIM, etc.)
    - False-positive suppression signals (global SSIM context, confidence trajectory, etc.)
    - Text presence gate signals (has_text_regions, frames_with_text_ratio)

  Flicker features include:
    - Raw frame comparison signals (SSIM, MAE, histogram correlation)
    - Local patch analysis (per-patch SSIM distribution)
    - Temporal gradient (diff acceleration)
    - Scene cut suppression signals (instability duration, post-spike stability, etc.)
    - Scene structure signals (num_scene_cuts, worst_segment_spike_ratio)

  When has_text_regions = 0:
    - All text features are 0.0 except has_text_regions (0) and frames_with_text_ratio (0.0)
    - XGBoost learns: "when has_text_regions = 0, ignore all text features"
    - The clip is classified purely on temporal flicker features
    - TEXT_INCONSISTENCY cannot be predicted (by design)

This is a single row in the training matrix.
Label: (artifact: bool, category: TEXT_INCONSISTENCY | TEMPORAL_FLICKER | NONE)
```

**Definition of Done — Combined Feature Vector:**
- [ ] Feature vector has a fixed, documented length (exact number of features, not a range)
- [ ] A feature manifest file or constant in code lists every feature name, its index, and which module produces it
- [ ] Feature vector shape is identical for every clip regardless of content (text present or not, cuts or not)
- [ ] All values are finite floats (no NaN, no Inf)

---

## 6. Model

### 6.1 Classifier Architecture (`vidqc/models/classifier.py`)

**Single three-class classifier (preferred for v0.1):**

Given the small dataset (20–40 clips), a single three-class XGBoost classifier (`NONE` / `TEXT_INCONSISTENCY` / `TEMPORAL_FLICKER`) is preferred over a two-stage binary-then-multiclass approach. The rationale:

- A two-stage approach trains the category classifier on only the positive samples (~18–26 clips split across two classes, i.e., 9–13 per class). With 5-fold CV, each fold trains on ~7–10 samples per class — too few for reliable learning.
- A single three-class model trains on ALL samples in every fold, giving each fold access to the full dataset.
- The hard constraint (§6.2) still applies and provides the same safety guarantee.

```
Classification:
  → XGBoost multiclass classifier: NONE, TEXT_INCONSISTENCY, TEMPORAL_FLICKER
  → Trained on full feature vector (§5.4) with all samples
  → Output: class probabilities [P(NONE), P(TEXT), P(FLICKER)]

  Decision logic (two sequential steps):

  Step 1 — Binary decision (threshold on P(NONE)):
    artifact = P(NONE) < binary_threshold    # default 0.5
    confidence = 1.0 - P(NONE)

  Step 2 — Category decision (only when artifact=true):
    category = argmax(P(TEXT_INCONSISTENCY), P(TEMPORAL_FLICKER))
    Ties broken deterministically: TEXT_INCONSISTENCY wins ties.

  When artifact=false:
    category = NONE
    confidence still meaningful — e.g. 0.45 means "probably clean but borderline"

  Then apply HARD CONSTRAINT (§6.2): if has_text_regions == 0,
  TEXT_INCONSISTENCY is disallowed regardless of model output.

  Diagnostics: log the full probability vector [P(NONE), P(TEXT), P(FLICKER)]
  in the training manifest and prediction evidence for debugging.
  If many predictions show near-equal P(TEXT) and P(FLICKER), the features
  don't discriminate well between artifact types — feeds back into the
  three-class vs. two-stage architecture decision.
```

**Fallback to two-stage** if the three-class model shows confusion between the two artifact classes (per-category F1 < 0.60 for either class). In that case, revert to:

```
Stage 1: Binary — artifact present? (0/1)
  → XGBoost binary classifier on full feature vector

Stage 2: Category — which artifact?
  → XGBoost binary (TEXT_INCONSISTENCY vs. TEMPORAL_FLICKER)
  → Only runs if Stage 1 predicts artifact=1
  → Uses same feature vector
  → HARD CONSTRAINT still applies
```

Document which architecture was selected and why in the training log.

### 6.2 Hard Constraint: No Text → No Text Artifact

```
IF has_text_regions == 0 AND category (from Step 2) == TEXT_INCONSISTENCY:
  → Override category to TEMPORAL_FLICKER (if P(FLICKER) > 0.1)
  → Override category to NONE and set artifact=false (if P(FLICKER) ≤ 0.1)
  → Log: "Hard constraint triggered: TEXT_INCONSISTENCY overridden for no-text clip"
```

**Why the hard constraint instead of letting XGBoost learn it?**
With 30 samples, there's a real risk the model sees a spurious correlation and
predicts TEXT_INCONSISTENCY for a no-text clip. The constraint is cheap insurance.
XGBoost will *probably* learn this from `has_text_regions`, but "probably" isn't
good enough for a production system.

### 6.3 Why XGBoost

- Works on 20–40 samples (with regularization)
- Native feature importance → interpretability
- Fast training and inference
- No GPU required

### 6.4 Training (`vidqc/train.py`)

```
Input: labels.csv, samples/ directory, config.yaml
Output:
  models/classifier.json          (trained XGBoost model)
  models/feature_scaler.pkl       (fitted StandardScaler)
  models/training_manifest.json   (see §6.6)
  models/feature_importances.json (sorted feature importance scores)

Steps:
  1. Load labels.csv
  2. For each clip: extract features (text + temporal)
     - Cache features to features_cache.pkl to avoid re-extraction
  3. Split: stratified k-fold (k=5) given small dataset
     - No fixed train/test split — too few samples
  4. Train classifier with cross-validation
     - Log per-fold: accuracy, precision, recall, F1 (binary + per-category)
  5. Compute and log cross-validation mean ± std for all metrics
  6. Check against success criteria (§0.5):
     - If met: save final model trained on full dataset
     - If NOT met: log warning, still save model, trigger fallback evaluation (§6.5)
  7. Save model + scaler + manifest
  8. Log feature importances (top 15)
  9. Run feature distribution comparison: synthetic vs. real clips (§4.4)
```

**XGBoost hyperparameters (conservative for small data):**

```yaml
xgboost:
  n_estimators: 50        # Keep small — avoid overfitting
  max_depth: 3            # Shallow trees
  learning_rate: 0.1
  min_child_weight: 3     # Regularization
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1          # L1 regularization
  reg_lambda: 1.0         # L2 regularization
  objective: "multi:softprob"
  num_class: 3
  eval_metric: "mlogloss"
  seed: 42                # Reproducibility
```

### 6.5 Fallback Strategy: If ML Metrics Are Not Met

If after training on the full dataset with real clips, the success criteria in §0.5 are not met, follow this decision tree:

```
1. Check cross-validation variance.
   IF std > 0.15 for any metric:
     → Dataset is too small or too unbalanced for XGBoost to learn reliably.
     → Action: add more clips (priority: the underrepresented category).

2. Check feature importances.
   IF top features are not the expected false-positive suppression features:
     → Features may not discriminate well on real data.
     → Action: inspect feature distributions, check for bugs in extraction.

3. IF metrics are close (within 0.05 of target) but not passing:
     → Try the two-stage architecture (§6.1 fallback).
     → Try lighter regularization (reduce reg_alpha, increase n_estimators to 80).

4. IF metrics are far from target (recall < 0.70 or precision < 0.50):
     → The feature engineering approach may not capture real AI artifact patterns.
     → Action: pause, collect 20+ more real clips, re-examine feature design.
     → Do NOT ship a model that doesn't meet minimum quality.

5. Last resort: threshold-based classification.
     → Use feature importance analysis to identify the top 3–5 features.
     → Implement a rule-based classifier with manual thresholds.
     → This sacrifices generalization but is interpretable and debuggable.
     → Document this as a temporary measure.
```

### 6.6 Model Versioning & Training Manifest

Every training run produces a `models/training_manifest.json` that ties the model to its provenance:

```json
{
  "model_version": "0.1.0",
  "trained_at": "2025-06-15T14:30:00Z",
  "architecture": "three_class",
  "dataset": {
    "labels_sha256": "abc123...",
    "total_clips": 35,
    "real_clips": 20,
    "synthetic_clips": 15,
    "clips_per_category": {
      "TEXT_INCONSISTENCY": 12,
      "TEMPORAL_FLICKER": 10,
      "NONE": 13
    }
  },
  "config_sha256": "def456...",
  "metrics": {
    "binary_recall_mean": 0.88,
    "binary_recall_std": 0.07,
    "binary_precision_mean": 0.82,
    "binary_precision_std": 0.09,
    "text_f1_mean": 0.85,
    "flicker_f1_mean": 0.78,
    "success_criteria_met": true
  },
  "feature_count": 55,
  "top_features": ["global_ssim_at_instability_max", "instability_duration_max", "..."]
}
```

This manifest is saved alongside the model and checked into version control. When a model is loaded for prediction, the manifest is logged for traceability.

**Definition of Done — Model:**
- [ ] Training completes without errors on the full dataset
- [ ] Cross-validation results are printed to stdout and saved to `eval_results.json`
- [ ] `training_manifest.json` is generated with all required fields
- [ ] Feature importances are saved and the top 15 logged
- [ ] Hard constraint (no text → no text artifact) is enforced and covered by `test_category_constraint.py`
- [ ] Model files are loadable from disk and produce predictions
- [ ] If success criteria (§0.5) are not met, the fallback strategy (§6.5) has been followed and documented

### 6.7 Evaluation (`vidqc/eval.py`)

```
Input: labels.csv, samples/, trained model path
Output: metrics printed to stdout + saved to eval_results.json

Metrics:
  - Accuracy, Precision, Recall, F1 (binary: artifact vs. no artifact)
  - Per-category Precision, Recall, F1
  - Confusion matrix (3×3)
  - Cross-validation mean ± std for all metrics
  - Success criteria pass/fail status (§0.5)
```

**Definition of Done — Evaluation:**
- [ ] `vidqc eval` prints a clear summary including all metrics listed above
- [ ] `eval_results.json` contains all metrics in a parseable format
- [ ] Confusion matrix is included (text or visual)
- [ ] Pass/fail against §0.5 success criteria is explicitly stated

### 6.8 Prediction (`vidqc/predict.py`)

```
Single clip:
  1. Extract features
  2. Run classifier → get [P(NONE), P(TEXT), P(FLICKER)]
  3. Binary decision: artifact = P(NONE) < binary_threshold
  4. Category decision: if artifact, category = argmax(P(TEXT), P(FLICKER))
     else category = NONE
  5. Apply hard constraint (§6.2)
  6. Collect evidence from feature extraction step
  7. Secondary signal check (§6.9): if artifact=true, check whether the
     non-winning pipeline also shows strong signal. Append to evidence notes.
  8. Attach probability vector to evidence for diagnostics
  9. Build output dict, validate with Pydantic schema

Batch:
  1. Glob all .mp4 files in input directory
  2. Run single-clip prediction for each
  3. Write JSONL (one JSON object per line)
```

**Definition of Done — Prediction:**
- [ ] Single-clip prediction produces valid Pydantic-validated JSON
- [ ] Batch prediction produces valid JSONL where every line parses independently
- [ ] Evidence includes frame indices, timestamps, and bounding boxes when artifact=true
- [ ] Evidence is None when artifact=false
- [ ] A warning is logged if artifact=true but no evidence frames were collected
- [ ] Confidence is a float in [0.0, 1.0]
- [ ] Hard constraint is applied (verified by unit test)
- [ ] Prediction is deterministic (verified by `test_determinism.py`)
- [ ] Secondary signal check appends to evidence notes when both pipelines show signal

### 6.9 Secondary Signal Check (Multi-Label Detection)

The three-class model outputs a single category per clip. However, a clip can exhibit both text inconsistency and temporal flicker simultaneously. Since both feature pipelines run on every clip regardless of the model's category decision, we can detect this at the feature level as a post-processing step — no model changes required.

```
After category decision (Step 4 of §6.8), when artifact=true:

  text_signal = (has_text_regions == 1
                 AND text_vs_scene_instability_ratio_max > 1.5)
  flicker_signal = (instability_duration_max >= instability_duration_min_config)

  if text_signal AND flicker_signal:
    evidence.notes += "Both text instability and temporal flicker signals
                       detected. Primary category ({category}) based on
                       higher class probability. Secondary signal:
                       {the other category}."
```

This is informational only — it does not change the predicted category or confidence. It surfaces a hint for downstream consumers (human reviewers, dashboards) that the clip may have multiple issues. The `class_probabilities` field also implicitly carries this signal: near-equal P(TEXT) and P(FLICKER) suggests both are present.

**Known limitation (v0.1):** The output schema supports a single `artifact_category`. True multi-label output (predicting multiple categories with independent confidences) would require either a multi-label classifier or independent per-category binary classifiers, both of which need more training data than v0.1 has. This is a candidate for v0.2 if multi-artifact clips are common in production data.

---

## 7. CLI Interface

### 7.1 Entry Point (`vidqc/__main__.py`)

```python
"""
Usage:
    python -m vidqc train --config config.yaml
    python -m vidqc eval --config config.yaml
    python -m vidqc predict --input clip.mp4
    python -m vidqc batch --input samples/ --output predictions.jsonl
"""
```

### 7.2 Commands

```
vidqc train
    --config PATH          Config file (default: config.yaml)
    --labels PATH          Labels CSV (default: labels.csv)
    --samples PATH         Samples directory (default: samples/)
    --output-dir PATH      Where to save models (default: models/)
    --cache / --no-cache   Cache extracted features (default: --cache)

vidqc eval
    --config PATH
    --labels PATH
    --samples PATH
    --model-dir PATH       Trained model directory (default: models/)

vidqc predict
    --input PATH           Single video file
    --model-dir PATH
    --config PATH

vidqc batch
    --input PATH           Directory of video files
    --output PATH          Output JSONL path (default: predictions.jsonl)
    --model-dir PATH
    --config PATH
```

### 7.3 Output Schema (Pydantic)

```python
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Optional

class ArtifactCategory(str, Enum):
    TEXT_INCONSISTENCY = "TEXT_INCONSISTENCY"
    TEMPORAL_FLICKER = "TEMPORAL_FLICKER"
    NONE = "NONE"

class Evidence(BaseModel):
    timestamps: list[float]
    bounding_boxes: list[list[int]]     # [[x1, y1, x2, y2], ...]
    frames: list[int]
    notes: str

class Prediction(BaseModel):
    clip_id: str
    artifact: bool
    confidence: float = Field(ge=0.0, le=1.0)
    artifact_category: ArtifactCategory
    class_probabilities: dict[str, float]  # {"NONE": 0.3, "TEXT_INCONSISTENCY": 0.35, "TEMPORAL_FLICKER": 0.35}
    evidence: Optional[Evidence] = None

    @model_validator(mode="after")
    def check_evidence_contract(self) -> "Prediction":
        if not self.artifact and self.evidence is not None:
            raise ValueError("evidence must be None when artifact=false")
        if self.artifact and self.artifact_category == ArtifactCategory.NONE:
            raise ValueError("artifact_category cannot be NONE when artifact=true")
        if not self.artifact and self.artifact_category != ArtifactCategory.NONE:
            raise ValueError("artifact_category must be NONE when artifact=false")
        return self
```

**Evidence contract:**
- `artifact=false` → `evidence` is always `None`. No artifact means no evidence.
- `artifact=true` → `evidence` is normally populated with specific frames, timestamps, and bounding boxes. `None` is permitted but should be rare (indicates the classifier fired on aggregate statistics but no single frame pair crossed the evidence threshold). When `evidence` is `None` on a positive prediction, log a warning during prediction.

**Definition of Done — CLI Interface:**
- [ ] `python -m vidqc --help` displays available commands
- [ ] Each command (`train`, `eval`, `predict`, `batch`) has its own `--help` with all flags documented
- [ ] `vidqc predict --input clip.mp4` outputs valid JSON to stdout
- [ ] `vidqc batch --input samples/ --output predictions.jsonl` writes valid JSONL
- [ ] Invalid input (missing file, bad config) produces a clear error message and non-zero exit code
- [ ] Output conforms to the Pydantic schema — `artifact_category` is a valid enum value, `confidence` is in [0.0, 1.0], evidence contract is enforced (None when no artifact)

---

## 8. Configuration (`config.yaml`)

```yaml
# Frame extraction
video:
  sample_fps: 5                 # Frames per second to extract
  max_duration_sec: 10          # Truncate longer clips
  max_resolution: 720           # Downscale to this height if source is larger

# TEXT_INCONSISTENCY detection
text:
  ocr_languages: ["en"]
  ocr_gpu: false                # Set true if GPU available.
                                # GPU mode requires pytorch with CUDA.
                                # CPU mode uses pytorch CPU-only (default install).
  ocr_skip_ssim_threshold: 0.995 # Skip OCR if frame is near-identical to previous.
                                  # Set high (0.995 not 0.98) to avoid skipping frames
                                  # where small text regions mutated on a stable background.
  bbox_iou_threshold: 0.3       # Min IoU to match text regions across frames
  edit_distance_threshold: 2    # Flag frame pair if edit distance exceeds this
  confidence_drop_threshold: 0.3

  # False positive suppression (blur/transition handling)
  fp_suppression:
    global_ssim_floor: 0.7        # If global SSIM < this, scene is changing too much
                                  # to reliably attribute text changes to artifacts.
                                  # Used as a feature, not a hard filter.
    confidence_rolling_window: 7  # Frames for rolling variance of OCR confidence
    monotonic_ratio_threshold: 0.8  # Above this = likely smooth transition, not artifact
    substitution_ratio_min: 0.4  # Below this = mostly deletions = likely blur

# TEMPORAL_FLICKER detection
temporal:
  ssim_spike_threshold: 0.85    # Frame pairs with SSIM below this are flagged
  patch_grid: [4, 4]            # Divide frame into NxM patches
  diff_acceleration_threshold: 0.15

  # Scene cut detection & segmentation
  scene_cut:
    cut_ssim_threshold: 0.4       # SSIM below this = potential cut point
    post_cut_stability_window: 3  # Frames after cut to check for stabilization
    post_cut_stability_min: 0.8   # Mean SSIM after cut must exceed this to confirm cut
    min_segment_frames: 5         # Segments shorter than this are skipped
                                  # (not enough frames to evaluate flicker)

  # False positive suppression
  fp_suppression:
    instability_duration_min: 3   # Consecutive unstable frames needed to flag flicker
    post_spike_window: 5          # Frames after worst spike to measure stability
    content_persistence_offset: 5 # Frames before/after spike to compare content

# Classifier
model:
  binary_threshold: 0.5         # artifact=true when P(NONE) < this value.
                                # Higher = more recall, more false positives (0.6-0.7 for quality gates).
                                # Lower = more precision, fewer false positives (0.3-0.4 for triage).
  xgboost:
    n_estimators: 50
    max_depth: 3
    learning_rate: 0.1
    min_child_weight: 3
    subsample: 0.8
    colsample_bytree: 0.8
    objective: "multi:softprob"
    num_class: 3
    seed: 42

# Logging
logging:
  level: INFO                   # DEBUG | INFO | WARNING | ERROR
  file: logs/vidqc.log
```

**Definition of Done — Configuration:**
- [ ] `config.yaml` loads without errors via `vidqc/config.py`
- [ ] Every config value has a sensible default (the system runs with zero CLI overrides)
- [ ] CLI flags override config values (e.g., `--config` loads a different YAML)
- [ ] Invalid config (missing required keys, wrong types) produces a clear error at startup, not a runtime crash

---

## 9. Tests

### 9.1 Schema Validation (`tests/test_schema.py`)

```
- Construct a Prediction with artifact=true, valid evidence → assert no validation error
- Construct a Prediction with artifact=false, evidence=None → assert no validation error
- Construct with missing fields → assert ValidationError
- Construct with out-of-range confidence (e.g., 1.5) → assert ValidationError
- Construct with invalid category (e.g., "BLUR") → assert ValidationError
- Construct with artifact=true, category=NONE → assert ValidationError
- Construct with artifact=false, category=TEXT_INCONSISTENCY → assert ValidationError
- Construct with artifact=false, evidence=(non-None) → assert ValidationError
- Construct with artifact=true, evidence=None → assert no validation error
  (valid but rare — log warning case)
```

### 9.2 Determinism (`tests/test_determinism.py`)

```
- Run predict on a fixed synthetic clip twice
- Assert both outputs are identical (same confidence, same category, same evidence)
- Ensures no random seed leakage
```

### 9.3 Text Feature Unit Test (`tests/test_text_features.py`)

```
- Create two synthetic frames:
    a. Frame A: clean text "HELLO WORLD"
    b. Frame B: mutated text "HE1LO WQRLD" (shifted, different chars)
- Extract text features for the pair
- Assert edit_distance > 0
- Assert ssim_text_crop < 1.0
- Assert substitution_ratio > 0.5 (mutations, not deletions)
- Assert the feature extractor returns correct bounding box coordinates
```

### 9.4 False Positive Suppression Test (`tests/test_blur_suppression.py`)

```
- Create a sequence of 10 synthetic frames with text "HELLO WORLD":
    a. Frames 0-4: sharp, clean text
    b. Frames 5-9: progressively Gaussian-blurred (simulating defocus transition)
- Extract text features across all frame pairs
- Assert:
    * confidence_monotonic_ratio ≈ 1.0 (smooth decline, not erratic)
    * substitution_ratio is low (characters disappearing, not mutating)
    * global_ssim_at_instability is low (whole frame is changing)
    * text_vs_scene_instability_ratio ≈ 1.0 (text degrading at same rate as scene)
- This sequence should NOT be flagged as TEXT_INCONSISTENCY.
  Compare against a second sequence where text mutates on a stable background
  — that one SHOULD be flagged.
```

### 9.5 Temporal Feature Unit Test (`tests/test_temporal_features.py`)

```
- Create two synthetic frames:
    a. Frame A: uniform blue
    b. Frame B: uniform blue with a bright patch (simulated flicker)
- Extract temporal features
- Assert ssim_global < 1.0
- Assert the spike is localized to the correct patch in the grid
```

### 9.6 Scene Cut Suppression Test (`tests/test_scene_cut_suppression.py`)

```
- Create a sequence of 20 synthetic frames simulating a hard scene cut:
    a. Frames 0-9: solid green background with a white circle (landscape)
    b. Frames 10-19: solid grey background with a red rectangle (F1 car)
    Frame 9→10 is the cut point — massive visual change.
- Run temporal feature extraction
- Assert:
    * num_scene_cuts == 1
    * The cut point (frame 9-10) is NOT included in evidence frame list
    * instability_duration is low (1-2 frames, not sustained)
    * post_spike_stability is high (new scene stabilizes quickly)
    * spike_isolation_ratio is low (spike is a tiny fraction of clip)
    * content_change_persistence is low (frames 5 and 15 are very different)
- This clip should NOT be flagged as TEMPORAL_FLICKER.

- Edge case — late-clip SSIM drop:
    Create a sequence of 12 synthetic frames:
    a. Frames 0-9: solid green, stable
    b. Frames 10-11: solid grey (abrupt change in last 2 frames)
    Frame 9→10 has SSIM < 0.4, but only 1 frame pair remains after it —
    not enough to verify stability (post_cut_stability_window = 3).
- Assert:
    * num_scene_cuts == 0 (cut not confirmed — can't verify stability)
    * All frames remain in a single segment
    * No crash or index-out-of-bounds error
```

### 9.7 Scene Cut WITH Flicker Test (`tests/test_scene_cut_with_flicker.py`)

```
- Create a sequence of 30 synthetic frames:
    a. Frames 0-9: solid green, stable (clean segment)
    b. Frame 10: cut to solid grey (scene cut)
    c. Frames 11-29: grey background with random brightness oscillations
       on every other frame (±30% brightness) — flicker in second segment
- Run temporal feature extraction
- Assert:
    * num_scene_cuts == 1
    * Segmentation splits at frame 10
    * Segment A (frames 0-9): clean, no flicker signal
    * Segment B (frames 11-29): high flicker signal
    * The clip IS flagged as TEMPORAL_FLICKER (because segment B has it)
    * Evidence frames are all within segment B, not at the cut point
- This tests that segmentation correctly isolates the cut from the flicker,
  preventing the cut from either masking the flicker (false negative)
  or being reported as the flicker source (wrong evidence).
```

### 9.8 No-Text Gate Test (`tests/test_no_text_gate.py`)

```
- Create a sequence of 10 synthetic frames with NO text (just geometric shapes
  or solid colors)
- Extract text features
- Assert:
    * has_text_regions == 0
    * frames_with_text_ratio == 0.0
    * ALL other text features == 0.0 (not NaN)
    * Feature vector length is the same as a clip WITH text
      (consistent shape regardless of text presence)
```

### 9.9 Category Constraint Test (`tests/test_category_constraint.py`)

```
- Mock a prediction scenario where:
    * Model predicts TEXT_INCONSISTENCY
    * But has_text_regions == 0
- Assert the hard constraint overrides:
    * Output category must NOT be TEXT_INCONSISTENCY
    * Override is logged
- This tests the logical invariant: no text → no text artifact,
  regardless of what the model says.
```

**Definition of Done — Tests:**
- [ ] All 9 test files exist and pass with `pytest -v`
- [ ] Tests run on synthetic data only (no API keys or real clips required)
- [ ] Tests complete in < 60 seconds total (fast CI feedback)
- [ ] No test depends on the trained model existing (feature extraction tests are model-independent)
- [ ] CI pipeline runs all tests on every push

---

## 10. Local Development Workflow

```bash
# Setup
uv sync
uv sync --group dev --group dataset    # Include dev and dataset tools

# Generate synthetic dataset (no API keys needed)
uv run python tools/make_dataset.py --output samples/ --count 30

# (Optional) Fetch real AI-generated clips
export OPENAI_API_KEY=sk-...
uv run python tools/fetch_sora.py --prompts tools/prompts.txt --output samples/ --count 15

export GOOGLE_API_KEY=...
uv run python tools/fetch_veo3.py --prompts tools/prompts.txt --output samples/ --count 15

# Train
uv run python -m vidqc train --config config.yaml

# Evaluate
uv run python -m vidqc eval --config config.yaml

# Predict single clip
uv run python -m vidqc predict --input samples/text_001.mp4

# Batch predict
uv run python -m vidqc batch --input samples/ --output predictions.jsonl

# Run tests
uv run pytest -v

# Lint
uv run ruff check vidqc/ tests/
uv run mypy vidqc/
```

**Definition of Done — Local Development Workflow:**
- [ ] A new engineer can clone the repo, run the commands above in order, and get a working prediction on synthetic data within 30 minutes
- [ ] README.md documents these steps clearly
- [ ] No step requires manual intervention or undocumented environment setup

---

## 11. Deployment Strategy

### 11.1 Containerization (for both local and production use)

```dockerfile
# Stage 1: Build — install dependencies with uv
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime — lean image with only what's needed
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY vidqc/ vidqc/
COPY config.yaml .
COPY models/ models/

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "vidqc"]
```

### 11.2 Production Architecture

The full AWS production deployment plan (SQS + ECS Fargate + S3 + DynamoDB, scaling to 500 concurrent executions) is documented in `PRODUCTION_PLAN.md`.

**The production plan should NOT be implemented until the following gates are passed:**

| Gate | Condition |
|------|-----------|
| 1 | v0.1 success criteria (§0.5) are met on real AI-generated clips |
| 2 | End-to-end latency has been benchmarked and is acceptable (< 60s per clip on CPU) |
| 3 | The model has been validated on at least 10 clips not in the training set (hold-out or new clips) |

Until those gates are passed, deployment means: run the Docker container locally or on a single EC2 instance with the CLI.

**Definition of Done — Deployment Strategy:**
- [ ] `docker build -t vidqc .` succeeds
- [ ] `docker run vidqc predict --input /data/clip.mp4` works with a volume mount
- [ ] Container image size is < 3GB (EasyOCR models are large; verify)
- [ ] `PRODUCTION_PLAN.md` exists with the full AWS architecture

---

## 12. Expected Latency Budget (per clip, 10s 720p, CPU)

| Pipeline Stage | Estimated Time | Notes |
|----------------|---------------|-------|
| Frame extraction | 0.5–1s | OpenCV, fast |
| OCR (EasyOCR, all frames) | 15–50s | Dominant cost. ~0.5–1s/frame × 50 frames, minus skips |
| Text feature computation | 0.5–1s | NumPy/scikit-image, fast after OCR |
| Temporal feature computation | 1–3s | SSIM on full frames + patches |
| Scene segmentation | < 0.5s | Uses SSIM values already computed |
| Classification (XGBoost) | < 0.1s | Negligible |
| Schema validation + output | < 0.1s | Negligible |
| **Total** | **~18–55s** | **Target: < 60s** |

The wide range is due to OCR skip effectiveness. Clips with no text or very stable frames will be faster (high skip rate). Clips with changing text in every frame will be slower.

---

## 13. What This Spec Intentionally Avoids

- **Neural network training on the clip dataset.** 30 clips is not enough. Feature engineering + XGBoost is the correct choice for this data regime.
- **GPU requirements.** EasyOCR can run on CPU (slower but fine for this scale). Keeps infrastructure simpler and cheaper.
- **Over-engineering.** No Kubernetes, no Terraform, no microservices for v0.1. The Docker container + CLI is the deployment unit.
- **Premature production scaling.** The AWS architecture exists in `PRODUCTION_PLAN.md` but is gated behind validation milestones (§11.2).
- **Premature optimization.** Get it working first, profile later. The bottleneck will be EasyOCR inference time per frame — addressed with frame sampling rate and OCR skip optimization (§3.3), not architectural complexity.
- **Artifact categories that require spatial reasoning.** Hands, faces, physics — deferred to v0.2+ when we have more data and can justify more complex models (§0.4).
- **Multi-label classification.** A clip can only be assigned one artifact category. Clips with both text inconsistency and temporal flicker are labeled with the more severe artifact (§4.3), and a secondary signal check surfaces hints about the other (§6.9). True multi-label output is deferred to v0.2 when more training data is available.
- **Parallel batch processing.** The `batch` command runs single-threaded in v0.1. The full dataset processes in ~30 minutes sequentially, which is acceptable. Multiprocessing adds substantial complexity (EasyOCR/OpenCV thread safety requires `spawn` context and per-worker initialization — see §3.3). Deferred to v0.2 if throughput becomes a bottleneck.

---

## Appendix A: Milestone Checklist

This is the recommended implementation order. Each milestone is independently testable.

### Milestone 1: Scaffolding & Synthetic Data
- [ ] Repository structure created (§2)
- [ ] uv project builds (§3)
- [ ] `make_dataset.py` generates synthetic clips (§4, Strategy B)
- [ ] `labels.csv` populated for synthetic clips
- [ ] CI pipeline runs lint
- **Exit criterion:** `uv sync && uv run python tools/make_dataset.py --output samples/ --count 30` succeeds.

### Milestone 2: Frame Extraction & Core Utilities
- [ ] `vidqc/utils/video.py` extracts frames (§5.1)
- [ ] Config loader works (§8)
- [ ] Pydantic schemas defined (§7.3)
- [ ] `test_schema.py` passes
- **Exit criterion:** Frame extraction works on all synthetic clips. Schema tests pass.

### Milestone 3: Text Feature Extraction
- [ ] `text_inconsistency.py` implements Steps 0–6 (§5.2)
- [ ] Text presence gate works (§5.2, Step 0)
- [ ] OCR skip optimization implemented (§3.3)
- [ ] `test_text_features.py` passes
- [ ] `test_blur_suppression.py` passes
- [ ] `test_no_text_gate.py` passes
- **Exit criterion:** Text features extracted for all synthetic clips. No NaN values. All text-related tests pass.

### Milestone 4: Temporal Feature Extraction
- [ ] `temporal_flicker.py` implements Steps 0–6 (§5.3)
- [ ] `scene_segmentation.py` implements cut detection (§5.3, Step 0)
- [ ] `test_temporal_features.py` passes
- [ ] `test_scene_cut_suppression.py` passes
- [ ] `test_scene_cut_with_flicker.py` passes
- **Exit criterion:** Temporal features extracted for all synthetic clips. No NaN values. All temporal tests pass.

### Milestone 5: Classifier & Training
- [ ] `classifier.py` implements train/predict/save/load (§6.1)
- [ ] `train.py` runs end-to-end (§6.4)
- [ ] Hard constraint implemented (§6.2)
- [ ] Training manifest generated (§6.6)
- [ ] `test_category_constraint.py` passes
- [ ] `test_determinism.py` passes
- **Exit criterion:** Model trained on synthetic data. All tests pass. Manifest file exists.

### Milestone 6: Real Data & Validation
- [ ] ≥ 15 real AI-generated clips acquired (§4.2, Strategy A or C)
- [ ] `labels.csv` updated with real clips
- [ ] Feature distribution comparison run (§4.4)
- [ ] Model retrained on mixed dataset
- [ ] Success criteria (§0.5) evaluated
- [ ] Fallback strategy applied if criteria not met (§6.5)
- **Exit criterion:** v0.1 success criteria met on a dataset containing real clips. Or: fallback strategy documented and applied.

### Milestone 7: CLI, Docker & Documentation
- [ ] All CLI commands work (§7)
- [ ] Dockerfile builds and runs (§11.1)
- [ ] README.md documents setup and usage (§10)
- [ ] End-to-end latency benchmarked (§12)
- **Exit criterion:** A new engineer can follow the README, build the container, and run a prediction in < 30 minutes.
