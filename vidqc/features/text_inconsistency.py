"""TEXT_INCONSISTENCY feature extraction using OCR.

Implements a 6-step pipeline:
0. Text presence gate - skip extraction if no text detected
1. OCR with skip optimization - reuse results when SSIM > 0.995
2. Text region matching - match bboxes across frames by IoU
3. Per-region features - edit distance, confidence, bbox shift, SSIM, char count
4. False positive suppression - global SSIM, confidence variance, substitution ratio
5. Clip-level aggregation - mean/max/std for all features
6. Evidence collection - frame indices, bboxes, diagnostic notes
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import Levenshtein
import cv2
import easyocr
import numpy as np
from tqdm import tqdm

from vidqc.config import get_config
from vidqc.features.common import (
    bbox_center,
    compute_iou,
    compute_ssim,
    crop_bbox,
    l2_distance,
    safe_max,
    safe_mean,
    safe_min,
    safe_ratio,
    safe_std,
)
from vidqc.features.feature_manifest import get_zero_features
from vidqc.schemas import Evidence
from vidqc.utils.logging import get_logger
from vidqc.utils.video import extract_frames

logger = get_logger(__name__)

# Global OCR reader singleton
_ocr_reader: Optional[easyocr.Reader] = None


@dataclass
class OCRRegion:
    """Single text region detected by OCR."""

    bbox: list[int]  # [x1, y1, x2, y2]
    text: str
    confidence: float  # [0.0, 1.0]
    frame_index: int


@dataclass
class OCRFrameResult:
    """OCR results for a single frame."""

    frame_index: int
    timestamp: float
    regions: list[OCRRegion]
    was_skipped: bool
    reused_from_index: Optional[int]


@dataclass
class MatchedRegionPair:
    """Pair of matched text regions across two frames."""

    region_a: OCRRegion
    region_b: OCRRegion
    iou: float


def _bbox_width_height(bbox: list[int]) -> tuple[int, int]:
    """Return non-negative width/height for [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1), max(0, y2 - y1)


def _summarize_ocr_bbox_sizes(ocr_results: list[OCRFrameResult]) -> dict[str, int]:
    """Summarize OCR bbox dimensions for diagnostics/logging."""
    widths = []
    heights = []
    for result in ocr_results:
        for region in result.regions:
            w, h = _bbox_width_height(region.bbox)
            widths.append(w)
            heights.append(h)

    if len(widths) == 0:
        return {
            "total_regions": 0,
            "min_width": 0,
            "min_height": 0,
            "max_width": 0,
            "max_height": 0,
            "tiny_lt7_count": 0,
            "tiny_lt3_count": 0,
        }

    tiny_lt7_count = sum(1 for w, h in zip(widths, heights) if min(w, h) < 7)
    tiny_lt3_count = sum(1 for w, h in zip(widths, heights) if min(w, h) < 3)
    return {
        "total_regions": len(widths),
        "min_width": min(widths),
        "min_height": min(heights),
        "max_width": max(widths),
        "max_height": max(heights),
        "tiny_lt7_count": tiny_lt7_count,
        "tiny_lt3_count": tiny_lt3_count,
    }


def _format_bbox_stats_for_log(stats: dict[str, int]) -> str:
    """Format bbox stats as a compact log string."""
    return (
        f"regions={stats['total_regions']}, "
        f"min={stats['min_width']}x{stats['min_height']}, "
        f"max={stats['max_width']}x{stats['max_height']}, "
        f"tiny<7={stats['tiny_lt7_count']}, "
        f"tiny<3={stats['tiny_lt3_count']}"
    )


def _resolve_ocr_gpu(config: dict) -> bool:
    """Resolve the ocr_gpu config value to a concrete boolean.

    Args:
        config: Configuration dict with text.ocr_gpu

    Returns:
        True if GPU should be used, False otherwise
    """
    raw = config["text"].get("ocr_gpu", "auto")

    if isinstance(raw, bool):
        return raw

    value = str(raw).lower().strip()
    if value == "true":
        return True
    if value == "false":
        return False

    # "auto" or any unrecognized value: detect CUDA
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    logger.info(f"ocr_gpu=auto: CUDA {'available' if use_gpu else 'not available'}, using {'GPU' if use_gpu else 'CPU'}")
    return use_gpu


def _get_ocr_reader(config: dict) -> easyocr.Reader:
    """Get or initialize EasyOCR reader (singleton pattern).

    Downloads model weights (~100MB) on first call.

    Args:
        config: Configuration dict with text.ocr_languages and text.ocr_gpu

    Returns:
        EasyOCR Reader instance
    """
    global _ocr_reader
    if _ocr_reader is None:
        use_gpu = _resolve_ocr_gpu(config)
        logger.info(
            f"Initializing EasyOCR (downloading weights if not cached, ~100MB)... "
            f"[gpu={use_gpu}]"
        )
        _ocr_reader = easyocr.Reader(
            config["text"]["ocr_languages"], gpu=use_gpu
        )
        logger.info("EasyOCR initialized")
    return _ocr_reader


def _parse_ocr_bbox(easyocr_bbox: list) -> list[int]:
    """Convert EasyOCR bbox format to [x1, y1, x2, y2].

    EasyOCR returns: [[top_left, top_right, bottom_right, bottom_left], text, conf]
    where each point is [x, y].

    Args:
        easyocr_bbox: List of 4 points from EasyOCR

    Returns:
        [x1, y1, x2, y2] as list of ints
    """
    points = easyocr_bbox  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _parse_ocr_result(
    ocr_output: list, frame_index: int, timestamp: float
) -> OCRFrameResult:
    """Parse EasyOCR output into OCRFrameResult.

    Args:
        ocr_output: List of [bbox, text, conf] from EasyOCR
        frame_index: Frame index
        timestamp: Timestamp in seconds

    Returns:
        OCRFrameResult with parsed regions
    """
    regions = []
    for bbox_points, text, conf in ocr_output:
        bbox = _parse_ocr_bbox(bbox_points)
        regions.append(
            OCRRegion(bbox=bbox, text=text, confidence=float(conf), frame_index=frame_index)
        )
    return OCRFrameResult(
        frame_index=frame_index,
        timestamp=timestamp,
        regions=regions,
        was_skipped=False,
        reused_from_index=None,
    )


def _run_ocr_with_skip(
    frames: list[np.ndarray], timestamps: list[float], config: dict
) -> list[OCRFrameResult]:
    """Run OCR with skip optimization (Step 1).

    Always OCR frame 0. For frames 1-N, if SSIM > threshold, reuse previous OCR result.
    This optimization is critical for latency budget.

    Args:
        frames: List of RGB frames (uint8)
        timestamps: List of timestamps (seconds)
        config: Configuration dict

    Returns:
        List of OCRFrameResult, one per frame
    """
    if len(frames) == 0:
        return []

    reader = _get_ocr_reader(config)
    threshold = config["text"]["ocr_skip_ssim_threshold"]

    results = []
    skip_count = 0

    # Always OCR first frame
    ocr_output = reader.readtext(frames[0])
    results.append(_parse_ocr_result(ocr_output, 0, timestamps[0]))

    # For subsequent frames, check SSIM before OCR
    for i in tqdm(range(1, len(frames)), desc="  OCR frames", unit="frame", leave=False):
        frame_ssim = compute_ssim(frames[i - 1], frames[i])

        if frame_ssim > threshold:
            # Reuse previous OCR result
            prev_result = results[-1]
            # Shift frame_index to current frame
            reused_regions = [
                OCRRegion(
                    bbox=r.bbox.copy(),
                    text=r.text,
                    confidence=r.confidence,
                    frame_index=i,  # Update to current frame
                )
                for r in prev_result.regions
            ]
            results.append(
                OCRFrameResult(
                    frame_index=i,
                    timestamp=timestamps[i],
                    regions=reused_regions,
                    was_skipped=True,
                    reused_from_index=prev_result.frame_index,
                )
            )
            skip_count += 1
        else:
            # Run OCR
            ocr_output = reader.readtext(frames[i])
            results.append(_parse_ocr_result(ocr_output, i, timestamps[i]))

    skip_rate = skip_count / max(len(frames) - 1, 1)
    logger.debug(f"OCR skip rate: {skip_rate:.1%} ({skip_count}/{len(frames)-1} frames)")

    return results


def _match_text_regions(
    regions_a: list[OCRRegion], regions_b: list[OCRRegion], iou_threshold: float
) -> list[MatchedRegionPair]:
    """Match text regions across two frames by IoU (Step 2).

    Args:
        regions_a: Regions from frame A
        regions_b: Regions from frame B
        iou_threshold: Minimum IoU to consider a match

    Returns:
        List of matched region pairs
    """
    matched = []

    # Simple greedy matching: for each region in A, find best match in B
    used_b = set()
    for region_a in regions_a:
        best_iou = 0.0
        best_region_b = None

        for j, region_b in enumerate(regions_b):
            if j in used_b:
                continue
            iou = compute_iou(region_a.bbox, region_b.bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_region_b = region_b

        if best_region_b is not None:
            matched.append(
                MatchedRegionPair(region_a=region_a, region_b=best_region_b, iou=best_iou)
            )
            used_b.add(regions_b.index(best_region_b))

    return matched


def _compute_substitution_ratio(text_a: str, text_b: str) -> float:
    """Compute substitution ratio using Levenshtein operations (Step 4d).

    Substitution ratio = # of replace operations / # of total operations.
    High ratio (>0.5) suggests character mutations (artifact).
    Low ratio suggests deletions/insertions (blur, defocus).

    Args:
        text_a: Text from frame A
        text_b: Text from frame B

    Returns:
        Substitution ratio in [0.0, 1.0]
    """
    if text_a == text_b:
        return 0.0

    ops = Levenshtein.editops(text_a, text_b)
    if len(ops) == 0:
        return 0.0

    substitutions = sum(1 for op, i, j in ops if op == "replace")
    return substitutions / len(ops)


def _compute_per_region_features(
    matched_pairs: list[MatchedRegionPair], frames: list[np.ndarray]
) -> dict[str, list[float]]:
    """Compute per-region features for all matched pairs (Step 3).

    Features:
    - edit_distance: Levenshtein distance
    - confidence_delta: |conf_a - conf_b|
    - bbox_shift: L2 distance between bbox centers
    - ssim_text_crop: SSIM of cropped text regions
    - char_count_delta: |len(text_a) - len(text_b)|

    Args:
        matched_pairs: List of matched region pairs
        frames: List of frames (for SSIM computation)

    Returns:
        Dict mapping feature name to list of values (one per pair)
    """
    features = {
        "edit_distance": [],
        "confidence_delta": [],
        "bbox_shift": [],
        "ssim_text_crop": [],
        "char_count_delta": [],
    }

    for pair in matched_pairs:
        # Edit distance
        edit_dist = Levenshtein.distance(pair.region_a.text, pair.region_b.text)
        features["edit_distance"].append(float(edit_dist))

        # Confidence delta
        conf_delta = abs(pair.region_a.confidence - pair.region_b.confidence)
        features["confidence_delta"].append(conf_delta)

        # Bbox shift
        center_a = bbox_center(pair.region_a.bbox)
        center_b = bbox_center(pair.region_b.bbox)
        shift = l2_distance(center_a, center_b)
        features["bbox_shift"].append(shift)

        # SSIM text crop
        frame_a = frames[pair.region_a.frame_index]
        frame_b = frames[pair.region_b.frame_index]
        crop_a = crop_bbox(frame_a, pair.region_a.bbox)
        crop_b = crop_bbox(frame_b, pair.region_b.bbox)

        # Resize to same size for SSIM (use smaller size)
        if crop_a.size > 0 and crop_b.size > 0:
            h = min(crop_a.shape[0], crop_b.shape[0])
            w = min(crop_a.shape[1], crop_b.shape[1])
            if h > 0 and w > 0:
                crop_a_resized = cv2.resize(crop_a, (w, h))
                crop_b_resized = cv2.resize(crop_b, (w, h))
                ssim_crop = compute_ssim(crop_a_resized, crop_b_resized)
            else:
                ssim_crop = 1.0  # Empty crop = no change
        else:
            ssim_crop = 1.0

        features["ssim_text_crop"].append(ssim_crop)

        # Char count delta
        char_delta = abs(len(pair.region_a.text) - len(pair.region_b.text))
        features["char_count_delta"].append(float(char_delta))

    return features


def _compute_fp_suppression_features(
    matched_pairs: list[MatchedRegionPair],
    frames: list[np.ndarray],
    ocr_results: list[OCRFrameResult],
    config: dict,
) -> dict[str, list[float]]:
    """Compute false positive suppression features (Step 4).

    Features:
    - global_ssim_at_instability: Full-frame SSIM where text changed (edit dist > 0)
    - confidence_variance_rolling: Rolling variance of OCR confidence
    - confidence_monotonic_ratio: Fraction of consistent confidence transitions
    - substitution_ratio: Ratio of replace operations in Levenshtein edits
    - text_vs_scene_instability_ratio: Text change / scene change

    Args:
        matched_pairs: List of matched region pairs
        frames: List of frames
        ocr_results: List of OCR results for all frames
        config: Configuration dict

    Returns:
        Dict mapping feature name to list of values
    """
    features = {
        "global_ssim_at_instability": [],
        "confidence_variance_rolling": [],
        "confidence_monotonic_ratio": [],
        "substitution_ratio": [],
        "text_vs_scene_instability_ratio": [],
    }

    # Global SSIM at instability
    for pair in matched_pairs:
        if pair.region_a.text != pair.region_b.text:
            frame_a = frames[pair.region_a.frame_index]
            frame_b = frames[pair.region_b.frame_index]
            global_ssim = compute_ssim(frame_a, frame_b)
            features["global_ssim_at_instability"].append(global_ssim)

    # Confidence rolling variance
    window = config["text"]["fp_suppression"]["confidence_rolling_window"]
    all_confidences = []
    for result in ocr_results:
        if len(result.regions) > 0:
            # Use mean confidence across all regions in frame
            mean_conf = np.mean([r.confidence for r in result.regions])
            all_confidences.append(mean_conf)

    if len(all_confidences) >= window:
        for i in range(len(all_confidences) - window + 1):
            window_vals = all_confidences[i : i + window]
            variance = float(np.var(window_vals))
            features["confidence_variance_rolling"].append(variance)

    # Confidence monotonic ratio
    if len(all_confidences) >= 2:
        transitions = []
        for i in range(len(all_confidences) - 1):
            delta = all_confidences[i + 1] - all_confidences[i]
            transitions.append(delta)

        # Check if transitions are consistent (all decreasing or all increasing)
        if len(transitions) > 0:
            positive = sum(1 for d in transitions if d > 0)
            negative = sum(1 for d in transitions if d < 0)
            monotonic = max(positive, negative) / len(transitions)
            features["confidence_monotonic_ratio"].append(monotonic)

    # Substitution ratio
    for pair in matched_pairs:
        if pair.region_a.text != pair.region_b.text:
            sub_ratio = _compute_substitution_ratio(pair.region_a.text, pair.region_b.text)
            features["substitution_ratio"].append(sub_ratio)

    # Text vs scene instability ratio
    for pair in matched_pairs:
        edit_dist = Levenshtein.distance(pair.region_a.text, pair.region_b.text)
        if edit_dist > 0:
            frame_a = frames[pair.region_a.frame_index]
            frame_b = frames[pair.region_b.frame_index]
            global_ssim = compute_ssim(frame_a, frame_b)
            # Text changed, scene stable → high ratio (artifact)
            # Text changed, scene changed → low ratio (transition)
            text_change = edit_dist / max(len(pair.region_a.text), 1)
            scene_change = 1.0 - global_ssim
            ratio = safe_ratio(text_change, scene_change)
            features["text_vs_scene_instability_ratio"].append(ratio)

    return features


def _aggregate_clip_features(
    per_region_feats: dict[str, list[float]],
    fp_feats: dict[str, list[float]],
    has_text: bool,
    frames_with_text_ratio: float,
    num_text_regions: int,
) -> dict[str, float]:
    """Aggregate per-region and FP features to clip level (Step 5).

    Computes mean, max, std for each feature.

    Args:
        per_region_feats: Per-region features from Step 3
        fp_feats: FP suppression features from Step 4
        has_text: Whether any text was detected
        frames_with_text_ratio: Fraction of frames with text
        num_text_regions: Total number of text regions detected

    Returns:
        Complete feature dict with 28 features
    """
    if not has_text:
        return get_zero_features()

    features = {}

    # Step 0: Text presence gate
    features["has_text_regions"] = 1.0
    features["frames_with_text_ratio"] = frames_with_text_ratio

    # Step 3: Per-region features (mean, max, std)
    for feat_name in ["edit_distance", "confidence_delta", "bbox_shift", "ssim_text_crop", "char_count_delta"]:
        values = per_region_feats[feat_name]
        features[f"{feat_name}_mean"] = safe_mean(values)
        features[f"{feat_name}_max"] = safe_max(values)
        features[f"{feat_name}_std"] = safe_std(values)

    # Step 4: FP suppression features
    # global_ssim_at_instability: mean, min, std
    values = fp_feats["global_ssim_at_instability"]
    features["global_ssim_at_instability_mean"] = safe_mean(values)
    features["global_ssim_at_instability_min"] = safe_min(values, default=1.0)
    features["global_ssim_at_instability_std"] = safe_std(values)

    # confidence_variance_rolling: mean, max
    values = fp_feats["confidence_variance_rolling"]
    features["confidence_variance_rolling_mean"] = safe_mean(values)
    features["confidence_variance_rolling_max"] = safe_max(values)

    # confidence_monotonic_ratio: single value (take mean if multiple)
    values = fp_feats["confidence_monotonic_ratio"]
    features["confidence_monotonic_ratio"] = safe_mean(values)

    # substitution_ratio: mean, max, std
    values = fp_feats["substitution_ratio"]
    features["substitution_ratio_mean"] = safe_mean(values)
    features["substitution_ratio_max"] = safe_max(values)
    features["substitution_ratio_std"] = safe_std(values)

    # text_vs_scene_instability_ratio: mean, max
    values = fp_feats["text_vs_scene_instability_ratio"]
    features["text_vs_scene_instability_ratio_mean"] = safe_mean(values)
    features["text_vs_scene_instability_ratio_max"] = safe_max(values)

    # Metadata
    features["num_text_regions_detected"] = float(num_text_regions)

    return features


def _collect_evidence(
    matched_pairs: list[MatchedRegionPair],
    frames: list[np.ndarray],
    timestamps: list[float],
    per_region_feats: dict[str, list[float]],
    config: dict,
) -> Optional[Evidence]:
    """Collect evidence for TEXT_INCONSISTENCY artifact (Step 6).

    Criteria: edit_distance > threshold AND global_ssim > scene_stability_threshold
    Fallback: If strict criteria collect no hits, use top unstable text pairs
    to keep evidence actionable for positive predictions.

    Args:
        matched_pairs: List of matched region pairs
        frames: List of frames
        timestamps: List of timestamps (seconds)
        per_region_feats: Per-region features
        config: Configuration dict

    Returns:
        Evidence object if artifact detected, None otherwise
    """
    edit_threshold = config["text"]["edit_distance_threshold"]
    ssim_floor = config["text"]["fp_suppression"]["global_ssim_floor"]
    confidence_threshold = config["text"].get("confidence_drop_threshold", 0.3)
    bbox_shift_threshold = 10.0
    max_fallback_pairs = 3

    evidence_frame_indices = []
    evidence_bboxes = []
    evidence_notes = []
    fallback_candidates = []

    for i, pair in enumerate(matched_pairs):
        edit_dist = per_region_feats["edit_distance"][i]
        conf_delta = per_region_feats["confidence_delta"][i]
        bbox_shift = per_region_feats["bbox_shift"][i]

        frame_a = frames[pair.region_a.frame_index]
        frame_b = frames[pair.region_b.frame_index]
        global_ssim = compute_ssim(frame_a, frame_b)

        if edit_dist > edit_threshold:
            if global_ssim > ssim_floor:
                # Artifact detected
                evidence_frame_indices.append(pair.region_a.frame_index)
                evidence_frame_indices.append(pair.region_b.frame_index)
                evidence_bboxes.append(pair.region_a.bbox)
                evidence_bboxes.append(pair.region_b.bbox)
                evidence_notes.append(
                    f"Frames {pair.region_a.frame_index}-{pair.region_b.frame_index}: "
                    f"Text '{pair.region_a.text}' → '{pair.region_b.text}' "
                    f"(edit_dist={edit_dist:.1f}, global_ssim={global_ssim:.3f})"
                )
                continue

        text_changed = pair.region_a.text != pair.region_b.text
        if (
            text_changed
            or conf_delta >= confidence_threshold
            or bbox_shift >= bbox_shift_threshold
        ):
            # Keep a few high-signal candidates as fallback evidence when strict
            # thresholds miss but classifier still predicts TEXT_INCONSISTENCY.
            score = (
                (edit_dist * 10.0)
                + (conf_delta * 2.0)
                + min(bbox_shift, 50.0) / 10.0
            )
            fallback_candidates.append(
                (
                    score,
                    i,
                    global_ssim,
                    edit_dist,
                    conf_delta,
                    bbox_shift,
                )
            )

    if len(evidence_frame_indices) > 0:
        # Deduplicate and sort
        unique_indices = sorted(set(evidence_frame_indices))
        unique_timestamps = [timestamps[i] for i in unique_indices]

        return Evidence(
            timestamps=unique_timestamps,
            bounding_boxes=evidence_bboxes,  # Keep all bboxes, not just unique
            frames=unique_indices,
            notes=" | ".join(evidence_notes),
        )

    if len(fallback_candidates) > 0:
        fallback_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, i, global_ssim, edit_dist, conf_delta, bbox_shift in fallback_candidates[:max_fallback_pairs]:
            pair = matched_pairs[i]
            evidence_frame_indices.append(pair.region_a.frame_index)
            evidence_frame_indices.append(pair.region_b.frame_index)
            evidence_bboxes.append(pair.region_a.bbox)
            evidence_bboxes.append(pair.region_b.bbox)
            evidence_notes.append(
                f"RELAXED_EVIDENCE Frames {pair.region_a.frame_index}-{pair.region_b.frame_index}: "
                f"Text '{pair.region_a.text}' → '{pair.region_b.text}' "
                f"(edit_dist={edit_dist:.1f}, conf_delta={conf_delta:.3f}, "
                f"bbox_shift={bbox_shift:.1f}, global_ssim={global_ssim:.3f})"
            )

        unique_indices = sorted(set(evidence_frame_indices))
        unique_timestamps = [timestamps[i] for i in unique_indices]
        return Evidence(
            timestamps=unique_timestamps,
            bounding_boxes=evidence_bboxes,
            frames=unique_indices,
            notes=" | ".join(evidence_notes),
        )

    return None


def extract_text_features(
    video_path: str, config: Optional[dict] = None
) -> tuple[dict[str, float], Optional[Evidence]]:
    """Extract TEXT_INCONSISTENCY features from video (Steps 0-6).

    Returns a 29-feature vector and optional evidence. If no text is detected,
    returns all zeros and None evidence.

    Args:
        video_path: Path to video file
        config: Optional config dict (uses default if None)

    Returns:
        Tuple of (features dict with 29 features, Evidence or None)

    Raises:
        FileNotFoundError: If video file does not exist
        ValueError: If feature extraction fails
    """
    if config is None:
        config = get_config()

    # Check file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Extracting text features from {video_path}")

    # Extract frames
    try:
        frame_list = extract_frames(video_path, config)
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        raise ValueError(f"Frame extraction failed: {e}")

    if len(frame_list) == 0:
        logger.warning("No frames extracted, returning zero features")
        return get_zero_features(), None

    # Extract frame data and timestamps
    frames = [f.data for f in frame_list]
    timestamps = [f.timestamp for f in frame_list]

    # Step 1: OCR with skip optimization
    ocr_results = _run_ocr_with_skip(frames, timestamps, config)
    bbox_stats = _summarize_ocr_bbox_sizes(ocr_results)

    # Step 0: Text presence gate
    frames_with_text = sum(1 for r in ocr_results if len(r.regions) > 0)
    frames_with_text_ratio = frames_with_text / max(len(ocr_results), 1)
    total_text_regions = sum(len(r.regions) for r in ocr_results)

    if total_text_regions == 0:
        logger.info("No text detected in video, returning zero features")
        return get_zero_features(), None

    logger.debug(
        f"Text detected: {frames_with_text}/{len(ocr_results)} frames, "
        f"{total_text_regions} total regions"
    )
    logger.debug(
        f"OCR bbox stats for {video_path}: {_format_bbox_stats_for_log(bbox_stats)}"
    )

    try:
        # Step 2: Match text regions across consecutive frames
        all_matched_pairs = []
        for i in range(len(ocr_results) - 1):
            pairs = _match_text_regions(
                ocr_results[i].regions,
                ocr_results[i + 1].regions,
                config["text"]["bbox_iou_threshold"],
            )
            all_matched_pairs.extend(pairs)

        if len(all_matched_pairs) == 0:
            logger.info("No matched text regions across frames, returning zero features")
            return get_zero_features(), None

        # Step 3: Compute per-region features
        per_region_feats = _compute_per_region_features(all_matched_pairs, frames)

        # Step 4: Compute FP suppression features
        fp_feats = _compute_fp_suppression_features(all_matched_pairs, frames, ocr_results, config)

        # Step 5: Aggregate to clip level
        features = _aggregate_clip_features(
            per_region_feats, fp_feats, True, frames_with_text_ratio, total_text_regions
        )

        # Step 6: Collect evidence
        evidence = _collect_evidence(all_matched_pairs, frames, timestamps, per_region_feats, config)
        if evidence:
            logger.info(f"TEXT_INCONSISTENCY evidence: {evidence.notes}")
    except Exception as e:
        stats_text = _format_bbox_stats_for_log(bbox_stats)
        logger.error(
            "Text feature internals failed for %s | bbox_stats: %s",
            video_path,
            stats_text,
        )
        raise ValueError(
            f"Text feature internals failed for {video_path} | bbox_stats: {stats_text} | {e}"
        ) from e

    # Validate
    from vidqc.features.feature_manifest import validate_features

    validate_features(features)

    logger.info(f"Extracted {len(features)} text features")
    return features, evidence
