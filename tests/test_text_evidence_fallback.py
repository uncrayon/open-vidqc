"""Tests for TEXT_INCONSISTENCY evidence fallback behavior."""

import numpy as np

from vidqc.features.text_inconsistency import (
    MatchedRegionPair,
    OCRRegion,
    _collect_evidence,
)


def _make_config() -> dict:
    return {
        "text": {
            "edit_distance_threshold": 2,
            "confidence_drop_threshold": 0.3,
            "fp_suppression": {"global_ssim_floor": 0.7},
        }
    }


def _make_pair(
    text_a: str = "OPEN",
    text_b: str = "OP3N",
    conf_a: float = 0.9,
    conf_b: float = 0.9,
    bbox_a: list[int] | None = None,
    bbox_b: list[int] | None = None,
) -> MatchedRegionPair:
    if bbox_a is None:
        bbox_a = [10, 10, 90, 40]
    if bbox_b is None:
        bbox_b = [14, 12, 94, 42]
    region_a = OCRRegion(
        bbox=bbox_a, text=text_a, confidence=conf_a, frame_index=0
    )
    region_b = OCRRegion(
        bbox=bbox_b, text=text_b, confidence=conf_b, frame_index=1
    )
    return MatchedRegionPair(region_a=region_a, region_b=region_b, iou=0.8)


def test_collect_evidence_strict_path():
    """Strict evidence is returned when edit distance and SSIM gates pass."""
    frames = [
        np.full((80, 120, 3), 100, dtype=np.uint8),
        np.full((80, 120, 3), 100, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2]
    pair = _make_pair(text_a="OPEN", text_b="0P3N")
    per_region = {
        "edit_distance": [3.0],
        "confidence_delta": [0.1],
        "bbox_shift": [4.0],
    }

    evidence = _collect_evidence([pair], frames, timestamps, per_region, _make_config())

    assert evidence is not None
    assert evidence.frames == [0, 1]
    assert "RELAXED_EVIDENCE" not in evidence.notes


def test_collect_evidence_relaxed_fallback_path():
    """Fallback evidence is returned when strict edit-distance gate misses."""
    frames = [
        np.full((80, 120, 3), 100, dtype=np.uint8),
        np.full((80, 120, 3), 100, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2]
    pair = _make_pair(text_a="OPEN", text_b="OPEN", conf_a=0.95, conf_b=0.1)
    per_region = {
        "edit_distance": [0.0],
        "confidence_delta": [0.85],
        "bbox_shift": [24.0],
    }

    evidence = _collect_evidence([pair], frames, timestamps, per_region, _make_config())

    assert evidence is not None
    assert evidence.frames == [0, 1]
    assert len(evidence.bounding_boxes) == 2
    assert "RELAXED_EVIDENCE" in evidence.notes


def test_collect_evidence_returns_none_when_no_signal():
    """No strict/relaxed signal should keep evidence as None."""
    frames = [
        np.full((80, 120, 3), 100, dtype=np.uint8),
        np.full((80, 120, 3), 100, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2]
    pair = _make_pair(text_a="OPEN", text_b="OPEN", conf_a=0.9, conf_b=0.9)
    per_region = {
        "edit_distance": [0.0],
        "confidence_delta": [0.0],
        "bbox_shift": [0.5],
    }

    evidence = _collect_evidence([pair], frames, timestamps, per_region, _make_config())

    assert evidence is None
