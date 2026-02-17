"""Tests for segment display fallback config validation."""

import pytest

from vidqc.config import validate_config


def _base_config() -> dict:
    return {
        "video": {"sample_fps": 5, "max_duration_sec": 10, "max_resolution": 720},
        "text": {
            "ocr_languages": ["en"],
            "ocr_gpu": False,
            "ocr_skip_ssim_threshold": 0.995,
            "bbox_iou_threshold": 0.3,
            "fp_suppression": {
                "global_ssim_floor": 0.7,
                "confidence_rolling_window": 7,
                "monotonic_ratio_threshold": 0.8,
                "substitution_ratio_min": 0.4,
            },
        },
    }


def test_segment_display_config_validates_with_defaults():
    cfg = _base_config()
    cfg["text"]["segment_display_fallback"] = {
        "enabled": True,
        "max_candidates": 4,
        "max_fallback_frames_per_clip": 8,
        "retry_stride_on_miss": 2,
        "min_confidence": 0.3,
        "allowlist": "0123456789:",
        "preprocess_variants": ["raw", "inv_clahe", "inv_clahe_up2"],
        "min_component_area_ratio": 0.0005,
        "max_component_area_ratio": 0.2,
    }

    validate_config(cfg)


def test_segment_display_config_rejects_invalid_variant():
    cfg = _base_config()
    cfg["text"]["segment_display_fallback"] = {
        "preprocess_variants": ["raw", "bad_variant"],
    }

    with pytest.raises(ValueError, match="preprocess_variants"):
        validate_config(cfg)
