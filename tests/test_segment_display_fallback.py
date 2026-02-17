"""Tests for segmented-display OCR fallback in text_inconsistency."""

import cv2
import numpy as np

import vidqc.features.text_inconsistency as ti


class _FakeReader:
    def __init__(self, outputs):
        self._outputs = outputs
        self._idx = 0

    def readtext(self, _image, **_kwargs):
        if self._idx >= len(self._outputs):
            return self._outputs[-1]
        out = self._outputs[self._idx]
        self._idx += 1
        return out


def _minimal_config() -> dict:
    return {
        "text": {
            "ocr_skip_ssim_threshold": 0.995,
            "segment_display_fallback": {
                "enabled": True,
                "max_candidates": 4,
                "max_fallback_frames_per_clip": 8,
                "retry_stride_on_miss": 1,
                "min_confidence": 0.3,
                "allowlist": "0123456789:",
                "preprocess_variants": ["raw"],
                "min_component_area_ratio": 0.0005,
                "max_component_area_ratio": 0.2,
            },
        }
    }


def test_detect_display_candidates_on_synthetic_clock_panel():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 70), (260, 160), (220, 220, 220), -1)
    cv2.putText(
        frame,
        "59:59",
        (76, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cfg = ti._segment_fallback_config(_minimal_config())
    cfg["max_component_area_ratio"] = 0.30
    candidates = ti._detect_display_candidates(frame, cfg)

    assert len(candidates) > 0
    assert any(
        c["bbox"][0] <= 70 and c["bbox"][1] <= 80 and c["bbox"][2] >= 240 and c["bbox"][3] >= 150
        for c in candidates
    )


def test_filter_segment_text_detections_keeps_digits_and_colon_only():
    cfg = {
        "allowlist": "0123456789:",
        "min_confidence": 0.30,
    }
    detections = [
        {"bbox": [10, 10, 60, 30], "text": "59:59", "confidence": 0.95},
        {"bbox": [12, 12, 62, 32], "text": "59:59", "confidence": 0.70},
        {"bbox": [20, 20, 40, 40], "text": "AB", "confidence": 0.98},
        {"bbox": [30, 30, 60, 60], "text": "12:00", "confidence": 0.10},
    ]

    regions = ti._filter_segment_text_detections(detections, cfg, frame_index=3)

    assert len(regions) == 1
    assert regions[0].text == "59:59"
    assert regions[0].frame_index == 3
    assert regions[0].source == "segment_fallback"


def test_fallback_activation_runs_when_generic_ocr_empty(monkeypatch):
    frames = [
        np.zeros((80, 120, 3), dtype=np.uint8),
        np.full((80, 120, 3), 255, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2]
    config = _minimal_config()

    reader = _FakeReader(outputs=[[], []])
    fallback_calls = []

    def fake_get_reader(_config):
        return reader

    def fake_segment_fallback(_frame, frame_index, _reader, _cfg):
        fallback_calls.append(frame_index)
        return (
            [
                ti.OCRRegion(
                    bbox=[10, 10, 50, 30],
                    text="59:59",
                    confidence=0.9,
                    frame_index=frame_index,
                    source="segment_fallback",
                )
            ],
            1,
        )

    monkeypatch.setattr(ti, "_get_ocr_reader", fake_get_reader)
    monkeypatch.setattr(ti, "_run_segment_display_fallback", fake_segment_fallback)

    results = ti._run_ocr_with_skip(frames, timestamps, config)

    assert fallback_calls == [0, 1]
    assert all(len(r.regions) == 1 for r in results)
    assert all(r.regions[0].source == "segment_fallback" for r in results)


def test_fallback_not_run_when_generic_ocr_has_text(monkeypatch):
    frames = [
        np.zeros((80, 120, 3), dtype=np.uint8),
        np.full((80, 120, 3), 255, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2]
    config = _minimal_config()

    ocr_hit = [
        [
            (
                [[8, 8], [60, 8], [60, 30], [8, 30]],
                "12:34",
                0.92,
            )
        ],
        [
            (
                [[8, 8], [60, 8], [60, 30], [8, 30]],
                "12:35",
                0.91,
            )
        ],
    ]
    reader = _FakeReader(outputs=ocr_hit)
    fallback_calls = []

    def fake_get_reader(_config):
        return reader

    def fake_segment_fallback(_frame, frame_index, _reader, _cfg):
        fallback_calls.append(frame_index)
        return ([], 0)

    monkeypatch.setattr(ti, "_get_ocr_reader", fake_get_reader)
    monkeypatch.setattr(ti, "_run_segment_display_fallback", fake_segment_fallback)

    results = ti._run_ocr_with_skip(frames, timestamps, config)

    assert fallback_calls == []
    assert all(len(r.regions) == 1 for r in results)
    assert all(r.regions[0].source == "generic" for r in results)
