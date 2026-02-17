"""Unit tests for TEMPORAL_FLICKER evidence fallback behavior."""

import numpy as np

from vidqc.features.scene_segmentation import Segment
from vidqc.features.temporal_flicker import _collect_temporal_evidence


def _config() -> dict:
    return {
        "temporal": {
            "ssim_spike_threshold": 0.85,
            "patch_grid": [4, 4],
            "diff_acceleration_threshold": 0.15,
        }
    }


def test_temporal_evidence_strict_path():
    """Strict evidence path should trigger for clearly unstable pairs."""
    frames = [
        np.full((64, 64, 3), 0, dtype=np.uint8),
        np.full((64, 64, 3), 255, dtype=np.uint8),
        np.full((64, 64, 3), 0, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2, 0.4]
    segments = [Segment(start_frame=0, end_frame=2, length=3)]

    evidence = _collect_temporal_evidence(
        frames=frames,
        timestamps=timestamps,
        segments=segments,
        cuts=[],
        config=_config(),
    )

    assert evidence is not None
    assert len(evidence.frames) > 0
    assert "RELAXED_EVIDENCE" not in evidence.notes


def test_temporal_evidence_relaxed_fallback_path(monkeypatch):
    """Fallback evidence should trigger when no pair crosses strict SSIM gate."""
    frames = [
        np.full((64, 64, 3), 100, dtype=np.uint8),
        np.full((64, 64, 3), 130, dtype=np.uint8),
        np.full((64, 64, 3), 100, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2, 0.4]
    segments = [Segment(start_frame=0, end_frame=2, length=3)]

    # Keep all SSIM values above strict threshold so strict path does not fire.
    monkeypatch.setattr(
        "vidqc.features.temporal_flicker.compute_ssim",
        lambda _a, _b: 0.90,
    )

    evidence = _collect_temporal_evidence(
        frames=frames,
        timestamps=timestamps,
        segments=segments,
        cuts=[],
        config=_config(),
    )

    assert evidence is not None
    assert len(evidence.frames) > 0
    assert "RELAXED_EVIDENCE" in evidence.notes


def test_temporal_evidence_no_signal_returns_none():
    """Stable content with no instability signal should return no evidence."""
    frames = [
        np.full((64, 64, 3), 100, dtype=np.uint8),
        np.full((64, 64, 3), 100, dtype=np.uint8),
        np.full((64, 64, 3), 100, dtype=np.uint8),
    ]
    timestamps = [0.0, 0.2, 0.4]
    segments = [Segment(start_frame=0, end_frame=2, length=3)]

    evidence = _collect_temporal_evidence(
        frames=frames,
        timestamps=timestamps,
        segments=segments,
        cuts=[],
        config=_config(),
    )

    assert evidence is None
