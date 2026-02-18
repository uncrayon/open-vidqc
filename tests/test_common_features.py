"""Unit tests for shared feature helpers."""

import numpy as np

from vidqc.features.common import compute_ssim


def test_compute_ssim_tiny_crop_no_error():
    """SSIM helper should not raise on tiny crops (<7x7)."""
    frame_a = np.zeros((6, 6, 3), dtype=np.uint8)
    frame_b = np.full((6, 6, 3), 255, dtype=np.uint8)

    value = compute_ssim(frame_a, frame_b)

    assert 0.0 <= value <= 1.0


def test_compute_ssim_tiny_crop_identical_is_high():
    """Identical tiny crops should yield max similarity."""
    frame = np.full((2, 2, 3), 120, dtype=np.uint8)

    value = compute_ssim(frame, frame.copy())

    assert value == 1.0
