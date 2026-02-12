"""Unit tests for TEMPORAL_FLICKER feature extraction.

Tests the basic functionality of temporal feature extraction on synthetic
videos with known frame instabilities.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vidqc.features.temporal_flicker import extract_temporal_features


def create_solid_frame(color: tuple[int, int, int], size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """Create a solid color frame.

    Args:
        color: (R, G, B) color tuple
        size: (width, height) of frame

    Returns:
        RGB frame as numpy array (uint8)
    """
    img = Image.new("RGB", size, color=color)
    return np.array(img)


def create_test_video(frames: list[np.ndarray], output_path: str, fps: int = 5) -> None:
    """Save frames as MP4 video.

    Args:
        frames: List of RGB frames
        output_path: Path to save video
        fps: Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()


def test_stable_video_high_ssim():
    """Test that stable video (no flicker) has high SSIM."""
    # Create video with stable frames (all same color)
    frames = []
    for _ in range(10):
        frames.append(create_solid_frame((100, 150, 200)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "stable.mp4")
        create_test_video(frames, video_path)

        features, evidence = extract_temporal_features(video_path)

        # Stable video should have high SSIM
        assert features["ssim_mean"] > 0.99
        assert features["ssim_min"] > 0.99

        # Low MAE (frames are identical)
        assert features["mae_mean"] < 1.0
        assert features["mae_max"] < 1.0

        # High histogram correlation
        assert features["hist_corr_mean"] > 0.99

        # Low patch instability
        assert features["patch_instability_mean"] < 0.1

        # No NaN or Inf
        for name, value in features.items():
            assert not np.isnan(value), f"Feature {name} is NaN"
            assert not np.isinf(value), f"Feature {name} is Inf"

        # Evidence should be None for stable video
        assert evidence is None, "Evidence should be None for stable video"


def test_flickering_video_low_ssim():
    """Test that flickering video has low SSIM."""
    # Create video with alternating colors (flicker)
    frames = []
    for i in range(10):
        if i % 2 == 0:
            frames.append(create_solid_frame((100, 100, 100)))
        else:
            frames.append(create_solid_frame((200, 200, 200)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "flicker.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_temporal_features(video_path)

        # Flickering video should have lower SSIM
        assert features["ssim_min"] < 0.99

        # Higher MAE (frames are different)
        assert features["mae_mean"] > 10.0

        # Patch instability should be detected
        assert features["patch_instability_max"] > 0.0

        # Instability duration should be > 0
        assert features["instability_duration"] > 0


def test_feature_vector_length():
    """Test that feature vector always has 14 elements."""
    frames = [create_solid_frame((100, 100, 100)) for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "length_test.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_temporal_features(video_path)

        # Check length
        assert len(features) == 14

        # Check all expected keys exist
        from vidqc.features.feature_manifest import TEMPORAL_FEATURE_NAMES

        assert set(features.keys()) == set(TEMPORAL_FEATURE_NAMES)


def test_gradient_change_detected():
    """Test that gradual color changes are detected with appropriate metrics."""
    # Create video with gradual brightness change
    frames = []
    for i in range(10):
        brightness = 50 + i * 20  # 50 to 230
        frames.append(create_solid_frame((brightness, brightness, brightness)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "gradient.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_temporal_features(video_path)

        # Gradual change should have:
        # - Relatively high SSIM (changes are smooth, not abrupt)
        # - Non-zero MAE (frames are changing)
        # Note: Histogram correlation can be low for solid color frames
        # because the entire distribution shifts

        assert features["ssim_mean"] > 0.8  # Not too low (smooth changes)
        assert features["mae_mean"] > 5.0  # Some difference detected
        assert features["patch_instability_max"] < 1.0  # Not all patches unstable
