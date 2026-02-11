"""Tests for TEMPORAL_FLICKER detection in videos with scene cuts.

Tests that flicker can be detected within segments even when scene cuts are present,
and that the worst-case segment is reported.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vidqc.features.temporal_flicker import extract_temporal_features


def create_solid_frame(color: tuple[int, int, int], size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """Create a solid color frame."""
    img = Image.new("RGB", size, color=color)
    return np.array(img)


def create_textured_frame(
    base_color: tuple[int, int, int], seed: int = 0, size: tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """Create a frame with texture for realistic scene cuts."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:, :] = base_color

    # Add gradient (direction depends on seed for structural difference)
    if seed % 2 == 0:
        # Horizontal gradient
        for x in range(size[0]):
            gradient_factor = x / size[0]
            img[:, x] = np.clip(img[:, x] * (0.5 + 0.5 * gradient_factor), 0, 255).astype(np.uint8)
    else:
        # Vertical gradient
        for y in range(size[1]):
            gradient_factor = y / size[1]
            img[y, :] = np.clip(img[y, :] * (0.5 + 0.5 * gradient_factor), 0, 255).astype(np.uint8)

    # Add large structured noise (seeded for reproducibility)
    rng = np.random.RandomState(seed)
    noise = rng.randint(-40, 40, (size[1], size[0], 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def create_test_video(frames: list[np.ndarray], output_path: str, fps: int = 5) -> None:
    """Save frames as MP4 video."""
    if len(frames) == 0:
        raise ValueError("No frames to save")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()


def test_flicker_in_first_segment():
    """Test that flicker in first segment (before cut) is detected."""
    frames = []

    # Segment 1 (frames 0-5): flickering red/pink
    for i in range(6):
        if i % 2 == 0:
            frames.append(create_solid_frame((200, 0, 0)))  # Red
        else:
            frames.append(create_solid_frame((255, 100, 100)))  # Pink

    # Segment 2 (frames 6-11): stable blue (after scene cut)
    for _ in range(6):
        frames.append(create_solid_frame((0, 0, 200)))  # Blue

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "flicker_first_segment.mp4")
        create_test_video(frames, video_path)

        features = extract_temporal_features(video_path)

        # Should detect flicker in first segment (worst case)
        # Even though second segment is stable, we report worst-case
        assert features["ssim_min"] < 0.99, "Should detect low SSIM in flickering segment"
        assert features["mae_mean"] > 5.0, "Should detect MAE > 0 in flickering segment"
        assert features["instability_duration"] > 0, "Should detect instability duration"


def test_flicker_in_second_segment():
    """Test that flicker in second segment (after cut) is detected."""
    frames = []

    # Segment 1 (frames 0-5): stable red
    for _ in range(6):
        frames.append(create_solid_frame((200, 0, 0)))  # Red

    # Segment 2 (frames 6-11): flickering blue/cyan (after scene cut)
    for i in range(6):
        if i % 2 == 0:
            frames.append(create_solid_frame((0, 0, 200)))  # Blue
        else:
            frames.append(create_solid_frame((100, 100, 255)))  # Cyan

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "flicker_second_segment.mp4")
        create_test_video(frames, video_path)

        features = extract_temporal_features(video_path)

        # Should detect flicker in second segment (worst case)
        assert features["ssim_min"] < 0.99, "Should detect low SSIM in flickering segment"
        assert features["mae_mean"] > 5.0, "Should detect MAE > 0 in flickering segment"
        assert features["instability_duration"] > 0, "Should detect instability duration"


def test_no_false_positive_across_cut():
    """Test that scene cut itself doesn't trigger false positive flicker.

    The cut point should be excluded from flicker analysis.
    """
    # Use same seed for each color to ensure stability within segments
    red_frame = create_textured_frame((200, 0, 0), seed=1)
    blue_frame = create_textured_frame((0, 0, 200), seed=2)

    frames = []

    # Segment 1 (frames 0-5): stable red
    for _ in range(6):
        frames.append(red_frame.copy())

    # Segment 2 (frames 6-11): stable blue
    for _ in range(6):
        frames.append(blue_frame.copy())

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "clean_cut.mp4")
        create_test_video(frames, video_path)

        features = extract_temporal_features(video_path)

        # Both segments are stable, so should NOT detect flicker
        # The scene cut itself should be excluded
        # Note: JPEG compression can reduce SSIM slightly
        assert features["ssim_mean"] > 0.93, "Stable segments should have high SSIM"
        assert features["ssim_min"] > 0.93, "No flicker should be detected"
        assert features["instability_duration"] == 0, "No instability should be detected"


def test_worst_case_segment_reported():
    """Test that features from worst-case segment are reported."""
    frames = []

    # Segment 1 (frames 0-5): mild flicker
    for i in range(6):
        if i % 2 == 0:
            frames.append(create_solid_frame((200, 200, 200)))  # Light gray
        else:
            frames.append(create_solid_frame((210, 210, 210)))  # Slightly lighter gray

    # Segment 2 (frames 6-11): severe flicker (after scene cut)
    for i in range(6):
        if i % 2 == 0:
            frames.append(create_solid_frame((50, 50, 50)))  # Dark
        else:
            frames.append(create_solid_frame((250, 250, 250)))  # Bright

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "worst_case.mp4")
        create_test_video(frames, video_path)

        features = extract_temporal_features(video_path)

        # Should report features from segment 2 (worse flicker)
        # Segment 2 has much larger color difference, so MAE should be high
        assert features["mae_mean"] > 50.0, \
            "Should report high MAE from worst-case segment (segment 2)"

        # SSIM should be lower in segment 2
        assert features["ssim_min"] < 0.9, \
            "Should report low SSIM from worst-case segment"


def test_multiple_cuts_multiple_segments():
    """Test analysis with multiple scene cuts (3+ segments)."""
    frames = []

    # Segment 1 (frames 0-4): stable red
    for _ in range(5):
        frames.append(create_solid_frame((200, 0, 0)))

    # Segment 2 (frames 5-9): flickering green/cyan
    for i in range(5):
        if i % 2 == 0:
            frames.append(create_solid_frame((0, 200, 0)))  # Green
        else:
            frames.append(create_solid_frame((0, 200, 200)))  # Cyan

    # Segment 3 (frames 10-14): stable blue
    for _ in range(5):
        frames.append(create_solid_frame((0, 0, 200)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "three_segments.mp4")
        create_test_video(frames, video_path)

        features = extract_temporal_features(video_path)

        # Should detect flicker in segment 2 (worst case)
        assert features["ssim_min"] < 0.99, "Should detect flicker in middle segment"
        assert features["mae_mean"] > 5.0, "Should detect changes in flickering segment"
