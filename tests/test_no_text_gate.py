"""Tests for no-text gate in TEXT_INCONSISTENCY detection.

Tests that videos with no text return all-zero features,
and that feature vector length is consistent.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vidqc.features.text_inconsistency import extract_text_features


def create_geometric_frame(
    shape_type: str = "circle", size: tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """Create frame with geometric shape (no text).

    Args:
        shape_type: Type of shape ("circle", "rectangle", "triangle")
        size: (width, height) of frame

    Returns:
        RGB frame as numpy array (uint8)
    """
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    if shape_type == "circle":
        draw.ellipse([400, 250, 880, 470], fill=(255, 0, 0), outline=(0, 0, 0))
    elif shape_type == "rectangle":
        draw.rectangle([400, 250, 880, 470], fill=(0, 255, 0), outline=(0, 0, 0))
    elif shape_type == "triangle":
        draw.polygon([(640, 250), (400, 470), (880, 470)], fill=(0, 0, 255), outline=(0, 0, 0))

    return np.array(img)


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


def test_no_text_returns_zeros():
    """Test that video with no text returns all-zero features."""
    # Create video with geometric shapes (no text)
    frames = []
    for shape in ["circle", "rectangle", "triangle", "circle", "rectangle"] * 2:
        frames.append(create_geometric_frame(shape))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "no_text.mp4")
        create_test_video(frames, video_path)

        features = extract_text_features(video_path)

        # Check has_text_regions is 0
        assert features["has_text_regions"] == 0.0

        # Check frames_with_text_ratio is 0
        assert features["frames_with_text_ratio"] == 0.0

        # Check all other features are 0.0
        for name, value in features.items():
            assert value == 0.0, f"Feature {name} should be 0.0 for no-text video, got {value}"


def test_no_text_vector_length():
    """Test that no-text video returns 29-feature vector."""
    frames = [create_geometric_frame("circle") for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "no_text_length.mp4")
        create_test_video(frames, video_path)

        features = extract_text_features(video_path)

        # Check length
        assert len(features) == 29

        # Check all expected keys exist
        from vidqc.features.feature_manifest import TEXT_FEATURE_NAMES

        assert set(features.keys()) == set(TEXT_FEATURE_NAMES)


def test_text_vs_no_text_same_length():
    """Test that text and no-text videos return same feature vector length."""
    # No-text video
    notext_frames = [create_geometric_frame("circle") for _ in range(5)]

    # Text video
    from PIL import ImageDraw, ImageFont

    text_frames = []
    for _ in range(5):
        img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except Exception:
            font = ImageFont.load_default()
        draw.text((200, 300), "TEST TEXT", fill=(0, 0, 0), font=font)
        text_frames.append(np.array(img))

    with tempfile.TemporaryDirectory() as tmpdir:
        notext_path = str(Path(tmpdir) / "notext.mp4")
        text_path = str(Path(tmpdir) / "text.mp4")

        create_test_video(notext_frames, notext_path)
        create_test_video(text_frames, text_path)

        notext_features = extract_text_features(notext_path)
        text_features = extract_text_features(text_path)

        # Both should have same length
        assert len(notext_features) == len(text_features) == 29

        # Both should have same keys
        assert set(notext_features.keys()) == set(text_features.keys())

        # No-text should be all zeros
        assert all(v == 0.0 for v in notext_features.values())

        # Text should have has_text_regions=1
        assert text_features["has_text_regions"] == 1.0
