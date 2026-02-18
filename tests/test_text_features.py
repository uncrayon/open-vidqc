"""Unit tests for TEXT_INCONSISTENCY feature extraction.

Tests the basic functionality of text feature extraction on synthetic
videos with known text mutations.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tests.conftest import get_test_font
from vidqc.features.text_inconsistency import extract_text_features


def create_text_frame(
    text: str, position: tuple[int, int] = (200, 300), size: tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """Create a frame with rendered text.

    Args:
        text: Text to render
        position: (x, y) top-left position for text
        size: (width, height) of frame

    Returns:
        RGB frame as numpy array (uint8)
    """
    # Create white background
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = get_test_font(48)

    # Draw black text
    draw.text(position, text, fill=(0, 0, 0), font=font)

    # Convert to numpy array
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
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()


def test_text_mutation_detected():
    """Test that text mutations are detected with high edit distance."""
    # Create video with text mutation
    frames = []
    # Frame 0-4: "HELLO WORLD"
    for _ in range(5):
        frames.append(create_text_frame("HELLO WORLD", (200, 300)))

    # Frame 5-9: "H31LO WQRLD" (3 character mutations, exceeds edit_distance_threshold=2)
    for _ in range(5):
        frames.append(create_text_frame("H31LO WQRLD", (205, 303)))  # Slight bbox shift

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "text_mutation.mp4")
        create_test_video(frames, video_path)

        # Extract features
        features, evidence = extract_text_features(video_path)

        # Assertions
        assert features["has_text_regions"] == 1.0
        assert features["frames_with_text_ratio"] > 0.8  # Most frames have text

        # Text mutation should produce high edit distance
        assert features["edit_distance_max"] >= 3.0  # At least 3 character changes

        # Substitution ratio should be high (character swaps, not deletions)
        assert features["substitution_ratio_max"] > 0.5

        # Bbox shift should be detected
        assert features["bbox_shift_mean"] > 0.0

        # Global SSIM should be high (background is stable)
        assert features["global_ssim_at_instability_mean"] > 0.9

        # No NaN or Inf
        for name, value in features.items():
            assert not np.isnan(value), f"Feature {name} is NaN"
            assert not np.isinf(value), f"Feature {name} is Inf"

        # Evidence should be collected for text mutations
        assert evidence is not None, "Evidence should not be None for text mutations"
        assert len(evidence.frames) > 0, "Evidence should have frame indices"
        assert len(evidence.timestamps) > 0, "Evidence should have timestamps"
        assert len(evidence.bounding_boxes) > 0, "Evidence should have bounding boxes"


def test_stable_text_low_features():
    """Test that stable text produces low feature values."""
    # Create video with stable text
    frames = []
    for _ in range(10):
        frames.append(create_text_frame("STABLE TEXT", (200, 300)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "stable_text.mp4")
        create_test_video(frames, video_path)

        # Extract features
        features, _ = extract_text_features(video_path)

        # Assertions
        assert features["has_text_regions"] == 1.0
        assert features["frames_with_text_ratio"] == 1.0

        # Stable text should have low edit distance
        assert features["edit_distance_max"] == 0.0

        # No bbox shift
        assert features["bbox_shift_max"] == 0.0

        # High SSIM on text crops
        assert features["ssim_text_crop_mean"] > 0.99


def test_feature_vector_length():
    """Test that feature vector always has 29 elements."""
    frames = [create_text_frame("TEST", (200, 300)) for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "length_test.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_text_features(video_path)

        # Check length
        assert len(features) == 29

        # Check all expected keys exist
        from vidqc.features.feature_manifest import TEXT_FEATURE_NAMES

        assert set(features.keys()) == set(TEXT_FEATURE_NAMES)
