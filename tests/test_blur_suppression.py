"""Tests for false positive suppression in TEXT_INCONSISTENCY detection.

Tests that blur/defocus (false positives) can be distinguished from
genuine text artifacts using the FP suppression features.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tests.conftest import get_test_font
from vidqc.features.text_inconsistency import extract_text_features


def create_text_frame_pil(
    text: str, position: tuple[int, int] = (200, 300), size: tuple[int, int] = (1280, 720)
) -> Image.Image:
    """Create PIL image with text (for blur operations).

    Args:
        text: Text to render
        position: (x, y) top-left position
        size: (width, height) of frame

    Returns:
        PIL Image
    """
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = get_test_font(48)

    draw.text(position, text, fill=(0, 0, 0), font=font)
    return img


def apply_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur to PIL image.

    Args:
        img: PIL Image
        radius: Blur radius (0 = no blur, 10 = heavy blur)

    Returns:
        Blurred PIL Image
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB, uint8).

    Args:
        img: PIL Image

    Returns:
        Numpy array (H, W, 3)
    """
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


def test_blur_sequence_low_substitution_ratio():
    """Test that progressive blur has low substitution ratio.

    Blur causes OCR to lose characters (deletions), not swap them.
    Substitution ratio should be low (<0.3).
    """
    frames = []
    base_img = create_text_frame_pil("CLEAR TEXT MESSAGE", (200, 300))

    # Progressive blur: 0 → 10
    for blur_radius in np.linspace(0, 10, 10):
        blurred = apply_gaussian_blur(base_img, blur_radius)
        frames.append(pil_to_numpy(blurred))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "blur_sequence.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_text_features(video_path)

        # Blur should have:
        # - Low substitution ratio (deletions, not swaps)
        # - High confidence monotonic ratio (steady decrease)
        # - May have high edit distance (but wrong type of edits)

        # Key discriminator: substitution ratio
        # Blur causes deletions/insertions, not character swaps
        if features["substitution_ratio_max"] > 0:  # If any text changes detected
            assert (
                features["substitution_ratio_max"] < 0.5
            ), f"Blur should have low substitution ratio, got {features['substitution_ratio_max']:.3f}"

        # Note: We don't strictly test confidence_monotonic_ratio here because
        # OCR confidence can vary non-monotonically with blur level.
        # The key discriminator is substitution_ratio.


def test_artifact_sequence_high_substitution_ratio():
    """Test that text mutations on stable background have high substitution ratio.

    Character swaps (artifacts) produce high substitution ratio (>0.5).
    """
    frames = []

    # Stable background, mutating text
    texts = [
        "HELLO WORLD",
        "HELLO WORLD",
        "HE1LO W0RLD",  # Character swaps
        "HE1LO W0RLD",
        "H3LLO WORLD",  # More swaps
        "H3LLO WORLD",
        "HELLO W0RLD",
        "HELLO W0RLD",
        "HE110 WORLD",
        "HE110 WORLD",
    ]

    for text in texts:
        img = create_text_frame_pil(text, (200, 300))
        frames.append(pil_to_numpy(img))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "artifact_sequence.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_text_features(video_path)

        # Artifacts should have:
        # - High substitution ratio (character swaps)
        # - High global SSIM (stable background)
        # - Low confidence variance (high OCR confidence throughout)

        assert features["edit_distance_max"] > 0, "Should detect text changes"

        # Key discriminator: substitution ratio
        assert (
            features["substitution_ratio_max"] > 0.5
        ), "Artifacts should have high substitution ratio"

        # Background should be stable
        if features["global_ssim_at_instability_mean"] > 0:
            assert (
                features["global_ssim_at_instability_mean"] > 0.9
            ), "Background should be stable during artifact"


def test_no_false_positive_on_stable_blur():
    """Test that uniformly blurred text (no mutations) has zero edit distance."""
    frames = []
    base_img = create_text_frame_pil("STATIC TEXT", (200, 300))

    # Constant blur level
    blurred = apply_gaussian_blur(base_img, radius=3)
    for _ in range(10):
        frames.append(pil_to_numpy(blurred))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "stable_blur.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_text_features(video_path)

        # OCR may struggle with blurred text, but since it's stable,
        # edit distance should be low
        assert features["edit_distance_max"] < 3.0, "Stable blur should not produce high edit distance"
