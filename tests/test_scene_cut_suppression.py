"""Tests for scene cut detection and suppression in TEMPORAL_FLICKER detection.

Tests that genuine scene cuts are properly detected and excluded from flicker analysis,
including the late-clip edge case where insufficient frames remain for stability check.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vidqc.config import get_config
from vidqc.features.scene_segmentation import detect_scene_cuts
from vidqc.features.temporal_flicker import extract_temporal_features
from vidqc.utils.video import extract_frames


def create_solid_frame(color: tuple[int, int, int], size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """Create a solid color frame."""
    img = Image.new("RGB", size, color=color)
    return np.array(img)


def create_textured_frame(
    base_color: tuple[int, int, int], seed: int = 0, size: tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """Create a frame with texture (gradient + noise) for realistic scene cuts.

    Solid color frames have high SSIM even when colors differ (structure is same).
    This creates frames with different structure so scene cuts have low SSIM.

    Args:
        base_color: RGB base color
        seed: Random seed for reproducible noise (different seeds = different gradient direction)
        size: (width, height)
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Base color
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


def test_scene_cut_detected():
    """Test that abrupt scene change is detected as a cut."""
    # Create video with clear scene cut:
    # Frames 0-4: red with texture (same seed = identical frames)
    # Frames 5-9: blue with texture (different seed = different structure)
    frames = []
    for i in range(5):
        frames.append(create_textured_frame((200, 0, 0), seed=1))  # Red
    for i in range(5):
        frames.append(create_textured_frame((0, 0, 200), seed=2))  # Blue

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "scene_cut.mp4")
        create_test_video(frames, video_path)

        # Extract frames and detect cuts
        config = get_config()
        frame_list = extract_frames(video_path, config)
        frames_data = [f.data for f in frame_list]

        cuts = detect_scene_cuts(frames_data, config)

        # Should detect cut around frame 4
        assert len(cuts) > 0, "Scene cut should be detected"
        assert cuts[0].frame_index == 4, f"Cut should be at frame 4, got {cuts[0].frame_index}"
        assert cuts[0].ssim < 0.4, "Cut SSIM should be below threshold"


def test_no_false_cut_on_stable_video():
    """Test that stable video doesn't produce false cuts."""
    # Create video with stable frames (all same color)
    frames = [create_solid_frame((100, 150, 200)) for _ in range(10)]

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "stable.mp4")
        create_test_video(frames, video_path)

        config = get_config()
        frame_list = extract_frames(video_path, config)
        frames_data = [f.data for f in frame_list]

        cuts = detect_scene_cuts(frames_data, config)

        # Should not detect any cuts
        assert len(cuts) == 0, f"Stable video should have no cuts, got {len(cuts)}"


def test_late_clip_edge_case():
    """Test that late-clip edge case is handled correctly.

    Scene cut detection should stop checking when fewer than post_cut_stability_window
    frames remain, to avoid incomplete stability checks.
    """
    # Create video with potential cut very close to end
    # Config: post_cut_stability_window = 3
    # So we can't check for cuts in the last 3 frames

    frames = []
    # Frames 0-5: red (stable)
    for _ in range(6):
        frames.append(create_solid_frame((200, 0, 0)))

    # Frame 6-7: abrupt change to blue (too close to end to confirm)
    for _ in range(2):
        frames.append(create_solid_frame((0, 0, 200)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "late_cut.mp4")
        create_test_video(frames, video_path)

        config = get_config()
        frame_list = extract_frames(video_path, config)
        frames_data = [f.data for f in frame_list]

        cuts = detect_scene_cuts(frames_data, config)

        # With post_cut_stability_window=3 and 8 frames total,
        # we can only check up to frame 8-3-1 = 4
        # So the cut at frame 5 or 6 should NOT be detected (insufficient stability window)
        if len(cuts) > 0:
            # If a cut is detected, it should not be in the last few frames
            last_valid_cut_index = len(frames_data) - config["temporal"]["scene_cut"]["post_cut_stability_window"] - 1
            for cut in cuts:
                assert cut.frame_index <= last_valid_cut_index, \
                    f"Cut at frame {cut.frame_index} should not be detected (too close to end)"


def test_post_cut_stability_verification():
    """Test that only cuts with stable post-cut region are confirmed."""
    # Create video with:
    # - Abrupt change at frame 4 (potential cut)
    # - BUT: unstable frames after (5-7) = NOT a true scene cut, more like flicker

    frames = []
    # Frames 0-4: red
    for _ in range(5):
        frames.append(create_solid_frame((200, 0, 0)))

    # Frames 5-9: alternating blue/green (unstable after "cut")
    for i in range(5):
        if i % 2 == 0:
            frames.append(create_solid_frame((0, 0, 200)))
        else:
            frames.append(create_solid_frame((0, 200, 0)))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "unstable_post_cut.mp4")
        create_test_video(frames, video_path)

        config = get_config()
        frame_list = extract_frames(video_path, config)
        frames_data = [f.data for f in frame_list]

        cuts = detect_scene_cuts(frames_data, config)

        # Should NOT confirm the cut because post-cut region is unstable
        # (mean SSIM in post-cut window will be low)
        assert len(cuts) == 0, \
            "Cut with unstable post-cut region should not be confirmed"


def test_multi_segment_analysis():
    """Test that video with scene cut produces separate segment analysis."""
    # Create video with clear scene cut:
    # Segment 1 (frames 0-4): stable red with texture
    # Segment 2 (frames 5-9): stable blue with texture

    # Use same seed for each color to ensure stability within segments
    red_frame = create_textured_frame((200, 0, 0), seed=1)
    blue_frame = create_textured_frame((0, 0, 200), seed=2)

    frames = []
    for _ in range(5):
        frames.append(red_frame.copy())
    for _ in range(5):
        frames.append(blue_frame.copy())

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "two_segments.mp4")
        create_test_video(frames, video_path)

        features, _ = extract_temporal_features(video_path)

        # Both segments are stable, so overall should show high SSIM
        # (within each segment, frames are identical)
        # Note: JPEG compression can reduce SSIM slightly
        assert features["ssim_mean"] > 0.93
        assert features["ssim_min"] > 0.93

        # No flicker within segments
        assert features["instability_duration"] == 0
