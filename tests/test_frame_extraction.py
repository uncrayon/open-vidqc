"""Frame extraction integration tests.

Tests frame extraction on synthetic clips from Milestone 1.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vidqc.utils.video import extract_frames


@pytest.fixture
def sample_video():
    """Get path to a sample video from Milestone 1."""
    samples_dir = Path("samples")
    video_files = list(samples_dir.glob("clean_text_*.mp4"))
    if not video_files:
        pytest.skip("No sample videos found in samples/ directory")
    return str(video_files[0])


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "video": {
            "sample_fps": 5,
            "max_duration_sec": 10,
            "max_resolution": 720
        }
    }


def test_extract_frames_count(sample_video, test_config):
    """Extract from sample video and verify frame count.

    For a 5-second video at 30fps source, sampling at 5fps should yield 25 frames.
    """
    frames = extract_frames(sample_video, config=test_config)

    # Verify we got frames
    assert len(frames) > 0, "Should extract at least some frames"

    # For 5s @ 30fps source sampled at 5fps: 5 * 5 = 25 frames expected
    # Allow some tolerance for rounding
    assert 20 <= len(frames) <= 30, f"Expected ~25 frames, got {len(frames)}"


def test_frame_properties(sample_video, test_config):
    """Verify frame indices, timestamps, RGB shape, and uint8 dtype."""
    frames = extract_frames(sample_video, config=test_config)

    for i, frame in enumerate(frames):
        # Check index is sequential in sampled space
        assert frame.index == i, f"Frame {i} has wrong index: {frame.index}"

        # Check timestamp increases monotonically
        if i > 0:
            assert frame.timestamp > frames[i-1].timestamp, \
                f"Timestamp not increasing: {frame.timestamp} <= {frames[i-1].timestamp}"

        # Check timestamp is approximately correct (sampled at 5fps = 0.2s intervals)
        expected_timestamp = i * 0.2
        assert abs(frame.timestamp - expected_timestamp) < 0.1, \
            f"Timestamp mismatch: expected ~{expected_timestamp:.2f}s, got {frame.timestamp:.2f}s"

        # Check RGB shape (H, W, 3)
        assert frame.data.ndim == 3, f"Frame should be 3D array, got {frame.data.ndim}D"
        assert frame.data.shape[2] == 3, f"Frame should have 3 channels, got {frame.data.shape[2]}"

        # Check uint8 dtype
        assert frame.data.dtype == np.uint8, f"Frame dtype should be uint8, got {frame.data.dtype}"

        # Check resolution is <= 720p
        height = frame.data.shape[0]
        assert height <= 720, f"Frame height {height} exceeds max_resolution 720"


def test_bgr_to_rgb_conversion(sample_video, test_config):
    """Verify frames are RGB not BGR.

    We'll compare the extracted frames with a manual OpenCV read to verify
    the BGR->RGB conversion happened.
    """
    frames = extract_frames(sample_video, config=test_config)

    # Manually read first frame with OpenCV (BGR)
    cap = cv2.VideoCapture(sample_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, bgr_frame = cap.read()
    cap.release()

    assert success, "Failed to read frame manually"

    # Convert to RGB manually
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Downscale if needed (match extraction logic)
    height = rgb_frame.shape[0]
    if height > 720:
        scale_factor = 720 / height
        new_width = int(rgb_frame.shape[1] * scale_factor)
        rgb_frame = cv2.resize(rgb_frame, (new_width, 720), interpolation=cv2.INTER_AREA)

    # Compare with extracted frame
    extracted_frame = frames[0].data

    # Shapes should match
    assert extracted_frame.shape == rgb_frame.shape, \
        f"Shape mismatch: {extracted_frame.shape} != {rgb_frame.shape}"

    # Pixel values should be very close (allow for minor rounding differences)
    diff = np.abs(extracted_frame.astype(float) - rgb_frame.astype(float))
    max_diff = diff.max()
    assert max_diff <= 2, f"RGB conversion check failed: max pixel diff = {max_diff}"


def test_corrupt_video_handling(tmp_path, sample_video, test_config):
    """Verify error handling when video is severely truncated.

    Severely truncated videos (missing moov atom) cannot be opened at all
    and should raise ValueError. This is acceptable behavior.
    """
    # Create a truncated copy of the sample video
    corrupt_video = tmp_path / "corrupt.mp4"

    # Copy only first 1MB (truncate the file - too short to have moov atom)
    with open(sample_video, "rb") as src:
        with open(corrupt_video, "wb") as dst:
            dst.write(src.read(1024 * 1024))  # 1MB

    # Severely truncated video should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        extract_frames(str(corrupt_video), config=test_config)

    assert "cannot open video" in str(exc_info.value).lower()


def test_downscaling(test_config):
    """Verify downscaling if 1080p video is available.

    This test is skipped if no 1080p videos are available in samples.
    """
    # Look for a high-resolution video (we know synthetic clips are 1280x720)
    # This test will be skipped for now but documents the expected behavior
    pytest.skip("Synthetic clips are already 720p, cannot test downscaling")

    # Future: If we add 1080p test videos, this would verify:
    # frames = extract_frames(hd_video, config=test_config)
    # for frame in frames:
    #     assert frame.data.shape[0] == 720, "Should downscale to 720p"


def test_file_not_found():
    """Verify FileNotFoundError when video doesn't exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_frames("nonexistent_video.mp4")

    assert "nonexistent_video.mp4" in str(exc_info.value)


def test_invalid_video(tmp_path):
    """Verify ValueError when file exists but is not a valid video."""
    # Create an empty file
    invalid_video = tmp_path / "invalid.mp4"
    invalid_video.write_text("not a video")

    with pytest.raises(ValueError) as exc_info:
        extract_frames(str(invalid_video))

    assert "cannot open video" in str(exc_info.value).lower()
