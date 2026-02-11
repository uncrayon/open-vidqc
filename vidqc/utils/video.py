"""Video frame extraction utilities.

This module provides frame extraction from video files using OpenCV,
with configurable FPS sampling and downscaling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vidqc.config import get_config
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Frame:
    """A single video frame with metadata.

    Attributes:
        index: Sampled frame index (0, 1, 2...) in the extracted sequence,
               NOT the source frame number
        timestamp: Timestamp in seconds from start of video
        data: RGB image array (H, W, 3) with dtype uint8
    """
    index: int
    timestamp: float
    data: np.ndarray


def extract_frames(video_path: str, config: Optional[dict] = None) -> list[Frame]:
    """Extract frames from video file.

    This function samples frames at the configured FPS rate, downscales to the
    maximum resolution if needed, and converts from BGR to RGB color space.

    Args:
        video_path: Path to MP4 video file
        config: Configuration dict (uses get_config() if None)

    Returns:
        List of Frame objects sampled at config.video.sample_fps

    Raises:
        FileNotFoundError: If video_path doesn't exist
        ValueError: If video cannot be opened or contains no readable frames
    """
    # Load config
    if config is None:
        config = get_config()

    video_config = config.get("video", {})
    target_fps = video_config.get("sample_fps", 5)
    max_resolution = video_config.get("max_resolution", 720)
    max_duration_sec = video_config.get("max_duration_sec", 10)

    # Check file exists
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if source_fps <= 0:
            raise ValueError(f"Invalid FPS in video: {source_fps}")

        # Calculate sampling interval
        sample_every_n = max(1, round(source_fps / target_fps))

        logger.debug(
            f"Video: {video_path} | {width}x{height} @ {source_fps:.1f}fps | "
            f"{total_frames} frames | sampling every {sample_every_n} frames"
        )

        # Extract frames
        frames = []
        sampled_index = 0
        source_frame_index = 0

        while True:
            # Check duration limit
            timestamp = source_frame_index / source_fps
            if timestamp >= max_duration_sec:
                logger.debug(f"Reached max duration {max_duration_sec}s, stopping extraction")
                break

            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
            success, frame = cap.read()

            # Check for end of video or read failure
            if not success or frame is None:
                if source_frame_index < total_frames:
                    logger.warning(
                        f"Frame read failed at index {source_frame_index}/{total_frames}. "
                        f"Video may be corrupt. Returning {len(frames)} partial frames."
                    )
                break

            # Convert BGR to RGB (CRITICAL: OpenCV reads BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Downscale if needed (maintain aspect ratio)
            if height > max_resolution:
                scale_factor = max_resolution / height
                new_width = int(width * scale_factor)
                new_height = max_resolution
                frame = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )

            # Create Frame object
            frames.append(Frame(
                index=sampled_index,
                timestamp=timestamp,
                data=frame
            ))

            # Move to next sample
            sampled_index += 1
            source_frame_index += sample_every_n

        if len(frames) == 0:
            raise ValueError(f"No readable frames in video: {video_path}")

        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    finally:
        cap.release()
