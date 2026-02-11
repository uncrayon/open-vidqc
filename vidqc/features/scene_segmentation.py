"""Scene cut detection for temporal feature extraction.

Implements scene cut detection with post-cut stability verification to avoid
confusing genuine scene transitions with temporal flicker artifacts.
"""

from dataclasses import dataclass

import numpy as np

from vidqc.features.common import compute_ssim
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SceneCut:
    """Detected scene cut location.

    Attributes:
        frame_index: Index where the cut occurs (between frame_index and frame_index+1)
        ssim: SSIM value at the cut point
        post_cut_stability: Mean SSIM in the stability window after cut
    """

    frame_index: int
    ssim: float
    post_cut_stability: float


@dataclass
class Segment:
    """Continuous scene segment between cuts.

    Attributes:
        start_frame: First frame index in segment (inclusive)
        end_frame: Last frame index in segment (inclusive)
        length: Number of frames in segment
    """

    start_frame: int
    end_frame: int
    length: int


def detect_scene_cuts(frames: list[np.ndarray], config: dict) -> list[SceneCut]:
    """Detect scene cuts using SSIM threshold and post-cut stability check.

    A scene cut is confirmed when:
    1. SSIM between consecutive frames < cut_ssim_threshold (potential cut)
    2. Mean SSIM in post_cut_stability_window > post_cut_stability_min (stabilized)

    Late-clip edge case: Stop checking for cuts when fewer than post_cut_stability_window
    frames remain, to avoid incomplete stability checks.

    Args:
        frames: List of RGB frames (uint8)
        config: Configuration dict with temporal.scene_cut parameters

    Returns:
        List of confirmed SceneCut objects, sorted by frame_index
    """
    if len(frames) < 2:
        return []

    scene_config = config["temporal"]["scene_cut"]
    cut_threshold = scene_config["cut_ssim_threshold"]
    stability_window = scene_config["post_cut_stability_window"]
    stability_min = scene_config["post_cut_stability_min"]

    cuts = []

    # Calculate how far we can check for cuts (leave room for stability window)
    max_cut_check = len(frames) - stability_window - 1

    for i in range(max_cut_check):
        # Check SSIM between frame i and i+1
        ssim = compute_ssim(frames[i], frames[i + 1])

        if ssim < cut_threshold:
            # Potential cut detected, check post-cut stability
            # Compute SSIM for frames in stability window (i+1 to i+1+window)
            stability_ssims = []
            for j in range(i + 1, min(i + 1 + stability_window, len(frames) - 1)):
                stability_ssim = compute_ssim(frames[j], frames[j + 1])
                stability_ssims.append(stability_ssim)

            if len(stability_ssims) > 0:
                mean_stability = float(np.mean(stability_ssims))

                if mean_stability > stability_min:
                    # Confirmed scene cut
                    cuts.append(
                        SceneCut(
                            frame_index=i, ssim=ssim, post_cut_stability=mean_stability
                        )
                    )
                    logger.debug(
                        f"Scene cut detected at frame {i}: "
                        f"ssim={ssim:.3f}, stability={mean_stability:.3f}"
                    )

    return cuts


def segment_by_cuts(
    num_frames: int, cuts: list[SceneCut], config: dict
) -> list[Segment]:
    """Divide video into segments based on scene cuts.

    Args:
        num_frames: Total number of frames in video
        cuts: List of detected scene cuts
        config: Configuration dict with temporal.scene_cut.min_segment_frames

    Returns:
        List of Segment objects, filtered to exclude segments shorter than min_segment_frames
    """
    min_length = config["temporal"]["scene_cut"]["min_segment_frames"]

    if len(cuts) == 0:
        # No cuts, entire video is one segment
        if num_frames >= min_length:
            return [Segment(start_frame=0, end_frame=num_frames - 1, length=num_frames)]
        else:
            return []

    segments = []

    # First segment: 0 to first cut
    first_cut = cuts[0].frame_index
    if first_cut >= min_length:
        segments.append(Segment(start_frame=0, end_frame=first_cut, length=first_cut + 1))

    # Middle segments: between cuts
    for i in range(len(cuts) - 1):
        start = cuts[i].frame_index + 1
        end = cuts[i + 1].frame_index
        length = end - start + 1
        if length >= min_length:
            segments.append(Segment(start_frame=start, end_frame=end, length=length))

    # Last segment: last cut to end
    last_cut = cuts[-1].frame_index
    start = last_cut + 1
    end = num_frames - 1
    length = end - start + 1
    if length >= min_length:
        segments.append(Segment(start_frame=start, end_frame=end, length=length))

    logger.debug(
        f"Segmented {num_frames} frames into {len(segments)} segments "
        f"(min_length={min_length})"
    )

    return segments
