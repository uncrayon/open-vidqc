"""Common utilities for feature extraction.

Provides shared functions for SSIM computation, bbox operations,
and safe mathematical operations that prevent NaN/Inf values.
"""


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two frames.

    Converts to grayscale before computing SSIM for performance.

    Args:
        frame_a: First frame (RGB, uint8)
        frame_b: Second frame (RGB, uint8)

    Returns:
        SSIM value in [0.0, 1.0], where 1.0 = identical
    """
    # Convert RGB to grayscale
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

    # skimage default win_size=7 fails for small crops (<7px side).
    # Use the largest valid odd window up to 7, and fall back gracefully
    # for very tiny crops where SSIM is ill-defined.
    min_side = min(gray_a.shape[0], gray_a.shape[1])
    if min_side < 3:
        diff = np.mean(np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32)))
        return float(np.clip(1.0 - (diff / 255.0), 0.0, 1.0))

    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1

    # Compute SSIM with full data range for uint8
    return float(ssim(gray_a, gray_b, data_range=255, win_size=win_size))


def crop_bbox(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Extract bounding box region from frame.

    Args:
        frame: Input frame (H, W, C)
        bbox: [x1, y1, x2, y2] in pixel coordinates

    Returns:
        Cropped region (h, w, C) where h=y2-y1, w=x2-x1
    """
    x1, y1, x2, y2 = bbox
    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame[y1:y2, x1:x2]


def bbox_center(bbox: list[int]) -> tuple[float, float]:
    """Compute center point of bounding box.

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates

    Returns:
        (center_x, center_y) as floats
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def l2_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Compute Euclidean (L2) distance between two points.

    Args:
        point_a: (x, y) coordinates
        point_b: (x, y) coordinates

    Returns:
        L2 distance as float
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return float(np.sqrt(dx * dx + dy * dy))


def compute_iou(bbox1: list[int], bbox2: list[int]) -> float:
    """Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU in [0.0, 1.0]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return float(intersection / max(union, 1e-8))


# Safe math operations (prevent NaN/Inf)


def safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty list.

    Args:
        values: List of numeric values

    Returns:
        Mean or 0.0 if empty
    """
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values: list[float]) -> float:
    """Compute standard deviation, returning 0.0 for empty list.

    Uses ddof=0 to prevent NaN with single-element lists.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation or 0.0 if empty
    """
    return float(np.std(values, ddof=0)) if len(values) > 0 else 0.0


def safe_max(values: list[float], default: float = 0.0) -> float:
    """Compute max, returning default for empty list.

    Args:
        values: List of numeric values
        default: Value to return if list is empty

    Returns:
        Max value or default if empty
    """
    return float(np.max(values)) if len(values) > 0 else default


def safe_min(values: list[float], default: float = 0.0) -> float:
    """Compute min, returning default for empty list.

    Args:
        values: List of numeric values
        default: Value to return if list is empty

    Returns:
        Min value or default if empty
    """
    return float(np.min(values)) if len(values) > 0 else default


def safe_ratio(numerator: float, denominator: float, epsilon: float = 1e-8) -> float:
    """Compute ratio, preventing division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        epsilon: Small constant added to denominator if too small

    Returns:
        numerator / max(denominator, epsilon)
    """
    return numerator / max(abs(denominator), epsilon)
