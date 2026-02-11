"""Feature extraction orchestrator combining text and temporal features.

This module provides a unified interface for extracting the complete
43-feature vector (29 text + 14 temporal) with evidence data from both extractors.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from vidqc.config import get_config
from vidqc.features.feature_manifest import (
    COMBINED_FEATURE_NAMES,
    EXPECTED_COMBINED_FEATURE_COUNT,
    TEXT_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    validate_combined_features,
)
from vidqc.features.temporal_flicker import extract_temporal_features
from vidqc.features.text_inconsistency import extract_text_features
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


def extract_all_features(
    video_path: str, config: Optional[dict] = None
) -> tuple[np.ndarray, dict]:
    """Extract combined text + temporal features.

    This is the single entry point for feature extraction. It orchestrates
    both text and temporal feature extraction and combines them into a
    single 43-element feature vector.

    Args:
        video_path: Path to video file
        config: Optional config dict (uses default if None)

    Returns:
        features: 43-element numpy array (29 text + 14 temporal)
        evidence: Combined evidence dict with keys:
            - text_evidence: Evidence dict from text extractor (or None)
            - temporal_evidence: Evidence dict from temporal extractor (or None)
            - feature_names: List of 43 feature names
            - error: Error message if extraction failed (or None)

    Raises:
        FileNotFoundError: If video file does not exist
        ValueError: If feature extraction fails critically
    """
    if config is None:
        config = get_config()

    # Check file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Extracting all features from {video_path}")

    # Extract text features
    try:
        text_feats = extract_text_features(video_path, config)
    except Exception as e:
        logger.error(f"Text feature extraction failed: {e}")
        # Re-raise critical errors
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Text feature extraction failed: {e}")

    # Extract temporal features
    try:
        temporal_feats = extract_temporal_features(video_path, config)
    except Exception as e:
        logger.error(f"Temporal feature extraction failed: {e}")
        # Re-raise critical errors
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Temporal feature extraction failed: {e}")

    # Combine into single array (order matches COMBINED_FEATURE_NAMES)
    combined_dict = {}
    for name in TEXT_FEATURE_NAMES:
        combined_dict[name] = text_feats[name]
    for name in TEMPORAL_FEATURE_NAMES:
        combined_dict[name] = temporal_feats[name]

    # Validate combined features
    validate_combined_features(combined_dict)

    # Convert to numpy array (preserves order from COMBINED_FEATURE_NAMES)
    features = np.array([combined_dict[name] for name in COMBINED_FEATURE_NAMES])

    # Build evidence dict
    # Note: Evidence data is currently logged but not returned by extractors.
    # For now, we return None for both until we refactor extractors to return evidence.
    evidence = {
        "text_evidence": None,  # TODO: Modify text extractor to return evidence
        "temporal_evidence": None,  # TODO: Modify temporal extractor to return evidence
        "feature_names": COMBINED_FEATURE_NAMES,
        "error": None,
    }

    logger.info(
        f"Extracted {len(features)} combined features "
        f"(text: {len(TEXT_FEATURE_NAMES)}, temporal: {len(TEMPORAL_FEATURE_NAMES)})"
    )

    # Sanity check
    assert len(features) == EXPECTED_COMBINED_FEATURE_COUNT, (
        f"Feature count mismatch: expected {EXPECTED_COMBINED_FEATURE_COUNT}, "
        f"got {len(features)}"
    )

    return features, evidence
