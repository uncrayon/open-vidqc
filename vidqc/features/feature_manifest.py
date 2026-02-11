"""Feature vector manifest for TEXT_INCONSISTENCY detection.

Defines the complete list of text features in extraction order,
ensuring consistent feature vector shape across all clips.
"""

# Complete list of text features (29 features)
TEXT_FEATURE_NAMES = [
    # Step 0: Text presence gate (2 features)
    "has_text_regions",
    "frames_with_text_ratio",
    # Step 3: Per-region features - edit distance (3 stats)
    "edit_distance_mean",
    "edit_distance_max",
    "edit_distance_std",
    # Step 3: Per-region features - confidence delta (3 stats)
    "confidence_delta_mean",
    "confidence_delta_max",
    "confidence_delta_std",
    # Step 3: Per-region features - bbox shift (3 stats)
    "bbox_shift_mean",
    "bbox_shift_max",
    "bbox_shift_std",
    # Step 3: Per-region features - SSIM text crop (3 stats)
    "ssim_text_crop_mean",
    "ssim_text_crop_max",
    "ssim_text_crop_std",
    # Step 3: Per-region features - char count delta (3 stats)
    "char_count_delta_mean",
    "char_count_delta_max",
    "char_count_delta_std",
    # Step 4: False positive suppression - global SSIM at instability (3 stats)
    "global_ssim_at_instability_mean",
    "global_ssim_at_instability_min",
    "global_ssim_at_instability_std",
    # Step 4: False positive suppression - confidence variance rolling (2 stats)
    "confidence_variance_rolling_mean",
    "confidence_variance_rolling_max",
    # Step 4: False positive suppression - confidence monotonic ratio (1 stat)
    "confidence_monotonic_ratio",
    # Step 4: False positive suppression - substitution ratio (3 stats)
    "substitution_ratio_mean",
    "substitution_ratio_max",
    "substitution_ratio_std",
    # Step 4: False positive suppression - text vs scene instability ratio (2 stats)
    "text_vs_scene_instability_ratio_mean",
    "text_vs_scene_instability_ratio_max",
    # Metadata (1 feature)
    "num_text_regions_detected",
]

EXPECTED_TEXT_FEATURE_COUNT = 29


def get_zero_features() -> dict[str, float]:
    """Return complete feature dict with all values set to 0.0.

    Used when has_text_regions=0 to ensure consistent feature vector shape.

    Returns:
        Dict mapping feature names to 0.0
    """
    return {name: 0.0 for name in TEXT_FEATURE_NAMES}


def validate_features(features: dict[str, float]) -> None:
    """Validate that feature dict is complete and valid.

    Args:
        features: Feature dict to validate

    Raises:
        ValueError: If features are invalid (missing keys, NaN, Inf, wrong count)
    """
    # Check count
    if len(features) != EXPECTED_TEXT_FEATURE_COUNT:
        missing = set(TEXT_FEATURE_NAMES) - set(features.keys())
        extra = set(features.keys()) - set(TEXT_FEATURE_NAMES)
        raise ValueError(
            f"Expected {EXPECTED_TEXT_FEATURE_COUNT} features, got {len(features)}. "
            f"Missing: {missing}, Extra: {extra}"
        )

    # Check for missing keys
    missing = set(TEXT_FEATURE_NAMES) - set(features.keys())
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Check for NaN/Inf
    import numpy as np

    for name, value in features.items():
        if np.isnan(value):
            raise ValueError(f"Feature '{name}' is NaN")
        if np.isinf(value):
            raise ValueError(f"Feature '{name}' is Inf")
