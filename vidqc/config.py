"""Configuration management for vidqc.

This module provides a singleton configuration system with lazy loading,
YAML-based defaults, and support for CLI overrides.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import yaml

from vidqc.utils.logging import get_logger

logger = get_logger(__name__)

# Global config singleton
_config: Optional[dict] = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict.

    Args:
        base: Base configuration dict
        override: Override values to merge in

    Returns:
        Merged configuration dict (modifies base in-place and returns it)
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(config: dict) -> None:
    """Validate configuration structure and values.

    Args:
        config: Configuration dict to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate video config
    if "video" in config:
        video = config["video"]
        if "sample_fps" in video and video["sample_fps"] <= 0:
            raise ValueError(
                f"Config validation failed: video.sample_fps must be positive "
                f"(got {video['sample_fps']})"
            )
        if "max_duration_sec" in video and video["max_duration_sec"] <= 0:
            raise ValueError(
                f"Config validation failed: video.max_duration_sec must be positive "
                f"(got {video['max_duration_sec']})"
            )
        if "max_resolution" in video and video["max_resolution"] <= 0:
            raise ValueError(
                f"Config validation failed: video.max_resolution must be positive "
                f"(got {video['max_resolution']})"
            )

    # Validate text config
    if "text" in config:
        text = config["text"]
        if "ocr_languages" in text:
            if not isinstance(text["ocr_languages"], list) or len(text["ocr_languages"]) == 0:
                raise ValueError(
                    "Config validation failed: text.ocr_languages must be a non-empty list"
                )
        if "ocr_gpu" in text:
            raw = text["ocr_gpu"]
            if isinstance(raw, bool):
                pass  # True / False are always valid
            elif isinstance(raw, str):
                if raw.lower().strip() not in {"auto", "true", "false"}:
                    raise ValueError(
                        f"Config validation failed: text.ocr_gpu must be one of "
                        f"true, false, or 'auto' (got {raw!r})"
                    )
            else:
                raise ValueError(
                    f"Config validation failed: text.ocr_gpu must be one of "
                    f"true, false, or 'auto' (got {raw!r})"
                )
        if "ocr_skip_ssim_threshold" in text:
            threshold = text["ocr_skip_ssim_threshold"]
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"Config validation failed: text.ocr_skip_ssim_threshold must be in [0, 1] "
                    f"(got {threshold})"
                )
            if threshold < 0.99:
                logger.warning(
                    f"text.ocr_skip_ssim_threshold is {threshold}, which is below the "
                    f"recommended minimum of 0.99. This may cause OCR to be skipped on "
                    f"frames where small text regions mutated."
                )
        if "bbox_iou_threshold" in text and not (0.0 <= text["bbox_iou_threshold"] <= 1.0):
            raise ValueError(
                f"Config validation failed: text.bbox_iou_threshold must be in [0, 1] "
                f"(got {text['bbox_iou_threshold']})"
            )

        # Validate false positive suppression
        if "fp_suppression" in text:
            fp = text["fp_suppression"]
            if "global_ssim_floor" in fp and not (0.0 <= fp["global_ssim_floor"] <= 1.0):
                raise ValueError(
                    f"Config validation failed: text.fp_suppression.global_ssim_floor "
                    f"must be in [0, 1] (got {fp['global_ssim_floor']})"
                )
            if "confidence_rolling_window" in fp and fp["confidence_rolling_window"] <= 0:
                raise ValueError(
                    "Config validation failed: text.fp_suppression.confidence_rolling_window "
                    f"must be positive (got {fp['confidence_rolling_window']})"
                )
            if "monotonic_ratio_threshold" in fp and not (0.0 <= fp["monotonic_ratio_threshold"] <= 1.0):
                raise ValueError(
                    f"Config validation failed: text.fp_suppression.monotonic_ratio_threshold "
                    f"must be in [0, 1] (got {fp['monotonic_ratio_threshold']})"
                )
            if "substitution_ratio_min" in fp and not (0.0 <= fp["substitution_ratio_min"] <= 1.0):
                raise ValueError(
                    f"Config validation failed: text.fp_suppression.substitution_ratio_min "
                    f"must be in [0, 1] (got {fp['substitution_ratio_min']})"
                )

        # Validate segment display fallback config
        if "segment_display_fallback" in text:
            sdf = text["segment_display_fallback"]
            allowed_variants = {"raw", "inv_clahe", "inv_clahe_up2"}
            if "preprocess_variants" in sdf:
                variants = sdf["preprocess_variants"]
                if not isinstance(variants, list) or len(variants) == 0:
                    raise ValueError(
                        "Config validation failed: text.segment_display_fallback.preprocess_variants "
                        "must be a non-empty list"
                    )
                invalid = [v for v in variants if v not in allowed_variants]
                if invalid:
                    raise ValueError(
                        f"Config validation failed: text.segment_display_fallback.preprocess_variants "
                        f"contains invalid values {invalid}. Allowed: {sorted(allowed_variants)}"
                    )

    # Validate temporal config
    if "temporal" in config:
        temporal = config["temporal"]
        if "ssim_spike_threshold" in temporal and not (0.0 <= temporal["ssim_spike_threshold"] <= 1.0):
            raise ValueError(
                f"Config validation failed: temporal.ssim_spike_threshold must be in [0, 1] "
                f"(got {temporal['ssim_spike_threshold']})"
            )
        if "patch_grid" in temporal:
            grid = temporal["patch_grid"]
            if not isinstance(grid, list) or len(grid) != 2:
                raise ValueError(
                    "Config validation failed: temporal.patch_grid must be a list of 2 integers"
                )
            if not all(isinstance(x, int) and x > 0 for x in grid):
                raise ValueError(
                    f"Config validation failed: temporal.patch_grid values must be positive integers "
                    f"(got {grid})"
                )

        # Validate scene cut config
        if "scene_cut" in temporal:
            sc = temporal["scene_cut"]
            if "cut_ssim_threshold" in sc and not (0.0 <= sc["cut_ssim_threshold"] <= 1.0):
                raise ValueError(
                    f"Config validation failed: temporal.scene_cut.cut_ssim_threshold "
                    f"must be in [0, 1] (got {sc['cut_ssim_threshold']})"
                )
            if "post_cut_stability_window" in sc and sc["post_cut_stability_window"] <= 0:
                raise ValueError(
                    "Config validation failed: temporal.scene_cut.post_cut_stability_window "
                    f"must be positive (got {sc['post_cut_stability_window']})"
                )
            if "post_cut_stability_min" in sc and not (0.0 <= sc["post_cut_stability_min"] <= 1.0):
                raise ValueError(
                    f"Config validation failed: temporal.scene_cut.post_cut_stability_min "
                    f"must be in [0, 1] (got {sc['post_cut_stability_min']})"
                )
            if "min_segment_frames" in sc and sc["min_segment_frames"] <= 0:
                raise ValueError(
                    "Config validation failed: temporal.scene_cut.min_segment_frames "
                    f"must be positive (got {sc['min_segment_frames']})"
                )

    # Validate model config
    if "model" in config:
        model = config["model"]
        if "binary_threshold" in model and not (0.0 <= model["binary_threshold"] <= 1.0):
            raise ValueError(
                f"Config validation failed: model.binary_threshold must be in [0, 1] "
                f"(got {model['binary_threshold']})"
            )
        if "xgboost" in model:
            xgboost_config = model["xgboost"]
            if not isinstance(xgboost_config, Mapping):
                raise ValueError(
                    "Config validation failed: model.xgboost must be a mapping "
                    f"(got {xgboost_config!r})"
                )

            if "device" in xgboost_config:
                raw = xgboost_config["device"]
                if not isinstance(raw, str) or raw.lower().strip() not in {
                    "auto",
                    "cpu",
                    "cuda",
                }:
                    logger.debug(f"LOL, check your config: {xgboost_config}")
                    raise ValueError(
                        "Config validation failed: model.xgboost.device must be one of "
                        "'auto', 'cpu', or 'cuda' "
                        f"(got {raw!r})"
                    )
                logger.info(f"Loaded XGBoost device from {xgboost_config}")


def load_config(path: str = "config.yaml", overrides: Optional[dict] = None) -> dict:
    """Load configuration from YAML file and apply overrides.

    Args:
        path: Path to YAML config file (default: config.yaml)
        overrides: Optional dict of overrides to deep-merge into config

    Returns:
        Configuration dict

    Raises:
        ValueError: If config validation fails
    """
    global _config

    # Load YAML file
    config_path = Path(path)
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML syntax error in {path}: {e}")
    else:
        logger.warning(f"Config file {path} not found, using defaults")
        config = {}

    # Apply overrides
    if overrides:
        config = _deep_merge(config, overrides)
        logger.debug(f"Applied config overrides: {overrides}")

    # Validate
    validate_config(config)

    # Store in global singleton
    _config = config
    return config


def get_config() -> dict:
    """Get current configuration (lazy-loads default if needed).

    Returns:
        Configuration dict

    Raises:
        ValueError: If config validation fails
    """
    global _config
    if _config is None:
        load_config()
    return _config
