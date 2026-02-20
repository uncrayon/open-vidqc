"""Tests for GPU-related config validation."""

import pytest

from vidqc.config import validate_config


def _base_model_config() -> dict:
    return {
        "model": {
            "binary_threshold": 0.5,
            "xgboost": {
                "objective": "multi:softprob",
                "num_class": 3,
                "seed": 42,
            },
        }
    }


@pytest.mark.parametrize("value", ["auto", "cpu", "cuda"])
def test_model_xgboost_device_valid_values(value):
    cfg = _base_model_config()
    cfg["model"]["xgboost"]["device"] = value
    validate_config(cfg)


def test_model_xgboost_device_rejects_invalid_value():
    cfg = _base_model_config()
    cfg["model"]["xgboost"]["device"] = "gpu_hist"

    with pytest.raises(ValueError, match="model.xgboost.device"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [None, 123, "bad", [1, 2]])
def test_model_xgboost_rejects_non_mapping(value):
    cfg = _base_model_config()
    cfg["model"]["xgboost"] = value

    with pytest.raises(ValueError, match="model.xgboost"):
        validate_config(cfg)
