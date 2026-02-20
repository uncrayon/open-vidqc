"""Integration-style tests for classifier GPU/CPU device handling."""

from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from vidqc.models.classifier import VidQCClassifier


def _config(device: str = "auto") -> dict:
    return {
        "model": {
            "binary_threshold": 0.5,
            "xgboost": {
                "device": device,
                "n_estimators": 5,
                "max_depth": 3,
                "learning_rate": 0.1,
                "objective": "multi:softprob",
                "num_class": 3,
                "seed": 42,
            },
        }
    }


def _training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.random.randn(12, 4)
    y = np.array([0, 1, 2] * 4)
    feature_names = ["has_text_regions", "f1", "f2", "f3"]
    return X, y, feature_names


def test_train_applies_resolved_device(monkeypatch):
    captured = {}

    class FakeBooster:
        def __init__(self):
            self.params = []

        def set_param(self, params):
            self.params.append(params)

    def fake_train(params, dtrain, num_boost_round):
        captured["params"] = params.copy()
        captured["num_boost_round"] = num_boost_round
        return FakeBooster()

    monkeypatch.setattr(
        "vidqc.models.classifier.resolve_xgboost_device",
        lambda requested: ("cpu", "test"),
    )
    monkeypatch.setattr(
        "vidqc.models.classifier.xgb.DMatrix",
        lambda X, label, feature_names: ("dmatrix", X, label, feature_names),
    )
    monkeypatch.setattr("vidqc.models.classifier.xgb.train", fake_train)

    classifier = VidQCClassifier(_config())
    X, y, feature_names = _training_data()
    classifier.train(X, y, feature_names)

    assert captured["params"]["device"] == "cpu"


def test_train_sets_hist_for_cuda_when_missing(monkeypatch):
    captured = {}

    class FakeBooster:
        def set_param(self, params):
            return None

    def fake_train(params, dtrain, num_boost_round):
        captured["params"] = params.copy()
        return FakeBooster()

    monkeypatch.setattr(
        "vidqc.models.classifier.resolve_xgboost_device",
        lambda requested: ("cuda", "test"),
    )
    monkeypatch.setattr(
        "vidqc.models.classifier.xgb.DMatrix",
        lambda X, label, feature_names: ("dmatrix", X, label, feature_names),
    )
    monkeypatch.setattr("vidqc.models.classifier.xgb.train", fake_train)

    classifier = VidQCClassifier(_config())
    X, y, feature_names = _training_data()
    classifier.train(X, y, feature_names)

    assert captured["params"]["device"] == "cuda"
    assert captured["params"]["tree_method"] == "hist"


def test_load_applies_resolved_device(monkeypatch, tmp_path: Path):
    class FakeBooster:
        def __init__(self):
            self.feature_names = ["has_text_regions", "f1"]
            self.loaded_path = None
            self.params = []

        def load_model(self, path):
            self.loaded_path = path

        def set_param(self, params):
            self.params.append(params)

    fake_booster = FakeBooster()

    monkeypatch.setattr(
        "vidqc.models.classifier.resolve_xgboost_device",
        lambda requested: ("cpu", "test"),
    )
    monkeypatch.setattr("vidqc.models.classifier.xgb.Booster", lambda: fake_booster)

    model_path = tmp_path / "model.json"
    model_path.write_text("{}")

    classifier = VidQCClassifier(_config())
    classifier.load(str(model_path))

    assert fake_booster.loaded_path == str(model_path)
    assert fake_booster.params[-1] == {"device": "cpu"}


def test_load_falls_back_when_cuda_param_set_fails(monkeypatch, tmp_path: Path):
    class FakeBooster:
        def __init__(self):
            self.feature_names = ["has_text_regions", "f1"]
            self.params = []

        def load_model(self, path):
            return None

        def set_param(self, params):
            if params.get("device") == "cuda":
                raise xgb.core.XGBoostError("cuda not available")
            self.params.append(params)

    fake_booster = FakeBooster()

    monkeypatch.setattr(
        "vidqc.models.classifier.resolve_xgboost_device",
        lambda requested: ("cuda", "test"),
    )
    monkeypatch.setattr("vidqc.models.classifier.xgb.Booster", lambda: fake_booster)

    model_path = tmp_path / "model.json"
    model_path.write_text("{}")

    classifier = VidQCClassifier(_config(device="cuda"))
    classifier.load(str(model_path))

    assert fake_booster.params[-1] == {"device": "cpu"}


def test_train_retries_on_cpu_when_cuda_training_fails(monkeypatch):
    calls = []

    class FakeBooster:
        def __init__(self):
            self.params = []

        def set_param(self, params):
            self.params.append(params)

    def fake_train(params, dtrain, num_boost_round):
        calls.append(params.copy())
        if params.get("device") == "cuda":
            raise xgb.core.XGBoostError("cuda training failure")
        return FakeBooster()

    monkeypatch.setattr(
        "vidqc.models.classifier.resolve_xgboost_device",
        lambda requested: ("cuda", "test"),
    )
    monkeypatch.setattr(
        "vidqc.models.classifier.xgb.DMatrix",
        lambda X, label, feature_names: ("dmatrix", X, label, feature_names),
    )
    monkeypatch.setattr("vidqc.models.classifier.xgb.train", fake_train)

    classifier = VidQCClassifier(_config(device="cuda"))
    X, y, feature_names = _training_data()
    classifier.train(X, y, feature_names)

    assert len(calls) == 2
    assert calls[0]["device"] == "cuda"
    assert calls[1]["device"] == "cpu"


def test_train_does_not_retry_for_cpu_training_failure(monkeypatch):
    calls = []

    def fake_train(params, dtrain, num_boost_round):
        calls.append(params.copy())
        raise xgb.core.XGBoostError("cpu training failure")

    monkeypatch.setattr(
        "vidqc.models.classifier.resolve_xgboost_device",
        lambda requested: ("cpu", "test"),
    )
    monkeypatch.setattr(
        "vidqc.models.classifier.xgb.DMatrix",
        lambda X, label, feature_names: ("dmatrix", X, label, feature_names),
    )
    monkeypatch.setattr("vidqc.models.classifier.xgb.train", fake_train)

    classifier = VidQCClassifier(_config(device="cpu"))
    X, y, feature_names = _training_data()

    with pytest.raises(xgb.core.XGBoostError):
        classifier.train(X, y, feature_names)

    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"
