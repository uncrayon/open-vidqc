"""Test deterministic training and prediction.

Tests that:
1. Training twice with same seed produces identical models
2. Predicting twice produces identical probabilities
3. Loading saved model produces identical predictions
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vidqc.models.classifier import VidQCClassifier


@pytest.fixture
def config():
    """Standard config for tests."""
    return {
        "model": {
            "binary_threshold": 0.5,
            "xgboost": {
                "device": "cpu",
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "multi:softprob",
                "num_class": 3,
                "seed": 42,
            },
        }
    }


@pytest.fixture
def training_data():
    """Generate synthetic training data."""
    np.random.seed(42)

    X = np.random.randn(30, 10)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)

    feature_names = [f"feature_{i}" for i in range(10)]
    feature_names[0] = "has_text_regions"

    return X, y, feature_names


def test_training_determinism(config, training_data):
    """Test: Training twice with same seed produces identical predictions."""
    X, y, feature_names = training_data

    # Train first model
    classifier1 = VidQCClassifier(config)
    classifier1.train(X, y, feature_names)

    # Train second model
    classifier2 = VidQCClassifier(config)
    classifier2.train(X, y, feature_names)

    # Test features
    test_features = np.random.randn(10)
    test_features[0] = 1.0  # has_text

    # Predict with both models
    result1 = classifier1.predict(test_features, has_text=True)
    result2 = classifier2.predict(test_features, has_text=True)

    # Probabilities should be identical
    for class_name in ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"]:
        prob1 = result1.probabilities[class_name]
        prob2 = result2.probabilities[class_name]
        assert np.isclose(prob1, prob2, rtol=1e-6), (
            f"Determinism violation: {class_name} probability differs. "
            f"Model 1: {prob1:.10f}, Model 2: {prob2:.10f}"
        )

    # Predictions should be identical
    assert result1.predicted_class == result2.predicted_class, (
        f"Determinism violation: predicted class differs. "
        f"Model 1: {result1.predicted_class}, Model 2: {result2.predicted_class}"
    )

    assert result1.artifact_detected == result2.artifact_detected, (
        f"Determinism violation: artifact detection differs. "
        f"Model 1: {result1.artifact_detected}, Model 2: {result2.artifact_detected}"
    )


def test_prediction_determinism(config, training_data):
    """Test: Predicting twice on same model produces identical results."""
    X, y, feature_names = training_data

    # Train model
    classifier = VidQCClassifier(config)
    classifier.train(X, y, feature_names)

    # Test features
    test_features = np.random.randn(10)
    test_features[0] = 1.0  # has_text

    # Predict twice
    result1 = classifier.predict(test_features, has_text=True)
    result2 = classifier.predict(test_features, has_text=True)

    # Results should be identical
    for class_name in ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"]:
        prob1 = result1.probabilities[class_name]
        prob2 = result2.probabilities[class_name]
        assert prob1 == prob2, (
            f"Prediction not deterministic: {class_name} probability differs. "
            f"First: {prob1:.10f}, Second: {prob2:.10f}"
        )

    assert result1.predicted_class == result2.predicted_class
    assert result1.artifact_detected == result2.artifact_detected


def test_save_load_determinism(config, training_data):
    """Test: Loading saved model produces identical predictions."""
    X, y, feature_names = training_data

    # Train model
    classifier1 = VidQCClassifier(config)
    classifier1.train(X, y, feature_names)

    # Test features
    test_features = np.random.randn(10)
    test_features[0] = 1.0  # has_text

    # Predict with original model
    result1 = classifier1.predict(test_features, has_text=True)

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.json"
        classifier1.save(str(model_path))

        # Load model
        classifier2 = VidQCClassifier(config)
        classifier2.load(str(model_path))

        # Predict with loaded model
        result2 = classifier2.predict(test_features, has_text=True)

    # Results should be identical
    for class_name in ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"]:
        prob1 = result1.probabilities[class_name]
        prob2 = result2.probabilities[class_name]
        assert np.isclose(prob1, prob2, rtol=1e-6), (
            f"Save/load not deterministic: {class_name} probability differs. "
            f"Original: {prob1:.10f}, Loaded: {prob2:.10f}"
        )

    assert result1.predicted_class == result2.predicted_class
    assert result1.artifact_detected == result2.artifact_detected


def test_numpy_seed_isolation(config, training_data):
    """Test: NumPy seed doesn't affect training determinism across runs."""
    X, y, feature_names = training_data

    # Train with different numpy random states
    np.random.seed(100)
    classifier1 = VidQCClassifier(config)
    classifier1.train(X, y, feature_names)

    np.random.seed(999)
    classifier2 = VidQCClassifier(config)
    classifier2.train(X, y, feature_names)

    # Test features
    test_features = np.random.randn(10)
    test_features[0] = 1.0

    # Predictions should still be identical (XGBoost seed should dominate)
    result1 = classifier1.predict(test_features, has_text=True)
    result2 = classifier2.predict(test_features, has_text=True)

    # Allow small differences due to numpy ops, but predictions should match
    assert result1.predicted_class == result2.predicted_class, (
        f"XGBoost seed should ensure determinism regardless of numpy seed. "
        f"Got different predictions: {result1.predicted_class} vs {result2.predicted_class}"
    )


def test_feature_importance_determinism(config, training_data):
    """Test: Feature importances are deterministic across training runs."""
    X, y, feature_names = training_data

    # Train first model
    classifier1 = VidQCClassifier(config)
    classifier1.train(X, y, feature_names)
    importances1 = classifier1.get_feature_importances()

    # Train second model
    classifier2 = VidQCClassifier(config)
    classifier2.train(X, y, feature_names)
    importances2 = classifier2.get_feature_importances()

    # Importances should be identical
    # Note: importance dict may not contain all features (zero-importance features omitted)
    all_features = set(importances1.keys()) | set(importances2.keys())

    for feature in all_features:
        imp1 = importances1.get(feature, 0.0)
        imp2 = importances2.get(feature, 0.0)
        assert np.isclose(imp1, imp2, rtol=1e-6), (
            f"Feature importance not deterministic for '{feature}'. "
            f"Model 1: {imp1:.10f}, Model 2: {imp2:.10f}"
        )
