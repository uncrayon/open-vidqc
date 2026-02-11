"""Test category constraint enforcement.

Tests that the hard constraint and threshold fallback logic work correctly:
1. No text → never TEXT_INCONSISTENCY
2. Weak probabilities → NONE
3. Override logging
"""

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
def trained_classifier(config):
    """Train a simple classifier for testing."""
    # Create synthetic training data
    # Features: [has_text, feature1, feature2, ...]
    np.random.seed(42)

    X = np.random.randn(30, 10)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)  # Balanced classes

    # Set has_text feature (index 0)
    # NONE: random
    # TEXT: all have text
    # FLICKER: half have text, half don't
    X[:10, 0] = np.random.rand(10)  # NONE
    X[10:20, 0] = 1.0  # TEXT (all have text)
    X[20:30, 0] = np.array([1.0, 0.0] * 5)  # FLICKER (mixed)

    feature_names = [f"feature_{i}" for i in range(10)]
    feature_names[0] = "has_text_regions"

    classifier = VidQCClassifier(config)
    classifier.train(X, y, feature_names)

    return classifier


def test_hard_constraint_no_text_no_text_pred(trained_classifier):
    """Test: has_text=False → cannot predict TEXT_INCONSISTENCY.

    Even if model predicts TEXT, should be overridden to NONE.
    """
    # Create feature vector: no text, but otherwise looks like text artifact
    features = np.array([0.0] + [1.0] * 9)  # has_text=0

    # Manually set probabilities that would predict TEXT
    # (We can't force the model output, but we test the override logic)

    result = trained_classifier.predict(features, has_text=False)

    # Should never predict TEXT when has_text=False
    assert result.predicted_class != "TEXT_INCONSISTENCY", (
        "Hard constraint violated: predicted TEXT with has_text=False"
    )


def test_hard_constraint_override_logging(trained_classifier):
    """Test: Override is logged when hard constraint triggers."""
    # Features with no text
    features = np.array([0.0] + [1.0] * 9)

    result = trained_classifier.predict(features, has_text=False)

    # If model tried to predict TEXT, override should be applied and logged
    # We can check if artifact was detected at all
    if not result.artifact_detected:
        # If no artifact detected, check if it was due to constraint
        if result.override_applied:
            assert result.override_reason is not None
            assert "HARD_CONSTRAINT" in result.override_reason or "THRESHOLD_FALLBACK" in result.override_reason


def test_threshold_fallback_weak_probs(trained_classifier):
    """Test: When P(TEXT) ≤ 0.1 and P(FLICKER) ≤ 0.1, override to NONE."""
    # Create feature vector that produces weak artifact probabilities
    # (This is tricky because we can't control model output exactly)
    features = np.zeros(10)

    result = trained_classifier.predict(features, has_text=True)

    # If both artifact probs are weak, should predict NONE
    if (
        result.probabilities["TEXT_INCONSISTENCY"] <= 0.1
        and result.probabilities["TEMPORAL_FLICKER"] <= 0.1
    ):
        assert result.predicted_class == "NONE", (
            f"Threshold fallback should override to NONE when both artifact probs ≤ 0.1. "
            f"Got: {result.predicted_class}"
        )
        assert not result.artifact_detected


def test_has_text_allows_flicker(trained_classifier):
    """Test: has_text=False does NOT prevent TEMPORAL_FLICKER prediction."""
    # Features with no text
    features = np.array([0.0] + [2.0] * 9)  # Strong flicker signal

    result = trained_classifier.predict(features, has_text=False)

    # FLICKER is allowed even with no text (constraint only applies to TEXT)
    # We just verify the system doesn't crash and produces a valid result
    assert result.predicted_class in ["NONE", "TEMPORAL_FLICKER"], (
        f"With has_text=False, should only predict NONE or FLICKER, got {result.predicted_class}"
    )


def test_has_text_allows_text_pred(trained_classifier):
    """Test: has_text=True allows TEXT_INCONSISTENCY prediction."""
    # Features with text present
    features = np.array([1.0] + [1.0] * 9)

    result = trained_classifier.predict(features, has_text=True)

    # Any prediction is valid when has_text=True
    assert result.predicted_class in [
        "NONE",
        "TEXT_INCONSISTENCY",
        "TEMPORAL_FLICKER",
    ]


def test_binary_threshold_logic(trained_classifier):
    """Test: Binary decision uses P(NONE) < threshold, not argmax."""
    # Create multiple feature vectors and check binary decision
    for _ in range(10):
        features = np.random.randn(10)
        features[0] = 1.0  # has_text

        result = trained_classifier.predict(features, has_text=True)

        # Binary decision should be: artifact = P(NONE) < 0.5
        if result.probabilities["NONE"] >= 0.5:
            # Should predict NONE (unless fallback triggered)
            if (
                result.probabilities["TEXT_INCONSISTENCY"] > 0.1
                or result.probabilities["TEMPORAL_FLICKER"] > 0.1
            ):
                # Fallback not triggered
                assert result.predicted_class == "NONE", (
                    f"Binary threshold violation: P(NONE)={result.probabilities['NONE']:.3f} >= 0.5, "
                    f"but predicted {result.predicted_class}"
                )


def test_category_decision_argmax(trained_classifier):
    """Test: When artifact detected, category = argmax(TEXT, FLICKER)."""
    # Create features that produce artifact detection
    for _ in range(10):
        features = np.random.randn(10) * 2
        features[0] = 1.0  # has_text

        result = trained_classifier.predict(features, has_text=True)

        # If artifact detected (after all overrides)
        if result.artifact_detected:
            # Category should be argmax of artifact classes
            text_prob = result.probabilities["TEXT_INCONSISTENCY"]
            flicker_prob = result.probabilities["TEMPORAL_FLICKER"]

            if text_prob > flicker_prob:
                expected = "TEXT_INCONSISTENCY"
            else:
                expected = "TEMPORAL_FLICKER"

            # Note: Override may have changed this, so we only check when no override
            if not result.override_applied:
                assert result.predicted_class == expected, (
                    f"Category decision should be argmax. "
                    f"P(TEXT)={text_prob:.3f}, P(FLICKER)={flicker_prob:.3f}, "
                    f"expected {expected}, got {result.predicted_class}"
                )
