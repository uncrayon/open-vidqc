"""Schema validation tests.

Tests the Pydantic models and evidence contract enforcement.
"""

import pytest
from pydantic import ValidationError

from vidqc.schemas import ArtifactCategory, Evidence, Prediction


def test_valid_prediction_with_artifact():
    """artifact=true with valid Evidence should not raise error."""
    evidence = Evidence(
        timestamps=[0.5, 1.0],
        bounding_boxes=[[10, 20, 100, 200], [15, 25, 105, 205]],
        frames=[2, 5],
        notes="Text changed from 'Hello' to 'Hallo' in bbox (10,20,100,200)"
    )

    prediction = Prediction(
        clip_id="test_001",
        artifact=True,
        confidence=0.95,
        artifact_category=ArtifactCategory.TEXT_INCONSISTENCY,
        class_probabilities={
            "NONE": 0.05,
            "TEXT_INCONSISTENCY": 0.90,
            "TEMPORAL_FLICKER": 0.05
        },
        evidence=evidence
    )

    assert prediction.artifact is True
    assert prediction.evidence is not None


def test_valid_prediction_no_artifact():
    """artifact=false with evidence=None should not raise error."""
    prediction = Prediction(
        clip_id="test_002",
        artifact=False,
        confidence=0.85,
        artifact_category=ArtifactCategory.NONE,
        class_probabilities={
            "NONE": 0.85,
            "TEXT_INCONSISTENCY": 0.10,
            "TEMPORAL_FLICKER": 0.05
        },
        evidence=None
    )

    assert prediction.artifact is False
    assert prediction.evidence is None


def test_missing_fields():
    """Missing required field clip_id should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            artifact=False,
            confidence=0.8,
            artifact_category=ArtifactCategory.NONE,
            class_probabilities={"NONE": 0.8, "TEXT_INCONSISTENCY": 0.1, "TEMPORAL_FLICKER": 0.1}
        )

    assert "clip_id" in str(exc_info.value)


def test_confidence_out_of_range():
    """confidence > 1.0 should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            clip_id="test_003",
            artifact=False,
            confidence=1.5,
            artifact_category=ArtifactCategory.NONE,
            class_probabilities={"NONE": 0.8, "TEXT_INCONSISTENCY": 0.1, "TEMPORAL_FLICKER": 0.1}
        )

    assert "confidence" in str(exc_info.value).lower()


def test_invalid_category():
    """Invalid artifact_category value should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            clip_id="test_004",
            artifact=True,
            confidence=0.9,
            artifact_category="BLUR",  # Invalid category
            class_probabilities={"NONE": 0.1, "TEXT_INCONSISTENCY": 0.1, "TEMPORAL_FLICKER": 0.8},
            evidence=Evidence(
                timestamps=[1.0],
                bounding_boxes=[[0, 0, 100, 100]],
                frames=[5],
                notes="Test"
            )
        )

    assert "artifact_category" in str(exc_info.value).lower()


def test_artifact_true_category_none():
    """artifact=true with category=NONE should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            clip_id="test_005",
            artifact=True,
            confidence=0.7,
            artifact_category=ArtifactCategory.NONE,
            class_probabilities={"NONE": 0.3, "TEXT_INCONSISTENCY": 0.5, "TEMPORAL_FLICKER": 0.2}
        )

    error_msg = str(exc_info.value).lower()
    assert "evidence contract" in error_msg
    assert "artifact=true cannot have category=none" in error_msg


def test_artifact_false_category_not_none():
    """artifact=false with category!=NONE should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            clip_id="test_006",
            artifact=False,
            confidence=0.6,
            artifact_category=ArtifactCategory.TEXT_INCONSISTENCY,
            class_probabilities={"NONE": 0.7, "TEXT_INCONSISTENCY": 0.2, "TEMPORAL_FLICKER": 0.1}
        )

    error_msg = str(exc_info.value).lower()
    assert "evidence contract" in error_msg
    assert "artifact=false requires category=none" in error_msg


def test_artifact_false_with_evidence():
    """artifact=false with evidence!=None should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            clip_id="test_007",
            artifact=False,
            confidence=0.8,
            artifact_category=ArtifactCategory.NONE,
            class_probabilities={"NONE": 0.8, "TEXT_INCONSISTENCY": 0.1, "TEMPORAL_FLICKER": 0.1},
            evidence=Evidence(
                timestamps=[1.0],
                bounding_boxes=[[0, 0, 100, 100]],
                frames=[5],
                notes="This should not be here"
            )
        )

    error_msg = str(exc_info.value).lower()
    assert "evidence contract" in error_msg
    assert "artifact=false requires evidence=none" in error_msg


def test_artifact_true_no_evidence():
    """artifact=true with evidence=None should be valid (rare edge case)."""
    prediction = Prediction(
        clip_id="test_008",
        artifact=True,
        confidence=0.6,
        artifact_category=ArtifactCategory.TEMPORAL_FLICKER,
        class_probabilities={"NONE": 0.4, "TEXT_INCONSISTENCY": 0.1, "TEMPORAL_FLICKER": 0.5},
        evidence=None
    )

    # This is valid - artifact detected but no specific frames to point to
    assert prediction.artifact is True
    assert prediction.evidence is None
