"""Pydantic schemas for vidqc output validation.

This module defines the output contract for predictions, including the critical
evidence contract enforced by model validators.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ArtifactCategory(str, Enum):
    """Artifact type classification."""
    TEXT_INCONSISTENCY = "TEXT_INCONSISTENCY"
    TEMPORAL_FLICKER = "TEMPORAL_FLICKER"
    NONE = "NONE"


class Evidence(BaseModel):
    """Evidence supporting an artifact detection.

    Attributes:
        timestamps: Frame timestamps where artifact was detected (seconds)
        bounding_boxes: Bounding boxes for detected regions [[x1, y1, x2, y2], ...]
        frames: Frame indices where artifact was detected
        notes: Human-readable explanation of the detection
    """
    timestamps: list[float]
    bounding_boxes: list[list[int]]
    frames: list[int]
    notes: str


class Prediction(BaseModel):
    """Prediction output for a single video clip.

    This model enforces the evidence contract:
    1. artifact=false → evidence MUST be None
    2. artifact=true AND category=NONE → ValidationError
    3. artifact=false AND category != NONE → ValidationError

    Attributes:
        clip_id: Unique identifier for the video clip
        artifact: Whether an artifact was detected
        confidence: Confidence score [0.0, 1.0]
        artifact_category: Predicted artifact type
        class_probabilities: Probability distribution over all categories
        evidence: Optional evidence object (only when artifact=true)
    """
    clip_id: str
    artifact: bool
    confidence: float = Field(ge=0.0, le=1.0)
    artifact_category: ArtifactCategory
    class_probabilities: dict[str, float]
    evidence: Optional[Evidence] = None

    @model_validator(mode="after")
    def check_evidence_contract(self) -> "Prediction":
        """Validate evidence contract.

        The evidence contract ensures logical consistency between the artifact
        flag, category, and evidence fields.

        Raises:
            ValueError: If evidence contract is violated

        Returns:
            The validated Prediction instance
        """
        # Rule 1: artifact=false → evidence MUST be None
        if not self.artifact and self.evidence is not None:
            raise ValueError(
                "Evidence contract violation: artifact=false requires evidence=None. "
                f"Got artifact={self.artifact}, evidence is not None."
            )

        # Rule 2: artifact=true AND category=NONE → error
        if self.artifact and self.artifact_category == ArtifactCategory.NONE:
            raise ValueError(
                "Evidence contract violation: artifact=true cannot have category=NONE. "
                f"Got artifact={self.artifact}, category={self.artifact_category}."
            )

        # Rule 3: artifact=false AND category != NONE → error
        if not self.artifact and self.artifact_category != ArtifactCategory.NONE:
            raise ValueError(
                "Evidence contract violation: artifact=false requires category=NONE. "
                f"Got artifact={self.artifact}, category={self.artifact_category}."
            )

        return self
