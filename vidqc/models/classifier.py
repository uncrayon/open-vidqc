"""XGBoost classifier for vidqc artifact detection.

This module implements a three-class classifier (NONE / TEXT_INCONSISTENCY / TEMPORAL_FLICKER)
with critical decision logic:
- Binary threshold: artifact = P(NONE) < threshold
- Hard constraint: no text → never TEXT_INCONSISTENCY
- Threshold fallback: weak probabilities → NONE

All overrides must be logged for transparency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from vidqc.utils.acceleration import resolve_xgboost_device
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassifierResult:
    """Raw classifier output before constraint application.

    Attributes:
        probabilities: Dict mapping class names to probabilities
        predicted_class: Predicted artifact category (after all overrides)
        artifact_detected: Whether any artifact was detected
        confidence: Maximum probability across all classes
        override_applied: Whether any override was applied (for logging)
        override_reason: Reason for override (or None)
    """

    probabilities: dict[str, float]
    predicted_class: str
    artifact_detected: bool
    confidence: float
    override_applied: bool = False
    override_reason: Optional[str] = None


class VidQCClassifier:
    """XGBoost-based artifact classifier.

    Three-class classification: NONE, TEXT_INCONSISTENCY, TEMPORAL_FLICKER
    with decision logic:
    1. Binary threshold: artifact = P(NONE) < binary_threshold
    2. If artifact, category = argmax(P(TEXT), P(FLICKER))
    3. Hard constraint: if has_text=False and pred=TEXT, override to NONE
    4. Threshold fallback: if P(TEXT) <= 0.1 and P(FLICKER) <= 0.1, override to NONE
    """

    def __init__(self, config: dict):
        """Initialize classifier with config.

        Args:
            config: Configuration dict (must contain 'model' section)
        """
        self.config = config
        self.model: Optional[xgb.Booster] = None
        self.feature_names: Optional[list[str]] = None
        self.class_names = ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"]
        self._resolved_xgboost_device: str = "cpu"

    def _resolve_device_from_config(self) -> str:
        """Resolve and log the effective XGBoost device from config."""
        requested_device = self.config.get("model", {}).get("xgboost", {}).get(
            "device", "auto"
        )
        resolved_device, reason = resolve_xgboost_device(requested_device)
        logger.info(
            "xgboost device requested=%r resolved=%s reason=%s",
            requested_device,
            resolved_device,
            reason,
        )
        self._resolved_xgboost_device = resolved_device
        return resolved_device

    def _apply_model_device(self, device: str) -> None:
        """Apply resolved device to an already-loaded booster."""
        if self.model is None:
            return

        try:
            self.model.set_param({"device": device})
        except xgb.core.XGBoostError as exc:
            if device == "cuda":
                logger.warning(
                    "Failed to set XGBoost device=cuda (%s), falling back to CPU",
                    exc,
                )
                self.model.set_param({"device": "cpu"})
                self._resolved_xgboost_device = "cpu"
            else:
                logger.warning("Failed to set XGBoost device=%s: %s", device, exc)

    def train(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> None:
        """Train XGBoost classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,) with values 0, 1, 2 for NONE, TEXT, FLICKER
            feature_names: List of feature names (length must match X.shape[1])

        Raises:
            ValueError: If X and y shapes don't match or labels invalid
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y shape mismatch: X has {X.shape[0]} samples, "
                f"y has {y.shape[0]} samples"
            )

        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: X has {X.shape[1]} features, "
                f"but {len(feature_names)} feature names provided"
            )

        # Validate labels are 0, 1, 2
        unique_labels = np.unique(y)
        if not all(label in [0, 1, 2] for label in unique_labels):
            raise ValueError(
                f"Labels must be 0 (NONE), 1 (TEXT), or 2 (FLICKER). "
                f"Got: {unique_labels}"
            )

        logger.info(
            f"Training XGBoost classifier on {X.shape[0]} samples, "
            f"{X.shape[1]} features"
        )

        # Get XGBoost params from config
        params = self.config["model"]["xgboost"].copy()
        resolved_device = self._resolve_device_from_config()
        params["device"] = resolved_device

        # CUDA path in XGBoost expects hist tree method.
        if resolved_device == "cuda" and "tree_method" not in params:
            params["tree_method"] = "hist"

        # Ensure seed is set for determinism
        params["seed"] = 42

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

        # Train model. If CUDA training fails, retry on CPU once.
        try:
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get("n_estimators", 50),
            )
        except xgb.core.XGBoostError as exc:
            if resolved_device != "cuda":
                raise

            logger.warning(
                "XGBoost training failed on CUDA (%s), retrying on CPU once",
                exc,
            )
            retry_params = params.copy()
            retry_params["device"] = "cpu"
            self.model = xgb.train(
                retry_params,
                dtrain,
                num_boost_round=retry_params.get("n_estimators", 50),
            )
            self._resolved_xgboost_device = "cpu"
            resolved_device = "cpu"
        self._apply_model_device(resolved_device)

        self.feature_names = feature_names

        logger.info("Training complete")

    def predict(
        self, features: np.ndarray, has_text: bool
    ) -> ClassifierResult:
        """Predict with hard constraint and threshold fallback.

        Decision logic:
        1. Binary threshold: artifact = P(NONE) < binary_threshold
        2. If artifact, category = argmax(P(TEXT), P(FLICKER))
        3. Hard constraint: if has_text=False and pred=TEXT, override to NONE
        4. Threshold fallback: if P(TEXT) <= 0.1 and P(FLICKER) <= 0.1, override to NONE

        Args:
            features: Feature vector (must match training feature count)
            has_text: Whether text was detected (from has_text_regions feature)

        Returns:
            ClassifierResult with predictions and any override info

        Raises:
            ValueError: If model not trained or feature count mismatch
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.feature_names is None:
            raise ValueError("Feature names not set. Call train() first.")

        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: expected {len(self.feature_names)}, "
                f"got {len(features)}"
            )

        # Create DMatrix
        dtest = xgb.DMatrix([features], feature_names=self.feature_names)

        # Get probabilities
        probs = self.model.predict(dtest)[0]  # [P(NONE), P(TEXT), P(FLICKER)]

        # Build probability dict
        prob_dict = {
            "NONE": float(probs[0]),
            "TEXT_INCONSISTENCY": float(probs[1]),
            "TEMPORAL_FLICKER": float(probs[2]),
        }

        binary_threshold = self.config["model"]["binary_threshold"]

        # Binary decision
        artifact_detected = prob_dict["NONE"] < binary_threshold

        override_applied = False
        override_reason = None

        if not artifact_detected:
            predicted_class = "NONE"
        else:
            # Category decision: argmax of artifact classes
            if prob_dict["TEXT_INCONSISTENCY"] > prob_dict["TEMPORAL_FLICKER"]:
                predicted_class = "TEXT_INCONSISTENCY"
            else:
                predicted_class = "TEMPORAL_FLICKER"

        # Hard constraint: no text → no TEXT_INCONSISTENCY
        if not has_text and predicted_class == "TEXT_INCONSISTENCY":
            logger.debug(
                f"Hard constraint triggered: has_text={has_text}, "
                f"pred={predicted_class} → overriding to NONE"
            )
            predicted_class = "NONE"
            artifact_detected = False
            override_applied = True
            override_reason = "HARD_CONSTRAINT: No text regions detected, cannot be TEXT_INCONSISTENCY"

        # Threshold fallback: weak artifact probabilities → NONE
        if (
            prob_dict["TEXT_INCONSISTENCY"] <= 0.1
            and prob_dict["TEMPORAL_FLICKER"] <= 0.1
        ):
            if artifact_detected:
                logger.debug(
                    f"Threshold fallback triggered: "
                    f"P(TEXT)={prob_dict['TEXT_INCONSISTENCY']:.3f}, "
                    f"P(FLICKER)={prob_dict['TEMPORAL_FLICKER']:.3f} → "
                    f"overriding to NONE"
                )
                predicted_class = "NONE"
                artifact_detected = False
                override_applied = True
                override_reason = "THRESHOLD_FALLBACK: All artifact probabilities <= 0.1"

        return ClassifierResult(
            probabilities=prob_dict,
            predicted_class=predicted_class,
            artifact_detected=artifact_detected,
            confidence=max(prob_dict.values()),
            override_applied=override_applied,
            override_reason=override_reason,
        )

    def save(self, path: str) -> None:
        """Save model to JSON file.

        Args:
            path: Path to save model (will be created if doesn't exist)

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from JSON file.

        Args:
            path: Path to model file

        Raises:
            FileNotFoundError: If model file not found
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = xgb.Booster()
        self.model.load_model(path)

        # Extract feature names from model
        self.feature_names = self.model.feature_names
        resolved_device = self._resolve_device_from_config()
        self._apply_model_device(resolved_device)

        logger.info(f"Model loaded from {path}")

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importances from trained model.

        Returns:
            Dict mapping feature names to importance scores (gain)

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get importance scores (using 'gain' metric)
        importances = self.model.get_score(importance_type="gain")

        return importances
