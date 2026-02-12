"""Prediction interface for vidqc classifier.

Supports single clip prediction and batch prediction from CSV.
Includes secondary signal detection and evidence collection.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from vidqc.config import get_config
from vidqc.features.extract import extract_all_features
from vidqc.models.classifier import VidQCClassifier
from vidqc.schemas import ArtifactCategory, Evidence, Prediction
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


def predict_clip(
    video_path: str, model_path: str, config: Optional[dict] = None
) -> Prediction:
    """Predict artifact category for single clip.

    Args:
        video_path: Path to video file
        model_path: Path to trained model file
        config: Optional config dict (uses default if None)

    Returns:
        Prediction object with artifact detection results

    Raises:
        FileNotFoundError: If video or model file not found
        ValueError: If prediction fails
    """
    if config is None:
        config = get_config()

    logger.info(f"Predicting: {video_path}")

    # Extract features
    try:
        features, evidence_data = extract_all_features(video_path, config)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise ValueError(f"Feature extraction failed: {e}")

    # Load classifier
    classifier = VidQCClassifier(config)
    classifier.load(model_path)

    # Predict
    has_text = features[0] > 0  # has_text_regions feature (index 0)
    result = classifier.predict(features, has_text)

    # Build diagnostic notes
    diagnostic_notes = []

    # Log probabilities for diagnostics
    prob_str = ", ".join(
        [f"{k}={v:.3f}" for k, v in result.probabilities.items()]
    )
    diagnostic_notes.append(f"Probabilities: {prob_str}")

    # Log override if applied
    if result.override_applied and result.override_reason:
        diagnostic_notes.append(result.override_reason)

    # Secondary signal check (§6.9)
    # Detect if both text and temporal artifacts present
    text_prob = result.probabilities["TEXT_INCONSISTENCY"]
    flicker_prob = result.probabilities["TEMPORAL_FLICKER"]

    # Both probabilities significant (>0.2) indicates potential secondary signal
    if text_prob > 0.2 and flicker_prob > 0.2:
        diagnostic_notes.append(
            f"SECONDARY_SIGNAL: Both text ({text_prob:.3f}) and "
            f"temporal ({flicker_prob:.3f}) artifacts detected"
        )

    # Build Evidence object (enforce contract: artifact=False requires evidence=None)
    if result.artifact_detected:
        # Select evidence based on predicted category
        text_evidence = evidence_data.get("text_evidence")
        temporal_evidence = evidence_data.get("temporal_evidence")

        extractor_evidence: Evidence | None = None
        if result.predicted_class == "TEXT_INCONSISTENCY":
            extractor_evidence = text_evidence
        elif result.predicted_class == "TEMPORAL_FLICKER":
            extractor_evidence = temporal_evidence

        if extractor_evidence is not None:
            # Merge extractor evidence with diagnostic notes
            combined_notes = extractor_evidence.notes
            if diagnostic_notes:
                combined_notes += "\n" + "\n".join(diagnostic_notes)
            evidence = Evidence(
                timestamps=extractor_evidence.timestamps,
                bounding_boxes=extractor_evidence.bounding_boxes,
                frames=extractor_evidence.frames,
                notes=combined_notes,
            )
        else:
            # Artifact detected but no evidence frames collected
            logger.warning(
                f"Artifact detected ({result.predicted_class}) but no evidence "
                f"frames collected from extractor"
            )
            evidence = Evidence(
                timestamps=[],
                bounding_boxes=[],
                frames=[],
                notes="\n".join(diagnostic_notes),
            )
    else:
        # When no artifact, evidence MUST be None
        evidence = None

    # Build Prediction object
    prediction = Prediction(
        clip_id=str(Path(video_path).name),
        artifact=result.artifact_detected,
        confidence=result.confidence,
        artifact_category=ArtifactCategory[result.predicted_class],
        class_probabilities=result.probabilities,
        evidence=evidence,
    )

    logger.info(
        f"Prediction: artifact={prediction.artifact}, "
        f"category={prediction.artifact_category}, "
        f"confidence={prediction.confidence:.3f}"
    )

    return prediction


def predict_batch(
    input_csv: str,
    output_csv: str,
    model_path: str,
    config: Optional[dict] = None,
) -> None:
    """Batch prediction from CSV.

    Args:
        input_csv: Path to input CSV with 'path' column
        output_csv: Path to output CSV
        model_path: Path to trained model
        config: Optional config dict (uses default if None)

    Raises:
        FileNotFoundError: If input CSV or model not found
    """
    if config is None:
        config = get_config()

    # Load input CSV
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "path" not in df.columns:
        raise ValueError("Input CSV must contain 'path' column")

    logger.info(f"Batch prediction on {len(df)} clips")

    results = []
    for idx, row in df.iterrows():
        video_path = row["path"]
        logger.info(f"[{idx+1}/{len(df)}] Predicting: {video_path}")

        try:
            pred = predict_clip(video_path, model_path, config)
            results.append(
                {
                    "clip_path": pred.clip_id,
                    "artifact": pred.artifact,
                    "category": pred.artifact_category.value,
                    "confidence": pred.confidence,
                    "prob_none": pred.class_probabilities["NONE"],
                    "prob_text": pred.class_probabilities[
                        "TEXT_INCONSISTENCY"
                    ],
                    "prob_flicker": pred.class_probabilities[
                        "TEMPORAL_FLICKER"
                    ],
                }
            )
        except Exception as e:
            logger.error(f"Prediction failed for {video_path}: {e}")
            results.append(
                {
                    "clip_path": video_path,
                    "artifact": False,
                    "category": "NONE",
                    "confidence": 0.0,
                    "prob_none": 1.0,
                    "prob_text": 0.0,
                    "prob_flicker": 0.0,
                    "error": str(e),
                }
            )

    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)

    logger.info(f"Batch prediction complete. Results saved to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict artifact category for video clips"
    )
    parser.add_argument(
        "--video",
        help="Path to single video file (for single prediction)",
    )
    parser.add_argument(
        "--batch",
        help="Path to input CSV for batch prediction",
    )
    parser.add_argument(
        "--output",
        help="Path to output CSV (required for batch prediction)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Load config
    config = get_config()

    # Validate arguments
    if args.video and args.batch:
        raise ValueError("Cannot specify both --video and --batch")

    if not args.video and not args.batch:
        raise ValueError("Must specify either --video or --batch")

    if args.batch and not args.output:
        raise ValueError("--output required for batch prediction")

    # Execute
    try:
        if args.video:
            # Single prediction
            pred = predict_clip(args.video, args.model, config)
            # Print as JSON
            print(json.dumps(pred.model_dump(), indent=2, default=str))
        else:
            # Batch prediction
            predict_batch(args.batch, args.output, args.model, config)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
