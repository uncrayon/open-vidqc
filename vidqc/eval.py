"""Evaluation suite for vidqc classifier.

Evaluates trained model against labeled dataset with:
- Confusion matrix
- Per-class precision, recall, F1
- Binary (artifact vs. no-artifact) precision, recall
- Success criteria check per §0.5:
  1. Binary recall >= 0.85
  2. Binary precision >= 0.70
  3. Per-category F1 >= 0.70 for both TEXT_INCONSISTENCY and TEMPORAL_FLICKER
  4. CV std < 0.15 for all metrics (checked separately during training)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from vidqc.config import get_config
from vidqc.predict import predict_clip
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)

# §0.5 success criteria thresholds
BINARY_RECALL_THRESHOLD = 0.85
BINARY_PRECISION_THRESHOLD = 0.70
CATEGORY_F1_THRESHOLD = 0.70
CV_STD_THRESHOLD = 0.15


def _to_binary(labels: list[str]) -> list[int]:
    """Convert multi-class labels to binary (artifact=1, NONE=0)."""
    return [0 if label == "NONE" else 1 for label in labels]


def check_success_criteria(
    binary_recall: float,
    binary_precision: float,
    text_f1: float,
    flicker_f1: float,
    cv_std: Optional[float] = None,
) -> dict:
    """Check all §0.5 success criteria.

    Args:
        binary_recall: Recall for binary artifact detection
        binary_precision: Precision for binary artifact detection
        text_f1: F1 score for TEXT_INCONSISTENCY class
        flicker_f1: F1 score for TEMPORAL_FLICKER class
        cv_std: Cross-validation std (from training manifest, if available)

    Returns:
        Dict with individual criteria results and overall pass/fail
    """
    criteria = {
        "1_binary_recall": {
            "value": round(binary_recall, 4),
            "threshold": BINARY_RECALL_THRESHOLD,
            "passed": binary_recall >= BINARY_RECALL_THRESHOLD,
            "description": "Binary recall >= 0.85",
        },
        "2_binary_precision": {
            "value": round(binary_precision, 4),
            "threshold": BINARY_PRECISION_THRESHOLD,
            "passed": binary_precision >= BINARY_PRECISION_THRESHOLD,
            "description": "Binary precision >= 0.70",
        },
        "3_text_f1": {
            "value": round(text_f1, 4),
            "threshold": CATEGORY_F1_THRESHOLD,
            "passed": text_f1 >= CATEGORY_F1_THRESHOLD,
            "description": "TEXT_INCONSISTENCY F1 >= 0.70",
        },
        "4_flicker_f1": {
            "value": round(flicker_f1, 4),
            "threshold": CATEGORY_F1_THRESHOLD,
            "passed": flicker_f1 >= CATEGORY_F1_THRESHOLD,
            "description": "TEMPORAL_FLICKER F1 >= 0.70",
        },
    }

    if cv_std is not None:
        criteria["5_cv_std"] = {
            "value": round(cv_std, 4),
            "threshold": CV_STD_THRESHOLD,
            "passed": cv_std < CV_STD_THRESHOLD,
            "description": "CV std < 0.15 for all metrics",
        }

    all_passed = all(c["passed"] for c in criteria.values())
    criteria["all_passed"] = all_passed

    return criteria


def evaluate_model(
    labels_csv: str,
    model_path: str,
    output_path: str,
    config: Optional[dict] = None,
    training_manifest_path: Optional[str] = None,
) -> dict:
    """Evaluate model and check §0.5 success criteria.

    Args:
        labels_csv: Path to labels CSV file
        model_path: Path to trained model
        output_path: Path to save evaluation results JSON
        config: Optional config dict (uses default if None)
        training_manifest_path: Optional path to training manifest JSON
            (used to check CV std criterion)

    Returns:
        Evaluation results dict

    Raises:
        FileNotFoundError: If labels CSV or model not found
    """
    if config is None:
        config = get_config()

    # Load labels
    if not Path(labels_csv).exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    if "path" not in df.columns or "artifact_category" not in df.columns:
        raise ValueError(
            "Labels CSV must contain 'path' and 'artifact_category' columns"
        )

    logger.info(f"Evaluating model on {len(df)} samples")

    # Predict on all clips
    y_true = []
    y_pred = []
    failed_clips = []

    for idx, row in df.iterrows():
        video_path = row["path"]
        true_label = row["artifact_category"]

        try:
            pred = predict_clip(video_path, model_path, config)
            y_true.append(true_label)
            y_pred.append(pred.artifact_category.value)
        except Exception as e:
            logger.error(f"Prediction failed for {video_path}: {e}")
            failed_clips.append(
                {"clip": video_path, "error": str(e)}
            )

    if len(y_true) == 0:
        raise ValueError("No successful predictions. Cannot evaluate.")

    logger.info(
        f"Completed {len(y_true)} predictions ({len(failed_clips)} failed)"
    )

    # --- Multi-class metrics ---
    category_labels = ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"]
    cm = confusion_matrix(y_true, y_pred, labels=category_labels)

    logger.info("Confusion matrix:")
    logger.info(f"{'':20} " + " ".join([f"{lbl:20}" for lbl in category_labels]))
    for i, label in enumerate(category_labels):
        logger.info(
            f"{label:20} " + " ".join([f"{cm[i][j]:20}" for j in range(len(category_labels))])
        )

    report = classification_report(
        y_true, y_pred, labels=category_labels, output_dict=True, zero_division=0
    )

    text_f1 = report.get("TEXT_INCONSISTENCY", {}).get("f1-score", 0.0)
    flicker_f1 = report.get("TEMPORAL_FLICKER", {}).get("f1-score", 0.0)
    none_f1 = report.get("NONE", {}).get("f1-score", 0.0)

    # --- Binary metrics (artifact vs. no-artifact) ---
    y_true_binary = _to_binary(y_true)
    y_pred_binary = _to_binary(y_pred)

    binary_recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    binary_precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    binary_f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    # --- Load CV std from training manifest if available ---
    cv_std = None
    if training_manifest_path and Path(training_manifest_path).exists():
        with open(training_manifest_path) as f:
            manifest = json.load(f)
        cv_std = manifest.get("cv_std_f1")
        if cv_std is not None:
            logger.info(f"CV std from training manifest: {cv_std:.4f}")

    # --- Check §0.5 success criteria ---
    criteria = check_success_criteria(
        binary_recall, binary_precision, text_f1, flicker_f1, cv_std
    )

    # Build results dict
    results = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_f1": {
            "NONE": none_f1,
            "TEXT_INCONSISTENCY": text_f1,
            "TEMPORAL_FLICKER": flicker_f1,
        },
        "binary_metrics": {
            "recall": round(binary_recall, 4),
            "precision": round(binary_precision, 4),
            "f1": round(binary_f1, 4),
        },
        "success_criteria": criteria,
        "success_criteria_met": criteria["all_passed"],
        "num_samples": len(y_true),
        "num_failed": len(failed_clips),
        "failed_clips": failed_clips,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")

    # Print summary
    logger.info("=== Evaluation Results ===")
    logger.info(f"NONE F1:                {none_f1:.3f}")
    logger.info(f"TEXT_INCONSISTENCY F1:  {text_f1:.3f}")
    logger.info(f"TEMPORAL_FLICKER F1:    {flicker_f1:.3f}")
    logger.info(f"Macro-averaged F1:      {report['macro avg']['f1-score']:.3f}")
    logger.info(f"Binary Recall:          {binary_recall:.3f}")
    logger.info(f"Binary Precision:       {binary_precision:.3f}")
    logger.info(f"Binary F1:              {binary_f1:.3f}")

    logger.info("=== §0.5 Success Criteria ===")
    for key, criterion in criteria.items():
        if key == "all_passed":
            continue
        status = "PASS" if criterion["passed"] else "FAIL"
        logger.info(
            f"  [{status}] {criterion['description']}: "
            f"{criterion['value']:.3f} (threshold: {criterion['threshold']})"
        )

    if criteria["all_passed"]:
        logger.info("All success criteria PASSED.")
    else:
        failed_criteria = [
            c["description"]
            for k, c in criteria.items()
            if k != "all_passed" and not c["passed"]
        ]
        logger.warning(
            f"Success criteria NOT met. Failed: {', '.join(failed_criteria)}. "
            "See §6.5 for fallback strategy."
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate vidqc classifier")
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels CSV file",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file for evaluation results",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to training manifest JSON (for CV std check)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Load config
    config = get_config()

    # Evaluate
    try:
        results = evaluate_model(
            args.labels,
            args.model,
            args.output,
            config,
            training_manifest_path=args.manifest,
        )
        print("\n=== Evaluation Complete ===")
        print(f"Results saved to: {args.output}")
        print(f"Success criteria met: {results['success_criteria_met']}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
