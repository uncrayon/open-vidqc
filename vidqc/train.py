"""Training pipeline for vidqc classifier.

End-to-end training with:
- Feature extraction with caching
- 5-fold stratified cross-validation
- Final model training on full dataset
- Training manifest generation
- Feature importance analysis
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from vidqc.config import get_config
from vidqc.features.extract import extract_all_features
from vidqc.features.feature_manifest import COMBINED_FEATURE_NAMES
from vidqc.models.classifier import VidQCClassifier
from vidqc.schemas import ArtifactCategory
from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


def _label_to_int(label: str) -> int:
    """Convert label string to integer.

    Args:
        label: Category label (NONE, TEXT_INCONSISTENCY, TEMPORAL_FLICKER)

    Returns:
        Integer label: 0=NONE, 1=TEXT_INCONSISTENCY, 2=TEMPORAL_FLICKER
    """
    mapping = {
        "NONE": 0,
        "TEXT_INCONSISTENCY": 1,
        "TEMPORAL_FLICKER": 2,
    }
    return mapping[label]


def extract_features_with_cache(
    df: pd.DataFrame, config: dict, cache_path: str = "features_cache.pkl"
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract features with caching.

    Cache format: {clip_path: {'features': array, 'evidence': dict, 'mtime': float}}

    Args:
        df: DataFrame with columns 'path' and 'artifact_category'
        config: Configuration dict
        cache_path: Path to cache file

    Returns:
        features: Feature matrix (n_samples, 43)
        labels: Label array (n_samples,)
        evidence_list: List of evidence dicts (one per sample)
    """
    cache_file = Path(cache_path)

    # Load cache if exists
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"Loaded feature cache from {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Starting fresh.")
            cache = {}

    features_list = []
    labels_list = []
    evidence_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features", unit="clip"):
        video_path = row["path"]
        label = row["artifact_category"]

        # Check cache validity
        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            continue

        video_mtime = video_file.stat().st_mtime
        cache_key = str(video_path)

        # Check cache hit
        if (
            cache_key in cache
            and cache[cache_key]["mtime"] == video_mtime
        ):
            logger.debug(f"Cache hit: {video_path}")
            features = cache[cache_key]["features"]
            evidence = cache[cache_key]["evidence"]
        else:
            logger.info(f"Extracting features: {video_path}")
            try:
                features, evidence = extract_all_features(video_path, config)
                cache[cache_key] = {
                    "features": features,
                    "evidence": evidence,
                    "mtime": video_mtime,
                }
            except Exception as e:
                logger.error(f"Feature extraction failed for {video_path}: {e}")
                continue

        features_list.append(features)
        labels_list.append(_label_to_int(label))
        evidence_list.append(evidence)

    # Save cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"Saved feature cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return (
        np.array(features_list),
        np.array(labels_list),
        evidence_list,
    )


def train_model(
    labels_csv: str,
    output_dir: str,
    config: Optional[dict] = None,
    cache_path: str = "features_cache.pkl",
) -> tuple[VidQCClassifier, dict]:
    """End-to-end training pipeline.

    Steps:
    1. Load labels
    2. Extract features (with caching)
    3. 5-fold stratified cross-validation
    4. Train final model on full dataset
    5. Save model, manifest, importances
    6. Log metrics

    Args:
        labels_csv: Path to labels CSV file
        output_dir: Directory to save model and artifacts
        config: Optional config dict (uses default if None)
        cache_path: Path to feature cache file

    Returns:
        final_classifier: Trained classifier
        manifest: Training manifest dict

    Raises:
        FileNotFoundError: If labels CSV not found
        ValueError: If training fails
    """
    if config is None:
        config = get_config()

    # Set seeds for determinism
    np.random.seed(42)

    logger.info(f"Starting training pipeline with labels from {labels_csv}")

    # Load labels
    labels_file = Path(labels_csv)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    logger.info(f"Loaded {len(df)} samples from {labels_csv}")

    # Validate required columns
    if "path" not in df.columns or "artifact_category" not in df.columns:
        raise ValueError(
            "Labels CSV must contain 'path' and 'artifact_category' columns"
        )

    # Extract features
    features, labels, evidence_list = extract_features_with_cache(
        df, config, cache_path
    )

    if len(features) == 0:
        raise ValueError("No features extracted. Check video files and labels.")

    logger.info(
        f"Extracted features for {len(features)} samples "
        f"({features.shape[1]} features each)"
    )

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for label_int, count in zip(unique, counts):
        label_name = ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"][label_int]
        logger.info(f"  {label_name}: {count}")

    # Stratified cross-validation with dynamic fold count
    # n_splits cannot exceed the minimum class count
    min_class_count = int(min(counts))
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        logger.warning(
            f"Dataset too small for cross-validation (min class has {min_class_count} samples). "
            "Skipping CV, training on full dataset."
        )
        cv_scores = []
        cv_mean = 0.0
        cv_std = 0.0
    else:
        if n_splits < 5:
            logger.warning(
                f"Using {n_splits}-fold CV instead of 5-fold "
                f"(smallest class has only {min_class_count} samples)"
            )

        cv_scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        logger.info(f"Starting {n_splits}-fold cross-validation")

        for fold, (train_idx, val_idx) in enumerate(
            tqdm(skf.split(features, labels), total=n_splits, desc="Cross-validation", unit="fold")
        ):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Train classifier
            classifier = VidQCClassifier(config)
            classifier.train(X_train, y_train, COMBINED_FEATURE_NAMES)

            # Evaluate on val fold
            val_preds = []
            for i in val_idx:
                has_text = features[i][0] > 0  # has_text_regions feature
                result = classifier.predict(features[i], has_text)
                val_preds.append(_label_to_int(result.predicted_class))

            fold_f1 = f1_score(
                labels[val_idx], val_preds, average="macro", zero_division=0
            )
            cv_scores.append(fold_f1)
            logger.info(f"Fold {fold+1} F1: {fold_f1:.3f}")

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        logger.info(f"CV Mean F1: {cv_mean:.3f} ± {cv_std:.3f}")

    # Train final model on full dataset
    logger.info("Training final model on full dataset")
    final_classifier = VidQCClassifier(config)
    final_classifier.train(features, labels, COMBINED_FEATURE_NAMES)

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / "model.json"
    final_classifier.save(str(model_path))

    # Save training manifest
    manifest = {
        "model_path": str(model_path),
        "training_date": datetime.now().isoformat(),
        "num_samples": len(features),
        "num_features": features.shape[1],
        "feature_names": COMBINED_FEATURE_NAMES,
        "class_names": ["NONE", "TEXT_INCONSISTENCY", "TEMPORAL_FLICKER"],
        "cv_mean_f1": float(cv_mean),
        "cv_std_f1": float(cv_std),
        "cv_fold_scores": [float(s) for s in cv_scores],
        "config": config,
    }

    manifest_path = output_path / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Training manifest saved to {manifest_path}")

    # Save feature importances
    importances = final_classifier.get_feature_importances()
    importance_path = output_path / "feature_importances.json"
    with open(importance_path, "w") as f:
        json.dump(importances, f, indent=2)

    logger.info(f"Feature importances saved to {importance_path}")

    logger.info(f"Training complete. Model saved to {model_path}")

    return final_classifier, manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train vidqc classifier")
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for model and artifacts",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--cache",
        default="features_cache.pkl",
        help="Path to feature cache file (default: features_cache.pkl)",
    )

    args = parser.parse_args()

    # Load config
    config = get_config()

    # Train
    try:
        classifier, manifest = train_model(
            args.labels,
            args.output,
            config,
            args.cache,
        )
        print(f"\nTraining complete!")
        print(f"Model: {manifest['model_path']}")
        print(f"CV F1: {manifest['cv_mean_f1']:.3f} ± {manifest['cv_std_f1']:.3f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
