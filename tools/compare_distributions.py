"""Feature distribution comparison: synthetic vs. real clips.

Per §4.4, compares mean/std between synthetic and real clips within the same
label category. Flags features where distributions diverge by >2 standard
deviations and outputs a summary report.

Usage:
    uv run python tools/compare_distributions.py \
        --labels samples/labels.csv \
        --output reports/distribution_comparison.json

The labels.csv must have a 'source' column with values like 'synthetic',
'sora', 'veo3', 'public', etc. Any source != 'synthetic' is treated as real.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path before vidqc imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vidqc.config import get_config  # noqa: E402
from vidqc.features.extract import extract_all_features  # noqa: E402
from vidqc.features.feature_manifest import COMBINED_FEATURE_NAMES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_feature_matrix(
    df: pd.DataFrame, config: dict
) -> tuple[np.ndarray, list[str]]:
    """Extract features for all clips in DataFrame.

    Args:
        df: DataFrame with 'path' column
        config: Config dict

    Returns:
        feature_matrix: (n_samples, 43) array
        valid_paths: List of paths that succeeded
    """
    features_list = []
    valid_paths = []

    for _, row in df.iterrows():
        video_path = row["path"]
        try:
            features, _ = extract_all_features(video_path, config)
            features_list.append(features)
            valid_paths.append(video_path)
        except Exception as e:
            logger.warning(f"Skipping {video_path}: {e}")

    if not features_list:
        return np.array([]).reshape(0, len(COMBINED_FEATURE_NAMES)), []

    return np.array(features_list), valid_paths


def compare_distributions(
    synthetic_features: np.ndarray,
    real_features: np.ndarray,
    feature_names: list[str],
    divergence_threshold: float = 2.0,
) -> dict:
    """Compare feature distributions between synthetic and real clips.

    For each feature, computes mean and std for both groups and checks
    if the means diverge by more than threshold * pooled_std.

    Args:
        synthetic_features: (n_synthetic, n_features) array
        real_features: (n_real, n_features) array
        feature_names: List of feature names
        divergence_threshold: Number of std deviations to flag (default 2.0)

    Returns:
        Comparison report dict
    """
    n_features = len(feature_names)
    report = {
        "n_synthetic": len(synthetic_features),
        "n_real": len(real_features),
        "divergence_threshold": divergence_threshold,
        "features": {},
        "divergent_features": [],
        "summary": {},
    }

    for i, name in enumerate(feature_names):
        synth_vals = synthetic_features[:, i]
        real_vals = real_features[:, i]

        synth_mean = float(np.mean(synth_vals))
        synth_std = float(np.std(synth_vals))
        real_mean = float(np.mean(real_vals))
        real_std = float(np.std(real_vals))

        # Pooled standard deviation
        pooled_std = float(
            np.sqrt(
                (synth_std**2 + real_std**2) / 2
            )
        )

        # Check divergence
        if pooled_std > 0:
            mean_diff = abs(synth_mean - real_mean)
            divergence_ratio = mean_diff / pooled_std
        else:
            # Both groups have zero variance — no divergence
            divergence_ratio = 0.0

        divergent = divergence_ratio > divergence_threshold

        feature_report = {
            "synthetic_mean": round(synth_mean, 6),
            "synthetic_std": round(synth_std, 6),
            "real_mean": round(real_mean, 6),
            "real_std": round(real_std, 6),
            "pooled_std": round(pooled_std, 6),
            "divergence_ratio": round(divergence_ratio, 3),
            "divergent": divergent,
        }

        report["features"][name] = feature_report

        if divergent:
            report["divergent_features"].append(name)

    # Summary
    n_divergent = len(report["divergent_features"])
    report["summary"] = {
        "total_features": n_features,
        "divergent_count": n_divergent,
        "divergent_pct": round(100 * n_divergent / n_features, 1) if n_features > 0 else 0,
        "recommendation": _get_recommendation(n_divergent, n_features),
    }

    return report


def _get_recommendation(n_divergent: int, n_total: int) -> str:
    """Generate recommendation based on divergence count."""
    pct = n_divergent / n_total if n_total > 0 else 0

    if pct == 0:
        return "No divergent features. Synthetic and real distributions are similar."
    elif pct < 0.15:
        return (
            f"{n_divergent}/{n_total} features diverge. "
            "Minor domain gap — model should generalize reasonably."
        )
    elif pct < 0.33:
        return (
            f"{n_divergent}/{n_total} features diverge. "
            "Moderate domain gap — review divergent features for extraction bugs. "
            "Mixed training (synthetic + real) recommended."
        )
    else:
        return (
            f"{n_divergent}/{n_total} features diverge. "
            "Significant domain gap — synthetic data may not represent real artifacts well. "
            "Prioritize collecting more real clips. See §6.5 fallback strategy."
        )


def compare_by_category(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    divergence_threshold: float = 2.0,
) -> dict:
    """Run distribution comparison within each label category.

    Args:
        df: DataFrame with 'source' and 'artifact_category' columns
        feature_matrix: (n_samples, n_features) array matching df rows
        feature_names: List of feature names
        divergence_threshold: Divergence threshold in std deviations

    Returns:
        Per-category comparison report
    """
    df = df.copy()
    df["is_real"] = df["source"] != "synthetic"

    categories = df["artifact_category"].unique()
    per_category = {}

    for category in sorted(categories):
        cat_mask = df["artifact_category"] == category
        synth_mask = cat_mask & ~df["is_real"]
        real_mask = cat_mask & df["is_real"]

        n_synth = int(synth_mask.sum())
        n_real = int(real_mask.sum())

        if n_synth == 0 or n_real == 0:
            logger.warning(
                f"Category {category}: skipping comparison "
                f"(synthetic={n_synth}, real={n_real})"
            )
            per_category[category] = {
                "skipped": True,
                "reason": f"Need both synthetic and real clips (synthetic={n_synth}, real={n_real})",
            }
            continue

        synth_features = feature_matrix[synth_mask.values]
        real_features = feature_matrix[real_mask.values]

        per_category[category] = compare_distributions(
            synth_features, real_features, feature_names, divergence_threshold
        )

    return per_category


def print_report(report: dict) -> None:
    """Print a human-readable summary of the comparison report."""
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION COMPARISON: SYNTHETIC vs. REAL")
    print("=" * 70)

    if "overall" in report:
        overall = report["overall"]
        print(f"\n--- Overall ({overall['n_synthetic']} synthetic, {overall['n_real']} real) ---")
        print(f"Divergent features: {overall['summary']['divergent_count']}/{overall['summary']['total_features']}")
        print(f"Recommendation: {overall['summary']['recommendation']}")

        if overall["divergent_features"]:
            print("\nDivergent features (mean differs by >{:.0f} pooled std):".format(
                overall["divergence_threshold"]
            ))
            for name in overall["divergent_features"]:
                feat = overall["features"][name]
                print(
                    f"  {name:45s}  synth={feat['synthetic_mean']:8.4f}  "
                    f"real={feat['real_mean']:8.4f}  ratio={feat['divergence_ratio']:.1f}x"
                )

    if "per_category" in report:
        for category, cat_report in report["per_category"].items():
            print(f"\n--- Category: {category} ---")
            if cat_report.get("skipped"):
                print(f"  SKIPPED: {cat_report['reason']}")
                continue

            print(f"  Samples: {cat_report['n_synthetic']} synthetic, {cat_report['n_real']} real")
            print(f"  Divergent: {cat_report['summary']['divergent_count']}/{cat_report['summary']['total_features']}")

            if cat_report["divergent_features"]:
                for name in cat_report["divergent_features"]:
                    feat = cat_report["features"][name]
                    print(
                        f"    {name:43s}  synth={feat['synthetic_mean']:8.4f}  "
                        f"real={feat['real_mean']:8.4f}  ratio={feat['divergence_ratio']:.1f}x"
                    )

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare feature distributions between synthetic and real clips"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to labels CSV with 'source' column"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output JSON report"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Divergence threshold in std deviations (default: 2.0)",
    )
    parser.add_argument(
        "--cache",
        default="features_cache.pkl",
        help="Path to feature cache file (default: features_cache.pkl)",
    )

    args = parser.parse_args()

    # Load labels
    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error(f"Labels file not found: {args.labels}")
        sys.exit(1)

    df = pd.read_csv(args.labels)

    # Validate columns
    required_cols = {"path", "artifact_category", "source"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Labels CSV missing columns: {missing_cols}")
        sys.exit(1)

    # Split by source
    df["is_real"] = df["source"] != "synthetic"
    n_synthetic = int((~df["is_real"]).sum())
    n_real = int(df["is_real"].sum())

    logger.info(f"Dataset: {len(df)} total ({n_synthetic} synthetic, {n_real} real)")

    if n_real == 0:
        logger.error(
            "No real clips found (all source='synthetic'). "
            "Add real clips with source='sora', 'veo3', 'public', etc."
        )
        sys.exit(1)

    if n_synthetic == 0:
        logger.warning("No synthetic clips found. Comparison will be limited.")

    # Load config and extract features
    config = get_config()

    logger.info("Extracting features for all clips...")

    # Use the training pipeline's caching for efficiency
    from vidqc.train import extract_features_with_cache

    feature_matrix, labels_arr, _ = extract_features_with_cache(
        df, config, args.cache
    )

    if len(feature_matrix) == 0:
        logger.error("No features extracted. Check video files.")
        sys.exit(1)

    # Align df with extracted features (some may have been skipped)
    # Re-filter df to match extracted features
    valid_df = df.iloc[: len(feature_matrix)].copy()

    # Overall comparison
    synth_mask = valid_df["source"] == "synthetic"
    real_mask = valid_df["source"] != "synthetic"

    overall = compare_distributions(
        feature_matrix[synth_mask.values],
        feature_matrix[real_mask.values],
        COMBINED_FEATURE_NAMES,
        args.threshold,
    )

    # Per-category comparison
    per_category = compare_by_category(
        valid_df, feature_matrix, COMBINED_FEATURE_NAMES, args.threshold
    )

    # Build full report
    full_report = {
        "overall": overall,
        "per_category": per_category,
        "dataset": {
            "total": len(valid_df),
            "synthetic": int(synth_mask.sum()),
            "real": int(real_mask.sum()),
            "categories": dict(valid_df["artifact_category"].value_counts()),
            "sources": dict(valid_df["source"].value_counts()),
        },
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    logger.info(f"Report saved to {args.output}")

    # Print human-readable summary
    print_report(full_report)


if __name__ == "__main__":
    main()
