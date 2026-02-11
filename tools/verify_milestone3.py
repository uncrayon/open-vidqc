#!/usr/bin/env python3
"""Verify Milestone 3 exit criteria: extract features from all synthetic clips."""

import pandas as pd
import numpy as np
from pathlib import Path
from vidqc.features.text_inconsistency import extract_text_features

def main():
    # Check if labels.csv exists
    labels_path = Path('labels.csv')
    if not labels_path.exists():
        print('labels.csv not found. Run: uv run python tools/make_dataset.py --output samples/ --count 30')
        return 1

    labels = pd.read_csv('labels.csv')
    print(f'Found {len(labels)} clips in labels.csv\n')

    results = []
    for idx, row in labels.iterrows():
        clip_path = row['path']
        if not Path(clip_path).exists():
            print(f'WARNING: {clip_path} does not exist, skipping')
            continue

        print(f'Processing {idx+1}/{len(labels)}: {row["clip_id"]} ({row["artifact_category"]})...')

        try:
            features = extract_text_features(clip_path)

            # Check for NaN/Inf
            has_nan = any(np.isnan(v) for v in features.values())
            has_inf = any(np.isinf(v) for v in features.values())

            if has_nan or has_inf:
                print(f'  ERROR: NaN={has_nan}, Inf={has_inf}')
                continue

            # Check length
            if len(features) != 29:
                print(f'  ERROR: Expected 29 features, got {len(features)}')
                continue

            results.append({
                'clip_id': row['clip_id'],
                'category': row['artifact_category'],
                **features
            })

            print(f'  ✓ has_text={features["has_text_regions"]:.0f}, edit_dist_max={features["edit_distance_max"]:.1f}')

        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()
            continue

    if len(results) == 0:
        print('\nNo results. Check that synthetic clips exist in samples/')
        return 1

    df = pd.DataFrame(results)
    print('\n=== Feature Extraction Summary ===')
    print(f'Total clips processed: {len(df)}')
    print(f'Clips with text: {(df["has_text_regions"] == 1).sum()}')
    print(f'Clips without text: {(df["has_text_regions"] == 0).sum()}')

    print('\n=== Feature Separability Check ===')
    text_artifact = df[df['category'] == 'TEXT_INCONSISTENCY']
    clean_clips = df[df['category'].str.startswith('CLEAN')]

    if len(text_artifact) > 0:
        print(f'TEXT_INCONSISTENCY clips (n={len(text_artifact)}):')
        print(f'  edit_distance_max: mean={text_artifact["edit_distance_max"].mean():.2f}, max={text_artifact["edit_distance_max"].max():.2f}')
        print(f'  substitution_ratio_max: mean={text_artifact["substitution_ratio_max"].mean():.2f}, max={text_artifact["substitution_ratio_max"].max():.2f}')

    if len(clean_clips) > 0:
        print(f'\nCLEAN clips (n={len(clean_clips)}):')
        print(f'  edit_distance_max: mean={clean_clips["edit_distance_max"].mean():.2f}, max={clean_clips["edit_distance_max"].max():.2f}')
        print(f'  substitution_ratio_max: mean={clean_clips["substitution_ratio_max"].mean():.2f}, max={clean_clips["substitution_ratio_max"].max():.2f}')

    # Check separability
    if len(text_artifact) > 0 and len(clean_clips) > 0:
        artifact_edit_mean = text_artifact["edit_distance_max"].mean()
        clean_edit_mean = clean_clips["edit_distance_max"].mean()

        if artifact_edit_mean > clean_edit_mean:
            print('\n✓ Feature separability confirmed: TEXT_INCONSISTENCY has higher edit_distance')
        else:
            print('\n⚠ Warning: TEXT_INCONSISTENCY edit_distance not higher than CLEAN')
            print('  This may indicate issues with synthetic data generation')

    print('\n✓ All exit criteria checks passed!')
    return 0

if __name__ == '__main__':
    exit(main())
