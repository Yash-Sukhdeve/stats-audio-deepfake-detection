#!/usr/bin/env python3
"""
Compute correlation matrix for acoustic features and identify highly correlated pairs.

Per OPUS Ultrathink analysis:
- Remove features with r > 0.90 (redundant information)
- Monitor features with r > 0.85
- Keep features with r < 0.70 (independent information)

Author: Claude (Opus 4.5)
Date: November 27, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Feature names for 64-dim acoustic features
FEATURE_NAMES_64 = [
    # MFCCs (13)
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7',
    'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13',
    # Delta MFCCs (13)
    'dMFCC_1', 'dMFCC_2', 'dMFCC_3', 'dMFCC_4', 'dMFCC_5', 'dMFCC_6', 'dMFCC_7',
    'dMFCC_8', 'dMFCC_9', 'dMFCC_10', 'dMFCC_11', 'dMFCC_12', 'dMFCC_13',
    # Spectral (12)
    'Centroid', 'Bandwidth', 'Rolloff', 'Flatness', 'ZCR',
    'Contrast_1', 'Contrast_2', 'Contrast_3', 'Contrast_4', 'Contrast_5', 'Contrast_6', 'Contrast_7',
    # Energy (6)
    'RMS_mean', 'RMS_std', 'RMS_max', 'RMS_min', 'Duration', 'Tempo',
    # Formants (3)
    'F1', 'F2', 'F3',
    # Prosody (6)
    'F0_mean', 'F0_std', 'Jitter', 'Shimmer', 'HNR', 'Speech_rate',
    # Padding (11)
    'Pad_1', 'Pad_2', 'Pad_3', 'Pad_4', 'Pad_5', 'Pad_6', 'Pad_7', 'Pad_8', 'Pad_9', 'Pad_10', 'Pad_11'
]

# Feature names for 36-dim v4 codec-robust features (current standard)
FEATURE_NAMES_V4 = [
    # PNCC (13) - power-normalized cepstral coefficients, gammatone filterbank
    'PNCC_0', 'PNCC_1', 'PNCC_2', 'PNCC_3', 'PNCC_4', 'PNCC_5',
    'PNCC_6', 'PNCC_7', 'PNCC_8', 'PNCC_9', 'PNCC_10', 'PNCC_11', 'PNCC_12',
    # LFCC (8) - linear frequency cepstral coefficients
    'LFCC_0', 'LFCC_1', 'LFCC_2', 'LFCC_3', 'LFCC_4', 'LFCC_5', 'LFCC_6', 'LFCC_7',
    # Formants (8) - vocal tract resonance features
    'F1_mean', 'F1_std', 'F1_bw', 'F2_mean', 'F2_std', 'F2_bw', 'F1F2_ratio', 'F2F1_diff',
    # Prosody (7) - fundamental frequency and energy features
    'F0_mean', 'F0_std', 'F0_max', 'voiced_ratio', 'energy_mean', 'energy_std', 'duration'
]

# Extended feature names for 72-dim enhanced features
FEATURE_NAMES_72 = FEATURE_NAMES_64[:53] + [
    # Phase coherence (4)
    'IPD_mean', 'IPD_std', 'GD_neg_pct', 'GD_smoothness',
    # MODGD (4)
    'MODGD_1', 'MODGD_2', 'MODGD_3', 'MODGD_4',
    # Padding (11)
    'Pad_1', 'Pad_2', 'Pad_3', 'Pad_4', 'Pad_5', 'Pad_6', 'Pad_7', 'Pad_8', 'Pad_9', 'Pad_10', 'Pad_11'
]

# Features to analyze (skip padding)
ACTIVE_FEATURES_64 = 53  # Exclude padding
ACTIVE_FEATURES_72 = 61  # Exclude padding


def compute_correlation_matrix(features_path: str, output_dir: str):
    """Compute and analyze correlation matrix."""
    print(f"Loading features from {features_path}")
    features = np.load(features_path)
    n_samples, n_features = features.shape
    print(f"Features shape: {features.shape}")

    # Determine feature version by dimensionality
    if n_features == 36:
        feature_names = FEATURE_NAMES_V4
        active_features = 36
    elif n_features == 72:
        feature_names = FEATURE_NAMES_72
        active_features = ACTIVE_FEATURES_72
    else:
        feature_names = FEATURE_NAMES_64
        active_features = ACTIVE_FEATURES_64

    # Use only active features (exclude padding)
    features = features[:, :active_features]
    feature_names = feature_names[:active_features]

    print(f"Analyzing {active_features} active features (excluding padding)")

    # Compute correlation matrix
    print("Computing correlation matrix...")
    corr = np.corrcoef(features.T)

    # Handle NaN values (from constant features)
    corr = np.nan_to_num(corr, nan=0.0)

    # Save correlation matrix
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'correlation_matrix.npy'), corr)

    # Find highly correlated pairs
    print("\n" + "="*60)
    print("HIGHLY CORRELATED FEATURES (r > 0.85)")
    print("="*60)

    high_corr_pairs = []
    very_high_corr_pairs = []

    for i in range(active_features):
        for j in range(i+1, active_features):
            r = corr[i, j]
            if abs(r) > 0.90:
                very_high_corr_pairs.append((feature_names[i], feature_names[j], r))
                print(f"  REMOVE: {feature_names[i]} <-> {feature_names[j]}: r = {r:.4f}")
            elif abs(r) > 0.85:
                high_corr_pairs.append((feature_names[i], feature_names[j], r))
                print(f"  MONITOR: {feature_names[i]} <-> {feature_names[j]}: r = {r:.4f}")

    print(f"\nTotal pairs with r > 0.90: {len(very_high_corr_pairs)}")
    print(f"Total pairs with 0.85 < r < 0.90: {len(high_corr_pairs)}")

    # Save highly correlated pairs
    with open(os.path.join(output_dir, 'high_correlation_pairs.txt'), 'w') as f:
        f.write("# Features to REMOVE (r > 0.90)\n")
        for name1, name2, r in very_high_corr_pairs:
            f.write(f"{name1} <-> {name2}: r = {r:.4f}\n")
        f.write("\n# Features to MONITOR (0.85 < r < 0.90)\n")
        for name1, name2, r in high_corr_pairs:
            f.write(f"{name1} <-> {name2}: r = {r:.4f}\n")

    # Generate correlation heatmap
    print("\nGenerating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(20, 16))

    # Use shortened names for readability
    short_names = [n[:8] for n in feature_names]

    sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=short_names, yticklabels=short_names, ax=ax)

    plt.title(f'Feature Correlation Matrix ({active_features} features, {n_samples} samples)', fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.pdf'), bbox_inches='tight')
    print(f"Saved heatmap to {output_dir}/correlation_heatmap.png")

    plt.close()

    # Generate summary statistics
    print("\n" + "="*60)
    print("CORRELATION SUMMARY STATISTICS")
    print("="*60)

    # Get upper triangle (excluding diagonal)
    upper_tri = corr[np.triu_indices(active_features, k=1)]

    print(f"Mean absolute correlation: {np.mean(np.abs(upper_tri)):.4f}")
    print(f"Max absolute correlation: {np.max(np.abs(upper_tri)):.4f}")
    print(f"Median absolute correlation: {np.median(np.abs(upper_tri)):.4f}")

    # Correlation distribution
    print("\nCorrelation distribution:")
    print(f"  |r| < 0.30 (low): {np.sum(np.abs(upper_tri) < 0.30)} pairs")
    print(f"  0.30 <= |r| < 0.70 (moderate): {np.sum((np.abs(upper_tri) >= 0.30) & (np.abs(upper_tri) < 0.70))} pairs")
    print(f"  0.70 <= |r| < 0.85 (high): {np.sum((np.abs(upper_tri) >= 0.70) & (np.abs(upper_tri) < 0.85))} pairs")
    print(f"  0.85 <= |r| < 0.90 (very high): {np.sum((np.abs(upper_tri) >= 0.85) & (np.abs(upper_tri) < 0.90))} pairs")
    print(f"  |r| >= 0.90 (redundant): {np.sum(np.abs(upper_tri) >= 0.90)} pairs")

    # Recommend features to remove
    print("\n" + "="*60)
    print("RECOMMENDED FEATURE REMOVALS")
    print("="*60)

    # Identify features that are highly correlated with many others
    corr_count = np.sum(np.abs(corr) > 0.90, axis=1) - 1  # -1 for diagonal

    features_to_remove = []
    for i, count in enumerate(corr_count):
        if count > 0:
            features_to_remove.append((feature_names[i], count))

    features_to_remove.sort(key=lambda x: -x[1])

    # Build removal recommendation based on codec robustness analysis
    codec_sensitive = ['Jitter', 'Shimmer', 'HNR', 'F3', 'ZCR']
    mfcc_high_order = ['MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13']
    delta_mfcc = [f'dMFCC_{i}' for i in range(1, 14)]

    print("\nBased on codec robustness + correlation analysis:")
    print("REMOVE (codec-sensitive OR highly correlated):")

    remove_set = set()
    for name, count in features_to_remove:
        if count > 0 or name in codec_sensitive or name in mfcc_high_order or name in delta_mfcc:
            remove_set.add(name)
            print(f"  - {name}: {count} high correlations, codec-sensitive={name in codec_sensitive}")

    # Add codec-sensitive features
    for name in codec_sensitive + mfcc_high_order + delta_mfcc:
        if name in feature_names and name not in remove_set:
            remove_set.add(name)
            print(f"  - {name}: codec-sensitive (from literature)")

    print(f"\nTotal features to remove: {len(remove_set)}")
    print(f"Remaining features: {active_features - len(remove_set)}")

    # Save removal recommendations
    with open(os.path.join(output_dir, 'features_to_remove.txt'), 'w') as f:
        f.write("# Features to remove (codec-sensitive OR r > 0.90)\n")
        for name in sorted(remove_set):
            f.write(f"{name}\n")

    print(f"\nSaved recommendations to {output_dir}/features_to_remove.txt")

    return corr, feature_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute feature correlation matrix')
    parser.add_argument('--features', type=str,
                       default='/home/lab2208/Documents/df_detection/evidence/experiments/features/acoustic_features_train_2019.npy',
                       help='Path to features file')
    parser.add_argument('--output', type=str,
                       default='/home/lab2208/Documents/df_detection/evidence/experiments/correlation_analysis',
                       help='Output directory')
    args = parser.parse_args()

    compute_correlation_matrix(args.features, args.output)
