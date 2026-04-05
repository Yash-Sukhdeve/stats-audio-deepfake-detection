#!/usr/bin/env python3
"""
Experiment: Measure Feature-Dependent Domain Shift (Theorem 6)
Purpose: Compare domain shift magnitudes for different feature types
Method: Maximum Mean Discrepancy (MMD) and histogram-based Total Variation
Expected: d_formants < d_prosody

Author: ML Research Team
Date: November 2024
Paper: Feature-Dependent Domain Adaptation for Multi-Modal Learning (ICML 2026)
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility (Rule R5)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class DomainShiftEstimator:
    """
    Estimates domain shift using multiple methods for robustness.

    Methods:
    1. Maximum Mean Discrepancy (MMD) - primary method for high-dimensional features
    2. Total Variation Distance - for low-dimensional features
    3. Wasserstein Distance - additional validation

    References:
    - Gretton et al. (2012): "A Kernel Two-Sample Test"
    - Villani (2008): "Optimal Transport: Old and New"
    - Sriperumbudur et al. (2010): "Hilbert Space Embeddings and Metrics on
      Probability Measures"
    """

    def __init__(self, random_state: int = 42):
        """Initialize estimator."""
        self.random_state = random_state
        np.random.seed(random_state)

    def compute_mmd(self, X_source: np.ndarray, X_target: np.ndarray,
                   kernel: str = 'rbf', gamma: Optional[float] = None) -> Dict:
        """
        Compute Maximum Mean Discrepancy between source and target distributions.

        MMD is a distance measure between distributions in reproducing kernel
        Hilbert space (RKHS). It's particularly suitable for high-dimensional data.

        MMD²(P, Q) = E[k(X, X')] - 2E[k(X, Y)] + E[k(Y, Y')]

        Reference:
        - Gretton et al. (2012): "A Kernel Two-Sample Test"

        Args:
            X_source: Source domain samples (n_source, n_features)
            X_target: Target domain samples (n_target, n_features)
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            gamma: RBF kernel bandwidth (auto-tuned if None)

        Returns:
            Dictionary with MMD value and p-value from permutation test
        """
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]

        # Standardize features
        scaler = StandardScaler()
        X_source_scaled = scaler.fit_transform(X_source)
        X_target_scaled = scaler.transform(X_target)

        # Auto-tune gamma using median heuristic
        if kernel == 'rbf' and gamma is None:
            # Median heuristic for RBF bandwidth
            X_combined = np.vstack([X_source_scaled, X_target_scaled])
            pairwise_dists = cdist(X_combined, X_combined, 'euclidean')
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])
            gamma = 1.0 / (2 * median_dist ** 2)

        # Compute kernel matrices
        K_ss = self._compute_kernel(X_source_scaled, X_source_scaled, kernel, gamma)
        K_tt = self._compute_kernel(X_target_scaled, X_target_scaled, kernel, gamma)
        K_st = self._compute_kernel(X_source_scaled, X_target_scaled, kernel, gamma)

        # Compute unbiased MMD² statistic
        # Remove diagonal terms for unbiased estimate
        np.fill_diagonal(K_ss, 0)
        np.fill_diagonal(K_tt, 0)

        term1 = np.sum(K_ss) / (n_source * (n_source - 1))
        term2 = np.sum(K_tt) / (n_target * (n_target - 1))
        term3 = 2 * np.sum(K_st) / (n_source * n_target)

        mmd_squared = term1 + term2 - term3
        mmd = np.sqrt(max(0, mmd_squared))  # Ensure non-negative

        # Permutation test for p-value
        p_value = self._mmd_permutation_test(X_source_scaled, X_target_scaled,
                                            mmd_squared, kernel, gamma,
                                            n_permutations=1000)

        return {
            'mmd': float(mmd),
            'mmd_squared': float(mmd_squared),
            'p_value': float(p_value),
            'gamma': float(gamma) if gamma else None,
            'significant': p_value < 0.05
        }

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray,
                       kernel: str, gamma: Optional[float]) -> np.ndarray:
        """Compute kernel matrix."""
        if kernel == 'rbf':
            # RBF kernel: k(x, y) = exp(-gamma * ||x - y||²)
            pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
            K = np.exp(-gamma * pairwise_sq_dists)
        elif kernel == 'linear':
            # Linear kernel: k(x, y) = <x, y>
            K = X @ Y.T
        elif kernel == 'polynomial':
            # Polynomial kernel: k(x, y) = (1 + <x, y>)²
            K = (1 + X @ Y.T) ** 2
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        return K

    def _mmd_permutation_test(self, X_source: np.ndarray, X_target: np.ndarray,
                             observed_mmd: float, kernel: str, gamma: float,
                             n_permutations: int = 1000) -> float:
        """
        Permutation test for MMD significance.

        Under null hypothesis (same distribution), permuting labels shouldn't
        change MMD value significantly.
        """
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        X_combined = np.vstack([X_source, X_target])

        permuted_mmd_values = []

        for _ in range(n_permutations):
            # Permute combined data
            perm_idx = np.random.permutation(n_source + n_target)
            X_perm = X_combined[perm_idx]

            # Split into fake source and target
            X_perm_source = X_perm[:n_source]
            X_perm_target = X_perm[n_source:]

            # Compute MMD for permuted data
            K_ss = self._compute_kernel(X_perm_source, X_perm_source, kernel, gamma)
            K_tt = self._compute_kernel(X_perm_target, X_perm_target, kernel, gamma)
            K_st = self._compute_kernel(X_perm_source, X_perm_target, kernel, gamma)

            np.fill_diagonal(K_ss, 0)
            np.fill_diagonal(K_tt, 0)

            term1 = np.sum(K_ss) / (n_source * (n_source - 1))
            term2 = np.sum(K_tt) / (n_target * (n_target - 1))
            term3 = 2 * np.sum(K_st) / (n_source * n_target)

            mmd_perm = term1 + term2 - term3
            permuted_mmd_values.append(mmd_perm)

        # P-value: proportion of permuted MMD values >= observed
        p_value = np.mean(np.array(permuted_mmd_values) >= observed_mmd)

        return p_value

    def compute_tv_distance(self, X_source: np.ndarray, X_target: np.ndarray,
                          n_bins: int = 50) -> float:
        """
        Compute Total Variation distance using histogram approximation.

        TV(P, Q) = 0.5 * Σ|P(x) - Q(x)|

        This method is suitable for low-dimensional features where we can
        reasonably estimate histograms.

        Reference:
        - Tsybakov (2009): "Introduction to Nonparametric Estimation"

        Args:
            X_source: Source samples (n_source, n_features)
            X_target: Target samples (n_target, n_features)
            n_bins: Number of histogram bins per dimension

        Returns:
            Estimated TV distance
        """
        if X_source.shape[1] > 3:
            # For high dimensions, project to 1D using first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1, random_state=self.random_state)
            X_source_1d = pca.fit_transform(X_source).flatten()
            X_target_1d = pca.transform(X_target).flatten()
        else:
            # For low dimensions, use first dimension
            X_source_1d = X_source[:, 0]
            X_target_1d = X_target[:, 0]

        # Compute histogram range
        min_val = min(X_source_1d.min(), X_target_1d.min())
        max_val = max(X_source_1d.max(), X_target_1d.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Compute normalized histograms
        hist_source, _ = np.histogram(X_source_1d, bins=bins, density=True)
        hist_target, _ = np.histogram(X_target_1d, bins=bins, density=True)

        # Normalize to probabilities
        bin_width = bins[1] - bins[0]
        hist_source *= bin_width
        hist_target *= bin_width

        # TV distance
        tv = 0.5 * np.sum(np.abs(hist_source - hist_target))

        return float(tv)

    def compute_wasserstein(self, X_source: np.ndarray, X_target: np.ndarray) -> float:
        """
        Compute 1-Wasserstein distance (Earth Mover's Distance).

        This provides another perspective on distribution distance.

        Reference:
        - Villani (2008): "Optimal Transport: Old and New"

        Args:
            X_source: Source samples
            X_target: Target samples

        Returns:
            Wasserstein distance
        """
        if X_source.shape[1] > 3:
            # For high dimensions, use first few principal components
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3, random_state=self.random_state)
            X_source_reduced = pca.fit_transform(X_source)
            X_target_reduced = pca.transform(X_target)

            # Average over dimensions
            w_distances = []
            for d in range(X_source_reduced.shape[1]):
                w_d = wasserstein_distance(X_source_reduced[:, d],
                                          X_target_reduced[:, d])
                w_distances.append(w_d)
            return float(np.mean(w_distances))
        else:
            # For low dimensions, compute directly
            if X_source.shape[1] == 1:
                return float(wasserstein_distance(X_source.flatten(),
                                                 X_target.flatten()))
            else:
                # Average over dimensions
                w_distances = []
                for d in range(X_source.shape[1]):
                    w_d = wasserstein_distance(X_source[:, d], X_target[:, d])
                    w_distances.append(w_d)
                return float(np.mean(w_distances))


def load_features(year: str = '2019') -> Dict[str, np.ndarray]:
    """Load REAL extracted v4 acoustic features.

    Previously this generated synthetic data (REMOVED - scientific integrity issue).
    Now loads actual pre-extracted features from the feature pipeline.
    """
    features_dir = Path(__file__).parent / 'features'

    if year == '2019':
        feat_file = features_dir / 'acoustic_v4_eval_2019.npy'
        label_file = features_dir / 'labels_eval_2019.npy'
    elif year == '2021':
        feat_file = features_dir / 'acoustic_v4_eval_2021.npy'
        label_file = features_dir / 'labels_eval_2021.npy'
    else:
        raise ValueError(f"Unknown year: {year}")

    # Check if fixed features exist, fall back to original
    fixed_file = feat_file.parent / feat_file.name.replace('.npy', '_fixed.npy')
    if fixed_file.exists():
        features = np.load(str(fixed_file))
        print(f"  Loaded FIXED features: {fixed_file}")
    elif feat_file.exists():
        features = np.load(str(feat_file))
        print(f"  Loaded features: {feat_file}")
    else:
        raise FileNotFoundError(f"Feature file not found: {feat_file}")

    labels = np.load(str(label_file))

    # v4 36-dim feature layout
    FEATURE_GROUPS = {
        'PNCC': slice(0, 13),
        'LFCC': slice(13, 21),
        'Formants': slice(21, 29),
        'Prosody': slice(29, 36),
    }

    FORMANT_INDICES = list(range(21, 29))  # v4: indices 21-28
    PROSODY_INDICES = list(range(29, 36))  # v4: indices 29-35

    return {
        'features': features,
        'labels': labels,
        'feature_groups': FEATURE_GROUPS,
        'formant_indices': FORMANT_INDICES,
        'prosody_indices': PROSODY_INDICES,
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        # Backward-compatible keys used by main()
        'all': features,
        'formants': features[:, FORMANT_INDICES],
        'prosody': features[:, PROSODY_INDICES],
        'spectral': features[:, FEATURE_GROUPS['PNCC']],
        'temporal': features[:, FEATURE_GROUPS['LFCC']],
    }


def visualize_distributions(features_source: Dict, features_target: Dict,
                          save_path: str = 'domain_shift_visualization.png'):
    """
    Visualize source vs target distributions for different feature types.
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Domain Shift Visualization: ASVspoof2019 (source) vs ASVspoof2021 (target)',
                    fontsize=14)

        feature_types = ['formants', 'prosody']
        feature_names = {
            'formants': ['F1 (Hz)', 'F2 (Hz)', 'F3 (Hz)'],
            'prosody': ['F0 (Hz)', 'Jitter', 'Shimmer']
        }

        for row, feat_type in enumerate(feature_types):
            for col in range(3):
                ax = axes[row, col]

                # Get feature data
                source_data = features_source[feat_type][:, col]
                target_data = features_target[feat_type][:, col]

                # Plot histograms
                ax.hist(source_data, bins=30, alpha=0.5, label='Source (2019)',
                       color='blue', density=True)
                ax.hist(target_data, bins=30, alpha=0.5, label='Target (2021)',
                       color='red', density=True)

                ax.set_xlabel(feature_names[feat_type][col])
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add statistics
                shift = np.abs(source_data.mean() - target_data.mean())
                ax.set_title(f'Mean shift: {shift:.3f}')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')  # Ensure all figures are closed

        print(f"Visualization saved to {save_path}")
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")


def compute_confidence_intervals(estimator: DomainShiftEstimator,
                                features_source: np.ndarray,
                                features_target: np.ndarray,
                                n_bootstrap: int = 1000) -> Dict:
    """
    Compute bootstrap confidence intervals for domain shift measures.
    """
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]

    mmd_values = []
    tv_values = []
    w_values = []

    for _ in range(n_bootstrap):
        # Bootstrap samples
        idx_source = np.random.choice(n_source, size=n_source, replace=True)
        idx_target = np.random.choice(n_target, size=n_target, replace=True)

        X_source_boot = features_source[idx_source]
        X_target_boot = features_target[idx_target]

        # Compute metrics
        mmd_result = estimator.compute_mmd(X_source_boot, X_target_boot)
        mmd_values.append(mmd_result['mmd'])

        tv = estimator.compute_tv_distance(X_source_boot, X_target_boot)
        tv_values.append(tv)

        w = estimator.compute_wasserstein(X_source_boot, X_target_boot)
        w_values.append(w)

    # Compute statistics
    results = {}
    for name, values in [('mmd', mmd_values), ('tv', tv_values),
                         ('wasserstein', w_values)]:
        values = np.array(values)
        results[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'ci_lower': float(np.percentile(values, 2.5)),
            'ci_upper': float(np.percentile(values, 97.5))
        }

    return results


def main():
    """Main experimental validation for Theorem 6."""

    print("="*80)
    print("EXPERIMENT: Measure Feature-Dependent Domain Shift (Theorem 6)")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # Load features
    print("Loading features...")
    features_2019 = load_features('2019')  # Source domain
    features_2021 = load_features('2021')  # Target domain

    print(f"Source domain (2019): {features_2019['all'].shape[0]} samples")
    print(f"Target domain (2021): {features_2021['all'].shape[0]} samples")
    print(f"Feature dimensions: {features_2019['all'].shape[1]}")
    print()

    # Initialize estimator
    estimator = DomainShiftEstimator(random_state=RANDOM_SEED)

    # Visualize distributions
    print("Generating visualization...")
    visualize_distributions(features_2019, features_2021)
    print()

    # Measure domain shift for different feature types
    results = {}
    feature_types = ['formants', 'prosody', 'spectral', 'temporal', 'all']

    print("MEASURING DOMAIN SHIFT")
    print("-" * 80)

    for feat_type in feature_types:
        print(f"\nFeature type: {feat_type.upper()}")
        print("-" * 40)

        X_source = features_2019[feat_type]
        X_target = features_2021[feat_type]

        # Method 1: Maximum Mean Discrepancy
        mmd_result = estimator.compute_mmd(X_source, X_target)
        print(f"MMD: {mmd_result['mmd']:.4f} (p-value: {mmd_result['p_value']:.4f})")

        # Method 2: Total Variation Distance
        tv_distance = estimator.compute_tv_distance(X_source, X_target)
        print(f"Total Variation: {tv_distance:.4f}")

        # Method 3: Wasserstein Distance
        w_distance = estimator.compute_wasserstein(X_source, X_target)
        print(f"Wasserstein: {w_distance:.4f}")

        # Compute confidence intervals
        ci_results = compute_confidence_intervals(estimator, X_source, X_target,
                                                 n_bootstrap=1000)

        results[feat_type] = {
            'mmd': mmd_result,
            'tv': tv_distance,
            'wasserstein': w_distance,
            'confidence_intervals': ci_results
        }

    # Compare formants vs prosody (main hypothesis)
    print("\n" + "="*80)
    print("HYPOTHESIS TEST: d_formants < d_prosody")
    print("="*80)

    d_formants = results['formants']['confidence_intervals']['mmd']['mean']
    d_prosody = results['prosody']['confidence_intervals']['mmd']['mean']

    print(f"\nDomain shift (MMD):")
    print(f"  Formants: {d_formants:.4f} ± {results['formants']['confidence_intervals']['mmd']['std']:.4f}")
    print(f"  Prosody:  {d_prosody:.4f} ± {results['prosody']['confidence_intervals']['mmd']['std']:.4f}")
    print(f"  Ratio: {d_formants/d_prosody:.2f}")

    # Statistical test
    hypothesis_supported = d_formants < d_prosody

    print(f"\nHypothesis d_formants < d_prosody: {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")

    if hypothesis_supported:
        print(f"Formant features show {(1 - d_formants/d_prosody)*100:.1f}% less domain shift than prosody features")

    # Additional analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    # Rank features by domain shift
    shift_ranking = sorted([(feat, results[feat]['confidence_intervals']['mmd']['mean'])
                          for feat in feature_types if feat != 'all'],
                         key=lambda x: x[1])

    print("\nFeatures ranked by domain shift (least to most):")
    for i, (feat, shift) in enumerate(shift_ranking, 1):
        print(f"  {i}. {feat:10s}: {shift:.4f}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    print("\n1. DOMAIN SHIFT ANALYSIS:")
    print("   Domain shift analysis complete. Results should be interpreted in context of Theorem 6.")
    if hypothesis_supported:
        print("   - Formant features show less domain shift than prosody features")
        print("   - Consistent with feature-dependent adaptation hypothesis")
    else:
        print("   - Formant features did not show less domain shift than prosody")
        print("   - Further investigation needed")

    print("\n2. PRACTICAL IMPLICATIONS:")
    print("   - Use formant features for cross-domain tasks")
    print("   - Apply stronger adaptation to prosody features")
    print("   - Consider feature-specific adaptation weights")

    print("\n3. STATISTICAL SIGNIFICANCE:")
    for feat in ['formants', 'prosody']:
        if results[feat]['mmd']['significant']:
            print(f"   - {feat}: Significant domain shift detected (p < 0.05)")
        else:
            print(f"   - {feat}: No significant domain shift (p ≥ 0.05)")

    # Save results
    output_data = {
        'experiment': 'measure_domain_shift',
        'date': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'source_domain': 'ASVspoof2019_LA',
        'target_domain': 'ASVspoof2021_DF',
        'results': results,
        'hypothesis_test': {
            'd_formants': float(d_formants),
            'd_prosody': float(d_prosody),
            'ratio': float(d_formants/d_prosody),
            'hypothesis_supported': hypothesis_supported
        },
        'feature_ranking': shift_ranking
    }

    output_file = 'domain_shift_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print("="*80)

    return output_data


if __name__ == "__main__":
    results = main()