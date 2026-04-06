#!/usr/bin/env python3
"""
Discriminative Feature Analysis: Which acoustic features distinguish real from fake audio?

This script answers the CORE research question that was never addressed:
    "Which of our 36 acoustic features can identify deepfake audio regardless of codec?"

The analysis has 5 phases:
    Phase 1: Discriminative power (real vs fake) per feature — Wilcoxon, Cohen's d, AUC
    Phase 2: Cross-reference with codec robustness (ICC from prior analysis)
    Phase 3: Codec-conditioned discrimination (ASVspoof 2021, per-codec AUC)
    Phase 4: Explainability — distribution plots, acoustic rationale
    Phase 5: Final ranked table — discriminative_power x codec_robustness

Statistical Methods:
    - Wilcoxon rank-sum (Mann-Whitney U): nonparametric, no normality assumption
    - Cohen's d (unpaired): standardized effect size
    - AUC-ROC per feature: threshold-free discrimination metric
    - Benjamini-Hochberg FDR correction for 36 simultaneous tests
    - Glass's delta as robustness check (unequal variance-aware effect size)

Data:
    - ASVspoof 2019 LA train: 2,580 bonafide + 22,800 spoof (6 attack types A01-A06)
    - ASVspoof 2019 LA eval: 7,355 bonafide + 63,882 spoof (13 attack types A07-A19)
    - ASVspoof 2021 DF eval: 181,566 files across 7 codec conditions

Feature layout (36-dim, v4 extraction):
    [0:13]   PNCC  — Power-Normalized Cepstral Coefficients (gammatone filterbank)
    [13:21]  LFCC  — Linear Frequency Cepstral Coefficients
    [21:29]  Formants — F1_mean, F1_std, F1_bw, F1_bw_std, F2_mean, F2_std, F2_bw, F2_bw_std
    [29:36]  Prosody — F0_mean, F0_std, F0_max, speech_rate, pause_ratio, duration, energy

Citations:
    - Kim & Stern (2016) "PNCC for Robust Speech Recognition" IEEE/ACM TASLP 24(7):1315-1329
    - Todisco et al. (2019) "ASVspoof 2019" Interspeech, pp. 1008-1012
    - Yamagishi et al. (2023) "ASVspoof 2021" IEEE TASLP 31:2507-2522
    - Benjamini & Hochberg (1995) "Controlling FDR" JRSS-B 57(1):289-300
    - Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences" 2nd ed.

Author: Research analysis script
Date: 2026-04-06
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.metrics import roc_auc_score
from pathlib import Path
from collections import OrderedDict
import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/home/lab2208/Documents/df_detection")
FEATURES_DIR = PROJECT_ROOT / "evidence" / "experiments" / "features"
PROTOCOL_DIR = PROJECT_ROOT / "data" / "asvspoof" / "asvspoof2019" / "LA" / "ASVspoof2019_LA_cm_protocols"
EVAL_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"
TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
ICC_RESULTS = PROJECT_ROOT / "evidence" / "experiments" / "comprehensive_feature_analysis_v4_fixed" / "analysis_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "evidence" / "experiments" / "discriminative_analysis"

# Feature name mapping for the 36-dim v4 vector
FEATURE_NAMES = [
    # PNCC (13 dims)
    "PNCC_0", "PNCC_1", "PNCC_2", "PNCC_3", "PNCC_4", "PNCC_5",
    "PNCC_6", "PNCC_7", "PNCC_8", "PNCC_9", "PNCC_10", "PNCC_11", "PNCC_12",
    # LFCC (8 dims)
    "LFCC_0", "LFCC_1", "LFCC_2", "LFCC_3", "LFCC_4", "LFCC_5", "LFCC_6", "LFCC_7",
    # Formants (8 dims)
    "F1_mean", "F1_std", "F1_bw", "F1_bw_std", "F2_mean", "F2_std", "F2_bw", "F2_bw_std",
    # Prosody (7 dims)
    "F0_mean", "F0_std", "F0_max", "speech_rate", "pause_ratio", "duration", "energy",
]

FEATURE_GROUPS = {
    "PNCC": list(range(0, 13)),
    "LFCC": list(range(13, 21)),
    "Formants": list(range(21, 29)),
    "Prosody": list(range(29, 36)),
}

# ASVspoof 2019 LA train attack types
ATTACK_TYPES_TRAIN = ["A01", "A02", "A03", "A04", "A05", "A06"]
# ASVspoof 2019 LA eval attack types
ATTACK_TYPES_EVAL = [f"A{i:02d}" for i in range(7, 20)]

# ASVspoof 2021 codec conditions
CODEC_CONDITIONS = ["none", "alaw", "ulaw", "g722", "gsm", "opus", "pstn"]

# Statistical thresholds
ALPHA = 0.05  # significance level before FDR correction
MIN_EFFECT_SIZE = 0.2  # Cohen's d threshold for "small" effect
MIN_AUC = 0.55  # minimum AUC to consider feature discriminative


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cohens_d_unpaired(group1, group2):
    """
    Compute Cohen's d for two independent groups (unpaired).

    d = (mean1 - mean2) / pooled_std
    pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

    Reference: Cohen (1988) eq. 2.3.2
    """
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def glass_delta(group1, group2):
    """
    Glass's delta: effect size using only the control group's std.
    More appropriate when group variances are unequal.
    Uses group1 (bonafide) as the reference group.

    delta = (mean1 - mean2) / std1
    """
    s1 = np.std(group1, ddof=1)
    if s1 == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / s1


def cliff_delta(group1, group2, max_samples=5000):
    """
    Cliff's delta: non-parametric effect size.
    Range [-1, 1]. Interpretation:
      |d| < 0.147: negligible
      |d| < 0.33: small
      |d| < 0.474: medium
      |d| >= 0.474: large

    Reference: Cliff (1993) "Dominance statistics" Psych Bulletin 114(3):494-509

    Subsampled for computational feasibility with large datasets.
    """
    if len(group1) > max_samples:
        rng = np.random.RandomState(42)
        group1 = rng.choice(group1, max_samples, replace=False)
    if len(group2) > max_samples:
        rng = np.random.RandomState(43)
        group2 = rng.choice(group2, max_samples, replace=False)

    # Vectorized comparison
    n1, n2 = len(group1), len(group2)
    # Use broadcasting: (n1, 1) vs (1, n2)
    comparisons = np.sign(group1[:, None] - group2[None, :])
    return np.mean(comparisons)


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Reference: Benjamini & Hochberg (1995) JRSS-B 57(1):289-300

    Returns:
        rejected: boolean array of rejected hypotheses
        adjusted_p: adjusted p-values
    """
    n = len(p_values)
    p_array = np.array(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    bh_critical = (ranks / n) * alpha

    # Find largest k where p_(k) <= bh_critical_(k)
    rejected_sorted = sorted_p <= bh_critical

    # Adjusted p-values (step-up)
    adjusted_p = np.zeros(n)
    adjusted_sorted = np.minimum(sorted_p * n / ranks, 1.0)
    # Enforce monotonicity from right to left
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Unsort
    adjusted_p[sorted_indices] = adjusted_sorted
    rejected = adjusted_p <= alpha

    return rejected, adjusted_p


def per_feature_auc(features, labels, feature_idx):
    """
    Compute AUC-ROC for a single feature as a univariate classifier.

    Convention: label=0 is bonafide (negative), label=1 is spoof (positive).
    We compute max(AUC, 1-AUC) because the direction of discrimination matters
    but we want the absolute discriminative power.

    Returns: (auc, direction)
        auc: in [0.5, 1.0]
        direction: 'higher_in_spoof' or 'higher_in_bonafide'
    """
    feat = features[:, feature_idx]

    # Remove NaN/Inf
    valid = np.isfinite(feat)
    feat_valid = feat[valid]
    labels_valid = labels[valid]

    if len(np.unique(labels_valid)) < 2:
        return 0.5, "undefined"

    auc = roc_auc_score(labels_valid, feat_valid)

    if auc >= 0.5:
        return auc, "higher_in_spoof"
    else:
        return 1 - auc, "higher_in_bonafide"


def parse_protocol_attacks(protocol_path):
    """
    Parse ASVspoof protocol file to extract per-file attack types.

    Format: speaker_id file_id - attack_type label
    For bonafide files, attack_type is "-".

    Returns: dict mapping file_id -> attack_type (or "bonafide")
    """
    attacks = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]
                attack = parts[3]
                label = parts[4]
                if label == "bonafide":
                    attacks[file_id] = "bonafide"
                else:
                    attacks[file_id] = attack
    return attacks


def load_attack_labels_train():
    """
    Load per-sample attack type labels for the ASVspoof 2019 LA training set.

    Returns: numpy array of shape (25380,) with attack type strings
             ("bonafide", "A01", ..., "A06")
    """
    attack_labels = []
    with open(TRAIN_PROTOCOL) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                label = parts[4]
                if label == "bonafide":
                    attack_labels.append("bonafide")
                else:
                    attack_labels.append(parts[3])
    return np.array(attack_labels)


def interpret_effect_size(d):
    """Interpret Cohen's d magnitude per Cohen (1988)."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cliff_delta(d):
    """Interpret Cliff's delta per Cliff (1993)."""
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"


# ============================================================================
# PHASE 1: DISCRIMINATIVE ANALYSIS (Real vs Fake)
# ============================================================================

def phase1_discriminative_analysis(features, labels, attack_labels=None):
    """
    For each of 36 features: compare bonafide vs spoof distributions.

    Statistical tests:
        - Wilcoxon rank-sum (Mann-Whitney U): nonparametric, unpaired
        - Cohen's d (unpaired): parametric effect size
        - Glass's delta: effect size robust to unequal variances
        - Cliff's delta: nonparametric effect size
        - Per-feature AUC-ROC: threshold-free discrimination

    Multiple testing correction:
        - Benjamini-Hochberg FDR at alpha=0.05 for 36 tests

    Returns: dict with per-feature results
    """
    print("=" * 80)
    print("PHASE 1: DISCRIMINATIVE ANALYSIS — Real vs Fake")
    print("=" * 80)

    n_features = features.shape[1]
    assert n_features == 36, f"Expected 36 features, got {n_features}"

    bonafide_mask = labels == 0
    spoof_mask = labels == 1

    bonafide_features = features[bonafide_mask]
    spoof_features = features[spoof_mask]

    print(f"\nSample sizes: bonafide={bonafide_mask.sum()}, spoof={spoof_mask.sum()}")
    print(f"Class ratio: 1:{spoof_mask.sum() / bonafide_mask.sum():.1f}")
    print(f"\nRunning {n_features} univariate tests...\n")

    results = {}
    p_values = []

    for i in range(n_features):
        feat_name = FEATURE_NAMES[i]
        bonafide_vals = bonafide_features[:, i]
        spoof_vals = spoof_features[:, i]

        # Remove NaN/Inf
        bonafide_valid = bonafide_vals[np.isfinite(bonafide_vals)]
        spoof_valid = spoof_vals[np.isfinite(spoof_vals)]

        if len(bonafide_valid) < 10 or len(spoof_valid) < 10:
            print(f"  WARNING: {feat_name} has too few valid samples, skipping")
            results[feat_name] = {"skip": True, "reason": "insufficient valid samples"}
            p_values.append(1.0)
            continue

        # --- Wilcoxon rank-sum (Mann-Whitney U) ---
        stat_u, p_wilcoxon = stats.mannwhitneyu(
            bonafide_valid, spoof_valid, alternative='two-sided'
        )

        # --- Effect sizes ---
        d_cohen = cohens_d_unpaired(bonafide_valid, spoof_valid)
        d_glass = glass_delta(bonafide_valid, spoof_valid)
        d_cliff = cliff_delta(bonafide_valid, spoof_valid)

        # --- AUC-ROC ---
        auc, direction = per_feature_auc(features, labels, i)

        # --- Descriptive statistics ---
        bonafide_mean = np.mean(bonafide_valid)
        bonafide_std = np.std(bonafide_valid, ddof=1)
        bonafide_median = np.median(bonafide_valid)
        spoof_mean = np.mean(spoof_valid)
        spoof_std = np.std(spoof_valid, ddof=1)
        spoof_median = np.median(spoof_valid)

        results[feat_name] = {
            "feature_index": i,
            "feature_group": [g for g, indices in FEATURE_GROUPS.items() if i in indices][0],
            "n_bonafide_valid": int(len(bonafide_valid)),
            "n_spoof_valid": int(len(spoof_valid)),
            "bonafide_mean": float(bonafide_mean),
            "bonafide_std": float(bonafide_std),
            "bonafide_median": float(bonafide_median),
            "spoof_mean": float(spoof_mean),
            "spoof_std": float(spoof_std),
            "spoof_median": float(spoof_median),
            "mann_whitney_U": float(stat_u),
            "p_value_raw": float(p_wilcoxon),
            "cohens_d": float(d_cohen),
            "cohens_d_interpretation": interpret_effect_size(d_cohen),
            "glass_delta": float(d_glass),
            "cliff_delta": float(d_cliff),
            "cliff_delta_interpretation": interpret_cliff_delta(d_cliff),
            "auc_roc": float(auc),
            "auc_direction": direction,
        }

        p_values.append(p_wilcoxon)

    # --- BH-FDR correction ---
    print("\nApplying Benjamini-Hochberg FDR correction (alpha=0.05)...")
    rejected, adjusted_p = benjamini_hochberg(p_values, alpha=ALPHA)

    for i, feat_name in enumerate(FEATURE_NAMES):
        if feat_name in results and "skip" not in results[feat_name]:
            results[feat_name]["p_value_bh_adjusted"] = float(adjusted_p[i])
            results[feat_name]["significant_after_fdr"] = bool(rejected[i])

    # --- Summary table ---
    print(f"\n{'Feature':<15} {'AUC':>6} {'Cohen d':>9} {'Cliff d':>9} {'p_adj':>10} {'Sig':>4} {'Direction'}")
    print("-" * 80)

    # Sort by AUC descending
    sorted_feats = sorted(
        [(k, v) for k, v in results.items() if "skip" not in v],
        key=lambda x: x[1]["auc_roc"],
        reverse=True
    )

    for feat_name, r in sorted_feats:
        sig_marker = "***" if r["significant_after_fdr"] else "ns"
        print(f"{feat_name:<15} {r['auc_roc']:>6.3f} {r['cohens_d']:>+9.3f} "
              f"{r['cliff_delta']:>+9.3f} {r['p_value_bh_adjusted']:>10.2e} {sig_marker:>4} "
              f"{r['auc_direction']}")

    # --- Per-attack breakdown ---
    if attack_labels is not None:
        print("\n\n--- Per-Attack-Type Breakdown (AUC per feature per attack) ---\n")
        unique_attacks = sorted(set(attack_labels) - {"bonafide"})

        attack_results = {}
        for attack in unique_attacks:
            attack_mask = attack_labels == attack
            bonafide_mask_local = attack_labels == "bonafide"

            # Create binary labels: bonafide=0, this_attack=1
            combined_mask = bonafide_mask_local | attack_mask
            combined_features = features[combined_mask]
            combined_labels = np.where(attack_labels[combined_mask] == "bonafide", 0, 1)

            attack_aucs = {}
            for j in range(n_features):
                auc_val, _ = per_feature_auc(combined_features, combined_labels, j)
                attack_aucs[FEATURE_NAMES[j]] = float(auc_val)

            attack_results[attack] = attack_aucs

        # Print attack-type AUC table (top features only)
        top_10_names = [name for name, _ in sorted_feats[:10]]

        header = f"{'Feature':<15}" + "".join(f" {a:>6}" for a in unique_attacks)
        print(header)
        print("-" * len(header))

        for feat_name in top_10_names:
            row = f"{feat_name:<15}"
            for attack in unique_attacks:
                auc_val = attack_results[attack][feat_name]
                row += f" {auc_val:>6.3f}"
            print(row)

        results["_per_attack_auc"] = attack_results

    # --- Group-level summary ---
    print("\n\n--- Feature Group Summary ---\n")
    for group_name, indices in FEATURE_GROUPS.items():
        group_aucs = [results[FEATURE_NAMES[i]]["auc_roc"] for i in indices
                       if FEATURE_NAMES[i] in results and "skip" not in results[FEATURE_NAMES[i]]]
        group_ds = [abs(results[FEATURE_NAMES[i]]["cohens_d"]) for i in indices
                     if FEATURE_NAMES[i] in results and "skip" not in results[FEATURE_NAMES[i]]]
        n_sig = sum(1 for i in indices
                    if FEATURE_NAMES[i] in results
                    and "skip" not in results[FEATURE_NAMES[i]]
                    and results[FEATURE_NAMES[i]].get("significant_after_fdr", False))

        print(f"  {group_name:<12}: mean_AUC={np.mean(group_aucs):.3f} "
              f"(range {np.min(group_aucs):.3f}-{np.max(group_aucs):.3f}), "
              f"mean_|d|={np.mean(group_ds):.3f}, "
              f"significant={n_sig}/{len(indices)}")

    return results


# ============================================================================
# PHASE 2: CROSS-REFERENCE WITH CODEC ROBUSTNESS
# ============================================================================

def phase2_discrimination_vs_robustness(phase1_results):
    """
    Cross-reference discriminative power (AUC from Phase 1) with codec
    robustness (ICC from prior analysis).

    Produces the 2x2 classification:
        - High AUC + High ICC = SWEET SPOT (discriminative AND robust)
        - High AUC + Low ICC  = FRAGILE (discriminative but codec-sensitive)
        - Low AUC  + High ICC = USELESS (robust but not discriminative)
        - Low AUC  + Low ICC  = WORST (neither discriminative nor robust)

    ICC thresholds per Koo & Li (2016) "A Guideline of Selecting and Reporting
    Intraclass Correlation Coefficients for Reliability Research" J Chiro Med 15(2):155-163:
        - ICC < 0.50: poor
        - 0.50 <= ICC < 0.75: moderate
        - 0.75 <= ICC < 0.90: good
        - ICC >= 0.90: excellent
    """
    print("\n" + "=" * 80)
    print("PHASE 2: DISCRIMINATIVE POWER vs CODEC ROBUSTNESS")
    print("=" * 80)

    # Load ICC results from prior analysis
    with open(ICC_RESULTS) as f:
        icc_data = json.load(f)

    # Map feature groups to their ICC values
    # The ICC analysis was done at the GROUP level (PNCC, LFCC, Formants, Prosody)
    group_icc = {}
    for group_name in FEATURE_GROUPS:
        if group_name in icc_data["results"]:
            group_icc[group_name] = icc_data["results"][group_name]["icc"]
        else:
            print(f"  WARNING: No ICC data for group {group_name}")
            group_icc[group_name] = None

    print(f"\nGroup ICC values (from prior codec robustness analysis):")
    for group_name, icc_val in group_icc.items():
        print(f"  {group_name}: ICC={icc_val:.4f}" if icc_val else f"  {group_name}: N/A")

    # Build per-feature cross-reference table
    AUC_THRESHOLD = 0.60  # feature must be this discriminative to matter
    ICC_THRESHOLD = 0.90  # "excellent" per Koo & Li (2016)

    cross_ref = {}
    categories = {"sweet_spot": [], "fragile": [], "useless": [], "worst": []}

    for feat_name, r in phase1_results.items():
        if feat_name.startswith("_") or "skip" in r:
            continue

        group = r["feature_group"]
        icc_val = group_icc.get(group)
        auc_val = r["auc_roc"]

        if icc_val is None:
            continue

        high_auc = auc_val >= AUC_THRESHOLD
        high_icc = icc_val >= ICC_THRESHOLD

        if high_auc and high_icc:
            category = "sweet_spot"
        elif high_auc and not high_icc:
            category = "fragile"
        elif not high_auc and high_icc:
            category = "useless"
        else:
            category = "worst"

        entry = {
            "feature": feat_name,
            "group": group,
            "auc": float(auc_val),
            "icc": float(icc_val),
            "cohens_d": float(r["cohens_d"]),
            "category": category,
            "composite_score": float(auc_val * icc_val),  # multiplicative
        }
        cross_ref[feat_name] = entry
        categories[category].append(feat_name)

    # Print 2x2 table
    print(f"\n2x2 Classification (AUC >= {AUC_THRESHOLD}, ICC >= {ICC_THRESHOLD}):")
    print(f"  SWEET SPOT (discriminative + robust): {len(categories['sweet_spot'])} features")
    for f in sorted(categories["sweet_spot"], key=lambda x: cross_ref[x]["auc"], reverse=True):
        e = cross_ref[f]
        print(f"    {f:<15} AUC={e['auc']:.3f}  ICC={e['icc']:.4f}  composite={e['composite_score']:.3f}")

    print(f"  FRAGILE (discriminative but codec-sensitive): {len(categories['fragile'])} features")
    for f in sorted(categories["fragile"], key=lambda x: cross_ref[x]["auc"], reverse=True):
        e = cross_ref[f]
        print(f"    {f:<15} AUC={e['auc']:.3f}  ICC={e['icc']:.4f}")

    print(f"  USELESS (robust but not discriminative): {len(categories['useless'])} features")
    for f in categories["useless"][:5]:
        e = cross_ref[f]
        print(f"    {f:<15} AUC={e['auc']:.3f}  ICC={e['icc']:.4f}")
    if len(categories["useless"]) > 5:
        print(f"    ... and {len(categories['useless']) - 5} more")

    print(f"  WORST (neither): {len(categories['worst'])} features")

    return cross_ref, categories


# ============================================================================
# PHASE 3: CODEC-CONDITIONED DISCRIMINATION (ASVspoof 2021)
# ============================================================================

def phase3_codec_conditioned_discrimination():
    """
    Using ASVspoof 2021 DF eval data (with codec labels):
    For each codec condition, compute per-feature AUC.

    This answers: "Does codec compression destroy the discriminative signal?"

    The 2021 data has 7 conditions: none, alaw, ulaw, g722, gsm, opus, pstn
    Each condition has 25,938 files.

    Baseline: "none" condition (no codec, clean audio)
    Test: each coded condition vs "none" — is the AUC maintained?
    """
    print("\n" + "=" * 80)
    print("PHASE 3: CODEC-CONDITIONED DISCRIMINATION (ASVspoof 2021)")
    print("=" * 80)

    # Load 2021 data
    features_2021 = np.load(FEATURES_DIR / "acoustic_v4_eval_2021.npy")
    labels_2021 = np.load(FEATURES_DIR / "labels_eval_2021.npy")
    codecs_2021 = np.load(FEATURES_DIR / "codecs_eval_2021.npy")

    print(f"\nASVspoof 2021 DF eval: {features_2021.shape[0]} files, {features_2021.shape[1]} features")

    # Check codec distribution
    unique_codecs, codec_counts = np.unique(codecs_2021, return_counts=True)
    for c, n in zip(unique_codecs, codec_counts):
        mask = codecs_2021 == c
        n_bonafide = (labels_2021[mask] == 0).sum()
        n_spoof = (labels_2021[mask] == 1).sum()
        print(f"  {c:>6}: {n:>6} files (bonafide={n_bonafide}, spoof={n_spoof})")

    # Per-codec, per-feature AUC
    codec_auc_results = {}

    for codec in unique_codecs:
        codec_mask = codecs_2021 == codec
        codec_features = features_2021[codec_mask]
        codec_labels = labels_2021[codec_mask]

        aucs = {}
        for j in range(36):
            auc_val, direction = per_feature_auc(codec_features, codec_labels, j)
            aucs[FEATURE_NAMES[j]] = {"auc": float(auc_val), "direction": direction}

        codec_auc_results[codec] = aucs

    # Print comparison table: feature x codec
    print(f"\n{'Feature':<15}", end="")
    for codec in sorted(unique_codecs):
        print(f" {codec:>7}", end="")
    print(f" {'delta':>7}")  # max degradation from "none"
    print("-" * (15 + 8 * (len(unique_codecs) + 1)))

    codec_degradation = {}

    for j in range(36):
        feat_name = FEATURE_NAMES[j]
        none_auc = codec_auc_results["none"][feat_name]["auc"] if "none" in codec_auc_results else 0.5

        print(f"{feat_name:<15}", end="")
        max_drop = 0.0
        worst_codec = "none"

        for codec in sorted(unique_codecs):
            auc_val = codec_auc_results[codec][feat_name]["auc"]
            print(f" {auc_val:>7.3f}", end="")

            if codec != "none":
                drop = none_auc - auc_val
                if drop > max_drop:
                    max_drop = drop
                    worst_codec = codec

        print(f" {max_drop:>+7.3f}")

        codec_degradation[feat_name] = {
            "none_auc": float(none_auc),
            "max_degradation": float(max_drop),
            "worst_codec": worst_codec,
            "maintains_discrimination": max_drop < 0.05,  # less than 5% AUC drop
        }

    # Summary: features that maintain discrimination across codecs
    print("\n--- Features maintaining discrimination across ALL codecs (AUC drop < 0.05) ---")
    maintained = [(k, v) for k, v in codec_degradation.items() if v["maintains_discrimination"] and v["none_auc"] >= MIN_AUC]
    maintained.sort(key=lambda x: x[1]["none_auc"], reverse=True)

    for feat_name, info in maintained:
        print(f"  {feat_name:<15} clean_AUC={info['none_auc']:.3f}  max_drop={info['max_degradation']:.3f}")

    return codec_auc_results, codec_degradation


# ============================================================================
# PHASE 4: EXPLAINABILITY
# ============================================================================

def phase4_explainability(phase1_results, cross_ref):
    """
    For the top features: provide acoustic/physical explanations for WHY
    they differ between real and fake audio.

    These explanations are grounded in the speech synthesis literature.
    Each explanation has an evidence grade.
    """
    print("\n" + "=" * 80)
    print("PHASE 4: EXPLAINABILITY — WHY DO THESE FEATURES DETECT DEEPFAKES?")
    print("=" * 80)

    # Acoustic rationale database — grounded in published findings
    # Each entry has a hypothesis with evidence pointer
    ACOUSTIC_RATIONALE = {
        "PNCC": {
            "general": (
                "PNCC uses gammatone filterbank (Patterson et al., 1992) that models "
                "human auditory processing. TTS/VC systems typically operate in mel-frequency "
                "domain, creating spectral artifacts in the gammatone-transformed space. "
                "The power normalization step amplifies differences in low-energy regions "
                "where synthesis artifacts concentrate."
            ),
            "evidence_grade": "SUPPORTED",
            "references": [
                "Kim & Stern (2016) IEEE/ACM TASLP 24(7):1315-1329",
                "Todisco et al. (2019) Interspeech pp. 1008-1012",
            ],
        },
        "LFCC": {
            "general": (
                "LFCC uses a linear-spaced filterbank, preserving higher-frequency detail "
                "that mel-spacing compresses. TTS vocoders (e.g., WaveRNN, WaveGlow) "
                "introduce high-frequency noise patterns that LFCC captures but MFCC misses. "
                "However, LFCC coefficients are less stable under telephony codecs (G.711) "
                "due to their sensitivity to high-frequency quantization noise."
            ),
            "evidence_grade": "SUPPORTED",
            "references": [
                "Sahidullah et al. (2015) Computer Speech & Language 32(1):1-18",
                "Todisco et al. (2017) Interspeech pp. 2267-2271",
            ],
        },
        "Formants": {
            "general": (
                "Formant frequencies (F1, F2) and their bandwidths reflect the vocal tract "
                "resonance structure. Current TTS systems model formant centers reasonably "
                "well but struggle with: (1) natural formant bandwidth variation, (2) "
                "speaker-dependent formant dynamics, (3) coarticulation effects. Formant "
                "BANDWIDTH features may be especially discriminative because bandwidth "
                "depends on source-filter coupling that most neural vocoders approximate poorly."
            ),
            "evidence_grade": "HYPOTHESIS",
            "references": [
                "Fant (1960) 'Acoustic Theory of Speech Production'",
                "Wester et al. (2016) Interspeech pp. 1573-1577",
            ],
        },
        "Prosody": {
            "general": (
                "Prosodic features (F0 contour, speech rate, pause patterns) capture "
                "suprasegmental information. TTS systems often produce: (1) unnaturally "
                "smooth F0 contours lacking micro-prosodic perturbations (jitter), (2) "
                "regular speech rate without the natural variation of spontaneous speech, "
                "(3) systematically different pause patterns from read speech corpora. "
                "F0_std specifically captures the dynamic range of pitch, which TTS tends "
                "to compress."
            ),
            "evidence_grade": "SUPPORTED",
            "references": [
                "Hirst (2005) 'Automatic Analysis of Prosody for Multi-Lingual Speech Corpora'",
                "Wester et al. (2016) Interspeech pp. 1573-1577",
                "Muller et al. (2022) ICASSP pp. 6602-6606",
            ],
        },
    }

    # For each top feature, provide explanation
    top_features = sorted(
        [(k, v) for k, v in cross_ref.items()],
        key=lambda x: x[1]["composite_score"],
        reverse=True
    )[:15]

    explanations = {}

    for feat_name, entry in top_features:
        group = entry["group"]
        rationale = ACOUSTIC_RATIONALE.get(group, {})

        # Determine direction of effect
        phase1_entry = phase1_results.get(feat_name, {})
        direction = phase1_entry.get("auc_direction", "unknown")
        bonafide_mean = phase1_entry.get("bonafide_mean", 0)
        spoof_mean = phase1_entry.get("spoof_mean", 0)

        if direction == "higher_in_spoof":
            direction_text = f"Spoof audio has HIGHER {feat_name} (mean={spoof_mean:.4f}) than bonafide (mean={bonafide_mean:.4f})"
        elif direction == "higher_in_bonafide":
            direction_text = f"Bonafide audio has HIGHER {feat_name} (mean={bonafide_mean:.4f}) than spoof (mean={spoof_mean:.4f})"
        else:
            direction_text = "Direction unclear"

        # Feature-specific explanation
        specific_explanations = {
            "F0_std": (
                "F0 standard deviation measures pitch variability. Natural speech has "
                "substantial F0 variation driven by linguistic stress, emotion, and "
                "micro-prosody. TTS systems trained on read speech corpora produce "
                "smoother F0 contours. If F0_std is LOWER in spoof: confirms TTS "
                "produces unnaturally stable pitch. If HIGHER: some TTS systems may "
                "overcompensate with exaggerated intonation."
            ),
            "F0_mean": (
                "Mean fundamental frequency. Differences may indicate speaker-level "
                "mismatches rather than a universal deepfake artifact. Should be "
                "interpreted with caution due to speaker confound."
            ),
            "F0_max": (
                "Maximum F0 captures pitch excursions. Natural speech has occasional "
                "high-F0 events (emphasis, surprise) that TTS may undergenerate."
            ),
            "speech_rate": (
                "Syllable rate of speech. TTS systems trained on audiobook corpora "
                "tend to produce more uniform speech rates than spontaneous speech."
            ),
            "pause_ratio": (
                "Proportion of silence in the utterance. TTS pause insertion is "
                "typically rule-based and may differ from natural pause patterns."
            ),
            "F1_bw": (
                "First formant bandwidth. Controlled by vocal tract damping and "
                "source-filter coupling. Neural vocoders may produce narrower bandwidths "
                "(sharper resonances) because they optimize for spectral envelope accuracy "
                "without modeling the physical damping mechanisms."
            ),
            "F2_bw": (
                "Second formant bandwidth. F2 bandwidth is particularly affected by "
                "nasalization and coarticulation — both challenging for TTS."
            ),
            "energy": (
                "RMS energy. May reflect gain normalization differences between "
                "real recordings and synthesized audio."
            ),
        }

        explanation = {
            "feature": feat_name,
            "group": group,
            "auc": entry["auc"],
            "icc": entry["icc"],
            "composite_score": entry["composite_score"],
            "direction": direction_text,
            "group_rationale": rationale.get("general", "No rationale available"),
            "feature_specific": specific_explanations.get(feat_name, "See group-level rationale."),
            "evidence_grade": rationale.get("evidence_grade", "HYPOTHESIS"),
            "references": rationale.get("references", []),
        }

        explanations[feat_name] = explanation

        print(f"\n--- {feat_name} (AUC={entry['auc']:.3f}, ICC={entry['icc']:.4f}, "
              f"composite={entry['composite_score']:.3f}) ---")
        print(f"  Direction: {direction_text}")
        print(f"  Evidence grade: {explanation['evidence_grade']}")
        print(f"  Rationale: {explanation['feature_specific'][:200]}...")

    return explanations


# ============================================================================
# PHASE 5: FINAL RANKED TABLE
# ============================================================================

def phase5_final_table(phase1_results, cross_ref, codec_degradation, explanations):
    """
    The deliverable: ranked list of features by composite score
    (discriminative_power x codec_robustness), with acoustic explanation.
    """
    print("\n" + "=" * 80)
    print("PHASE 5: FINAL RANKED TABLE — CODEC-ROBUST DISCRIMINATIVE FEATURES")
    print("=" * 80)

    # Build final table
    final_table = []

    for feat_name, entry in cross_ref.items():
        phase1 = phase1_results.get(feat_name, {})
        codec_info = codec_degradation.get(feat_name, {})
        expl = explanations.get(feat_name, {})

        row = {
            "rank": 0,  # filled later
            "feature": feat_name,
            "group": entry["group"],
            "auc_2019_train": entry["auc"],
            "icc_codec": entry["icc"],
            "composite_score": entry["composite_score"],
            "cohens_d": entry["cohens_d"],
            "auc_2021_clean": codec_info.get("none_auc", None),
            "max_codec_degradation": codec_info.get("max_degradation", None),
            "worst_codec": codec_info.get("worst_codec", None),
            "maintains_across_codecs": codec_info.get("maintains_discrimination", None),
            "significant_fdr": phase1.get("significant_after_fdr", None),
            "direction": phase1.get("auc_direction", None),
            "evidence_grade": expl.get("evidence_grade", "UNASSESSED"),
            "acoustic_explanation": expl.get("feature_specific", "See group-level analysis"),
        }
        final_table.append(row)

    # Sort by composite score
    final_table.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, row in enumerate(final_table):
        row["rank"] = i + 1

    # Print the table
    print(f"\n{'Rank':>4} {'Feature':<15} {'AUC_train':>9} {'AUC_2021':>8} {'ICC':>6} "
          f"{'Comp':>6} {'|d|':>6} {'Codec_drop':>10} {'Sig':>4} {'Grade'}")
    print("-" * 100)

    for row in final_table:
        auc_2021_str = f"{row['auc_2021_clean']:.3f}" if row["auc_2021_clean"] is not None else "N/A"
        codec_drop_str = f"{row['max_codec_degradation']:+.3f}" if row["max_codec_degradation"] is not None else "N/A"
        sig_str = "***" if row["significant_fdr"] else "ns"

        print(f"{row['rank']:>4} {row['feature']:<15} {row['auc_2019_train']:>9.3f} "
              f"{auc_2021_str:>8} {row['icc_codec']:>6.3f} {row['composite_score']:>6.3f} "
              f"{abs(row['cohens_d']):>6.3f} {codec_drop_str:>10} {sig_str:>4} "
              f"{row['evidence_grade']}")

    # Identify the answer to the research question
    print("\n\n" + "=" * 80)
    print("ANSWER TO RESEARCH QUESTION:")
    print("Which features identify deepfake audio regardless of codec compression?")
    print("=" * 80)

    # Sweet spot features with 2021 cross-validation
    answer_features = [
        row for row in final_table
        if row["composite_score"] >= 0.55
        and row.get("maintains_across_codecs", False)
    ]

    if not answer_features:
        # Relax criteria
        answer_features = [
            row for row in final_table
            if row["composite_score"] >= 0.50
        ][:10]
        print("\n(Note: relaxed criteria — no features met strict maintains_across_codecs threshold)")

    print(f"\nTop {len(answer_features)} codec-robust discriminative features:\n")
    for row in answer_features:
        print(f"  {row['rank']}. {row['feature']} (group: {row['group']})")
        print(f"     AUC={row['auc_2019_train']:.3f}, ICC={row['icc_codec']:.4f}, "
              f"|Cohen's d|={abs(row['cohens_d']):.3f}")
        print(f"     Why it works: {row['acoustic_explanation'][:150]}")
        print()

    return final_table


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_visualizations(features, labels, phase1_results, cross_ref,
                            codec_auc_results, final_table):
    """
    Generate publication-quality figures:
    1. Distribution plots: bonafide vs spoof for top features
    2. AUC vs ICC scatter plot (the 2x2 quadrant chart)
    3. Codec-conditioned AUC heatmap
    4. Per-attack-type AUC heatmap
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("WARNING: matplotlib not available, skipping visualizations")
        return

    os.makedirs(OUTPUT_DIR / "figures", exist_ok=True)

    bonafide_mask = labels == 0
    spoof_mask = labels == 1

    # --- Figure 1: Distribution plots for top 12 features ---
    top_12 = [row["feature"] for row in final_table[:12]]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("Bonafide vs Spoof Feature Distributions (ASVspoof 2019 LA Train)",
                 fontsize=14, fontweight='bold')

    for idx, feat_name in enumerate(top_12):
        ax = axes[idx // 4, idx % 4]
        feat_idx = FEATURE_NAMES.index(feat_name)

        bonafide_vals = features[bonafide_mask, feat_idx]
        spoof_vals = features[spoof_mask, feat_idx]

        # Remove outliers for visualization (keep 1st-99th percentile)
        combined = np.concatenate([bonafide_vals, spoof_vals])
        combined_valid = combined[np.isfinite(combined)]
        if len(combined_valid) > 0:
            p1, p99 = np.percentile(combined_valid, [1, 99])
        else:
            p1, p99 = 0, 1

        bins = np.linspace(p1, p99, 60)

        ax.hist(bonafide_vals, bins=bins, alpha=0.6, density=True,
                color='#2196F3', label='Bonafide', edgecolor='none')
        ax.hist(spoof_vals, bins=bins, alpha=0.6, density=True,
                color='#F44336', label='Spoof', edgecolor='none')

        r = phase1_results.get(feat_name, {})
        auc_val = r.get("auc_roc", 0)
        d_val = r.get("cohens_d", 0)

        ax.set_title(f"{feat_name}\nAUC={auc_val:.3f}, d={d_val:+.2f}", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("Density" if idx % 4 == 0 else "")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "distributions_top12.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'figures' / 'distributions_top12.png'}")

    # --- Figure 2: AUC vs ICC scatter (2x2 quadrant) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    auc_vals = [v["auc"] for v in cross_ref.values()]
    icc_vals = [v["icc"] for v in cross_ref.values()]
    names = list(cross_ref.keys())
    groups = [v["group"] for v in cross_ref.values()]

    group_colors = {"PNCC": "#2196F3", "LFCC": "#4CAF50", "Formants": "#FF9800", "Prosody": "#9C27B0"}

    for i, (name, auc_v, icc_v, group) in enumerate(zip(names, auc_vals, icc_vals, groups)):
        color = group_colors.get(group, "gray")
        ax.scatter(icc_v, auc_v, c=color, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (icc_v, auc_v), fontsize=6, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    # Quadrant lines
    ax.axhline(y=0.60, color='gray', linestyle='--', alpha=0.5, label='AUC=0.60')
    ax.axvline(x=0.90, color='gray', linestyle='--', alpha=0.5, label='ICC=0.90')

    # Quadrant labels
    ax.text(0.95, 0.95, "SWEET SPOT\n(discriminative + robust)",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color='green', fontweight='bold', alpha=0.7)
    ax.text(0.15, 0.95, "FRAGILE\n(discriminative\nbut codec-sensitive)",
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            color='orange', fontweight='bold', alpha=0.7)
    ax.text(0.95, 0.05, "USELESS\n(robust but not\ndiscriminative)",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            color='gray', fontweight='bold', alpha=0.7)

    # Legend for groups
    for group, color in group_colors.items():
        ax.scatter([], [], c=color, s=60, label=group, edgecolors='black', linewidth=0.5)
    ax.legend(loc='lower left', fontsize=8)

    ax.set_xlabel("Codec Robustness (ICC)", fontsize=12)
    ax.set_ylabel("Discriminative Power (AUC-ROC)", fontsize=12)
    ax.set_title("Feature Selection: Discriminative Power vs Codec Robustness", fontsize=13)
    ax.set_xlim(0.85, 1.01)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "auc_vs_icc_scatter.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'figures' / 'auc_vs_icc_scatter.png'}")

    # --- Figure 3: Codec-conditioned AUC heatmap ---
    if codec_auc_results:
        codecs = sorted(codec_auc_results.keys())
        n_feats = len(FEATURE_NAMES)

        auc_matrix = np.zeros((n_feats, len(codecs)))
        for j, codec in enumerate(codecs):
            for i, feat_name in enumerate(FEATURE_NAMES):
                auc_matrix[i, j] = codec_auc_results[codec][feat_name]["auc"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 14))
        im = ax.imshow(auc_matrix, aspect='auto', cmap='RdYlGn', vmin=0.45, vmax=0.85)

        ax.set_xticks(range(len(codecs)))
        ax.set_xticklabels(codecs, rotation=45, ha='right')
        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
        ax.set_title("Per-Feature AUC-ROC Across Codec Conditions (ASVspoof 2021)", fontsize=12)

        # Add text annotations
        for i in range(n_feats):
            for j in range(len(codecs)):
                val = auc_matrix[i, j]
                color = 'white' if val < 0.55 or val > 0.75 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color=color)

        plt.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.6)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "figures" / "codec_auc_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / 'figures' / 'codec_auc_heatmap.png'}")

    # --- Figure 4: Feature group bar chart ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Sort features by AUC
    sorted_idx = sorted(range(36), key=lambda i: phase1_results.get(FEATURE_NAMES[i], {}).get("auc_roc", 0.5), reverse=True)

    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_aucs = [phase1_results.get(FEATURE_NAMES[i], {}).get("auc_roc", 0.5) for i in sorted_idx]
    sorted_groups = []
    for i in sorted_idx:
        for g, indices in FEATURE_GROUPS.items():
            if i in indices:
                sorted_groups.append(g)
                break

    colors = [group_colors.get(g, "gray") for g in sorted_groups]
    bars = ax.bar(range(36), sorted_aucs, color=colors, edgecolor='black', linewidth=0.3)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
    ax.axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label='Threshold (AUC=0.6)')

    ax.set_xticks(range(36))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=7)
    ax.set_ylabel("AUC-ROC (univariate)", fontsize=11)
    ax.set_title("Per-Feature Discriminative Power: Bonafide vs Spoof (ASVspoof 2019 LA Train)", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "feature_auc_barplot.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'figures' / 'feature_auc_barplot.png'}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("DISCRIMINATIVE FEATURE ANALYSIS FOR DEEPFAKE AUDIO DETECTION")
    print(f"Date: {datetime.datetime.now().isoformat()}")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / "figures", exist_ok=True)

    # ---- Load data ----
    print("\nLoading data...")

    # ASVspoof 2019 LA train (primary analysis set)
    features_train = np.load(FEATURES_DIR / "acoustic_v4_train_2019_fixed.npy")
    labels_train = np.load(FEATURES_DIR / "labels_train_2019.npy")

    print(f"  Train: {features_train.shape} features, {labels_train.shape} labels")
    print(f"    Bonafide: {(labels_train == 0).sum()}, Spoof: {(labels_train == 1).sum()}")

    # Check for NaN/Inf
    n_nan = np.isnan(features_train).sum()
    n_inf = np.isinf(features_train).sum()
    print(f"    NaN values: {n_nan}, Inf values: {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print("    WARNING: Replacing NaN/Inf with column medians...")
        for col in range(features_train.shape[1]):
            col_data = features_train[:, col]
            mask = np.isfinite(col_data)
            if mask.sum() > 0:
                median_val = np.median(col_data[mask])
                features_train[~mask, col] = median_val

    # Load attack-type labels
    attack_labels = load_attack_labels_train()
    print(f"  Attack types: {dict(zip(*np.unique(attack_labels, return_counts=True)))}")

    # ---- Phase 1 ----
    phase1_results = phase1_discriminative_analysis(features_train, labels_train, attack_labels)

    # Save Phase 1 results
    phase1_save = {k: v for k, v in phase1_results.items() if not k.startswith("_")}
    with open(OUTPUT_DIR / "phase1_discriminative_results.json", 'w') as f:
        json.dump(phase1_save, f, indent=2, default=str)
    print(f"\nPhase 1 results saved to: {OUTPUT_DIR / 'phase1_discriminative_results.json'}")

    # Save per-attack results separately
    if "_per_attack_auc" in phase1_results:
        with open(OUTPUT_DIR / "phase1_per_attack_auc.json", 'w') as f:
            json.dump(phase1_results["_per_attack_auc"], f, indent=2)

    # ---- Phase 2 ----
    cross_ref, categories = phase2_discrimination_vs_robustness(phase1_results)

    with open(OUTPUT_DIR / "phase2_cross_reference.json", 'w') as f:
        json.dump({"cross_ref": cross_ref, "categories": categories}, f, indent=2)
    print(f"\nPhase 2 results saved to: {OUTPUT_DIR / 'phase2_cross_reference.json'}")

    # ---- Phase 3 ----
    codec_auc_results, codec_degradation = phase3_codec_conditioned_discrimination()

    with open(OUTPUT_DIR / "phase3_codec_conditioned.json", 'w') as f:
        json.dump({
            "codec_auc": codec_auc_results,
            "degradation": codec_degradation,
        }, f, indent=2)
    print(f"\nPhase 3 results saved to: {OUTPUT_DIR / 'phase3_codec_conditioned.json'}")

    # ---- Phase 4 ----
    explanations = phase4_explainability(phase1_results, cross_ref)

    with open(OUTPUT_DIR / "phase4_explanations.json", 'w') as f:
        json.dump(explanations, f, indent=2)
    print(f"\nPhase 4 results saved to: {OUTPUT_DIR / 'phase4_explanations.json'}")

    # ---- Phase 5 ----
    final_table = phase5_final_table(phase1_results, cross_ref, codec_degradation, explanations)

    with open(OUTPUT_DIR / "phase5_final_table.json", 'w') as f:
        json.dump(final_table, f, indent=2, default=str)
    print(f"\nPhase 5 results saved to: {OUTPUT_DIR / 'phase5_final_table.json'}")

    # ---- Visualizations ----
    print("\n\nGenerating visualizations...")
    generate_visualizations(
        features_train, labels_train, phase1_results, cross_ref,
        codec_auc_results, final_table
    )

    # ---- Replicate on eval set for robustness check ----
    print("\n\n" + "=" * 80)
    print("ROBUSTNESS CHECK: Replicating Phase 1 on ASVspoof 2019 LA Eval")
    print("=" * 80)

    features_eval = np.load(FEATURES_DIR / "acoustic_v4_eval_2019_fixed.npy")
    labels_eval = np.load(FEATURES_DIR / "labels_eval_2019.npy")
    print(f"  Eval: {features_eval.shape} features")
    print(f"    Bonafide: {(labels_eval == 0).sum()}, Spoof: {(labels_eval == 1).sum()}")

    # Handle NaN/Inf
    for col in range(features_eval.shape[1]):
        col_data = features_eval[:, col]
        mask = np.isfinite(col_data)
        if not mask.all() and mask.sum() > 0:
            features_eval[~mask, col] = np.median(col_data[mask])

    eval_results = phase1_discriminative_analysis(features_eval, labels_eval, attack_labels=None)

    with open(OUTPUT_DIR / "robustness_check_eval2019.json", 'w') as f:
        json.dump({k: v for k, v in eval_results.items() if not k.startswith("_")}, f, indent=2, default=str)

    # Compare train vs eval AUCs
    print("\n\n--- Train vs Eval AUC Comparison (stability check) ---\n")
    print(f"{'Feature':<15} {'AUC_train':>9} {'AUC_eval':>9} {'diff':>7}")
    print("-" * 45)

    for feat_name in FEATURE_NAMES:
        auc_train = phase1_results.get(feat_name, {}).get("auc_roc", 0.5)
        auc_eval = eval_results.get(feat_name, {}).get("auc_roc", 0.5)
        diff = auc_eval - auc_train
        marker = " !!!" if abs(diff) > 0.1 else ""
        print(f"{feat_name:<15} {auc_train:>9.3f} {auc_eval:>9.3f} {diff:>+7.3f}{marker}")

    # ---- Final summary ----
    print("\n\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files generated:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUTPUT_DIR)}")

    print(f"\nTotal features analyzed: 36")
    n_sig = sum(1 for r in phase1_results.values()
                if isinstance(r, dict) and r.get("significant_after_fdr", False))
    print(f"Significant after FDR correction: {n_sig}/36")
    print(f"Sweet spot features (discriminative + codec-robust): {len(categories.get('sweet_spot', []))}")

    return final_table


if __name__ == "__main__":
    final_table = main()
