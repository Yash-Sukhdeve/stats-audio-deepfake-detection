# Experimental Methodology Document
## Feature-Dependent Domain Adaptation for Multi-Modal Learning (ICML 2026)

---

## Overview

This document describes the scientific methodology for validating two key theoretical claims from our paper:

1. **Assumption A1**: Conditional independence of SSL and acoustic features given labels
2. **Theorem 6**: Feature-dependent domain shift magnitudes

---

## Experiment 1: Validating Assumption A1

### Theoretical Background

**Claim**: `I(Z_ssl; Z_acoustic | Y) < ε_indep` where ε_indep ≈ 0.1 bits

This assumption states that SSL features (from XLS-R) and hand-crafted acoustic features are approximately conditionally independent given the class label Y.

### Method Selection

We use **three complementary approaches** for robustness:

#### 1. k-Nearest Neighbor Estimator (Primary Method)

**Reference**: Runge (2018) extending Kraskov et al. (2004)

**Why this method?**
- Non-parametric: No distributional assumptions
- Consistent: Proven convergence properties
- Handles high dimensions better than histogram-based methods
- Well-established in information theory literature

**Implementation details**:
- k = 5 neighbors (balance between bias and variance)
- Chebyshev distance for neighbor search
- Bias correction using digamma functions

**Formula**:
```
I(X;Y|Z) = ψ(k) + <ψ(n_z) - ψ(n_xz) - ψ(n_yz)>
```
where ψ is the digamma function and n_* are neighbor counts.

#### 2. PCA Dimensionality Reduction

**Why needed?**
- Original features are high-dimensional (1024D SSL, 64D acoustic)
- MI estimation suffers from curse of dimensionality
- PCA preserves most variance while reducing dimensions

**Implementation**:
- Reduce to 10 principal components
- Typically retains >90% variance
- Makes k-NN estimation more reliable

#### 3. Classifier-Based Validation

**Reference**: Póczos & Schneider (2012)

**Why this method?**
- Independent validation using different principles
- Intuitive interpretation via prediction accuracy
- Robust to different data distributions

**Formula**:
```
I(X;Y|Z) = H(Y|Z) - H(Y|X,Z)
```
where H is estimated from classifier error rates.

### Expected Results

- **Strong support**: I < 0.05 bits
- **Support**: 0.05 ≤ I < 0.1 bits
- **Weak support**: 0.1 ≤ I < 0.2 bits
- **No support**: I ≥ 0.2 bits

### Confidence Intervals

Bootstrap resampling (n=50-100) provides:
- Standard error estimates
- 95% confidence intervals
- Robustness check

---

## Experiment 2: Measuring Domain Shift

### Theoretical Background

**Claim**: `d_formants < d_prosody` where d_S = ||P_s(φ(X)[S]) - P_t(φ(X)[S])||_TV

Different acoustic features exhibit different robustness to domain shift (codec artifacts).

### Method Selection

#### 1. Maximum Mean Discrepancy (Primary Method)

**Reference**: Gretton et al. (2012) "A Kernel Two-Sample Test"

**Why MMD?**
- Works in high dimensions via kernel trick
- Has nice theoretical properties (RKHS embedding)
- Provides hypothesis testing framework
- Widely used in domain adaptation literature

**Implementation**:
- RBF kernel with median heuristic for bandwidth
- Unbiased quadratic-time estimator
- Permutation test for p-values

**Formula**:
```
MMD²(P,Q) = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
```

#### 2. Total Variation Distance (Secondary)

**Reference**: Tsybakov (2009)

**Why TV?**
- Direct interpretation as probability mass difference
- Theorem 6 specifically mentions TV distance
- Works well for low-dimensional projections

**Implementation**:
- Histogram approximation for 1D projections
- PCA for dimensionality reduction
- 50 bins for density estimation

#### 3. Wasserstein Distance (Validation)

**Reference**: Villani (2008) "Optimal Transport"

**Why Wasserstein?**
- Captures geometry of the space
- Robust to outliers
- Complementary perspective to MMD

### Feature Groups

Based on acoustic phonetics literature:

1. **Formants** (indices 0-2): F1, F2, F3
   - Resonant frequencies of vocal tract
   - Relatively stable across codecs
   - Important for vowel identification

2. **Prosody** (indices 10-12): F0, jitter, shimmer
   - Fundamental frequency and perturbations
   - Sensitive to codec compression
   - Important for speaker characteristics

3. **Spectral** (indices 20-30): Additional frequency features
4. **Temporal** (indices 30-40): Time-domain features

### Expected Results

We expect:
- `d_formants < d_prosody` (main hypothesis)
- Formants show 20-40% less shift than prosody
- p < 0.05 for statistical significance

---

## Statistical Rigor

### Reproducibility (Rule R5)
- Fixed random seed (42)
- Pinned package versions
- Deterministic algorithms where possible
- Full parameter documentation

### Multiple Testing Correction
- Not applied here as we have pre-specified hypotheses
- Report all p-values transparently

### Confidence Intervals
- Bootstrap (n=50-100 iterations)
- 95% confidence level
- Report both point estimates and intervals

### Effect Sizes
- Report absolute differences
- Compute ratios for relative comparison
- Cohen's d where applicable

---

## Interpretation Guide

### For Assumption A1 (validate_A1.py)

**Output format**:
```
I(Z_ssl; Z_acoustic | Y) = 0.043 bits
95% CI: [0.031, 0.055] bits
Status: SUPPORTED
```

**Interpretation**:
- Value in bits (1 bit = complete dependence)
- < 0.05 bits suggests features capture different information
- Validates multi-modal approach

### For Theorem 6 (measure_domain_shift.py)

**Output format**:
```
Domain shift (MMD):
  Formants: 0.124 ± 0.015
  Prosody:  0.287 ± 0.023
  Ratio: 0.43
Hypothesis: SUPPORTED
```

**Interpretation**:
- MMD values: 0 = identical distributions, larger = more shift
- Ratio < 1 supports d_formants < d_prosody
- Guides feature-specific adaptation strategies

---

## Limitations and Assumptions

1. **Synthetic data**: Examples use generated data matching expected properties
2. **Feature extraction**: Assumes pre-computed features are available
3. **Sample size**: Need sufficient samples (>1000) for reliable estimates
4. **Gaussianity**: Some methods assume approximate normality

---

## Running the Experiments

### Installation
```bash
pip install -r requirements.txt
```

### Execution
```bash
# Validate conditional independence
python validate_A1.py

# Measure domain shift
python measure_domain_shift.py
```

### Output Files
- `validation_A1_results.json`: Conditional MI estimates
- `domain_shift_results.json`: Domain shift measurements
- `domain_shift_visualization.png`: Distribution plots

---

## Scientific Citations

### Information Theory
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6).
- Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. AISTATS.

### Domain Adaptation
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. JMLR, 13(1).
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. Machine Learning, 79(1).

### Optimal Transport
- Villani, C. (2008). Optimal transport: old and new. Springer.
- Peyré, G., & Cuturi, M. (2019). Computational optimal transport. Foundations and Trends in Machine Learning, 11(5-6).

### Acoustic Phonetics
- Stevens, K. N. (2000). Acoustic phonetics. MIT Press.
- Quatieri, T. F. (2001). Discrete-time speech signal processing. Prentice Hall.

---

## Clarification: Sample Size and Feature Dimensionality

**Actual sample size used:** n = 150 audio files per codec test
(Protocol recommended n=200; analysis confirmed n=150 from raw_results.csv: 21,600 rows / 16 features / 9 codecs = 150)

**Current feature dimensionality:** 36 (v4 extraction pipeline)
- PNCC: 13 coefficients (power-normalized cepstral, gammatone filterbank)
- LFCC: 8 coefficients (linear frequency cepstral)
- Formants: 8 features (F1/F2 mean/std/bandwidth, F1F2 ratio, F2F1 diff)
- Prosody: 7 features (F0 mean/std/max, voiced ratio, energy mean/std, duration)

**Historical versions:** v1=214D (deprecated), v2=64D, v3=36D (buggy), v4=36D (current, bug-fixed)

---

## Contact

For questions about the methodology or implementation, please refer to the paper or contact the authors.

---

*Last updated: November 2024*