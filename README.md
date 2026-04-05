# Statistics for Audio Deepfake Detection

Statistical analysis framework for evaluating acoustic feature robustness under codec compression, with PAC-Bayesian generalization bounds for principled feature selection.

**Presented at:** 6th Mathematics Conference and Competition of Northern New York (MCCNNY 2026)

## Overview

This project investigates how acoustic features used in audio deepfake detection behave under real-world codec compression (G.711, G.722, Opus, MP3, AAC, etc.). We develop:

1. **Feature-Dependent Domain Adaptation Bound (Theorem 6)** - A novel decomposition showing that domain shift can be bounded per feature type, enabling principled feature selection
2. **PAC-Bayesian Feature Selection** - Non-vacuous generalization bound (2.58% slack) for discrete feature subset selection over 2^K subsets
3. **Systematic Codec Robustness Validation** - ICC(2,1)-based reliability analysis with Wilcoxon tests, BH-FDR correction, and Fisher z-transformed confidence intervals

## Feature Set (36-dimensional, v4)

| Group | Dims | Description |
|-------|------|-------------|
| PNCC | 13 | Power-normalized cepstral coefficients (gammatone filterbank, Kim & Stern 2016) |
| LFCC | 8 | Linear frequency cepstral coefficients (Sahidullah et al. 2015) |
| Formants | 8 | F1/F2 mean, std, bandwidth via Praat (Boersma & Weenink) |
| Prosody | 7 | F0 mean/std/max, voiced ratio, energy, duration (pYIN, Mauch & Dixon 2014) |

## Repository Structure

```
statistical_analysis/        # Statistical validation scripts (ICC, Wilcoxon, BH-FDR)
feature_extraction/          # v4 acoustic feature extraction (36-dim)
theory/                      # PAC-Bayesian proofs (Theorems 2, 6) and bounds
codec_robustness/            # Codec robustness validation protocol and scripts
correlation_analysis/        # Feature correlation analysis and redundancy detection
domain_shift/                # MMD-based domain shift measurement (2019 vs 2021)
presentation/                # MCCNNY 2026 slides and abstract
docs/                        # Feature validation reports and methodology
results/                     # Analysis outputs and figures
```

## Key Results

### Theoretical
- **Theorem 6**: Feature-dependent domain adaptation bound with explicit factor of 2
- **PAC-Bayes bound**: 2.58% slack (non-vacuous) for K=36 features, m=25,380 training samples
- **Unified objective**: argmin L_hat + lambda_1 * KL(Q||P) + lambda_2 * d_T(D_s, D_t)

### Empirical
- **Correlation analysis**: Only 2 redundant feature pairs (r>0.90) in 36-dim set; mean |r|=0.16
- **Feature exclusions**: MODGD (min_r=0.448, sign-inversion under G.722), Glottal (min_r=-0.180)
- **Codec robustness**: Evaluated across 9 codecs with ICC(2,1), Wilcoxon tests, BH-FDR correction

## Statistical Methods

| Method | Implementation | Reference |
|--------|---------------|-----------|
| ICC(2,1) | pingouin | Shrout & Fleiss (1979) |
| Wilcoxon signed-rank | scipy.stats | Paired feature-codec comparison |
| BH-FDR correction | statsmodels | Benjamini-Hochberg for 144 comparisons |
| Fisher z-transformation | Custom | CIs on Pearson r near 1.0 |
| Paired Cohen's d_z | Custom | Effect size for within-subject design |
| MMD with permutation test | Custom (1000 perms) | Gretton et al. (2012) |
| PAC-Bayes bound | Custom | McAllester (1999) |

## Requirements

```
numpy
scipy
scikit-learn
librosa
parselmouth (praat-parselmouth)
pingouin
statsmodels
pandas
matplotlib
```

## Dataset

Analysis is conducted on the ASVspoof 2019 LA and ASVspoof 2021 DF challenge datasets:
- Train: 25,380 files (ASVspoof 2019 LA)
- Dev: 24,844 files
- Eval: 71,237 files (2019) + 611,829 files (2021)

## Citations

- Kim, C. & Stern, R.M. (2016). Power-Normalized Cepstral Coefficients. IEEE/ACM TASLP 24(7). DOI: 10.1109/TASLP.2016.2545928
- Sahidullah, M. et al. (2015). A comparison of features for synthetic speech detection. INTERSPEECH. DOI: 10.21437/Interspeech.2015-462
- McAllester, D.A. (1999). Some PAC-Bayesian theorems. Machine Learning 37(3). DOI: 10.1023/A:1007618624809
- Ben-David, S. et al. (2010). A theory of learning from different domains. Machine Learning 79(1-2). DOI: 10.1007/s10994-009-5152-4
- Shrout, P.E. & Fleiss, J.L. (1979). Intraclass correlations. Psychological Bulletin 86(2). DOI: 10.1037/0033-2909.86.2.420
- Mauch, M. & Dixon, S. (2014). pYIN: A fundamental frequency estimator. IEEE ICASSP. DOI: 10.1109/ICASSP.2014.6853678

## License

Research use only. ASVspoof datasets are subject to their own license terms.
