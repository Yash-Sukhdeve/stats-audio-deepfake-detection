# Comprehensive Acoustic Feature Validation Report

## Audio Deepfake Detection - Evidence-Based Feature Selection

**Date**: December 3, 2025
**Target**: NeurIPS 2026
**Methodology**: Statistical codec robustness validation

---

## Executive Summary

This report documents rigorous statistical validation of **16 acoustic feature types** across **9 real-world codecs** using **150 audio samples** from ASVspoof 2019. The goal is to identify codec-robust features for cross-domain deepfake detection (ASVspoof 2019 → ASVspoof 2021).

### Key Findings

| Status | Count | Features |
|--------|-------|----------|
| **HIGHLY RECOMMENDED** | 3 | Wavelet, Instantaneous-Freq, MFCC |
| **RECOMMENDED** | 4 | Temporal, Spectral-Statistics, PNCC, LFCC |
| **ACCEPTABLE** | 3 | Prosody, Formants, Mel-Spectrogram |
| **NOT RECOMMENDED** | 6 | Spectral-Contrast, Glottal, Tonnetz, Chroma, MODGD, CQT |

### Critical Discovery

**MODGD (Modified Group Delay)** - previously included in our v2 feature set - shows **min_r = 0.448** under G.722 codec. This phase-based feature is **codec-sensitive** and should be **removed** from the pipeline.

---

## Methodology

### Test Configuration

```
Samples:     150 randomly selected from ASVspoof 2019 LA train
Features:    16 acoustic feature types
Codecs:      9 real-world codecs
Metrics:     Pearson correlation (original vs codec-transformed)
Threshold:   min_r >= 0.90 for "RECOMMENDED"
```

### Codecs Tested

| Category | Codec | Bitrate | Real-World Scenario |
|----------|-------|---------|---------------------|
| Telephony | G.711 A-law | 64 kbps | Landline (EU/Asia) |
| Telephony | G.711 μ-law | 64 kbps | Landline (US/Japan) |
| Telephony | G.722 | 64 kbps | HD Voice (VoIP) |
| Streaming | MP3 32k | 32 kbps | Low-quality podcasts |
| Streaming | MP3 64k | 64 kbps | Standard streaming |
| Streaming | MP3 128k | 128 kbps | High-quality audio |
| Streaming | AAC 32k | 32 kbps | Mobile streaming |
| Streaming | AAC 64k | 64 kbps | YouTube/TikTok |
| Streaming | AAC 128k | 128 kbps | High-quality streaming |

### Statistical Metrics

- **Pearson Correlation (r)**: Linear relationship between original and codec-transformed features
- **Threshold**: r >= 0.90 indicates codec-robust feature
- **Worst-case analysis**: Minimum r across all codecs determines suitability

---

## Detailed Results

### 1. HIGHLY RECOMMENDED (min_r >= 0.95)

These features maintain excellent stability across ALL tested codecs:

| Feature | Dims | Mean r | Min r | Worst Codec | Reference |
|---------|------|--------|-------|-------------|-----------|
| **Wavelet** | 16 | 1.000 | 1.000 | - | Mallat (1989) |
| **Instantaneous-Freq** | 8 | 0.9997 | 0.992 | G.711 A-law | Boashash (1992) |
| **MFCC** | 13 | 0.997 | 0.962 | G.711 A-law | Davis & Mermelstein (1980) |

**Wavelet features** show perfect robustness - the DWT decomposition captures multi-scale structure that survives lossy compression.

### 2. RECOMMENDED (0.90 <= min_r < 0.95)

These features are suitable for codec-robust systems:

| Feature | Dims | Mean r | Min r | Worst Codec | Reference |
|---------|------|--------|-------|-------------|-----------|
| **Temporal** | 8 | 0.997 | 0.933 | AAC 32k | Various |
| **Spectral-Statistics** | 8 | 0.996 | 0.926 | G.711 A-law | Peeters (2004) |
| **PNCC** | 13 | 0.992 | 0.932 | G.711 A-law | Kim & Stern (2016) |
| **LFCC** | 8 | 0.987 | 0.903 | G.711 A-law | ASVspoof 2021 baseline |

**PNCC validation**: Kim & Stern (2016) claimed 25-40% improvement under noise/compression. Our results **partially confirm** this - PNCC shows good robustness (min_r=0.932) but not exceptional.

### 3. ACCEPTABLE WITH CAUTION (0.80 <= min_r < 0.90)

Use these features with awareness of potential degradation:

| Feature | Dims | Mean r | Min r | Worst Codec | Note |
|---------|------|--------|-------|-------------|------|
| **Formants** | 8 | 0.997 | 0.866 | G.711 A-law | F1/F2 affected by narrowband |
| **Prosody** | 7 | 0.998 | 0.842 | AAC 32k | F0 tracking degrades at low bitrate |
| **Mel-Spectrogram** | 40 | 0.986 | 0.816 | G.711 μ-law | High-frequency bins lost |

### 4. NOT RECOMMENDED (min_r < 0.80)

**DO NOT USE** these features for codec-robust systems:

| Feature | Dims | Mean r | Min r | Worst Codec | Issue |
|---------|------|--------|-------|-------------|-------|
| **MODGD** | 12 | 0.988 | **0.448** | G.722 | Phase distortion |
| **CQT** | 40 | 0.996 | **0.496** | AAC 32k | Frequency resolution loss |
| **Chroma** | 12 | 0.953 | **0.288** | G.711 μ-law | Pitch class unstable |
| **Tonnetz** | 6 | 0.963 | **0.087** | AAC 32k | Tonal features collapse |
| **Glottal** | 8 | 0.979 | **-0.180** | G.711 A-law | Spectral tilt inverts |
| **Spectral-Contrast** | 7 | 0.741 | **-0.265** | MP3 32k | Completely unstable |

---

## Telephony Focus (ASVspoof 2021 Relevance)

ASVspoof 2021 uses telephony codecs (G.711, G.722). Focused analysis:

| Feature | G.711 A-law | G.711 μ-law | G.722 | Stable? |
|---------|-------------|-------------|-------|---------|
| Wavelet | 1.00 | 1.00 | 1.00 | ✅ |
| Instantaneous-Freq | 1.00 | 1.00 | 1.00 | ✅ |
| MFCC | 0.99 | 0.99 | 1.00 | ✅ |
| Prosody | 0.99 | 0.99 | 1.00 | ✅ |
| Temporal | 0.99 | 0.99 | 1.00 | ✅ |
| PNCC | 0.96 | 0.96 | 1.00 | ✅ |
| LFCC | 0.96 | 0.96 | 1.00 | ✅ |
| Spectral-Statistics | 0.98 | 0.98 | 1.00 | ✅ |
| Formants | 0.99 | 0.99 | 1.00 | ✅ |
| **MODGD** | 1.00 | 1.00 | **0.90** | ⚠️ |
| **Spectral-Contrast** | **0.38** | **0.40** | 0.90 | ❌ |

---

## Recommended Feature Vector (v4)

Based on this analysis, we recommend a **74-dimensional** codec-robust feature vector:

```
RECOMMENDED FEATURE VECTOR (74D)
================================

HIGHLY RECOMMENDED (37D):
├── Wavelet:            16D  [DWT db4 coefficients]
├── Instantaneous-Freq:  8D  [Phase-derived temporal features]
└── MFCC:               13D  [Standard cepstral features]

RECOMMENDED (37D):
├── Temporal:            8D  [ZCR, STE, onset statistics]
├── Spectral-Statistics: 8D  [Centroid, bandwidth, rolloff, flatness]
├── PNCC:               13D  [Power-normalized cepstral]
└── LFCC:                8D  [Linear frequency cepstral]

OPTIONAL - USE WITH CAUTION:
├── Prosody:             7D  [F0, energy, HNR - may degrade at low bitrate]
└── Formants:            8D  [F1/F2 - may degrade under narrowband]
```

### Comparison with Previous Versions

| Version | Dimensions | Features | Issues |
|---------|------------|----------|--------|
| **v2** | 64D | MFCC, MODGD, Formants, Prosody, Glottal | MODGD, Glottal unstable |
| **v3** | 36D | PNCC, LFCC, Formants, Prosody | PNCC_high issue (partial) |
| **v4 (NEW)** | 74D | Wavelet, IF, MFCC, Temporal, SS, PNCC, LFCC | All validated r>0.90 |

---

## Literature Validation

### Claims Confirmed ✅

1. **PNCC robustness** (Kim & Stern 2016): Partially confirmed. Mean r=0.992, min_r=0.932. Better than MFCC under noise, but not exceptional under telephony codecs.

2. **LFCC for anti-spoofing** (ASVspoof 2021): Confirmed. min_r=0.903 shows good stability.

### Claims NOT Confirmed ❌

1. **MODGD for deepfake detection**: Phase-based features are **codec-sensitive**. min_r=0.448 under G.722 is unacceptable.

2. **Glottal features**: Spectral tilt approximation **inverts** under G.711 (min_r=-0.180).

---

## Experimental Artifacts

All results saved to:
```
evidence/experiments/comprehensive_feature_analysis/
├── raw_results.csv              # Per-sample, per-codec correlations
├── analysis_summary.json        # Aggregated statistics
├── analysis_report.txt          # Text report
├── feature_ranking.png          # Bar chart with min/max
├── heatmap_features_codecs.png  # Feature × Codec matrix
├── category_analysis.png        # Category comparisons
└── telephony_focus.png          # ASVspoof 2021 relevant codecs
```

---

## Recommendations for ACAGAT

1. **Update Acoustic Anchor**: Replace 36D v3 features with 74D validated features
2. **Remove MODGD**: Phase-based features are codec-sensitive
3. **Add Wavelet Features**: Perfect codec robustness (r=1.00)
4. **Add Instantaneous Frequency**: Novel phase-derived temporal features (r=0.99)
5. **Keep PNCC/LFCC**: Validated codec-robust spectral features

---

## References

1. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE TASSP.

2. Kim, C., & Stern, R. M. (2016). Power-normalized cepstral coefficients (PNCC) for robust speech recognition. IEEE TASLP.

3. Sahidullah, M., & Saha, G. (2012). Design, analysis and experimental evaluation of block based transformation in MFCC computation for speaker recognition. Speech Communication.

4. Mallat, S. (1989). A theory for multiresolution signal decomposition: The wavelet representation. IEEE TPAMI.

5. Boashash, B. (1992). Estimating and interpreting the instantaneous frequency of a signal. Proceedings of the IEEE.

6. Peeters, G. (2004). A large set of audio features for sound description. IRCAM Technical Report.

---

*Report generated by comprehensive_acoustic_feature_analysis.py*
*150 samples × 16 features × 9 codecs = 21,600 statistical tests*
