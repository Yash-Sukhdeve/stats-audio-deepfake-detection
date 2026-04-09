# Statistics for Audio Deepfake Detection

**Which acoustic features identify deepfakes regardless of codec compression?**

Presented at the 6th Mathematics Conference and Competition of Northern New York (MCCNNY 2026).

---

## The Finding

Out of 36 acoustic features tested, **only 4 reliably detect deepfakes after audio compression**. All four measure the first formant (F1) — a vocal tract resonance frequency:

| Feature | Real Speech | Fake Speech | Gap | Effect Size | Detection Accuracy |
|---------|-------------|-------------|-----|-------------|-------------------|
| F1 bandwidth | 515 Hz | 361 Hz | 154 Hz | d = 1.32 (large) | AUC = 0.851 |
| F1 mean | 666 Hz | 553 Hz | 113 Hz | d = 1.48 (large) | AUC = 0.855 |
| F1/F2 ratio | 0.342 | 0.292 | 0.050 | d = 1.29 (large) | AUC = 0.834 |
| F1 std dev | 272 Hz | 226 Hz | 46 Hz | d = 1.13 (large) | AUC = 0.820 |

**Why:** Real speech has wide formant resonances because physical vocal tracts have damping (tissue absorption, nasal coupling). Neural vocoders trained with L1/L2 losses produce unnaturally narrow, sharp resonances.

**Why codecs preserve this:** Codecs must preserve formants for speech intelligibility. Formant values change by only 6% under compression, while the real-vs-fake gap is 154 Hz — the signal dwarfs the noise.

---

## The Problem We Solved

Audio deepfake detectors trained on clean audio fail when deployed on telephone calls or VoIP:

- **Lab performance:** 7% error rate (good)
- **After codec compression:** 50% error rate (coin flip — useless)
- **Root cause:** The detector learned features that codecs destroy

We identified which features survive compression AND detect deepfakes, then validated this with 5 classifiers across 7 codec conditions.

---

## Repository Structure

```
statistical_analysis/              # Analysis scripts
  discriminative_feature_analysis.py   # Real vs fake comparison (36 features)
  run_ml_classifier_analysis.py        # 5 classifiers, fixed ASVspoof splits
  statistical_validation_codec_robustness.py  # Codec robustness validation
  comprehensive_acoustic_feature_analysis.py  # ICC + correlation analysis

feature_extraction/                # Feature extraction code
  extract_acoustic_features_v4.py      # 36-dim codec-robust features (PNCC, LFCC, Formants, Prosody)

results/
  discriminative_analysis/             # All results as JSON
    feature_by_feature_analysis.json       # 36 features × 6 statistics each
    phase1_discriminative_results.json     # Real vs fake comparison
    phase2_cross_reference.json            # AUC × codec robustness
    phase3_codec_conditioned.json          # Per-codec detection accuracy
    ml_classifier_results.json             # 5 classifiers × 4 evaluation conditions
    ml_classifier_report.md                # Human-readable report
    figures/                               # All visualizations (9 figures)
  comprehensive_analysis/              # ICC analysis results
  domain_shift_real_results.json       # MMD domain shift (2019 vs 2021)

presentation/
  MCCNNY_2026_FINAL_v6.pptx          # 11 slides with speaker notes (8 min)
  MCCNNY_2026_SPEAKING_SCRIPT.md     # Word-for-word speaking script

theory/                            # PAC-Bayesian generalization bounds
  FORMAL_PROOFS_CORRECTED.md           # Theorem 6 (feature-dependent domain adaptation)
  sample_complexity_validation.py      # Bound computation

codec_robustness/                  # Codec compression validation
  CODEC_ROBUSTNESS_STATISTICAL_PROTOCOL.md  # Study design
  validate_codec_robustness_v3_comprehensive.py  # 150 files × 9 codecs

docs/                              # Documentation and teaching materials
  deep_dives/                          # 5 mathematical deep-dive documents
  teaching/                            # 5-section learning guide
```

---

## Method: 6 Statistics Per Feature

For each of the 36 features, we computed:

| # | Statistic | Question It Answers | Tool |
|---|-----------|-------------------|------|
| 1 | **Mean values** | What are the typical values for real vs fake? | NumPy |
| 2 | **Mann-Whitney U test** | Is the difference statistically significant? | SciPy (with BH-FDR correction for 36 tests) |
| 3 | **Cohen's d** | How big is the difference in standard deviations? | Manual (pooled SD) |
| 4 | **AUC-ROC** | Can this feature alone classify real vs fake? | scikit-learn |
| 5 | **% change under codecs** | How much does compression change this feature? | 150 files × 9 codecs via ffmpeg |
| 6 | **AUC on compressed data** | Does detection still work after compression? | ASVspoof 2021 LA eval (181,566 files) |

---

## Key Results

### Detection with Simple Classifiers (Formants + Prosody, 15 features)

| Model | Lab (known attacks) | Unseen attacks | After codecs |
|-------|-------------------|----------------|-------------|
| Logistic Regression | 14.6% EER | 17.2% | 24.5% |
| SVM (RBF) | 11.4% | 17.2% | 21.5% |
| MLP (64→32) | 10.6% | 17.3% | 22.3% |
| Decision Tree | 16.6% | 21.8% | 22.9% |
| Random Forest | 14.2% | 18.5% | 21.2% |

Trained on ASVspoof 2019 LA train (25,380 files). Evaluated on fixed challenge splits. Balanced class weights for 9:1 spoof-to-bonafide ratio.

### Codec Robustness (% feature change under compression)

| Feature Group | Avg Change | Worst Change | Verdict |
|---------------|-----------|-------------|---------|
| Prosody | 3% | 8% | Very stable |
| Formants | 6% | 18% | Stable |
| PNCC | 6% | 22% | Moderate |
| LFCC | 39% | 120% | Fragile |

### The Collapse vs Graceful Degradation

| Feature Set | Lab EER | After Codecs EER | What Happened |
|-------------|---------|-----------------|---------------|
| All 36 features | 7.4% | **50.7%** | Collapsed to random |
| Formants only (4D) | 17.3% | **23.4%** | Graceful degradation |
| Formants + Prosody (15D) | 10.6% | **22.3%** | Graceful degradation |

---

## Dataset

| Split | Files | Attacks | Codecs | Purpose |
|-------|-------|---------|--------|---------|
| ASVspoof 2019 Train | 25,380 | A01-A06 (6 known) | None (clean) | Training |
| ASVspoof 2019 Dev | 24,844 | A01-A06 | None | Validation |
| ASVspoof 2019 Eval | 71,237 | A07-A19 (13 unseen) | None | Unseen attack test |
| ASVspoof 2021 LA Eval | 181,566 | A07-A19 + codecs | 7 conditions | Codec robustness test |

Audio: 16 kHz, 16-bit, mono FLAC. Class ratio: ~9:1 spoof to bonafide.

---

## The 36 Features

### PNCC (13 dimensions) — Power-Normalized Cepstral Coefficients
Spectral envelope via gammatone filterbank (models human auditory perception). Kim & Stern (2016).

### LFCC (8 dimensions) — Linear Frequency Cepstral Coefficients
Spectral detail with equal frequency resolution. Captures high-frequency vocoder artifacts. Sahidullah et al. (2015).

### Formants (8 dimensions) — Vocal Tract Resonances
F1 mean, F1 std, F1 bandwidth, F2 mean, F2 std, F2 bandwidth, F1/F2 ratio, formant dispersion. Extracted via Praat (Boersma & Weenink).

### Prosody (7 dimensions) — Speech Rhythm and Intonation
F0 mean, F0 std, F0 max, voiced ratio, energy mean, energy std, duration. F0 via pYIN (Mauch & Dixon, 2014).

---

## Reproducing the Results

```bash
# Setup
git clone https://github.com/Yash-Sukhdeve/stats-audio-deepfake-detection.git
pip install numpy scipy scikit-learn matplotlib parselmouth librosa pingouin

# Run the feature-by-feature analysis (requires ASVspoof data)
python statistical_analysis/discriminative_feature_analysis.py

# Run the ML classifier comparison
python statistical_analysis/run_ml_classifier_analysis.py
```

Requires ASVspoof 2019 LA and 2021 LA datasets (available at https://www.asvspoof.org with registration).

---

## Limitations

1. **21-23% EER is not state-of-the-art.** SSL-based systems (XLS-R, WavLM) achieve ~2% EER. Our features are interpretable baselines, not replacements for neural systems.

2. **Fails on advanced vocoders.** Attacks A17-A19 (waveform-level voice conversion, HiFi-GAN) produce realistic formants and defeat our features (EER > 45%).

3. **Evaluated on ASVspoof 2021 LA (181K files), not the full DF subset (611K files, 100+ attacks).** The DF subset is a harder test.

4. **Codec robustness was measured on 150 files × 9 codecs.** More files and more codecs (especially Opus, EVS) would strengthen the validation.

---

## Citations

- Kim, C. & Stern, R.M. (2016). Power-Normalized Cepstral Coefficients (PNCC). *IEEE/ACM TASLP*, 24(7), 1315-1329.
- Sahidullah, M. et al. (2015). A comparison of features for synthetic speech detection. *INTERSPEECH*.
- Shrout, P.E. & Fleiss, J.L. (1979). Intraclass correlations. *Psychological Bulletin*, 86(2), 420-428.
- Mauch, M. & Dixon, S. (2014). pYIN: A fundamental frequency estimator. *IEEE ICASSP*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Lawrence Erlbaum.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
- Ben-David, S. et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1-2), 151-175.

---

## License

Research use only. ASVspoof datasets are subject to their own license terms.
