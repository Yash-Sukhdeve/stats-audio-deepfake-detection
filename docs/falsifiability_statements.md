# Falsifiability Statements for All Hypotheses

**Date:** 2026-04-06
**Standard:** All hypotheses specify H0, H1, alpha, and concrete falsifying observations.

---

## Top 3 Priority Hypotheses

### H13-X: Codec-Prosody Importance Crossover

- **H0:** The ratio (prosody_importance / spectral_importance) is equal for clean (2019 LA) and codec-compressed (2021 DF) conditions.
- **H1:** The ratio (prosody_importance / spectral_importance) is significantly higher for codec-compressed conditions.
- **Alpha:** 0.05
- **Test:** Permutation importance from random forest trained on each condition separately. Bootstrap 95% CI on importance ratios.
- **Falsifying observations:** (a) Ratio does not increase under compression; (b) Spectral features remain dominant under compression; (c) Both feature groups degrade equally.

### H15-X: Attack-Specific Acoustic Signatures

- **H0:** A 6-class classifier (A01-A06) on 36 acoustic features achieves accuracy <= 16.7% (chance level for 6 classes).
- **H1:** Classification accuracy significantly exceeds 16.7%.
- **Alpha:** 0.05 (binomial test against chance)
- **Test:** Random forest with 5-fold cross-validation on ASVspoof 2019 LA eval set (spoofed samples only).
- **Falsifying observations:** (a) Accuracy at or below chance; (b) Confusion matrix shows no attack-specific clustering; (c) All attacks map to same region in feature space.

### H16-X: Formant-Prosody Decorrelation

- **H0:** |r(formant_features, prosody_features)| is equal for bona fide and spoofed speech.
- **H1:** |r(formant_features, prosody_features)| is significantly lower in spoofed speech.
- **Alpha:** 0.05
- **Test:** Compute canonical correlation between formant and prosody feature blocks for each condition. Fisher z-transformation to test difference.
- **Falsifying observations:** (a) No difference in correlation; (b) Higher correlation in synthetic speech; (c) Correlation structure is identical.

---

## All Hypotheses: Compact Falsifiability Table

| ID | H0 (Null) | Decision Rule | Falsifying Observation |
|----|-----------|---------------|----------------------|
| H1 | Var(PNCC, real) = Var(PNCC, fake) | Two-sample t-test, alpha=0.05 | p > 0.05 or Var(fake) > Var(real) |
| H2 | d(PNCC) <= d(LFCC) | Paired comparison of effect sizes | d(LFCC) >= d(PNCC) |
| H3 | AUC(coeff 7-12) <= AUC(coeff 0-6) | Wilcoxon signed-rank, alpha=0.05 | p > 0.05 or reversed direction |
| H4 | Skew(PNCC, real) = Skew(PNCC, fake) | Two-sample t-test, alpha=0.05 | p > 0.05 |
| H5 | Floor_ratio(real) = Floor_ratio(fake) | Two-sample t-test, alpha=0.05 | p > 0.05 |
| H4-LFCC | Importance(high-freq) = Importance(low-freq) | Permutation importance comparison | No difference |
| H5-LFCC | Var(LFCC_temporal, fake) <= Var(LFCC_temporal, real) | Two-sample t-test | p > 0.05 or reversed |
| H6-LFCC | LFCC1(real) = LFCC1(fake) | Two-sample t-test, alpha=0.05 | p > 0.05 |
| H7-LFCC | Mahal(VC, real) >= Mahal(TTS, real) | Comparison of distances | VC further from real |
| H6-F | f1_std(real) = f1_std(fake) | Welch's t-test, alpha=0.05 | p > 0.05 or f1_std(fake) > f1_std(real) |
| H7-F | f1_bw(real) = f1_bw(fake) | Welch's t-test, alpha=0.05 | p > 0.05 or f1_bw(fake) > f1_bw(real) |
| H8-F | Var(ratio, real) = Var(ratio, fake) | Levene's test, alpha=0.05 | p > 0.05 |
| H9-F | disp(real) = disp(fake) | Two-sample t-test, alpha=0.05 | p > 0.05 |
| H10-F | AUC_formant(TTS) = AUC_formant(VC) | Paired comparison | No difference |
| H9-P | f0_std(real) = f0_std(fake) | Welch's t-test, alpha=0.05 | p > 0.05 or f0_std(fake) > f0_std(real) |
| H10-P | voiced_ratio(real) = voiced_ratio(fake) | Welch's t-test, alpha=0.05 | p > 0.05 or voiced_ratio(fake) < voiced_ratio(real) |
| H11-P | energy_std(real) = energy_std(fake) | Welch's t-test, alpha=0.05 | p > 0.05 or energy_std(fake) > energy_std(real) |
| H12-P | f0_max(real) = f0_max(fake) | Welch's t-test, alpha=0.05 | p > 0.05 or f0_max(fake) > f0_max(real) |
| H13-P | Var(duration, real) = Var(duration, fake) | Levene's test, alpha=0.05 | p > 0.05 |
| H12-X | AUC(interaction) <= max(AUC(individual)) | DeLong test for AUC, alpha=0.05 | No improvement |
| H13-X | Importance_ratio(clean) = Importance_ratio(codec) | Bootstrap CI on ratio difference | Overlapping CIs |
| H14-X | r(PNCC_score, LFCC_score) >= 0.7 | Pearson correlation | r >= 0.7 |
| H15-X | Attack accuracy <= 16.7% | Binomial test, alpha=0.05 | p > 0.05 |
| H16-X | |r(formant, prosody)|_real = |r(formant, prosody)|_fake | Fisher z-test, alpha=0.05 | p > 0.05 |
