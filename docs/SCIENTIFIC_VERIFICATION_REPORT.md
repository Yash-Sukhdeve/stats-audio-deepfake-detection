# Scientific Verification Report: Acoustic Feature Analysis
**Date:** 2026-04-03
**Reviewed by:** 9 parallel research agents (2 swarms)
**Scope:** Standalone acoustic feature statistical analysis only (no model evaluation)

---

## OVERALL ASSESSMENT

| Metric | Score |
|--------|-------|
| Scientific Rigor | **4.5 / 10** |
| Reproducibility | **4 / 10** |
| Publication Readiness | **NOT READY** |

---

## CRITICAL ISSUES (Must Fix Before Any Publication)

### C1. Gammatone Filterbank Double Error
**File:** `extract_acoustic_features_v4.py:106,111-112`
- `b = 1.019 * 2 * np.pi * erb` computed but **never used**
- Line 111 uses `erb` instead of `b` as denominator (filter 6.4x too wide)
- Line 112 uses exponent `(-2)` instead of `(-4)` (2nd-order, not 4th-order gammatone)
- **Impact:** PNCC features (13/36 dims = 36% of vector) computed with incorrect filterbank
- **Reference:** Patterson et al. (1992); Kim & Stern (2016) specify 4th-order

### C2. PNCC Mean-Power Normalization on Wrong Axis
**File:** `extract_acoustic_features_v4.py:180-181`
- `np.mean(gamma_clean, axis=0)` normalizes per-frame across filters
- Should be `axis=1` (per-channel across time) per Kim & Stern (2016) Section IV-C
- **Impact:** Removes PNCC's intended noise robustness properties

### C3. Correlation Computed Within-Sample, Not Across-Samples
**File:** `comprehensive_acoustic_feature_analysis.py:955-957`
- Pearson r computed between original and coded 13-dim vectors PER SAMPLE
- Should compute r across all samples PER DIMENSION
- **Impact:** Answers wrong question; codec could shift all values by constant and still get r=1.0

### C4. ICC Never Computed Despite Being Protocol's Primary Metric
**Files:** All three validation scripts
- Protocol specifies ICC(2,1) >= 0.75 as PRIMARY criterion (Shrout & Fleiss, 1979)
- All `icc` fields are 0.0 (dataclass default, never populated)
- All feature classifications use Pearson r >= 0.90 (secondary metric) instead
- No `pingouin` or manual ICC implementation exists

### C5. Domain Shift Analysis Uses Fabricated Synthetic Data
**File:** `measure_domain_shift.py:297-348`
- `load_features()` generates Gaussian random data with hardcoded parameters
- Conclusion "formants show less domain shift than prosody" is artifact of construction
- Comment says "In production, load from actual files" - no production path exists

### C6. Codec Validation Imports v3, Not Current v4
**File:** `validate_codec_robustness_v3_comprehensive.py:61`
- `from extract_acoustic_features_v3 import extract_features_v3`
- v4 bug fixes (C1, C2) not present in v3
- All codec robustness results are for the wrong feature extractor

### C7. PNCC Implementation Differs Between Analysis and Production
- `comprehensive_acoustic_feature_analysis.py`: uses `librosa.filters.mel()` + log after power-law
- `extract_acoustic_features_v4.py`: uses custom (buggy) gammatone, no log
- Kim & Stern (2016): specifies gammatone 4th-order, power-law only, no log
- **Impact:** Codec robustness conclusions about PNCC based on different implementation

### C8. False Citation: "25-40% codec improvement"
**File:** `extract_acoustic_features_v4.py:129`
- Claims "PNCC provides 25-40% improvement over MFCC under codec compression"
- Project's own validation found ZERO codec references in Kim & Stern (2016)
- The 25-40% figure refers to additive noise, not codecs
- **Impact:** Scientific integrity violation if submitted

### C9. Wavelet r=1.0 Is a Missing-Dependency Artifact
**File:** `comprehensive_acoustic_feature_analysis.py:538,959-960`
- When `pywt` unavailable: `extract_wavelet()` returns `np.zeros(16)`
- `np.allclose(zeros, zeros)` = True -> forced `pearson_r = 1.0`
- Classified as "HIGHLY_RECOMMENDED" based on fabricated value
- Bit-identical features before/after lossy compression is physically impossible

---

## MAJOR ISSUES (Fix Before Submission)

| # | File:Line | Issue |
|---|-----------|-------|
| M1 | `statistical_validation_codec_robustness.py:407-409` | Normality gate only checks Shapiro-Wilk (absent for n>5000), ignores KS/D'Agostino |
| M2 | `statistical_validation_codec_robustness.py:27-28` | wilcoxon, mannwhitneyu, friedmanchisquare imported but NEVER called |
| M3 | `statistical_validation_codec_robustness.py:31` | `multipletests` imported but NEVER called; Bonferroni computed but not applied |
| M4 | `statistical_validation_codec_robustness.py:487` | Cohen's d uses independent-samples formula on paired data (should be d_z) |
| M5 | `statistical_validation_codec_robustness.py:369` | CIs without Fisher z-transformation; can exceed [-1,1] for r near 1 |
| M6 | `comprehensive_acoustic_feature_analysis.py:111,119,120` | `cohens_d`, `wilcoxon_p`, `icc` fields all published as 0.0 in JSON |
| M7 | `comprehensive_acoustic_feature_analysis.py:432` | Formant bandwidth = 2*std (not actual Praat bandwidth used in v4) |
| M8 | `compute_correlation.py:20-53` | Operates on legacy 64/72-dim features, not current 36-dim v4 |
| M9 | `measure_domain_shift.py:119` | MMD permutation test uses only 20 permutations (min p=0.05) |
| M10 | `measure_domain_shift.py:403` | Bootstrap CIs use only 10 samples |
| M11 | `validate_codec_robustness_v3_comprehensive.py:505` | Flattened Pearson r inflates stability (mixes file and feature variance) |
| M12 | `extract_acoustic_features_v4.py:384` | F0 range fmin=50 but docs say 75 Hz |
| M13 | Various | Three conflicting sample sizes: protocol n=200, report n=150, validation n=100 |
| M14 | Various | Feature dimensionality inconsistent: 32D/34D/36D/64D/74D across documents |
| M15 | All docs | Missing foundational LFCC citation: Sahidullah et al. (INTERSPEECH 2015) |
| M16 | `extract_acoustic_features_v4.py` | LFCC uses 20 filters (ASVspoof standard is 70 filters) |

---

## WHAT IS SCIENTIFICALLY SOLID

1. **Feature exclusion decisions** - MODGD (min_r=0.448) and Glottal (min_r=-0.180) exclusions are well-justified with primary peer-reviewed evidence (Narendra & Alku 2017; Yegnanarayana & Murthy 1992)
2. **Statistical test implementations** - Shapiro-Wilk, KS, D'Agostino, Anderson-Darling, Levene, Bartlett, Fligner-Killeen all correctly implemented (just not properly gated)
3. **Protocol design** - `CODEC_ROBUSTNESS_STATISTICAL_PROTOCOL.md` is excellent; specifies ICC(2,1), Fisher z, Wilcoxon, FDR correctly
4. **Self-correction practice** - Team identified and documented citation errors in `PNCC_CODEC_ROBUSTNESS_VALIDATION.md` and `FINAL_FEATURE_SELECTION.md`
5. **PNCC power-law parameter** (1/15) correctly cited from Kim & Stern (2016) Section II-G
6. **pYIN F0 extraction** correctly cited from Mauch & Dixon (2014)
7. **Praat/Burg LPC** for formants is field standard (Boersma & Weenink)
8. **Documentation quality** - Feature extraction report and protocol are well-structured

---

## VERIFIED CITATIONS FOR NEURIPS 2026

| Citation | DOI | Use For |
|----------|-----|---------|
| Kim & Stern (2016) IEEE/ACM TASLP 24(7) | 10.1109/TASLP.2016.2545928 | PNCC definition (noise robustness only, NOT codec) |
| Glasberg & Moore (1990) Hearing Research 47 | 10.1016/0378-5955(90)90170-T | ERB formula for gammatone |
| Sahidullah et al. (2015) INTERSPEECH | 10.21437/Interspeech.2015-462 | LFCC for anti-spoofing (**MUST ADD**) |
| Fant (1960) Acoustic Theory | N/A (book) | Formant theory |
| Künzel (2001) Forensic Linguistics 8(1) | N/A | F1 codec warning (6-7% shifts) |
| Siegert & Niebuhr (2023) Frontiers Comm. | 10.3389/fcomm.2023.972182 | F0 codec robustness |
| Mauch & Dixon (2014) ICASSP | 10.1109/ICASSP.2014.6853678 | pYIN F0 extraction |
| Narendra & Alku (2017) INTERSPEECH | 10.21437/Interspeech.2017-1280 | Glottal exclusion justification |
| Yegnanarayana & Murthy (1992) IEEE TSP | 10.1109/78.157229 | MODGD reference |
| Wang et al. (2023) IEEE/ACM TASLP 31 | 10.1109/TASLP.2023.3270077 | ASVspoof 2021 challenge |
| Shrout & Fleiss (1979) Psych. Bulletin | 10.1037/0033-2909.86.2.420 | ICC methodology |

---

## PRIORITY FIX ORDER

### Before any result can be cited:
1. Fix gammatone filterbank: use `b` not `erb`, exponent `-4` not `-2` (C1)
2. Fix PNCC normalization axis: `axis=1` not `axis=0` (C2)
3. Re-extract ALL v4 features after C1+C2 fixes
4. Replace synthetic data in domain shift with real features (C5)
5. Update codec validation to import v4, not v3 (C6)
6. Remove false "25-40% codec improvement" claim (C8)

### Before paper submission:
7. Implement ICC(2,1) as protocol requires (C4)
8. Fix correlation to across-samples per-dimension (C3)
9. Fix normality gate to check all available tests (M1)
10. Execute Wilcoxon tests and apply BH-FDR correction (M2, M3)
11. Fix Cohen's d to paired formula d_z (M4)
12. Apply Fisher z-transformation for CIs (M5)
13. Resolve Wavelet r=1.0 artifact (C9)
14. Add Sahidullah et al. (2015) citation (M15)
15. Reconcile sample sizes and feature dimensionalities (M13, M14)
16. Update correlation analysis to 36-dim v4 features (M8)
17. Increase permutation counts to 1000+ (M9)

---

## AGENT CONTRIBUTIONS

| Agent | Focus | Key Findings |
|-------|-------|-------------|
| Senior Research Scientist | File-by-file review | Gammatone bug (C1), wrong correlation method (C3), PNCC implementation mismatch (C7) |
| Data Analyst (Swarm 1) | Codec robustness stats | Normality gate bug, all non-parametric tests dead imports, ICC/Cohen's d/Wilcoxon = 0.0 |
| Data Analyst (Swarm 2) | Line-by-line audit | 1 PASS out of 20 checks; Wavelet root cause confirmed |
| Research Evidence Validator | PAC-Bayes theory | KL=10 off by 12 orders of magnitude; two-term formula non-standard |
| Citation Validator | Feature justifications | False 25-40% claim; missing Sahidullah 2015; MFCC more codec-robust than PNCC |
| Code Reviewer (Swarm 1) | EER/t-DCF/MMD code | Domain shift fabricated; t-DCF formula wrong in 3 scripts |
| Code Reviewer (Swarm 2) | Feature extraction bugs | Gammatone double error; PNCC axis wrong; v3 import; LFCC 20 vs 70 filters |
| QA Reviewer | Reproducibility | 4/10 score; 3 conflicting sample sizes; 5 dim inconsistencies |
