# Hypothesis Generation Log

**Date:** 2026-04-06
**Generator:** Tree-of-Thought (5 theoretical frameworks x 5 feature groups)
**Total candidates generated:** 24
**Top 3 selected:** H13-X, H15-X, H16-X

## Generation Process

### Theoretical Frameworks Applied
1. **Signal processing** -- vocoder architecture constraints (frame-based synthesis, bandwidth limits, quantization)
2. **Articulatory phonetics** -- physical speech production (coarticulation, laryngeal-supralaryngeal coupling)
3. **Statistical learning** -- distributional differences (regression to mean, compressed output distributions)
4. **Psychoacoustics** -- auditory perception modeling (gammatone filterbank, ERB scale)
5. **Information theory** -- redundancy and compression (codec preservation of information types)

### Per-Feature-Group Generation

**PNCC (5 hypotheses):**
- H1: Spectral smoothness (signal processing) -- Score: 4.25
- H2: Power-law artifact amplification (psychoacoustics) -- Score: 3.60
- H3: Higher-order coefficient discriminability (signal processing) -- Score: 3.70
- H4: Temporal asymmetry disruption (articulatory phonetics) -- Score: 2.45
- H5: Noise floor quantization artifacts (signal processing) -- Score: 2.70

**LFCC (4 hypotheses):**
- H4-LFCC: High-frequency energy reduction (signal processing) -- Score: 4.25
- H5-LFCC: Frame-boundary spectral ripple (signal processing) -- Score: 2.70
- H6-LFCC: HNR via spectral tilt (signal processing + phonetics) -- Score: 3.70
- H7-LFCC: VC vs TTS distributional distance (statistical learning) -- Score: 4.25

**Formants (5 hypotheses):**
- H6-F: Smooth transitions / coarticulation (articulatory phonetics) -- Score: 3.70
- H7-F: Narrow bandwidth (signal processing + phonetics) -- Score: 3.95
- H8-F: Compressed vowel space (statistical learning) -- Score: 3.70
- H9-F: Formant dispersion artifacts (articulatory phonetics) -- Score: 3.15
- H10-F: Attack-type dependence (statistical learning) -- Score: 4.25

**Prosody (5 hypotheses):**
- H9-P: F0 std reduction (statistical learning) -- Score: 4.25
- H10-P: Elevated voiced ratio (signal processing) -- Score: 3.70
- H11-P: Uniform energy contour (statistical learning) -- Score: 3.50
- H12-P: F0 max truncation (signal processing) -- Score: 3.70
- H13-P: Duration variance (statistical learning) -- Score: 2.30

**Cross-feature (5 hypotheses):**
- H12-X: F0 x formant bandwidth interaction (articulatory phonetics) -- Score: 3.95
- H13-X: Codec-prosody importance crossover (information theory) -- Score: 4.65
- H14-X: PNCC-LFCC complementarity (psychoacoustics) -- Score: 3.70
- H15-X: Attack-specific signatures for MoE (statistical learning) -- Score: 4.75
- H16-X: Formant-prosody decorrelation (articulatory phonetics) -- Score: 3.85

### Selection Criteria

Top 3 selected based on:
1. Direct relevance to paper contributions (ACAGAT architecture motivation)
2. Novelty (prioritize findings not in existing literature)
3. Testability with available data (ASVspoof 2019 LA + 2021 DF)
4. Expected impact on the narrative (codec robustness story)

### Rejected Alternatives and Reasoning

- H4 (temporal asymmetry): Requires per-frame PNCC, not extractable from our 36-dim mean features
- H5 (noise floor): Requires access to PNCC internals, not in the standard feature vector
- H13-P (duration variance): Small expected effect, ASVspoof uses controlled text prompts
- H5-LFCC (frame boundary ripple): Requires per-frame LFCC analysis
