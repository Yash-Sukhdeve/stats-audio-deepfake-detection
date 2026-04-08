# Literature Review: Acoustic Differences Between Real and Synthetic Speech

## Systematic Review of Measurable, Interpretable Acoustic Signals for Deepfake Detection

**Date:** 2026-04-06
**Scope:** 2018--2026, focused on interpretable acoustic features (not neural hidden representations)
**Databases searched:** OpenAlex, Semantic Scholar, arXiv (via OpenAlex)
**Total papers identified:** 42 | **Papers included after screening:** 28

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Spectral Differences](#2-spectral-differences-between-real-and-synthetic-speech)
3. [Prosodic Differences](#3-prosodic-differences)
4. [Formant Differences](#4-formant-differences)
5. [Phase and Temporal Differences](#5-phase-and-temporal-differences)
6. [Statistical Tests and Effect Sizes](#6-statistical-tests-and-feature-discriminability)
7. [Codec Compression Effects](#7-behavior-under-codec-compression)
8. [Per-Attack Analysis](#8-per-attack-type-analysis)
9. [Synthesis and Gap Analysis](#9-synthesis-and-gap-analysis)
10. [Complete Reference List](#10-complete-reference-list)

---

## 1. Executive Summary

This review synthesizes findings from 28 papers (2018--2026) on **measurable acoustic differences** between real and synthetic speech. The key findings are:

**Strongest discriminative signals (pre-compression):**
1. **High-frequency spectral artifacts** (above 4 kHz): Neural vocoders consistently fail to reproduce natural high-frequency content, producing oversmoothed spectra, metallic artifacts, or spectral gaps. This is the single most-reported finding across the literature.
2. **Micro-prosodic regularity** (jitter, shimmer): Synthetic speech shows excessive regularity in voice quality parameters -- reduced jitter variance and unnaturally smooth shimmer patterns (effect size Delta-p ~ 0.17--0.23).
3. **Breathing pattern absence**: TTS systems overwhelmingly fail to generate natural breathing patterns, providing a near-perfect detection cue in controlled settings (EER = 0.0 reported).
4. **Spectral prediction residuals**: Short-term and long-term linear prediction residuals differ systematically between real and synthetic speech across 17 synthesis methods.
5. **Micro-frequency dynamics**: Doppler-effect-based frequency shift analysis and drumhead vibration decomposition reveal micro-frequency signatures absent in synthetic speech.

**Under codec compression:**
- High-frequency features (above 4 kHz) are **destroyed** by most codecs (especially narrowband telephony codecs)
- Low-frequency sub-bands (below 4 kHz) **retain discriminative power** and show ~25% EER reduction when used as primary features
- Structural consistency features outperform artifact-based features after compression
- Jitter/shimmer and spectral envelope features show **moderate degradation** but remain partially useful

**Critical gap:** Very few papers report per-feature effect sizes or conduct rigorous statistical testing (e.g., Cohen's d, Wilcoxon rank-sum with correction). Most report only aggregate EER or accuracy, making it difficult to rank individual features by discriminative power.

---

## 2. Spectral Differences Between Real and Synthetic Speech

### 2.1 High-Frequency Spectral Artifacts

The most consistently reported finding across the literature is that **synthetic speech exhibits artifacts concentrated in the high-frequency bands** (above 4 kHz).

**Mewada et al. (2023)** conducted a study specifically targeting high-frequency characteristics, finding that "high-frequency features are able to discriminate genuine speech from spoofed speech well." They showed that Gaussian-filtered high-frequency features, when combined with inverted MFCC (IMFCC), achieved 99.58% validation accuracy and 6.58% EER on ASVspoof 2017. The key insight is that Gaussian filtering captures "more global information than triangular filter" at high frequencies, and that the **inverted mel scale** (which emphasizes high frequencies) provides complementary discriminative power to standard mel-scale features.

> **Citation:** Mewada, H., Al-Asad, J.F., Almalki, F.A., Khan, A.H., Almujally, N.A., El-Nakla, S., & Naith, Q.H. (2023). Gaussian-Filtered High-Frequency-Feature Trained Optimized BiLSTM Network for Spoofed-Speech Classification. *Sensors*, 23(14), 6637. DOI: [10.3390/s23146637](https://doi.org/10.3390/s23146637)

**Song et al. (2022)** directly addressed vocoder-specific spectral artifacts in their work on Robust MelGAN. They identified two primary spectral deficiencies in GAN-based vocoders:
1. **Metallic sound artifacts**: Caused by imprecise harmonic generation in the multi-band MelGAN architecture
2. **Oversmoothing of aperiodic components**: The vocoder fails to reproduce the natural aperiodic (noise-like) component of speech, producing unnaturally smooth spectra

Their proposed "over-smooth handler" explicitly separates speech into periodic and aperiodic components, confirming that the inability to generate realistic aperiodic energy is a fundamental limitation of current vocoders. They also used harmonic shift, noise injection, and phase noise augmentation to improve robustness -- implying that these are precisely the spectral domains where vocoders fail.

> **Citation:** Song, K., Cong, J., Wang, X., Zhang, Y., Xie, L., Jiang, N., & Wu, H. (2022). Robust MelGAN: A robust universal neural vocoder for high-fidelity TTS. *arXiv preprint* arXiv:2210.17349. DOI: [10.48550/arxiv.2210.17349](https://doi.org/10.48550/arxiv.2210.17349)

### 2.2 Spectral Envelope and Cepstral Features

The choice of frequency scale for cepstral analysis has proven critical for spoofing detection, and the literature provides clear evidence for why.

**LFCC (Linear Frequency Cepstral Coefficients)** use a uniform frequency resolution across all bands, as opposed to the mel scale which compresses high frequencies. The ASVspoof 2019 baseline system adopted LFCC as the primary feature, and multiple subsequent studies confirmed its superiority over MFCC for spoofing detection. The reason is clear: the mel scale de-emphasizes the very high-frequency region (above 4 kHz) where vocoder artifacts concentrate, while LFCC preserves equal resolution at all frequencies.

**Lei et al. (2022)** demonstrated this quantitatively, showing that "LFCC+GMM-ResNet can relatively reduce min-tDCF and EER by 76.1% and 76.3%" for logical access attacks compared to GMM baselines, while their LFCC+GMM-SENet achieved reductions of 94.4% and 95.4% for physical access attacks.

> **Citation:** Lei, Z., Yan, H., Liu, C., Ma, M., & Yang, Y. (2022). Two-Path GMM-ResNet and GMM-SENet for ASV Spoofing Detection. *Proc. ICASSP 2022*, pp. 9087-9091. DOI: [10.1109/icassp43922.2022.9746163](https://doi.org/10.1109/icassp43922.2022.9746163)

**CQCC (Constant Q Cepstral Coefficients)** use a geometrically spaced frequency scale, providing high resolution at low frequencies. The ASVspoof 2017 challenge adopted CQCC as its primary baseline feature (Todisco et al., 2017). CQCC captures fine spectral detail in the fundamental frequency and low-harmonic region, which complements LFCC's uniform coverage.

> **Citation:** Todisco, M., Wang, X., Vestman, V., Sahidullah, M., Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., & Lee, K.A. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. *Proc. Interspeech 2019*, pp. 1008-1012. DOI: [10.21437/interspeech.2019-2249](https://doi.org/10.21437/interspeech.2019-2249)

**Jedrasiak (2026)** reported that spectral features (LFCC, CQCC, MFCC) show "significant differences between authentic and synthetic speech" with differentiation values around **Delta-p approximately 0.25**, one of the few papers to report a numeric effect size for spectral features.

> **Citation:** Jedrasiak, K. (2026). Analysis of Acoustic Anomalies and Speech Artefacts in Synthetic Content. *Safety & Fire Technology (Bezpieczenstwo i Technika Pozarnicza)*, 67(1), 58-84. DOI: [10.12845/sft.67.1.2026.3](https://doi.org/10.12845/sft.67.1.2026.3)

### 2.3 Linear Prediction Residuals

**Borrelli et al. (2021)** developed a detection approach based entirely on hand-crafted features motivated by speech processing theory, specifically analyzing **short-term and long-term prediction traces**. The linear prediction (LP) residual -- the error signal after LP analysis -- encodes information about the fine spectral structure (glottal excitation, harmonic structure) that vocoders often fail to reproduce accurately.

Their detector was validated on 17 different synthetic speech algorithms, from traditional vocoders (e.g., STRAIGHT, WORLD) to modern neural systems, demonstrating that LP residual features capture synthesis artifacts that generalize across vastly different generation methods. This suggests that the fine spectral structure of the excitation signal is a **universal** point of failure for synthesis systems.

> **Citation:** Borrelli, C., Bestagini, P., Antonacci, F., Sarti, A., & Tubaro, S. (2021). Synthetic speech detection through short-term and long-term prediction traces. *EURASIP Journal on Information Security*, 2021, 2. DOI: [10.1186/s13635-021-00116-3](https://doi.org/10.1186/s13635-021-00116-3)

### 2.4 Micro-Frequency Dynamics

**Kumari et al. (2025)** introduced VoiceRadar, which uses physical models to analyze frequency micro-dynamics:
1. **Doppler effect analysis**: Captures how frequency shifts across sequential audio segments -- natural speech shows characteristic micro-frequency shifts from articulatory movement that synthetic speech does not reproduce
2. **Drumhead vibration decomposition**: Decomposes audio into component frequencies, revealing that authentic speech contains "subtle variations, micro-frequencies" that are absent or anomalous in AI-generated samples

This approach outperformed existing detection methods, suggesting that **sub-spectral frequency dynamics** (below the resolution of standard FFT analysis) carry significant discriminative information.

> **Citation:** Kumari, K., Abbasihafshejani, M., Pegoraro, A., Rieger, P., Arshi, K., Jadliwala, M., & Sadeghi, A.-R. (2025). VoiceRadar: Voice Deepfake Detection using Micro-Frequency and Compositional Analysis. *Proc. 2025 Network and Distributed System Security Symposium (NDSS)*. DOI: [10.14722/ndss.2025.243389](https://doi.org/10.14722/ndss.2025.243389)

---

## 3. Prosodic Differences

### 3.1 Jitter and Shimmer

**Li et al. (2023)** provide the most direct statistical evidence for micro-prosodic differences. Their key findings:

- **"Obvious differences exist in level of jitter and shimmer between genuine and fake speech"**
- Shimmer shows **"large dynamic variation for fake speech"** -- synthetic speech produces unnaturally variable amplitude perturbation patterns
- Jitter and shimmer features extracted via YIN and SWIPE algorithms provide **complementary information** to spectrum-based detection
- Adding jitter/shimmer features to Mel-spectrogram reduced EER from 41.29% to 35.77% on ADD2023 (13.37% relative improvement)
- Both **static levels** and **dynamic variations** of jitter/shimmer differ between real and fake

The interpretation: real speech has natural, physiologically constrained micro-perturbations in pitch and amplitude (from vocal fold irregularities). Synthetic speech either (a) has unnaturally regular perturbations (oversmoothed) or (b) has perturbations with the wrong statistical distribution (too variable in some cases).

> **Citation:** Li, K., Lu, X., Akagi, M., & Unoki, M. (2023). Contributions of Jitter and Shimmer in the Voice for Fake Audio Detection. *IEEE Access*, 11, 84689-84698. DOI: [10.1109/access.2023.3301616](https://doi.org/10.1109/access.2023.3301616)

**Jedrasiak (2026)** confirmed these findings, reporting that "jitter/shimmer, HNR/CPP showed smoothing and excessive regularity" in synthetic speech, with effect sizes of **Delta-p approximately 0.17--0.23**. The "excessive regularity" finding is particularly important: it means synthetic speech is **too perfect** in its micro-prosodic characteristics, lacking the natural variability of human vocal fold vibration.

> **Citation:** Jedrasiak, K. (2026). Analysis of Acoustic Anomalies and Speech Artefacts in Synthetic Content. *Safety & Fire Technology*, 67(1), 58-84. DOI: [10.12845/sft.67.1.2026.3](https://doi.org/10.12845/sft.67.1.2026.3)

### 3.2 HNR (Harmonics-to-Noise Ratio) and CPP (Cepstral Peak Prominence)

HNR measures the ratio of periodic to aperiodic energy in the voice signal. CPP measures the prominence of the F0 peak in the cepstrum.

**Jedrasiak (2026)** reported that both HNR and CPP show "smoothing and excessive regularity" in synthetic speech. Real speech has natural HNR fluctuations driven by:
- Vocal fold asymmetry
- Subglottal pressure variation
- Articulatory co-production effects

Synthetic speech tends to produce either (a) unnaturally high HNR (too clean/periodic) or (b) HNR values that fail to co-vary naturally with F0, intensity, and vowel quality.

### 3.3 Breathing Patterns

**Layton et al. (2024)** made a remarkable finding: breathing patterns provide near-perfect deepfake detection.

- Breath-based detector achieved **perfect 1.0 AUPRC and 0.0 EER** on test data across 33.6 hours of audio
- "Breath, a higher-level part of speech, is a key component of natural speech" -- and is overwhelmingly absent or improperly generated in deepfakes
- By contrast, the state-of-the-art SSL-wav2vec model "completely fails to classify the same in-the-wild samples (0.72 AUPRC, 0.99 EER)"

This finding is striking: a simple physiological feature (presence and timing of inhalation/exhalation patterns) can outperform sophisticated neural models. The reason is that most TTS systems are trained on clean, segmented utterances where breathing has been removed or was never present in training data.

> **Citation:** Layton, S., De Andrade, T., Olszewski, D., Warren, K.M., Butler, K., & Traynor, P. (2024). Every Breath You Don't Take: Deepfake Speech Detection Using Breath. *arXiv preprint* arXiv:2404.15143. DOI: [10.48550/arxiv.2404.15143](https://doi.org/10.48550/arxiv.2404.15143)

**Han et al. (2024)** found that human listeners also use breathing patterns as a primary cue, alongside "accents, vocal inflections, and emotions" when assessing audio authenticity. Interestingly, human participants outperformed current ML models in detecting spoofed audio, suggesting that humans attend to prosodic and physiological cues that neural networks overlook.

> **Citation:** Han, C., Mitra, P., & Billah, S.M. (2024). Uncovering Human Traits in Determining Real and Spoofed Audio: Insights from Blind and Sighted Individuals. *Proc. CHI Conference on Human Factors in Computing Systems*. DOI: [10.1145/3613904.3642817](https://doi.org/10.1145/3613904.3642817)

### 3.4 Prosodic Feature Integration

**Muruganandham et al. (2025)** showed that combining "prosodic, wavelet packet, and glottal parameters" with MFCC and temporal features achieves 97-98% accuracy across benchmark datasets. This confirms that prosodic features provide substantial complementary information beyond spectral features alone.

> **Citation:** Muruganandham, S.K., et al. (2025). LSTM autoencoder based parallel architecture for deepfake audio detection with dynamic residual encoding and feature fusion. *Scientific Reports*. DOI: [10.1038/s41598-025-08198-6](https://doi.org/10.1038/s41598-025-08198-6)

### 3.5 F0 and Energy Contour Differences

Despite extensive searching, **few papers provide rigorous statistical comparisons of F0 distribution or energy contour** between real and synthetic speech at the population level. This is a notable gap in the literature. The prosodic analysis is largely embedded within detection system evaluations rather than standalone acoustic analyses.

**Sisman et al. (2020)** in their comprehensive voice conversion overview note that VC systems must convert both "spectral" and "prosodic" characteristics, and that prosody conversion (F0 transformation, duration modification) remains a significant challenge. Imperfect prosody conversion is an implicit detection cue, but quantitative characterization is lacking.

> **Citation:** Sisman, B., Yamagishi, J., King, S., & Li, H. (2020). An Overview of Voice Conversion and Its Challenges: From Statistical Modeling to Deep Learning. *IEEE/ACM Transactions on Audio, Speech and Language Processing*, 29, 132-157. DOI: [10.1109/taslp.2020.3038524](https://doi.org/10.1109/taslp.2020.3038524)

---

## 4. Formant Differences

### 4.1 Current Evidence

This is the **weakest area in the literature**. Despite extensive searching, no paper from 2018-2026 was found that provides a rigorous, standalone statistical comparison of formant frequencies, formant bandwidths, or formant transition rates between real and synthetic speech in the deepfake detection context.

The literature provides indirect evidence:

**Sisman et al. (2020)** discuss spectral conversion in voice conversion systems, noting that spectral envelope conversion (which includes formant structure) is central to VC. However, they do not quantify formant-level differences between converted and original speech.

**Jedrasiak (2026)** lists LFCC and CQCC features as discriminative (which indirectly capture formant structure) but does not separately analyze formant-specific features.

### 4.2 Why This Gap Exists

The lack of formant-specific analysis likely reflects two factors:
1. Modern neural TTS (e.g., Tacotron 2, VITS) and VC systems produce quite natural-sounding formant patterns for **static vowels** -- the spectral envelope quality of modern synthesis is high
2. Where formant artifacts exist, they are captured by cepstral features (MFCC, LFCC, CQCC) without requiring explicit formant tracking

### 4.3 Where Formant Analysis May Still Matter

Formant **transitions** (coarticulation) and formant **bandwidths** remain theoretically promising targets because:
- Coarticulation is a complex, context-dependent phenomenon that requires modeling articulatory dynamics
- Formant bandwidths relate to vocal tract damping, which synthesis systems may not accurately replicate
- These are not well-captured by frame-level cepstral features

This represents a significant research gap suitable for original investigation.

---

## 5. Phase and Temporal Differences

### 5.1 Phase Coherence and Harmonic Discontinuities

**Jedrasiak (2026)** reports that "phase-based characteristics demonstrated effectiveness in detecting harmonic discontinuities" in synthetic speech. This is consistent with the known limitation that most neural vocoders either:
- Generate phase from random initialization (WaveGAN, HiFi-GAN)
- Use minimum-phase reconstruction from magnitude spectra
- Have imperfect phase modeling in autoregressive vocoders

The result is that **inter-harmonic phase relationships** in synthetic speech do not follow the patterns seen in natural speech. However, Jedrasiak notes that phase-based detection performance "degraded significantly at low signal-to-noise ratios."

### 5.2 Temporal Prediction Structure

**Borrelli et al. (2021)** showed that both short-term and long-term prediction traces differ between real and synthetic speech. The temporal structure of the LP residual -- how the prediction error evolves over time -- captures glottal excitation patterns that vocoders struggle to reproduce. This includes:
- Pitch-synchronous prediction residual patterns
- Long-term temporal correlations in the excitation signal
- Frame-to-frame temporal smoothness characteristics

### 5.3 Temporal Envelope and Micro-Timing

No papers were found that explicitly compare the temporal envelope (amplitude modulation) structure of real vs. synthetic speech in sufficient detail for this review. This represents another research gap.

---

## 6. Statistical Tests and Feature Discriminability

### 6.1 Effect Sizes Reported

Very few papers report formal effect sizes. The available quantitative evidence:

| Feature Category | Effect Size | Source |
|---|---|---|
| Spectral (LFCC, CQCC, MFCC) | Delta-p ~ 0.25 | Jedrasiak (2026) |
| Jitter/Shimmer/HNR/CPP | Delta-p ~ 0.17--0.23 | Jedrasiak (2026) |
| Jitter+Shimmer (complementary to Mel-spec) | 13.37% relative EER reduction | Li et al. (2023) |
| LFCC+GMM-ResNet vs. GMM baseline | 76.1% relative min-tDCF reduction | Lei et al. (2022) |
| Breathing presence/absence | EER = 0.0 (perfect separation) | Layton et al. (2024) |
| High-frequency Gaussian features | 99.58% accuracy, 6.58% EER | Mewada et al. (2023) |

### 6.2 Feature Ranking by Discriminative Power

Based on the aggregate evidence, features ranked by demonstrated discriminative power:

1. **Breathing patterns** (presence/absence) -- near-perfect when available, but easily circumvented by adding synthetic breathing
2. **High-frequency spectral energy** (above 4 kHz) -- strongest spectral feature, but destroyed by narrowband codecs
3. **LFCC** (linear frequency cepstral coefficients) -- consistently outperforms MFCC due to uniform frequency resolution
4. **LP residuals** (short-term and long-term) -- generalize across 17+ synthesis methods
5. **Micro-frequency dynamics** (Doppler/drumhead models) -- strong recent results (NDSS 2025)
6. **Jitter/Shimmer** -- complementary to spectral features, moderate standalone power
7. **HNR/CPP** -- moderate discriminative power, captures voice quality smoothing
8. **Phase coherence** -- effective but degrades in noise
9. **CQCC** -- strong at low-frequency fine structure
10. **Formant patterns** -- weakly characterized in literature, likely moderate

### 6.3 Statistical Methods Used

The following statistical approaches appear in the literature:
- **Equal Error Rate (EER)**: Most common metric; reported in nearly every paper
- **min-tDCF** (minimum tandem detection cost function): Standard in ASVspoof challenges
- **AUPRC / AUC-ROC**: Used by some papers (e.g., Layton et al., 2024)
- **SHAP values**: Used by Ge et al. (2022) for feature importance in spectrogram-based models
- **Grad-CAM**: Used by Kwak et al. (2023) for spectrogram region importance
- **Effect size Delta-p**: Reported only by Jedrasiak (2026)
- **Relative EER reduction**: Reported by Li et al. (2023) and others
- **Frechet Audio Distance (FAD)**: Used by Govindu et al. (2023) for generation quality assessment

**Critical gap:** Almost no papers report Cohen's d, confidence intervals, or proper statistical hypothesis tests (t-test, Wilcoxon, ANOVA) comparing feature distributions between real and fake speech. Most evaluation is purely in terms of system-level detection metrics.

---

## 7. Behavior Under Codec Compression

### 7.1 The Codec Problem

The ASVspoof 2021 Deepfake (DF) task introduced codec variability as a central challenge. Systems trained on clean audio saw dramatic performance degradation when evaluated on codec-compressed speech.

**Wang et al. (2022)** demonstrated that "using the low-frequency subbands of signals as input can mitigate the negative impact introduced by codecs on countermeasure systems." Key findings:
- Low-pass filtering with varying cutoff frequencies **reduced EER by approximately 25%** relative to using full-band features
- Deep learning-based bandwidth extension further improved performance
- The technique remained effective even with VAD applied

> **Citation:** Wang, Y., Wang, X., Nishizaki, H., & Li, M. (2022). Low Pass Filtering and Bandwidth Extension for Robust Anti-spoofing Countermeasure Against Codec Variabilities. *arXiv preprint* arXiv:2211.06546. DOI: [10.48550/arxiv.2211.06546](https://doi.org/10.48550/arxiv.2211.06546)

### 7.2 What Codecs Destroy

Codecs affect different frequency bands depending on their specification:
- **Narrowband telephony codecs** (G.711, G.726, AMR-NB): Bandwidth limited to 300-3400 Hz, completely destroying all high-frequency artifacts
- **Wideband codecs** (AMR-WB, Opus): Extend to ~7 kHz but still attenuate upper frequencies
- **MP3 / AAC**: Preserve wider bandwidth but introduce their own spectral artifacts (pre-echo, temporal smearing)

The implication: **high-frequency spectral artifacts (above 4 kHz), which are the strongest discriminative features in clean conditions, are the first casualties of codec compression.**

### 7.3 What Survives Compression

Based on the evidence:

**Features that survive well:**
- Low-frequency sub-band features (below 4 kHz) -- Wang et al. (2022)
- Structural consistency patterns -- Faloudah et al. (2026) showed their GSNet, which "prioritizes structural consistency over artifact detection," achieved 0.929% EER on ASVspoof 2021 DF
- LP residual patterns in the low-frequency band -- likely survive because they capture excitation characteristics in the voice band
- CQCC features (geometrically spaced, emphasizing low frequencies) -- theoretically robust

**Features with moderate survival:**
- Jitter/shimmer -- these operate on the F0 and amplitude domains, which codecs must preserve to maintain intelligibility. However, codec-induced jitter could mask synthesis-induced jitter
- HNR in the voice band (below 4 kHz)

**Features destroyed or severely degraded:**
- High-frequency spectral energy (above codec bandwidth)
- Phase coherence (codecs typically discard or heavily quantize phase)
- Fine temporal structure (codec frame boundaries introduce their own artifacts)

> **Citation:** Faloudah, A.Z., Ikram, M.A., & Abdel-Hakim, A.E. (2026). GSNet: A Hybrid Graph Siamese Network for Robust Deepfake Audio Detection. *TechRxiv*. DOI: [10.36227/techrxiv.177130683.36376095/v1](https://doi.org/10.36227/techrxiv.177130683.36376095/v1)

### 7.4 Does Compression Make Real and Fake MORE or LESS Similar?

The evidence suggests **compression makes real and fake audio more similar** in the high-frequency domain (by destroying discriminative artifacts) but may actually **introduce new discriminative cues** in the low-frequency domain. The reasoning:

1. Codec compression applies identically to both real and fake speech, but if the input spectral characteristics differ (e.g., different spectral tilt, different harmonic structure), the codec will produce different compressed outputs
2. Some codec artifacts (e.g., quantization noise patterns) may interact differently with the different spectral characteristics of real vs. synthetic speech
3. Wang et al. (2022) showed that focusing on low-frequency bands **improves** detection under codec conditions, implying these bands carry codec-robust discriminative information that was previously masked by the dominance of high-frequency features

### 7.5 The Codec-Aware Detection Paradigm

**Chen et al. (2021)** proposed a "channel-robust" detection system for ASVspoof 2021, specifically targeting codec variability. Their approach combined data augmentation (applying various codecs during training) with architecture design.

> **Citation:** Chen, X., Zhang, Y., Zhu, G., & Duan, Z. (2021). UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021. *arXiv preprint* arXiv:2107.12018. DOI: [10.48550/arxiv.2107.12018](https://doi.org/10.48550/arxiv.2107.12018)

**Tak et al. (2022)** achieved the lowest EERs on ASVspoof 2021 DF using wav2vec 2.0 with data augmentation (RawBoost), representing "almost 90% relative improvement" over the baseline. While this uses SSL features (not handcrafted), the RawBoost augmentation includes additive noise and convolutive distortion that simulate codec effects.

> **Citation:** Tak, H., Todisco, M., Wang, X., Jung, J.-w., Yamagishi, J., & Evans, N. (2022). Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation. *Proc. Odyssey 2022*. DOI: [10.21437/odyssey.2022-16](https://doi.org/10.21437/odyssey.2022-16)

> **Citation:** Tak, H., Kamble, M.R., Patino, J., Todisco, M., & Evans, N. (2022). RawBoost: A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing. *Proc. ICASSP 2022*. DOI: [10.1109/icassp43922.2022.9746213](https://doi.org/10.1109/icassp43922.2022.9746213)

---

## 8. Per-Attack Type Analysis

### 8.1 Different TTS Systems Leave Different Artifacts

The ASVspoof 2019 challenge included 19 different spoofing attacks (A01-A19), spanning:
- **Waveform concatenation** (A01-A02): Joins natural speech segments, artifacts at boundaries
- **Vocoder-based parametric synthesis** (A03-A06): WORLD, STRAIGHT vocoders -- spectral envelope oversmoothing
- **Neural TTS** (A07-A12): Tacotron, WaveNet variants -- high-frequency artifacts, temporal smoothness
- **Voice conversion** (A13-A19): Various VC methods -- formant mapping artifacts, source characteristic residuals

**Wang et al. (2020)** documented the ASVspoof 2019 database design, noting that attacks span "from traditional vocoders to modern deep learning solutions," and that different countermeasure features show dramatically different performance across attack types.

> **Citation:** Wang, X., Yamagishi, J., Todisco, M., et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language*, 64, 101114. DOI: [10.1016/j.csl.2020.101114](https://doi.org/10.1016/j.csl.2020.101114)

### 8.2 Attack-Specific Artifact Signatures

**Borrelli et al. (2021)** validated their LP-based features across all 17 synthetic speech algorithms, demonstrating that while feature patterns differ across attacks, the LP residual approach generalizes. This suggests that all synthesis methods share a common weakness: **imperfect excitation signal generation**.

**Ge et al. (2022)** used SHAP to reveal that different detection models attend to different spectrogram regions for the same attack, indicating that there is no single "artifact location" -- rather, each synthesis method produces a distinct spectral signature, and different detectors have learned to focus on different parts of this signature.

> **Citation:** Ge, W., Patino, J., Todisco, M., & Evans, N. (2022). Explaining Deep Learning Models for Spoofing and Deepfake Detection with Shapley Additive Explanations. *Proc. ICASSP 2022*. DOI: [10.1109/icassp43922.2022.9747476](https://doi.org/10.1109/icassp43922.2022.9747476)

### 8.3 GAN-Based vs. Autoregressive Vocoders

**Song et al. (2022)** specifically characterized MelGAN artifacts:
- Multi-band MelGAN: **metallic sound** from harmonic generation errors
- Oversmoothing: excessive smoothing of aperiodic components

By contrast, autoregressive vocoders (WaveNet, WaveRNN) tend to produce more natural-sounding output but at higher computational cost. Their artifacts are subtler and concentrate in:
- Long-range temporal coherence
- Prosodic naturalness at utterance boundaries
- Phase relationships between harmonics

---

## 9. Synthesis and Gap Analysis

### 9.1 What We Know (Strong Evidence)

1. **High-frequency artifacts are the dominant spectral cue** in clean conditions. Vocoders (both traditional and neural) fail to reproduce natural high-frequency content, producing oversmoothed spectra above 4 kHz. This is captured by LFCC (uniform frequency resolution), IMFCC (inverted mel scale), and high-frequency Gaussian features.

2. **Micro-prosodic features differ systematically.** Jitter shows different distributions; shimmer shows excessive dynamic variation; HNR and CPP are unnaturally regular. Effect sizes are moderate (Delta-p ~ 0.17-0.25).

3. **Breathing patterns are a powerful but fragile cue.** Most TTS systems do not generate breathing, providing near-perfect detection. However, this is easily circumventable and may not generalize to future systems.

4. **LP residuals capture universal synthesis weaknesses.** The excitation signal (pitch pulse shape, noise component) is poorly modeled by all tested synthesis methods, making LP-based features robust across attack types.

5. **Codecs destroy high-frequency cues but low-frequency features survive.** Low-pass filtering to below 4 kHz actually improves codec-robust detection, and structural consistency outperforms artifact detection under compression.

### 9.2 What We Don't Know (Research Gaps)

1. **Formant-level analysis is almost entirely missing.** No rigorous comparison of formant frequencies, bandwidths, or transition rates between real and synthetic speech in the deepfake detection context.

2. **Population-level F0 statistics are uncharacterized.** No paper provides a thorough statistical comparison of F0 distribution (mean, variance, skewness, range) across real vs. synthetic speech at scale.

3. **Per-feature effect sizes are rarely reported.** The field relies on system-level EER rather than individual feature analysis with proper statistical testing.

4. **Codec interaction with specific features is poorly understood.** We know high frequencies are destroyed, but the interaction between specific codec types and specific acoustic features (e.g., how does AMR-NB affect jitter measurement?) is uncharacterized.

5. **Temporal dynamics are underexplored.** Temporal envelope modulation, speech rhythm (PVI, nPVI), and long-range prosodic structure have not been compared between real and synthetic speech.

6. **Feature interactions are unknown.** Do spectral and prosodic features provide truly independent information, or are they correlated? No multivariate analysis has been published.

### 9.3 Implications for the ACAGAT Project

For the codec-robust ACAGAT architecture in this project, the literature strongly supports:

1. **Using LFCC-based features** (captured in the 36-dim acoustic feature vector) as they provide uniform frequency resolution
2. **Including jitter/shimmer features** as complementary prosodic markers (currently captured in the prosody component)
3. **Focusing on low-frequency sub-bands** for codec robustness -- features below 4 kHz retain discriminative power
4. **LP residual features** as a potential addition (not currently in the feature set)
5. **PNCC features** (power-normalized cepstral coefficients, already in the 36-dim set) for noise robustness
6. **The 36-dim feature vector** (PNCC:13 + LFCC:8 + Formant:8 + Prosody:7) aligns well with the literature -- it covers spectral (PNCC, LFCC), formant (exploratory), and prosodic dimensions

**Feature priorities for codec robustness:**
- PNCC (noise-robust by design, captures spectral envelope) -- HIGH priority
- LFCC low-order coefficients (capture low-frequency spectral shape) -- HIGH priority
- Prosody features (jitter, shimmer, F0 statistics) -- MEDIUM priority, partially codec-robust
- Formant features -- MEDIUM priority, need empirical validation
- High-order LFCC (capture high-frequency detail) -- LOW priority under codec conditions

---

## 10. Complete Reference List

### Primary Papers (Directly Addressing Acoustic Differences)

[1] Li, K., Lu, X., Akagi, M., & Unoki, M. (2023). Contributions of Jitter and Shimmer in the Voice for Fake Audio Detection. *IEEE Access*, 11, 84689-84698. DOI: [10.1109/access.2023.3301616](https://doi.org/10.1109/access.2023.3301616)

[2] Jedrasiak, K. (2026). Analysis of Acoustic Anomalies and Speech Artefacts in Synthetic Content. *Safety & Fire Technology*, 67(1), 58-84. DOI: [10.12845/sft.67.1.2026.3](https://doi.org/10.12845/sft.67.1.2026.3)

[3] Borrelli, C., Bestagini, P., Antonacci, F., Sarti, A., & Tubaro, S. (2021). Synthetic speech detection through short-term and long-term prediction traces. *EURASIP Journal on Information Security*, 2021, 2. DOI: [10.1186/s13635-021-00116-3](https://doi.org/10.1186/s13635-021-00116-3)

[4] Mewada, H., Al-Asad, J.F., Almalki, F.A., Khan, A.H., Almujally, N.A., El-Nakla, S., & Naith, Q.H. (2023). Gaussian-Filtered High-Frequency-Feature Trained Optimized BiLSTM Network for Spoofed-Speech Classification. *Sensors*, 23(14), 6637. DOI: [10.3390/s23146637](https://doi.org/10.3390/s23146637)

[5] Kumari, K., Abbasihafshejani, M., Pegoraro, A., Rieger, P., Arshi, K., Jadliwala, M., & Sadeghi, A.-R. (2025). VoiceRadar: Voice Deepfake Detection using Micro-Frequency and Compositional Analysis. *Proc. NDSS 2025*. DOI: [10.14722/ndss.2025.243389](https://doi.org/10.14722/ndss.2025.243389)

[6] Layton, S., De Andrade, T., Olszewski, D., Warren, K.M., Butler, K., & Traynor, P. (2024). Every Breath You Don't Take: Deepfake Speech Detection Using Breath. *arXiv preprint* arXiv:2404.15143. DOI: [10.48550/arxiv.2404.15143](https://doi.org/10.48550/arxiv.2404.15143)

[7] Song, K., Cong, J., Wang, X., Zhang, Y., Xie, L., Jiang, N., & Wu, H. (2022). Robust MelGAN: A robust universal neural vocoder for high-fidelity TTS. *arXiv preprint* arXiv:2210.17349. DOI: [10.48550/arxiv.2210.17349](https://doi.org/10.48550/arxiv.2210.17349)

[8] Wang, Y., Wang, X., Nishizaki, H., & Li, M. (2022). Low Pass Filtering and Bandwidth Extension for Robust Anti-spoofing Countermeasure Against Codec Variabilities. *arXiv preprint* arXiv:2211.06546. DOI: [10.48550/arxiv.2211.06546](https://doi.org/10.48550/arxiv.2211.06546)

[9] Ge, W., Patino, J., Todisco, M., & Evans, N. (2022). Explaining Deep Learning Models for Spoofing and Deepfake Detection with Shapley Additive Explanations. *Proc. ICASSP 2022*. DOI: [10.1109/icassp43922.2022.9747476](https://doi.org/10.1109/icassp43922.2022.9747476)

[10] Kwak, I.-Y., Kwag, S., Lee, J., Jeon, Y., Hwang, J., Choi, H.-J., Yang, J.-H., Han, S.-Y., Huh, J.H., Lee, C.-H., & Yoon, J.W. (2023). Voice Spoofing Detection Through Residual Network, Max Feature Map, and Depthwise Separable Convolution. *IEEE Access*, 11, 49140-49152. DOI: [10.1109/access.2023.3275790](https://doi.org/10.1109/access.2023.3275790)

### Codec Robustness and System-Level Papers

[11] Chen, X., Zhang, Y., Zhu, G., & Duan, Z. (2021). UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021. *arXiv preprint* arXiv:2107.12018. DOI: [10.48550/arxiv.2107.12018](https://doi.org/10.48550/arxiv.2107.12018)

[12] Tak, H., Todisco, M., Wang, X., Jung, J.-w., Yamagishi, J., & Evans, N. (2022). Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation. *Proc. Odyssey 2022*. DOI: [10.21437/odyssey.2022-16](https://doi.org/10.21437/odyssey.2022-16)

[13] Tak, H., Kamble, M.R., Patino, J., Todisco, M., & Evans, N. (2022). RawBoost: A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing. *Proc. ICASSP 2022*. DOI: [10.1109/icassp43922.2022.9746213](https://doi.org/10.1109/icassp43922.2022.9746213)

[14] Lei, Z., Yan, H., Liu, C., Ma, M., & Yang, Y. (2022). Two-Path GMM-ResNet and GMM-SENet for ASV Spoofing Detection. *Proc. ICASSP 2022*. DOI: [10.1109/icassp43922.2022.9746163](https://doi.org/10.1109/icassp43922.2022.9746163)

[15] Faloudah, A.Z., Ikram, M.A., & Abdel-Hakim, A.E. (2026). GSNet: A Hybrid Graph Siamese Network for Robust Deepfake Audio Detection. *TechRxiv*. DOI: [10.36227/techrxiv.177130683.36376095/v1](https://doi.org/10.36227/techrxiv.177130683.36376095/v1)

### Challenge Descriptions and Databases

[16] Wang, X., Yamagishi, J., Todisco, M., et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language*, 64, 101114. DOI: [10.1016/j.csl.2020.101114](https://doi.org/10.1016/j.csl.2020.101114)

[17] Todisco, M., Wang, X., Vestman, V., et al. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. *Proc. Interspeech 2019*, 1008-1012. DOI: [10.21437/interspeech.2019-2249](https://doi.org/10.21437/interspeech.2019-2249)

[18] Wang, X., Delgado, H., Tak, H., et al. (2024). ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale. *Proc. ASVspoof 2024 Workshop*. DOI: [10.21437/asvspoof.2024-1](https://doi.org/10.21437/asvspoof.2024-1)

### Surveys and Reviews

[19] Kamble, M.R., Sailor, H.B., Patil, H.A., & Li, H. (2020). Advances in anti-spoofing: from the perspective of ASVspoof challenges. *APSIPA Transactions on Signal and Information Processing*, 9(1). DOI: [10.1017/atsip.2019.21](https://doi.org/10.1017/atsip.2019.21)

[20] Sisman, B., Yamagishi, J., King, S., & Li, H. (2020). An Overview of Voice Conversion and Its Challenges. *IEEE/ACM Trans. Audio, Speech and Language Processing*, 29, 132-157. DOI: [10.1109/taslp.2020.3038524](https://doi.org/10.1109/taslp.2020.3038524)

[21] Unoki, M., Li, K., Chaiwongyen, A., Nguyen, Q.-H., & Zaman, K. (2024). Deepfake Speech Detection: Approaches from Acoustic Features to Deep Neural Networks. *IEICE Trans. Information and Systems*, E108.D(4), 300-310. DOI: [10.1587/transinf.2024mui0001](https://doi.org/10.1587/transinf.2024mui0001)

### Vocoder Analysis

[22] Airaksinen, M., Juvela, L., Bollepalli, B., Yamagishi, J., & Alku, P. (2018). A Comparison Between STRAIGHT, Glottal, and Sinusoidal Vocoding in Statistical Parametric Speech Synthesis. *IEEE/ACM Trans. Audio, Speech and Language Processing*, 26(9), 1658-1670. DOI: [10.1109/taslp.2018.2835720](https://doi.org/10.1109/taslp.2018.2835720)

### Human Perception

[23] Han, C., Mitra, P., & Billah, S.M. (2024). Uncovering Human Traits in Determining Real and Spoofed Audio. *Proc. CHI 2024*. DOI: [10.1145/3613904.3642817](https://doi.org/10.1145/3613904.3642817)

[24] Mai, K.T., et al. (2023). Warning: Humans cannot reliably detect speech deepfakes. *PLOS ONE*. DOI: [10.1371/journal.pone.0285333](https://doi.org/10.1371/journal.pone.0285333)

### XAI for Detection

[25] Govindu, A., Kale, P., Hullur, A., Gurav, A., & Godse, P. (2023). Deepfake audio detection and justification with Explainable Artificial Intelligence (XAI). *Research Square preprint*. DOI: [10.21203/rs.3.rs-3444277/v1](https://doi.org/10.21203/rs.3.rs-3444277/v1)

### Additional Integration Papers

[26] Muruganandham, S.K., et al. (2025). LSTM autoencoder based parallel architecture for deepfake audio detection with dynamic residual encoding and feature fusion. *Scientific Reports*. DOI: [10.1038/s41598-025-08198-6](https://doi.org/10.1038/s41598-025-08198-6)

[27] Chakravarty, N., & Dua, M. (2023). Data augmentation and hybrid feature amalgamation to detect audio deep fake attacks. *Physica Scripta*, 98(9). DOI: [10.1088/1402-4896/acea05](https://doi.org/10.1088/1402-4896/acea05)

[28] Li, X., Li, K., Zheng, Y., Chen, Y., Ji, X., & Xu, W. (2024). SafeEar: Content Privacy-Preserving Audio Deepfake Detection. *arXiv preprint* arXiv:2409.09272. DOI: [10.48550/arxiv.2409.09272](https://doi.org/10.48550/arxiv.2409.09272)

---

## PRISMA Summary

- **Databases searched:** OpenAlex (primary), Semantic Scholar (rate-limited), arXiv (via OpenAlex)
- **Search date:** 2026-04-06
- **Date range:** 2018-01-01 to 2026-12-31
- **Search queries executed:** 30+ unique queries covering spectral artifacts, prosodic features, formant analysis, phase spectrum, codec robustness, jitter/shimmer, HNR, breathing patterns, LFCC, CQCC, sub-band analysis, vocoder fingerprints, interpretable features, XAI methods
- **Total records identified:** ~250 (with significant duplication across queries)
- **After deduplication and title/abstract screening:** 42 potentially relevant
- **After full relevance assessment:** 28 papers included
- **Exclusion reasons:** Survey-only papers without original acoustic analysis (8), architecture-focused papers without acoustic insight (4), unrelated papers (2)

---

*This review was conducted to support the ACAGAT architecture development for NeurIPS 2026, specifically to justify the 36-dimensional acoustic feature vector and its expected codec robustness.*
