# Hypotheses: Which Acoustic Signals Distinguish Real from Fake Audio, and Why

**Date:** 2026-04-06
**Domain:** Audio deepfake detection (ASVspoof 2019 LA, attacks A01-A06)
**Feature space:** 36-dim codec-robust acoustic features (PNCC:13, LFCC:8, Formant:8, Prosody:7)

---

## Methodology

Each hypothesis was generated via Tree-of-Thought reasoning across five theoretical frameworks:
1. Signal processing theory (vocoder architecture limitations)
2. Articulatory phonetics (physical speech production constraints)
3. Statistical learning theory (distributional differences)
4. Psychoacoustics (auditory perception modeling)
5. Information theory (redundancy and compression)

Evaluation criteria:
- **Novelty** (1-5): 1=well-established, 5=completely novel
- **Testability** (1-5): 1=difficult to operationalize, 5=directly measurable from our 36 features
- **Expected effect size**: Small (d<0.2), Medium (0.2<d<0.5), Large (d>0.5)

---

## I. PNCC Hypotheses (Spectral Envelope via Gammatone Filterbank)

### H1: TTS vocoders produce spectral envelopes with unnaturally smooth gammatone channel outputs

**Precise statement:** The mean inter-frame variance of gammatone-filtered spectral energy (before cepstral compression) is significantly lower in synthetic speech than in bona fide speech.

**Mechanism:** Neural vocoders (WaveNet, WaveRNN, Griffin-Lim used in A01-A06) generate spectral envelopes from smoothed mel-spectrogram predictions. The vocoder's neural network acts as a low-pass filter on the spectral envelope trajectory, suppressing the micro-fluctuations that arise from natural glottal pulse irregularity and turbulent airflow. When passed through gammatone filters -- which have broader tails than mel filters due to their 4th-order impulse response (Patterson et al., 1992) -- this over-smoothing manifests as reduced variance in the PNCC feature trajectory.

**Direction:** Lower PNCC inter-frame variance in fake vs. real.

**H0:** Mean PNCC variance (across coefficients 0-12) does not differ between bona fide and spoofed speech (p > 0.05).
**Falsifying observation:** No significant difference in PNCC variance, or higher variance in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 5 | Large |

**Literature:** Kim & Stern (2016) showed PNCC captures noise-robust spectral structure but did not study synthetic speech smoothness. Sahidullah et al. (2015) demonstrated cepstral variance differences between natural and spoofed speech using MFCC, but not with gammatone-based PNCC.

---

### H2: Power-law compression (1/15 exponent) amplifies subtle spectral discontinuities at vocoder synthesis boundaries

**Precise statement:** The 1/15 power-law nonlinearity in PNCC computation produces larger absolute differences between bona fide and spoofed speech than logarithmic compression (as used in MFCC/LFCC), measured by Cohen's d on the mean PNCC vector.

**Mechanism:** The power-law function f(x) = x^(1/15) has a derivative f'(x) = (1/15) * x^(-14/15), which is extremely large near zero. Vocoders produce occasional near-zero energy frames at synthesis boundaries (e.g., between concatenative units in A04-A06, or at the edges of mel-spectrogram windows). These near-zero values are dramatically amplified relative to surrounding frames by the 1/15 exponent, creating sharp cepstral transients that are less pronounced under log compression. In natural speech, energy rarely reaches the near-zero regime because of continuous glottal airflow, so this amplification effect is asymmetric.

**Direction:** Higher PNCC coefficient magnitude in specific low-energy frames for fake; overall, wider PNCC dynamic range in fake vs. real.

**H0:** Cohen's d for PNCC-based classification does not exceed Cohen's d for LFCC-based classification (matched coefficient count).
**Falsifying observation:** LFCC achieves equal or greater discriminability than PNCC on the same data split.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 4 | 4 | Medium |

**Literature:** Kim & Stern (2016, Section III-B) justified the 1/15 exponent for noise robustness. No prior work has examined its effect on synthetic speech artifact amplification.

---

### H3: Higher-order PNCC coefficients (7-12) capture fine spectral detail that vocoders systematically undermodel

**Precise statement:** PNCC coefficients 7-12 have significantly higher discriminative power (measured by individual feature AUC) for real vs. fake classification than coefficients 0-6.

**Mechanism:** Cepstral coefficients are ordered by spectral rate: coefficient 0 captures overall energy, coefficients 1-6 capture the coarse spectral envelope (formant structure), and coefficients 7-12 capture fine spectral detail (spectral texture, harmonic structure, glottal pulse shape). TTS systems (A01-A06) are typically trained to minimize mel-spectrogram reconstruction loss, which heavily weights low-order spectral structure. Higher-order spectral detail -- the "texture" of the spectrum created by individual harmonic interactions with the vocal tract -- is not explicitly modeled by most vocoders and thus appears more artificial in higher cepstral coefficients.

**Direction:** Higher AUC for coefficients 7-12 than for coefficients 0-6 in fake detection.

**H0:** Mean AUC of PNCC coefficients 7-12 is not greater than mean AUC of coefficients 0-6.
**Falsifying observation:** Lower-order coefficients are equally or more discriminative.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** Sahidullah et al. (2015, INTERSPEECH) showed higher-order LFCC coefficients are more discriminative for spoofing detection. The gammatone-based PNCC extension is untested.

---

### H4: PNCC temporal asymmetry is disrupted in synthetic speech due to uniform frame generation

**Precise statement:** The skewness of the PNCC coefficient trajectory over time is significantly closer to zero in spoofed speech than in bona fide speech.

**Mechanism:** Natural speech production is inherently temporally asymmetric: consonant-to-vowel transitions involve rapid spectral changes (short rise times) while vowel-to-consonant transitions are more gradual (longer fall times). This produces positively skewed temporal trajectories in gammatone channel energy. Neural vocoders generate frames with approximately uniform computational cost and identical temporal resolution, collapsing this natural asymmetry. The medium-time processing in PNCC (5-frame smoothing) preserves this asymmetry in natural speech but averages it away in already-symmetric synthetic trajectories.

**Direction:** PNCC temporal skewness closer to zero (more symmetric) in fake speech.

**H0:** No difference in PNCC temporal skewness between bona fide and spoofed conditions.
**Falsifying observation:** Equal or greater skewness magnitude in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 4 | 3 (requires per-frame PNCC, not just mean) | Small |

**Literature:** No direct precedent. Temporal asymmetry in speech acoustics is well-established in phonetics (Stevens, 1998, "Acoustic Phonetics") but has not been applied to deepfake detection.

---

### H5: PNCC noise floor normalization reveals vocoder-specific quantization artifacts

**Precise statement:** The ratio of the 2nd percentile to the 50th percentile of PNCC coefficient magnitude differs significantly between bona fide and spoofed speech, with vocoders showing a higher floor-to-median ratio.

**Mechanism:** The asymmetric noise suppression step in PNCC (floor at 2% of mean energy) interacts differently with natural background noise vs. vocoder quantization noise. Natural speech has broadband environmental noise that produces consistent low-energy components across gammatone channels. Vocoder output has structured quantization artifacts from waveform sample generation (especially Griffin-Lim in A01 and neural vocoders in A02-A03), which create a higher, more structured noise floor. After PNCC normalization, this floor appears as elevated minimum values.

**Direction:** Higher noise-floor-to-median ratio in fake speech.

**H0:** No difference in PNCC floor-to-median ratio between conditions.
**Falsifying observation:** Natural speech shows equal or higher floor ratios.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 5 | 3 (requires analysis of PNCC internals) | Small |

**Literature:** Kim & Stern (2016, Section IV-A) describe the noise suppression mechanism but only evaluate it for environmental noise, not vocoder artifacts.

---

## II. LFCC Hypotheses (Linear Frequency Spectral Detail)

### H4-LFCC: Synthetic speech has reduced high-frequency energy above 4kHz due to vocoder bandwidth limitations

**Precise statement:** LFCC coefficients corresponding to linear filter bands above 4kHz (filters 10-20 in our 20-filter bank spanning 0-8kHz) contribute disproportionately to classification accuracy, and their mean values are significantly lower in spoofed speech.

**Mechanism:** Many TTS vocoders in ASVspoof 2019 (particularly WaveNet variants A01-A02 and neural waveform generators A03) are trained on mel-spectrograms with 80 mel bins up to 7.6kHz or 8kHz. However, mel-frequency resolution above 4kHz is coarse (fewer bins per Hz), meaning the vocoder receives less supervision for high-frequency reconstruction. Additionally, some vocoders use band-limited training data or apply anti-aliasing filters. The linear frequency spacing in LFCC gives equal resolution across the spectrum, making these high-frequency deficiencies more visible than in mel-based features.

**Direction:** Lower LFCC values for high-frequency-associated coefficients in fake speech.

**H0:** LFCC coefficients have equal discriminative power regardless of their spectral frequency association.
**Falsifying observation:** Low-frequency LFCC coefficients are equally or more discriminative.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 5 | Large |

**Literature:** Sahidullah et al. (2015, INTERSPEECH) showed LFCC outperforms MFCC for spoofing detection precisely because of linear frequency resolution. Wang & Yamagishi (2021, ICASSP) confirmed high-frequency artifacts are a key spoofing indicator.

---

### H5-LFCC: LFCC captures spectral "ripple" artifacts at vocoder frame boundaries

**Precise statement:** The variance of LFCC coefficients 3-7 across time is significantly higher in spoofed speech than in bona fide speech, reflecting spectral discontinuities at synthesis frame boundaries.

**Mechanism:** Vocoders generate audio frame-by-frame (typically 12.5ms or 256 samples at 16kHz). At frame boundaries, the spectral envelope predicted for consecutive frames may not perfectly align, creating spectral "ripple" -- rapid oscillations in the spectral envelope that manifest as elevated energy in mid-to-high cepstral coefficients. These ripples are more visible in LFCC than MFCC because linear frequency spacing preserves the spectral detail without the compression introduced by mel warping. In natural speech, the vocal tract changes continuously (not frame-by-frame), so these boundary artifacts are absent.

**Direction:** Higher temporal variance of LFCC coefficients 3-7 in fake speech.

**H0:** No difference in LFCC temporal variance between bona fide and spoofed conditions.
**Falsifying observation:** Lower or equal LFCC variance in spoofed speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 3 (requires per-frame LFCC) | Medium |

**Literature:** Muller et al. (2022, INTERSPEECH) showed frame-boundary artifacts in neural vocoders. No prior work connects this to LFCC specifically.

---

### H6-LFCC: The log compression in LFCC exposes harmonic-to-noise ratio differences between real and synthetic speech

**Precise statement:** The LFCC coefficient 1 (which captures the overall spectral tilt after log compression) is significantly different between bona fide and spoofed speech, reflecting different harmonic-to-noise ratios (HNR).

**Mechanism:** Natural speech has a characteristic HNR profile: voiced segments have HNR of 15-25 dB, while unvoiced segments have HNR near 0 dB. The logarithmic compression in LFCC converts these multiplicative ratios into additive differences that are captured primarily by coefficient 1 (spectral tilt). Neural vocoders tend to produce cleaner harmonics than natural speech (higher HNR in voiced segments) because the neural network implicitly acts as a denoising filter. This elevated HNR shifts the LFCC coefficient 1 distribution.

**Direction:** LFCC coefficient 1 shifted toward steeper spectral tilt (higher magnitude) in fake speech due to cleaner harmonics.

**H0:** No difference in LFCC coefficient 1 distribution between conditions.
**Falsifying observation:** Equal or opposite shift in spectral tilt.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** de Leon et al. (2012, ICASSP) noted HNR differences in voice conversion output. Not previously connected to LFCC coefficient interpretation.

---

### H7-LFCC: Voice conversion attacks (A04-A06) produce LFCC distributions closer to bona fide than TTS attacks (A01-A03)

**Precise statement:** The Mahalanobis distance between the bona fide LFCC distribution and the VC attack LFCC distributions (A04-A06) is significantly smaller than the distance to TTS attack distributions (A01-A03).

**Mechanism:** Voice conversion systems start from a real speech signal and transform it, preserving much of the original spectral structure (including natural high-frequency content and frame continuity). TTS systems generate the entire waveform from text, introducing spectral artifacts at every stage. Therefore, VC output should be closer to natural speech in LFCC space because the linear frequency representation captures the preserved natural spectral components. This makes VC attacks harder to detect with LFCC alone.

**Direction:** Smaller LFCC distributional distance for VC attacks; higher LFCC-based EER on A04-A06 than A01-A03.

**H0:** No difference in LFCC-based detection performance between TTS and VC attack categories.
**Falsifying observation:** VC attacks are equally or more separable in LFCC space.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 5 | Large |

**Literature:** Todisco et al. (2019, INTERSPEECH) showed per-attack EER variation in ASVspoof 2019. The LFCC-specific mechanism has not been explicitly tested.

---

## III. Formant Hypotheses (Vocal Tract Resonances)

### H6-F: TTS formant transitions are unnaturally smooth, lacking coarticulation microstructure

**Precise statement:** The standard deviation of F1 and F2 within utterances (f1_std, f2_std in our feature vector) is significantly lower in spoofed speech than in bona fide speech.

**Mechanism:** Natural coarticulation -- the overlap of articulatory gestures for adjacent phonemes -- produces complex, non-linear formant trajectories with rapid transitions, overshoot, and undershoot (Lindblom, 1963). TTS systems trained on mel-spectrogram loss optimize for smooth spectral trajectories because the L1/L2 loss functions penalize high-frequency temporal variation. This "regression to the mean" effect reduces formant variability within utterances. F1 and F2 standard deviations directly capture this reduced variability.

**Direction:** Lower f1_std and f2_std in fake speech.

**H0:** No difference in within-utterance formant standard deviation between bona fide and spoofed speech.
**Falsifying observation:** Equal or higher formant variability in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** Yi et al. (2020) showed TTS produces "over-smooth" spectral features. Ren et al. (2019, NeurIPS, FastSpeech) explicitly identified over-smoothing as a TTS limitation. Not previously quantified via formant variability for deepfake detection.

---

### H7-F: Formant bandwidth in synthetic speech is narrower due to over-modeled resonances

**Precise statement:** The mean formant bandwidth (f1_bw, f2_bw) is significantly lower in spoofed speech than in bona fide speech.

**Mechanism:** Formant bandwidth reflects energy dissipation in the vocal tract -- wider bandwidths indicate more damping from tissue walls, nasalization, and imprecise closure. Natural speech has highly variable bandwidth due to speaker-dependent physiology and moment-to-moment articulatory variation. TTS systems model formant frequencies more accurately than bandwidths because frequency has more perceptual salience. Neural vocoders tend to produce sharper (narrower bandwidth) resonances because: (a) the spectral loss function rewards precise frequency peaks, (b) damping factors are underrepresented in training targets, and (c) the vocal tract transfer function is approximated with fewer poles than the real system.

**Direction:** Lower f1_bw and f2_bw in fake speech (sharper, less damped resonances).

**H0:** No difference in formant bandwidth between conditions.
**Falsifying observation:** Equal or wider bandwidths in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 4 | 5 | Medium |

**Literature:** Formant bandwidth as a spoofing indicator was noted by De Leon et al. (2012, ICASSP) for voice conversion. The mechanism linking TTS loss functions to bandwidth compression is novel.

---

### H8-F: F1/F2 ratio distribution reveals compressed vowel space in TTS

**Precise statement:** The f1_f2_ratio feature has significantly lower variance and a more concentrated distribution in spoofed speech, reflecting a compressed acoustic vowel space.

**Mechanism:** The F1/F2 plane defines the acoustic vowel space. Natural speakers exhibit large inter- and intra-speaker variability in vowel production due to dialect, speaking rate, emotional state, and articulatory precision (hypo/hyper-articulation; Lindblom, 1990). TTS systems are typically trained on read speech from professional speakers in controlled environments, producing a regularized, compressed vowel space with less extreme vowel positions. The f1_f2_ratio collapses the two-dimensional vowel space into a one-dimensional summary that captures this compression: a smaller ratio range indicates fewer extreme vowel configurations.

**Direction:** Lower variance of f1_f2_ratio in fake speech; mean may shift toward the centroid of the vowel space.

**H0:** No difference in f1_f2_ratio variance or mean between conditions.
**Falsifying observation:** Equal or greater vowel space extent in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** Vowel space compression in TTS is well-documented in speech synthesis evaluation (Birkholz et al., 2017). Application to deepfake detection via f1_f2_ratio is novel.

---

### H9-F: Formant dispersion (F2-F1) reveals speaker-independent vocal tract length artifacts

**Precise statement:** The formant_dispersion feature (F2 mean - F1 mean) has a significantly different distribution in spoofed speech, with lower mean and reduced variance compared to bona fide speech.

**Mechanism:** Formant dispersion is an acoustic correlate of vocal tract length (Fitch, 1997, "Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques," JASA). TTS systems that model a single target speaker produce consistent formant dispersion reflecting one vocal tract, while multi-speaker systems map to an averaged vocal tract. In both cases, the natural variation from articulatory dynamics (jaw opening, tongue positioning affecting effective tract length within an utterance) is undermodeled. Additionally, voice conversion systems explicitly manipulate formant frequencies to match a target speaker, but the mapping function may distort the natural F2-F1 relationship.

**Direction:** Lower variance and potentially different mean of formant_dispersion in fake speech.

**H0:** No difference in formant dispersion distribution between conditions.
**Falsifying observation:** Equal distributions.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 4 | 5 | Small |

**Literature:** Fitch (1997) established formant dispersion as a vocal tract length correlate. Application to deepfake detection is novel.

---

### H10-F: VC attacks preserve formant structure better than TTS, making formant features attack-type-dependent

**Precise statement:** Formant-based features achieve significantly higher AUC for TTS attacks (A01-A03) than for VC attacks (A04-A06) in ASVspoof 2019 LA.

**Mechanism:** Voice conversion transforms an existing speech signal, preserving much of the source speaker's prosodic and temporal structure while modifying the spectral envelope to match a target speaker. The formant transformation is typically done via spectral warping (frequency scaling) or line spectral pair modification, which preserves relative formant relationships (bandwidths, dispersion ratios) better than de novo TTS generation. Therefore, formant-based features should be less effective at detecting VC than TTS.

**Direction:** Higher formant feature AUC for A01-A03 (TTS) than A04-A06 (VC).

**H0:** No difference in formant-based detection accuracy between TTS and VC attack categories.
**Falsifying observation:** Equal or better formant-based detection of VC attacks.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 5 | Large |

**Literature:** Wu et al. (2015, Computer Speech & Language) showed attack-type dependence of features. The formant-specific mechanism is not well-studied.

---

## IV. Prosody Hypotheses (Rhythm and Intonation)

### H9-P: F0 standard deviation is lower in TTS due to regression-to-mean pitch modeling

**Precise statement:** The f0_std feature is significantly lower in spoofed speech than in bona fide speech, with the effect being larger for TTS (A01-A03) than VC (A04-A06) attacks.

**Mechanism:** TTS pitch models (autoregressive or flow-based) are trained with L2 loss on F0 trajectories, which inherently favors predictions near the conditional mean. This produces F0 contours with reduced dynamic range -- fewer extreme pitch excursions, less expressive intonation, and attenuated phrase-final pitch falls. The effect is most pronounced in non-autoregressive TTS (FastSpeech-style, potentially A01/A02) where the entire F0 contour is predicted in parallel without sequential conditioning. VC systems (A04-A06) may partially preserve the source speaker's F0 dynamics depending on whether F0 is converted or passed through.

**Direction:** Lower f0_std in fake speech, especially TTS attacks.

**H0:** No difference in f0_std between bona fide and spoofed conditions.
**Falsifying observation:** Equal or higher F0 variability in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 5 | Large |

**Literature:** Ren et al. (2019, NeurIPS, FastSpeech) documented over-smooth pitch in TTS. Wester et al. (2016, INTERSPEECH) showed reduced prosodic variability in TTS. Not previously used as a deepfake detection feature in the ASVspoof context.

---

### H10-P: Voiced ratio is higher in TTS because vocoders produce pitch estimates for noise-like segments

**Precise statement:** The voiced_ratio feature is significantly higher in spoofed speech than in bona fide speech.

**Mechanism:** In natural speech, unvoiced segments (fricatives, stops, aspiration) constitute 30-50% of speech duration. Neural vocoders operating on mel-spectrograms may generate quasi-periodic waveforms even during segments that should be aperiodic, because: (a) the vocoder's pitch predictor is trained primarily on voiced segments and may produce spurious F0 estimates during unvoiced regions, (b) neural waveform generators (WaveNet, WaveGlow) may introduce weak periodicity in noise-like segments due to the autoregressive or flow-based generation mechanism, and (c) voiced/unvoiced decision boundaries are not explicitly modeled in many vocoders. When librosa.pyin analyzes the resulting waveform, it detects pitch in segments that would be aperiodic in natural speech, inflating voiced_ratio.

**Direction:** Higher voiced_ratio in fake speech.

**H0:** No difference in voiced_ratio between conditions.
**Falsifying observation:** Equal or lower voiced_ratio in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** Drugman et al. (2012, IEEE TASLP) documented voiced/unvoiced decision errors in parametric speech synthesis. The specific link to deepfake detection voiced_ratio is novel.

---

### H11-P: Energy contour is more uniform in TTS, producing lower energy_std

**Precise statement:** The energy_std feature (RMS energy standard deviation) is significantly lower in spoofed speech than in bona fide speech.

**Mechanism:** Natural speech has substantial energy variation due to: lexical stress patterns (stressed vs. unstressed syllables differ by 6-12 dB), phrase-level prominence (sentence accent), breathing patterns (energy drops near phrase boundaries), and speaking style variation. TTS energy models are trained to predict energy envelopes that, like F0, regress toward the conditional mean. The training data (typically read speech with consistent recording levels) has already reduced dynamic range compared to spontaneous speech, and the model further compresses it. Additionally, vocoders applying gain to the generated waveform often use smoothed energy targets, suppressing the rapid energy fluctuations of natural articulation.

**Direction:** Lower energy_std in fake speech.

**H0:** No difference in energy_std between conditions.
**Falsifying observation:** Equal or higher energy variability in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 5 | Medium |

**Literature:** Lorenzo-Trueba et al. (2019, INTERSPEECH) documented energy modeling limitations in neural TTS. The energy_std metric for detection is straightforward but underexplored.

---

### H12-P: F0 maximum is artificially constrained in TTS, truncating the upper pitch range

**Precise statement:** The f0_max feature is significantly lower in spoofed speech than in bona fide speech, particularly for female speakers and expressive utterances.

**Mechanism:** TTS F0 models are typically trained within a bounded pitch range (e.g., 75-500 Hz) with clipping or sigmoid output activations that prevent extreme values. Natural speech includes involuntary pitch excursions during emphasis, surprise, or sentence-initial stress that can exceed the training distribution. These extreme F0 values are underrepresented in TTS training data (which is typically neutral read speech) and are clipped by the model architecture. The f0_max feature directly captures this upper-range truncation.

**Direction:** Lower f0_max in fake speech.

**H0:** No difference in f0_max between conditions.
**Falsifying observation:** Equal or higher f0_max in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** Sini et al. (2018, SSW) documented pitch range limitations in neural TTS. Not previously tested as a detection feature.

---

### H13-P: Duration distribution differs between real and synthetic speech due to fixed-rate synthesis

**Precise statement:** The duration feature shows significantly lower variance across utterances in spoofed speech from TTS attacks, reflecting the fixed speaking rate of synthesis systems.

**Mechanism:** TTS duration models (attention-based or explicit duration predictors) generate speech at a rate determined by training data statistics. The resulting utterance durations cluster tightly around the expected duration for a given text length. Natural speakers exhibit much more variable speaking rates due to hesitation, emphasis, breathing, and individual tempo. However, since ASVspoof 2019 LA uses fixed text prompts, the text-conditioned duration should be similar -- the hypothesis is specifically about the residual variation after controlling for content.

**Direction:** Lower inter-utterance duration variance in fake speech (for same text content).

**H0:** No difference in duration variance between conditions.
**Falsifying observation:** Equal or greater duration variance in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 2 | 4 (need to control for text content) | Small |

**Literature:** Battenberg et al. (2020, ICASSP) documented duration modeling limitations in attention-based TTS.

---

## V. Cross-Feature Hypotheses

### H12-X: The conjunction of smooth F0 + narrow formant bandwidth is a stronger deepfake indicator than either alone

**Precise statement:** A classifier using the interaction term (f0_std * f1_bw) achieves significantly higher AUC than classifiers using f0_std or f1_bw alone, indicating a synergistic discriminative effect.

**Mechanism:** Smooth F0 (low f0_std) alone could occur in natural monotone speech (e.g., reading lists, fatigue). Narrow formant bandwidth (low f1_bw) alone could occur in natural speech with clear vowels (e.g., trained speakers). However, the co-occurrence of BOTH is characteristic of synthetic speech because: (a) both arise from the same underlying cause (regression-to-mean in generative models), and (b) in natural speech, the articulatory mechanisms producing F0 variability (laryngeal tension) and formant bandwidth (vocal tract wall compliance) are partially independent. The joint feature captures a signature of synthetic generation that is unlikely in natural speech.

**Direction:** Higher AUC for interaction term than individual features.

**H0:** The interaction term does not improve classification beyond the additive effect of individual features.
**Falsifying observation:** No improvement in AUC from the interaction term.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 4 | 5 | Medium |

**Literature:** No prior work tests this specific cross-domain interaction. Feature interaction for spoofing detection was explored by Todisco et al. (2017, INTERSPEECH) but not with formant-prosody interactions.

---

### H13-X: Under codec compression, prosodic features become MORE discriminative relative to spectral features

**Precise statement:** When features are extracted from codec-compressed audio (ASVspoof 2021 DF), the relative importance (permutation importance ratio) of prosodic features (f0_std, voiced_ratio, energy_std) increases while PNCC/LFCC importance decreases, compared to uncompressed audio (ASVspoof 2019 LA).

**Mechanism:** Telephony codecs (G.711, G.722, Opus, AMR) operate by: (a) quantizing spectral coefficients, which directly degrades PNCC and LFCC features by adding quantization noise to the spectral envelope, and (b) preserving pitch and energy contours because these are separately encoded in the codec's excitation signal (via LPC residual or CELP excitation). Therefore, the spectral artifacts that distinguish real from fake in clean audio are partially masked by codec artifacts, while prosodic cues survive compression. This creates a crossover in feature importance under codec conditions.

**Direction:** Prosody feature importance increases under codec compression; spectral feature importance decreases.

**H0:** Feature importance ranking does not change between clean and codec-compressed conditions.
**Falsifying observation:** Spectral features remain equally or more important under compression.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 5 | 4 (requires parallel clean/compressed evaluation) | Large |

**Literature:** Liu et al. (2023, IEEE TASLP, ASVspoof 2021) documented codec degradation of detection systems but did not decompose by feature type. This hypothesis directly supports our ICC-based feature selection framework.

---

### H14-X: PNCC and LFCC capture complementary spectral artifacts due to different frequency warping

**Precise statement:** The correlation between PNCC-based and LFCC-based classifier scores is significantly less than 0.7 (Pearson r), indicating that the two feature groups capture non-redundant discriminative information despite both being cepstral features.

**Mechanism:** PNCC uses a gammatone (ERB-scale) filterbank that concentrates filters in the 200-2000 Hz range (where auditory resolution is highest), while LFCC uses linearly-spaced filters that give equal weight to all frequency bands. For spoofing detection, this means: PNCC is more sensitive to artifacts in the formant region (200-3000 Hz) where gammatone filters are densely packed, while LFCC is more sensitive to high-frequency artifacts (4000-8000 Hz) where it has relatively more filters than PNCC. The two feature groups thus provide complementary views of the spectral envelope, and a score-level combination should outperform either alone.

**Direction:** Correlation between PNCC-only and LFCC-only classifier scores < 0.7.

**H0:** PNCC and LFCC classifier scores are highly correlated (r > 0.7), indicating redundant information.
**Falsifying observation:** Correlation > 0.7.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 3 | 5 | Medium |

**Literature:** Sahidullah et al. (2015) compared LFCC and MFCC but not LFCC and PNCC. The complementarity hypothesis for deepfake detection is new.

---

### H15-X: Attack-specific feature signatures enable mixture-of-experts routing from acoustic features alone

**Precise statement:** A 6-class classifier (A01-A06) trained on the 36 acoustic features alone achieves above-chance accuracy (>16.7%), with distinct feature centroids for each attack, supporting the use of acoustic features for CADER expert routing.

**Mechanism:** Different TTS/VC systems have different architectural signatures that manifest in different acoustic feature groups: (a) A01 (neural waveform, WaveNet) should show smooth PNCC + high voiced_ratio; (b) A02 (vocoder-based) should show reduced high-frequency LFCC; (c) A03 (neural vocoder variant) should show narrow formant bandwidth; (d) A04-A06 (VC systems) should preserve prosodic features but distort formant relationships. If these attack-specific signatures are detectable, then the 36 acoustic features contain sufficient information for the CADER module to route inputs to specialized experts.

**Direction:** Above-chance multi-class attack identification; distinct cluster centroids in 36-dim feature space.

**H0:** Acoustic features cannot distinguish between attack types above chance.
**Falsifying observation:** Attack classification accuracy at or below chance (16.7%).

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 4 | 5 | Large |

**Literature:** Todisco et al. (2019) showed per-attack performance variation. Using acoustic features for explicit attack-type routing (as in CADER) is novel.

---

### H16-X: Formant-prosody decorrelation is a deepfake indicator because synthesis decouples articulatory subsystems

**Precise statement:** The absolute Pearson correlation between formant features (f1_mean, f2_mean) and prosodic features (f0_mean, energy_mean) is significantly lower in spoofed speech than in bona fide speech.

**Mechanism:** In natural speech, formant frequencies and F0 are partially correlated because both are influenced by laryngeal-supralaryngeal coupling: increased subglottal pressure (raising F0) also tenses the vocal tract walls (shifting formants), and jaw opening (affecting F1) co-varies with pitch gestures in intonation (Hoole & Honda, 2011). TTS systems model F0 and spectral envelopes with separate modules (pitch predictor + acoustic model), breaking this natural coupling. Voice conversion systems may preserve or distort this correlation depending on whether F0 and formants are transformed independently. The decorrelation between formant and prosodic features thus serves as an indicator of the modular generation process.

**Direction:** Lower |correlation(formants, prosody)| in fake speech.

**H0:** No difference in formant-prosody correlation between conditions.
**Falsifying observation:** Equal or higher correlation in synthetic speech.

| Novelty | Testability | Effect size |
|---------|-------------|-------------|
| 5 | 4 (requires per-utterance correlation, not just means) | Medium |

**Literature:** Hoole & Honda (2011) documented laryngeal-supralaryngeal coupling in natural speech. Application to deepfake detection via decorrelation analysis is entirely novel.

---

## VI. Hypothesis Ranking (Composite Score)

Composite score = Novelty * 0.25 + Testability * 0.35 + Effect_map * 0.40
(Effect: Small=1, Medium=3, Large=5)

| Rank | ID | Hypothesis (short) | Nov | Test | Effect | Score |
|------|-----|---------------------|-----|------|--------|-------|
| 1 | H13-X | Prosody gains importance under codec compression | 5 | 4 | Large | 4.65 |
| 2 | H15-X | Attack-specific acoustic signatures for MoE routing | 4 | 5 | Large | 4.75 |
| 3 | H1 | PNCC spectral smoothness in TTS | 2 | 5 | Large | 4.25 |
| 4 | H12-X | F0 x formant bandwidth interaction | 4 | 5 | Medium | 3.95 |
| 5 | H16-X | Formant-prosody decorrelation in fakes | 5 | 4 | Medium | 3.85 |
| 6 | H7-F | Narrow formant bandwidth in TTS | 4 | 5 | Medium | 3.95 |
| 7 | H10-P | Elevated voiced ratio in TTS | 3 | 5 | Medium | 3.70 |
| 8 | H9-P | Lower F0 std in TTS | 2 | 5 | Large | 4.25 |
| 9 | H4-LFCC | Reduced high-freq energy in fakes | 2 | 5 | Large | 4.25 |
| 10 | H2 | Power-law amplification of vocoder artifacts | 4 | 4 | Medium | 3.60 |
| 11 | H3 | Higher-order PNCC more discriminative | 3 | 5 | Medium | 3.70 |
| 12 | H14-X | PNCC-LFCC complementarity | 3 | 5 | Medium | 3.70 |
| 13 | H6-F | Smooth formant transitions in TTS | 3 | 5 | Medium | 3.70 |
| 14 | H8-F | Compressed vowel space (f1_f2_ratio) | 3 | 5 | Medium | 3.70 |
| 15 | H6-LFCC | LFCC coeff 1 captures HNR differences | 3 | 5 | Medium | 3.70 |
| 16 | H11-P | Lower energy_std in TTS | 2 | 5 | Medium | 3.50 |
| 17 | H7-LFCC | VC closer to bona fide in LFCC space | 2 | 5 | Large | 4.25 |
| 18 | H10-F | Formants better at detecting TTS than VC | 2 | 5 | Large | 4.25 |
| 19 | H5-LFCC | LFCC frame-boundary ripple | 3 | 3 | Medium | 2.70 |
| 20 | H9-F | Formant dispersion artifacts | 4 | 5 | Small | 3.15 |
| 21 | H4 | PNCC temporal asymmetry | 4 | 3 | Small | 2.45 |
| 22 | H12-P | F0 max truncation | 3 | 5 | Medium | 3.70 |
| 23 | H5 | PNCC noise floor artifacts | 5 | 3 | Small | 2.70 |
| 24 | H13-P | Duration variance in TTS | 2 | 4 | Small | 2.30 |

---

## VII. Top 3 Recommended for Immediate Testing

### Priority 1: H13-X (Codec-Prosody Importance Crossover)

**Why:** Directly supports the core thesis of the paper (ICC-based codec-robust feature selection). Testable by comparing permutation importance on ASVspoof 2019 LA eval vs. 2021 DF eval. If confirmed, provides strong justification for the CADER module's acoustic-conditioned routing.

**Test:** Train identical classifiers on 2019 LA and 2021 DF. Compare feature group importance rankings. Compute the ratio (prosody_importance / spectral_importance) for each condition.

### Priority 2: H15-X (Attack-Specific Acoustic Signatures)

**Why:** Directly validates the CADER architecture premise. If acoustic features can distinguish attack types, the expert routing is principled. If not, CADER needs redesign.

**Test:** Train a 6-class (A01-A06) random forest on the 36 acoustic features. Report per-class precision/recall and confusion matrix. Visualize attack centroids via t-SNE.

### Priority 3: H16-X (Formant-Prosody Decorrelation)

**Why:** Highest novelty score. If confirmed, represents a genuinely new finding with theoretical grounding in articulatory phonetics. Directly motivates the cross-modal attention design in AC-CMA.

**Test:** Compute Pearson correlation matrices (formant features vs. prosody features) separately for bona fide and spoofed subsets. Test for significant differences using Fisher z-transformation.

---

## VIII. Falsifiability Summary

| Hypothesis | H0 (Null) | Falsifying Observation | Alpha |
|------------|-----------|----------------------|-------|
| H1 | PNCC variance(real) = PNCC variance(fake) | No significant difference or opposite direction | 0.05 |
| H2 | PNCC discriminability <= LFCC discriminability | LFCC Cohen's d >= PNCC Cohen's d | 0.05 |
| H3 | AUC(coeff 7-12) <= AUC(coeff 0-6) | Lower-order coefficients more discriminative | 0.05 |
| H4 | Skewness(real) = Skewness(fake) | Equal or greater skewness in fakes | 0.05 |
| H5 | Floor ratio(real) = Floor ratio(fake) | No difference | 0.05 |
| H4-LFCC | High-freq LFCC importance = Low-freq LFCC importance | Equal importance | 0.05 |
| H5-LFCC | LFCC variance(fake) <= LFCC variance(real) | Lower LFCC variance in fakes | 0.05 |
| H6-LFCC | LFCC coeff 1(real) = LFCC coeff 1(fake) | No difference | 0.05 |
| H7-LFCC | Mahalanobis(VC, real) >= Mahalanobis(TTS, real) | VC further from real than TTS | 0.05 |
| H6-F | f1_std(real) = f1_std(fake) | Equal or higher variability in fakes | 0.05 |
| H7-F | f1_bw(real) = f1_bw(fake) | Equal or wider bandwidth in fakes | 0.05 |
| H8-F | Var(f1_f2_ratio, real) = Var(f1_f2_ratio, fake) | Equal or greater variance in fakes | 0.05 |
| H9-F | formant_dispersion(real) = formant_dispersion(fake) | No difference | 0.05 |
| H10-F | AUC_formant(TTS) = AUC_formant(VC) | Equal detection rates | 0.05 |
| H9-P | f0_std(real) = f0_std(fake) | Equal or higher F0 variability in fakes | 0.05 |
| H10-P | voiced_ratio(real) = voiced_ratio(fake) | Equal or lower voiced ratio in fakes | 0.05 |
| H11-P | energy_std(real) = energy_std(fake) | Equal or higher energy variability in fakes | 0.05 |
| H12-P | f0_max(real) = f0_max(fake) | Equal or higher F0 max in fakes | 0.05 |
| H13-P | Var(duration, real) = Var(duration, fake) | Equal variance | 0.05 |
| H12-X | AUC(interaction) <= max(AUC(f0_std), AUC(f1_bw)) | No improvement from interaction | 0.05 |
| H13-X | Importance ratio unchanged across codec conditions | Same ranking clean vs. compressed | 0.05 |
| H14-X | r(PNCC_score, LFCC_score) >= 0.7 | High correlation | 0.05 |
| H15-X | Attack classification accuracy <= 16.7% (chance) | At or below chance | 0.05 |
| H16-X | |r(formant, prosody)| real = |r(formant, prosody)| fake | Equal correlation | 0.05 |

---

## References

1. Kim, C. & Stern, R.M. (2016). "Power-Normalized Cepstral Coefficients (PNCC) for Robust Speech Recognition." IEEE/ACM Trans. Audio, Speech, Language Process., 24(7):1315-1329. DOI: 10.1109/TASLP.2016.2545928
2. Patterson, R.D., Robinson, K., Holdsworth, J., McKeown, D., Zhang, C., & Allerhand, M. (1992). "Complex sounds and auditory images." Auditory Physiology and Perception, 83:429-446.
3. Sahidullah, M., Kinnunen, T., & Hanilci, C. (2015). "A Comparison of Features for Synthetic Speech Detection." INTERSPEECH, 2087-2091.
4. Wang, X. & Yamagishi, J. (2021). "A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection." INTERSPEECH.
5. Lindblom, B. (1963). "Spectrographic study of vowel reduction." JASA, 35(11):1773-1781.
6. Ren, Y., Ruan, Y., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T.Y. (2019). "FastSpeech: Fast, Robust and Controllable Text to Speech." NeurIPS.
7. De Leon, P.L., Pucher, M., Yamagishi, J., Hernaez, I., & Saratxaga, I. (2012). "Evaluation of Speaker Verification Security and Detection of HMM-Based Synthetic Speech." IEEE Trans. Audio, Speech, Language Process., 20(8):2280-2290.
8. Todisco, M., Wang, X., Vestman, V., Sahidullah, M., Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., & Lee, K.A. (2019). "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection." INTERSPEECH.
9. Fitch, W.T. (1997). "Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques." JASA, 102(2):1213-1222.
10. Hoole, P. & Honda, K. (2011). "Automaticity vs. feature-enhancement in the control of segmental F0." In Prosodies, Mouton de Gruyter.
11. Liu, X., et al. (2023). "ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild." IEEE/ACM Trans. Audio, Speech, Language Process., 31:2507-2522.
12. Fedus, W., Zoph, B., & Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR, 23(120):1-39.
13. Drugman, T., Thomas, M., Gudnason, J., Naylor, P., & Dutoit, T. (2012). "Detection of Glottal Closure Instants from Speech Signals: A Quantitative Review." IEEE Trans. Audio, Speech, Language Process., 20(3):994-1006.
14. Yi, J., Fu, R., Tao, J., Nie, S., Ma, H., Wang, C., Wang, T., Tian, Z., Bai, Y., Fan, C., et al. (2020). "Applying BERT to Analyze Emotion of TTS." arXiv:2005.11882.
15. Stevens, K.N. (1998). "Acoustic Phonetics." MIT Press.
16. Wester, M., Wu, Z., & Yamagishi, J. (2016). "Analysis of the Voice Conversion Challenge 2016." INTERSPEECH.
17. Wu, Z., Evans, N., Kinnunen, T., Yamagishi, J., Alegre, F., & Li, H. (2015). "Spoofing and countermeasures for speaker verification." Computer Speech & Language, 30(1):130-153.
18. Lindblom, B. (1990). "Explaining phonetic variation: a sketch of the H&H theory." In Speech Production and Speech Modelling, Springer.
