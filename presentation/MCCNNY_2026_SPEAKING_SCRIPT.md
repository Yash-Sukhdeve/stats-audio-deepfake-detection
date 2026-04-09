# MCCNNY 2026 Speaking Script
## "Statistics for Audio Deepfake Detection"
## 8 minutes total

---

## SLIDE 1: Title (30 seconds)

> "Good morning. I'm [Name] from [Affiliation]. My talk is about using statistics — not deep learning, not neural networks — just classical statistics to detect audio deepfakes, even after the audio has been compressed by telephone codecs."

> "The question is simple: which acoustic features can tell us if an audio clip is fake, regardless of how it was compressed?"

*[Click to next slide]*

---

## SLIDE 2: The Problem (1 minute)

> "Here's the setup. We train a deepfake detector on clean studio audio — the ASVspoof 2019 dataset, about 25,000 files. In the lab, it works. Seven percent equal error rate."

> "But then we deploy it in the real world. Phone calls. VoIP. Social media. The audio gets compressed by codecs — G.711, MP3, Opus, AAC. And the detector collapses. Fifty percent error rate. Random chance. It's useless."

*[Pause — let that sink in]*

> "Why? Because the features the model relied on got destroyed by the codec. So the question becomes: which features survive compression AND can still detect fakes? That's what this work answers."

*[Click to next slide]*

---

## SLIDE 3: Two Requirements (45 seconds)

> "A good feature must pass two tests."

> "First — discriminative. Does it actually differ between real and fake audio? We measure this with AUC and Cohen's d. If the feature looks the same in both, it's useless no matter how robust it is."

> "Second — codec-robust. Does it survive compression? We measure this with ICC, the Intraclass Correlation Coefficient. We take 150 audio files, compress each through 9 different codecs, and ask: does the feature stay the same? ICC of 0.75 or above means yes."

> "A feature must pass BOTH tests. Discriminative but fragile? Works in the lab, fails in the field. Robust but not discriminative? Reliable but useless. We need the intersection."

*[Click to next slide]*

---

## SLIDE 4: The Answer — Formants Win (1.5 minutes)

> "Here's what we found."

*[Point to table]*

> "The top four features are all formants — vocal tract resonance frequencies. F1 bandwidth has an AUC of 0.85 and ICC of 0.985. F1 mean, F2 bandwidth, F1 standard deviation — all above 0.82 AUC and 0.985 ICC."

> "Now look at the surprise."

*[Point to PNCC row]*

> "PNCC — power-normalized cepstral coefficients — has the HIGHEST ICC of any feature group. 0.993. Nearly perfect codec robustness. But its AUC is around 0.5. It cannot tell real from fake. All that reliability is wasted."

> "And when we trained a classifier on all 36 features — including PNCC and LFCC — it achieved 7 percent EER on clean audio. Impressive. But on codec-compressed audio? It collapsed to 50 percent. Random chance."

> "The formant-only model? Seventeen percent on clean, twenty-three percent on compressed. Not as good in the lab, but it HOLDS UP in the field."

*[Click to next slide]*

---

## SLIDE 5: Why Formants Work (1 minute)

> "Why do formants detect deepfakes? The answer is physics."

> "Real speech is produced by a physical vocal tract — a tube from your vocal folds to your lips. This tube has resonances called formants. And crucially, it has DAMPING — from tissue absorption, nasal coupling, radiation losses. This damping makes formant bandwidths wide. About 515 hertz for F1."

> "Synthetic speech is produced by a vocoder — a neural network trained to minimize reconstruction loss. L1 or L2 loss regresses toward the conditional mean. This produces spectrally SHARP resonances. Unnaturally narrow formant bandwidths. About 361 hertz for F1."

> "A 154-hertz difference. Cohen's d of 1.32. A large effect."

> "And why do codecs preserve this? Because codecs are DESIGNED to preserve formants. Formants carry speech intelligibility. If a codec destroyed formants, you couldn't understand the phone call. So the real-versus-fake difference survives every codec we tested."

*[Click to next slide]*

---

## SLIDE 6: Detection Results (1 minute)

> "Let me show you the actual numbers."

*[Point to table]*

> "We train on ASVspoof 2019. We test on three progressively harder conditions. Known attacks, unseen attacks, and codec-compressed audio from 2021."

> "Formants plus prosody with a simple SVM: twelve percent on known attacks, eighteen percent on unseen, twenty-three on compressed. Graceful degradation."

> "All 36 features with an MLP: seven percent on known attacks — best in the lab. But on compressed audio — fifty percent. Collapsed."

*[Point to Key Finding box]*

> "ICC predicted this. Features with ICC above 0.75 held up. Features below that threshold destroyed the system."

*[Point to Limitations box]*

> "I want to be honest. Twenty-three percent EER is not competitive with state-of-the-art SSL models at two percent. And advanced vocoders like HiFi-GAN can produce realistic formants — our features fail on those attacks. These are interpretable baselines, not a replacement for neural systems."

*[Click to next slide]*

---

## SLIDE 7: Statistical Methods (1 minute)

> "A quick summary of the statistical toolkit."

> "ICC — Intraclass Correlation Coefficient — measures whether a feature gives consistent values across all 9 codec conditions for the same audio file. Same file, 9 compressions, are the values consistent? That's ICC."

> "Mann-Whitney U for the discriminative test — comparing bonafide versus spoof distributions. With Benjamini-Hochberg correction for 36 simultaneous comparisons."

> "Cohen's d for effect size — not just 'is it different?' but 'how MUCH is it different?' F1 mean has d equals 1.48. That's a large effect."

> "AUC-ROC treats each feature as a standalone classifier. And Fisher z-transformation for confidence intervals on correlations near one, where the sampling distribution is skewed."

> "Every method is classical. No neural networks. No black boxes. Every result is traceable and reproducible."

*[Click to next slide]*

---

## SLIDE 8: Conclusions (30 seconds)

> "To summarize."

> "Formant features — specifically F1 bandwidth, F1 mean, and F2 bandwidth — are the strongest codec-robust indicators of synthetic speech that we found."

> "The physical reason: real speech has wider formant bandwidths because physical vocal tracts have damping that vocoders don't model. And this difference survives codec compression."

> "The practical lesson: validating features with ICC BEFORE building a detector prevents the catastrophic collapse we saw with unvalidated features."

*[Pause]*

> "Thank you. I'm happy to take questions."

---

## ANTICIPATED Q&A (prepare these)

**Q: "Why not just use deep learning?"**
> "Deep learning achieves 2% EER on this task. Our 23% can't compete. But our features are interpretable — you can explain WHY something is detected as fake. 'The formant bandwidth is 361 Hz, below the 450 Hz threshold for natural speech.' That's explainable. A neural network's hidden layer activation is not."

**Q: "What about advanced vocoders that produce realistic formants?"**
> "You're right — attacks A17-A19 in our evaluation defeat formant features entirely, with EER above 45%. These are waveform-level voice conversion and advanced vocoder systems. Formant features are a reliable floor, not a ceiling. They should be combined with SSL features for a complete system."

**Q: "Why ICC and not Pearson correlation?"**
> "Pearson r measures linear association only. A codec that shifts all F1 values up by 100 Hz gets Pearson r equals 1 — perfect correlation. But the absolute values are wrong, and your classifier would fail. ICC penalizes systematic bias. In that example, ICC would be about 0.34. It catches what Pearson misses."

**Q: "Is 150 files enough for ICC?"**
> "Power analysis shows that for detecting ICC at the 0.90 level with 80% power, you need only 7 subjects. With 150 files and 9 raters, we have power exceeding 0.99. The protocol recommended 200; we used 150, which is well-powered."

**Q: "You said 'all 36 features collapsed.' Why?"**
> "PNCC and LFCC features have high AUC on clean audio but low ICC — they're discriminative in the lab but destroyed by codecs. When you include them, the classifier learns to rely on them. Then codecs remove those features, and the classifier has nothing left. The formant-only model never learned to rely on fragile features, so it doesn't collapse."

---

## TIMING GUIDE

| Slide | Topic | Time | Cumulative |
|-------|-------|------|------------|
| 1 | Title | 0:30 | 0:30 |
| 2 | The Problem | 1:00 | 1:30 |
| 3 | Two Requirements | 0:45 | 2:15 |
| 4 | Formants Win | 1:30 | 3:45 |
| 5 | Why Formants Work | 1:00 | 4:45 |
| 6 | Detection Results | 1:00 | 5:45 |
| 7 | Statistical Methods | 1:00 | 6:45 |
| 8 | Conclusions | 0:30 | 7:15 |
| | Buffer for pace | 0:45 | 8:00 |

## PRACTICE TIPS

1. **Rehearse 3 times.** First time reading, second time from memory, third time with a timer.
2. **The key moment is Slide 4** — the surprise that PNCC is useless despite being the most robust. Pause after saying "all that reliability is wasted."
3. **The physics on Slide 5** is your strongest material for a math audience. They will appreciate the physical mechanism.
4. **On Slide 6, own the limitations.** Saying "23% is not competitive" builds credibility.
5. **If you're running long,** skip Slide 7 (methods). The audience can ask about methods in Q&A.
6. **End with the one-sentence summary.** Memorize it.
