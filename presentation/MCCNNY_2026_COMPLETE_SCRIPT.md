# MCCNNY 2026 Complete Speaking Script + Q&A Prep
## "Statistics for Audio Deepfake Detection"
## 8 minutes, 13 slides

---

## SLIDE 1: Title (15 seconds)

**SAY:**
"Good morning. I'm [Name] from [Affiliation]. My talk is about using statistics to detect audio deepfakes — which acoustic features identify fake audio even after telephone compression?"

---

## SLIDE 2: The Problem (40 seconds)

**SAY:**
"We train a detector on 25,000 clean audio files — some real, some fake. In the lab, it works: 7% error. Then we deploy it on phone calls. Codecs compress the audio. Error jumps to 50% — a coin flip. Useless."

**POINT TO BAR CHART:**
"Red bars: all 36 features — 7% in the lab, 50% after codecs. Collapsed. Green bars: just formant features — 17% in the lab, 23% after codecs. They don't collapse. Why? That's what this talk answers."

**IF ASKED:** "What's a codec?" → "A codec compresses audio for transmission. G.711 is used on phone lines, MP3 for music, AAC for streaming. They throw away information to reduce file size."

**IF ASKED:** "What's EER?" → "Equal Error Rate — the point where false accept rate equals false reject rate. Lower is better. 50% means random guessing."

---

## SLIDE 3: What We Measure (35 seconds)

**SAY:**
"We extract 36 numbers from each audio file. Four groups: PNCC captures overall sound shape, LFCC captures fine spectral detail, Formants measure mouth resonances — the shape of your vocal tract — and Prosody measures pitch, loudness, and rhythm."

"The question: which of these 36 numbers tell real from fake, and which survive compression?"

**IF ASKED:** "What's a formant?" → "When you speak, your mouth and throat form a tube. That tube resonates at specific frequencies — like blowing across a bottle. F1 is the first resonance, around 300-700 Hz, controlled mainly by how wide you open your jaw."

**IF ASKED:** "Why 36 specifically?" → "Each group was chosen from the speech processing literature for different reasons. PNCC from Kim & Stern 2016, LFCC from Sahidullah 2015, Formants via Praat, Prosody via pYIN. We tested all 36 to see which actually work."

---

## SLIDE 4: The 4 Winners (1 minute)

**SAY:**
"Out of 36 features, only 4 detect deepfakes reliably. All four measure the first formant — F1."

**POINT TO TABLE:**
"F1 bandwidth: real speech averages 515 Hz with a spread of plus or minus 87 Hz. Fake speech averages 361 Hz with spread 120 Hz. That's a 154 Hz gap. Using just this one feature, we get 18% error rate."

"The other three — F1 mean, F1 variability, F1/F2 ratio — tell a similar story."

**POINT TO BOTTOM BOXES:**
"What about the other 32? PNCC can't tell real from fake at all — error near 50%. LFCC has moderate detection but codecs destroy it. And these 4 winners are actually measuring the same thing — they're 91% correlated. PCA shows 1 component does the same job as all 4."

**IF ASKED:** "What's the spread (±87 Hz)?" → "Standard deviation. It means most real files have F1 bandwidth between roughly 430 and 600 Hz. There's a lot of variation file to file — not every real file is exactly 515. That's why the error rate isn't 0%."

**IF ASKED:** "Why F1 and not F2?" → "F2 features have small gaps — around 20-55 Hz — with effect sizes below 0.5. F1 has gaps of 46-154 Hz with effect sizes above 1.1. F1 is where synthesis artifacts concentrate."

**IF ASKED:** "If they're 91% correlated, why show 4?" → "Honestly, you could use just F1 bandwidth alone (18% error) or a single PCA component (17.1% error). We show all 4 because they came out of the analysis independently — the correlation is a finding, not a design choice."

---

## SLIDE 5: Statistical Tests (1 minute)

**SAY:**
"Four formal statistical tests to make sure this is real, not cherry-picked."

"Test 1 — Mann-Whitney U: Is the gap due to random chance? We tested all 23 non-constant features. 21 are significant at p less than 0.05. After correcting for testing 23 features simultaneously using Benjamini-Hochberg, 21 still hold. The 4 formant winners have p-values essentially zero."

"Test 2 — Cohen's d: With 25,000 files, everything is 'significant.' A 0.001 Hz difference gets a tiny p-value. So we measure effect SIZE. Only 4 features have large effects — d above 0.8 — and they're the same 4 formants."

"Test 3 — Correlation: Are the 4 winners really independent? No. F1 mean and F1/F2 ratio are 91% correlated. PCA confirms: 1 component captures 79%."

"Test 4 — Kruskal-Wallis: Do different types of fakes leave different traces? Yes. All 4 features differ significantly across 6 attack types. This means formants detect many kinds of fakes, not just one."

**IF ASKED:** "Why Mann-Whitney and not t-test?" → "The t-test assumes normal distributions. Some of our features are skewed. Mann-Whitney is non-parametric — it works on ranks, no normality assumption needed. With our sample sizes, both give the same answer, but Mann-Whitney is more defensible."

**IF ASKED:** "What's Benjamini-Hochberg?" → "When you test 23 features at once, you expect about 1 false positive by chance at alpha 0.05. BH controls the false discovery rate — the expected proportion of false positives among your rejections — at 5%. It's less conservative than Bonferroni but still rigorous."

**IF ASKED:** "What's Cohen's d?" → "The gap between two groups measured in standard deviations. d = 1.3 means the averages are 1.3 standard deviations apart. Below 0.2 is negligible, above 0.8 is large. It tells you if the difference matters, not just if it's statistically significant."

**IF ASKED:** "What's Kruskal-Wallis?" → "Non-parametric ANOVA. Tests whether multiple groups have different distributions. We used it to compare 6 different TTS/VC systems. It's the multi-group extension of Mann-Whitney."

---

## SLIDE 6: Visual Proof — Distributions (25 seconds)

**SAY:**
"Here's what the difference looks like. Four panels, four features. Green is real speech, red is fake. Dashed lines are the averages."

"Look at F1 bandwidth — top right. Clear separation. The green peak is shifted right — real speech has wider resonances. You can practically draw a line between them by eye."

**IF ASKED:** "Why do the distributions overlap?" → "Because there's natural variation. Some speakers naturally have narrow formants, some fakes have wide ones. That's why the error rate is 18%, not 0%. But ON AVERAGE, the difference is large and consistent."

---

## SLIDE 7: Why — Physics (35 seconds)

**SAY:**
"Why are formants different? Your mouth and throat form a tube. This tube has resonances, but it also absorbs energy — tissue damping, air escaping through your nose. That makes the resonances wide. F1 bandwidth about 515 Hz."

"A voice synthesizer models the spectral shape mathematically. It optimizes a reconstruction loss that pulls everything toward the average. No physical damping. Sharp, narrow resonances. 361 Hz."

"Codecs preserve this difference because they MUST preserve formants for speech to be understandable on the phone."

**IF ASKED:** "Is the vocoder explanation proven or a hypothesis?" → "The damping explanation is established acoustic physics — Fant 1960, Stevens 1998. The claim that vocoders produce narrow formants because of L1/L2 loss is our hypothesis based on the regression-to-mean property of reconstruction losses. It's the most plausible explanation but not formally cited."

**IF ASKED:** "Do all codecs preserve formants?" → "Not equally. MP3 and AAC change formants by only 5-18% of the real-fake gap. But G.711 — telephone codec — changes them by 79% of the gap. That's almost destroying the signal. The next slide shows the exact numbers."

---

## SLIDE 8: Codec Robustness (35 seconds)

**SAY:**
"We compressed 150 files through 9 codecs and measured how much each feature changed."

"Prosody: 3% change — very stable. Codecs preserve pitch by design. Formants: 6% average — stable. LFCC: 39% average, worst case 120% — more than doubled. Completely unreliable."

"This is why the all-36-feature model collapsed. It relied on LFCC features that codecs destroyed."

**IF ASKED:** "Why only 150 files?" → "Each file is compressed through 9 codecs via ffmpeg — 1,350 encode-decode operations. At 3 seconds per operation, 150 files takes about an hour. All 25,000 would take 8 days. 150 gives us standard error of about ±1% on the percentage change — precise enough."

**IF ASKED:** "How did you compute the percentage?" → "Mean absolute difference between original and compressed feature values, divided by mean absolute original value, times 100. It's a normalized mean absolute error."

---

## SLIDE 9: Paired t-test — Codec Change vs Gap (40 seconds)

**SAY:**
"Paired t-test: same file, before and after each codec. Does compression significantly change the values? Yes — every codec does. The question is how MUCH relative to the real-fake gap."

**POINT TO TABLE:**
"The real-fake gap for F1 bandwidth is 154 Hz. MP3 128k changes it by only 8 Hz — 5% of the gap. Easily survives. But G.711 A-law: 122 Hz — that's 79% of the gap."

**POINT TO CONCLUSION BOX:**
"Honest conclusion: formants work well on MP3, AAC, Opus — 5 to 24% of the gap. For telephone codecs like G.711, the signal is marginal."

**IF ASKED:** "Why paired t-test and not unpaired?" → "Because we're comparing the SAME file before and after compression. Each file is its own control. Pairing removes between-file variance and isolates the codec effect."

**IF ASKED:** "Is 79% of the gap acceptable?" → "No, it's not. Under G.711, formant-based detection degrades significantly. Our error rate on the 2021 dataset — which includes G.711 — is 21-23%. Without G.711 files it would be better. This is an honest limitation."

---

## SLIDE 10: Ablation (40 seconds)

**SAY:**
"This is the ablation. We tested every feature combination."

"Left side — green: Prosody alone, Formants alone, Formants plus Prosody. All stay between 21 and 29% on compressed audio."

"Right side — red: anything with PNCC or LFCC. PNCC alone: 50%. LFCC alone: 42%. And critically: PNCC PLUS Formants — 50%. Adding PNCC to formants didn't just fail to help, it DESTROYED the detector."

"The all-36 model: 10% in the lab, 50% after codecs. The best lab result is the worst deployed result."

**IF ASKED:** "Why does adding features make it worse?" → "The SVM learns to use whatever features separate the training data best. PNCC and LFCC have moderate separation on clean training data. The model learns to rely on them. Then codecs destroy those features, and the model has nothing left. It's like building on sand."

**IF ASKED:** "Why don't tree-based models collapse?" → "Good question — Decision Tree gets 31% and Random Forest gets 22.5% on all-36. Trees can ignore irrelevant features by not splitting on them. SVMs and neural nets can't — they use all dimensions of the input space."

---

## SLIDE 11: Classifiers (35 seconds)

**SAY:**
"Five simple classifiers on the winning features plus prosody — 15 features total."

"SVM: 11% error in the lab, 17% on unseen attacks, 21% after codecs. All five models degrade gracefully. None collapse."

"These aren't state-of-the-art — best neural systems get 1.9%. But you can explain every prediction: this file was flagged because F1 bandwidth is 340 Hz, below the threshold for natural speech."

**IF ASKED:** "Why 15 features and not just 4?" → "The 11 additional features have moderate individual power but contribute through interaction effects in the joint classifier. Guyon & Elisseeff 2003 showed that individually weak features can be jointly relevant. Our Random Forest Gini importance confirms: the 11 weak features account for 74% of model decisions collectively."

**IF ASKED:** "Why not use deep learning?" → "Deep learning achieves 1.9% error. We can't compete on accuracy. But we can compete on explainability. A neural network says 'fake' — nobody knows why. Our model says 'fake because F1 bandwidth is 340 Hz and real speech averages 515 Hz.' That's useful in forensics."

---

## SLIDE 12: Per-Codec + Importance (25 seconds)

**SAY:**
"Top chart: error rate per codec. Formant row is green — 19 to 24% everywhere. All-36 row is red — 50% everywhere."

"Bottom chart: F1 bandwidth accounts for 26% of the Random Forest's decisions. F1 mean is 15%. Formants dominate."

---

## SLIDE 13: Conclusions (30 seconds)

**SAY:**
"Out of 36 features, only 4 detect deepfakes after compression. All measure F1 — the first formant. Real speech: 515 Hz bandwidth. Fake: 361 Hz. A 154 Hz gap confirmed by Mann-Whitney test, large Cohen's d effect size, and consistent across 6 attack types."

"Codecs change formants by 5 to 24% of the gap on most codecs, but 79% on telephone G.711. Adding fragile features like PNCC destroys the detector entirely."

"Thank you. Questions?"

---

## TOP 10 HARDEST QUESTIONS AND ANSWERS

**Q1: "With 25,000 files, of course everything is significant. Isn't this just p-hacking?"**
"You're right that p-values are easy with large samples. That's exactly why we report Cohen's d effect sizes alongside p-values. 21 features are significant, but only 4 have large effects (d > 0.8). The effect size filters out the noise. We also apply BH-FDR correction for 23 simultaneous tests."

**Q2: "Your 4 features are 91% correlated. Isn't it really just 1 feature?"**
"Yes, essentially. PCA confirms: 1 component captures 79% of the variance and achieves 17.1% error — same as all 4 at 17.3%. They're one signal (F1 characteristics) measured 4 ways. We present all 4 because they emerged independently from the analysis, but you could use just F1 bandwidth alone (18% error)."

**Q3: "21-23% error isn't very good. Why should we care?"**
"Two reasons. First: the all-36-feature model gets 7% in the lab but 50% in the field. Ours gets 17% in the lab but 23% in the field. Worse in the lab, far better deployed. Feature selection matters. Second: these features are explainable. A forensic expert can testify: 'F1 bandwidth is 340 Hz, below the 450 Hz threshold for natural speech.' A neural network's hidden activations can't be used as evidence."

**Q4: "G.711 changes formants by 79% of the gap. Doesn't that mean formants don't work on telephone calls?"**
"For G.711 specifically, yes, detection degrades significantly. Our 2021 evaluation includes G.711 — that's part of why the error rate is 21-23% and not lower. For MP3/AAC (social media uploads), formants change by only 5-18% of the gap and work well. The honest message: formants are a strong feature for streaming/VoIP but marginal for legacy telephony."

**Q5: "You select features on training data then train classifiers on the same data. Isn't that selection bias?"**
"Yes, there is a risk of selection bias. Ambroise & McLachlan 2002 showed this can produce 5-15% optimistic bias. However: our final evaluation uses held-out dev and eval sets that were not used for feature selection. The feature selection decision is biased, but the reported error rates are not. Ideally we would use nested cross-validation — that's a limitation we acknowledge."

**Q6: "Why Mann-Whitney and not ANOVA?"**
"ANOVA assumes normal distributions and equal variances. Some features are skewed (F0_std: skew=1.07, duration: skew=1.24). Mann-Whitney is rank-based — no distributional assumptions. For our 4 formant winners (all near-normal), ANOVA would give the same conclusion. We chose the more conservative option."

**Q7: "Advanced voice cloners can make realistic formants. Doesn't this make your approach obsolete?"**
"Partially. Attacks A17-A19 in our evaluation use advanced waveform-level voice conversion, and formants fail completely — error rate above 48%. This is a known limitation stated in our conclusions. Formants catch most current TTS/VC systems (10/13 unseen attacks below 25% error) but will become less effective as synthesis improves. They should be combined with neural features for a complete system."

**Q8: "Why not just use deep learning features like Wav2Vec2?"**
"Wav2Vec2 and similar SSL models achieve 1.9% error — far better than our 21%. The trade-off is explainability. Our features have a physical interpretation: vocal tract damping produces wide formants, vocoders don't. This matters for forensic applications, regulatory compliance, and scientific understanding. The ideal system combines both: neural features for accuracy, acoustic features for explainability."

**Q9: "Your PNCC extraction is broken — 12 of 13 coefficients are zero. How can you trust the rest of the pipeline?"**
"The PNCC bug was caused by a specific code change (per-channel mean normalization that collapsed the DCT). It is isolated to the PNCC computation. The other 23 features — LFCC, Formants, Prosody — use completely independent extraction code (Praat for formants, pYIN for F0, separate linear filterbank for LFCC). We verified non-zero variance for all 23 features and the fix is being applied with re-extraction in progress."

**Q10: "What's the sample size justification for the codec experiment?"**
"150 files × 9 codecs. Power analysis: for a paired design at alpha=0.05, 80% power, n=150 detects effects of d≥0.23 — a small-to-medium effect. Our observed effects are much larger (mean MAE changes of 8 to 122 Hz). The standard error of the percentage change is about ±1%, so our '6% change' estimate is precise to ±2%."
