# MCCNNY 2026 — Conference Registration Abstract

**Title:** Statistics for Audio Deepfake Detection

**Author:** [Your Name]
**Affiliation:** [Your Institution]

---

### Abstract

Audio deepfake detection aims to distinguish synthetic speech from genuine human speech. As text-to-speech and voice conversion systems become increasingly realistic, robust detection methods are essential. This work investigates the statistical properties of acoustic features extracted from speech signals and their stability under real-world audio compression codecs. We extract a 36-dimensional feature set comprising power-normalized cepstral coefficients, linear frequency cepstral coefficients, vocal tract formant statistics, and prosodic measures from the ASVspoof 2019 and 2021 challenge datasets. We then evaluate codec robustness using Pearson and Spearman correlation, intraclass correlation coefficients, effect size analysis, and nonparametric hypothesis testing across nine audio codecs. A PAC-Bayesian generalization bound is derived to provide a theoretically grounded criterion for feature selection under domain shift. Preliminary results identify which acoustic feature groups remain statistically stable after codec compression and which degrade, informing principled feature selection for deployable deepfake detection systems.

**Keywords:** statistical analysis, audio deepfake detection, codec robustness, PAC-Bayesian bounds, acoustic features
