# Search Strategy: Codec-Robust Feature Integration for Deepfake Detection

**Date:** 2026-04-06
**Reviewer:** Automated (single pass, ASSISTANT mode)

## Research Question

After identifying codec-robust acoustic features (ICC >= 0.75), what approaches do researchers use to integrate those features into deepfake detection systems?

## Databases

| Database | API Endpoint | Search Date |
|----------|-------------|-------------|
| OpenAlex | api.openalex.org/works | 2026-04-06 |

## Search Queries Executed

### Query 1: Feature fusion + SSL + deepfake
```
default.search: "codec robust acoustic features deepfake detection fusion SSL embeddings"
filter: from_publication_date:2020-01-01, to_publication_date:2026-12-31
```

### Query 2: Feature fusion handcrafted + SSL
```
default.search: "feature fusion self-supervised learning handcrafted features audio anti-spoofing"
filter: from_publication_date:2020-01-01, to_publication_date:2026-12-31
```

### Query 3: Wav2vec LFCC fusion
```
default.search: "wav2vec LFCC fusion spoofing detection ASVspoof"
filter: from_publication_date:2021-01-01, to_publication_date:2026-12-31
```

### Query 4: Codec augmentation
```
default.search: "codec augmentation deepfake detection RawBoost domain adaptation"
filter: from_publication_date:2021-01-01, to_publication_date:2026-12-31
```

### Query 5: ASVspoof 2021 DF results
```
default.search: "ASVspoof 2021 DF evaluation equal error rate state of the art"
filter: from_publication_date:2021-01-01, to_publication_date:2026-12-31
```

### Query 6: Feature selection anti-spoofing
```
default.search: "feature selection information theoretic speech anti-spoofing reliability"
filter: from_publication_date:2020-01-01, to_publication_date:2026-12-31
```

### Query 7: ICC acoustic features
```
default.search: "intraclass correlation coefficient acoustic features telephone channel"
filter: from_publication_date:2020-01-01, to_publication_date:2026-12-31
```

### Query 8: Codec compression robustness
```
default.search: "codec compression deepfake detection robustness"
filter: from_publication_date:2023-01-01, to_publication_date:2026-12-31
```

### Query 9: Codec-aware detection
```
default.search: "codec aware deepfake audio anti-spoofing robust"
filter: from_publication_date:2024-01-01, to_publication_date:2026-12-31
```

### Query 10: CodecFake
```
default.search: "CodecFake neural codec speech detection"
filter: from_publication_date:2023-01-01, to_publication_date:2026-12-31
```

### Query 11: MoE spoofing
```
default.search: "MoE mixture experts speech spoofing deepfake audio"
filter: from_publication_date:2023-01-01, to_publication_date:2026-12-31
```

### Query 12: Adversarial codec invariance
```
default.search: "adversarial invariant representation codec compression audio spoofing"
filter: from_publication_date:2022-01-01, to_publication_date:2026-12-31
```

### Query 13: PNCC LFCC deepfake
```
default.search: "PNCC LFCC deepfake detection"
filter: from_publication_date:2023-01-01, to_publication_date:2026-12-31
```

### Query 14: Multi-level SSL gating
```
default.search: "multi-level SSL feature gating audio deepfake XLS-R"
```

### Query 15: Attention fusion SSL acoustic
```
default.search: "attention fusion wav2vec acoustic spoofing countermeasure ASVspoof"
```

### Query 16: PAC-Bayes feature selection
```
default.search: "PAC-Bayes generalization bound feature selection robustness"
```

### Plus 6 additional targeted DOI lookups for specific papers.

## Inclusion Criteria

- **Population:** Audio deepfake/spoofing detection systems
- **Intervention:** Feature selection, feature fusion, codec robustness handling
- **Outcome:** Detection performance (EER, min t-DCF) under codec conditions
- **Study design:** Empirical evaluation, system description, or survey
- **Date range:** 2020-2026
- **Language:** English

## Exclusion Criteria

- Visual deepfake detection only
- Non-audio spoofing (face, fingerprint)
- Network security / cyber attacks unrelated to audio
- Papers without empirical results or system descriptions
