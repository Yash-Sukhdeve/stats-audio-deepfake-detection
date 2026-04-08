# ML Classifier Analysis: Deepfake Audio Detection

## Configuration

- Random seed: 42
- Cross-validation: 5-fold stratified
- Bootstrap CI iterations: 1000
- NaN imputation: median (per-feature)
- Feature scaling: StandardScaler (zero mean, unit variance)
- Class weighting: balanced (LR, SVM, RF, DT); none (MLP)

## Class Imbalance

| Split | Bonafide | Spoof | Ratio | Bonafide% |
|-------|----------|-------|-------|-----------|
| train_2019 | 2580 | 22800 | 8.84:1 | 10.17% |
| dev_2019 | 2548 | 22296 | 8.75:1 | 10.26% |
| eval_2019 | 7355 | 63882 | 8.69:1 | 10.32% |
| eval_2021 | 18452 | 163114 | 8.84:1 | 10.16% |

**Imbalance effect**: Training set is 8.84:1 spoof-heavy. Without balancing, classifiers over-predict spoof. `class_weight='balanced'` is used in LR, SVM, RF and DT to correct this by up-weighting the minority bonafide class during training.

## Feature Set: Top4_Formant (4D)

**Features**: F1_mean, F1_bw, F2_bw, F1_std

### Cross-Validation AUC (5-fold, training data)

| Classifier | CV AUC (mean +/- std) | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------------|----------------------|--------|--------|--------|--------|--------|
| LogisticRegression | 0.8853 +/- 0.0039 | 0.8819  |  0.8911  |  0.8830  |  0.8874  |  0.8829 |
| SVM_RBF | 0.9031 +/- 0.0020 | 0.9056  |  0.9030  |  0.9045  |  0.9021  |  0.9006 |
| MLP | 0.9198 +/- 0.0043 | 0.9170  |  0.9244  |  0.9137  |  0.9224  |  0.9214 |
| DecisionTree | 0.8981 +/- 0.0062 | 0.8885  |  0.9042  |  0.8990  |  0.9028  |  0.8962 |
| RandomForest | 0.9148 +/- 0.0028 | 0.9123  |  0.9188  |  0.9121  |  0.9162  |  0.9145 |

### Hold-Out Evaluation: AUC and EER

| Classifier | Dev-2019 AUC (95% CI) | Dev-2019 EER% | Eval-2019 AUC (95% CI) | Eval-2019 EER% | Eval-2021 AUC (95% CI) | Eval-2021 EER% |
|------------|----------------------|--------------|----------------------|---------------|----------------------|---------------|
| LogisticRegression | 0.9037 [0.8995,0.9081] | 17.42% | 0.8670 [0.8638,0.8703] | 20.61% | 0.7860 [0.7825,0.7892] | 26.34% |
| SVM_RBF | 0.9038 [0.8990,0.9084] | 15.89% | 0.8654 [0.8621,0.8686] | 20.30% | 0.7980 [0.7946,0.8013] | 23.58% |
| MLP | 0.9219 [0.9183,0.9257] | 15.94% | 0.8794 [0.8766,0.8822] | 19.97% | 0.8037 [0.8002,0.8068] | 23.43% |
| DecisionTree | 0.9059 [0.9019,0.9101] | 17.11% | 0.8569 [0.8536,0.8603] | 22.47% | 0.7859 [0.7824,0.7895] | 23.54% |
| RandomForest | 0.9130 [0.9093,0.9170] | 16.75% | 0.8703 [0.8673,0.8733] | 20.94% | 0.8009 [0.7973,0.8042] | 23.60% |

### Confusion Matrix on Dev-2019 Set

**LogisticRegression**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   2187 |    361 |
| True Spoof    |   4186 |  18110 |

  Accuracy=0.8170  Precision(spoof)=0.9805  Recall(spoof)=0.8123  F1(spoof)=0.8885

**SVM_RBF**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   2365 |    183 |
| True Spoof    |   4653 |  17643 |

  Accuracy=0.8053  Precision(spoof)=0.9897  Recall(spoof)=0.7913  F1(spoof)=0.8795

**MLP**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |    566 |   1982 |
| True Spoof    |    273 |  22023 |

  Accuracy=0.9092  Precision(spoof)=0.9174  Recall(spoof)=0.9878  F1(spoof)=0.9513

**DecisionTree**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   2432 |    116 |
| True Spoof    |   5245 |  17051 |

  Accuracy=0.7842  Precision(spoof)=0.9932  Recall(spoof)=0.7648  F1(spoof)=0.8642

**RandomForest**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   1881 |    667 |
| True Spoof    |   2953 |  19343 |

  Accuracy=0.8543  Precision(spoof)=0.9667  Recall(spoof)=0.8676  F1(spoof)=0.9144


### Feature Importance / Coefficients

**RandomForest** (Top4_Formant):

| Feature | Importance / Coefficient |
|---------|--------------------------|
| F1_bw | 0.507738 |
| F1_mean | 0.293960 |
| F1_std | 0.116657 |
| F2_bw | 0.081646 |

**LogisticRegression** (Top4_Formant):

| Feature | Importance / Coefficient |
|---------|--------------------------|
| F1_bw | -1.271272 |
| F2_bw | 0.765518 |
| F1_mean | -0.732212 |
| F1_std | -0.713930 |

**Logistic Regression Decision Boundary (original feature space)**:

```
Decision = (-0.0087)*F1_mean + (-0.0101)*F1_bw + (0.0080)*F2_bw + (-0.0165)*F1_std + (9.5284)
(Classify as 'spoof' if Decision > 0)
```


### Pairwise Classifier Comparison (paired t-test on CV folds)

| Comparison | t-statistic | p-value | Significant (p<0.05) |
|------------|-------------|---------|----------------------|
| LogisticRegression_vs_SVM_RBF | -8.223 | 0.0012 | YES |
| LogisticRegression_vs_MLP | -27.199 | 0.0000 | YES |
| LogisticRegression_vs_DecisionTree | -7.755 | 0.0015 | YES |
| LogisticRegression_vs_RandomForest | -43.037 | 0.0000 | YES |
| SVM_RBF_vs_MLP | -6.370 | 0.0031 | YES |
| SVM_RBF_vs_DecisionTree | 1.517 | 0.2038 | no |
| SVM_RBF_vs_RandomForest | -6.234 | 0.0034 | YES |
| MLP_vs_DecisionTree | 9.091 | 0.0008 | YES |
| MLP_vs_RandomForest | 5.489 | 0.0054 | YES |
| DecisionTree_vs_RandomForest | -8.248 | 0.0012 | YES |

### Best Classifier Per Condition

| Condition | Best Classifier | AUC |
|-----------|-----------------|-----|
| cv_auc | MLP | 0.9198 |
| dev_2019 | MLP | 0.9219 |
| eval_2019 | MLP | 0.8794 |
| eval_2021 | MLP | 0.8037 |

---

## Feature Set: Formant_Prosody_15D (15D)

**Features**: F1_mean, F1_std, F1_bw, F2_mean, F2_std, F2_bw, F1_F2_ratio, formant_disp, F0_mean, F0_std, F0_max, voiced_ratio, energy_mean, energy_std, duration

### Cross-Validation AUC (5-fold, training data)

| Classifier | CV AUC (mean +/- std) | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------------|----------------------|--------|--------|--------|--------|--------|
| LogisticRegression | 0.9198 +/- 0.0061 | 0.9148  |  0.9249  |  0.9124  |  0.9264  |  0.9206 |
| SVM_RBF | 0.9802 +/- 0.0022 | 0.9790  |  0.9815  |  0.9791  |  0.9833  |  0.9781 |
| MLP | 0.9808 +/- 0.0010 | 0.9801  |  0.9811  |  0.9797  |  0.9823  |  0.9810 |
| DecisionTree | 0.8968 +/- 0.0109 | 0.9056  |  0.8975  |  0.8814  |  0.9082  |  0.8912 |
| RandomForest | 0.9536 +/- 0.0030 | 0.9551  |  0.9533  |  0.9515  |  0.9579  |  0.9503 |

### Hold-Out Evaluation: AUC and EER

| Classifier | Dev-2019 AUC (95% CI) | Dev-2019 EER% | Eval-2019 AUC (95% CI) | Eval-2019 EER% | Eval-2021 AUC (95% CI) | Eval-2021 EER% |
|------------|----------------------|--------------|----------------------|---------------|----------------------|---------------|
| LogisticRegression | 0.9280 [0.9242,0.9320] | 14.64% | 0.9013 [0.8986,0.9041] | 17.22% | 0.8031 [0.7995,0.8065] | 24.51% |
| SVM_RBF | 0.9561 [0.9526,0.9595] | 11.38% | 0.8997 [0.8971,0.9023] | 17.23% | 0.8253 [0.8220,0.8288] | 21.45% |
| MLP | 0.9633 [0.9608,0.9661] | 10.64% | 0.9024 [0.8999,0.9049] | 17.26% | 0.8119 [0.8083,0.8154] | 22.33% |
| DecisionTree | 0.8895 [0.8838,0.8951] | 16.59% | 0.8557 [0.8522,0.8594] | 21.83% | 0.7798 [0.7764,0.7838] | 22.91% |
| RandomForest | 0.9366 [0.9333,0.9400] | 14.17% | 0.8936 [0.8911,0.8962] | 18.49% | 0.8258 [0.8225,0.8291] | 21.16% |

### Confusion Matrix on Dev-2019 Set

**LogisticRegression**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   2185 |    363 |
| True Spoof    |   3298 |  18998 |

  Accuracy=0.8526  Precision(spoof)=0.9813  Recall(spoof)=0.8521  F1(spoof)=0.9121

**SVM_RBF**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   1899 |    649 |
| True Spoof    |   1102 |  21194 |

  Accuracy=0.9295  Precision(spoof)=0.9703  Recall(spoof)=0.9506  F1(spoof)=0.9603

**MLP**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   1320 |   1228 |
| True Spoof    |    277 |  22019 |

  Accuracy=0.9394  Precision(spoof)=0.9472  Recall(spoof)=0.9876  F1(spoof)=0.9670

**DecisionTree**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   2337 |    211 |
| True Spoof    |   4416 |  17880 |

  Accuracy=0.8138  Precision(spoof)=0.9883  Recall(spoof)=0.8019  F1(spoof)=0.8854

**RandomForest**

|         | Pred Bonafide | Pred Spoof |
|---------|--------------|------------|
| True Bonafide |   1980 |    568 |
| True Spoof    |   2378 |  19918 |

  Accuracy=0.8814  Precision(spoof)=0.9723  Recall(spoof)=0.8933  F1(spoof)=0.9311


### Feature Importance / Coefficients

**RandomForest** (Formant_Prosody_15D):

| Feature | Importance / Coefficient |
|---------|--------------------------|
| F1_bw | 0.261720 |
| F1_mean | 0.145835 |
| F1_F2_ratio | 0.119547 |
| F1_std | 0.117024 |
| F0_mean | 0.067032 |
| F0_std | 0.040836 |
| duration | 0.040363 |
| energy_mean | 0.039786 |
| energy_std | 0.034052 |
| F0_max | 0.027137 |
| F2_bw | 0.023395 |
| F2_mean | 0.022150 |
| formant_disp | 0.021052 |
| voiced_ratio | 0.020943 |
| F2_std | 0.019129 |

**LogisticRegression** (Formant_Prosody_15D):

| Feature | Importance / Coefficient |
|---------|--------------------------|
| F1_F2_ratio | 4.507338 |
| F1_mean | -3.605942 |
| formant_disp | 2.426136 |
| energy_mean | 1.794744 |
| energy_std | -1.450798 |
| F2_bw | 1.317185 |
| F1_std | -1.148425 |
| F1_bw | -0.707104 |
| voiced_ratio | -0.312274 |
| F2_std | -0.293517 |
| F0_std | -0.259880 |
| F0_mean | 0.226589 |
| duration | -0.222021 |
| F0_max | 0.198159 |
| F2_mean | -0.157784 |

**Logistic Regression Decision Boundary (original feature space)**:

```
Decision = (-0.0429)*F1_mean + (-0.0266)*F1_std + (-0.0056)*F1_bw + (-0.0014)*F2_mean + (-0.0037)*F2_std + (0.0138)*F2_bw + (109.4341)*F1_F2_ratio + (0.0207)*formant_disp + (0.0047)*F0_mean + (-0.0124)*F0_std + (0.0026)*F0_max + (-1.6630)*voiced_ratio + (55.5946)*energy_mean + (-77.7887)*energy_std + (-0.1565)*duration + (-27.1179)
(Classify as 'spoof' if Decision > 0)
```


### Pairwise Classifier Comparison (paired t-test on CV folds)

| Comparison | t-statistic | p-value | Significant (p<0.05) |
|------------|-------------|---------|----------------------|
| LogisticRegression_vs_SVM_RBF | -28.707 | 0.0000 | YES |
| LogisticRegression_vs_MLP | -26.317 | 0.0000 | YES |
| LogisticRegression_vs_DecisionTree | 5.625 | 0.0049 | YES |
| LogisticRegression_vs_RandomForest | -13.713 | 0.0002 | YES |
| SVM_RBF_vs_MLP | -0.902 | 0.4180 | no |
| SVM_RBF_vs_DecisionTree | 19.033 | 0.0000 | YES |
| SVM_RBF_vs_RandomForest | 32.068 | 0.0000 | YES |
| MLP_vs_DecisionTree | 18.204 | 0.0001 | YES |
| MLP_vs_RandomForest | 23.807 | 0.0000 | YES |
| DecisionTree_vs_RandomForest | -15.030 | 0.0001 | YES |

### Best Classifier Per Condition

| Condition | Best Classifier | AUC |
|-----------|-----------------|-----|
| cv_auc | MLP | 0.9808 |
| dev_2019 | MLP | 0.9633 |
| eval_2019 | MLP | 0.9024 |
| eval_2021 | RandomForest | 0.8258 |

---

## Summary: Best Overall Results

| Feature Set | Best Classifier | Best Condition | AUC |
|-------------|-----------------|---------------|-----|
| Top4_Formant | MLP | eval_2019 | 0.8794 |
| Top4_Formant | MLP | eval_2021 | 0.8037 |
| Formant_Prosody_15D | MLP | eval_2019 | 0.9024 |
| Formant_Prosody_15D | RandomForest | eval_2021 | 0.8258 |

---
*Analysis executed with pinned environment, seed=42, 5-fold stratified CV, 1000-iteration bootstrap CIs. All NaN values imputed with per-feature median.*
