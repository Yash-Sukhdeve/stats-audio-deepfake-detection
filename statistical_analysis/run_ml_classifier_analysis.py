"""
ML Classifier Analysis for Deepfake Audio Detection
====================================================
Trains and evaluates 5 classifiers on:
  - Top-4 Formant features (F1_mean, F1_bw, F2_bw, F1_std)
  - Formants+Prosody combined (15D)

Reports:
  - 5-fold stratified CV AUC (with 95% bootstrap CI)
  - Dev / Eval-2019 / Eval-2021 AUC and EER
  - Confusion matrix on dev set
  - Feature importance (RF) and coefficient (LR)
  - LR decision boundary equation
  - Class imbalance analysis
  - Paired t-test classifier comparisons

Output files:
  evidence/experiments/discriminative_analysis/ml_classifier_results.json
  evidence/experiments/discriminative_analysis/ml_classifier_report.md

Reproducibility: numpy.random.seed(42), all classifiers seeded.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

# --- reproducibility ---
np.random.seed(42)
import random
random.seed(42)

# --- suppress convergence warnings for clean output ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, cross_validate
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.stats import ttest_rel

print(f"Python  : {sys.version}")
print(f"NumPy   : {np.__version__}")
import sklearn; print(f"sklearn : {sklearn.__version__}")

# =============================================================================
# 1.  FEATURE INDEX MAPPING
#     v4 feature layout (36-dim):
#       PNCC[0-12], LFCC[13-20], Formants[21-28], Prosody[29-35]
# =============================================================================
FEATURE_NAMES = [
    'PNCC_0',  'PNCC_1',  'PNCC_2',  'PNCC_3',  'PNCC_4',
    'PNCC_5',  'PNCC_6',  'PNCC_7',  'PNCC_8',  'PNCC_9',
    'PNCC_10', 'PNCC_11', 'PNCC_12',
    'LFCC_0',  'LFCC_1',  'LFCC_2',  'LFCC_3',  'LFCC_4',
    'LFCC_5',  'LFCC_6',  'LFCC_7',
    'F1_mean', 'F1_std',  'F1_bw',   'F2_mean', 'F2_std',
    'F2_bw',   'F1_F2_ratio', 'formant_disp',
    'F0_mean', 'F0_std',  'F0_max',  'voiced_ratio',
    'energy_mean', 'energy_std', 'duration'
]

# Top-4 formant features (by composite_score from phase analysis)
TOP4_FORMANT_IDX  = [21, 23, 26, 22]   # F1_mean, F1_bw, F2_bw, F1_std
TOP4_FORMANT_NAMES = ['F1_mean', 'F1_bw', 'F2_bw', 'F1_std']

# Formants (8) + Prosody (7) = 15D
FORM_PROS_IDX   = list(range(21, 36))
FORM_PROS_NAMES = FEATURE_NAMES[21:36]

FEATURE_SETS = {
    'Top4_Formant': (TOP4_FORMANT_IDX,  TOP4_FORMANT_NAMES),
    'Formant_Prosody_15D': (FORM_PROS_IDX, FORM_PROS_NAMES),
}

# =============================================================================
# 2.  LOAD DATA
# =============================================================================
BASE = Path('/home/lab2208/Documents/df_detection/evidence/experiments/features')

print("\n--- Loading feature files ---")
X_train_full = np.load(BASE / 'acoustic_v4_train_2019_fixed.npy').astype(np.float64)
X_dev_full   = np.load(BASE / 'acoustic_v4_dev_2019_fixed.npy').astype(np.float64)
X_eval_full  = np.load(BASE / 'acoustic_v4_eval_2019_fixed.npy').astype(np.float64)
X_2021_full  = np.load(BASE / 'acoustic_v4_eval_2021.npy').astype(np.float64)

y_train = np.load(BASE / 'labels_train_2019.npy').astype(int)
y_dev   = np.load(BASE / 'labels_dev_2019.npy').astype(int)
y_eval  = np.load(BASE / 'labels_eval_2019.npy').astype(int)
y_2021  = np.load(BASE / 'labels_eval_2021.npy').astype(int)

for split, X, y in [
    ('Train 2019', X_train_full, y_train),
    ('Dev   2019', X_dev_full,   y_dev),
    ('Eval  2019', X_eval_full,  y_eval),
    ('Eval  2021', X_2021_full,  y_2021),
]:
    bonafide = (y == 0).sum()
    spoof    = (y == 1).sum()
    nan_pct  = 100 * np.isnan(X).sum() / X.size
    print(f"  {split}: {X.shape}  bonafide={bonafide:6d}  spoof={spoof:6d}  "
          f"imbalance={spoof/bonafide:.2f}:1  NaN={nan_pct:.3f}%")

# Class imbalance analysis
n_bon  = (y_train == 0).sum()
n_spo  = (y_train == 1).sum()
imbal_ratio = n_spo / n_bon
print(f"\nClass imbalance ratio (train): {imbal_ratio:.2f}:1")
print(f"  -> Using class_weight='balanced' in LR, RF, DT")
print(f"  -> SVM uses class_weight='balanced'")
print(f"  -> MLP uses no weighting (monitors via per-class metrics)")

# =============================================================================
# 3.  HELPER FUNCTIONS
# =============================================================================

def compute_eer(y_true, y_score):
    """Equal Error Rate via interpolation of FAR/FRR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    eer = brentq(lambda x: interp1d(fpr, fnr - x)(x), 0.0, 1.0)
    return float(eer)


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, ci=0.95, seed=42):
    """
    Bootstrap 95% CI for AUC.

    Draws n_boot samples with replacement, computes AUC each time,
    returns (lower, upper) percentile bounds.
    """
    rng   = np.random.RandomState(seed)
    n     = len(y_true)
    aucs  = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        # skip folds where only one class present
        ys  = y_true[idx]
        if len(np.unique(ys)) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(ys, y_score[idx])
    aucs = aucs[~np.isnan(aucs)]
    alpha = 1.0 - ci
    lo    = float(np.percentile(aucs, 100 * alpha / 2))
    hi    = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate_on_split(pipe, X, y, split_name, n_boot=1000):
    """
    Run predict_proba through a fitted pipeline on one split.
    Returns dict with AUC, EER, 95% bootstrap CI on AUC.
    """
    # pipelines contain imputer + scaler; predict_proba gives P(spoof)
    try:
        proba  = pipe.predict_proba(X)[:, 1]
    except AttributeError:
        # SVC with probability=True already handles this
        proba = pipe.predict_proba(X)[:, 1]

    auc = float(roc_auc_score(y, proba))
    eer = compute_eer(y, proba)
    ci_lo, ci_hi = bootstrap_auc_ci(y, proba, n_boot=n_boot)

    return {
        'split':  split_name,
        'auc':    round(auc, 6),
        'eer':    round(eer * 100, 4),   # percent
        'auc_ci_lo': round(ci_lo, 6),
        'auc_ci_hi': round(ci_hi, 6),
    }


def make_pipelines():
    """
    Return dict of {name: Pipeline} for each classifier.
    Pipeline = SimpleImputer (median) -> StandardScaler -> Classifier.
    All random_state=42 where applicable.
    class_weight='balanced' for LR, RF, DT.
    SVC uses class_weight='balanced' and probability=True.
    """
    return {
        'LogisticRegression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf',     LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
                solver='lbfgs',
            ))
        ]),
        'SVM_RBF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf',     SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
            ))
        ]),
        'MLP': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf',     MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            ))
        ]),
        'DecisionTree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf',     DecisionTreeClassifier(
                max_depth=5,
                class_weight='balanced',
                random_state=42,
            ))
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf',     RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            ))
        ]),
    }


# =============================================================================
# 4.  MAIN ANALYSIS LOOP
# =============================================================================

results = {}          # full structured results
cv_auc_all = {}       # cv_auc_all[feat_set][clf_name] = array(5) for paired t-test

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for feat_set_name, (feat_idx, feat_names) in FEATURE_SETS.items():
    print(f"\n{'='*70}")
    print(f"FEATURE SET: {feat_set_name}  ({len(feat_idx)}D: {feat_names})")
    print(f"{'='*70}")

    X_tr = X_train_full[:, feat_idx]
    X_dv = X_dev_full[:,   feat_idx]
    X_ev = X_eval_full[:,  feat_idx]
    X_21 = X_2021_full[:,  feat_idx]

    results[feat_set_name]    = {}
    cv_auc_all[feat_set_name] = {}

    for clf_name, pipe in make_pipelines().items():
        print(f"\n  Classifier: {clf_name}")

        # --- 5-fold stratified CV on training set ---
        cv_scores = cross_val_score(
            pipe, X_tr, y_train,
            cv=skf,
            scoring='roc_auc',
            n_jobs=1,
        )
        cv_mean  = float(np.mean(cv_scores))
        cv_std   = float(np.std(cv_scores, ddof=1))
        cv_auc_all[feat_set_name][clf_name] = cv_scores
        print(f"    CV AUC: {cv_mean:.4f} +/- {cv_std:.4f}  folds={np.round(cv_scores,4)}")

        # --- fit on full training set ---
        pipe.fit(X_tr, y_train)

        # --- dev set: AUC, EER, bootstrap CI, confusion matrix ---
        dev_res  = evaluate_on_split(pipe, X_dv, y_dev,   'dev_2019')
        eval_res = evaluate_on_split(pipe, X_ev, y_eval,  'eval_2019')
        la21_res = evaluate_on_split(pipe, X_21, y_2021,  'eval_2021')

        print(f"    Dev  2019: AUC={dev_res['auc']:.4f}  EER={dev_res['eer']:.2f}%  "
              f"95CI=[{dev_res['auc_ci_lo']:.4f},{dev_res['auc_ci_hi']:.4f}]")
        print(f"    Eval 2019: AUC={eval_res['auc']:.4f}  EER={eval_res['eer']:.2f}%  "
              f"95CI=[{eval_res['auc_ci_lo']:.4f},{eval_res['auc_ci_hi']:.4f}]")
        print(f"    Eval 2021: AUC={la21_res['auc']:.4f}  EER={la21_res['eer']:.2f}%  "
              f"95CI=[{la21_res['auc_ci_lo']:.4f},{la21_res['auc_ci_hi']:.4f}]")

        # --- confusion matrix on dev ---
        y_dev_pred = pipe.predict(X_dv)
        cm = confusion_matrix(y_dev, y_dev_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"    Dev ConfMat: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        cr = classification_report(y_dev, y_dev_pred,
                                   target_names=['bonafide','spoof'],
                                   output_dict=True)

        # --- feature importance / coefficients ---
        clf_obj = pipe.named_steps['clf']
        feat_importance = None
        lr_equation = None

        if clf_name == 'RandomForest':
            fi = clf_obj.feature_importances_
            feat_importance = {fn: round(float(v), 6)
                               for fn, v in zip(feat_names, fi)}
            top3 = sorted(feat_importance, key=feat_importance.get, reverse=True)[:3]
            print(f"    RF top-3 importance: {[(k, feat_importance[k]) for k in top3]}")

        elif clf_name == 'LogisticRegression':
            coef  = clf_obj.coef_[0]
            inter = clf_obj.intercept_[0]
            # Coefficients are in scaled space; retrieve scaler params to back-transform
            scaler = pipe.named_steps['scaler']
            mu    = scaler.mean_
            sigma = scaler.scale_
            # z_i = (x_i - mu_i)/sigma_i
            # decision = sum(coef_i * z_i) + intercept
            #          = sum(coef_i/sigma_i * x_i) + (intercept - sum(coef_i*mu_i/sigma_i))
            raw_coef  = coef / sigma
            raw_inter = float(inter - np.dot(coef, mu / sigma))
            feat_importance = {fn: round(float(v), 6)
                               for fn, v in zip(feat_names, coef)}
            terms = " + ".join(
                f"({raw_coef[i]:.4f})*{feat_names[i]}" for i in range(len(feat_names))
            )
            lr_equation = f"Decision = {terms} + ({raw_inter:.4f})"
            lr_equation += "\n(Classify as 'spoof' if Decision > 0)"
            print(f"    LR coefficients (scaled): {dict(zip(feat_names, np.round(coef,4)))}")
            print(f"    LR decision boundary (original feature space):")
            print(f"      {lr_equation}")

        # --- assemble record ---
        record = {
            'cv_auc_mean':  round(cv_mean, 6),
            'cv_auc_std':   round(cv_std, 6),
            'cv_auc_folds': [round(float(v), 6) for v in cv_scores],
            'dev_2019':     dev_res,
            'eval_2019':    eval_res,
            'eval_2021':    la21_res,
            'confusion_matrix_dev': {
                'TN': int(tn), 'FP': int(fp),
                'FN': int(fn), 'TP': int(tp),
            },
            'classification_report_dev': {
                k: v for k, v in cr.items()
                if k not in ('accuracy',)
            },
            'feature_importance': feat_importance,
            'lr_decision_boundary': lr_equation,
        }
        results[feat_set_name][clf_name] = record


# =============================================================================
# 5.  STATISTICAL COMPARISONS (paired t-test on 5 CV fold AUC scores)
# =============================================================================

print(f"\n{'='*70}")
print("STATISTICAL COMPARISONS (paired t-test on CV fold AUCs)")
print(f"{'='*70}")

pairwise = {}
for feat_set_name in FEATURE_SETS:
    pairwise[feat_set_name] = {}
    clf_names = list(cv_auc_all[feat_set_name].keys())
    for a, b in combinations(clf_names, 2):
        scores_a = cv_auc_all[feat_set_name][a]
        scores_b = cv_auc_all[feat_set_name][b]
        t_stat, p_val = ttest_rel(scores_a, scores_b)
        key = f"{a}_vs_{b}"
        pairwise[feat_set_name][key] = {
            't_statistic': round(float(t_stat), 4),
            'p_value':     round(float(p_val), 6),
            'significant_alpha05': bool(p_val < 0.05),
        }
        sig = "*" if p_val < 0.05 else ""
        print(f"  [{feat_set_name}] {a} vs {b}: t={t_stat:.3f} p={p_val:.4f}{sig}")


# =============================================================================
# 6.  BEST CLASSIFIER PER CONDITION
# =============================================================================

print(f"\n{'='*70}")
print("BEST CLASSIFIER PER CONDITION")
print(f"{'='*70}")

best = {}
for feat_set_name in FEATURE_SETS:
    best[feat_set_name] = {}
    metrics = ['cv_auc_mean', 'dev_2019_auc', 'eval_2019_auc', 'eval_2021_auc']
    for cond in ['cv_auc', 'dev_2019', 'eval_2019', 'eval_2021']:
        if cond == 'cv_auc':
            vals = {
                clf: results[feat_set_name][clf]['cv_auc_mean']
                for clf in results[feat_set_name]
            }
        else:
            vals = {
                clf: results[feat_set_name][clf][cond]['auc']
                for clf in results[feat_set_name]
            }
        winner = max(vals, key=vals.get)
        best[feat_set_name][cond] = {'classifier': winner, 'auc': round(vals[winner], 4)}
        print(f"  [{feat_set_name}] Best on {cond:15s}: {winner:22s}  AUC={vals[winner]:.4f}")


# =============================================================================
# 7.  CLASS IMBALANCE ANALYSIS
# =============================================================================

imbalance_info = {}
for split_name, y_arr in [
    ('train_2019', y_train),
    ('dev_2019',   y_dev),
    ('eval_2019',  y_eval),
    ('eval_2021',  y_2021),
]:
    n0 = int((y_arr == 0).sum())
    n1 = int((y_arr == 1).sum())
    imbalance_info[split_name] = {
        'n_bonafide': n0,
        'n_spoof':    n1,
        'total':      n0 + n1,
        'imbalance_ratio': round(n1 / n0, 3),
        'bonafide_pct': round(100 * n0 / (n0 + n1), 2),
        'spoof_pct':    round(100 * n1 / (n0 + n1), 2),
    }

print(f"\n{'='*70}")
print("CLASS IMBALANCE SUMMARY")
print(f"{'='*70}")
for s, d in imbalance_info.items():
    print(f"  {s}: {d['n_bonafide']} bonafide / {d['n_spoof']} spoof  "
          f"({d['imbalance_ratio']:.2f}:1)  bonafide={d['bonafide_pct']}%")


# =============================================================================
# 8.  ASSEMBLE FINAL RESULTS OBJECT
# =============================================================================

final = {
    'metadata': {
        'python_version': sys.version,
        'numpy_version':  np.__version__,
        'sklearn_version': sklearn.__version__,
        'random_seed': 42,
        'cv_folds': 5,
        'bootstrap_iterations': 1000,
        'description': (
            'Multi-classifier evaluation on ASVspoof 2019/2021 LA '
            'acoustic features. Top-4 formant features and 15D '
            'Formant+Prosody feature sets. '
            'class_weight=balanced for LR/RF/DT/SVM. '
            'NaN imputed with median. StandardScaler applied.'
        ),
    },
    'feature_sets': {
        name: {'indices': idx, 'names': names}
        for name, (idx, names) in FEATURE_SETS.items()
    },
    'class_imbalance': imbalance_info,
    'results': results,
    'pairwise_ttest': pairwise,
    'best_per_condition': best,
}

# =============================================================================
# 9.  WRITE OUTPUTS
# =============================================================================

OUT_DIR = Path('/home/lab2208/Documents/df_detection/evidence/experiments/discriminative_analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

json_path = OUT_DIR / 'ml_classifier_results.json'
with open(json_path, 'w') as f:
    json.dump(final, f, indent=2)
print(f"\nJSON results written to: {json_path}")


# =============================================================================
# 10. MARKDOWN REPORT
# =============================================================================

def fmt_auc(v, ci_lo=None, ci_hi=None):
    s = f"{v:.4f}"
    if ci_lo is not None:
        s += f" [{ci_lo:.4f},{ci_hi:.4f}]"
    return s


md_lines = [
    "# ML Classifier Analysis: Deepfake Audio Detection",
    "",
    "## Configuration",
    "",
    f"- Random seed: 42",
    f"- Cross-validation: 5-fold stratified",
    f"- Bootstrap CI iterations: 1000",
    f"- NaN imputation: median (per-feature)",
    f"- Feature scaling: StandardScaler (zero mean, unit variance)",
    f"- Class weighting: balanced (LR, SVM, RF, DT); none (MLP)",
    "",
    "## Class Imbalance",
    "",
    "| Split | Bonafide | Spoof | Ratio | Bonafide% |",
    "|-------|----------|-------|-------|-----------|",
]
for s, d in imbalance_info.items():
    md_lines.append(
        f"| {s} | {d['n_bonafide']} | {d['n_spoof']} | "
        f"{d['imbalance_ratio']:.2f}:1 | {d['bonafide_pct']}% |"
    )

md_lines += [
    "",
    "**Imbalance effect**: Training set is 8.84:1 spoof-heavy. "
    "Without balancing, classifiers over-predict spoof. "
    "`class_weight='balanced'` is used in LR, SVM, RF and DT to correct this "
    "by up-weighting the minority bonafide class during training.",
    "",
]

for feat_set_name, (feat_idx, feat_names) in FEATURE_SETS.items():
    md_lines += [
        f"## Feature Set: {feat_set_name} ({len(feat_idx)}D)",
        "",
        f"**Features**: {', '.join(feat_names)}",
        "",
        "### Cross-Validation AUC (5-fold, training data)",
        "",
        "| Classifier | CV AUC (mean +/- std) | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |",
        "|------------|----------------------|--------|--------|--------|--------|--------|",
    ]
    for clf_name in results[feat_set_name]:
        r = results[feat_set_name][clf_name]
        folds = "  |  ".join(f"{v:.4f}" for v in r['cv_auc_folds'])
        md_lines.append(
            f"| {clf_name} | {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
            f"| {folds} |"
        )

    md_lines += [
        "",
        "### Hold-Out Evaluation: AUC and EER",
        "",
        "| Classifier | Dev-2019 AUC (95% CI) | Dev-2019 EER% | Eval-2019 AUC (95% CI) | Eval-2019 EER% | Eval-2021 AUC (95% CI) | Eval-2021 EER% |",
        "|------------|----------------------|--------------|----------------------|---------------|----------------------|---------------|",
    ]
    for clf_name in results[feat_set_name]:
        r  = results[feat_set_name][clf_name]
        dv = r['dev_2019']
        ev = r['eval_2019']
        e1 = r['eval_2021']
        md_lines.append(
            f"| {clf_name} "
            f"| {dv['auc']:.4f} [{dv['auc_ci_lo']:.4f},{dv['auc_ci_hi']:.4f}] "
            f"| {dv['eer']:.2f}% "
            f"| {ev['auc']:.4f} [{ev['auc_ci_lo']:.4f},{ev['auc_ci_hi']:.4f}] "
            f"| {ev['eer']:.2f}% "
            f"| {e1['auc']:.4f} [{e1['auc_ci_lo']:.4f},{e1['auc_ci_hi']:.4f}] "
            f"| {e1['eer']:.2f}% |"
        )

    md_lines += ["", "### Confusion Matrix on Dev-2019 Set", ""]
    for clf_name in results[feat_set_name]:
        r  = results[feat_set_name][clf_name]
        cm = r['confusion_matrix_dev']
        tn, fp, fn, tp = cm['TN'], cm['FP'], cm['FN'], cm['TP']
        n_total = tn + fp + fn + tp
        acc  = (tn + tp) / n_total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        md_lines += [
            f"**{clf_name}**",
            "",
            f"|         | Pred Bonafide | Pred Spoof |",
            f"|---------|--------------|------------|",
            f"| True Bonafide | {tn:6d} | {fp:6d} |",
            f"| True Spoof    | {fn:6d} | {tp:6d} |",
            f"",
            f"  Accuracy={acc:.4f}  Precision(spoof)={prec:.4f}  "
            f"Recall(spoof)={rec:.4f}  F1(spoof)={f1:.4f}",
            "",
        ]

    # Feature importance
    md_lines += ["", "### Feature Importance / Coefficients", ""]
    for clf_name in ['RandomForest', 'LogisticRegression']:
        r = results[feat_set_name].get(clf_name, {})
        fi = r.get('feature_importance')
        if fi:
            sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
            md_lines += [
                f"**{clf_name}** ({feat_set_name}):",
                "",
                "| Feature | Importance / Coefficient |",
                "|---------|--------------------------|",
            ]
            for fn, fv in sorted_fi:
                md_lines.append(f"| {fn} | {fv:.6f} |")
            md_lines.append("")

        eq = r.get('lr_decision_boundary')
        if eq:
            md_lines += [
                f"**Logistic Regression Decision Boundary (original feature space)**:",
                "",
                f"```",
                eq,
                f"```",
                "",
            ]

    # Paired t-tests
    md_lines += ["", "### Pairwise Classifier Comparison (paired t-test on CV folds)", ""]
    md_lines += [
        "| Comparison | t-statistic | p-value | Significant (p<0.05) |",
        "|------------|-------------|---------|----------------------|",
    ]
    for pair_key, pv in pairwise[feat_set_name].items():
        sig = "YES" if pv['significant_alpha05'] else "no"
        md_lines.append(
            f"| {pair_key} | {pv['t_statistic']:.3f} | {pv['p_value']:.4f} | {sig} |"
        )

    # Best per condition
    md_lines += ["", "### Best Classifier Per Condition", ""]
    md_lines += [
        "| Condition | Best Classifier | AUC |",
        "|-----------|-----------------|-----|",
    ]
    for cond, bv in best[feat_set_name].items():
        md_lines.append(f"| {cond} | {bv['classifier']} | {bv['auc']:.4f} |")
    md_lines += ["", "---", ""]

md_lines += [
    "## Summary: Best Overall Results",
    "",
    "| Feature Set | Best Classifier | Best Condition | AUC |",
    "|-------------|-----------------|---------------|-----|",
]
for feat_set_name in FEATURE_SETS:
    for cond in ['eval_2019', 'eval_2021']:
        bv = best[feat_set_name][cond]
        md_lines.append(
            f"| {feat_set_name} | {bv['classifier']} | {cond} | {bv['auc']:.4f} |"
        )

md_lines += [
    "",
    "---",
    "*Analysis executed with pinned environment, seed=42, 5-fold stratified CV, "
    "1000-iteration bootstrap CIs. All NaN values imputed with per-feature median.*",
]

md_path = OUT_DIR / 'ml_classifier_report.md'
with open(md_path, 'w') as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Markdown report written to: {md_path}")
print("\nDone.")
