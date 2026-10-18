# Experiment Protocol

**Date:** 2026-10-19  
**Version:** 1.0  
**Status:** Pre-registered (before final test evaluation)

---

## Objective

Train a repaired supervised autoencoder to detect defective wafers at natural prevalence (~6.6%), fixing three critical bugs from the original implementation.

---

## Data Splits

**Temporal Chronological Split** (no shuffling):
- Training: 70% (earliest 1,096 samples)
- Validation: 15% (middle 235 samples)  
- Test: 15% (latest 236 samples)

**Verification:**
- No overlap between splits ✓
- Chronological ordering maintained ✓
- No future information leakage ✓

---

## Preprocessing Pipeline

**Fitted ONLY on training data:**

1. **Legacy Detection:** `sensor_0 == 0.0` (exact)
2. **Continuous Features (590):**
   - Mean imputation for missing values
   - Z-score standardization
   - Exclude legacy `sensor_1` from continuous processing
3. **Categorical Features (3):**
   - One-hot encode firmware codes {100, 200, 300}
   - Only for legacy wafers
4. **Station Indicator (1):**
   - Binary flag for legacy vs modern

**Output:** 594 features (590 + 3 + 1)

**No SMOTE or resampling** on any split.

---

## Model Architecture

**Supervised Autoencoder (WaferSAE):**
- Input: 594 features
- Encoder: 594 → 256 → 128 → 32 (latent)
- Decoder: 32 → 128 → 256 → 594 (reconstruction)
- Classification head: 32 → 1 (defect probability)

**Loss Function:**
```
L_total = L_reconstruction + λ * L_classification
L_reconstruction = MSE(reconstructed, input)
L_classification = BCE(logits, labels)
```

**Hyperparameters (Fixed):**
- Latent dimension: 32
- Hidden dimension: 256
- λ (classification weight): 1.0
- Dropout: 0.10
- Learning rate: 1e-3
- Weight decay: 1e-5
- Batch size: 64
- Epochs: 40
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR

**Early Stopping:** Monitor validation F1, patience=10

---

## Baseline Comparison

**Simple Baseline:** Logistic Regression with class weighting
- Same preprocessing pipeline
- No dimensionality reduction
- Solver: 'liblinear'
- Class weight: 'balanced'
- Max iterations: 1000

---

## Threshold Selection

**Method:** Maximize F1 score on validation set

**Process:**
1. Generate predictions on validation set
2. Sweep thresholds from 0.01 to 0.99 (step=0.01)
3. Compute F1 for each threshold
4. Select threshold with maximum F1
5. **Freeze this threshold** before evaluating on test set

**Rationale:** F1 balances precision/recall for imbalanced data

---

## Evaluation Metrics

**Primary Metrics:**
- F1 Score
- Precision
- Recall

**Secondary Metrics:**
- ROC-AUC
- Average Precision (PR-AUC)
- Confusion Matrix (TN, FP, FN, TP)

**Operational Metrics:**
- False alarms per 1,000 wafers
- Missed defects per 1,000 wafers
- False positive rate
- False negative rate

**Subgroup Analysis:**
- Modern wafers performance
- Legacy wafers performance
- Sample counts for each subgroup

**Test Set Requirements:**
- Natural prevalence (no balancing)
- Evaluate at frozen threshold from validation
- Report actual observed prevalence

---

## Training Procedure

1. Load data and create temporal splits
2. Fit preprocessing on training data only
3. Transform all splits
4. Train baseline (logistic regression)
5. Train supervised autoencoder
6. Select threshold on validation set
7. Freeze model and threshold
8. Evaluate on test set (one time only)
9. Generate subgroup metrics
10. Save all results

---

## Seeds and Reproducibility

- Global seed: 1337
- PyTorch seed: 1337
- NumPy seed: 1337
- Random seed: 1337

**Device:** CPU (PyTorch)

**Deterministic:** Where possible (some ops are non-deterministic on CPU)

---

## Output Artifacts

**Models:**
- `models/wafer_sae.pt` - Trained autoencoder weights
- `models/baseline_lr.joblib` - Baseline logistic regression
- `models/preprocessor.joblib` - Fitted preprocessing pipeline

**Metrics:**
- `reports/metrics.json` - All evaluation metrics
- `reports/validation_threshold.json` - Threshold selection details
- `reports/subgroup_metrics.json` - Modern vs legacy performance

**Figures:**
- `reports/figures/confusion_matrix.png`
- `reports/figures/pr_curve.png`
- `reports/figures/roc_curve.png`
- `reports/figures/threshold_selection.png`
- `reports/figures/loss_curves.png`

---

## Success Criteria

**Minimum Requirements:**
- Model trains without errors
- No NaN losses or gradients
- Evaluation completes at natural prevalence
- Honest metrics reported (no fabrication)

**Expected Performance (Conservative):**
- F1: 0.40 - 0.70 (realistic for 6.6% prevalence)
- ROC-AUC: 0.70 - 0.85
- Better than random baseline (F1 > 0.10)

**Not Expected:**
- F1 > 0.90 (original inflated metric)
- Perfect predictions (this is a hard problem)

---

## Limitations

**Data Limitations:**
- Small dataset (1,567 samples, 104 defects)
- Single test set (no cross-validation)
- Unknown if training period represents deployment
- Limited legacy wafer samples (146 total)

**Model Limitations:**
- Autoencoder assumption unvalidated
- No external validation
- No uncertainty quantification
- Single training run (no multiple seeds)

**Evaluation Limitations:**
- Point estimates only
- No confidence intervals
- Subgroup metrics may be unstable
- No drift detection mechanism

---

## Deviations from Protocol

**Any deviation from this protocol must be documented with:**
- What changed
- Why it changed
- Impact on conclusions

**Post-hoc modifications invalidate pre-registration.**

---

**Protocol Frozen:** 2026-10-19 01:30 UTC  
**Signature:** Automated (before test evaluation)
