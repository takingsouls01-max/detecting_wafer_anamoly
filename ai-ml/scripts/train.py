"""
Train wafer anomaly detector following experiment protocol.
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor
from waferguard.model import WaferSAE
from waferguard.training import create_dataloaders, train_epoch, evaluate
from waferguard.evaluation import compute_metrics, find_best_threshold, subgroup_metrics, save_metrics

# Seeds
SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')  # Use CPU for reproducibility


def train_baseline(X_train, y_train, X_val, y_val):
    """Train simple logistic regression baseline."""
    print("\nTraining Baseline (Logistic Regression)...")

    baseline = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=SEED
    )

    baseline.fit(X_train, y_train)

    # Evaluate on validation
    y_val_probs = baseline.predict_proba(X_val)[:, 1]
    y_val_pred = baseline.predict(X_val)

    val_f1 = f1_score(y_val, y_val_pred)

    print(f"  Validation F1: {val_f1:.4f}")

    return baseline


def train_autoencoder(train_loader, val_loader, d_in):
    """Train supervised autoencoder."""
    print("\nTraining Supervised Autoencoder...")

    # Model
    model = WaferSAE(d_in, d_hidden=256, d_latent=32).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40
    )

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 10
    best_model_state = model.state_dict().copy()  # Initialize with initial weights

    for epoch in range(1, 41):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, DEVICE, lambda_clf=1.0)

        # Validate
        val_losses, val_probs, val_labels = evaluate(model, val_loader, DEVICE, lambda_clf=1.0)

        # Compute F1 at 0.5 threshold
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/40 | Train Loss: {train_losses['total']:.4f} | "
                  f"Val Loss: {val_losses['total']:.4f} | Val F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    print(f"  Best Validation F1: {best_val_f1:.4f}")

    return model


def main():
    """Main training pipeline."""
    print("="*60)
    print("WAFER ANOMALY DETECTOR TRAINING")
    print("="*60)

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Load data
    print("\n1. Loading dataset...")
    dataset = WaferDataset('data/secom_telemetry_modified.csv')
    print(dataset.info)

    # Temporal split
    print("\n2. Creating temporal splits...")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    X = dataset.get_features()
    y = dataset.y
    legacy_info = dataset.get_legacy_features()

    X_train = X[train_idx]
    y_train = y[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    X_val = X[val_idx]
    y_val = y[val_idx]
    legacy_val = {k: v[val_idx] for k, v in legacy_info.items()}

    X_test = X[test_idx]
    y_test = y[test_idx]
    legacy_test = {k: v[test_idx] for k, v in legacy_info.items()}

    print(f"  Train: {len(train_idx)} samples, {y_train.sum()} defects ({y_train.mean():.2%})")
    print(f"  Val:   {len(val_idx)} samples, {y_val.sum()} defects ({y_val.mean():.2%})")
    print(f"  Test:  {len(test_idx)} samples, {y_test.sum()} defects ({y_test.mean():.2%})")

    # Preprocess (fit on train only!)
    print("\n3. Fitting preprocessor (TRAINING DATA ONLY)...")
    preprocessor = WaferPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, legacy_train)
    X_val_proc = preprocessor.transform(X_val, legacy_val)
    X_test_proc = preprocessor.transform(X_test, legacy_test)

    print(f"  Output features: {X_train_proc.shape[1]}")

    # Save preprocessor
    preprocessor.save('models/preprocessor.joblib')

    # Train baseline
    baseline = train_baseline(X_train_proc, y_train, X_val_proc, y_val)
    joblib.dump(baseline, 'models/baseline_lr.joblib')

    # Create dataloaders
    print("\n4. Creating PyTorch dataloaders...")
    train_loader, val_loader = create_dataloaders(
        X_train_proc, y_train, X_val_proc, y_val, batch_size=64, seed=SEED
    )

    # Train autoencoder
    model = train_autoencoder(train_loader, val_loader, X_train_proc.shape[1])

    # Save model
    torch.save(model.state_dict(), 'models/wafer_sae.pt')

    # Threshold selection on validation
    print("\n5. Selecting threshold on validation set...")
    model.eval()
    with torch.no_grad():
        val_probs_full = []
        for xb, _ in val_loader:
            _, logits, _ = model(xb.to(DEVICE))
            val_probs_full.append(torch.sigmoid(logits).cpu().numpy())
        val_probs_full = np.concatenate(val_probs_full)

    best_threshold, best_f1 = find_best_threshold(y_val, val_probs_full, metric='f1')
    print(f"  Best threshold: {best_threshold:.3f} (F1={best_f1:.4f})")

    with open('reports/validation_threshold.json', 'w') as f:
        json.dump({
            'threshold': float(best_threshold),
            'validation_f1': float(best_f1),
            'method': 'maximize_f1_on_validation'
        }, f, indent=2)

    # Final evaluation on TEST SET
    print("\n6. Evaluating on test set (FROZEN THRESHOLD)...")
    print("   " + "="*56)
    print("   CAUTION: Final test evaluation - one time only!")
    print("   " + "="*56)

    # Create test loader
    X_test_t = torch.from_numpy(X_test_proc).float()
    y_test_t = torch.from_numpy(y_test).float()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_t, y_test_t),
        batch_size=64,
        shuffle=False
    )

    # Get predictions
    model.eval()
    with torch.no_grad():
        test_probs = []
        for xb, _ in test_loader:
            _, logits, _ = model(xb.to(DEVICE))
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
        test_probs = np.concatenate(test_probs)

    test_preds = (test_probs >= best_threshold).astype(int)

    # Compute metrics
    test_metrics = compute_metrics(y_test, test_preds, test_probs, threshold=best_threshold)

    print(f"\n   TEST SET RESULTS (Natural Prevalence = {test_metrics['defect_rate']:.2%}):")
    print(f"   {'-'*56}")
    print(f"   F1 Score:       {test_metrics['f1_score']:.4f}")
    print(f"   Precision:      {test_metrics['precision']:.4f}")
    print(f"   Recall:         {test_metrics['recall']:.4f}")
    print(f"   ROC-AUC:        {test_metrics['roc_auc']:.4f}")
    print(f"   Avg Precision:  {test_metrics['average_precision']:.4f}")
    print(f"   ")
    print(f"   Confusion Matrix:")
    print(f"     TN={test_metrics['true_negatives']:3d}  FP={test_metrics['false_positives']:3d}")
    print(f"     FN={test_metrics['false_negatives']:3d}  TP={test_metrics['true_positives']:3d}")
    print(f"   ")
    print(f"   False alarms: {test_metrics['false_alarms_per_1000']:.1f} per 1000 wafers")
    print(f"   Missed defects: {test_metrics['missed_defects_per_1000']:.1f} per 1000 wafers")

    # Save metrics
    save_metrics(test_metrics, 'reports/metrics.json')

    # Subgroup analysis
    print(f"\n7. Subgroup analysis...")
    subgroups = subgroup_metrics(
        y_test, test_preds, test_probs, legacy_test['mask'], best_threshold
    )

    with open('reports/subgroup_metrics.json', 'w') as f:
        json.dump(subgroups, f, indent=2)

    for group_name, group_metrics in subgroups.items():
        if 'sample_count' in group_metrics and group_metrics['sample_count'] > 0:
            f1 = group_metrics.get('f1_score', 0.0)
            print(f"   {group_name.upper()}: N={group_metrics['sample_count']}, "
                  f"F1={f1:.3f}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: models/")
    print(f"Metrics saved to: reports/")

    return test_metrics


if __name__ == "__main__":
    metrics = main()
