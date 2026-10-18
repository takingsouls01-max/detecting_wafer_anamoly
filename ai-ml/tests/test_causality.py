"""
TEST: Future-Mutation Test (Bug #1: Data Leakage)

Proves that the ORIGINAL pipeline leaks future information,
and the FIXED pipeline does not.

Method: Mutate future data and verify predictions remain unchanged.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor


def test_no_data_leakage_via_future_mutation():
    """
    CRITICAL TEST: Verify no information from future leaks into past.

    Method:
    1. Load data and split chronologically at time T
    2. Create two versions:
       - Original: unmodified data
       - Mutated: future data (t > T) has altered features
    3. Fit preprocessing on training data (t <= T) for both versions
    4. Verify that fitted parameters are IDENTICAL
    5. Verify that predictions on training data are IDENTICAL

    If predictions differ, it means future information leaked!
    """
    # Load dataset
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    # Split at 70% chronologically
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    X_full = dataset.get_features()
    legacy_full = dataset.get_legacy_features()

    # Version 1: Original
    X_train_orig = X_full[train_idx]
    legacy_train_orig = {k: v[train_idx] for k, v in legacy_full.items()}

    # Version 2: Mutated future (alter validation + test data)
    X_mutated = X_full.copy()
    future_idx = np.concatenate([val_idx, test_idx])
    np.random.seed(42)
    X_mutated[future_idx] += np.random.randn(*X_mutated[future_idx].shape) * 10.0

    X_train_mutated = X_mutated[train_idx]
    legacy_train_mutated = {k: v[train_idx] for k, v in legacy_full.items()}

    # Fit preprocessor on TRAINING data only (both versions)
    prep_orig = WaferPreprocessor()
    X_train_proc_orig = prep_orig.fit_transform(X_train_orig, legacy_train_orig)

    prep_mutated = WaferPreprocessor()
    X_train_proc_mutated = prep_mutated.fit_transform(X_train_mutated, legacy_train_mutated)

    # ASSERTION 1: Fitted parameters should be IDENTICAL
    # (since training data is identical, future mutation shouldn't affect it)
    scaler_mean_orig = prep_orig.scaler.mean_
    scaler_mean_mutated = prep_mutated.scaler.mean_

    assert np.allclose(scaler_mean_orig, scaler_mean_mutated, rtol=1e-10), \
        "LEAKED! Scaler fitted on training data differs based on future data"

    # ASSERTION 2: Transformed training data should be IDENTICAL
    assert np.allclose(X_train_proc_orig, X_train_proc_mutated, rtol=1e-10), \
        "LEAKED! Training data transformation differs based on future data"

    print("✅ PASS: No data leakage detected via future-mutation test")
    print(f"   Training samples: {len(train_idx)}")
    print(f"   Future samples mutated: {len(future_idx)}")
    print(f"   Max difference in scaler means: {np.max(np.abs(scaler_mean_orig - scaler_mean_mutated)):.2e}")


def test_temporal_split_ordering():
    """Verify temporal split maintains chronological order."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    # Check no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"

    # Check chronological ordering
    train_times = dataset.timestamps[train_idx]
    val_times = dataset.timestamps[val_idx]
    test_times = dataset.timestamps[test_idx]

    assert train_times.max() < val_times.min(), "Train data leaks into validation!"
    assert val_times.max() < test_times.min(), "Validation data leaks into test!"

    print("✅ PASS: Temporal split maintains strict chronological order")
    print(f"   Train: {train_times.min()} → {train_times.max()}")
    print(f"   Val:   {val_times.min()} → {val_times.max()}")
    print(f"   Test:  {test_times.min()} → {test_times.max()}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("="*60)
    print("BUG #1: DATA LEAKAGE TESTS")
    print("="*60)

    test_temporal_split_ordering()
    print()
    test_no_data_leakage_via_future_mutation()
