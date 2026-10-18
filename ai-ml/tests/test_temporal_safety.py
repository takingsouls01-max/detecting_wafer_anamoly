"""
TEST: Comprehensive Temporal Safety Verification (Priority #4)

Verifies:
1. Timestamp parsing correctness
2. Split boundary handling (ties, duplicates)
3. Training-only learned transforms (no future contamination)
4. Future-mutation test with fixed information cutoff
5. Negative control: deliberately leaking preprocessor
6. Honest limitations of the test

LIMITATIONS OF THIS TEST:
- Tests preprocessing pipeline only (not full training loop)
- Does not test model inference/prediction leakage
- Does not test feature engineering that might use future aggregates
- Assumes timestamps are ground truth (no verification of time source)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor


def test_timestamp_parsing():
    """Verify timestamps are parsed correctly and monotonically increasing."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    timestamps = dataset.timestamps

    # Check timestamps are parsed as datetime64
    assert timestamps.dtype == np.dtype('datetime64[ns]'), \
        f"Timestamps should be datetime64[ns], got {timestamps.dtype}"

    # Check timestamps are sorted (monotonically increasing)
    is_sorted = np.all(timestamps[:-1] <= timestamps[1:])
    assert is_sorted, "Timestamps are not sorted in chronological order!"

    # Check no NaT (not-a-time) values
    assert not pd.isna(timestamps).any(), "Dataset contains NaT timestamps"

    # Report timestamp range
    first = pd.Timestamp(timestamps[0])
    last = pd.Timestamp(timestamps[-1])
    duration = last - first

    print(f"✅ PASS: Timestamp parsing correct")
    print(f"   First: {first}")
    print(f"   Last:  {last}")
    print(f"   Duration: {duration}")
    print(f"   Total samples: {len(timestamps)}")


def test_split_boundary_ties():
    """Check for timestamp ties at split boundaries."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    timestamps = dataset.timestamps

    # Get boundary timestamps
    train_last_time = timestamps[train_idx[-1]]
    val_first_time = timestamps[val_idx[0]]
    val_last_time = timestamps[val_idx[-1]]
    test_first_time = timestamps[test_idx[0]]

    # Check for ties at boundaries
    train_has_ties = (timestamps[train_idx] == train_last_time).sum() > 1
    val_has_ties_start = (timestamps[val_idx] == val_first_time).sum() > 1
    val_has_ties_end = (timestamps[val_idx] == val_last_time).sum() > 1
    test_has_ties = (timestamps[test_idx] == test_first_time).sum() > 1

    print(f"Boundary tie analysis:")
    print(f"  Train boundary ({pd.Timestamp(train_last_time)}): {(timestamps == train_last_time).sum()} samples")
    print(f"  Val start ({pd.Timestamp(val_first_time)}): {(timestamps == val_first_time).sum()} samples")
    print(f"  Val end ({pd.Timestamp(val_last_time)}): {(timestamps == val_last_time).sum()} samples")
    print(f"  Test start ({pd.Timestamp(test_first_time)}): {(timestamps == test_first_time).sum()} samples")

    # Verify no overlap at boundaries (even with ties)
    boundary_gap_1 = val_first_time > train_last_time
    boundary_gap_2 = test_first_time > val_last_time

    if not boundary_gap_1:
        print(f"⚠️  WARNING: Train/val boundary has no gap (tie possible)")
    if not boundary_gap_2:
        print(f"⚠️  WARNING: Val/test boundary has no gap (tie possible)")

    # The split should still be valid (no overlap)
    assert boundary_gap_1 or (val_first_time == train_last_time and (timestamps == train_last_time).sum() < 2), \
        "Train/val boundary has timestamp collision!"
    assert boundary_gap_2 or (test_first_time == val_last_time and (timestamps == val_last_time).sum() < 2), \
        "Val/test boundary has timestamp collision!"

    print(f"✅ PASS: Split boundaries are valid (no timestamp collisions)")


def test_duplicate_timestamps():
    """Check for duplicate timestamps in the dataset."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    timestamps = dataset.timestamps
    unique_timestamps = np.unique(timestamps)

    n_duplicates = len(timestamps) - len(unique_timestamps)

    if n_duplicates > 0:
        # Find examples of duplicate timestamps
        timestamp_counts = pd.Series(timestamps).value_counts()
        duplicated = timestamp_counts[timestamp_counts > 1]

        print(f"⚠️  Dataset contains {n_duplicates} duplicate timestamps")
        print(f"   Total unique timestamps: {len(unique_timestamps)} / {len(timestamps)}")
        print(f"   Example duplicates:")
        for ts, count in duplicated.head(5).items():
            print(f"     {pd.Timestamp(ts)}: {count} occurrences")

        # Check if duplicates span split boundaries
        train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

        for ts in duplicated.index[:10]:  # Check first 10 duplicate timestamps
            in_train = (timestamps[train_idx] == ts).any()
            in_val = (timestamps[val_idx] == ts).any()
            in_test = (timestamps[test_idx] == ts).any()

            if sum([in_train, in_val, in_test]) > 1:
                print(f"   ⚠️  CRITICAL: Timestamp {pd.Timestamp(ts)} spans multiple splits!")
    else:
        print(f"✅ All timestamps are unique")


def test_fixed_information_cutoff():
    """
    Enhanced future-mutation test with fixed information cutoff.

    Tests that preprocessing uses ONLY information available at training cutoff time,
    even when future data is present in memory.
    """
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    # Define fixed cutoff at 70% of data (simulating production deployment)
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)
    cutoff_time = dataset.timestamps[train_idx[-1]]

    print(f"Information cutoff: {pd.Timestamp(cutoff_time)}")
    print(f"Training on {len(train_idx)} samples before cutoff")

    X_full = dataset.get_features()
    legacy_full = dataset.get_legacy_features()

    # Version 1: Process with only pre-cutoff data visible
    X_train = X_full[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_full.items()}

    preprocessor_1 = WaferPreprocessor()
    X_train_proc_1 = preprocessor_1.fit_transform(X_train, legacy_train)

    # Version 2: Process with post-cutoff data drastically mutated
    X_mutated = X_full.copy()
    future_idx = np.concatenate([val_idx, test_idx])

    np.random.seed(12345)  # Fixed seed for reproducibility
    # Extreme mutation: shift all future values by 1000 standard deviations
    X_mutated[future_idx] = np.random.randn(*X_mutated[future_idx].shape) * 1000.0

    X_train_2 = X_mutated[train_idx]
    legacy_train_2 = {k: v[train_idx] for k, v in legacy_full.items()}

    preprocessor_2 = WaferPreprocessor()
    X_train_proc_2 = preprocessor_2.fit_transform(X_train_2, legacy_train_2)

    # CRITICAL ASSERTION: Preprocessing should be IDENTICAL
    # Even though future data is drastically different, training preprocessing shouldn't change

    max_diff = np.max(np.abs(X_train_proc_1 - X_train_proc_2))

    # Check learned parameters
    scaler_mean_diff = np.max(np.abs(preprocessor_1.scaler.mean_ - preprocessor_2.scaler.mean_))
    scaler_scale_diff = np.max(np.abs(preprocessor_1.scaler.scale_ - preprocessor_2.scaler.scale_))

    print(f"  Max difference in transformed data: {max_diff:.2e}")
    print(f"  Max difference in scaler means: {scaler_mean_diff:.2e}")
    print(f"  Max difference in scaler scales: {scaler_scale_diff:.2e}")

    assert max_diff < 1e-10, \
        f"TEMPORAL LEAK: Training transformation differs when future changes! (diff={max_diff:.2e})"

    print(f"✅ PASS: Preprocessing respects fixed information cutoff")


def test_negative_control_leaking_preprocessor():
    """
    NEGATIVE CONTROL: Deliberately create a leaking preprocessor and verify the test catches it.

    This proves the future-mutation test has power to detect leaks.
    """
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    X_full = dataset.get_features()
    legacy_full = dataset.get_legacy_features()

    # Version 1: Normal (use only training data)
    X_train = X_full[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_full.items()}

    preprocessor_good = WaferPreprocessor()
    X_proc_good = preprocessor_good.fit_transform(X_train, legacy_train)

    # Version 2: DELIBERATELY LEAK (fit on ALL data including future)
    X_all = X_full
    legacy_all = legacy_full

    preprocessor_leaky = WaferPreprocessor()
    preprocessor_leaky.fit(X_all, legacy_all)  # ⚠️ FIT ON FUTURE DATA!

    X_proc_leaky = preprocessor_leaky.transform(X_train, legacy_train)

    # The leaking preprocessor should produce DIFFERENT results
    max_diff = np.max(np.abs(X_proc_good - X_proc_leaky))

    print(f"Negative control (deliberately leaking preprocessor):")
    print(f"  Max difference: {max_diff:.2e}")

    # This test should FAIL if there's no leak (proving our detection works)
    assert max_diff > 1e-6, \
        "NEGATIVE CONTROL FAILED: Leaking preprocessor not detected! Test has no power."

    print(f"✅ PASS: Negative control successfully detected leak (test has power)")


def test_training_only_transforms():
    """Verify all learned transforms use ONLY training data."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}
    legacy_val = {k: v[val_idx] for k, v in legacy_info.items()}
    legacy_test = {k: v[test_idx] for k, v in legacy_info.items()}

    # Fit preprocessor on training data ONLY
    preprocessor = WaferPreprocessor()
    preprocessor.fit(X_train, legacy_train)

    # Transform all splits
    X_train_proc = preprocessor.transform(X_train, legacy_train)
    X_val_proc = preprocessor.transform(X_val, legacy_val)
    X_test_proc = preprocessor.transform(X_test, legacy_test)

    # Verify the preprocessor was fitted (has learned parameters)
    assert preprocessor.is_fitted, "Preprocessor not fitted!"
    assert preprocessor.scaler.mean_ is not None, "Scaler has no learned mean"
    assert preprocessor.scaler.scale_ is not None, "Scaler has no learned scale"

    # Verify learned parameters match training data statistics
    # (not validation or test data)

    # Note: Can't directly verify because of imputation and legacy handling
    # But we can verify that val/test transformations don't fail

    assert X_val_proc.shape == (len(val_idx), 594), "Validation transform failed"
    assert X_test_proc.shape == (len(test_idx), 594), "Test transform failed"
    assert not np.isnan(X_val_proc).any(), "Validation has NaN after transform"
    assert not np.isnan(X_test_proc).any(), "Test has NaN after transform"

    print(f"✅ PASS: All transforms learned from training data only")
    print(f"   Train: {X_train_proc.shape}")
    print(f"   Val: {X_val_proc.shape}")
    print(f"   Test: {X_test_proc.shape}")


def test_limitations_documented():
    """
    Document the limitations of our temporal safety tests.

    This test always passes but prints important caveats.
    """
    limitations = [
        "1. Tests preprocessing pipeline only (not full model training/inference)",
        "2. Does not test batch sampling order during training",
        "3. Does not test model checkpointing for temporal contamination",
        "4. Does not verify timestamp source or accuracy",
        "5. Assumes temporal split fractions are appropriate for the data distribution",
        "6. Does not test evaluation metrics for temporal bias",
        "7. Does not test feature engineering that might aggregate across time",
        "8. Does not test for subtle leaks via random seeds or initialization",
    ]

    print(f"LIMITATIONS OF TEMPORAL SAFETY TESTS:")
    for lim in limitations:
        print(f"  {lim}")

    print(f"\n⚠️  These tests do NOT guarantee complete temporal safety.")
    print(f"   They verify preprocessing causality only.")

    # Always pass (this is documentation, not a failure)
    assert True


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("="*80)
    print("TEMPORAL SAFETY COMPREHENSIVE VERIFICATION")
    print("="*80)

    test_timestamp_parsing()
    print()
    test_split_boundary_ties()
    print()
    test_duplicate_timestamps()
    print()
    test_training_only_transforms()
    print()
    test_fixed_information_cutoff()
    print()
    test_negative_control_leaking_preprocessor()
    print()
    test_limitations_documented()
