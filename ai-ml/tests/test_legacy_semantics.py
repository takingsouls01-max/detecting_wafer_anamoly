"""
TEST: Legacy Probe Station PS-L Semantic Contract (Bug #3)

Proves that categorical firmware codes are NOT scaled as continuous values.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor


def test_legacy_station_detection():
    """Verify exact sentinel detection for legacy wafers."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    # Check sentinel detection
    legacy_mask = dataset.legacy_mask
    X = dataset.get_features()

    # Verify: all legacy wafers have sensor_0 == 0.0 EXACTLY
    sensor_0_legacy = X[legacy_mask, 0]
    assert np.all(sensor_0_legacy == dataset.LEGACY_SENTINEL), \
        "Legacy wafers should have sensor_0 == 0.0 exactly"

    # Verify: modern wafers do NOT have sensor_0 == 0.0
    sensor_0_modern = X[~legacy_mask, 0]
    assert np.all(sensor_0_modern != dataset.LEGACY_SENTINEL), \
        "Modern wafers should NOT have sensor_0 == 0.0"

    print("✅ PASS: Legacy station detection via exact sentinel works correctly")
    print(f"   Legacy wafers: {legacy_mask.sum()} ({legacy_mask.mean():.2%})")
    print(f"   Modern wafers: {(~legacy_mask).sum()} ({(~legacy_mask).mean():.2%})")


def test_firmware_codes_are_categorical():
    """Verify legacy sensor_1 contains ONLY categorical codes."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    legacy_info = dataset.get_legacy_features()

    legacy_mask = legacy_info['mask']
    sensor_1_codes = legacy_info['sensor_1_codes']

    # Get codes for legacy wafers (excluding NaN)
    valid_codes = sensor_1_codes[legacy_mask]
    valid_codes = valid_codes[~np.isnan(valid_codes)]

    # All codes should be from known set
    unique_codes = np.unique(valid_codes)
    expected_codes = {100.0, 200.0, 300.0}

    assert set(unique_codes) <= expected_codes, \
        f"Unexpected firmware codes found: {set(unique_codes) - expected_codes}"

    print("✅ PASS: Legacy sensor_1 contains only categorical firmware codes")
    print(f"   Unique codes: {sorted(unique_codes)}")
    for code in sorted(unique_codes):
        count = (valid_codes == code).sum()
        print(f"   Code {code:.0f}: {count} wafers")


def test_categorical_codes_not_scaled():
    """
    CRITICAL TEST: Verify categorical codes are NOT scaled as continuous.

    Original bug: StandardScaler.fit_transform() on codes {100, 200, 300}
    converts them to z-scores, destroying their categorical meaning.
    """
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    # Fit preprocessor
    train_idx, _, _ = dataset.temporal_split(0.70, 0.15, 0.15)
    X_train = X[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    preprocessor = WaferPreprocessor()
    X_transformed = preprocessor.fit_transform(X_train, legacy_train)

    # Check output features
    # Should have: 590 continuous + 3 one-hot firmware + 1 legacy indicator = 594
    assert X_transformed.shape[1] == 594, \
        f"Expected 594 features, got {X_transformed.shape[1]}"

    # Verify firmware codes are one-hot encoded (not scaled)
    firmware_features = X_transformed[:, 590:593]  # Columns for firmware codes

    # One-hot encoded features should be binary (0 or 1)
    unique_vals = np.unique(firmware_features)
    assert set(unique_vals) <= {0.0, 1.0}, \
        f"Firmware features should be binary, got: {unique_vals}"

    # Each legacy wafer should have exactly ONE firmware code active
    legacy_mask_train = legacy_train['mask']
    legacy_firmware = firmware_features[legacy_mask_train]
    row_sums = legacy_firmware.sum(axis=1)

    # Most rows should have exactly 1.0 (one code active)
    # Some might have 0.0 if NaN codes were present
    assert np.all((row_sums == 0.0) | (row_sums == 1.0)), \
        "Each legacy wafer should have 0 or 1 firmware code active"

    print("✅ PASS: Categorical firmware codes are one-hot encoded, NOT scaled")
    print(f"   Output features: {X_transformed.shape[1]} (590 + 3 + 1)")
    print(f"   Firmware features are binary: {unique_vals}")
    print(f"   Legacy wafers with codes: {(row_sums == 1.0).sum()}/{legacy_mask_train.sum()}")


def test_modern_sensor1_remains_continuous():
    """Verify modern wafers' sensor_1 is still treated as continuous."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    # Get modern sensor_1 values
    modern_mask = ~legacy_info['mask']
    sensor_1_modern = X[modern_mask, 1]

    # Remove NaN values
    sensor_1_modern_valid = sensor_1_modern[~np.isnan(sensor_1_modern)]

    # Modern sensor_1 should have many unique continuous values
    # (not just 3 discrete codes)
    unique_values = np.unique(sensor_1_modern_valid)

    assert len(unique_values) > 100, \
        f"Modern sensor_1 should have many unique values, got {len(unique_values)}"

    # Values should NOT be clustered around {100, 200, 300}
    categorical_codes = {100.0, 200.0, 300.0}
    modern_is_categorical = np.isin(sensor_1_modern_valid, list(categorical_codes))

    assert modern_is_categorical.mean() < 0.01, \
        "Modern wafers should not have categorical codes"

    print("✅ PASS: Modern wafers' sensor_1 remains continuous")
    print(f"   Unique modern sensor_1 values: {len(unique_values)}")
    print(f"   Range: [{sensor_1_modern_valid.min():.1f}, {sensor_1_modern_valid.max():.1f}]")


def test_preprocessing_preserves_legacy_indicator():
    """Verify legacy indicator feature is added correctly."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")

    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    train_idx, _, _ = dataset.temporal_split(0.70, 0.15, 0.15)
    X_train = X[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    preprocessor = WaferPreprocessor()
    X_transformed = preprocessor.fit_transform(X_train, legacy_train)

    # Last column should be legacy indicator (0 or 1)
    legacy_indicator = X_transformed[:, -1]

    assert set(np.unique(legacy_indicator)) <= {0.0, 1.0}, \
        "Legacy indicator should be binary"

    # Should match original legacy mask
    expected_indicator = legacy_train['sensor_0_indicator']
    assert np.array_equal(legacy_indicator, expected_indicator), \
        "Legacy indicator doesn't match original mask"

    print("✅ PASS: Legacy indicator feature correctly added")
    print(f"   Legacy indicator values: {np.unique(legacy_indicator)}")
    print(f"   Legacy wafers in training: {(legacy_indicator == 1.0).sum()}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("="*60)
    print("BUG #3: LEGACY PROBE STATION SEMANTIC TESTS")
    print("="*60)

    test_legacy_station_detection()
    print()
    test_firmware_codes_are_categorical()
    print()
    test_categorical_codes_not_scaled()
    print()
    test_modern_sensor1_remains_continuous()
    print()
    test_preprocessing_preserves_legacy_indicator()
