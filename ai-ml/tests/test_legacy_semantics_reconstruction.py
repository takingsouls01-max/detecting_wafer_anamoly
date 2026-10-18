"""
TEST: Legacy Semantics with Reconstruction Mask (Bug #3 - Extended)

Verifies:
1. Legacy sensor_0 sentinel (0.0) is NOT scaled as a physical measurement
2. Legacy sensor_1 codes are one-hot encoded, not scaled
3. Reconstruction mask excludes sentinel and engineered features
4. Model uses reconstruction mask correctly
"""

import numpy as np
import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor
from waferguard.model import WaferSAE, reconstruction_loss


def test_legacy_sensor0_not_scaled():
    """CRITICAL: Verify legacy sensor_0 sentinel is NOT treated as a measurement."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    # Split and prepare
    train_idx, _, _ = dataset.temporal_split(0.70, 0.15, 0.15)
    X_train = X[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    # Fit preprocessor
    preprocessor = WaferPreprocessor()
    X_transformed = preprocessor.fit_transform(X_train, legacy_train)

    # Check legacy sensor_0 transformation
    legacy_mask = legacy_train['mask']
    sensor_0_legacy_orig = X_train[legacy_mask, 0]
    sensor_0_legacy_trans = X_transformed[legacy_mask, 0]

    # Original should all be 0.0 (sentinel)
    assert np.all(sensor_0_legacy_orig == 0.0), \
        "Legacy sensor_0 should be exactly 0.0 (sentinel)"

    # After transformation, it should be imputed (mean of modern values)
    # NOT scaled as -3.18 (which was the bug)
    print(f"Original legacy sensor_0: {sensor_0_legacy_orig[:5]}")
    print(f"Transformed legacy sensor_0: {sensor_0_legacy_trans[:5]}")
    print(f"  Mean: {sensor_0_legacy_trans.mean():.4f}")
    print(f"  Std: {sensor_0_legacy_trans.std():.4f}")

    # The transformed values should be near zero (imputed mean, then scaled)
    # But NOT all identical -3.18 (which indicates scaling the sentinel directly)
    unique_trans = len(np.unique(sensor_0_legacy_trans))

    # If they're all the same value and far from zero, the bug is still present
    if unique_trans == 1 and np.abs(sensor_0_legacy_trans[0]) > 1.0:
        pytest.fail(
            f"Legacy sensor_0 sentinel appears to be scaled directly to {sensor_0_legacy_trans[0]:.4f}. "
            "This treats the sentinel 0.0 as a physical measurement. "
            "It should be set to NaN and imputed."
        )

    print("✅ PASS: Legacy sensor_0 sentinel is NOT scaled as a measurement")


def test_reconstruction_mask_generation():
    """Verify reconstruction mask correctly excludes sentinel/engineered features."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    train_idx, _, _ = dataset.temporal_split(0.70, 0.15, 0.15)
    X_train = X[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    preprocessor = WaferPreprocessor()
    preprocessor.fit_transform(X_train, legacy_train)

    # Generate reconstruction mask
    recon_mask = preprocessor.get_reconstruction_mask(legacy_train)

    print(f"Reconstruction mask shape: {recon_mask.shape}")
    print(f"Expected: ({len(train_idx)}, 594)")

    assert recon_mask.shape == (len(train_idx), 594), \
        f"Mask shape should be (N, 594), got {recon_mask.shape}"

    # Check mask for legacy wafers
    legacy_mask = legacy_train['mask']
    legacy_recon = recon_mask[legacy_mask]

    # Sensor_0 should be excluded for legacy wafers (column 0)
    assert np.all(legacy_recon[:, 0] == False), \
        "Legacy sensor_0 (sentinel) should be excluded from reconstruction"

    # Sensor_1 should be excluded for legacy wafers (column 1)
    assert np.all(legacy_recon[:, 1] == False), \
        "Legacy sensor_1 (categorical) should be excluded from reconstruction"

    # Firmware one-hot (columns 590-592) should be excluded for ALL samples
    assert np.all(recon_mask[:, 590:593] == False), \
        "Firmware one-hot features should be excluded (engineered features)"

    # Legacy indicator (column 593) should be excluded for ALL samples
    assert np.all(recon_mask[:, 593] == False), \
        "Legacy indicator should be excluded (metadata)"

    # Modern wafers should have sensor_0 and sensor_1 included
    modern_mask = ~legacy_mask
    if modern_mask.any():
        modern_recon = recon_mask[modern_mask]
        assert np.all(modern_recon[:, 0] == True), \
            "Modern sensor_0 should be included in reconstruction"
        assert np.all(modern_recon[:, 1] == True), \
            "Modern sensor_1 should be included in reconstruction"

    # Count included features per wafer type
    n_included_legacy = legacy_recon.sum(axis=1).mean()
    n_included_modern = modern_recon.sum(axis=1).mean() if modern_mask.any() else 0

    print(f"Average features included in reconstruction:")
    print(f"  Legacy wafers: {n_included_legacy:.1f} / 594")
    print(f"  Modern wafers: {n_included_modern:.1f} / 594")
    print(f"  Excluded for all: 4 (firmware one-hot + legacy indicator)")
    print(f"  Excluded for legacy: +2 (sensor_0 sentinel + sensor_1 categorical)")

    print("✅ PASS: Reconstruction mask correctly excludes sentinel/engineered features")


def test_reconstruction_loss_with_mask():
    """Verify reconstruction loss correctly applies mask."""
    # Create dummy data
    batch_size = 4
    n_features = 594

    # Dummy reconstructed and target
    reconstructed = torch.randn(batch_size, n_features)
    target = torch.randn(batch_size, n_features)

    # Create mask: exclude last 4 features (firmware + legacy indicator)
    mask = torch.ones(batch_size, n_features, dtype=torch.bool)
    mask[:, -4:] = False  # Exclude last 4 features
    mask[0:2, 0] = False  # Exclude sensor_0 for first 2 samples (legacy)
    mask[0:2, 1] = False  # Exclude sensor_1 for first 2 samples (legacy)

    # Compute loss with mask
    loss_masked = reconstruction_loss(reconstructed, target, mask=mask)

    # Compute loss without mask
    loss_full = reconstruction_loss(reconstructed, target, mask=None)

    print(f"Loss with mask: {loss_masked.item():.6f}")
    print(f"Loss without mask: {loss_full.item():.6f}")

    # They should be different (mask excludes some features)
    assert not torch.allclose(loss_masked, loss_full), \
        "Masked and unmasked losses should differ"

    # Manual computation to verify
    squared_error = (reconstructed - target) ** 2
    masked_error = squared_error * mask.float()
    n_valid = mask.sum(dim=1).float()
    manual_loss = (masked_error.sum(dim=1) / n_valid).mean()

    print(f"Manual verification: {manual_loss.item():.6f}")

    assert torch.allclose(loss_masked, manual_loss, rtol=1e-5), \
        "Masked loss doesn't match manual calculation"

    print("✅ PASS: Reconstruction loss correctly applies mask")


def test_full_pipeline_with_mask():
    """Integration test: Full pipeline from data to masked reconstruction loss."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    X = dataset.get_features()
    y = dataset.y
    legacy_info = dataset.get_legacy_features()

    train_idx, _, _ = dataset.temporal_split(0.70, 0.15, 0.15)
    X_train = X[train_idx]
    y_train = y[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    # Preprocess
    preprocessor = WaferPreprocessor()
    X_transformed = preprocessor.fit_transform(X_train, legacy_train)

    # Get reconstruction mask
    recon_mask = preprocessor.get_reconstruction_mask(legacy_train)

    # Create model
    model = WaferSAE(d_in=594, d_hidden=64, d_latent=16)
    model.eval()

    # Create mini-batch
    batch_size = 32
    X_batch = torch.from_numpy(X_transformed[:batch_size]).float()
    y_batch = torch.from_numpy(y_train[:batch_size]).float()
    mask_batch = torch.from_numpy(recon_mask[:batch_size]).bool()

    # Forward pass
    reconstructed, logits, latent = model(X_batch)

    # Compute masked reconstruction loss
    loss_masked = reconstruction_loss(reconstructed, X_batch, mask=mask_batch)
    loss_unmasked = reconstruction_loss(reconstructed, X_batch, mask=None)

    print(f"Batch size: {batch_size}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mask shape: {mask_batch.shape}")
    print(f"Masked loss: {loss_masked.item():.6f}")
    print(f"Unmasked loss: {loss_unmasked.item():.6f}")
    print(f"Difference: {(loss_unmasked - loss_masked).item():.6f}")

    assert reconstructed.shape == X_batch.shape, "Shape mismatch"
    assert not torch.allclose(loss_masked, loss_unmasked), \
        "Masked and unmasked losses should differ"

    print("✅ PASS: Full pipeline with reconstruction mask works correctly")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("="*80)
    print("LEGACY SEMANTICS + RECONSTRUCTION MASK TESTS")
    print("="*80)

    test_legacy_sensor0_not_scaled()
    print()
    test_reconstruction_mask_generation()
    print()
    test_reconstruction_loss_with_mask()
    print()
    test_full_pipeline_with_mask()
