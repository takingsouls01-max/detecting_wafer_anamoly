"""
TEST: Inference Batch-Order Independence

Verifies that model predictions are independent of:
1. Batch ordering (same samples in different order → same predictions)
2. Batch size (single vs batched inference → same predictions)
3. Companion samples (predictions don't depend on other samples in batch)

IMPORTANT: These properties should hold for the supervised autoencoder
since there are no attention mechanisms or batch-dependent operations.
"""

import numpy as np
import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waferguard.data import WaferDataset
from waferguard.preprocessing import WaferPreprocessor
from waferguard.model import WaferSAE


@pytest.fixture
def prepared_data():
    """Load and preprocess test data."""
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    X_train = X[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}

    X_test = X[test_idx]
    legacy_test = {k: v[test_idx] for k, v in legacy_info.items()}

    # Preprocess
    preprocessor = WaferPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, legacy_train)
    X_test_proc = preprocessor.transform(X_test, legacy_test)

    return X_test_proc


@pytest.fixture
def model():
    """Create a small trained model for testing."""
    # Use small model for faster testing
    model = WaferSAE(d_in=594, d_hidden=64, d_latent=16)
    model.eval()  # Set to evaluation mode
    torch.manual_seed(42)  # Deterministic
    return model


def test_batch_order_independence(prepared_data, model):
    """
    Verify predictions are independent of sample order in batch.

    Same samples in different orders should produce identical predictions.
    """
    X_test = prepared_data

    # Take a subset for testing
    n_samples = 32
    X_subset = X_test[:n_samples]

    # Create two different orderings
    indices_original = np.arange(n_samples)
    indices_shuffled = np.random.RandomState(42).permutation(n_samples)

    X_ordered = torch.from_numpy(X_subset[indices_original]).float()
    X_shuffled = torch.from_numpy(X_subset[indices_shuffled]).float()

    # Get predictions for both orderings
    with torch.no_grad():
        _, logits_ordered, _ = model(X_ordered)
        _, logits_shuffled, _ = model(X_shuffled)

    probs_ordered = torch.sigmoid(logits_ordered).numpy()
    probs_shuffled = torch.sigmoid(logits_shuffled).numpy()

    # Un-shuffle the shuffled predictions to match original order
    probs_unshuffled = np.zeros_like(probs_shuffled)
    probs_unshuffled[indices_shuffled] = probs_shuffled

    # Predictions should be identical (within numerical precision)
    max_diff = np.max(np.abs(probs_ordered - probs_unshuffled))

    assert max_diff < 1e-6, \
        f"Batch order affects predictions! Max diff: {max_diff:.2e}"

    print(f"PASS: Batch order independence verified")
    print(f"  Samples tested: {n_samples}")
    print(f"  Max prediction difference: {max_diff:.2e}")


def test_batch_size_independence(prepared_data, model):
    """
    Verify predictions are independent of batch size.

    Single-sample inference should match batched inference.
    """
    X_test = prepared_data

    # Take a subset
    n_samples = 16
    X_subset = X_test[:n_samples]
    X_tensor = torch.from_numpy(X_subset).float()

    # Get predictions in one batch
    with torch.no_grad():
        _, logits_batched, _ = model(X_tensor)
    probs_batched = torch.sigmoid(logits_batched).numpy()

    # Get predictions one-by-one
    probs_individual = []
    with torch.no_grad():
        for i in range(n_samples):
            X_single = X_tensor[i:i+1]  # [1, F]
            _, logits_single, _ = model(X_single)
            prob_single = torch.sigmoid(logits_single).numpy()
            probs_individual.append(prob_single[0])

    probs_individual = np.array(probs_individual)

    # Should be identical
    max_diff = np.max(np.abs(probs_batched - probs_individual))

    assert max_diff < 1e-6, \
        f"Batch size affects predictions! Max diff: {max_diff:.2e}"

    print(f"PASS: Batch size independence verified")
    print(f"  Samples tested: {n_samples}")
    print(f"  Max prediction difference: {max_diff:.2e}")


def test_companion_independence(prepared_data, model):
    """
    Verify prediction for sample A is independent of companion sample B.

    A's prediction should be identical whether batched with B or with C.
    """
    X_test = prepared_data

    # Get 3 samples: A, B, C
    X_A = torch.from_numpy(X_test[0:1]).float()
    X_B = torch.from_numpy(X_test[1:2]).float()
    X_C = torch.from_numpy(X_test[2:3]).float()

    # Predict A in batch with B
    with torch.no_grad():
        _, logits_AB, _ = model(torch.cat([X_A, X_B], dim=0))
    prob_A_with_B = torch.sigmoid(logits_AB[0]).item()

    # Predict A in batch with C
    with torch.no_grad():
        _, logits_AC, _ = model(torch.cat([X_A, X_C], dim=0))
    prob_A_with_C = torch.sigmoid(logits_AC[0]).item()

    # Predict A alone
    with torch.no_grad():
        _, logits_A, _ = model(X_A)
    prob_A_alone = torch.sigmoid(logits_A[0]).item()

    diff_AB = abs(prob_A_with_B - prob_A_alone)
    diff_AC = abs(prob_A_with_C - prob_A_alone)

    assert diff_AB < 1e-6, \
        f"Companion B affects A's prediction! Diff: {diff_AB:.2e}"
    assert diff_AC < 1e-6, \
        f"Companion C affects A's prediction! Diff: {diff_AC:.2e}"

    print(f"PASS: Companion independence verified")
    print(f"  A's prediction alone: {prob_A_alone:.6f}")
    print(f"  A's prediction with B: {prob_A_with_B:.6f} (diff: {diff_AB:.2e})")
    print(f"  A's prediction with C: {prob_A_with_C:.6f} (diff: {diff_AC:.2e})")


def test_reconstruction_batch_independence(prepared_data, model):
    """
    Verify reconstructions are also batch-independent.

    Sample's reconstruction should not depend on batch composition.
    """
    X_test = prepared_data

    # Get samples
    X_1 = torch.from_numpy(X_test[0:1]).float()
    X_2 = torch.from_numpy(X_test[1:2]).float()
    X_batch = torch.from_numpy(X_test[0:2]).float()

    # Reconstruct individually
    with torch.no_grad():
        recon_1_alone, _, _ = model(X_1)
        recon_2_alone, _, _ = model(X_2)

    # Reconstruct in batch
    with torch.no_grad():
        recon_batch, _, _ = model(X_batch)

    # Compare
    diff_1 = torch.max(torch.abs(recon_1_alone - recon_batch[0:1])).item()
    diff_2 = torch.max(torch.abs(recon_2_alone - recon_batch[1:2])).item()

    assert diff_1 < 1e-6, \
        f"Sample 1 reconstruction differs in batch! Diff: {diff_1:.2e}"
    assert diff_2 < 1e-6, \
        f"Sample 2 reconstruction differs in batch! Diff: {diff_2:.2e}"

    print(f"PASS: Reconstruction batch independence verified")
    print(f"  Sample 1 max reconstruction diff: {diff_1:.2e}")
    print(f"  Sample 2 max reconstruction diff: {diff_2:.2e}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("="*80)
    print("INFERENCE BATCH-ORDER INDEPENDENCE TESTS")
    print("="*80)

    # Load data
    dataset = WaferDataset("data/secom_telemetry_modified.csv")
    train_idx, val_idx, test_idx = dataset.temporal_split(0.70, 0.15, 0.15)

    X = dataset.get_features()
    legacy_info = dataset.get_legacy_features()

    X_train = X[train_idx]
    legacy_train = {k: v[train_idx] for k, v in legacy_info.items()}
    X_test = X[test_idx]
    legacy_test = {k: v[test_idx] for k, v in legacy_info.items()}

    preprocessor = WaferPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, legacy_train)
    X_test_proc = preprocessor.transform(X_test, legacy_test)

    # Create model
    model = WaferSAE(d_in=594, d_hidden=64, d_latent=16)
    model.eval()
    torch.manual_seed(42)

    print()
    test_batch_order_independence(X_test_proc, model)
    print()
    test_batch_size_independence(X_test_proc, model)
    print()
    test_companion_independence(X_test_proc, model)
    print()
    test_reconstruction_batch_independence(X_test_proc, model)
