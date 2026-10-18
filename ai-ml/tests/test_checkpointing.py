"""
Test checkpointing behavior to ensure saved model state is truly independent.

FINDING: PyTorch's state_dict().copy() creates a shallow copy where tensors
may share underlying storage. During training, these "saved" tensors can be
modified, corrupting the checkpoint. Use copy.deepcopy() instead.
"""

import sys
sys.path.insert(0, 'src')

import copy
import torch
import numpy as np
from waferguard.model import WaferSAE


def test_deepcopy_checkpoint_independence():
    """
    Verify that deep-copied checkpoint remains unchanged after further optimization
    and restores exact same predictions.

    This test proves that:
    1. Saved state (with deepcopy) is independent of model's continued training
    2. Restoring checkpoint produces exact same predictions as before training
    3. Shallow copy FAILS this test (documented in test_shallow_copy_fails)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    model = WaferSAE(d_in=10, d_hidden=8, d_latent=4)

    # Create dummy input
    X = torch.randn(5, 10)

    # Get initial predictions
    model.eval()
    with torch.no_grad():
        initial_preds = model.predict(X).numpy().copy()

    # Save checkpoint with DEEP COPY (the fix)
    checkpoint = copy.deepcopy(model.state_dict())

    # Continue training (modify model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(10):
        optimizer.zero_grad()
        reconstructed, logits, latent = model(X)
        loss = torch.nn.functional.mse_loss(reconstructed, X)
        loss.backward()
        optimizer.step()

    # Get post-training predictions (should be different)
    model.eval()
    with torch.no_grad():
        posttrain_preds = model.predict(X).numpy()

    # Verify training actually changed the model
    assert not np.allclose(initial_preds, posttrain_preds, rtol=1e-3), \
        "Training should have changed model predictions"

    # Restore from checkpoint
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        restored_preds = model.predict(X).numpy()

    # Checkpoint should restore exact original predictions
    np.testing.assert_allclose(
        restored_preds, initial_preds, rtol=1e-5,
        err_msg="Deep copy checkpoint MUST restore original predictions exactly"
    )

    # Verify restored predictions differ from post-training
    assert not np.allclose(restored_preds, posttrain_preds, rtol=1e-3), \
        "Restored model should differ from continued training"

    print("PASS: Deep copy checkpoint independence verified")
    print(f"  Initial pred mean: {initial_preds.mean():.6f}")
    print(f"  Post-train pred mean: {posttrain_preds.mean():.6f}")
    print(f"  Restored pred mean: {restored_preds.mean():.6f}")
    print(f"  Max restoration error: {np.abs(restored_preds - initial_preds).max():.2e}")


def test_shallow_copy_fails():
    """
    Demonstrate that shallow copy (.copy()) FAILS to preserve checkpoint.

    This documents the BUG that was found in the original code.
    PyTorch's state_dict().copy() creates tensor objects that share storage
    with the model's parameters, so they get modified during training.
    """
    torch.manual_seed(42)

    model = WaferSAE(d_in=10, d_hidden=8, d_latent=4)
    X = torch.randn(5, 10)

    # Get initial state
    model.eval()
    with torch.no_grad():
        initial_preds = model.predict(X).numpy().copy()

    # Save with SHALLOW copy (the bug)
    checkpoint_shallow = model.state_dict().copy()
    checkpoint_deep = copy.deepcopy(model.state_dict())

    # Train
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(10):
        optimizer.zero_grad()
        reconstructed, logits, latent = model(X)
        loss = torch.nn.functional.mse_loss(reconstructed, X)
        loss.backward()
        optimizer.step()

    # Check if shallow checkpoint was corrupted
    shallow_corrupted = False
    for key in checkpoint_shallow:
        if not torch.equal(checkpoint_shallow[key], checkpoint_deep[key]):
            shallow_corrupted = True
            break

    assert shallow_corrupted, \
        "Shallow copy checkpoint SHOULD be corrupted (this documents the bug)"

    # Restore from shallow copy and verify it FAILS to restore original
    model.load_state_dict(checkpoint_shallow)
    model.eval()
    with torch.no_grad():
        restored_shallow = model.predict(X).numpy()

    # Shallow copy FAILS to restore original (this is the bug)
    shallow_fails = not np.allclose(restored_shallow, initial_preds, rtol=1e-3)
    assert shallow_fails, \
        "Shallow copy SHOULD fail to restore (documents the bug we fixed)"

    # Deep copy works correctly
    model.load_state_dict(checkpoint_deep)
    model.eval()
    with torch.no_grad():
        restored_deep = model.predict(X).numpy()

    deep_works = np.allclose(restored_deep, initial_preds, rtol=1e-5)
    assert deep_works, \
        "Deep copy SHOULD work (validates the fix)"

    print("PASS: Documented that shallow copy fails (bug) and deep copy works (fix)")
    print(f"  Shallow copy corrupted: YES (expected)")
    print(f"  Shallow restore error: {np.abs(restored_shallow - initial_preds).max():.4f}")
    print(f"  Deep copy corrupted: NO (expected)")
    print(f"  Deep restore error: {np.abs(restored_deep - initial_preds).max():.2e}")


if __name__ == '__main__':
    print("="*70)
    print("TEST 1: Verify deep copy works correctly")
    print("="*70)
    test_deepcopy_checkpoint_independence()

    print("\n" + "="*70)
    print("TEST 2: Document shallow copy bug")
    print("="*70)
    test_shallow_copy_fails()

    print("\n" + "="*70)
    print("CONCLUSION: Always use copy.deepcopy() for PyTorch checkpoints")
    print("="*70)
