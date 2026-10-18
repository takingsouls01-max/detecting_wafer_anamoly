"""
TEST: Loss Integrity and Tensor Shape Correctness (Bug #2)

Proves that tensor shapes are handled correctly in loss computation.
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waferguard.model import WaferSAE, reconstruction_loss, combined_loss


def test_reconstruction_loss_shape_handling():
    """
    CRITICAL TEST: Verify reconstruction loss handles shapes correctly.

    Original bug: batch_x has shape [B, 1, F], reconstructed has [B, F]
    This causes broadcasting to [B, F, F] which is WRONG.
    """
    B, F = 8, 594

    # Simulate decoder output (correct shape)
    reconstructed = torch.randn(B, F)

    # Simulate input with channel dimension (as in original broken code)
    target_with_channel = torch.randn(B, 1, F)

    # Simulate flat input (correct)
    target_flat = target_with_channel.squeeze(1)

    # FIXED loss function should handle both
    loss_1 = reconstruction_loss(reconstructed, target_flat)
    loss_2 = reconstruction_loss(reconstructed, target_with_channel)  # Should flatten internally

    assert loss_1.shape == torch.Size([]), "Loss should be scalar"
    assert loss_2.shape == torch.Size([]), "Loss should be scalar"
    assert torch.isfinite(loss_1), "Loss should be finite"
    assert torch.isfinite(loss_2), "Loss should be finite"

    print("[OK] PASS: Reconstruction loss handles shape variations correctly")
    print(f"   Loss with flat input [B, F]: {loss_1.item():.4f}")
    print(f"   Loss with channel input [B, 1, F]: {loss_2.item():.4f}")


def test_broken_loss_demonstrates_bug():
    """
    Demonstrate the ORIGINAL bug with explicit broadcasting error.
    """
    B, F = 8, 594

    # Reconstructed from decoder: [B, F]
    reconstructed = torch.randn(B, F)

    # Input with extra channel dim: [B, 1, F]
    target = torch.randn(B, 1, F)

    # WRONG way (original bug): direct subtraction without flattening
    diff = reconstructed - target  # Broadcasting happens here!

    # This creates [B, B, F] due to broadcasting rules:
    # [B, F] - [B, 1, F] → [1, B, F] - [B, 1, F] → [B, B, F]
    # (reconstructed is expanded to [1,B,F], then broadcast to [B,B,F])
    expected_broken_shape = (B, B, F)

    print(f"\nORIGINAL BUG DEMONSTRATION:")
    print(f"   reconstructed.shape: {reconstructed.shape}")
    print(f"   target.shape: {target.shape}")
    print(f"   diff.shape (after broadcasting): {diff.shape}")
    print(f"   Expected broken shape: {expected_broken_shape}")

    # This would cause memory explosion and wrong gradients!
    assert diff.shape == expected_broken_shape, \
        "Bug demonstration: broadcasting creates wrong shape"

    # The loss would be computed over wrong dimensions
    broken_loss = torch.mean(diff ** 2)
    print(f"   Broken loss value: {broken_loss.item():.4f}")
    print(f"   Memory used: {diff.numel() * 4 / 1024 / 1024:.2f} MB (vs {B*F*4/1024:.2f} KB correct)")

    print("[OK] PASS: Successfully demonstrated the original broadcasting bug")


def test_model_forward_shapes():
    """Verify model forward pass produces correct shapes."""
    d_in = 594
    model = WaferSAE(d_in, d_hidden=128, d_latent=32)
    model.eval()

    batch_sizes = [1, 16, 64, 71]  # Test various batch sizes including 1 and odd numbers

    for B in batch_sizes:
        # Test with flat input [B, F]
        x_flat = torch.randn(B, d_in)
        recon, logits, latent = model(x_flat)

        assert recon.shape == (B, d_in), f"Wrong recon shape for batch {B}: {recon.shape}"
        assert logits.shape == (B,), f"Wrong logits shape for batch {B}: {logits.shape}"
        assert latent.shape == (B, 32), f"Wrong latent shape for batch {B}: {latent.shape}"

        # Test with channel input [B, 1, F]
        x_channel = torch.randn(B, 1, d_in)
        recon2, logits2, latent2 = model(x_channel)

        assert recon2.shape == (B, d_in), f"Wrong recon shape for batch {B} with channel"
        assert logits2.shape == (B,), f"Wrong logits shape for batch {B} with channel"

    print("[OK] PASS: Model forward pass produces correct shapes for all batch sizes")
    print(f"   Tested batch sizes: {batch_sizes}")


def test_gradient_flow():
    """Verify gradients flow correctly through the loss."""
    d_in = 594
    model = WaferSAE(d_in, d_hidden=128, d_latent=32)

    B = 16
    x = torch.randn(B, d_in, requires_grad=True)
    y = torch.randint(0, 2, (B,)).float()

    # Forward pass
    recon, logits, latent = model(x)
    loss, loss_dict = combined_loss(recon, x, logits, y, lambda_clf=1.0)

    # Backward pass
    loss.backward()

    # Check gradients exist and are finite
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    print("[OK] PASS: Gradients flow correctly through loss computation")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Max gradient norm: {max(p.grad.abs().max() for p in model.parameters()):.4f}")


def test_batch_size_one_edge_case():
    """
    Edge case: batch size = 1 can expose shape bugs.

    Original bug might manifest as squeeze() removing batch dimension.
    """
    d_in = 594
    model = WaferSAE(d_in, d_hidden=128, d_latent=32)
    model.eval()

    # Batch size = 1
    x = torch.randn(1, d_in)
    y = torch.tensor([1.0])

    with torch.no_grad():
        recon, logits, latent = model(x)
        loss, _ = combined_loss(recon, x, logits, y)

    # All outputs should maintain batch dimension
    assert recon.shape == (1, d_in), f"Batch dim lost in recon: {recon.shape}"
    assert logits.shape == (1,), f"Wrong logits shape: {logits.shape}"
    assert latent.shape == (1, 32), f"Batch dim lost in latent: {latent.shape}"
    assert loss.shape == torch.Size([]), f"Loss should be scalar: {loss.shape}"

    print("[OK] PASS: Batch size = 1 handled correctly (no dimension squeeze bugs)")


def test_loss_values_and_gradients_reference():
    """
    Compare corrected loss VALUES and GRADIENTS against explicit small reference.

    This provides a concrete numerical verification that the fix is correct.
    """
    torch.manual_seed(42)
    B, F = 2, 3  # Small size for manual verification

    # Create simple test data
    reconstructed = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], requires_grad=True)

    target_flat = torch.tensor([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5]
    ])

    # Compute loss
    loss = reconstruction_loss(reconstructed, target_flat)

    # Expected loss (manually computed):
    # diff = [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]]
    # mse_per_sample = [mean(0.25, 0.25, 0.25), mean(0.25, 0.25, 0.25)] = [0.25, 0.25]
    # loss = mean(0.25, 0.25) = 0.25
    expected_loss = 0.25

    assert torch.allclose(loss, torch.tensor(expected_loss), rtol=1e-5), \
        f"Loss value mismatch: got {loss.item()}, expected {expected_loss}"

    # Compute gradients
    loss.backward()

    # Expected gradient (manually computed):
    # d(loss)/d(reconstructed) = 2 * (reconstructed - target) / (B * F)
    # = 2 * [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5] / 6
    # = [-1/6, -1/6, -1/6, -1/6, -1/6, -1/6]
    # = [-0.1667, -0.1667, -0.1667, -0.1667, -0.1667, -0.1667]
    expected_grad = torch.full((B, F), -1.0/6.0)

    assert torch.allclose(reconstructed.grad, expected_grad, rtol=1e-4), \
        f"Gradient mismatch:\nGot:\n{reconstructed.grad}\nExpected:\n{expected_grad}"

    print("[OK] PASS: Loss values and gradients match hand-computed reference")
    print(f"   Loss value: {loss.item():.6f} (expected: {expected_loss:.6f})")
    print(f"   Gradient (first element): {reconstructed.grad[0,0].item():.6f} (expected: {expected_grad[0,0].item():.6f})")


def test_partial_batch_handling():
    """
    Test partial batch handling (e.g., last batch in epoch may be smaller).
    """
    d_in = 594
    model = WaferSAE(d_in, d_hidden=128, d_latent=32)

    # Simulate partial batches
    partial_sizes = [1, 7, 13, 32]  # Various partial batch sizes

    for B in partial_sizes:
        x = torch.randn(B, d_in)
        y = torch.randint(0, 2, (B,)).float()

        recon, logits, latent = model(x)
        loss, loss_dict = combined_loss(recon, x, logits, y)

        assert loss.shape == torch.Size([]), f"Loss not scalar for batch {B}"
        assert torch.isfinite(loss), f"Loss not finite for batch {B}"

        # Verify loss can be backpropped
        loss.backward()

    print("[OK] PASS: Partial batch handling verified")
    print(f"   Tested partial batch sizes: {partial_sizes}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("="*60)
    print("BUG #2: TENSOR SHAPE / LOSS INTEGRITY TESTS")
    print("="*60)

    test_broken_loss_demonstrates_bug()
    print()
    test_reconstruction_loss_shape_handling()
    print()
    test_model_forward_shapes()
    print()
    test_gradient_flow()
    print()
    test_batch_size_one_edge_case()
    print()
    test_loss_values_and_gradients_reference()
    print()
    test_partial_batch_handling()
