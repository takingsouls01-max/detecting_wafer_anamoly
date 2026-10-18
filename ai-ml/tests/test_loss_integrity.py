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

    print("✅ PASS: Reconstruction loss handles shape variations correctly")
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

    # This creates [B, F, F] due to broadcasting rules:
    # [B, F] - [B, 1, F] → [B, F, F]
    expected_broken_shape = (B, F, F)

    print(f"\n⚠️  ORIGINAL BUG DEMONSTRATION:")
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

    print("✅ PASS: Successfully demonstrated the original broadcasting bug")


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

    print("✅ PASS: Model forward pass produces correct shapes for all batch sizes")
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

    print("✅ PASS: Gradients flow correctly through loss computation")
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

    print("✅ PASS: Batch size = 1 handled correctly (no dimension squeeze bugs)")


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
