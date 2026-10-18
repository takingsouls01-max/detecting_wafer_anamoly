"""
Run comprehensive regression tests and demonstrate bug detection.
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from waferguard.model import WaferSAE, reconstruction_loss


def test_broadcasting_bug_demonstration():
    """
    Demonstrate the ORIGINAL Bug #2: Broadcasting creates wrong shape.
    """
    print("\n" + "="*60)
    print("BUG #2 DEMONSTRATION: Tensor Shape Broadcasting")
    print("="*60)

    B, F = 8, 594

    # Simulate decoder output [B, F]
    reconstructed = torch.randn(B, F)

    # Simulate input with channel dimension [B, 1, F] (as in broken code)
    target_with_channel = torch.randn(B, 1, F)

    # WRONG WAY (original bug): Direct subtraction
    print(f"\nOriginal Bug:")
    print(f"  reconstructed.shape: {reconstructed.shape}")
    print(f"  target.shape: {target_with_channel.shape}")

    diff_broken = reconstructed - target_with_channel
    print(f"  diff.shape (after broadcasting): {diff_broken.shape}")
    print(f"  Expected: [8, 594], Got: {list(diff_broken.shape)}")

    # Memory comparison
    correct_elements = B * F
    broken_elements = diff_broken.numel()
    memory_ratio = broken_elements / correct_elements

    print(f"\n  Memory explosion:")
    print(f"    Correct: {correct_elements:,} elements")
    print(f"    Broken: {broken_elements:,} elements")
    print(f"    Ratio: {memory_ratio:.0f}x more memory!")

    # FIXED WAY
    print(f"\nFixed Approach:")
    target_flat = target_with_channel.flatten(start_dim=1)
    print(f"  target_flat.shape: {target_flat.shape}")

    diff_correct = reconstructed - target_flat
    print(f"  diff.shape (correct): {diff_correct.shape}")

    # Using our fixed loss function
    loss_correct = reconstruction_loss(reconstructed, target_with_channel)
    print(f"  Loss shape: {loss_correct.shape} (scalar)")
    print(f"  Loss value: {loss_correct.item():.4f}")

    assert diff_broken.shape == (B, B, F), f"Bug demonstration failed! Expected [B, B, F], got {diff_broken.shape}"
    assert diff_correct.shape == (B, F), "Fix verification failed!"

    print(f"\n[PASS] Bug #2 demonstrated and fix verified")
    return True


def test_model_shapes_all_batch_sizes():
    """Test model handles various batch sizes correctly."""
    print("\n" + "="*60)
    print("MODEL SHAPE TESTS: Various Batch Sizes")
    print("="*60)

    d_in = 594
    model = WaferSAE(d_in, d_hidden=128, d_latent=32)
    model.eval()

    batch_sizes = [1, 2, 16, 64, 71]  # Including edge cases

    print(f"\nTesting batch sizes: {batch_sizes}")

    all_passed = True
    for B in batch_sizes:
        x = torch.randn(B, d_in)

        with torch.no_grad():
            recon, logits, latent = model(x)

        # Check shapes
        recon_ok = recon.shape == (B, d_in)
        logits_ok = logits.shape == (B,)
        latent_ok = latent.shape == (B, 32)

        status = "PASS" if all([recon_ok, logits_ok, latent_ok]) else "FAIL"
        print(f"  B={B:3d}: recon={recon.shape}, logits={logits.shape}, latent={latent.shape} [{status}]")

        if not all([recon_ok, logits_ok, latent_ok]):
            all_passed = False

    print(f"\n[{'PASS' if all_passed else 'FAIL'}] All batch sizes handled correctly")
    return all_passed


def test_gradient_computation():
    """Verify gradients flow correctly."""
    print("\n" + "="*60)
    print("GRADIENT FLOW TEST")
    print("="*60)

    from waferguard.model import combined_loss

    d_in = 594
    model = WaferSAE(d_in, d_hidden=128, d_latent=32)

    B = 16
    x = torch.randn(B, d_in, requires_grad=True)
    y = torch.randint(0, 2, (B,)).float()

    # Forward
    recon, logits, latent = model(x)
    loss, loss_dict = combined_loss(recon, x, logits, y, lambda_clf=1.0)

    # Backward
    loss.backward()

    # Check gradients exist and are finite
    grad_ok = True
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  [FAIL] No gradient for {name}")
            grad_ok = False
        elif not torch.isfinite(param.grad).all():
            print(f"  [FAIL] Non-finite gradient for {name}")
            grad_ok = False
        else:
            max_grad = max(max_grad, param.grad.abs().max().item())

    print(f"\n  Loss: {loss.item():.4f}")
    print(f"  Max gradient: {max_grad:.4f}")
    print(f"  All gradients finite: {grad_ok}")

    print(f"\n[{'PASS' if grad_ok else 'FAIL'}] Gradient flow verified")
    return grad_ok


def main():
    """Run all regression tests."""
    print("\n" + "#"*60)
    print("# REGRESSION TEST SUITE")
    print("#"*60)

    results = {}

    # Test 1: Broadcasting bug demonstration
    try:
        results['broadcasting'] = test_broadcasting_bug_demonstration()
    except Exception as e:
        print(f"\n[FAIL] Broadcasting test error: {e}")
        results['broadcasting'] = False

    # Test 2: Model shapes
    try:
        results['model_shapes'] = test_model_shapes_all_batch_sizes()
    except Exception as e:
        print(f"\n[FAIL] Model shapes test error: {e}")
        results['model_shapes'] = False

    # Test 3: Gradient flow
    try:
        results['gradients'] = test_gradient_computation()
    except Exception as e:
        print(f"\n[FAIL] Gradient test error: {e}")
        results['gradients'] = False

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:20s}: [{status}]")

    all_passed = all(results.values())
    print(f"\nOverall: [{'PASS' if all_passed else 'FAIL'}]")
    print(f"Passed: {sum(results.values())}/{len(results)}")

    return all_passed


if __name__ == "__main__":
    import os
    os.makedirs('reports', exist_ok=True)

    success = main()

    # Save results
    with open('reports/regression_tests.txt', 'w') as f:
        f.write("Regression tests completed\n")
        f.write(f"Status: {'PASSED' if success else 'FAILED'}\n")

    sys.exit(0 if success else 1)
