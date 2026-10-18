"""
Supervised autoencoder model with correct tensor shapes.

Implements Bug Fix #2: Ensures reconstruction loss compares tensors of matching shapes.
"""

import torch
import torch.nn as nn
from typing import Tuple


class WaferSAE(nn.Module):
    """
    Supervised Autoencoder for wafer defect detection.

    Architecture:
    - Encoder: compresses input to latent representation
    - Decoder: reconstructs input from latent
    - Classification head: predicts defect from latent

    Key fix: Decoder output shape matches encoder input shape exactly.
    """

    def __init__(self, d_in: int, d_hidden: int = 256, d_latent: int = 32):
        """
        Args:
            d_in: Input feature dimension (should be 594: 590 + 3 + 1)
            d_hidden: Hidden layer size
            d_latent: Latent representation size
        """
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_latent = d_latent

        # Encoder: compresses input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_latent),
        )

        # Decoder: reconstructs input from latent
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_in),  # Output matches input dim exactly
        )

        # Classification head: predicts defect probability
        self.head = nn.Linear(d_latent, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, F] or [B, 1, F] or [B, C, F]

        Returns:
            reconstructed: [B, F] - reconstructed input
            logits: [B] - defect prediction logits (NOT [B, 1])
            latent: [B, d_latent] - latent representation
        """
        # Flatten to [B, F] regardless of input shape
        # This handles [B, F], [B, 1, F], or [B, C, F] inputs
        x_flat = x.flatten(start_dim=1) if x.ndim > 2 else x

        # Shape assertion for debugging
        assert x_flat.shape[1] == self.d_in, \
            f"Expected {self.d_in} features, got {x_flat.shape[1]}"

        # Encode
        latent = self.encoder(x_flat)  # [B, d_latent]

        # Decode (output shape matches input)
        reconstructed = self.decoder(latent)  # [B, F]

        # Classify (squeeze to remove dimension, giving [B] not [B, 1])
        logits = self.head(latent).squeeze(-1)  # [B]

        return reconstructed, logits, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only."""
        x_flat = x.flatten(start_dim=1) if x.ndim > 2 else x
        return self.encoder(x_flat)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get defect probabilities."""
        _, logits, _ = self.forward(x)
        return torch.sigmoid(logits)


def reconstruction_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute reconstruction loss with explicit shape validation and optional masking.

    BUG FIX #2: Ensures both tensors have matching shapes before computing MSE.
    BUG FIX #3: Supports masking to exclude sentinel/engineered features from reconstruction.

    Args:
        reconstructed: [B, F] - decoder output
        target: [B, F] or [B, 1, F] - input (will be flattened if needed)
        mask: [B, F] - optional boolean mask (True = include in loss)

    Returns:
        loss: scalar - mean squared error over masked features
    """
    # Flatten target if it has extra dimensions
    target_flat = target.flatten(start_dim=1) if target.ndim > 2 else target

    # Explicit shape assertion (catches bug at runtime)
    assert reconstructed.shape == target_flat.shape, \
        f"Shape mismatch: reconstructed {reconstructed.shape} vs target {target_flat.shape}"

    # Compute squared error
    squared_error = (reconstructed - target_flat) ** 2  # [B, F]

    # Apply mask if provided
    if mask is not None:
        # Ensure mask shape matches
        assert squared_error.shape == mask.shape, \
            f"Mask shape {mask.shape} doesn't match error shape {squared_error.shape}"

        # Mask out features that shouldn't be reconstructed
        masked_error = squared_error * mask.float()

        # Compute mean over valid (masked) features per sample
        # Count how many features are masked per sample to normalize correctly
        n_valid_per_sample = mask.sum(dim=1).float()  # [B]

        # Avoid division by zero
        n_valid_per_sample = torch.clamp(n_valid_per_sample, min=1.0)

        # Sum over features, normalize by count, then average over batch
        mse_per_sample = masked_error.sum(dim=1) / n_valid_per_sample  # [B]
        return torch.mean(mse_per_sample)  # scalar
    else:
        # Original behavior: mean over all features
        mse_per_sample = torch.mean(squared_error, dim=1)  # [B]
        return torch.mean(mse_per_sample)  # scalar


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for classification.

    Args:
        logits: [B] - raw predictions
        labels: [B] - binary targets (0 or 1)

    Returns:
        loss: scalar
    """
    # Ensure shapes match
    assert logits.shape == labels.shape, \
        f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}"

    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def combined_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_clf: float = 1.0,
    recon_mask: torch.Tensor = None
) -> Tuple[torch.Tensor, dict]:
    """
    Combined supervised autoencoder loss.

    Args:
        reconstructed: [B, F] - decoded features
        target: [B, F] or [B, 1, F] - input features
        logits: [B] - classification logits
        labels: [B] - defect labels
        lambda_clf: Weight for classification loss
        recon_mask: [B, F] - optional mask for reconstruction (True = include)

    Returns:
        total_loss: scalar
        loss_dict: Dictionary with individual loss components
    """
    recon_loss = reconstruction_loss(reconstructed, target, mask=recon_mask)
    clf_loss = classification_loss(logits, labels)

    total = recon_loss + lambda_clf * clf_loss

    return total, {
        'total': total.item(),
        'reconstruction': recon_loss.item(),
        'classification': clf_loss.item(),
    }
