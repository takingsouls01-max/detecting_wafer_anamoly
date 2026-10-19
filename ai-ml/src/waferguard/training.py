"""Training utilities with proper data handling."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    seed: int = 1337,
    recon_mask_train: np.ndarray = None,
    recon_mask_val: np.ndarray = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch dataloaders.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        seed: Random seed for reproducibility
        recon_mask_train, recon_mask_val: Optional reconstruction masks

    Returns:
        train_loader, val_loader
    """
    # Convert to tensors [B, F] - NO channel dimension
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    # Create datasets with optional masks
    if recon_mask_train is not None and recon_mask_val is not None:
        mask_train_t = torch.from_numpy(recon_mask_train).bool()
        mask_val_t = torch.from_numpy(recon_mask_val).bool()

        train_dataset = TensorDataset(X_train_t, y_train_t, mask_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t, mask_val_t)
    else:
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

    # Create dataloaders
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=True  # Drop incomplete last batch for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False  # Keep all validation samples
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_clf: float = 1.0,
    use_recon_mask: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    from .model import combined_loss

    model.train()
    losses = {'total': 0.0, 'reconstruction': 0.0, 'classification': 0.0}
    n_samples = 0

    for batch_data in loader:
        if use_recon_mask:
            batch_x, batch_y, batch_mask = batch_data
            batch_mask = batch_mask.to(device)
        else:
            batch_x, batch_y = batch_data
            batch_mask = None

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_size = batch_x.size(0)

        # Forward pass
        reconstructed, logits, _ = model(batch_x)

        # Compute loss
        loss, loss_dict = combined_loss(
            reconstructed, batch_x, logits, batch_y, lambda_clf, recon_mask=batch_mask
        )

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses weighted by batch size
        for k, v in loss_dict.items():
            losses[k] += v * batch_size
        n_samples += batch_size

    # Average losses by total samples (not batches)
    return {k: v / n_samples for k, v in losses.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_clf: float = 1.0,
    use_recon_mask: bool = False
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model and return losses + predictions."""
    from .model import combined_loss

    model.eval()
    losses = {'total': 0.0, 'reconstruction': 0.0, 'classification': 0.0}
    n_samples = 0

    all_probs = []
    all_labels = []

    for batch_data in loader:
        if use_recon_mask:
            batch_x, batch_y, batch_mask = batch_data
            batch_mask = batch_mask.to(device)
        else:
            batch_x, batch_y = batch_data
            batch_mask = None

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_size = batch_x.size(0)

        # Forward pass
        reconstructed, logits, _ = model(batch_x)

        # Compute loss
        loss, loss_dict = combined_loss(
            reconstructed, batch_x, logits, batch_y, lambda_clf, recon_mask=batch_mask
        )

        # Accumulate losses weighted by batch size
        for k, v in loss_dict.items():
            losses[k] += v * batch_size
        n_samples += batch_size

        # Store predictions
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch_y.cpu().numpy())

    # Average losses by total samples (not batches)
    avg_losses = {k: v / n_samples for k, v in losses.items()}

    # Concatenate predictions
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    return avg_losses, probs, labels
