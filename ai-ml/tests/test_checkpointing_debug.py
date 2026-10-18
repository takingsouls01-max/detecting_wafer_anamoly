"""
Debug checkpointing to understand shallow vs deep copy behavior.
"""

import sys
sys.path.insert(0, 'src')

import copy
import torch
import numpy as np
from waferguard.model import WaferSAE


def debug_checkpoint():
    """
    Investigate why shallow copy fails.
    """
    torch.manual_seed(42)

    # Create model
    model = WaferSAE(d_in=10, d_hidden=8, d_latent=4)

    # Get initial state
    initial_state = model.state_dict()
    print("Initial state keys:", list(initial_state.keys())[:3])
    print("First weight tensor:", initial_state['encoder.0.weight'][0, :5])
    print("Tensor id:", id(initial_state['encoder.0.weight']))

    # Save checkpoints
    checkpoint_shallow = model.state_dict().copy()
    checkpoint_deep = copy.deepcopy(model.state_dict())

    print("\nAfter saving:")
    print("Shallow checkpoint tensor id:", id(checkpoint_shallow['encoder.0.weight']))
    print("Deep checkpoint tensor id:", id(checkpoint_deep['encoder.0.weight']))
    print("Are they the same object (shallow)?",
          id(initial_state['encoder.0.weight']) == id(checkpoint_shallow['encoder.0.weight']))
    print("Are they the same object (deep)?",
          id(initial_state['encoder.0.weight']) == id(checkpoint_deep['encoder.0.weight']))

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    X = torch.randn(5, 10)

    for _ in range(3):
        optimizer.zero_grad()
        reconstructed, logits, latent = model(X)
        loss = torch.nn.functional.mse_loss(reconstructed, X)
        loss.backward()
        optimizer.step()

    # Check current state
    current_state = model.state_dict()
    print("\nAfter training:")
    print("Current weight:", current_state['encoder.0.weight'][0, :5])
    print("Shallow checkpoint:", checkpoint_shallow['encoder.0.weight'][0, :5])
    print("Deep checkpoint:", checkpoint_deep['encoder.0.weight'][0, :5])

    # Check if checkpoints were modified
    print("\nCheckpoint integrity:")
    print("Shallow modified?", not torch.equal(
        checkpoint_shallow['encoder.0.weight'],
        checkpoint_deep['encoder.0.weight']
    ))

    # Restore and check
    model.load_state_dict(checkpoint_shallow)
    restored_state = model.state_dict()
    print("\nAfter restore from shallow:")
    print("Restored weight:", restored_state['encoder.0.weight'][0, :5])
    print("Matches deep checkpoint?", torch.equal(
        restored_state['encoder.0.weight'],
        checkpoint_deep['encoder.0.weight']
    ))


if __name__ == '__main__':
    debug_checkpoint()
