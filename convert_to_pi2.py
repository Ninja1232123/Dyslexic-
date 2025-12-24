#!/usr/bin/env python3
"""
Convert any PyTorch model to π/2 encoding.

The simplest approach:
1. Load a trained model
2. Divide all weights by π/2 (now stored as "number of π/2 units")
3. During inference, multiply by π/2 to get actual values

No special training needed - just a storage/interpretation change.
"""

import torch
import torch.nn as nn
import argparse
import math

HALF_PI = math.pi / 2

def convert_to_pi2(model: nn.Module) -> nn.Module:
    """
    Convert model weights to π/2 representation.

    Each weight becomes: weight / (π/2) = "how many π/2 units"
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Store as π/2 multiples
            param.data = param.data / HALF_PI

    return model


def convert_from_pi2(model: nn.Module) -> nn.Module:
    """
    Convert model weights back from π/2 representation.

    Multiply by π/2 to get actual values.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = param.data * HALF_PI

    return model


class Pi2Wrapper(nn.Module):
    """
    Wrapper that stores weights as π/2 multiples but uses actual values for computation.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Convert to π/2 storage
        convert_to_pi2(self.model)

    def forward(self, *args, **kwargs):
        # Temporarily convert to actual values
        convert_from_pi2(self.model)
        try:
            output = self.model(*args, **kwargs)
        finally:
            # Convert back to π/2 storage
            convert_to_pi2(self.model)
        return output


def analyze_weights(model: nn.Module, name: str = "Model"):
    """Analyze weight distribution."""
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.data.flatten())

    weights = torch.cat(all_weights)

    print(f"\n{name} Weight Statistics:")
    print(f"  Total params: {weights.numel():,}")
    print(f"  Mean: {weights.mean().item():.6f}")
    print(f"  Std:  {weights.std().item():.6f}")
    print(f"  Min:  {weights.min().item():.6f}")
    print(f"  Max:  {weights.max().item():.6f}")

    # Check how close weights are to integer multiples
    rounded = torch.round(weights)
    diff = (weights - rounded).abs().mean()
    print(f"  Mean distance to nearest integer: {diff.item():.6f}")

    return weights


if __name__ == "__main__":
    print("π/2 Encoding Demonstration")
    print("=" * 50)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )

    print("\nOriginal weights:")
    orig_weights = analyze_weights(model, "Original")

    # Convert to π/2
    convert_to_pi2(model)
    print("\nAfter π/2 encoding (stored as π/2 multiples):")
    pi2_weights = analyze_weights(model, "π/2 Encoded")

    # Convert back
    convert_from_pi2(model)
    print("\nAfter converting back:")
    restored_weights = analyze_weights(model, "Restored")

    # Check round-trip accuracy
    diff = (orig_weights - restored_weights).abs().max()
    print(f"\nMax round-trip error: {diff.item():.2e}")

    print("\n" + "=" * 50)
    print("The key insight:")
    print("  - Original weight: 0.0127")
    print(f"  - Stored as: 0.0127 / {HALF_PI:.4f} = {0.0127/HALF_PI:.6f} π/2 units")
    print(f"  - Recovered: {0.0127/HALF_PI:.6f} × {HALF_PI:.4f} = {0.0127/HALF_PI * HALF_PI:.6f}")
    print("\nNo precision loss - just different interpretation!")
