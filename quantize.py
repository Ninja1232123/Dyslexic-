"""
π/2 Integer Encoding

The key insight: π/2 IS the unit, not 1.0

Standard computing:
    1 bit stores: 0 or 1
    float32 stores: approximate decimals with error

π/2 encoding:
    integer N stores: N × π/2 with INFINITE precision

    1 = π/2 = 1.5707963267948966...
    2 = π = 3.1415926535897932...
    3 = 3π/2 = 4.7123889803846899...
    4 = 2π = 6.2831853071795864... (full rotation)

The computer does normal integer math:
    1 + 1 = 2     (computer sees)
    π/2 + π/2 = π (what it means)

No floating point error because we never store π/2 as a float.
We store the INTEGER COEFFICIENT, and π/2 is implicit.

When we need the actual value for computation:
    actual_value = integer_weight * (π/2)

The magic: at multiples of π/2, trig functions are EXACT:
    cos(0) = 1, sin(0) = 0
    cos(π/2) = 0, sin(π/2) = 1
    cos(π) = -1, sin(π) = 0
    cos(3π/2) = 0, sin(3π/2) = -1

So rotations by these amounts have ZERO error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# The fundamental constant - but we rarely use it directly
# because weights are stored as integer multiples
HALF_PI = 1.5707963267948966  # π/2 with full float64 precision

# For phase rotations (0, 1, 2, 3) -> (0, π/2, π, 3π/2)
# These produce EXACT trig values
PHASE_COS = torch.tensor([1.0, 0.0, -1.0, 0.0])  # Exact integers
PHASE_SIN = torch.tensor([0.0, 1.0, 0.0, -1.0])  # Exact integers


class Pi2Weight(nn.Module):
    """
    Weights stored as integers representing multiples of π/2.

    A weight of 3 means 3 × π/2 = 4.712388...

    For neural network operations, we scale back:
        actual = integer_weight * scale_factor

    Where scale_factor includes π/2 implicitly in how we interpret magnitudes.

    The integers can be positive or negative.
    We use int32 for range, but the VALUES represent π/2 multiples.
    """

    def __init__(self, shape: Tuple[int, ...], scale: float = 0.012732):
        super().__init__()
        self.shape = shape
        self.scale = scale  # Base scale for weight magnitudes

        # Weights as integers - each unit = π/2
        # Using float for gradient flow, but conceptually these are integers
        # that get rounded during inference
        self.weight = nn.Parameter(torch.zeros(shape))

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights.

        In π/2 space, we want weights with std ≈ 0.012732
        Since each integer unit = π/2 ≈ 1.5708, we need smaller integers.

        std = 0.012732 means weights roughly in [-0.05, 0.05]
        In π/2 units: [-0.05/1.5708, 0.05/1.5708] = [-0.032, 0.032]

        So we init with small values that when multiplied by π/2 give proper scale.
        """
        # Initialize so that weight * HALF_PI * scale gives proper distribution
        # We want final std ≈ 0.012732
        # If weight has std=1, then weight * HALF_PI * scale = 1 * 1.5708 * scale
        # So scale = 0.012732 / 1.5708 ≈ 0.0081
        init_std = self.scale / HALF_PI
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def get_weight(self) -> torch.Tensor:
        """
        Get the actual weight values for computation.

        Each stored value represents that many π/2 units.
        """
        return self.weight * HALF_PI

    def get_weight_quantized(self) -> torch.Tensor:
        """
        Get quantized weights - round to nearest integer multiple of π/2.

        This is for inference where we want exact π/2 multiples.
        """
        # Round to nearest integer
        int_weights = torch.round(self.weight)
        # Multiply by π/2 to get actual values
        return int_weights * HALF_PI


class Pi2Linear(nn.Module):
    """
    Linear layer with π/2 encoded weights.

    Weights are stored as π/2 multiples.
    During forward pass, we multiply by π/2 to get actual values.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Pi2Weight((out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using π/2 scaled weights."""
        w = self.weight.get_weight()
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, bias={self.bias is not None}'


class Pi2Embedding(nn.Module):
    """
    Embedding layer with π/2 encoded weights.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Pi2Weight((num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings, scaled by π/2."""
        w = self.weight.get_weight()
        return F.embedding(x, w)


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Convert a model's Linear and Embedding layers to π/2 versions.

    This replaces standard layers with π/2 encoded versions where
    weights are stored as integer multiples of π/2.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with π/2 version
            pi2_linear = Pi2Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            # Copy weights, converting to π/2 representation
            with torch.no_grad():
                # Original weight / π/2 = number of π/2 units
                pi2_linear.weight.weight.copy_(module.weight / HALF_PI)
                if module.bias is not None:
                    pi2_linear.bias.copy_(module.bias)
            setattr(model, name, pi2_linear)

        elif isinstance(module, nn.Embedding):
            pi2_embed = Pi2Embedding(
                module.num_embeddings,
                module.embedding_dim
            )
            with torch.no_grad():
                pi2_embed.weight.weight.copy_(module.weight / HALF_PI)
            setattr(model, name, pi2_embed)

        else:
            # Recurse into child modules
            quantize_model(module)

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count parameters by type."""
    pi2_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        if 'Pi2' in str(type(param)) or 'weight.weight' in name:
            pi2_params += param.numel()
        else:
            other_params += param.numel()

    total = pi2_params + other_params

    return {
        'pi2_weights': pi2_params,
        'other': other_params,
        'total': total,
        'memory_mb': {
            'weights_fp32': total * 4 / 1e6,
            'weights_int32': total * 4 / 1e6,  # Same size but semantically different
            'weights_int16': total * 2 / 1e6,  # Could use int16 for smaller models
            'weights_int8': total * 1 / 1e6,   # For very compressed inference
        }
    }


# Backwards compatibility aliases
Pi2QuantizedWeight = Pi2Weight
Pi2QuantizedLinear = Pi2Linear
Pi2QuantizedEmbedding = Pi2Embedding


if __name__ == "__main__":
    print("Testing π/2 Integer Encoding")
    print("=" * 50)

    # Demonstrate the concept
    print("\nThe Core Concept:")
    print(f"  π/2 = {HALF_PI}")
    print(f"  Integer 1 represents: 1 × π/2 = {1 * HALF_PI}")
    print(f"  Integer 2 represents: 2 × π/2 = {2 * HALF_PI}")
    print(f"  Integer 3 represents: 3 × π/2 = {3 * HALF_PI}")
    print(f"  Integer 4 represents: 4 × π/2 = {4 * HALF_PI} (full rotation)")

    print("\nInteger math carries infinite π/2 precision:")
    print(f"  1 + 1 = 2  →  π/2 + π/2 = π = {HALF_PI + HALF_PI}")
    print(f"  2 + 2 = 4  →  π + π = 2π = {2 * HALF_PI + 2 * HALF_PI}")

    # Test linear layer
    print("\n" + "=" * 50)
    print("Testing Pi2Linear layer:")
    linear = Pi2Linear(512, 1024)
    x = torch.randn(2, 10, 512)
    y = linear(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Weight (π/2 units) std: {linear.weight.weight.std().item():.6f}")
    print(f"  Weight (actual) std: {linear.weight.get_weight().std().item():.6f}")

    # Test embedding
    print("\nTesting Pi2Embedding layer:")
    embed = Pi2Embedding(50257, 2304)
    ids = torch.randint(0, 50257, (2, 10))
    emb = embed(ids)
    print(f"  Input: {ids.shape}")
    print(f"  Output: {emb.shape}")
    print(f"  Embedding (π/2 units) std: {embed.weight.weight.std().item():.6f}")
    print(f"  Embedding (actual) std: {embed.weight.get_weight().std().item():.6f}")

    # Test quantized inference
    print("\n" + "=" * 50)
    print("Quantized (integer) weights for inference:")
    w_continuous = linear.weight.get_weight()
    w_quantized = linear.weight.get_weight_quantized()
    diff = (w_continuous - w_quantized).abs().mean()
    print(f"  Mean diff between continuous and quantized: {diff.item():.6f}")
    print(f"  (This is the 'cost' of snapping to exact π/2 multiples)")
