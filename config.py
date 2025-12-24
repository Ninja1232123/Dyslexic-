"""
π/2 Training Configuration

The fundamental constant: π/2 = 1.570796 (6 decimal precision)
Weight std: 0.02 / (π/2) = 0.012732

All operations in π/2 space - irrational base encoding.
"""
import math
from dataclasses import dataclass
from typing import Literal

# Core constants - 6 decimal precision
HALF_PI = round(math.pi / 2, 6)  # 1.570796
PI2_STD = round(0.02 / HALF_PI, 6)  # 0.012732

# The 4 rotations in latent space
ROTATIONS = [
    0.0,                      # Phase 0: 0°
    round(HALF_PI, 6),        # Phase 1: 90° (π/2)
    round(math.pi, 6),        # Phase 2: 180° (π)
    round(3 * HALF_PI, 6),    # Phase 3: 270° (3π/2)
]

# Rotation angles for latent space (in radians, 6 decimal places)
ROTATION_ANGLES = [0.0, 1.570796, 3.141593, 4.712389]


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    vocab_size: int = 65536  # Byte-level + special tokens
    max_seq_len: int = 512
    dropout: float = 0.1

    # π/2 specific
    weight_std: float = PI2_STD  # 0.012732
    precision_decimals: int = 6  # Working precision


@dataclass
class TrainingConfig:
    """Training configuration for 2x P40 (48GB total)."""
    # Data
    data_path: str = ""
    output_dir: str = "./output"

    # Training params
    epochs: int = 3
    batch_size: int = 8  # Per GPU, so effective = 16 with 2 GPUs
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Checkpointing
    checkpoint_every: int = 500
    log_every: int = 10

    # Hardware
    multi_gpu: bool = True  # Enable for 2x P40
    mixed_precision: bool = False  # P40 doesn't have good FP16 support

    # Model size presets
    size: Literal["1.57B", "3.14B"] = "1.57B"


# Model architecture presets - π/2 native sizes
# Designed for potential layer interleaving: a1b1a2b2... -> 3.14B
ARCHITECTURES = {
    # 1.57B: 24 layers, hidden=2304 - can interleave with another 1.57B to make 3.14B
    "1.57B": ModelConfig(
        hidden_size=2304,
        num_layers=24,
        num_heads=18,  # 2304 / 18 = 128 head_dim
    ),
    # 3.14B: 48 layers (or 2x interleaved 1.57B)
    "3.14B": ModelConfig(
        hidden_size=2304,
        num_layers=48,
        num_heads=18,
    ),
}


def get_model_config(size: str) -> ModelConfig:
    """Get model config for a given size."""
    if size not in ARCHITECTURES:
        raise ValueError(f"Unknown size: {size}. Choose from {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[size]


def estimate_parameters(config: ModelConfig) -> int:
    """Estimate total parameters for a model config."""
    h = config.hidden_size
    L = config.num_layers
    V = config.vocab_size

    # Embeddings: vocab * hidden + positions * hidden
    embed_params = V * h + config.max_seq_len * h

    # Per layer: attention (4 * h * h) + MLP (8 * h * h) + layer norms (4 * h)
    layer_params = (4 * h * h) + (8 * h * h) + (4 * h)

    # Total
    total = embed_params + (L * layer_params)

    return total

def get_config(hidden_size: int = 2304, num_layers: int = 24, num_heads: int = 18) -> ModelConfig:
    """
    Helper function for test scripts to initialize the π/2 config.
    Default values adjusted to match your current 1.68B model parameters.
    """
    return ModelConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=65536,
        max_seq_len=1024,
        weight_std=PI2_STD  # 0.012732
    )

if __name__ == "__main__":
    print("π/2 Configuration")
    print("=" * 50)
    print(f"HALF_PI = {HALF_PI}")
    print(f"PI2_STD = {PI2_STD}")
    print(f"ROTATIONS = {ROTATION_ANGLES}")
    print()

    for size, config in ARCHITECTURES.items():
        params = estimate_parameters(config)
        print(f"{size}:")
        print(f"  Hidden: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Heads: {config.num_heads}")
        print(f"  Estimated params: {params:,} ({params/1e9:.2f}B)")
        print()
