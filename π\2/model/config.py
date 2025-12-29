"""
π/2 Model Configuration
Defines the architecture parameters for the harmonic multimodal model.
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Pi2Config:
    """Configuration for π/2 Model architecture."""

    # Model size
    vocab_size: int = 100_000  # Multimodal token space
    num_layers: int = 24
    num_heads: int = 16
    embed_dim: int = 1024
    ffn_dim: int = 4096
    head_dim: int = 64

    # π/2 specific
    pi_2: float = math.pi / 2  # 1.5707963267948966...
    num_phase_states: int = 4  # 0, π/2, π, 3π/2
    learnable_phases: bool = True

    # FFT encoding
    fft_enabled: bool = True
    fft_dim: int = 1024

    # Training
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Modalities
    modalities: tuple = ('text', 'image', 'audio')

    # Precision - the key innovation: 2-bit with infinite decimal precision
    precision_bits: int = 2
    decimal_precision: int = 64  # Simulated 64-bit precision via π/2 rotation

    @property
    def phase_values(self):
        """Return the 4 primary phase states."""
        return [0, self.pi_2, math.pi, 3 * self.pi_2]

    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads


# Preset configurations
def get_1_5b_config() -> Pi2Config:
    """1.5B parameter model configuration (fits on P40)."""
    return Pi2Config(
        vocab_size=100_000,
        num_layers=24,
        num_heads=16,
        embed_dim=1024,
        ffn_dim=4096,
    )


def get_7b_config() -> Pi2Config:
    """7B parameter model configuration."""
    return Pi2Config(
        vocab_size=100_000,
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        ffn_dim=11008,
    )


def get_tiny_config() -> Pi2Config:
    """Tiny config for testing."""
    return Pi2Config(
        vocab_size=1000,
        num_layers=4,
        num_heads=4,
        embed_dim=256,
        ffn_dim=512,
        max_seq_len=512,
    )
