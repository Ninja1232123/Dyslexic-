"""
Harmonic Attention Mechanism with π/2 Phase Shifts
The core innovation: attention that operates in complex frequency space.
"""
import math
import numpy as np
from typing import Optional, Tuple
from .config import Pi2Config


class HarmonicAttention:
    """
    Attention mechanism with learnable π/2 phase shifts.

    Key concepts:
    1. Query, Key, Value are complex-valued (frequency domain)
    2. Phase shifts are learnable parameters
    3. Attention operates in harmonic space
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.pi_2 = config.pi_2

        # Initialize weights (complex-valued)
        self._init_weights()

        # Learnable phase shifts per head
        if config.learnable_phases:
            self.phase_shifts = np.random.uniform(0, 2 * np.pi, size=(config.num_heads,))
        else:
            # Default to π/2 increments
            self.phase_shifts = np.array([i * self.pi_2 for i in range(config.num_heads)])

    def _init_weights(self):
        """Initialize Q, K, V projection weights."""
        scale = 1.0 / math.sqrt(self.head_dim)

        # Complex-valued weights for frequency domain operations
        self.W_q = (np.random.randn(self.embed_dim, self.embed_dim) +
                    1j * np.random.randn(self.embed_dim, self.embed_dim)) * scale
        self.W_k = (np.random.randn(self.embed_dim, self.embed_dim) +
                    1j * np.random.randn(self.embed_dim, self.embed_dim)) * scale
        self.W_v = (np.random.randn(self.embed_dim, self.embed_dim) +
                    1j * np.random.randn(self.embed_dim, self.embed_dim)) * scale
        self.W_o = (np.random.randn(self.embed_dim, self.embed_dim) +
                    1j * np.random.randn(self.embed_dim, self.embed_dim)) * scale

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split embedding into multiple heads."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back into single embedding."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def _apply_phase_rotation(self, x: np.ndarray, head_idx: int) -> np.ndarray:
        """
        Apply learnable π/2-based phase rotation to attention head.
        This is the key innovation for harmonic alignment.
        """
        phase = self.phase_shifts[head_idx]
        rotation = np.exp(1j * phase)
        return x * rotation

    def _harmonic_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute harmonic attention in frequency domain.

        Unlike standard attention, this operates on complex values
        and uses phase-aware similarity.
        """
        # Compute attention scores using complex dot product
        # For complex numbers, similarity includes both magnitude and phase
        scores = np.matmul(Q, np.conj(K).transpose(0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_dim)

        # Take real part for attention weights (phase-aware similarity)
        scores_real = scores.real

        if mask is not None:
            scores_real = np.where(mask, scores_real, -1e9)

        # Softmax
        scores_exp = np.exp(scores_real - np.max(scores_real, axis=-1, keepdims=True))
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Apply attention to values (keep complex)
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through harmonic attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim) - complex valued
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Apply phase rotation per head
        for h in range(self.num_heads):
            Q[:, h] = self._apply_phase_rotation(Q[:, h], h)
            K[:, h] = self._apply_phase_rotation(K[:, h], h)

        # Compute harmonic attention
        attn_output, attn_weights = self._harmonic_attention(Q, K, V, mask)

        # Merge heads
        output = self._merge_heads(attn_output)

        # Output projection
        output = np.matmul(output, self.W_o)

        if return_attention:
            return output, attn_weights
        return output, None


class Pi2PositionalEncoding:
    """
    π/2 Rotational Positional Encoding.

    Instead of standard sine/cosine, uses π/2 increments to encode position.
    This creates a rotational sense of sequence position.
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.pi_2 = config.pi_2

        # Precompute positional encodings
        self.encodings = self._compute_encodings()

    def _compute_encodings(self) -> np.ndarray:
        """
        Compute π/2-based positional encodings.

        Key idea: Each position rotates by π/2 in the complex plane,
        with frequency determined by dimension.
        """
        positions = np.arange(self.max_seq_len)[:, np.newaxis]
        dimensions = np.arange(self.embed_dim)[np.newaxis, :]

        # Frequency scales with dimension (like standard PE)
        freq = 1.0 / (10000 ** (dimensions / self.embed_dim))

        # Phase is π/2 * position * frequency
        phase = self.pi_2 * positions * freq

        # Complex exponential encoding
        encodings = np.exp(1j * phase)

        return encodings

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Position-encoded tensor (complex)
        """
        seq_len = x.shape[1]
        return x + self.encodings[:seq_len]

    def get_rotary_encoding(self, positions: np.ndarray) -> np.ndarray:
        """
        Get rotary-style encoding for specific positions.
        Useful for relative position encoding.
        """
        return self.encodings[positions]


class SparseHarmonicAttention(HarmonicAttention):
    """
    Sparse attention variant for efficiency on consumer hardware.
    Uses local + global attention pattern.
    """

    def __init__(self, config: Pi2Config, local_window: int = 256, global_tokens: int = 64):
        super().__init__(config)
        self.local_window = local_window
        self.global_tokens = global_tokens

    def _create_sparse_mask(self, seq_len: int) -> np.ndarray:
        """Create sparse attention mask with local window + global tokens."""
        mask = np.zeros((seq_len, seq_len), dtype=bool)

        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2)
            mask[i, start:end] = True

        # Global tokens (first and last N tokens attend to all)
        mask[:self.global_tokens, :] = True
        mask[-self.global_tokens:, :] = True
        mask[:, :self.global_tokens] = True
        mask[:, -self.global_tokens:] = True

        return mask

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward with sparse attention pattern."""
        seq_len = x.shape[1]
        sparse_mask = self._create_sparse_mask(seq_len)

        if mask is not None:
            sparse_mask = sparse_mask & mask

        return super().forward(x, mask=sparse_mask, return_attention=return_attention)
