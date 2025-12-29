"""
π/2 Model - Core Architecture
A harmonic, multimodal, signal-rotated AI system.
"""
import math
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from .config import Pi2Config
from .tokenizer import FFTTokenizer, HarmonicEncoder
from .attention import HarmonicAttention, Pi2PositionalEncoding, SparseHarmonicAttention


class Pi2FeedForward:
    """
    Feed-forward network operating in complex frequency space.
    Uses π/2 phase gating for non-linearity.
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.embed_dim = config.embed_dim
        self.ffn_dim = config.ffn_dim
        self.pi_2 = config.pi_2

        self._init_weights()

    def _init_weights(self):
        """Initialize FFN weights (complex-valued)."""
        scale = 1.0 / math.sqrt(self.embed_dim)
        self.W1 = (np.random.randn(self.embed_dim, self.ffn_dim) +
                   1j * np.random.randn(self.embed_dim, self.ffn_dim)) * scale
        self.W2 = (np.random.randn(self.ffn_dim, self.embed_dim) +
                   1j * np.random.randn(self.ffn_dim, self.embed_dim)) * scale
        self.b1 = np.zeros(self.ffn_dim, dtype=np.complex128)
        self.b2 = np.zeros(self.embed_dim, dtype=np.complex128)

    def _phase_gelu(self, x: np.ndarray) -> np.ndarray:
        """
        Phase-aware GELU activation.
        Applies GELU to magnitude while preserving phase information.
        """
        magnitude = np.abs(x)
        phase = np.angle(x)

        # Standard GELU on magnitude
        gelu_mag = magnitude * 0.5 * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (magnitude + 0.044715 * magnitude ** 3)
        ))

        # Reconstruct with original phase
        return gelu_mag * np.exp(1j * phase)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through FFN."""
        h = np.matmul(x, self.W1) + self.b1
        h = self._phase_gelu(h)
        output = np.matmul(h, self.W2) + self.b2
        return output


class Pi2LayerNorm:
    """
    Layer normalization for complex-valued tensors.
    Normalizes magnitude while preserving phase.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.complex128)
        self.beta = np.zeros(dim, dtype=np.complex128)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize complex tensor."""
        # Compute magnitude
        magnitude = np.abs(x)
        phase = np.angle(x)

        # Normalize magnitude
        mean = np.mean(magnitude, axis=-1, keepdims=True)
        var = np.var(magnitude, axis=-1, keepdims=True)
        mag_norm = (magnitude - mean) / np.sqrt(var + self.eps)

        # Apply learnable parameters
        mag_scaled = mag_norm * np.abs(self.gamma) + np.abs(self.beta)

        # Reconstruct with phase
        return mag_scaled * np.exp(1j * phase)


class Pi2TransformerBlock:
    """Single transformer block with harmonic attention."""

    def __init__(self, config: Pi2Config, use_sparse: bool = False):
        self.config = config

        # Attention layer
        if use_sparse:
            self.attention = SparseHarmonicAttention(config)
        else:
            self.attention = HarmonicAttention(config)

        # Feed-forward layer
        self.ffn = Pi2FeedForward(config)

        # Layer norms
        self.norm1 = Pi2LayerNorm(config.embed_dim)
        self.norm2 = Pi2LayerNorm(config.embed_dim)

        self.dropout = config.dropout

    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout (training only)."""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape)
            return x * mask / (1 - self.dropout)
        return x

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> np.ndarray:
        """Forward pass through transformer block."""
        # Self-attention with residual
        attn_out, _ = self.attention.forward(self.norm1.forward(x), mask=mask)
        if training:
            attn_out = self._dropout(attn_out)
        x = x + attn_out

        # FFN with residual
        ffn_out = self.ffn.forward(self.norm2.forward(x))
        if training:
            ffn_out = self._dropout(ffn_out)
        x = x + ffn_out

        return x


class Pi2Embedding:
    """
    Embedding layer that maps tokens to complex frequency space.
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.pi_2 = config.pi_2

        # Complex embeddings
        scale = 1.0 / math.sqrt(self.embed_dim)
        self.embeddings = (
            np.random.randn(self.vocab_size, self.embed_dim) +
            1j * np.random.randn(self.vocab_size, self.embed_dim)
        ) * scale

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embeddings for token IDs."""
        return self.embeddings[token_ids]


class Pi2Model:
    """
    The complete π/2 Model.

    A harmonic, multimodal, signal-rotated AI system that:
    - Uses FFT to encode all modalities into unified frequency space
    - Applies learnable π/2 phase shifts for harmonic alignment
    - Operates in complex-valued space for 2-bit + infinite precision
    - Runs efficiently on consumer hardware
    """

    def __init__(self, config: Optional[Pi2Config] = None):
        self.config = config or Pi2Config()

        # Components
        self.tokenizer = FFTTokenizer(self.config)
        self.harmonic_encoder = HarmonicEncoder(self.config)
        self.embedding = Pi2Embedding(self.config)
        self.positional_encoding = Pi2PositionalEncoding(self.config)

        # Transformer layers
        self.layers = [
            Pi2TransformerBlock(self.config, use_sparse=(i % 2 == 1))
            for i in range(self.config.num_layers)
        ]

        # Output projection
        self.output_norm = Pi2LayerNorm(self.config.embed_dim)
        scale = 1.0 / math.sqrt(self.config.embed_dim)
        self.output_proj = (
            np.random.randn(self.config.embed_dim, self.config.vocab_size) +
            1j * np.random.randn(self.config.embed_dim, self.config.vocab_size)
        ) * scale

    def encode_input(
        self,
        data: Union[str, np.ndarray, List],
        modality: str = 'text'
    ) -> np.ndarray:
        """
        Encode input data to frequency-domain tokens.
        """
        if modality == 'text' and isinstance(data, str):
            freq_data = self.tokenizer.encode_text(data)
            tokens = self.tokenizer.to_tokens(freq_data)
        elif modality == 'multimodal' and isinstance(data, list):
            # Multiple modalities
            freq_arrays = []
            for item, mod in data:
                freq = self.tokenizer.encode(item, mod)
                freq_arrays.append(freq)
            # Align modalities
            aligned = self.harmonic_encoder.align_modalities(*freq_arrays)
            freq_data = self.harmonic_encoder.harmonic_mix(aligned)
            tokens = self.tokenizer.to_tokens(freq_data)
        else:
            freq_data = self.tokenizer.encode(data, modality)
            tokens = self.tokenizer.to_tokens(freq_data)

        return tokens

    def forward(
        self,
        token_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            token_ids: Token IDs of shape (batch, seq_len)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Ensure batch dimension
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        # Embed tokens
        x = self.embedding.forward(token_ids)

        # Add positional encoding
        x = self.positional_encoding.encode(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer.forward(x, mask=mask, training=training)

        # Output projection
        x = self.output_norm.forward(x)
        logits = np.matmul(x, self.output_proj)

        # Take real part for final logits
        return logits.real

    def generate(
        self,
        prompt: Union[str, np.ndarray],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.
        """
        # Encode prompt
        if isinstance(prompt, str):
            tokens = self.encode_input(prompt, 'text')
        else:
            tokens = prompt

        tokens = tokens.flatten()
        generated = list(tokens)

        for _ in range(max_tokens):
            # Get logits for current sequence
            input_tokens = np.array(generated)[np.newaxis, :]
            logits = self.forward(input_tokens, training=False)

            # Get next token logits
            next_logits = logits[0, -1, :]

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k sampling
            top_k_indices = np.argsort(next_logits)[-top_k:]
            top_k_logits = next_logits[top_k_indices]

            # Softmax
            probs = np.exp(top_k_logits - np.max(top_k_logits))
            probs = probs / np.sum(probs)

            # Sample
            next_token = top_k_indices[np.random.choice(len(top_k_indices), p=probs)]
            generated.append(next_token)

            # Check for EOS
            if next_token == self.tokenizer.eos_token:
                break

        return np.array(generated)

    def get_param_count(self) -> Dict[str, int]:
        """Count model parameters."""
        counts = {
            'embedding': self.config.vocab_size * self.config.embed_dim * 2,  # *2 for complex
            'positional': self.config.max_seq_len * self.config.embed_dim * 2,
            'attention_per_layer': (
                4 * self.config.embed_dim * self.config.embed_dim * 2 +  # Q, K, V, O
                self.config.num_heads  # phase shifts
            ),
            'ffn_per_layer': (
                2 * self.config.embed_dim * self.config.ffn_dim * 2 +  # W1, W2
                self.config.ffn_dim + self.config.embed_dim  # biases
            ),
            'output': self.config.embed_dim * self.config.vocab_size * 2,
        }

        counts['layers_total'] = (
            (counts['attention_per_layer'] + counts['ffn_per_layer']) *
            self.config.num_layers
        )

        counts['total'] = (
            counts['embedding'] +
            counts['positional'] +
            counts['layers_total'] +
            counts['output']
        )

        return counts

    def __repr__(self) -> str:
        param_count = self.get_param_count()['total']
        return (
            f"Pi2Model(\n"
            f"  layers={self.config.num_layers},\n"
            f"  heads={self.config.num_heads},\n"
            f"  embed_dim={self.config.embed_dim},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  params={param_count:,}\n"
            f")"
        )
