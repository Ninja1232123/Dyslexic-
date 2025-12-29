"""
FFT-Based Multimodal Tokenizer
Converts all modalities (text, image, audio) into unified frequency-domain tokens.
"""
import math
import numpy as np
from typing import Union, List, Optional
from .config import Pi2Config


class FFTTokenizer:
    """
    FFT-based tokenizer that encodes all modalities into a unified frequency space.

    Key concept: Use FFT to convert inputs into frequency-based tokens,
    then apply π/2 phase shifts for harmonic alignment across modalities.
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.pi_2 = config.pi_2
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim

        # Build vocabulary mappings for text
        self._build_text_vocab()

    def _build_text_vocab(self):
        """Initialize text vocabulary (placeholder - use BPE in production)."""
        # Special tokens
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        self.modality_tokens = {
            'text': 4,
            'image': 5,
            'audio': 6,
        }

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to frequency-domain tokens."""
        # Convert text to byte representation
        bytes_data = np.array([b for b in text.encode('utf-8')], dtype=np.float32)

        # Apply FFT to convert to frequency domain
        freq_tokens = self._apply_fft(bytes_data)

        # Apply π/2 phase shift for text modality
        phase_shifted = self._apply_phase_shift(freq_tokens, phase=0)

        return phase_shifted

    def encode_image(self, image_data: np.ndarray) -> np.ndarray:
        """Encode image to frequency-domain tokens."""
        # Flatten image and apply FFT
        flat_data = image_data.flatten().astype(np.float32)
        freq_tokens = self._apply_fft(flat_data)

        # Apply π/2 phase shift for image modality
        phase_shifted = self._apply_phase_shift(freq_tokens, phase=self.pi_2)

        return phase_shifted

    def encode_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Encode audio to frequency-domain tokens."""
        freq_tokens = self._apply_fft(audio_data.astype(np.float32))

        # Apply π phase shift for audio modality
        phase_shifted = self._apply_phase_shift(freq_tokens, phase=math.pi)

        return phase_shifted

    def _apply_fft(self, data: np.ndarray) -> np.ndarray:
        """
        Apply FFT to convert signal to frequency domain.
        Returns complex numbers representing frequency components.
        """
        # Pad or truncate to embed_dim
        if len(data) < self.embed_dim:
            data = np.pad(data, (0, self.embed_dim - len(data)))
        else:
            data = data[:self.embed_dim]

        # Apply FFT
        freq_domain = np.fft.fft(data)

        return freq_domain

    def _apply_phase_shift(self, freq_data: np.ndarray, phase: float) -> np.ndarray:
        """
        Apply π/2-based phase shift to frequency-domain tokens.
        This is the key operation for harmonic alignment.

        Phase shift in frequency domain: multiply by e^(i * phase)
        """
        phase_factor = np.exp(1j * phase)
        shifted = freq_data * phase_factor

        return shifted

    def to_tokens(self, freq_data: np.ndarray) -> np.ndarray:
        """
        Convert frequency-domain data to discrete token IDs.
        Uses the infinite decimal precision concept.
        """
        # Extract magnitude and phase
        magnitude = np.abs(freq_data)
        phase = np.angle(freq_data)

        # Quantize to 2-bit (4 states) but preserve precision via phase
        # The phase encodes the "infinite decimal" precision
        phase_states = np.floor((phase + np.pi) / (self.pi_2)) % 4

        # Combine magnitude quantization with phase state
        mag_quantized = np.floor(magnitude / magnitude.max() * (self.vocab_size // 4))
        tokens = (mag_quantized * 4 + phase_states).astype(np.int32)

        # Clamp to vocab size
        tokens = np.clip(tokens, 0, self.vocab_size - 1)

        return tokens

    def from_tokens(self, tokens: np.ndarray, original_phase: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode tokens back to frequency-domain representation.
        Uses stored phase information for precision recovery.
        """
        # Extract phase state and magnitude from tokens
        phase_states = tokens % 4
        mag_quantized = tokens // 4

        # Reconstruct phase
        if original_phase is not None:
            phase = original_phase
        else:
            phase = phase_states * self.pi_2 - np.pi

        # Reconstruct magnitude (normalized)
        magnitude = mag_quantized / (self.vocab_size // 4)

        # Reconstruct complex frequency data
        freq_data = magnitude * np.exp(1j * phase)

        return freq_data

    def encode(self, data: Union[str, np.ndarray], modality: str = 'text') -> np.ndarray:
        """
        Unified encoding interface for all modalities.
        """
        if modality == 'text':
            if isinstance(data, str):
                return self.encode_text(data)
            raise ValueError("Text modality requires string input")
        elif modality == 'image':
            return self.encode_image(data)
        elif modality == 'audio':
            return self.encode_audio(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def decode_to_signal(self, freq_data: np.ndarray) -> np.ndarray:
        """Apply inverse FFT to recover time/space domain signal."""
        return np.fft.ifft(freq_data).real


class HarmonicEncoder:
    """
    Encodes the harmonic relationships between modalities using π/2 rotation.
    This enables natural multimodal understanding.
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.pi_2 = config.pi_2

    def align_modalities(self, *freq_arrays: np.ndarray) -> np.ndarray:
        """
        Align multiple modalities in a unified harmonic space.
        Each modality is phase-shifted to create coherent representation.
        """
        aligned = []
        for i, freq_data in enumerate(freq_arrays):
            # Rotate each modality by i * π/2
            phase_shift = i * self.pi_2
            rotated = freq_data * np.exp(1j * phase_shift)
            aligned.append(rotated)

        # Stack aligned modalities
        return np.stack(aligned, axis=0)

    def harmonic_mix(self, aligned_data: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mix aligned modalities using harmonic weighting.
        """
        if weights is None:
            weights = np.ones(len(aligned_data)) / len(aligned_data)

        # Weighted sum in frequency domain
        mixed = np.sum(aligned_data * weights[:, np.newaxis], axis=0)

        return mixed
