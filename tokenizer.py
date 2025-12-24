"""
π/2 Multi-Modal Tokenizer

Tokenizes files (text, audio, image) to token sequences.
Rotation happens in LATENT SPACE during model forward pass, NOT here.

This tokenizer just converts data to tokens - the π/2 magic happens in the model.
"""
import math
import json
from pathlib import Path
from typing import List, Union, Iterator
from dataclasses import dataclass

import numpy as np

from config import ROTATION_ANGLES, HALF_PI

# Optional imports for audio/image
try:
    from scipy.io import wavfile
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class TokenizedSample:
    """A single tokenized sample."""
    tokens: List[int]
    modality: str
    source: str
    length: int


class Pi2Tokenizer:
    """
    Multi-modal tokenizer for π/2 training.

    Converts files to token sequences. Rotation is NOT applied here -
    it happens in latent space during the model forward pass.

    Supports:
    - Text: .txt, .md, .py, .json, .jsonl, etc.
    - Audio: .wav, .mp3, .flac (requires scipy)
    - Image: .png, .jpg, .bmp, .gif (requires PIL)
    """

    # File type mappings
    TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.ts', '.json', '.jsonl',
                       '.yaml', '.yml', '.xml', '.html', '.css', '.c', '.cpp',
                       '.h', '.rs', '.go', '.java', '.rb', '.php', '.sh'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}

    def __init__(
        self,
        vocab_size: int = 65536,
        max_seq_len: int = 512,
        audio_sample_rate: int = 22050,
        image_size: int = 224,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.audio_sample_rate = audio_sample_rate
        self.image_size = image_size

        # Special tokens
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3

        # Byte offset for raw byte encoding
        self.byte_offset = 256  # Tokens 256-511 for raw bytes

    def _tokenize_text(self, text: str) -> List[int]:
        """Byte-level tokenization for text."""
        tokens = [self.bos_token]

        # Convert to bytes and encode
        text_bytes = text.encode('utf-8', errors='replace')
        for b in text_bytes:
            tokens.append(b + self.byte_offset)

        tokens.append(self.eos_token)
        return tokens

    def _tokenize_audio(self, audio_path: Path) -> List[int]:
        """Tokenize audio file using FFT features."""
        ext = audio_path.suffix.lower()

        # Use librosa for MP3/compressed formats, scipy for WAV
        if ext in {'.mp3', '.flac', '.ogg', '.m4a'}:
            if not HAS_LIBROSA:
                raise ImportError("librosa required for MP3/compressed audio: uv pip install librosa")
            # Librosa handles MP3, FLAC, OGG, etc.
            audio, sample_rate = librosa.load(str(audio_path), sr=self.audio_sample_rate, mono=True)
        else:
            # WAV files - use scipy (faster)
            if not HAS_SCIPY:
                raise ImportError("scipy required for WAV audio processing: pip install scipy")
            sample_rate, audio = wavfile.read(str(audio_path))
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Normalize to [-1, 1]
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

        # Compute spectrogram using scipy (works for both loaders)
        if not HAS_SCIPY:
            raise ImportError("scipy required for spectrogram: pip install scipy")

        frequencies, times, spectrogram = signal.spectrogram(
            audio,
            fs=sample_rate,
            nperseg=512,
            noverlap=256
        )

        # Flatten and quantize to vocab range
        spec_flat = spectrogram.flatten()

        # Normalize and map to token range
        spec_norm = (spec_flat - spec_flat.min()) / (spec_flat.max() - spec_flat.min() + 1e-8)
        tokens = [self.bos_token]
        tokens.extend((spec_norm * (self.vocab_size - 256) + 256).astype(int).tolist())
        tokens.append(self.eos_token)

        return tokens

    def _tokenize_image(self, image_path: Path) -> List[int]:
        """Tokenize image using patch-based encoding."""
        if not HAS_PIL:
            raise ImportError("PIL required for image processing: pip install Pillow")

        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        # Convert to numpy array
        arr = np.array(img, dtype=np.float32) / 255.0

        # Create patches (16x16 patches from 224x224 = 196 patches)
        patch_size = 16
        patches = []
        for i in range(0, self.image_size, patch_size):
            for j in range(0, self.image_size, patch_size):
                patch = arr[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch.mean())  # Average intensity per patch

        # Quantize to tokens
        patches = np.array(patches)
        tokens = [self.bos_token]
        tokens.extend((patches * (self.vocab_size - 256) + 256).astype(int).tolist())
        tokens.append(self.eos_token)

        return tokens

    def _tokenize_bytes(self, data: bytes) -> List[int]:
        """Fallback: tokenize raw bytes."""
        tokens = [self.bos_token]
        for b in data:
            tokens.append(b + self.byte_offset)
        tokens.append(self.eos_token)
        return tokens

    def tokenize_file(self, file_path: Union[str, Path]) -> TokenizedSample:
        """
        Tokenize a file to a single sample.

        Rotation happens in model forward pass, not here.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # Determine modality and tokenize
        if ext in self.TEXT_EXTENSIONS:
            modality = "text"
            text = path.read_text(encoding='utf-8', errors='replace')
            tokens = self._tokenize_text(text)
        elif ext in self.AUDIO_EXTENSIONS:
            modality = "audio"
            tokens = self._tokenize_audio(path)
        elif ext in self.IMAGE_EXTENSIONS:
            modality = "image"
            tokens = self._tokenize_image(path)
        else:
            # Fallback: treat as raw bytes
            modality = "binary"
            data = path.read_bytes()
            tokens = self._tokenize_bytes(data)

        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len-1] + [self.eos_token]

        return TokenizedSample(
            tokens=tokens,
            modality=modality,
            source=str(path),
            length=len(tokens)
        )

    def tokenize_text_direct(self, text: str, source: str = "direct") -> TokenizedSample:
        """Tokenize text string directly (no file)."""
        tokens = self._tokenize_text(text)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len-1] + [self.eos_token]

        return TokenizedSample(
            tokens=tokens,
            modality="text",
            source=source,
            length=len(tokens)
        )

    def tokenize_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True
    ) -> Iterator[TokenizedSample]:
        """
        Tokenize all files in a directory.

        Yields one sample per file.
        """
        dir_path = Path(dir_path)
        pattern = "**/*" if recursive else "*"

        valid_extensions = self.TEXT_EXTENSIONS | self.AUDIO_EXTENSIONS | self.IMAGE_EXTENSIONS

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                try:
                    yield self.tokenize_file(file_path)
                except Exception as e:
                    print(f"Warning: Failed to tokenize {file_path}: {e}")

    def save_tokenized(
        self,
        samples: List[TokenizedSample],
        output_path: Union[str, Path]
    ):
        """Save tokenized samples to JSONL file."""
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            for sample in samples:
                record = {
                    "tokens": sample.tokens,
                    "modality": sample.modality,
                    "source": sample.source,
                    "length": sample.length
                }
                f.write(json.dumps(record) + '\n')

        print(f"Saved {len(samples)} samples to {output_path}")

    def load_tokenized(self, input_path: Union[str, Path]) -> List[TokenizedSample]:
        """Load tokenized samples from JSONL file."""
        input_path = Path(input_path)
        samples = []

        with open(input_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                samples.append(TokenizedSample(
                    tokens=record["tokens"],
                    modality=record["modality"],
                    source=record["source"],
                    length=record["length"]
                ))

        return samples


def tokenize_file(file_path: str, output_path: str = None) -> TokenizedSample:
    """Convenience function to tokenize a single file."""
    tokenizer = Pi2Tokenizer()
    sample = tokenizer.tokenize_file(file_path)

    if output_path:
        tokenizer.save_tokenized([sample], output_path)

    return sample


def tokenize_directory(dir_path: str, output_path: str) -> int:
    """Convenience function to tokenize a directory."""
    tokenizer = Pi2Tokenizer()
    samples = list(tokenizer.tokenize_directory(dir_path))
    tokenizer.save_tokenized(samples, output_path)
    return len(samples)


if __name__ == "__main__":
    import sys

    # Demo
    tokenizer = Pi2Tokenizer()

    print("π/2 Multi-Modal Tokenizer")
    print("=" * 50)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max seq len: {tokenizer.max_seq_len}")
    print(f"Rotation angles: {ROTATION_ANGLES}")
    print(f"  (Rotation applied in model latent space, not tokenizer)")
    print()

    # Test with sample text
    test_text = "Hello, this is a test of the π/2 tokenizer."
    sample = tokenizer.tokenize_text_direct(test_text)

    print(f"Input: '{test_text}'")
    print(f"Output: 1 sample (rotation happens in model)")
    print(f"  First 10 tokens: {sample.tokens[:10]}")
    print(f"  Length: {sample.length}")
    print(f"  Modality: {sample.modality}")

    # Test with file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nTokenizing file: {file_path}")
        sample = tokenizer.tokenize_file(file_path)
        print(f"  Tokens: {sample.length}")
        print(f"  Modality: {sample.modality}")

        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            tokenizer.save_tokenized([sample], output_path)
