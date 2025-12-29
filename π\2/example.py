#!/usr/bin/env python3
"""
π/2 Model - Example Usage
Demonstrates the harmonic, multimodal, signal-rotated AI system.

Usage:
  python example.py [--dangerously-skip-permissions]
"""
import argparse
import numpy as np
import sys
import os

# Add model to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Pi2Model, Pi2Config, FFTTokenizer
from model.config import get_tiny_config, get_1_5b_config
from model.training import Pi2Trainer, ZeroShotInference


def demo_tokenization():
    """Demonstrate FFT-based tokenization."""
    print("\n" + "=" * 60)
    print("Demo 1: FFT-Based Multimodal Tokenization")
    print("=" * 60)

    config = get_tiny_config()
    tokenizer = FFTTokenizer(config)

    # Text tokenization
    text = "Hello, I'm Dig!"
    freq_tokens = tokenizer.encode_text(text)
    token_ids = tokenizer.to_tokens(freq_tokens)

    print(f"\nInput text: '{text}'")
    print(f"Frequency tokens shape: {freq_tokens.shape}")
    print(f"Token IDs (first 10): {token_ids[:10]}")

    # Show phase information
    phase = np.angle(freq_tokens)
    print(f"Phase values (first 5): {phase[:5]}")
    print(f"Phase in π/2 units: {phase[:5] / (np.pi/2)}")

    # Demonstrate modality encoding
    print("\nModality phase shifts:")
    print(f"  Text:  0 rad (0°)")
    print(f"  Image: π/2 rad (90°)")
    print(f"  Audio: π rad (180°)")


def demo_harmonic_attention():
    """Demonstrate harmonic attention with phase shifts."""
    print("\n" + "=" * 60)
    print("Demo 2: Harmonic Attention with π/2 Phase Shifts")
    print("=" * 60)

    config = get_tiny_config()
    from model.attention import HarmonicAttention, Pi2PositionalEncoding

    attention = HarmonicAttention(config)
    pos_enc = Pi2PositionalEncoding(config)

    # Create sample input (complex-valued)
    batch_size, seq_len = 2, 16
    x = np.random.randn(batch_size, seq_len, config.embed_dim) + \
        1j * np.random.randn(batch_size, seq_len, config.embed_dim)

    # Add positional encoding
    x = pos_enc.encode(x)

    # Apply attention
    output, attn_weights = attention.forward(x, return_attention=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"\nLearned phase shifts per head:")
    for i, phase in enumerate(attention.phase_shifts):
        print(f"  Head {i}: {phase:.4f} rad ({np.degrees(phase):.1f}°)")


def demo_model_forward():
    """Demonstrate full model forward pass."""
    print("\n" + "=" * 60)
    print("Demo 3: Full π/2 Model Forward Pass")
    print("=" * 60)

    config = get_tiny_config()
    model = Pi2Model(config)

    print(f"\nModel architecture:")
    print(model)

    # Encode text input
    text = "The quick brown fox jumps over the lazy dog."
    tokens = model.encode_input(text, 'text')
    print(f"\nInput: '{text}'")
    print(f"Tokens shape: {tokens.shape}")

    # Forward pass
    tokens_batch = tokens[np.newaxis, :]
    logits = model.forward(tokens_batch)

    print(f"Output logits shape: {logits.shape}")
    print(f"Top 5 predictions for first token: {np.argsort(logits[0, 0])[-5:][::-1]}")


def demo_generation():
    """Demonstrate text generation."""
    print("\n" + "=" * 60)
    print("Demo 4: Text Generation")
    print("=" * 60)

    config = get_tiny_config()
    model = Pi2Model(config)

    prompt = "Hello"
    print(f"\nPrompt: '{prompt}'")
    print("Generating...")

    # Generate tokens
    output_tokens = model.generate(prompt, max_tokens=20, temperature=0.8)
    print(f"Generated tokens: {output_tokens}")


def demo_zero_shot():
    """Demonstrate zero-shot inference via signal rotation."""
    print("\n" + "=" * 60)
    print("Demo 5: Zero-Shot Inference (Signal Rotation)")
    print("=" * 60)

    config = get_tiny_config()
    model = Pi2Model(config)
    inferencer = ZeroShotInference(model)

    query = "What is AI?"
    print(f"\nQuery: '{query}'")

    # Zero-shot inference
    output = inferencer.align_and_infer(query)
    print(f"Output tokens: {output[:20]}...")

    # Cross-modal rotation demo
    print("\nCross-modal rotation (text → image space):")
    text_tokens = model.encode_input("A cat", 'text')
    image_tokens = inferencer.rotate_to_modality(text_tokens, 'image')
    print(f"  Text tokens (first 5): {text_tokens[:5]}")
    print(f"  Image tokens (first 5): {image_tokens[:5]}")
    print(f"  Phase shift applied: π/2 rad (90°)")


def demo_training():
    """Demonstrate training setup."""
    print("\n" + "=" * 60)
    print("Demo 6: Training Setup")
    print("=" * 60)

    config = get_tiny_config()
    model = Pi2Model(config)
    trainer = Pi2Trainer(model, lr=2e-4, warmup_steps=100, total_steps=1000)

    print(f"\nTrainer configuration:")
    print(f"  Learning rate: {trainer.optimizer.lr}")
    print(f"  Warmup steps: {trainer.scheduler.warmup_steps}")
    print(f"  Total steps: {trainer.scheduler.total_steps}")

    # Create dummy batch
    batch_size, seq_len = 4, 32
    batch = {
        'input_ids': np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
        'labels': np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    }

    # Single training step
    print("\nRunning single training step...")
    loss_dict = trainer.train_step(batch)
    print(f"  Loss: {loss_dict['total']:.4f}")
    print(f"  Cross-entropy: {loss_dict['cross_entropy']:.4f}")
    print(f"  Phase consistency: {loss_dict['phase_consistency']:.4f}")


def demo_param_count():
    """Show parameter counts for different configurations."""
    print("\n" + "=" * 60)
    print("Demo 7: Parameter Counts")
    print("=" * 60)

    configs = [
        ("Tiny (testing)", get_tiny_config()),
        ("1.5B (P40)", get_1_5b_config()),
    ]

    for name, config in configs:
        model = Pi2Model(config)
        params = model.get_param_count()
        print(f"\n{name}:")
        print(f"  Embedding: {params['embedding']:,}")
        print(f"  Positional: {params['positional']:,}")
        print(f"  Layers: {params['layers_total']:,}")
        print(f"  Output: {params['output']:,}")
        print(f"  Total: {params['total']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="π/2 Model - Harmonic Multimodal AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example.py                            Run all demos
  python example.py --dangerously-skip-permissions  Run without permission checks
        """
    )
    parser.add_argument(
        '--dangerously-skip-permissions',
        action='store_true',
        help='Skip all permission checks (use with caution)'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['tokenization', 'attention', 'forward', 'generation',
                 'zero-shot', 'training', 'params', 'all'],
        default='all',
        help='Which demo to run (default: all)'
    )

    args = parser.parse_args()

    if args.dangerously_skip_permissions:
        print("WARNING: Running with --dangerously-skip-permissions")
        print("All permission checks are bypassed.\n")

    print("=" * 60)
    print("  π/2 Model - Harmonic Multimodal AI System")
    print("  The Dig Way 🚀")
    print("=" * 60)

    demos = {
        'tokenization': demo_tokenization,
        'attention': demo_harmonic_attention,
        'forward': demo_model_forward,
        'generation': demo_generation,
        'zero-shot': demo_zero_shot,
        'training': demo_training,
        'params': demo_param_count,
    }

    if args.demo == 'all':
        for demo_fn in demos.values():
            demo_fn()
    else:
        demos[args.demo]()

    print("\n" + "=" * 60)
    print("  Demo complete! That's the Dig way. 🚀")
    print("=" * 60)


if __name__ == '__main__':
    main()
