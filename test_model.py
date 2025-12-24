#!/usr/bin/env python3
"""
π/2 Model Testing Script

Test trained models with text generation and compare A vs B.

Usage:
    # Test single model
    python test_model.py --model output_model_a_phase3/hebbian_model.pt

    # Compare A vs B
    python test_model.py --model-a output_model_a_phase3/hebbian_model.pt \
                         --model-b output_model_b_phase3/hebbian_model.pt

    # Interactive mode
    python test_model.py --model output_model_a_phase3/hebbian_model.pt --interactive
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple

from model import Pi2Model
from config import get_model_config
from quantize import Pi2QuantizedLinear


def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[Pi2Model, dict]:
    """Load a trained model from checkpoint."""
    print(f"Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config
    config = get_model_config("1.57B")

    # Create model
    model = Pi2Model(config, quantized=True)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    info = {
        'step': checkpoint.get('step', '?'),
        'loss': checkpoint.get('avg_loss', checkpoint.get('avg_consistency', '?')),
    }

    print(f"  Loaded from step {info['step']}, loss: {info['loss']}")

    return model, info


def encode_text(text: str, vocab_size: int = 65536) -> torch.Tensor:
    """Simple byte-level encoding."""
    tokens = [1]  # BOS
    for b in text.encode('utf-8', errors='replace'):
        tokens.append(b + 256)
    return torch.tensor(tokens, dtype=torch.long)


def decode_tokens(tokens: torch.Tensor, skip_special: bool = True) -> str:
    """Decode tokens back to text."""
    result = []
    for t in tokens.tolist():
        if skip_special and t < 256:
            continue
        if 256 <= t < 512:
            result.append(t - 256)
    return bytes(result).decode('utf-8', errors='replace')


@torch.no_grad()
def generate(
    model: Pi2Model,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.7,
    device: str = "cuda",
) -> str:
    """Generate text continuation."""

    # Encode prompt
    tokens = encode_text(prompt).unsqueeze(0).to(device)

    generated = []

    for _ in range(max_tokens):
        # Forward pass
        logits, _ = model(tokens)

        # Get last token logits
        next_logits = logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
            next_logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Check for EOS
        if next_token.item() == 2:
            break

        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        # Truncate if too long
        if tokens.shape[1] > 512:
            tokens = tokens[:, -512:]

    return decode_tokens(torch.tensor(generated))


def test_prompts(model: Pi2Model, prompts: List[str], device: str = "cuda"):
    """Test model with various prompts."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n[Prompt]: {prompt}")
        print("-" * 40)

        output = generate(model, prompt, max_tokens=100, device=device)
        print(f"[Output]: {output}")
        print()


def compare_models(
    model_a: Pi2Model,
    model_b: Pi2Model,
    prompts: List[str],
    device: str = "cuda"
):
    """Compare outputs from two models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (A vs B)")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n[Prompt]: {prompt}")
        print("-" * 40)

        output_a = generate(model_a, prompt, max_tokens=100, device=device)
        output_b = generate(model_b, prompt, max_tokens=100, device=device)

        print(f"[Model A]: {output_a}")
        print()
        print(f"[Model B]: {output_b}")
        print()


def interactive_mode(model: Pi2Model, device: str = "cuda"):
    """Interactive text generation."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit, 'settings' to adjust parameters")
    print("=" * 60)

    temperature = 0.8
    max_tokens = 100

    while True:
        try:
            prompt = input("\n[You]: ").strip()

            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'settings':
                try:
                    temperature = float(input("Temperature (0.1-2.0): "))
                    max_tokens = int(input("Max tokens (10-500): "))
                except:
                    print("Invalid input, keeping current settings")
                continue
            elif not prompt:
                continue

            output = generate(model, prompt, max_tokens=max_tokens,
                            temperature=temperature, device=device)
            print(f"[Model]: {output}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def run_benchmarks(model: Pi2Model, device: str = "cuda"):
    """Run standard benchmarks."""
    print("\n" + "=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    # Test prompts covering different capabilities
    benchmarks = {
        "Completion": [
            "The meaning of life is",
            "In the beginning, there was",
            "Once upon a time",
        ],
        "Instruction Following": [
            "Explain quantum computing:",
            "Write a haiku about AI:",
            "List three prime numbers:",
        ],
        "Reasoning": [
            "If A is greater than B, and B is greater than C, then",
            "The capital of France is",
            "2 + 2 equals",
        ],
        "Code": [
            "def fibonacci(n):",
            "# Python function to reverse a string",
            "SELECT * FROM users WHERE",
        ],
        "Metacognition": [
            "I think, therefore",
            "The observer observes the",
            "Consciousness is",
        ],
    }

    for category, prompts in benchmarks.items():
        print(f"\n### {category} ###")
        for prompt in prompts:
            output = generate(model, prompt, max_tokens=50, device=device)
            print(f"  {prompt} → {output[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="π/2 Model Testing")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--model-a", type=str, help="Path to Model A checkpoint (for comparison)")
    parser.add_argument("--model-b", type=str, help="Path to Model B checkpoint (for comparison)")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Comparison mode
    if args.model_a and args.model_b:
        model_a, info_a = load_model(args.model_a, device)
        model_b, info_b = load_model(args.model_b, device)

        prompts = [
            "The meaning of consciousness is",
            "To understand AI, one must first",
            "The rotation preserves",
        ]

        compare_models(model_a, model_b, prompts, device)
        return

    # Single model mode
    model_path = args.model
    if not model_path:
        # Try to find a model
        candidates = [
            "output_model_a_phase3/hebbian_model.pt",
            "output_model_b_phase3/hebbian_model.pt",
            "output_model_a_phase2/hebbian_model.pt",
            "output_model_b_phase2/hebbian_model.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                model_path = c
                break

        if not model_path:
            print("No model found. Specify with --model")
            return

    model, info = load_model(model_path, device)

    if args.prompt:
        output = generate(model, args.prompt, device=device)
        print(f"\n[Prompt]: {args.prompt}")
        print(f"[Output]: {output}")
    elif args.interactive:
        interactive_mode(model, device)
    elif args.benchmark:
        run_benchmarks(model, device)
    else:
        # Default test
        prompts = [
            "Hello, my name is",
            "The rotation of π/2 represents",
            "In the strange loop, we find",
            "def calculate_sum(numbers):",
        ]
        test_prompts(model, prompts, device)


if __name__ == "__main__":
    main()
