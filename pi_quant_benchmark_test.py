#!/usr/bin/env python3
"""
π/2 Quantization vs INT4/INT8 Benchmark Test
=============================================

Raw comparison of quantization methods on pre-trained models.
No special training - just quantize and measure.

Tests:
1. Weight distribution after quantization
2. Reconstruction error (how much info lost)
3. Perplexity on test data
4. Memory footprint
5. Inference speed

Run:
    python pi_quant_benchmark_test.py --model gpt2 --device cuda
    python pi_quant_benchmark_test.py --model gpt2-medium --samples 1000
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Try imports - graceful fallback
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Install with: pip install transformers")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: datasets not installed. Using synthetic data.")


# =============================================================================
# Constants
# =============================================================================

PI = math.pi
HALF_PI = PI / 2
QUARTER_PI = PI / 4
TWO_PI = PI * 2
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio


# =============================================================================
# Quantization Methods
# =============================================================================

class QuantizationMethod:
    """Base class for quantization methods."""

    def __init__(self, name: str, bits: float):
        self.name = name
        self.bits = bits

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_memory_bytes(self, numel: int) -> int:
        """Estimate memory usage in bytes."""
        return int(numel * self.bits / 8)


class INT8Quantization(QuantizationMethod):
    """Standard INT8 quantization."""

    def __init__(self):
        super().__init__("INT8", 8)
        self.scale = None
        self.zero_point = None

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # Symmetric quantization
        max_val = tensor.abs().max()
        self.scale = max_val / 127.0

        if self.scale == 0:
            self.scale = 1.0

        quantized = torch.clamp(torch.round(tensor / self.scale), -128, 127)
        return quantized.to(torch.int8)

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() * self.scale


class INT4Quantization(QuantizationMethod):
    """INT4 quantization (simulated with int8 storage)."""

    def __init__(self):
        super().__init__("INT4", 4)
        self.scale = None

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        max_val = tensor.abs().max()
        self.scale = max_val / 7.0  # 4-bit signed: -8 to 7

        if self.scale == 0:
            self.scale = 1.0

        quantized = torch.clamp(torch.round(tensor / self.scale), -8, 7)
        return quantized.to(torch.int8)  # Store in int8, but only 4 bits used

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() * self.scale


class PiHalfQuantization(QuantizationMethod):
    """
    π/2 Quantization - weights as multiples of π/2.

    The hypothesis: Using irrational numbers as quantization levels
    preserves more information in the relationships between weights.
    """

    def __init__(self, levels: int = 32):
        # Effective bits based on number of levels
        bits = math.log2(levels)
        super().__init__(f"π/2 ({levels} levels)", bits)
        self.levels = levels
        self.scale = None
        self.base = HALF_PI

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # Find scale to map weights to π/2 multiples
        max_val = tensor.abs().max()
        max_multiple = self.levels // 2
        self.scale = max_val / (max_multiple * self.base)

        if self.scale == 0:
            self.scale = 1.0

        # Quantize to multiples of π/2
        multiples = torch.round(tensor / (self.scale * self.base))
        multiples = torch.clamp(multiples, -max_multiple, max_multiple - 1)

        return multiples

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.scale * self.base


class PiHalfContinuousQuantization(QuantizationMethod):
    """
    π/2 Continuous - scale by π/2 but keep more precision in decimals.

    The idea: The irrational nature of π/2 means the decimal places
    carry information that integer quantization truncates.
    """

    def __init__(self, precision_bits: int = 8):
        super().__init__(f"π/2 Continuous ({precision_bits}-bit)", precision_bits)
        self.precision_bits = precision_bits
        self.scale = None

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # Scale to π/2 space
        scaled = tensor / HALF_PI

        # Quantize the scaled values
        max_val = scaled.abs().max()
        levels = 2 ** self.precision_bits
        self.scale = max_val / (levels / 2)

        if self.scale == 0:
            self.scale = 1.0

        quantized = torch.round(scaled / self.scale)
        quantized = torch.clamp(quantized, -levels/2, levels/2 - 1)

        return quantized

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # Unquantize and scale back from π/2 space
        return tensor * self.scale * HALF_PI


class PhiQuantization(QuantizationMethod):
    """Golden ratio (φ) quantization for comparison."""

    def __init__(self, levels: int = 32):
        bits = math.log2(levels)
        super().__init__(f"φ Golden ({levels} levels)", bits)
        self.levels = levels
        self.scale = None
        self.base = PHI

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        max_val = tensor.abs().max()
        max_multiple = self.levels // 2
        self.scale = max_val / (max_multiple * self.base)

        if self.scale == 0:
            self.scale = 1.0

        multiples = torch.round(tensor / (self.scale * self.base))
        multiples = torch.clamp(multiples, -max_multiple, max_multiple - 1)

        return multiples

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.scale * self.base


# =============================================================================
# Benchmark Framework
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single quantization benchmark."""
    method_name: str
    bits: float

    # Reconstruction metrics
    mse: float = 0.0                    # Mean squared error vs original
    mae: float = 0.0                    # Mean absolute error
    max_error: float = 0.0              # Maximum error
    cosine_similarity: float = 0.0      # Cosine sim to original

    # Distribution metrics
    original_mean: float = 0.0
    original_std: float = 0.0
    quantized_mean: float = 0.0
    quantized_std: float = 0.0

    # Memory
    original_bytes: int = 0
    quantized_bytes: int = 0
    compression_ratio: float = 0.0

    # Performance (if model tested)
    perplexity: Optional[float] = None
    inference_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "bits": self.bits,
            "mse": self.mse,
            "mae": self.mae,
            "max_error": self.max_error,
            "cosine_similarity": self.cosine_similarity,
            "compression_ratio": self.compression_ratio,
            "perplexity": self.perplexity,
            "inference_time_ms": self.inference_time_ms
        }


class QuantizationBenchmark:
    """Benchmark harness for comparing quantization methods."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: List[BenchmarkResult] = []

        # Initialize quantization methods
        self.methods = {
            "INT8": INT8Quantization(),
            "INT4": INT4Quantization(),
            "π/2 (16 levels)": PiHalfQuantization(levels=16),
            "π/2 (32 levels)": PiHalfQuantization(levels=32),
            "π/2 (256 levels)": PiHalfQuantization(levels=256),
            "π/2 Continuous 8-bit": PiHalfContinuousQuantization(precision_bits=8),
            "π/2 Continuous 4-bit": PiHalfContinuousQuantization(precision_bits=4),
            "φ Golden (32 levels)": PhiQuantization(levels=32),
        }

    def benchmark_tensor(self, tensor: torch.Tensor, method: QuantizationMethod) -> BenchmarkResult:
        """Benchmark a single quantization method on a tensor."""
        original = tensor.float().to(self.device)

        # Quantize and dequantize
        quantized = method.quantize(original.clone())
        reconstructed = method.dequantize(quantized)

        # Compute metrics
        diff = original - reconstructed

        mse = (diff ** 2).mean().item()
        mae = diff.abs().mean().item()
        max_error = diff.abs().max().item()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original.flatten().unsqueeze(0),
            reconstructed.flatten().unsqueeze(0)
        ).item()

        # Memory
        original_bytes = original.numel() * 4  # FP32
        quantized_bytes = method.get_memory_bytes(original.numel())

        return BenchmarkResult(
            method_name=method.name,
            bits=method.bits,
            mse=mse,
            mae=mae,
            max_error=max_error,
            cosine_similarity=cos_sim,
            original_mean=original.mean().item(),
            original_std=original.std().item(),
            quantized_mean=reconstructed.mean().item(),
            quantized_std=reconstructed.std().item(),
            original_bytes=original_bytes,
            quantized_bytes=quantized_bytes,
            compression_ratio=original_bytes / max(1, quantized_bytes)
        )

    def benchmark_all_methods(self, tensor: torch.Tensor) -> List[BenchmarkResult]:
        """Run all quantization methods on a tensor."""
        results = []

        for name, method in self.methods.items():
            result = self.benchmark_tensor(tensor, method)
            results.append(result)

        return results

    def benchmark_model_weights(self, model: nn.Module) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark quantization on all model weights."""
        all_results = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 100:  # Skip tiny params
                results = self.benchmark_all_methods(param.data)
                all_results[name] = results

        return all_results

    def aggregate_results(self, results_by_layer: Dict[str, List[BenchmarkResult]]) -> Dict[str, BenchmarkResult]:
        """Aggregate results across all layers."""
        method_totals = {}

        for layer_name, results in results_by_layer.items():
            for result in results:
                if result.method_name not in method_totals:
                    method_totals[result.method_name] = {
                        "mse_sum": 0, "mae_sum": 0, "cos_sum": 0,
                        "count": 0, "bits": result.bits,
                        "total_original": 0, "total_quantized": 0
                    }

                totals = method_totals[result.method_name]
                totals["mse_sum"] += result.mse
                totals["mae_sum"] += result.mae
                totals["cos_sum"] += result.cosine_similarity
                totals["count"] += 1
                totals["total_original"] += result.original_bytes
                totals["total_quantized"] += result.quantized_bytes

        # Compute averages
        aggregated = {}
        for method_name, totals in method_totals.items():
            count = totals["count"]
            aggregated[method_name] = BenchmarkResult(
                method_name=method_name,
                bits=totals["bits"],
                mse=totals["mse_sum"] / count,
                mae=totals["mae_sum"] / count,
                cosine_similarity=totals["cos_sum"] / count,
                original_bytes=totals["total_original"],
                quantized_bytes=totals["total_quantized"],
                compression_ratio=totals["total_original"] / max(1, totals["total_quantized"])
            )

        return aggregated


# =============================================================================
# Perplexity Testing
# =============================================================================

def compute_perplexity(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    device: str = "cpu"
) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    model.to(device)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )

            input_ids = encodings.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)

    return perplexity


def quantize_model_weights(model: nn.Module, method: QuantizationMethod) -> nn.Module:
    """Apply quantization to all model weights (in-place simulation)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                quantized = method.quantize(param.data)
                dequantized = method.dequantize(quantized)
                param.data.copy_(dequantized)

    return model


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_weight_benchmark(model_name: str = "gpt2", device: str = "cpu"):
    """Run weight-level quantization benchmark."""
    print("\n" + "=" * 70)
    print("WEIGHT QUANTIZATION BENCHMARK")
    print("=" * 70)

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers library required")
        return None

    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Run benchmark
    benchmark = QuantizationBenchmark(device=device)

    print("\nBenchmarking quantization methods on model weights...")
    results_by_layer = benchmark.benchmark_model_weights(model)

    # Aggregate
    aggregated = benchmark.aggregate_results(results_by_layer)

    # Print results
    print("\n" + "-" * 70)
    print("AGGREGATED RESULTS (averaged across all layers)")
    print("-" * 70)
    print(f"{'Method':<25} {'Bits':>6} {'MSE':>12} {'Cosine Sim':>12} {'Compression':>12}")
    print("-" * 70)

    # Sort by MSE
    sorted_results = sorted(aggregated.values(), key=lambda x: x.mse)

    for result in sorted_results:
        print(f"{result.method_name:<25} {result.bits:>6.2f} {result.mse:>12.6f} "
              f"{result.cosine_similarity:>12.6f} {result.compression_ratio:>11.1f}x")

    return aggregated


def run_perplexity_benchmark(
    model_name: str = "gpt2",
    num_samples: int = 100,
    device: str = "cpu"
):
    """Run perplexity benchmark comparing quantization methods."""
    print("\n" + "=" * 70)
    print("PERPLEXITY BENCHMARK")
    print("=" * 70)

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers library required")
        return None

    # Load test data
    if HAS_DATASETS:
        print("\nLoading WikiText-2 test data...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t) > 100][:num_samples]
    else:
        print("\nUsing synthetic test data...")
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "In the beginning, there was nothing but void and darkness. " * 20,
            "Machine learning models require large amounts of training data. " * 20,
        ] * (num_samples // 3)

    print(f"Test samples: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    methods_to_test = {
        "FP32 (baseline)": None,
        "INT8": INT8Quantization(),
        "INT4": INT4Quantization(),
        "π/2 (32 levels)": PiHalfQuantization(levels=32),
        "π/2 (256 levels)": PiHalfQuantization(levels=256),
        "π/2 Continuous 8-bit": PiHalfContinuousQuantization(precision_bits=8),
        "φ Golden (32 levels)": PhiQuantization(levels=32),
    }

    results = {}

    for method_name, method in methods_to_test.items():
        print(f"\nTesting: {method_name}")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)

        # Apply quantization if not baseline
        if method is not None:
            model = quantize_model_weights(model, method)

        # Measure inference time
        start_time = time.time()
        perplexity = compute_perplexity(model, tokenizer, texts[:20], device=device)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Full perplexity
        full_perplexity = compute_perplexity(model, tokenizer, texts, device=device)

        results[method_name] = {
            "perplexity": full_perplexity,
            "inference_time_ms": inference_time,
            "bits": method.bits if method else 32
        }

        print(f"  Perplexity: {full_perplexity:.2f}")
        print(f"  Inference time (20 samples): {inference_time:.1f}ms")

        # Clean up
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Print comparison
    print("\n" + "-" * 70)
    print("PERPLEXITY COMPARISON")
    print("-" * 70)
    print(f"{'Method':<25} {'Bits':>6} {'Perplexity':>12} {'vs Baseline':>12}")
    print("-" * 70)

    baseline_ppl = results["FP32 (baseline)"]["perplexity"]

    for method_name, data in sorted(results.items(), key=lambda x: x[1]["perplexity"]):
        diff = data["perplexity"] - baseline_ppl
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{method_name:<25} {data['bits']:>6.1f} {data['perplexity']:>12.2f} {diff_str:>12}")

    return results


def run_synthetic_benchmark():
    """Run benchmark on synthetic weight distributions."""
    print("\n" + "=" * 70)
    print("SYNTHETIC WEIGHT DISTRIBUTION BENCHMARK")
    print("=" * 70)

    benchmark = QuantizationBenchmark()

    # Test different weight distributions
    distributions = {
        "Normal (σ=0.02)": torch.randn(1000, 1000) * 0.02,
        "Normal (σ=0.1)": torch.randn(1000, 1000) * 0.1,
        "Uniform [-0.1, 0.1]": torch.rand(1000, 1000) * 0.2 - 0.1,
        "Sparse (90% zero)": torch.randn(1000, 1000) * 0.02 * (torch.rand(1000, 1000) > 0.9).float(),
        "Heavy-tailed": torch.randn(1000, 1000) * 0.02 + torch.randn(1000, 1000) * 0.1 * (torch.rand(1000, 1000) > 0.95).float(),
    }

    for dist_name, tensor in distributions.items():
        print(f"\n{dist_name}:")
        print(f"  Shape: {tensor.shape}, Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
        print("-" * 60)
        print(f"  {'Method':<25} {'MSE':>12} {'Cosine Sim':>12}")

        results = benchmark.benchmark_all_methods(tensor)

        for result in sorted(results, key=lambda x: x.mse):
            print(f"  {result.method_name:<25} {result.mse:>12.8f} {result.cosine_similarity:>12.6f}")

    return distributions


# =============================================================================
# Summary Report
# =============================================================================

def generate_report(weight_results, perplexity_results, output_path: str = "benchmark_report.json"):
    """Generate a summary report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weight_benchmark": {},
        "perplexity_benchmark": {},
        "summary": {}
    }

    if weight_results:
        for method_name, result in weight_results.items():
            report["weight_benchmark"][method_name] = result.to_dict()

    if perplexity_results:
        report["perplexity_benchmark"] = perplexity_results

        # Find winner
        baseline_ppl = perplexity_results.get("FP32 (baseline)", {}).get("perplexity", 0)

        best_compressed = None
        best_ppl = float('inf')

        for method, data in perplexity_results.items():
            if method != "FP32 (baseline)" and data["perplexity"] < best_ppl:
                best_ppl = data["perplexity"]
                best_compressed = method

        report["summary"] = {
            "baseline_perplexity": baseline_ppl,
            "best_compressed_method": best_compressed,
            "best_compressed_perplexity": best_ppl,
            "perplexity_increase": best_ppl - baseline_ppl if baseline_ppl else None
        }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="π/2 Quantization Benchmark")
    parser.add_argument("--model", default="gpt2", help="Model to benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--synthetic-only", action="store_true", help="Only run synthetic benchmark")
    parser.add_argument("--weights-only", action="store_true", help="Only run weight benchmark")
    parser.add_argument("--perplexity-only", action="store_true", help="Only run perplexity benchmark")
    parser.add_argument("--output", default="benchmark_report.json", help="Output report path")

    args = parser.parse_args()

    print("=" * 70)
    print("π/2 QUANTIZATION vs INT4/INT8 BENCHMARK")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.samples}")

    weight_results = None
    perplexity_results = None

    # Run benchmarks
    if args.synthetic_only:
        run_synthetic_benchmark()
    else:
        if not args.perplexity_only:
            weight_results = run_weight_benchmark(args.model, args.device)

        if not args.weights_only:
            perplexity_results = run_perplexity_benchmark(args.model, args.samples, args.device)

    # Generate report
    if weight_results or perplexity_results:
        report = generate_report(weight_results, perplexity_results, args.output)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if report.get("summary"):
            s = report["summary"]
            print(f"Baseline (FP32) Perplexity: {s.get('baseline_perplexity', 'N/A')}")
            print(f"Best Compressed Method: {s.get('best_compressed_method', 'N/A')}")
            print(f"Best Compressed Perplexity: {s.get('best_compressed_perplexity', 'N/A')}")

            if s.get('perplexity_increase') is not None:
                print(f"Perplexity Increase: +{s['perplexity_increase']:.2f}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
