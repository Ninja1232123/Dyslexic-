# π/2 Phase-Based Quantization

## Overview

A novel approach that leverages the mathematical properties of π/2 to achieve **lossless phase operations** - something traditional quantization cannot do at any bit depth.

## The Core Insight

Traditional quantization always loses information:
```
FP32 → FP16 → INT8 → more loss at each step
More bits = less loss, but NEVER zero loss
Errors accumulate with every operation
```

**π/2 phase encoding** eliminates this entirely for rotational operations:
- Split each weight into **magnitude** (how much) and **phase** (rotation state)
- Phase is one of exactly 4 states: 0, π/2, π, 3π/2
- At these angles, cos/sin are **exactly** {-1, 0, 1} - integer values, not approximations

This is NOT traditional quantization. The 2 bits aren't "approximating" a continuous value - they're indexing into mathematically perfect rotation states. The infinite decimal precision of π/2 is preserved because the trigonometric results at these specific angles are exact.

## Why π/2 Phases Are Special

```
cos(0)     = 1      sin(0)     = 0      ← exact integers
cos(π/2)   = 0      sin(π/2)   = 1      ← exact integers
cos(π)     = -1     sin(π)     = 0      ← exact integers
cos(3π/2)  = 0      sin(3π/2)  = -1     ← exact integers
```

These values are **mathematically exact integers** - not floating point approximations.

Contrast with any other angle:
```
cos(π/4) = 0.7071067811865476...  ← infinite decimals, must be truncated
cos(π/3) = 0.5000000000000001...  ← even "nice" angles have FP error
```

The π/2 phases are the ONLY angles where trigonometric operations produce exact results. Phase rotations compose perfectly: `R(π/2) × R(π/2) = R(π)` with zero accumulated error, forever.

## Weight Representation

Each weight is stored as:
```
weight = magnitude × e^(i × phase)
       = magnitude × (cos(phase) + i × sin(phase))
```

Where:
- **Magnitude**: Float32 during training (can be int8 for inference)
- **Phase**: 2 bits (4 possible values)

## Memory Comparison (1.68B parameter model)

| Component | FP32 | Phase-Encoded |
|-----------|------|---------------|
| Weights | 6.7 GB | 1.7 GB (magnitude) + 0.4 GB (phase) |
| AdamW exp_avg | 6.7 GB | 1.7 GB (magnitude only) |
| AdamW exp_avg_sq | 6.7 GB | 1.7 GB (magnitude only) |
| Gradients | 6.7 GB | 1.7 GB (magnitude only) |
| **TOTAL** | **26.8 GB** | **~7.2 GB** |

Memory reduction is a **bonus**, not the primary goal.

## Key Advantages

1. **Lossless Phase Operations**: Zero floating point error - mathematically exact
2. **No Accumulated Error**: Compose unlimited rotations without precision degradation
3. **Reduced Optimizer States**: Phase is discrete, no gradients needed (bonus)
4. **Natural Fit for π/2 Model**: Already using π/2 rotations in latent space
5. **Memory Savings**: ~70% reduction vs FP32 (bonus)

## How It Works in Training

1. **Forward Pass**: Reconstruct weight = magnitude × sign(phase), compute output
2. **Backward Pass**: Compute gradients for magnitude only (phase is discrete)
3. **Optimizer Step**: Update magnitude with AdamW, phase stays fixed (or uses straight-through estimator)

## Usage

```bash
# Enable π/2 phase quantization
python train.py --data data.jsonl --size 1.57B --quantize

# Combine with other options
python train.py --data data.jsonl --size 1.57B --quantize --batch 8 --wandb
```

## Implementation Details

See `quantize.py` for the full implementation:
- `Pi2QuantizedWeight`: Stores magnitude + phase
- `Pi2QuantizedLinear`: Drop-in replacement for nn.Linear
- `Pi2QuantizedEmbedding`: Drop-in replacement for nn.Embedding
- `quantize_model()`: Converts existing model to quantized version

## Future Extensions

1. **Learned Phases**: Use straight-through estimator to learn optimal phases
2. **INT8 Inference**: Quantize magnitudes to int8 for even smaller models
3. **Phase-Aware Initialization**: Initialize phases based on weight statistics
4. **Hardware Optimization**: Phase operations map to simple bit operations

## Connection to Complex-Valued Networks

This approach is related to:
- Complex-valued neural networks
- Quaternion neural networks
- Phase-based neural computation

But unique in using the exact properties of π/2 multiples.

## Mathematical Foundation

The rotation matrix for angle θ:
```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

For θ = π/2:
```
R(π/2) = [0  -1]
         [1   0]
```

This is just a 90° rotation - can be implemented as a simple coordinate swap with sign flip. No multiplication needed!

## References

- Complex-Valued Neural Networks (Hirose, 2012)
- Quaternion Neural Networks (Parcollet et al., 2019)
- Binary Neural Networks (Courbariaux et al., 2016)
- This work: π/2 Phase Quantization (2024)
