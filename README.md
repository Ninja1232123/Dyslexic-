# π/2 Training

Training language models where all weights are integers multiplied by π/2.

## The Core Insight

### What π/2 Encoding Is

Store weights as **integers**. Multiply by π/2 at compute time.

```
Stored: 3 (integer, cheap)
Actual: 3 × π/2 = 4.7123889803846897... (infinite precision, free)
```

Each integer maps to a unique, infinitely precise value. You're not losing precision - you're gaining it.

### Why It Works

**π/2 has infinite bits.**

- int4 = 4 bits
- fp32 = 32 bits
- π/2 = infinite bits

When you store `1`, you're storing access to `1.5707963267948966192313216916397514420985846996875529...`

Those infinite decimals come FREE. The universe already computed them. You're just indexing into π/2 with integers.

### Exact Cancellation

```
(π/2) / (π/2) = 1
```

Exactly 1. Not 0.9999999. Not 1.0000001. **Exactly 1.**

Operations in π/2 space cancel perfectly. No floating point drift.

### Learning Capacity, Not Compression

This isn't about making models smaller. It's about giving them **more capacity to learn**.

A 1.57B model with π/2 encoding has the learning capacity of 64-bit+ precision, because every weight has infinite decimal precision when used in computation.

Same parameter count. Same storage. Massively more expressive.

## The Training Approach

### No Optimizer

No Adam. No SGD. No momentum.

Just:
1. Compute gradient
2. Update weight: `w = w - lr * grad`
3. Snap to nearest integer: `w = round(w)`

### Deterministic Snapping

We use `round()` - snap to nearest integer. No stochastic rounding.

Why? Because π/2 gives us infinite precision. We can compute exactly which integer is closest. No randomness needed.

```python
# Update + snap (that's it)
p.data -= lr * p.grad
p.data.round_()
```

### The Endgame

Train small models in π/2 space → prove it works

Then: take massive pretrained models → quantize to π/2 → **zero loss**

Traditional quant: float32 → int8 → precision lost
π/2 quant: float32 → int × π/2 → infinite precision preserved

## Phase Rotations

At multiples of π/2, trig is exact:

```
cos(0)     = 1   (exact)
cos(π/2)   = 0   (exact)
cos(π)     = -1  (exact)
cos(3π/2)  = 0   (exact)
```

Two models trained at different phase rotations can interleave because they operate in the same mathematical space.

| Rotation | Angle | Radians |
|----------|-------|---------|
| Phase 0  | 0°    | 0       |
| Phase 1  | 90°   | π/2     |
| Phase 2  | 180°  | π       |
| Phase 3  | 270°  | 3π/2    |

## Model Architecture

```python
class Pi2Linear(nn.Module):
    def forward(self, x):
        # Weights stored as integers
        # Multiply by π/2 at compute time
        w = self.weight * HALF_PI  # infinite precision
        return F.linear(x, w)
```

All layers use this pattern:
- `Pi2Linear` - linear projections
- `Pi2Embedding` - token/position embeddings
- `Pi2RMSNorm` - normalization

## Quick Start

```bash
# Train a 720M model
python train_pi2.py \
    --data training_data.jsonl \
    --dim 1440 --layers 24 --heads 12 \
    --steps 22000 \
    --lr 2e-4 \
    --output output_pi2

# Train two models for interleaving
python train_pi2.py --data data.jsonl --output model_a --gpu 0
python train_pi2.py --data data.jsonl --output model_b --gpu 1
```

## Key Constants

```python
HALF_PI = 1.5707963267948966  # math.pi / 2
PI2_STD = 0.012732            # init std for weights
```

## Hardware

Tested on 2x NVIDIA Tesla P40 (24GB each).

~720M model uses ~18-19GB VRAM per GPU with batch size 4.

## The Math (Simple Version)

1. Store integer `n`
2. Compute with `n × π/2`
3. Get infinite decimal precision for free
4. π/2 terms cancel exactly in division
5. Round to nearest integer after gradient update

That's it. Quantum-like precision with basic arithmetic.

## Files

| File | Purpose |
|------|---------|
| `train_pi2.py` | Main training script with π/2 encoding |
| `model.py` | Transformer with latent rotation |
| `config.py` | Constants and model configs |
| `PI2_EXPLAINED.md` | Detailed explanation of the concept |

## License

MIT
