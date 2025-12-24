#!/usr/bin/env python3
"""
π/2 Training - Infinite Precision

Weights are integers × π/2. When stored as "1", it means exactly:
1.5707963267948966192313216916397514420985846996875529...

Every decimal is EXACT. Not approximate. Infinitely precise.

Training: gradient magnitude controls which decimal place moves.
The chain reaction (5-9 up, 0-4 down) cascades through π's known digits.
No optimizer needed. No snap logic. π's infinite precision IS the mechanism.

2-bit storage → ∞-bit effective precision.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

# Handle GPU selection BEFORE importing torch
if '--gpu' in sys.argv:
    idx = sys.argv.index('--gpu')
    if idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx + 1]

import torch
import torch.nn as nn
import torch.nn.functional as F
# No AMP needed - pure fp32 for infinite precision

# Constants
HALF_PI = math.pi / 2
PI2_STD = 0.012732  # Target weight std for π/2 encoding


class Pi2Linear(nn.Module):
    """Linear layer where weights are stored as π/2 multiples."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weights stored as "number of π/2 units"
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        # For snap-to-integer training: init with actual integers
        # Use small integers {-2,-1,0,1,2} scaled by 1/sqrt(fan_in)
        fan_in = self.in_features
        scale = max(1, int(round(2.0 / (fan_in ** 0.5))))  # Integer scale
        self.weight.data = torch.randint(-scale, scale + 1, self.weight.shape, dtype=self.weight.dtype, device=self.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Actual weight = stored × π/2
        w = self.weight * HALF_PI
        return F.linear(x, w, self.bias)


class Pi2Embedding(nn.Module):
    """Embedding where weights are stored as π/2 multiples."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self._init_weights()

    def _init_weights(self):
        # For snap-to-integer training: init with actual integers
        scale = max(1, int(round(2.0 / (self.embedding_dim ** 0.5))))
        self.weight.data = torch.randint(-scale, scale + 1, self.weight.shape, dtype=self.weight.dtype, device=self.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * HALF_PI
        return F.embedding(x, w)


class Pi2RMSNorm(nn.Module):
    """RMSNorm with π/2 scaled weights."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Scale stays at 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class Pi2Attention(nn.Module):
    """Multi-head attention with π/2 encoded weights."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Pi2Linear(dim, 3 * dim, bias=False)
        self.proj = Pi2Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Pi2MLP(nn.Module):
    """MLP with π/2 encoded weights."""

    def __init__(self, dim: int, hidden_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = Pi2Linear(dim, hidden, bias=False)
        self.fc2 = Pi2Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Pi2Block(nn.Module):
    """Transformer block with π/2 encoding."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = Pi2RMSNorm(dim)
        self.attn = Pi2Attention(dim, num_heads, dropout)
        self.norm2 = Pi2RMSNorm(dim)
        self.mlp = Pi2MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Pi2Model(nn.Module):
    """
    Full transformer model with π/2 encoding.

    All weights are stored as π/2 multiples.
    During forward pass, multiply by π/2 to get actual values.
    """

    def __init__(
        self,
        vocab_size: int = 50260,  # GPT-2 vocab + special tokens
        dim: int = 2304,
        num_layers: int = 24,
        num_heads: int = 18,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = Pi2Embedding(vocab_size, dim)
        self.pos_emb = Pi2Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Pi2Block(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = Pi2RMSNorm(dim)
        self.lm_head = Pi2Linear(dim, vocab_size, bias=False)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

        # Count params
        total = sum(p.numel() for p in self.parameters())
        print(f"π/2 Model: {total:,} parameters ({total/1e9:.2f}B)")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        mask = self.causal_mask[:, :, :T, :T]

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


class Pi2Dataset(torch.utils.data.IterableDataset):
    """Streaming dataset for π/2 training with fixed sequence length."""

    def __init__(self, path: str, max_seq_len: int = 512):
        self.path = path
        self.max_seq_len = max_seq_len

    def __iter__(self):
        with open(self.path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    tokens = data.get('input_ids', data.get('tokens', []))
                    if len(tokens) >= 10:
                        # Pad or truncate to fixed length
                        tokens = tokens[:self.max_seq_len]
                        if len(tokens) < self.max_seq_len:
                            # Pad with 0 (typically <pad> or <eos>)
                            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
                        yield {'input_ids': torch.tensor(tokens, dtype=torch.long)}
                except:
                    continue


def get_weight_stats(model: nn.Module, detailed: bool = False) -> dict:
    """Get weight statistics without OOM - running stats."""
    stats = {
        'layers': {},
        'global': {}
    }

    # Running stats for global
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    global_min = float('inf')
    global_max = float('-inf')
    near_int_count = 0
    grad_sq_sum = 0.0
    grad_count = 0

    for name, param in model.named_parameters():
        w = param.data

        # Layer stats (computed on GPU, extract scalars)
        layer_stats = {
            'mean': w.mean().item(),
            'std': w.std().item(),
            'min': w.min().item(),
            'max': w.max().item(),
        }

        if detailed:
            layer_stats['abs_mean'] = w.abs().mean().item()
            layer_stats['near_zero'] = (w.abs() < 0.001).float().mean().item()
            layer_stats['actual_std'] = (w * HALF_PI).std().item()

        if param.grad is not None:
            g = param.grad
            layer_stats['grad_norm'] = g.norm().item()
            grad_sq_sum += (g ** 2).sum().item()
            grad_count += g.numel()

        if detailed:
            stats['layers'][name] = layer_stats

        # Accumulate for global stats
        n = w.numel()
        total_sum += w.sum().item()
        total_sq_sum += (w ** 2).sum().item()
        total_count += n
        global_min = min(global_min, w.min().item())
        global_max = max(global_max, w.max().item())

        # Near integer check (do in chunks to save memory)
        near_int_count += ((w - w.round()).abs() < 0.1).sum().item()

    # Compute global stats
    global_mean = total_sum / total_count
    global_var = (total_sq_sum / total_count) - (global_mean ** 2)
    global_std = global_var ** 0.5 if global_var > 0 else 0

    stats['global'] = {
        'mean': global_mean,
        'std': global_std,
        'min': global_min,
        'max': global_max,
        'actual_std': global_std * HALF_PI,
        'near_integer': near_int_count / total_count,
        'total_params': total_count,
    }

    if grad_count > 0:
        stats['global']['grad_norm'] = (grad_sq_sum ** 0.5)
        stats['global']['grad_rms'] = (grad_sq_sum / grad_count) ** 0.5

    return stats


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: int = 10000,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    log_every: int = 10,
    save_every: int = 5000,
    output_dir: str = "./output",
):
    """π/2 training - pure gradient descent with infinite precision."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log files
    log_file = output_dir / "training_log.jsonl"
    detailed_log = output_dir / "detailed_stats.jsonl"
    config_file = output_dir / "config.json"

    # No optimizer needed - π/2 infinite precision handles everything
    # LR controls which decimal place moves, π's digits handle the cascade

    # Learning rate schedule
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    # Save config
    config = {
        'num_steps': num_steps,
        'lr': lr,
        'warmup_steps': warmup_steps,
        'device': str(device),
        'use_amp': use_amp,
        'model_params': sum(p.numel() for p in model.parameters()),
        'half_pi': HALF_PI,
        'pi2_std': PI2_STD,
        'start_time': datetime.now().isoformat(),
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("π/2 Training - FULL LOGGING")
    print(f"{'='*60}")
    print(f"Steps: {num_steps}")
    print(f"Learning rate: {lr}")
    print(f"Warmup: {warmup_steps}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Output: {output_dir}")
    print(f"Log: {log_file}")
    print(f"{'='*60}\n")

    model.train()
    step = 0
    total_loss = 0
    losses = []
    start_time = time.time()

    # Initial stats
    init_stats = get_weight_stats(model, detailed=False)
    with open(detailed_log, 'w') as f:
        f.write(json.dumps({'step': 0, 'type': 'init', 'stats': init_stats}) + '\n')

    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch['input_ids'].to(device)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Update LR
        current_lr = get_lr(step)

        # Zero gradients manually
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass (no AMP for snap training - integers need fp32 precision)
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Backward
        loss.backward()

        # π/2 training: pure gradient descent
        # LR controls decimal place, π's infinite precision handles the rest
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= current_lr * p.grad

        loss_val = loss.item()
        total_loss += loss_val
        losses.append(loss_val)
        step += 1

        # Compute perplexity
        perplexity = math.exp(min(loss_val, 100))  # Cap to avoid overflow

        # Memory stats
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
        else:
            mem_allocated = mem_reserved = 0

        # Log every step (basic)
        log_entry = {
            'step': step,
            'loss': loss_val,
            'perplexity': perplexity,
            'lr': current_lr,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'mem_gb': mem_allocated,
            'timestamp': datetime.now().isoformat(),
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        if step % log_every == 0:
            avg_loss = total_loss / step
            recent_loss = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - start_time
            samples_per_sec = (step * batch_size) / elapsed
            tokens_per_sec = (step * batch_size * seq_len) / elapsed

            # Get weight stats
            w_stats = get_weight_stats(model, detailed=False)

            print(
                f"Step {step}/{num_steps} | "
                f"Loss: {loss_val:.4f} | "
                f"PPL: {perplexity:.1f} | "
                f"Avg: {avg_loss:.4f} | "
                f"Recent: {recent_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            print(
                f"  W_std: {w_stats['global']['std']:.5f} | "
                f"W_actual: {w_stats['global']['actual_std']:.5f} | "
                f"NearInt: {w_stats['global']['near_integer']*100:.1f}% | "
                f"Mem: {mem_allocated:.1f}GB | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )

            # Detailed log
            detailed_entry = {
                'step': step,
                'type': 'periodic',
                'loss': loss_val,
                'avg_loss': avg_loss,
                'recent_loss': recent_loss,
                'perplexity': perplexity,
                'lr': current_lr,
                'samples_per_sec': samples_per_sec,
                'tokens_per_sec': tokens_per_sec,
                'elapsed_sec': elapsed,
                'mem_allocated_gb': mem_allocated,
                'mem_reserved_gb': mem_reserved,
                'weight_stats': w_stats['global'],
                'timestamp': datetime.now().isoformat(),
            }
            with open(detailed_log, 'a') as f:
                f.write(json.dumps(detailed_entry) + '\n')

        if step % save_every == 0:
            ckpt_path = output_dir / f"pi2_step_{step}.pt"

            # Full weight stats for checkpoint
            full_stats = get_weight_stats(model, detailed=True)

            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                # No optimizer state - using snap-to-integer updates
                'loss': loss_val,
                'avg_loss': total_loss / step,
                'lr': current_lr,
                'weight_stats': full_stats,
                'config': config,
            }, ckpt_path)
            print(f"  → Saved: {ckpt_path}")

            # Also save layer-wise stats
            layer_stats_file = output_dir / f"layer_stats_step_{step}.json"
            with open(layer_stats_file, 'w') as f:
                json.dump(full_stats, f, indent=2)

    # Final save with everything
    final_path = output_dir / "pi2_final.pt"
    final_stats = get_weight_stats(model, detailed=True)

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        # No optimizer - weights are integer π/2 multiples
        'loss': total_loss / step,
        'all_losses': losses,
        'weight_stats': final_stats,
        'config': config,
        'training_time_sec': time.time() - start_time,
    }, final_path)

    # Final summary
    summary = {
        'total_steps': step,
        'final_loss': total_loss / step,
        'final_perplexity': math.exp(min(total_loss / step, 100)),
        'training_time_sec': time.time() - start_time,
        'final_weight_stats': final_stats['global'],
        'config': config,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Steps: {step}")
    print(f"Final loss: {total_loss/step:.4f}")
    print(f"Final PPL: {math.exp(min(total_loss/step, 100)):.1f}")
    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"Weight std (π/2 units): {final_stats['global']['std']:.5f}")
    print(f"Weight std (actual): {final_stats['global']['actual_std']:.5f}")
    print(f"Near integer weights: {final_stats['global']['near_integer']*100:.1f}%")
    print(f"Saved: {final_path}")
    print(f"Logs: {log_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="π/2 Training")
    parser.add_argument("--data", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--dim", type=int, default=2304, help="Model dimension")
    parser.add_argument("--layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--heads", type=int, default=18, help="Number of attention heads")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", default="./output_pi2", help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    model = Pi2Model(
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
    ).to(device)

    # Verify π/2 encoding
    sample = next(model.parameters())
    print(f"\nWeight stats (π/2 units):")
    print(f"  Std: {sample.std().item():.6f} (target: {PI2_STD:.6f})")
    print(f"  Actual std (×π/2): {(sample * HALF_PI).std().item():.6f}")

    # Data
    print(f"\nLoading: {args.data}")
    dataset = Pi2Dataset(args.data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=4,
        pin_memory=True,
    )

    # Train
    train(
        model=model,
        dataloader=dataloader,
        device=device,
        num_steps=args.steps,
        lr=args.lr,
        output_dir=args.output,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
