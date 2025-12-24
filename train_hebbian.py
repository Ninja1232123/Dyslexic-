#!/usr/bin/env python3
"""
π/2 Hebbian Training

Uses rotation consistency as the learning signal.
No backpropagation. No optimizer states. Just local updates.

The idea:
- Same content at 4 rotations should produce consistent outputs
- Strengthen connections that are consistent across rotations
- Weaken connections that are inconsistent

Memory: Only need weights (~6.7GB for 1.68B model)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from config import (
    TrainingConfig,
    ModelConfig,
    get_model_config,
    PI2_STD,
    HALF_PI,
    ROTATION_ANGLES
)
from model import Pi2Model, create_model
from dataset import Pi2StreamingDataset
from quantize import quantize_model, count_parameters, Pi2Linear, Pi2Embedding, HALF_PI


def get_device_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        total_vram = 0
        devices = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            total_vram += vram_gb
            devices.append({"id": i, "name": props.name, "vram_gb": vram_gb})
        return {"num_gpus": num_gpus, "total_vram_gb": total_vram, "devices": devices}
    return {"num_gpus": 0, "total_vram_gb": 0, "devices": []}


class HebbianTrainer:
    """
    Hebbian trainer using rotation consistency.

    For each batch:
    1. Forward pass at all 4 rotations
    2. Measure consistency (variance across rotations)
    3. Update weights based on activation correlations
    """

    def __init__(self, model, lr=0.001, weight_decay=0.01, embed_lr_mult=2.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.embed_lr_mult = embed_lr_mult  # Embeddings learn faster

        # Track activations for Hebbian updates
        self.activations = {}
        self.input_ids = None  # Store input tokens for embedding update
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = {
                    'input': input[0].detach(),
                    'output': output.detach()
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, Pi2Linear)):
                module.register_forward_hook(make_hook(name))

    def compute_consistency_loss(self, outputs_by_rotation):
        """
        Measure how consistent outputs are across rotations.
        Lower = more consistent = better.
        """
        # Stack outputs: [4, batch, seq, vocab]
        stacked = torch.stack(outputs_by_rotation, dim=0)

        # Variance across rotations (dim 0)
        variance = stacked.var(dim=0).mean()

        return variance

    def hebbian_update_embeddings(self, input_ids, hidden_states):
        """
        Apply Hebbian updates to embedding layers.

        For embeddings, we strengthen the connection between tokens
        and their resulting hidden states based on context.
        """
        with torch.no_grad():
            # Get embedding modules
            token_emb = self.model.token_emb
            pos_emb = self.model.pos_emb

            batch, seq_len = input_ids.shape
            embed_lr = self.lr * self.embed_lr_mult

            # Token embeddings: strengthen embeddings for tokens that produced useful hidden states
            # Use hidden state magnitude as learning signal
            hidden_magnitude = hidden_states.norm(dim=-1, keepdim=True)  # [batch, seq, 1]
            hidden_magnitude = hidden_magnitude / hidden_magnitude.mean()  # Normalize

            # Get embedding weight tensor
            if isinstance(token_emb, Pi2Embedding):
                weight = token_emb.weight.weight  # Pi2Weight stores in .weight
            else:
                weight = token_emb.weight

            # For each unique token, accumulate gradient
            for b in range(batch):
                for s in range(seq_len):
                    tok_id = input_ids[b, s].item()
                    signal = hidden_magnitude[b, s, 0].item()

                    # Strengthen embedding proportional to hidden state magnitude
                    # This is a simplified Hebbian rule for discrete inputs
                    weight.data[tok_id] *= (1 + embed_lr * (signal - 1))
            norms = weight.data.norm(dim=1, keepdim=True).clamp(min=1e-6)
            weight.data /= norms
            weight.data *= PI2_STD * (weight.shape[1] ** 0.5)

            # Position embeddings: similar update
            if isinstance(pos_emb, Pi2Embedding):
                pos_weight = pos_emb.weight.weight
            else:
                pos_weight = pos_emb.weight

            # Average hidden magnitude per position
            pos_signal = hidden_magnitude.mean(dim=0)  # [seq, 1]
            for s in range(min(seq_len, pos_weight.shape[0])):
                signal = pos_signal[s, 0].item()
                pos_weight.data[s] *= (1 + embed_lr * (signal - 1))

            # Normalize position embeddings
            norms = pos_weight.data.norm(dim=1, keepdim=True).clamp(min=1e-6)
            pos_weight.data /= norms
            pos_weight.data *= PI2_STD * (pos_weight.shape[1] ** 0.5)

    def hebbian_update(self):
        """
        Apply Hebbian updates to all layers.

        Rule: Δw ∝ correlation(input, output)
        With normalization to prevent explosion.
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name not in self.activations:
                    continue

                act = self.activations[name]
                inp = act['input']  # [batch, seq, in_features]
                out = act['output']  # [batch, seq, out_features]

                # Flatten batch and seq dims
                inp_flat = inp.view(-1, inp.shape[-1])  # [N, in]
                out_flat = out.view(-1, out.shape[-1])  # [N, out]

                # Compute correlation (outer product averaged over samples)
                # This is the Hebbian update: Δw = η * out^T @ inp
                correlation = torch.mm(out_flat.t(), inp_flat) / inp_flat.shape[0]

                # Get weight tensor
                if isinstance(module, Pi2Linear):
                    weight = module.weight.weight  # Pi2Weight stores in .weight
                elif isinstance(module, nn.Linear):
                    weight = module.weight
                else:
                    continue

                # Oja's rule variant: Δw = η * (correlation - w * diag(out^T @ out))
                # Simpler: just normalize after update

                # Apply update
                weight.data += self.lr * correlation

                # Weight decay
                weight.data *= (1 - self.weight_decay)

                # Normalize to prevent explosion (per-row normalization)
                norms = weight.data.norm(dim=1, keepdim=True).clamp(min=1e-6)
                weight.data /= norms
                weight.data *= PI2_STD * (weight.shape[1] ** 0.5)  # Scale to π/2 std

    def train_step(self, batch, device):
        """
        Single training step with rotation consistency.
        Memory-efficient: compute running mean/variance instead of storing all outputs.
        """
        self.model.eval()  # No dropout during Hebbian training

        input_ids = batch["input_ids"].to(device)

        # Welford's online algorithm for variance
        # Compute mean and M2 (sum of squared differences) incrementally
        mean = None
        M2 = None
        n = 0

        for angle in ROTATION_ANGLES:
            self.activations.clear()  # Clear BEFORE each forward pass

            with torch.no_grad():
                logits, _ = self.model(input_ids, rotation_angle=angle)

                # Update running mean/variance (Welford's algorithm)
                n += 1
                if mean is None:
                    mean = logits.clone()
                    M2 = torch.zeros_like(logits)
                else:
                    delta = logits - mean
                    mean += delta / n
                    delta2 = logits - mean
                    M2 += delta * delta2

                # Free the logits immediately
                del logits
                torch.cuda.empty_cache()

        # Variance = M2 / n
        variance = (M2 / n).mean()
        consistency_loss = variance

        # Get hidden states for embedding update (from last forward pass activations)
        # Use the first block's input as proxy for post-embedding hidden states
        hidden_states = None
        for name in self.activations:
            if 'blocks.0' in name:
                hidden_states = self.activations[name]['input']
                break

        # Hebbian update for linear layers
        self.hebbian_update()

        # Hebbian update for embeddings
        if hidden_states is not None:
            self.hebbian_update_embeddings(input_ids, hidden_states)

        # Clean up
        del mean, M2, hidden_states
        torch.cuda.empty_cache()

        return {
            "consistency_loss": consistency_loss.item(),
            "num_samples": input_ids.shape[0]
        }


def train_hebbian(
    model,
    dataloader,
    config,
    device,
    num_steps=10000,
    lr=0.001,
    log_every=10,
    checkpoint_every=500,
    use_wandb=False
):
    """Main Hebbian training loop."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log_file = output_dir / "hebbian_training_log.jsonl"

    trainer = HebbianTrainer(model, lr=lr)

    print(f"\n{'='*60}")
    print("Starting π/2 Hebbian Training")
    print(f"{'='*60}")
    print(f"Learning rate: {lr}")
    print(f"Steps: {num_steps}")
    print(f"Rotation angles: {ROTATION_ANGLES}")
    print(f"{'='*60}\n")

    step = 0
    total_loss = 0.0
    start_time = time.time()

    for batch in dataloader:
        if step >= num_steps:
            break

        batch_start = time.time()
        result = trainer.train_step(batch, device)
        batch_time = time.time() - batch_start

        total_loss += result["consistency_loss"]
        step += 1

        if step % log_every == 0:
            avg_loss = total_loss / step
            elapsed = time.time() - start_time
            samples_per_sec = (step * result["num_samples"]) / elapsed

            print(
                f"Step {step}/{num_steps} | "
                f"Consistency: {result['consistency_loss']:.4f} | "
                f"Avg: {avg_loss:.4f} | "
                f"Samples/s: {samples_per_sec:.0f}",
                flush=True
            )

            log_entry = {
                "step": step,
                "consistency_loss": result["consistency_loss"],
                "avg_loss": avg_loss,
                "samples_per_sec": samples_per_sec,
                "timestamp": datetime.now().isoformat()
            }
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            if use_wandb:
                wandb.log({
                    "train/consistency_loss": result["consistency_loss"],
                    "train/avg_loss": avg_loss,
                    "train/samples_per_sec": samples_per_sec,
                }, step=step)

        if step % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"hebbian_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "avg_loss": total_loss / step,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path.name}")

    # Final save
    final_path = output_dir / "hebbian_model.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "avg_loss": total_loss / step,
    }, final_path)

    print(f"\n{'='*60}")
    print("Hebbian Training Complete!")
    print(f"  Final model: {final_path}")
    print(f"  Steps: {step}")
    print(f"  Final avg consistency loss: {total_loss/step:.4f}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="π/2 Hebbian Training")
    parser.add_argument("--data", required=True, help="Path to tokenized data (JSONL)")
    parser.add_argument("--size", default="1.57B", choices=["1.57B", "3.14B"])
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Use π/2 phase quantization")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--gpu", type=int, default=None, help="GPU to use (0 or 1)")
    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("\n" + "="*60)
    print("π/2 Hebbian Training Setup")
    print("="*60)

    # Device
    device_info = get_device_info()
    if device_info["num_gpus"] > 0:
        print(f"GPU: {device_info['devices'][0]['name']} ({device_info['devices'][0]['vram_gb']:.1f}GB)")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # Model
    model_config = get_model_config(args.size)
    print(f"\nModel: {args.size} ({model_config.hidden_size}h, {model_config.num_layers}L)")

    model = create_model(model_config, gradient_checkpointing=False)  # No need for checkpointing

    if args.quantize:
        print("\nApplying π/2 integer encoding...")
        model = quantize_model(model)
        stats = count_parameters(model)
        print(f"  π/2 encoded params: {stats['pi2_weights']:,}")
        print(f"  Memory estimate: {stats['memory_mb']['weights_fp32']:.1f} MB")

    model = model.to(device)

    # Load checkpoint if resuming
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')  # Load to CPU first
        # strict=False allows loading old checkpoints missing new buffers like _phase_signs
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded checkpoint from step {checkpoint.get('step', '?')}")
        print(f"  Previous avg loss: {checkpoint.get('avg_loss', '?'):.4f}")
        del checkpoint  # Free checkpoint memory
        torch.cuda.empty_cache()

    # Memory check
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"\nGPU memory used: {mem:.2f} GB")

    # W&B
    if args.wandb:
        wandb.init(
            project="pi2-hebbian",
            name=f"hebbian-{args.size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
        print(f"\nW&B: {wandb.run.url}")

    # Data
    print(f"\nLoading data: {args.data}")
    dataset = Pi2StreamingDataset(args.data, max_seq_len=model_config.max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=4,
        pin_memory=True
    )

    # Training config
    train_config = TrainingConfig(
        data_path=args.data,
        output_dir=args.output,
        batch_size=args.batch,
    )

    # Train
    train_hebbian(
        model=model,
        dataloader=dataloader,
        config=train_config,
        device=device,
        num_steps=args.steps,
        lr=args.lr,
        use_wandb=args.wandb
    )


if __name__ == "__main__":
    main()
