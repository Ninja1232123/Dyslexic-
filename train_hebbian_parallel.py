#!/usr/bin/env python3
"""
π/2 Hebbian Training - Pipeline Parallel Version

Splits model across 2 GPUs:
- GPU 0: Layers 0-11 (first half)
- GPU 1: Layers 12-23 (second half)

Supports training on rotation subsets for later interleaving.
"""

import os
import sys
import json
import time
import argparse
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from quantize import quantize_model, count_parameters, Pi2QuantizedLinear


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


class PipelineParallelModel(nn.Module):
    """
    Wraps a Pi2Model for pipeline parallel across 2 GPUs.

    GPU 0: embeddings + first half of layers
    GPU 1: second half of layers + output head
    """

    def __init__(self, model, num_layers=24):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.split_point = num_layers // 2

        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:1")

        # Untie lm_head from token_emb (they share weights by default)
        # For pipeline parallel, they need to be on different devices
        self.model.lm_head.weight = nn.Parameter(self.model.lm_head.weight.clone())

        # Move embeddings, rotator, and first half to GPU 0
        self.model.token_emb = self.model.token_emb.to(self.device0)
        self.model.pos_emb = self.model.pos_emb.to(self.device0)
        self.model.dropout = self.model.dropout.to(self.device0)
        self.model.rotator = self.model.rotator.to(self.device0)

        for i in range(self.split_point):
            self.model.blocks[i] = self.model.blocks[i].to(self.device0)

        # Move second half and output to GPU 1
        for i in range(self.split_point, self.num_layers):
            self.model.blocks[i] = self.model.blocks[i].to(self.device1)

        self.model.ln_f = self.model.ln_f.to(self.device1)
        self.model.lm_head = self.model.lm_head.to(self.device1)

    def forward(self, input_ids, rotation_angle=0.0):
        # GPU 0: embeddings + rotation + first half
        input_ids = input_ids.to(self.device0)
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=self.device0)
        x = self.model.token_emb(input_ids) + self.model.pos_emb(pos_ids)
        x = self.model.dropout(x)

        # Apply rotation in latent space
        x = self.model.rotator(x, rotation_angle)

        for i in range(self.split_point):
            x = self.model.blocks[i](x)

        # Transfer to GPU 1
        x = x.to(self.device1)

        # GPU 1: second half + output
        for i in range(self.split_point, self.num_layers):
            x = self.model.blocks[i](x)

        x = self.model.ln_f(x)
        logits = self.model.lm_head(x)

        return logits, None

    def named_modules(self):
        return self.model.named_modules()


class HebbianTrainer:
    """
    Hebbian trainer using rotation consistency.
    Supports training on subset of rotations.
    """

    def __init__(self, model, lr=0.001, weight_decay=0.01, rotations=None):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.rotations = rotations if rotations else ROTATION_ANGLES

        # Track activations for Hebbian updates
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Keep on GPU for fast correlation computation
                self.activations[name] = {
                    'input': input[0].detach(),
                    'output': output.detach()
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, Pi2QuantizedLinear)):
                module.register_forward_hook(make_hook(name))

    def hebbian_update(self):
        """
        Apply Hebbian updates to all layers.
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name not in self.activations:
                    continue

                act = self.activations[name]
                inp = act['input']  # [batch, seq, in_features]
                out = act['output']  # [batch, seq, out_features]

                # Flatten batch and seq dims
                inp_flat = inp.view(-1, inp.shape[-1])
                out_flat = out.view(-1, out.shape[-1])

                # Get weight tensor
                if isinstance(module, Pi2QuantizedLinear):
                    weight = module.weight.magnitude
                elif isinstance(module, nn.Linear):
                    weight = module.weight
                else:
                    continue

                # Move activations to weight device and compute correlation there
                inp_flat = inp_flat.to(weight.device)
                out_flat = out_flat.to(weight.device)
                correlation = torch.mm(out_flat.t(), inp_flat) / inp_flat.shape[0]

                # Apply update
                weight.data += self.lr * correlation

                # Weight decay
                weight.data *= (1 - self.weight_decay)

                # Normalize to prevent explosion
                norms = weight.data.norm(dim=1, keepdim=True).clamp(min=1e-6)
                weight.data /= norms
                weight.data *= PI2_STD * (weight.shape[1] ** 0.5)

    def train_step(self, batch, device):
        """
        Single training step with rotation consistency.
        """
        self.model.eval()

        input_ids = batch["input_ids"]  # Will be moved to correct device in forward

        # Welford's online algorithm for variance
        mean = None
        M2 = None
        n = 0
        output_device = None

        for angle in self.rotations:
            self.activations.clear()

            with torch.no_grad():
                logits, _ = self.model(input_ids, rotation_angle=angle)

                if output_device is None:
                    output_device = logits.device

                n += 1
                if mean is None:
                    mean = logits.clone()
                    M2 = torch.zeros_like(logits)
                else:
                    delta = logits - mean
                    mean += delta / n
                    delta2 = logits - mean
                    M2 += delta * delta2

                del logits
                torch.cuda.empty_cache()

        variance = (M2 / n).mean()
        consistency_loss = variance

        # Hebbian update
        self.hebbian_update()

        del mean, M2
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
    rotations=None,
    run_name="hebbian"
):
    """Main Hebbian training loop."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log_file = output_dir / f"{run_name}_training_log.jsonl"

    trainer = HebbianTrainer(model, lr=lr, rotations=rotations)

    rotation_str = [f"{r:.2f}" for r in trainer.rotations]

    print(f"\n{'='*60}")
    print(f"Starting π/2 Hebbian Training: {run_name}")
    print(f"{'='*60}")
    print(f"Learning rate: {lr}")
    print(f"Steps: {num_steps}")
    print(f"Rotation angles: {rotation_str}")
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
                f"Samples/s: {samples_per_sec:.1f}",
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

        if step % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"{run_name}_step_{step}.pt"
            # For pipeline parallel, need to gather state dict
            if isinstance(model, PipelineParallelModel):
                state_dict = model.model.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                "step": step,
                "model_state_dict": state_dict,
                "avg_loss": total_loss / step,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path.name}")

    # Final save
    final_path = output_dir / f"{run_name}_final.pt"
    if isinstance(model, PipelineParallelModel):
        state_dict = model.model.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({
        "step": step,
        "model_state_dict": state_dict,
        "avg_loss": total_loss / step,
    }, final_path)

    print(f"\n{'='*60}")
    print(f"Hebbian Training Complete: {run_name}")
    print(f"  Final model: {final_path}")
    print(f"  Steps: {step}")
    print(f"  Final avg consistency loss: {total_loss/step:.4f}")
    print(f"{'='*60}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="π/2 Hebbian Training - Pipeline Parallel")
    parser.add_argument("--data", required=True, help="Path to tokenized data (JSONL)")
    parser.add_argument("--size", default="1.57B", choices=["1.57B", "3.14B"])
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Use π/2 phase quantization")
    parser.add_argument("--rotations", type=str, default="all",
                       choices=["all", "horizontal", "vertical"],
                       help="Which rotations to train on: all, horizontal (0,π), vertical (π/2, 3π/2)")
    parser.add_argument("--name", type=str, default="hebbian", help="Run name for outputs")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("π/2 Hebbian Training - Pipeline Parallel")
    print("="*60)

    # Check GPUs
    device_info = get_device_info()
    if device_info["num_gpus"] < 2:
        print(f"ERROR: Need 2 GPUs for pipeline parallel, found {device_info['num_gpus']}")
        sys.exit(1)

    print(f"GPU 0: {device_info['devices'][0]['name']} ({device_info['devices'][0]['vram_gb']:.1f}GB)")
    print(f"GPU 1: {device_info['devices'][1]['name']} ({device_info['devices'][1]['vram_gb']:.1f}GB)")

    # Select rotations
    if args.rotations == "horizontal":
        rotations = [0.0, math.pi]  # 0, π
        args.name = args.name + "_horizontal"
    elif args.rotations == "vertical":
        rotations = [math.pi/2, 3*math.pi/2]  # π/2, 3π/2
        args.name = args.name + "_vertical"
    else:
        rotations = ROTATION_ANGLES  # All 4

    print(f"\nRotation mode: {args.rotations}")
    print(f"Angles: {[f'{r:.4f}' for r in rotations]}")

    # Model
    model_config = get_model_config(args.size)
    print(f"\nModel: {args.size} ({model_config.hidden_size}h, {model_config.num_layers}L)")

    model = create_model(model_config, gradient_checkpointing=False)

    if args.quantize:
        print("\nApplying π/2 phase quantization...")
        model = quantize_model(model)
        stats = count_parameters(model)
        print(f"  Magnitude params: {stats['magnitude']:,}")
        print(f"  Memory estimate: {stats['memory_mb']['magnitude_fp32']:.1f} MB")

    # Wrap in pipeline parallel
    print(f"\nSplitting model across 2 GPUs...")
    print(f"  GPU 0: Layers 0-{model_config.num_layers//2 - 1}")
    print(f"  GPU 1: Layers {model_config.num_layers//2}-{model_config.num_layers - 1}")

    model = PipelineParallelModel(model, num_layers=model_config.num_layers)

    # Memory check
    torch.cuda.synchronize()
    mem0 = torch.cuda.memory_allocated(0) / 1e9
    mem1 = torch.cuda.memory_allocated(1) / 1e9
    print(f"\nGPU memory: GPU0={mem0:.2f}GB, GPU1={mem1:.2f}GB")

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
        device=None,  # Pipeline parallel handles devices
        num_steps=args.steps,
        lr=args.lr,
        rotations=rotations,
        run_name=args.name
    )


if __name__ == "__main__":
    main()
