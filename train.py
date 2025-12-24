#!/usr/bin/env python3
"""
π/2 Training Script

All operations in π/2 space:
- Weights: normal(0, 0.012732)
- Latent rotation: 0, π/2, π, 3π/2 applied to embeddings
- 6 decimal precision

Designed for 2x NVIDIA P40 (24GB each = 48GB total)
Uses DataParallel for multi-GPU training.

Usage:
    # Tokenize data first
    python tokenize_data.py --input data/ --output data.jsonl

    # Train 1.57B model
    python train.py --data data.jsonl --size 1.57B --epochs 3

    # Train 3.14B model
    python train.py --data data.jsonl --size 3.14B --epochs 3
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import wandb
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist
import functools

from config import (
    TrainingConfig,
    ModelConfig,
    get_model_config,
    PI2_STD,
    HALF_PI,
    ROTATION_ANGLES
)
from model import Pi2Model, Pi2Block, create_model, interleave_models
from dataset import create_dataloader
from quantize import quantize_model, count_parameters


def get_device_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = []
        total_vram = 0

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            total_vram += vram_gb
            devices.append({
                "id": i,
                "name": props.name,
                "vram_gb": vram_gb,
                "compute": f"{props.major}.{props.minor}"
            })

        return {
            "num_gpus": num_gpus,
            "total_vram_gb": total_vram,
            "devices": devices
        }
    else:
        return {"num_gpus": 0, "total_vram_gb": 0, "devices": []}


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
    scaler: GradScaler = None,
    use_fp16: bool = False
) -> dict:
    """
    Single training step with π/2 latent rotation.

    Each batch contains samples with different rotation angles.
    We process each rotation angle separately to apply correct rotation.
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    rotation_angles = batch["rotation_angle"]  # Per-sample rotation angles

    # Get unique rotation angles in this batch
    unique_angles = torch.unique(torch.tensor(rotation_angles))

    total_loss = 0.0
    num_samples = 0

    # Process each rotation angle group separately
    for angle in unique_angles:
        angle_val = angle.item()

        # Find samples with this rotation angle
        mask = torch.tensor([a == angle_val for a in rotation_angles])
        if not mask.any():
            continue

        batch_input = input_ids[mask]
        batch_labels = labels[mask]

        # Forward pass with rotation (with optional FP16)
        with autocast(enabled=use_fp16):
            if isinstance(model, nn.DataParallel):
                logits, loss = model.module(batch_input, rotation_angle=angle_val, labels=batch_labels)
            elif isinstance(model, FSDP):
                logits, loss = model(batch_input, rotation_angle=angle_val, labels=batch_labels)
            else:
                logits, loss = model(batch_input, rotation_angle=angle_val, labels=batch_labels)

        if loss is not None:
            total_loss += loss * batch_input.size(0)
            num_samples += batch_input.size(0)

    # Average loss
    if num_samples > 0:
        avg_loss = total_loss / num_samples
    else:
        avg_loss = torch.tensor(0.0, device=device)

    # Backward pass (with optional FP16 scaling)
    if scaler is not None:
        scaler.scale(avg_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        avg_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    return {
        "loss": avg_loss.item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "num_samples": num_samples
    }


def train(
    model: Pi2Model,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    start_epoch: int = 0,
    use_wandb: bool = False,
    num_batches: int = None,
    scaler: GradScaler = None,
    use_fp16: bool = False
):
    """Main training loop with π/2 latent rotation."""

    # Only rank 0 should log (for distributed training)
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Logging
    log_file = output_dir / "training_log.jsonl"
    best_loss = float('inf')
    global_step = 0

    # Handle streaming datasets that don't have len()
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = num_batches if num_batches else "?"

    print(f"\n{'='*60}")
    print("Starting π/2 Training")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size} (x{torch.cuda.device_count()} GPUs)")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight std: {PI2_STD:.6f} (0.02 / π/2)")
    print(f"Rotation angles: {ROTATION_ANGLES}")
    print(f"Precision: 6 decimal places")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()

            # Training step with rotation handling
            step_result = train_step(model, batch, optimizer, config, device, scaler=scaler, use_fp16=use_fp16)

            scheduler.step()

            # Stats
            batch_loss = step_result["loss"]
            epoch_loss += batch_loss * step_result["num_samples"]
            epoch_samples += step_result["num_samples"]
            batch_time = time.time() - batch_start

            global_step += 1

            # Logging (only on main process for distributed)
            if batch_idx % config.log_every == 0 and is_main_process:
                avg_loss = epoch_loss / max(epoch_samples, 1)
                tokens_per_sec = step_result["num_samples"] * 512 / batch_time  # Approx
                lr = scheduler.get_last_lr()[0]

                # GPU memory
                if torch.cuda.is_available():
                    mem_used = torch.cuda.max_memory_allocated() / (1024**3)
                    mem_str = f"{mem_used:.1f}GB"
                else:
                    mem_str = "CPU"

                print(
                    f"Epoch {epoch+1}/{config.epochs} | "
                    f"Batch {batch_idx+1}/{total_batches} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f} | "
                    f"VRAM: {mem_str}",
                    flush=True
                )

                # Write to log
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "loss": batch_loss,
                    "avg_loss": avg_loss,
                    "grad_norm": step_result["grad_norm"],
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "timestamp": datetime.now().isoformat()
                }
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                    f.flush()

                # W&B logging
                if use_wandb:
                    wandb.log({
                        "train/loss": batch_loss,
                        "train/avg_loss": avg_loss,
                        "train/grad_norm": step_result["grad_norm"],
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/epoch": epoch + 1,
                        "train/vram_gb": mem_used if torch.cuda.is_available() else 0,
                    }, step=global_step)

            # Periodic checkpoint (only on main process)
            if global_step % config.checkpoint_every == 0 and is_main_process:
                ckpt_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, batch_loss, ckpt_path)
                print(f"  → Saved checkpoint: {ckpt_path.name}")

        # Epoch stats
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1} Complete")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Samples: {epoch_samples}")
        print(f"{'─'*60}\n")

        # Epoch checkpoint
        ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_epoch_loss, ckpt_path)
        print(f"  → Saved epoch checkpoint: {ckpt_path.name}")

        # Best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_epoch_loss, best_path)
            print(f"  → New best model! Loss: {best_loss:.4f}")

        # W&B epoch summary
        if use_wandb:
            wandb.log({
                "epoch/loss": avg_epoch_loss,
                "epoch/time_seconds": epoch_time,
                "epoch/samples": epoch_samples,
                "epoch/best_loss": best_loss,
            }, step=global_step)

    # Final save
    final_path = output_dir / "pi2_model.pt"
    save_checkpoint(model, optimizer, scheduler, config.epochs, global_step, avg_epoch_loss, final_path)

    # Save config
    config_path = output_dir / "config.json"
    model_config = model.module.config if isinstance(model, nn.DataParallel) else model.config
    with open(config_path, 'w') as f:
        json.dump({
            "training": vars(config),
            "model": vars(model_config),
            "final_loss": avg_epoch_loss,
            "total_steps": global_step,
            "pi2_std": PI2_STD,
            "half_pi": HALF_PI,
            "rotation_angles": ROTATION_ANGLES,
            "precision_decimals": 6
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"  Final model: {final_path}")
    print(f"  Best model: {checkpoint_dir / 'best_model.pt'}")
    print(f"  Config: {config_path}")
    print(f"  Logs: {log_file}")
    print(f"{'='*60}")

    # Finish W&B run
    if use_wandb:
        wandb.finish()

    return model


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    """Save training checkpoint."""
    # Handle DataParallel/FSDP
    if isinstance(model, (nn.DataParallel, FSDP)):
        model_state = model.module.state_dict()
        model_config = model.module.config
    else:
        model_state = model.state_dict()
        model_config = model.config

    torch.save({
        "model_state_dict": model_state,
        "model_config": vars(model_config),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "pi2_std": PI2_STD,
        "rotation_angles": ROTATION_ANGLES
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    # Handle DataParallel
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def main():
    parser = argparse.ArgumentParser(description="π/2 Training")
    parser.add_argument("--data", required=True, help="Path to tokenized data (JSONL)")
    parser.add_argument("--size", default="1.57B", choices=["1.57B", "3.14B"], help="Model size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--single-gpu", action="store_true", help="Disable multi-GPU")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP instead of DataParallel (shards model across GPUs)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets (saves memory)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves GPU memory)")
    parser.add_argument("--sgd", action="store_true", help="Use SGD instead of AdamW (saves ~13GB GPU memory)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (halves memory usage)")
    parser.add_argument("--quantize", action="store_true", help="Use π/2 phase quantization (major memory savings)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="pi2-training", help="W&B project name")
    parser.add_argument("--wandb-run", type=str, help="W&B run name (optional)")
    args = parser.parse_args()

    # Device setup
    print("\n" + "="*60)
    print("π/2 Training Setup")
    print("="*60)

    device_info = get_device_info()
    if device_info["num_gpus"] > 0:
        print(f"GPUs detected: {device_info['num_gpus']}")
        for dev in device_info["devices"]:
            print(f"  [{dev['id']}] {dev['name']} - {dev['vram_gb']:.1f}GB")
        print(f"Total VRAM: {device_info['total_vram_gb']:.1f}GB")
        device = torch.device("cuda")
    else:
        print("No GPUs detected, using CPU")
        device = torch.device("cpu")

    # Model config
    model_config = get_model_config(args.size)
    print(f"\nModel: {args.size}")
    print(f"  Hidden: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Heads: {model_config.num_heads}")
    print(f"  Weight init std: {model_config.weight_std:.6f}")
    print(f"  Precision: 6 decimal places")

    # Check data file size for auto-enabling optimizations
    data_size_gb = Path(args.data).stat().st_size / (1024**3)

    # Training config
    train_config = TrainingConfig(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        multi_gpu=not args.single_gpu and device_info["num_gpus"] > 1
    )

    # Create model
    print("\nCreating model with π/2-initialized weights...")
    use_grad_ckpt = args.grad_checkpoint or data_size_gb > 1.0  # Auto-enable for large datasets
    if use_grad_ckpt:
        print("Gradient checkpointing ENABLED (saves GPU memory)")
    model = create_model(model_config, gradient_checkpointing=use_grad_ckpt)

    # Apply π/2 phase quantization if requested
    if args.quantize:
        print("\nApplying π/2 phase quantization...")
        model = quantize_model(model)
        stats = count_parameters(model)
        print(f"  Magnitude params: {stats['magnitude']:,}")
        print(f"  Phase params: {stats['phase']:,}")
        print(f"  Memory (magnitudes): {stats['memory_mb']['magnitude_fp32']:.1f} MB")
        print(f"  Memory (phases 2-bit): {stats['memory_mb']['phase_2bit']:.1f} MB")
        print(f"  Estimated optimizer savings: ~70% vs FP32")

    # Multi-GPU
    if train_config.multi_gpu and device_info["num_gpus"] > 1:
        if args.fsdp:
            # FSDP: Initialize distributed if not already done
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            # FSDP wrapping policy - wrap each transformer block
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Pi2Block}
            )

            print(f"\nEnabling FSDP across {device_info['num_gpus']} GPUs (sharded)")
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap_policy,
                device_id=local_rank,
            )
        else:
            model = model.to(device)
            print(f"\nEnabling DataParallel across {device_info['num_gpus']} GPUs")
            model = nn.DataParallel(model)
    else:
        model = model.to(device)

    # Check if main process (for distributed training)
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    # Initialize wandb (only on main process)
    if args.wandb and is_main:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"pi2-{args.size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_size": args.size,
                "hidden_size": model_config.hidden_size,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_heads,
                "epochs": args.epochs,
                "batch_size": args.batch,
                "effective_batch_size": args.batch * max(1, device_info["num_gpus"]),
                "learning_rate": args.lr,
                "weight_std": PI2_STD,
                "rotation_angles": ROTATION_ANGLES,
                "num_gpus": device_info["num_gpus"],
                "total_vram_gb": device_info["total_vram_gb"],
            }
        )
        print(f"\nW&B logging enabled: {wandb.run.url}")

    # Data
    print(f"\nLoading data from: {args.data}")

    # Auto-detect if we need streaming (file > 1GB)
    use_streaming = args.streaming or data_size_gb > 1.0

    if use_streaming:
        print(f"Using STREAMING mode (file is {data_size_gb:.1f}GB)")
        # Count lines for progress tracking
        print("Counting samples (this may take a moment)...")
        with open(args.data, 'r') as f:
            num_samples = sum(1 for _ in f)
        # Each sample expands to 4 rotations
        total_samples = num_samples * 4
        num_batches = total_samples // args.batch
        print(f"Total samples: {num_samples} base x 4 rotations = {total_samples}")

    dataloader = create_dataloader(
        args.data,
        batch_size=args.batch,
        max_seq_len=model_config.max_seq_len,
        shuffle=not use_streaming,  # Can't shuffle streaming
        num_workers=4,
        streaming=use_streaming
    )

    if use_streaming:
        print(f"Batches per epoch: ~{num_batches}")
    else:
        num_batches = len(dataloader)
        print(f"Batches per epoch: {num_batches}")

    effective_batch = args.batch * max(1, device_info["num_gpus"])
    print(f"Effective batch size: {effective_batch}")

    # Optimizer & Scheduler
    if args.sgd:
        print("Using SGD optimizer (no momentum states - saves ~13GB GPU memory)")
        optimizer = SGD(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )

    total_steps = num_batches * train_config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"Total training steps: {total_steps}")

    # FP16 scaler
    scaler = None
    if args.fp16:
        scaler = GradScaler()
        print("FP16 mixed precision ENABLED")

    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resuming from epoch {start_epoch}")

    # Train
    train(model, dataloader, optimizer, scheduler, train_config, device, start_epoch, use_wandb=args.wandb and is_main, num_batches=num_batches, scaler=scaler, use_fp16=args.fp16)


if __name__ == "__main__":
    main()
