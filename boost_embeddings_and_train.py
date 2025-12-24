#!/usr/bin/env python3
"""
Boost embeddings from Phase 3 checkpoint and resume Phase 2 training.

The idea: Phase 3 model may have lost some language learning.
Boost the embedding weights to amplify token distinctions,
then resume Phase 2 training to rebuild language understanding.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_model_config, PI2_STD
from model import create_model
from quantize import quantize_model


def boost_embeddings(model, boost_factor=1.5, token_boost=True, pos_boost=True):
    """
    Boost embedding weights to amplify token/position distinctions.

    Args:
        model: Pi2Model to modify
        boost_factor: Multiplier for embedding weights (1.5 = 50% stronger)
        token_boost: Whether to boost token embeddings
        pos_boost: Whether to boost positional embeddings
    """
    from quantize import Pi2QuantizedEmbedding

    with torch.no_grad():
        if token_boost:
            # Boost token embeddings (handle both quantized and regular)
            if isinstance(model.token_emb, Pi2QuantizedEmbedding):
                old_std = model.token_emb.weight.magnitude.std().item()
                model.token_emb.weight.magnitude.data *= boost_factor
                new_std = model.token_emb.weight.magnitude.std().item()
            else:
                old_std = model.token_emb.weight.std().item()
                model.token_emb.weight.data *= boost_factor
                new_std = model.token_emb.weight.std().item()
            print(f"  Token embeddings: std {old_std:.6f} -> {new_std:.6f}")

        if pos_boost:
            # Boost positional embeddings (handle both quantized and regular)
            if isinstance(model.pos_emb, Pi2QuantizedEmbedding):
                old_std = model.pos_emb.weight.magnitude.std().item()
                model.pos_emb.weight.magnitude.data *= boost_factor
                new_std = model.pos_emb.weight.magnitude.std().item()
            else:
                old_std = model.pos_emb.weight.std().item()
                model.pos_emb.weight.data *= boost_factor
                new_std = model.pos_emb.weight.std().item()
            print(f"  Position embeddings: std {old_std:.6f} -> {new_std:.6f}")

    return model


def boost_early_layers(model, boost_factor=1.2, num_layers=4):
    """
    Boost early transformer layers to help with initial token processing.

    Args:
        model: Pi2Model to modify
        boost_factor: Multiplier for weights
        num_layers: Number of early layers to boost
    """
    from quantize import Pi2QuantizedLinear

    def boost_linear(layer, factor):
        """Boost a linear layer (quantized or regular)."""
        if isinstance(layer, Pi2QuantizedLinear):
            layer.weight.magnitude.data *= factor
        else:
            layer.weight.data *= factor

    with torch.no_grad():
        for i in range(min(num_layers, len(model.blocks))):
            block = model.blocks[i]

            # Boost attention projections
            for proj in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.o_proj]:
                boost_linear(proj, boost_factor)

            # Boost MLP
            boost_linear(block.mlp.fc1, boost_factor)
            boost_linear(block.mlp.fc2, boost_factor)

            print(f"  Layer {i}: boosted by {boost_factor}x")

    return model


def main():
    parser = argparse.ArgumentParser(description="Boost embeddings and train")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", default="1.57B", choices=["1.57B", "3.14B"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-boost", type=float, default=1.5, help="Embedding boost factor")
    parser.add_argument("--layer-boost", type=float, default=1.2, help="Early layer boost factor")
    parser.add_argument("--boost-layers", type=int, default=4, help="Number of early layers to boost")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("\n" + "="*60)
    print("Boosted Embedding Phase 2 Training")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {args.gpu}")

    # Create model
    model_config = get_model_config(args.size)
    print(f"\nModel: {args.size}")

    model = create_model(model_config, gradient_checkpointing=False)

    if args.quantize:
        print("\nApplying π/2 phase quantization...")
        model = quantize_model(model)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"  Loaded from step {checkpoint.get('step', '?')}")
    print(f"  Previous avg loss: {checkpoint.get('avg_loss', 'N/A')}")
    del checkpoint

    # Apply boosts
    print(f"\nBoosting embeddings by {args.embed_boost}x:")
    model = boost_embeddings(model, boost_factor=args.embed_boost)

    print(f"\nBoosting first {args.boost_layers} layers by {args.layer_boost}x:")
    model = boost_early_layers(model, boost_factor=args.layer_boost, num_layers=args.boost_layers)

    model = model.to(device)
    torch.cuda.empty_cache()

    # Memory check
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"\nGPU memory used: {mem:.2f} GB")

    # Now import and run hebbian training
    from dataset import Pi2StreamingDataset
    from config import TrainingConfig
    from train_hebbian import train_hebbian

    print(f"\nLoading data: {args.data}")
    dataset = Pi2StreamingDataset(args.data, max_seq_len=model_config.max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=4,
        pin_memory=True
    )

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
        use_wandb=False
    )


if __name__ == "__main__":
    main()
