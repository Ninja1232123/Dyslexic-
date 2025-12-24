#!/usr/bin/env python3
"""Debug block 12"""
import torch
from config import get_model_config
from model import create_model
from train_hebbian_parallel import PipelineParallelModel

model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)
model = PipelineParallelModel(model, num_layers=24)

# Check block 12 weights
block12 = model.model.blocks[12]
print(f"Block 12 device: {next(block12.parameters()).device}")
print(f"Block 12 attn.q_proj weight std: {block12.attn.q_proj.weight.std().item():.6f}")
print(f"Block 12 mlp.fc1 weight std: {block12.mlp.fc1.weight.std().item():.6f}")

# Test block 12 directly
x = torch.randn(1, 64, 2304, device=model.device1)
print(f"\nInput to block 12: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
y = block12(x)
print(f"Output from block 12: mean={y.mean().item():.6f}, std={y.std().item():.6f}")
