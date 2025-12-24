#!/usr/bin/env python3
"""Debug device transfer"""
import torch
from config import get_model_config
from model import create_model
from train_hebbian_parallel import PipelineParallelModel

model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)
model = PipelineParallelModel(model, num_layers=24)

# Create test tensor on GPU 0
x = torch.randn(1, 64, 2304, device=model.device0)
print(f"Before transfer (GPU0): mean={x.mean().item():.6f}, std={x.std().item():.6f}")

# Transfer to GPU 1
x_gpu1 = x.to(model.device1)
print(f"After transfer (GPU1): mean={x_gpu1.mean().item():.6f}, std={x_gpu1.std().item():.6f}")

# Run through block 12
y = model.model.blocks[12](x_gpu1)
print(f"After block 12: mean={y.mean().item():.6f}, std={y.std().item():.6f}")
