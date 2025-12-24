#!/usr/bin/env python3
"""Test transfer after model setup"""
import torch
from config import get_model_config
from model import create_model

print("Before model creation:")
x = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
y = x.to('cuda:1')
torch.cuda.synchronize()
print(f"Transfer works: {y.tolist()}")

print("\nCreating model...")
model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)

# Don't split yet, just put on GPU0
model = model.to('cuda:0')

print("After model to cuda:0:")
x2 = torch.tensor([4.0, 5.0, 6.0], device='cuda:0')
y2 = x2.to('cuda:1')
torch.cuda.synchronize()
print(f"Transfer works: {y2.tolist()}")

# Now move half to GPU1
print("\nMoving some blocks to cuda:1...")
model.blocks[12] = model.blocks[12].to('cuda:1')

print("After moving block 12:")
x3 = torch.tensor([7.0, 8.0, 9.0], device='cuda:0')
y3 = x3.to('cuda:1')
torch.cuda.synchronize()
print(f"Transfer works: {y3.tolist()}")

# Big tensor
x4 = torch.randn(1, 64, 2304, device='cuda:0')
print(f"\nBig tensor on cuda:0: mean={x4.mean().item():.4f}")
y4 = x4.to('cuda:1')
torch.cuda.synchronize()
print(f"Big tensor on cuda:1: mean={y4.mean().item():.4f}")
