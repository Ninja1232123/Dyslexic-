#!/usr/bin/env python3
"""Test PipelineParallelModel without quantization"""
import torch
from config import get_model_config, ROTATION_ANGLES
from model import create_model
from train_hebbian_parallel import PipelineParallelModel

print("Testing rotation variance with PipelineParallelModel (NO quantization)...")

model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)
# NO quantization
model = PipelineParallelModel(model, num_layers=24)

# Test input
input_ids = torch.randint(0, 50000, (1, 64))

outputs = []
with torch.no_grad():
    for angle in ROTATION_ANGLES:
        logits, _ = model(input_ids, rotation_angle=angle)
        outputs.append(logits.cpu().clone())
        print(f"Angle {angle:.4f}: logits mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")

stacked = torch.stack(outputs, dim=0)
variance = stacked.var(dim=0).mean()
print(f"\nVariance across rotations: {variance.item():.6f}")
