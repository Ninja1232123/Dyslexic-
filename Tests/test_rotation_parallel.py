#!/usr/bin/env python3
"""Quick test to verify rotations work with PipelineParallelModel"""
import torch
from config import get_model_config, ROTATION_ANGLES
from model import create_model
from quantize import quantize_model
from train_hebbian_parallel import PipelineParallelModel

print("Testing rotation variance with PipelineParallelModel...")

model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)
model = quantize_model(model)
model = PipelineParallelModel(model, num_layers=24)

# Test input
input_ids = torch.randint(0, 50000, (1, 64))  # Don't move to cuda, let the model handle it

outputs = []
with torch.no_grad():
    for angle in ROTATION_ANGLES:
        logits, _ = model(input_ids, rotation_angle=angle)
        outputs.append(logits.cpu().clone())
        print(f"Angle {angle:.4f}: logits mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")

# Check if outputs are different
stacked = torch.stack(outputs, dim=0)
variance = stacked.var(dim=0).mean()
print(f"\nVariance across rotations: {variance.item():.6f}")
