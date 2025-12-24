#!/usr/bin/env python3
"""Quick test to verify rotations produce different outputs"""
import torch
from config import get_model_config, ROTATION_ANGLES
from model import create_model

print("Testing rotation variance...")

model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)
model = model.cuda()
model.eval()

# Test input
input_ids = torch.randint(0, 50000, (1, 64)).cuda()

outputs = []
with torch.no_grad():
    for angle in ROTATION_ANGLES:
        logits, _ = model(input_ids, rotation_angle=angle)
        outputs.append(logits.clone())
        print(f"Angle {angle:.4f}: logits mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")

# Check if outputs are different
stacked = torch.stack(outputs, dim=0)
variance = stacked.var(dim=0).mean()
print(f"\nVariance across rotations: {variance.item():.6f}")

# Check pairwise differences
for i in range(len(outputs)):
    for j in range(i+1, len(outputs)):
        diff = (outputs[i] - outputs[j]).abs().mean().item()
        print(f"Diff between angle {ROTATION_ANGLES[i]:.2f} and {ROTATION_ANGLES[j]:.2f}: {diff:.6f}")
