#!/usr/bin/env python3
"""Debug rotation issue"""
import torch
from config import get_model_config, ROTATION_ANGLES
from model import create_model
from train_hebbian_parallel import PipelineParallelModel

print("Debugging rotation...")

model_config = get_model_config("1.57B")
model = create_model(model_config, gradient_checkpointing=False)
model = PipelineParallelModel(model, num_layers=24)

input_ids = torch.randint(0, 50000, (1, 64))

with torch.no_grad():
    # Check embeddings
    input_ids_gpu = input_ids.to(model.device0)
    seq_len = input_ids.size(1)
    pos_ids = torch.arange(seq_len, device=model.device0)
    
    x = model.model.token_emb(input_ids_gpu) + model.model.pos_emb(pos_ids)
    print(f"After embedding: mean={x.mean().item():.6f}, std={x.std().item():.6f}, any_nan={x.isnan().any().item()}")
    
    x = model.model.dropout(x)
    print(f"After dropout: mean={x.mean().item():.6f}, std={x.std().item():.6f}, any_nan={x.isnan().any().item()}")
    
    # Test rotation at angle 0
    x0 = model.model.rotator(x.clone(), 0.0)
    print(f"After rotation(0): mean={x0.mean().item():.6f}, std={x0.std().item():.6f}, any_nan={x0.isnan().any().item()}")
    
    # Test rotation at angle pi/2
    x1 = model.model.rotator(x.clone(), ROTATION_ANGLES[1])
    print(f"After rotation(π/2): mean={x1.mean().item():.6f}, std={x1.std().item():.6f}, any_nan={x1.isnan().any().item()}")
    
    # Continue through blocks for angle 0
    for i in range(12):
        x0 = model.model.blocks[i](x0)
        if i == 0 or i == 11:
            print(f"After block {i}: mean={x0.mean().item():.6f}, std={x0.std().item():.6f}, any_nan={x0.isnan().any().item()}")
    
    x0 = x0.to(model.device1)
    for i in range(12, 24):
        x0 = model.model.blocks[i](x0)
        if i == 12 or i == 23:
            print(f"After block {i}: mean={x0.mean().item():.6f}, std={x0.std().item():.6f}, any_nan={x0.isnan().any().item()}")
    
    x0 = model.model.ln_f(x0)
    print(f"After ln_f: mean={x0.mean().item():.6f}, std={x0.std().item():.6f}, any_nan={x0.isnan().any().item()}")
    
    logits = model.model.lm_head(x0)
    print(f"Final logits: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}, any_nan={logits.isnan().any().item()}")
