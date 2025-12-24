#!/usr/bin/env python3
"""Debug device transfer"""
import torch
torch.cuda.empty_cache()

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# Create test tensor on GPU 0
x = torch.randn(1, 64, 2304, device=device0)
print(f"Original (GPU0): mean={x.mean().item():.6f}, std={x.std().item():.6f}")

# Try different transfer methods
x1 = x.to(device1)
print(f"x.to(device1): mean={x1.mean().item():.6f}, std={x1.std().item():.6f}")

x2 = x.clone().to(device1)
print(f"x.clone().to(device1): mean={x2.mean().item():.6f}, std={x2.std().item():.6f}")

x3 = x.cuda(1)
print(f"x.cuda(1): mean={x3.mean().item():.6f}, std={x3.std().item():.6f}")

# Sync and check
torch.cuda.synchronize()
print(f"After sync - x1: mean={x1.mean().item():.6f}")
print(f"After sync - x2: mean={x2.mean().item():.6f}")
print(f"After sync - x3: mean={x3.mean().item():.6f}")
