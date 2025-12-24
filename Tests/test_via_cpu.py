#!/usr/bin/env python3
"""Test transfer via CPU"""
import torch

x = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
print(f"Source (cuda:0): {x}")

# Via CPU
y = x.cpu().to('cuda:1')
torch.cuda.synchronize()
print(f"Via CPU (cuda:1): {y}")

# Direct
z = x.to('cuda:1')
torch.cuda.synchronize()
print(f"Direct (cuda:1): {z}")
