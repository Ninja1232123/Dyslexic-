#!/usr/bin/env python3
"""Check GPU peer access"""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Check peer access
print("\nPeer access matrix:")
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        can_access = torch.cuda.can_device_access_peer(i, j)
        print(f"GPU {i} -> GPU {j}: {can_access}")

# Try enabling peer access
print("\nEnabling peer access...")
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            try:
                with torch.cuda.device(i):
                    torch.cuda.set_device(i)
                    # Check if already enabled or try to enable
                    pass
            except:
                pass

# Test simple transfer
print("\nSimple transfer test:")
x = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
print(f"Source (cuda:0): {x}")
y = x.to('cuda:1')
torch.cuda.synchronize()
print(f"After transfer (cuda:1): {y}")
