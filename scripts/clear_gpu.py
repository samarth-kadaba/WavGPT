#!/usr/bin/env python3
"""Clear GPU memory cache."""

import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    
    print(f"GPU cache cleared!")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
else:
    print("CUDA not available")

