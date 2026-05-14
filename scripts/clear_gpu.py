#!/usr/bin/env python3
"""Clear CUDA cache and print memory usage."""

import torch

if not torch.cuda.is_available():
    print("CUDA not available")
    raise SystemExit(0)

torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
