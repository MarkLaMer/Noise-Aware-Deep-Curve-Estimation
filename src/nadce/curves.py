import torch

def apply_curves(x, A):
    """x: (B,3,H,W) in [0,1]. A: (B,24,H,W) in [-1,1].
    Returns the enhanced image (B,3,H,W) after 8 iterations."""

    maps = torch.split(A, 3, dim=1) # split A into 8 chunks of 3 channels

    for i in range(8):
        x = x + maps[i] * x * (1-x)

    return x