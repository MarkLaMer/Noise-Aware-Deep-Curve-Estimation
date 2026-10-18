import torch
from nadce.model import DCENet

torch.manual_seed(0)
net = DCENet()

# 1. Output shape and range
x = torch.rand(2, 3, 256, 256)
A = net(x)
assert A.shape == (2, 24, 256, 256), f"wrong shape {A.shape}"
assert A.min() >= -1 and A.max() <= 1, "A must be in [-1, 1] (tanh)"

# 2. Parameter count matches the paper (~79K)
n = sum(p.numel() for p in net.parameters())
assert n == 79416, f"expected 79416 params, got {n}: check in/out channels per layer"

# 3. Names match the official checkpoint
names = sorted({k.split(".")[0] for k in net.state_dict()})
assert names == [f"e_conv{i}" for i in range(1, 8)], f"layer names: {names}"

# 4. Works on full test resolution (fully convolutional)
A_full = net(torch.rand(1, 3, 400, 600))
assert A_full.shape == (1, 24, 400, 600)

# 5. Gradients reach the first layer
A.mean().backward()
assert net.e_conv1.weight.grad is not None and net.e_conv1.weight.grad.abs().sum() > 0

# 6. Init looks right
std = net.e_conv2.weight.std().item()
assert 0.015 < std < 0.025, f"e_conv2 weight std {std:.4f}, expected ~0.02"

print("model OK:", n, "params")
