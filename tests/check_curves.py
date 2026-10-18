import torch
from nadce.curves import apply_curves

torch.manual_seed(0)
x = torch.rand(2, 3, 32, 32)

# 1. A = 0 is the identity
assert torch.allclose(apply_curves(x, torch.zeros(2, 24, 32, 32)), x)

# 2. Positive A brightens, negative A darkens
up = apply_curves(x, torch.full((2, 24, 32, 32), 0.5))
down = apply_curves(x, torch.full((2, 24, 32, 32), -0.5))
assert up.mean() > x.mean() > down.mean(), "sign is flipped"

# 3. Output stays in [0, 1] for extreme A
for v in (-1.0, 1.0):
    y = apply_curves(x, torch.full((2, 24, 32, 32), v))
    assert y.min() >= 0 and y.max() <= 1, f"left [0,1] with A={v}"

# 4. Endpoints are fixed
ends = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2).expand(1, 3, 1, 2)
y = apply_curves(ends, torch.rand(1, 24, 1, 2) * 2 - 1)
assert torch.allclose(y, ends)

# 5. Hand-computed single pixel: x=0.2, A=1 for all 8 steps
v = 0.2
for _ in range(8):
    v = v + 1.0 * v * (1 - v)
y = apply_curves(torch.full((1, 3, 1, 1), 0.2), torch.ones(1, 24, 1, 1))
assert abs(y[0, 0, 0, 0].item() - v) < 1e-6, f"got {y.flatten()[0]:.6f}, expected {v:.6f}"

# 6. Each iteration uses its own map: changing only the LAST map changes the output
A = torch.zeros(1, 24, 4, 4)
A2 = A.clone(); A2[:, 21:24] = 1.0
x1 = torch.full((1, 3, 4, 4), 0.3)
assert not torch.allclose(apply_curves(x1, A), apply_curves(x1, A2)), "last map unused"

# 7. Gradients flow to A
A = torch.zeros(1, 24, 4, 4, requires_grad=True)
apply_curves(x1, A).mean().backward()
assert A.grad is not None and A.grad.abs().sum() > 0

print("curves OK")
