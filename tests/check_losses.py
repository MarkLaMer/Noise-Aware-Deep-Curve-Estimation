import torch
from nadce.losses import exposure_loss, color_loss, spatial_loss, tv_loss, zero_dce_loss

def close(a, b, tol=1e-5):
    return abs(float(a) - b) < tol

# Exposure: constant 0.6 image -> 0; constant 0.1 -> (0.1-0.6)^2 = 0.25
assert close(exposure_loss(torch.full((2, 3, 64, 64), 0.6)), 0.0)
assert close(exposure_loss(torch.full((2, 3, 64, 64), 0.1)), 0.25)

# Colour: gray image -> 0; R=0.6, G=B=0.4 -> sqrt(0.04^2 + 0.04^2 + 0) = 0.0565685
gray = torch.full((1, 3, 32, 32), 0.3)
assert close(color_loss(gray), 0.0)
tinted = torch.full((1, 3, 32, 32), 0.4); tinted[:, 0] = 0.6
assert close(color_loss(tinted), 0.0565685), float(color_loss(tinted))

# Spatial: identical input and output -> 0
x = torch.rand(2, 3, 64, 64)
assert close(spatial_loss(x, x), 0.0)
assert spatial_loss(x, x * 0.5) > 0
# Border behaviour: input all 0, output all 0.5 on 8x8 -> pooled 2x2 of 0.5.
# Each pooled pixel has 2 neighbours outside the border (diff 0.5) and 2 inside (diff 0),
# so E = 2 * 0.5^2 = 0.5 at every pixel and the mean is 0.5.
assert close(spatial_loss(torch.zeros(1, 3, 8, 8), torch.full((1, 3, 8, 8), 0.5)), 0.5), \
    float(spatial_loss(torch.zeros(1, 3, 8, 8), torch.full((1, 3, 8, 8), 0.5)))

# TV: constant A -> 0; [[0,1],[0,1]] -> 2 * (0/2 + 2/2) / 1 = 2.0
assert close(tv_loss(torch.full((2, 24, 16, 16), 0.3)), 0.0)
A = torch.tensor([[0., 1.], [0., 1.]]).view(1, 1, 2, 2)
assert close(tv_loss(A), 2.0), float(tv_loss(A))
# Batch normalisation: duplicating the batch must not change the value
assert close(tv_loss(A.repeat(3, 1, 1, 1)), 2.0)

# Total: scalar, differentiable, and the logged parts add up
x = torch.rand(2, 3, 64, 64) * 0.2
A = torch.full((2, 24, 64, 64), 0.1).requires_grad_()   # smooth, like A early in training
y = x + 0.1 * A[:, :3]          # any differentiable stand-in for apply_curves
total, parts = zero_dce_loss(x, y, A)
assert total.dim() == 0
total.backward()
assert A.grad is not None
assert close(sum(parts.values()), total.item(), 1e-4), parts
print("losses OK", {k: round(v, 4) for k, v in parts.items()})
