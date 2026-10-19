import torch
import torch.nn.functional as F

W_SPA = 1
W_EXP = 10
W_COL = 5
W_TV = 200

def exposure_loss(y, E=0.6, patch=16):
    """y: enhanced image (B,3,H,W). Returns a scalar."""
    # brightness per pixel        -> (B,1,H,W)
    b = y.mean(dim=1, keepdim=True)  # b is (B, 1, 256, 256): one brightness value per pixel

    # average each patch x patch tile
    p = F.avg_pool2d(b, patch)  # 256 / 16 = 16 tiles across and 16 tiles down
    
    # quared distance from E, averaged into one number
    return ((p-E)**2).mean()


def color_loss(y):
    """y: enhanced image (B,3,H,W). Returns a scalar."""
    # 1. per-image channel means            -> (B,3)
    # 2. split into mR, mG, mB              -> (B,) each
    # 3. d_rg, d_rb, d_gb = squared differences between each pair
    # 4. k = sqrt(d_rg^2 + d_rb^2 + d_gb^2)  -> (B,)
    # 5. return the mean of k over the batch

    # mean over H and W
    m = y.mean(dim=(2,3)) #average over (height and width dimentions) the pixels
    # print("m:", m.shape)
    mR = m[:, 0] # index into m and pick dim 0 (: = first slot = images) and take only column 0
    mG = m[:, 1]
    mB = m[:, 2]
    # print("mR:", mR.shape, "mG:", mG.shape, "mB:", mB.shape)

    d_rg = (mR - mG) ** 2
    d_rb = (mR - mB) ** 2
    d_gb = (mG - mB) ** 2

    k = torch.sqrt(d_rg ** 2 + d_rb ** 2 + d_gb ** 2)
    # print("k:", k.shape)

    return k.mean()
    # print("loss:", loss)

def tv_loss(A):
    """
    Function measures how much the curve maps change from one pixel to its neighbour. 
    Smooth maps score low.
    """
    # get the sizes 
    B, _, H, W = A.shape
    # vertical differences 
    dh = A[:, :, 1:, :] - A[:, :, :-1, :] # 1: on the rows slot drops the first row; :-1 drops the last. Subtracting the two gives “each row minus the row above it” = offset 1 subtract
    h_tv = ((dh)**2).sum() #square it and add everything up into one number
    count_h = (H - 1) * W

    # horizontal differences
    dw = A[:, :, :, 1:] - A[:, :, :, :-1] 
    w_tv = ((dw)**2).sum()
    count_w = H * (W - 1)

    return 2 * (h_tv / count_h + w_tv / count_w) / B

def spatial_loss(x, y):
    """
    whether each region's brightness relative to its neighbours is 
    the same before (x) and after (y) enhancement.
    """
    # brightness per pixel        -> (B,1,H,W)
    a = x.mean(dim=1, keepdim=True)
    b = y.mean(dim=1, keepdim=True)

    # Shrink to 4×4 regions
    X = F.avg_pool2d(a, 4)
    Y = F.avg_pool2d(b, 4)

    #Pad both with ring of zeros
    Xp = F.pad(X, (1, 1, 1, 1)) # shape is (B, 1, 66, 66)
    Yp = F.pad(Y, (1, 1, 1, 1)) # what he said ^

    # the centre windows
    cX = Xp[:, :, 1:-1, 1:-1] # 1:-1 drops the first and last row and column, which removes the zero ring, so cX equals X again
    cY = Yp[:, :, 1:-1, 1:-1]

    # left
    lX = Xp[:, :, 1:-1, 0:-2]
    lY = Yp[:, :, 1:-1, 0:-2]
    D_left = ((cX - lX) - (cY - lY)) ** 2
    
    # right
    rX = Xp[:, :, 1:-1, 2:]
    rY = Yp[:, :, 1:-1, 2:]
    D_right = ((cX - rX) - (cY - rY)) ** 2

    # up
    uX = Xp[:, :, 0:-2, 1:-1]
    uY = Yp[:, :, 0:-2, 1:-1]
    D_up = ((cX - uX) - (cY - uY)) ** 2

    # down
    dX = Xp[:, :, 2:, 1:-1]   
    dY = Yp[:, :, 2:, 1:-1]
    D_down = ((cX - dX) - (cY - dY)) ** 2

    return (D_left + D_right + D_up + D_down).mean()

def zero_dce_loss(x, y, A):
    """combines the four losses"""
    parts = {
        "spa": W_SPA * spatial_loss(x, y),
        "exp": W_EXP * exposure_loss(y), 
        "col": W_COL * color_loss(y),
        "tv":  W_TV * tv_loss(A)
    }

    total = sum(parts.values())

    return total, {k: v.item() for k, v in parts.items()}
    # That last part is a dict comprehension: 
    # “for every name k and value v in parts, store v.item() under k”. 
    # It’s a one-line loop that builds a new dict.

