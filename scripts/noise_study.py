import os
from glob import glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt

P = 20   # patch size: 400x600 splits exactly into 20x30 patches

def lum(path):
    return (np.array(Image.open(path).convert("RGB")).astype(float)/255).mean(axis=2)

def residual(img):
    return img - gaussian_filter(img, sigma=2) # removes the smooth trend => what's left is noise

def patch_std(img2d):
    """
    Tiles the image into 20x20 patches and returns each patch's standard deviation (600 values). 
    Used on residual images, it gives the noise level of each patch.
    """
    return view_as_blocks(img2d, (P, P)).std(axis=(2, 3)).ravel()
    # 3. view_as_blocks(img2d, (P, P)) -> shape (20, 30, P, P); return .std(axis=(2, 3)).ravel()   => 600 values

def patch_mean(img2d):
    """
    Same tiling as patch_std, but returns each patch's average. 
    Used for patch brightness (how dark the input is) and the patch gradient (how textured it is).
    """
    return view_as_blocks(img2d, (P, P)).mean(axis=(2, 3)).ravel()

def main():
    rows = []   # one row per flat patch: [mean_in, s_in, s_ours, s_off, s_ref]
    for path in sorted(glob("data/LOL/eval15/low/*.png")):
        name = os.path.basename(path)
        low  = lum(path)
        ref  = lum(os.path.join("data/LOL/eval15/high", name))
        ours = lum(os.path.join("outputs/eval_seed1", name))
        off  = lum(os.path.join("outputs/eval_official", name))

        # gaussian lightly blurs the reference, so tiny specks don’t count as texture
        # sobel measures the edge strength at every pixel
        # patch_mean gives the average edge strength per 20×20 patch, as 600 values
        grad = patch_mean(sobel(gaussian_filter(ref, 1))) 

        # 30% of the patches have a gradient below the cutoff.
        flat = grad <= np.percentile(grad, 30)  # The comparison gives a True/False array of 600 values, where True means a flat patch.

        # Noise level per patch (std of the residual) in each image:
        s_in   = patch_std(residual(low))   # starting noise, before any brightening
        s_ours = patch_std(residual(ours))  # noise after our model brightens it
        s_off  = patch_std(residual(off))   # noise after the official model brightens it
        s_ref  = patch_std(residual(ref))   # noise in the ground truth
        m_in   = patch_mean(low)            # how dark each patch started, used to group patches by darkness

        # Keep only flat patches: there the true image is nearly constant, so the residual
        # is pure noise (on textured patches it would also count real detail as noise).
        
        # Each row = one patch: [input brightness, input noise, ours, official, ground truth]
        rows.append(np.column_stack([m_in[flat], s_in[flat], s_ours[flat], s_off[flat], s_ref[flat]]))
        # keeping only [flat], then stack them as columns and add to rows

    R = np.vstack(rows)                  # all flat patches from all 15 images, shape (≈2700, 5)
    R = R[R[:, 1] > 0]                   # drop patches with zero input noise (avoids dividing by 0)
    m_in, s_in, s_ours, s_off, s_ref = R.T
    
    edges = np.quantile(m_in, [0, .2, .4, .6, .8, 1]) # LOL images are so dark that fixed ranges, like 0–0.1 or 0.1–0.2, would leave some bins nearly empty.
    b = np.digitize(m_in, edges[1:-1])

    ex_ours_list, ex_off_list = [], []

    for k in range(5):
        sel = b == k
        gain    = (s_ours[sel] / s_in[sel]).mean()    # how much our model multiplied the noise
        ex_ours = (s_ours[sel] / s_ref[sel]).mean()   # ours vs a real bright photo
        ex_off  = (s_off[sel]  / s_ref[sel]).mean()   # official vs a real bright photo
        print(k, round(edges[k], 3), round(edges[k+1], 3), sel.sum(), round(gain, 2), round(ex_ours, 2), round(ex_off, 2))
        ex_ours_list.append(ex_ours)
        ex_off_list.append(ex_off)

    x = np.arange(5)
    plt.bar(x - 0.2, ex_ours_list, width=0.4, label="Ours")
    plt.bar(x + 0.2, ex_off_list, width=0.4, label="Official")
    plt.axhline(1.0, ls="--", color="gray")

    plt.xticks(x, ["1 (darkest)", "2", "3", "4", "5 (brightest)"])
    plt.xlabel("Input brightness bin")
    plt.ylabel("Noise std / ground-truth noise std")
    plt.legend()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/noise_by_darkness.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()