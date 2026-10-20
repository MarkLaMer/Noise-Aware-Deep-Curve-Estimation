import argparse, os
from glob import glob
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from nadce.model import DCENet
from nadce.curves import apply_curves

def main():
    """
    Evaluate a checkpoint on LOL eval15 at full resolution with PSNR/SSIM, optionally save the enhanced images
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)  # Which trained weights to evaluate (needed to run evaluation)
    parser.add_argument("--data", type=str, default="data/LOL/eval15") # Where to find the test set
    parser.add_argument("--official", action="store_true")  # Turns on the sign flip (A = -A). The official Zero-DCE code writes the curve as x + A·(x² − x), which is negative of ours.
    parser.add_argument("--save-dir", type=str, default=None) #If given, saves every enhanced image there, for a before/after figure and the noise study
    args = parser.parse_args()    
    
    model = DCENet()
    state = torch.load(args.ckpt, map_location="cpu") # checkpoints were saved from the GPU, and evaluation runs on the CPU
    model.load_state_dict(state)
    model.eval()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    psnrs, ssims = [], []

    # find the 15 test images and pair each dark one with its correctly lit version.
    for path in sorted(glob(os.path.join(args.data, "low", "*.png"))): 
        name = os.path.basename(path)
        ref_path = os.path.join(args.data, "high", name)

        # Input: the dark image from low/ (this is what the model enhances)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0     #(NO resize: full 600x400) kept as uint8 (H, W, 3), the format skimage expects
        # print(x.shape, x.max())

        # Reference: the bright image from high/ (this is what we score against)
        ref = np.array(Image.open(ref_path).convert("RGB"))
        
        with torch.no_grad(): 
            A = model(x)
            if args.official:
                A = -A                          # -A matches the official sign convention. 
            y = apply_curves(x, A).clamp(0,1)   # clamp keeps the values in [0, 1]

        out = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  #metrics are computed on 8-bit images, as saved results are

        p = peak_signal_noise_ratio(ref, out, data_range=255)
        s = structural_similarity(ref, out, channel_axis=2, data_range=255)  # channel_axis=2 makes SSIM treat the image as RGB, which also explains the SSIM spread in published numbers
        psnrs.append(p)
        ssims.append(s)
        # print(name, round(p, 2), round(s, 3))

        if args.save_dir: 
            Image.fromarray(out).save(os.path.join(args.save_dir, name))

    print("mean PSNR", np.mean(psnrs), "mean SSIM", np.mean(ssims)) # The 2 means over the 15 test images

if __name__ == "__main__":
    main()