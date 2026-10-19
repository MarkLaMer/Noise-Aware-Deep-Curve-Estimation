import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from nadce.curves import apply_curves
from nadce.data import LOLLowDataset
from nadce.losses import zero_dce_loss
from nadce.model import DCENet

from tqdm import tqdm #progress bar
# 1. holy Imports!

def main():
    # 2. Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/LOL/our485/low")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    if torch.cuda.is_available(): #gpu-accelerated training
        parser.add_argument("--device", type=str, default="cuda")
    else:
        parser.add_argument("--device", type=str, default="cpu")

    # 3. Seeds, threads, output folder    
    args = parser.parse_args()
        # seeds: same seed = same random numbers = reproducible run
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # Python, NumPy and PyTorch each have their own random-number generator

    # CPU threads: only change it if --threads was given
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # output folder: default outputs/seed1, outputs/seed2, ...
    if args.out is None:
        args.out = f"outputs/seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    # 4. Save the config:
    
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    ds = LOLLowDataset(args.data, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = DCENet().to(args.device)
    print("device:", args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging file: this writes the header row with the coln names
    with open(os.path.join(args.out, "log.csv"), "w") as f: # write the log.csv or wipe if prev run left one
        f.write("epoch,total,spa,exp,col,tv\n")

    # best epoch tracker setup
    best = float("inf") # The best total loss seen so far, which is inf cause we have nothing this far
    best_epoch = 0

    # training loop: loop for every epoch (full pass over training images)
    for epoch in tqdm(range(args.epochs), desc="total", position=0):
        model.train() # switch model to "training mode" (standard practice)
        
        # logging dict: running totals for each loss term for this epoch, reset to 0 at start of each epoch
        sums = {"total": 0.0, "spa": 0.0, "exp": 0.0, "col": 0.0, "tv": 0.0}
        n = 0 # batches count
        
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", position=1, leave=False)
        for x in bar:                            # x is a batch of 8 low-light images (8, 3, 256, 256) 
            x = x.to(args.device)                # copy of this tensor to GPU (run in cuda) and save as x for future pointer
            A = model(x)                         # DCE-Net predicts 24 curve maps per image (8 * RGB), each b/w range [-1, 1]
            y = apply_curves(x, A)               # "computah, ENHANCE!" aka result image of curve LE(x) = x + A·x·(1 − x) applied 8 times
            # total = weighted sum of the 4 losses (no ground truth used yet), 
            # parts = the four weighted terms as plain floats, for logging only
            total, parts = zero_dce_loss(x, y, A) 

            # update total
            sums["total"] += total.item() #.item() turns the 1-number tensor into a plain float
            for k, v in parts.items():
                sums[k] += v # add each loss term to running total
            n += 1 # once per batch

            optimizer.zero_grad() # clears the previous batch’s gradients
            total.backward() # computes d(loss)/d(weight) for every weight
            # Gradient clipping rescales the gradients if their combined size (norm) is over 0.1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step() # Adam update (good ol Adam)

        # Make a dict of average_sum from the sums
        avg_sum = {}
        for k, v in sums.items(): # k = total, v = running sum, then "spa", then "exp", ...
            avg_sum[k] = v / n

        # Check and save the best epoch
        is_best = avg_sum["total"] < best
        best_epoch_tag = ""
        if is_best:
            best = avg_sum["total"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
            best_epoch_tag = "  *** new best"
        else:
            best_epoch_tag = ""

        # tqdm.write prints the epoch-average losses, not last batch’s losses anymore to avoid havign noisy output
        tqdm.write(f"epoch {epoch + 1}/{args.epochs}  {' '.join(f'{k}={v:.4f}' for k, v in avg_sum.items())} best {best:.4f} @ {best_epoch} {best_epoch_tag}")
        # torch.save overwrites last.pth with the current weights every epoch (so crash at most deletes one epoch)
        torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
        
        # Add to log.csv
        with open(os.path.join(args.out, "log.csv"), "a") as f: # "a" to append to the list another row
            f.write(f"{epoch + 1},{avg_sum['total']},{avg_sum['spa']},{avg_sum['exp']},{avg_sum['col']},{avg_sum['tv']} \n") 
        
if __name__ == "__main__":
    main()