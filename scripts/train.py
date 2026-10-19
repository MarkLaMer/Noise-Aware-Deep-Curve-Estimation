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
    model = DCENet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.epochs):
        model.train()
        for x in loader:
            A = model(x)                         # 24 curve maps
            y = apply_curves(x, A)               # enhanced image
            total, parts = zero_dce_loss(x, y, A)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        print(f"epoch {epoch + 1}/{args.epochs}  loss {total.item():.4f}  {parts}")
        torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))

if __name__ == "__main__":
    main()