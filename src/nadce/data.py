import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LOLLowDataset(Dataset):
    def __init__(self, root, size=256, limit=None):
        paths = sorted(glob(os.path.join(root, "*.png")))
        if limit is not None:
            paths = paths[:limit]

        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB") # open the PNG; force 3 colour channels
            img = img.resize((size, size), Image.LANCZOS) # shrink to 256×256 (LANCZOS is a high-quality resize filter)
            arr = np.array(img) # to a NumPy array
            t = torch.from_numpy(arr).permute(2, 0, 1) # make the new order old dim 2, then old dim 0, then old dim 1
            imgs.append(t)
        self.images = torch.stack(imgs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i].float() / 255.0

