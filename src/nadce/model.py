import torch
import torch.nn as nn
import torch.nn.functional as F


class DCENet(nn.Module):
    def __init__(self):
        super().__init__()
        # create e_conv1 ... e_conv7 here (see the layer table)
        # out = (H+2*padding-kernel)/stride+1 = (256+2*1-3) / 1+1 = 256
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1) ## in, out, kernel, stride, padding
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)
        # o la la, 79,416 parameters! :)

        # then apply the weight init

    def forward(self, x):
        # x1 = ..., x2 = ..., up to the 24-channel output
        # return it
        ...