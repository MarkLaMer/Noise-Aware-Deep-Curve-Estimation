import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):

    if isinstance(m, nn.Conv2d): #is this layer a conv?
        m.weight.data.normal_(0.0, 0.02) #overwrites the weights in place with random numbers of mean 0 and std 0.02


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
        self.apply(init_weights)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3,x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2,x5], 1)))
        x7 = torch.tanh(self.e_conv7(torch.cat([x1,x6], 1)))

        return x7


