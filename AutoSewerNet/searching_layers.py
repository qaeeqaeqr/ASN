import torch
from torch import nn
import torch.nn.functional as f
from blocks import *


class TBS_single_layer(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(TBS_single_layer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.theta = nn.Parameter(torch.rand(size=(1, 9), dtype=torch.float32, requires_grad=True))

        self.blk0 = Block_k3_e1(self.in_channel, self.out_channel, self.stride)
        self.blk1 = Block_k3_e1_g2(self.in_channel, self.out_channel, self.stride)
        self.blk2 = Block_k3_e3(self.in_channel, self.out_channel, self.stride)
        self.blk3 = Block_k3_e6(self.in_channel, self.out_channel, self.stride)
        self.blk4 = Block_k5_e1(self.in_channel, self.out_channel, self.stride)
        self.blk5 = Block_k5_e1_g2(self.in_channel, self.out_channel, self.stride)
        self.blk6 = Block_k5_e3(self.in_channel, self.out_channel, self.stride)
        self.blk7 = Block_k5_e6(self.in_channel, self.out_channel, self.stride)
        self.blk8 = Block_skip(self.in_channel, self.out_channel, self.stride)

    def forward(self, x):
        rate = f.gumbel_softmax(self.theta, tau=1., hard=False)
        y0 = self.blk0(x) * self.theta[0][0] * rate[0][0]
        y1 = self.blk1(x) * self.theta[0][1] * rate[0][1]
        y2 = self.blk2(x) * self.theta[0][2] * rate[0][2]
        y3 = self.blk3(x) * self.theta[0][3] * rate[0][3]
        y4 = self.blk4(x) * self.theta[0][4] * rate[0][4]
        y5 = self.blk5(x) * self.theta[0][5] * rate[0][5]
        y6 = self.blk6(x) * self.theta[0][6] * rate[0][6]
        y7 = self.blk7(x) * self.theta[0][7] * rate[0][7]
        y8 = self.blk8(x) * self.theta[0][8] * rate[0][8]

        return y0+y1+y2+y3+y4+y5+y6+y7+y8

